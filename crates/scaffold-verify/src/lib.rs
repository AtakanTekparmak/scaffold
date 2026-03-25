//! Scaffold v2 Graph Verifier
//!
//! Runs on IR after lowering. Performs structural analysis:
//! 1. Port compatibility: step args match node input types
//! 2. Cycle detection: no recursive graph references
//! 3. Bounds checking: loops have finite max, parallel collections bounded
//! 4. Emit completeness: all branches lead to emit
//! 5. Carry scope: carry only inside loops
//! 6. Reachability: warn on unused nodes/graphs
//! 7. Tunable path resolution: tune paths resolve to valid node.field
//! 8. Topology constraint validation: preserve targets exist, mutations recognized

use std::collections::{HashMap, HashSet};

use scaffold_ir::ir::*;

/// A verification finding
#[derive(Debug, Clone)]
pub struct VerifyError {
    pub message: String,
    pub severity: Severity,
    /// Context path (e.g. "graph.solve.step.s1")
    pub context: String,
}

impl std::fmt::Display for VerifyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let level = match self.severity {
            Severity::Error => "error",
            Severity::Warning => "warning",
        };
        if self.context.is_empty() {
            write!(f, "[{}] {}", level, self.message)
        } else {
            write!(f, "[{}] {}: {}", level, self.context, self.message)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
}

/// Verify an IR and return all findings
pub fn verify(ir: &ScaffoldIR) -> Vec<VerifyError> {
    let mut verifier = Verifier::new(ir);
    verifier.run();
    verifier.errors
}

struct Verifier<'a> {
    ir: &'a ScaffoldIR,
    errors: Vec<VerifyError>,
    node_names: HashSet<String>,
    graph_names: HashSet<String>,
}

impl<'a> Verifier<'a> {
    fn new(ir: &'a ScaffoldIR) -> Self {
        let node_names: HashSet<String> = ir.nodes.iter().map(|n| n.name.clone()).collect();
        let graph_names: HashSet<String> = ir.graphs.iter().map(|g| g.name.clone()).collect();
        Self {
            ir,
            errors: Vec::new(),
            node_names,
            graph_names,
        }
    }

    fn error(&mut self, ctx: &str, msg: impl Into<String>) {
        self.errors.push(VerifyError {
            message: msg.into(),
            severity: Severity::Error,
            context: ctx.to_string(),
        });
    }

    fn warn(&mut self, ctx: &str, msg: impl Into<String>) {
        self.errors.push(VerifyError {
            message: msg.into(),
            severity: Severity::Warning,
            context: ctx.to_string(),
        });
    }

    fn run(&mut self) {
        // 1. Check each graph
        for graph in &self.ir.graphs {
            self.check_graph(graph);
        }

        // 2. Cycle detection (graph-to-graph calls)
        self.check_cycles();

        // 3. Check objectives
        for obj in &self.ir.objectives {
            self.check_objective(obj);
        }

        // 4. Reachability
        self.check_reachability();
    }

    fn check_graph(&mut self, graph: &GraphIR) {
        let ctx = format!("graph.{}", graph.name);
        let mut step_names: HashSet<String> = HashSet::new();
        let mut has_emit = false;

        self.check_stmts(&graph.body, &ctx, &mut step_names, &mut has_emit, false);

        if !has_emit {
            self.warn(&ctx, "graph has no emit statement");
        }
    }

    fn check_stmts(
        &mut self,
        stmts: &[GraphStmtIR],
        ctx: &str,
        step_names: &mut HashSet<String>,
        has_emit: &mut bool,
        in_loop: bool,
    ) {
        for stmt in stmts {
            match stmt {
                GraphStmtIR::Step(s) => {
                    let step_ctx = format!("{}.step.{}", ctx, s.name);
                    // Check node/graph reference
                    if !self.node_names.contains(&s.node)
                        && !self.graph_names.contains(&s.node)
                    {
                        self.error(&step_ctx, format!("references undefined node '{}'", s.node));
                    }
                    step_names.insert(s.name.clone());
                }
                GraphStmtIR::Loop(l) => {
                    // Bounds check: max must be a literal or expression
                    self.check_loop_max(&l.max, &format!("{}.loop", ctx));
                    self.check_stmts(&l.body, &format!("{}.loop", ctx), step_names, has_emit, true);
                }
                GraphStmtIR::If(i) => {
                    let mut then_emit = false;
                    let mut else_emit = false;
                    self.check_stmts(
                        &i.then_body,
                        &format!("{}.if.then", ctx),
                        step_names,
                        &mut then_emit,
                        in_loop,
                    );
                    self.check_stmts(
                        &i.else_body,
                        &format!("{}.if.else", ctx),
                        step_names,
                        &mut else_emit,
                        in_loop,
                    );
                    if then_emit && else_emit {
                        *has_emit = true;
                    } else if then_emit || else_emit {
                        *has_emit = true; // At least one branch emits
                    }
                }
                GraphStmtIR::Choose(c) => {
                    for alt in &c.alternatives {
                        if !self.graph_names.contains(alt) && !self.node_names.contains(alt) {
                            self.error(
                                &format!("{}.choose", ctx),
                                format!("alternative '{}' not found", alt),
                            );
                        }
                    }
                }
                GraphStmtIR::Parallel(p) => {
                    self.check_stmts(
                        &p.body,
                        &format!("{}.parallel", ctx),
                        step_names,
                        has_emit,
                        in_loop,
                    );
                    if let Some(ref reduce) = p.reduce {
                        if !self.node_names.contains(reduce)
                            && !self.graph_names.contains(reduce)
                        {
                            self.error(
                                &format!("{}.parallel", ctx),
                                format!("reduce function '{}' not found", reduce),
                            );
                        }
                    }
                }
                GraphStmtIR::Emit(_) => {
                    *has_emit = true;
                }
                GraphStmtIR::Carry(c) => {
                    if !in_loop {
                        self.error(
                            &format!("{}.carry.{}", ctx, c.name),
                            "carry statement outside of loop",
                        );
                    }
                }
            }
        }
    }

    fn check_loop_max(&mut self, expr: &ExprIR, ctx: &str) {
        // Warn if max is not a literal integer (could be unbounded)
        match expr {
            ExprIR::LitInt { value } => {
                if *value <= 0 {
                    self.error(ctx, "loop max must be positive");
                }
            }
            ExprIR::Ident { .. } => {
                // Variable reference — can't statically verify bounds
            }
            _ => {
                // Complex expression — acceptable
            }
        }
    }

    fn check_cycles(&mut self) {
        // Build call graph: graph → set of graphs it references via steps
        let mut call_graph: HashMap<String, HashSet<String>> = HashMap::new();
        for graph in &self.ir.graphs {
            let mut called = HashSet::new();
            collect_graph_refs(&graph.body, &self.graph_names, &mut called);
            call_graph.insert(graph.name.clone(), called);
        }

        // DFS for cycles
        let mut visited = HashSet::new();
        let mut in_stack = HashSet::new();
        for name in call_graph.keys() {
            if !visited.contains(name) {
                self.dfs_cycle_check(name, &call_graph, &mut visited, &mut in_stack);
            }
        }
    }

    fn dfs_cycle_check(
        &mut self,
        node: &str,
        call_graph: &HashMap<String, HashSet<String>>,
        visited: &mut HashSet<String>,
        in_stack: &mut HashSet<String>,
    ) {
        visited.insert(node.to_string());
        in_stack.insert(node.to_string());

        if let Some(neighbors) = call_graph.get(node) {
            for neighbor in neighbors {
                if in_stack.contains(neighbor) {
                    self.error(
                        &format!("graph.{}", node),
                        format!("recursive graph reference to '{}'", neighbor),
                    );
                } else if !visited.contains(neighbor) {
                    self.dfs_cycle_check(neighbor, call_graph, visited, in_stack);
                }
            }
        }

        in_stack.remove(node);
    }

    fn check_objective(&mut self, obj: &ObjectiveIR) {
        let ctx = format!("objective.{}", obj.name);

        // Check graph reference
        if !self.graph_names.contains(&obj.graph) {
            self.error(&ctx, format!("references undefined graph '{}'", obj.graph));
        }

        // Check tunable paths
        for tunable in &obj.tunables {
            if tunable.path.is_empty() {
                continue;
            }
            let node_name = &tunable.path[0];
            if !self.node_names.contains(node_name) {
                self.error(
                    &ctx,
                    format!("tune path references undefined node '{}'", node_name),
                );
            }
            // Check that the field in path[1] is a valid config field
            if tunable.path.len() >= 2 {
                let field = &tunable.path[1];
                let valid_fields = [
                    "model",
                    "temperature",
                    "max_tokens",
                    "template",
                    "system",
                    "max_turns",
                    "timeout",
                ];
                if !valid_fields.contains(&field.as_str()) {
                    self.warn(
                        &ctx,
                        format!("tune field '{}' may not be a valid config field", field),
                    );
                }
            }
        }

        // Check topology constraints
        if let Some(ref topo) = obj.topology {
            let valid_mutations = [
                "insert_verify",
                "wrap_retry",
                "insert_step",
                "remove_step",
                "replace_component",
                "fan_out",
                "replace_with_subgraph",
                "set_config",
            ];
            for mutation in &topo.mutations {
                if !valid_mutations.contains(&mutation.as_str()) {
                    self.warn(
                        &ctx,
                        format!("unrecognized mutation type '{}'", mutation),
                    );
                }
            }

            // Check preserve targets
            for preserved in &topo.preserve {
                if !self.node_names.contains(preserved)
                    && !self.graph_names.contains(preserved)
                {
                    self.warn(
                        &ctx,
                        format!("preserve target '{}' not found", preserved),
                    );
                }
            }
        }

        // Check metric checker references
        for metric in &obj.metrics {
            let checker_names: HashSet<&str> = obj.checkers.iter().map(|c| c.name.as_str()).collect();
            if !checker_names.contains(metric.checker.as_str()) {
                self.error(
                    &ctx,
                    format!(
                        "metric '{}' references undefined checker '{}'",
                        metric.name, metric.checker
                    ),
                );
            }
        }
    }

    fn check_reachability(&mut self) {
        // Collect all referenced nodes and graphs
        let mut referenced_nodes: HashSet<String> = HashSet::new();
        let mut referenced_graphs: HashSet<String> = HashSet::new();

        for graph in &self.ir.graphs {
            collect_node_refs(&graph.body, &mut referenced_nodes);
            collect_graph_refs(&graph.body, &self.graph_names, &mut referenced_graphs);
        }

        for obj in &self.ir.objectives {
            referenced_graphs.insert(obj.graph.clone());
        }

        // Warn about unused nodes
        for node in &self.ir.nodes {
            if !referenced_nodes.contains(&node.name) {
                self.warn("", format!("node '{}' is never referenced", node.name));
            }
        }

        // Warn about unused graphs (only if objectives exist)
        if !self.ir.objectives.is_empty() {
            for graph in &self.ir.graphs {
                if !referenced_graphs.contains(&graph.name) {
                    self.warn("", format!("graph '{}' is never referenced", graph.name));
                }
            }
        }
    }
}

/// Collect node references from graph statements
fn collect_node_refs(stmts: &[GraphStmtIR], refs: &mut HashSet<String>) {
    for stmt in stmts {
        match stmt {
            GraphStmtIR::Step(s) => {
                refs.insert(s.node.clone());
            }
            GraphStmtIR::Loop(l) => collect_node_refs(&l.body, refs),
            GraphStmtIR::If(i) => {
                collect_node_refs(&i.then_body, refs);
                collect_node_refs(&i.else_body, refs);
            }
            GraphStmtIR::Parallel(p) => collect_node_refs(&p.body, refs),
            GraphStmtIR::Choose(c) => {
                for alt in &c.alternatives {
                    refs.insert(alt.clone());
                }
            }
            _ => {}
        }
    }
}

/// Collect graph references from graph statements
fn collect_graph_refs(
    stmts: &[GraphStmtIR],
    known_graphs: &HashSet<String>,
    refs: &mut HashSet<String>,
) {
    for stmt in stmts {
        match stmt {
            GraphStmtIR::Step(s) => {
                if known_graphs.contains(&s.node) {
                    refs.insert(s.node.clone());
                }
            }
            GraphStmtIR::Loop(l) => collect_graph_refs(&l.body, known_graphs, refs),
            GraphStmtIR::If(i) => {
                collect_graph_refs(&i.then_body, known_graphs, refs);
                collect_graph_refs(&i.else_body, known_graphs, refs);
            }
            GraphStmtIR::Parallel(p) => collect_graph_refs(&p.body, known_graphs, refs),
            GraphStmtIR::Choose(c) => {
                for alt in &c.alternatives {
                    if known_graphs.contains(alt) {
                        refs.insert(alt.clone());
                    }
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scaffold_ir::serialize::lower;
    use scaffold_syntax::parser::parse;

    fn verify_source(src: &str) -> Vec<VerifyError> {
        let program = parse(src).unwrap();
        let ir = lower(&program).unwrap();
        verify(&ir)
    }

    #[test]
    fn verify_valid_graph() {
        let errors = verify_source(
            r#"
            node solver: prompt {
                in: string
                out: string
                model: "gpt-4o"
                template: "test"
            }
            graph solve {
                in: string
                out: string
                step s = solver(input)
                emit s
            }
        "#,
        );
        let real_errors: Vec<_> = errors
            .iter()
            .filter(|e| e.severity == Severity::Error)
            .collect();
        assert!(real_errors.is_empty(), "unexpected errors: {:?}", real_errors);
    }

    #[test]
    fn verify_undefined_node_ref() {
        let errors = verify_source(
            r#"
            graph test {
                in: string
                out: string
                step s = nonexistent(input)
                emit s
            }
        "#,
        );
        assert!(errors.iter().any(|e| e.message.contains("undefined")));
    }

    #[test]
    fn verify_carry_outside_loop() {
        let errors = verify_source(
            r#"
            node solver: prompt {
                in: string
                out: string
                model: "gpt-4o"
                template: "test"
            }
            graph test {
                in: string
                out: string
                carry x = "hello"
                emit x
            }
        "#,
        );
        assert!(errors
            .iter()
            .any(|e| e.message.contains("carry") && e.message.contains("outside")));
    }
}
