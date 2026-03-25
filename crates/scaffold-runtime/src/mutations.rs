//! Topology mutation operators for graph optimization.
//!
//! The optimizer works on `GraphIR` directly. Each mutation produces a new
//! `GraphIR` candidate that is re-verified before execution.

use scaffold_ir::ir::*;

/// A topology mutation to apply to a graph.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Mutation {
    /// Insert a verify node after a step.
    InsertVerify {
        after_step: String,
        verify_node: String,
        max_retries: u32,
    },

    /// Wrap a step in a retry loop with a verify gate.
    WrapRetry {
        step: String,
        verify_node: String,
        max_retries: u32,
    },

    /// Add a new step after an existing step.
    InsertStep {
        after_step: String,
        new_step_name: String,
        node: String,
    },

    /// Remove a step (reconnect data flow).
    RemoveStep { step: String },

    /// Replace which node a step invokes.
    ReplaceComponent { step: String, new_node: String },

    /// Fan out a step to parallel execution with reduce.
    FanOutParallel {
        step: String,
        split_node: String,
        reduce_node: String,
    },

    /// Replace a step with a subgraph reference.
    ReplaceWithSubgraph { step: String, graph: String },

    /// Change a node config field.
    SetConfig {
        node: String,
        field: String,
        value: serde_json::Value,
    },
}

/// Result of applying a mutation.
#[derive(Debug)]
pub enum MutationResult {
    /// Mutation applied successfully, producing a new graph.
    Ok(GraphIR),
    /// Mutation could not be applied (step not found, etc.)
    Skipped(String),
}

/// Apply a mutation to a graph, producing a new graph.
pub fn apply_mutation(graph: &GraphIR, mutation: &Mutation, ir: &ScaffoldIR) -> MutationResult {
    match mutation {
        Mutation::InsertVerify {
            after_step,
            verify_node,
            max_retries,
        } => apply_insert_verify(graph, after_step, verify_node, *max_retries),

        Mutation::WrapRetry {
            step,
            verify_node,
            max_retries,
        } => apply_wrap_retry(graph, step, verify_node, *max_retries),

        Mutation::InsertStep {
            after_step,
            new_step_name,
            node,
        } => apply_insert_step(graph, after_step, new_step_name, node),

        Mutation::RemoveStep { step } => apply_remove_step(graph, step),

        Mutation::ReplaceComponent { step, new_node } => {
            apply_replace_component(graph, step, new_node)
        }

        Mutation::FanOutParallel {
            step,
            split_node,
            reduce_node,
        } => apply_fan_out(graph, step, split_node, reduce_node),

        Mutation::ReplaceWithSubgraph { step, graph: sub } => {
            apply_replace_with_subgraph(graph, step, sub)
        }

        Mutation::SetConfig { node, field, value } => {
            apply_set_config(graph, ir, node, field, value)
        }
    }
}

/// Check if a mutation respects topology constraints.
pub fn check_constraints(
    graph: &GraphIR,
    topology: &TopologyIR,
) -> Vec<String> {
    let mut violations = Vec::new();

    // Check max_nodes
    if let Some(max) = topology.max_nodes {
        let count = count_steps(&graph.body);
        if count > max {
            violations.push(format!(
                "graph has {} steps, exceeding max_nodes={}",
                count, max
            ));
        }
    }

    // Check max_depth
    if let Some(max) = topology.max_depth {
        let depth = measure_depth(&graph.body);
        if depth > max {
            violations.push(format!(
                "graph has nesting depth {}, exceeding max_depth={}",
                depth, max
            ));
        }
    }

    // Check preserve (preserved steps must still exist)
    for preserved in &topology.preserve {
        if !step_exists(&graph.body, preserved) {
            violations.push(format!("preserved step '{}' was removed", preserved));
        }
    }

    violations
}

// ── Mutation implementations ──

fn apply_insert_verify(
    graph: &GraphIR,
    after_step: &str,
    verify_node: &str,
    max_retries: u32,
) -> MutationResult {
    let mut new_body = Vec::new();
    let mut found = false;

    for stmt in &graph.body {
        match insert_verify_in_stmts(stmt, after_step, verify_node, max_retries) {
            (new_stmts, true) => {
                new_body.extend(new_stmts);
                found = true;
            }
            (new_stmts, false) => {
                new_body.extend(new_stmts);
            }
        }
    }

    if !found {
        return MutationResult::Skipped(format!("step '{}' not found", after_step));
    }

    MutationResult::Ok(GraphIR {
        body: new_body,
        ..graph.clone()
    })
}

fn insert_verify_in_stmts(
    stmt: &GraphStmtIR,
    after_step: &str,
    verify_node: &str,
    max_retries: u32,
) -> (Vec<GraphStmtIR>, bool) {
    match stmt {
        GraphStmtIR::Step(s) if s.name == after_step => {
            let verify_step_name = format!("{}_verify", s.name);
            let verify_step = GraphStmtIR::Step(StepIR {
                name: verify_step_name.clone(),
                node: verify_node.to_string(),
                args: vec![StepArgIR::Positional {
                    value: ExprIR::Ident {
                        name: s.name.clone(),
                    },
                }],
            });

            if max_retries > 0 {
                // Wrap in retry loop
                let loop_body = vec![
                    stmt.clone(),
                    verify_step,
                    GraphStmtIR::If(IfIR {
                        cond: ExprIR::FieldAccess {
                            base: Box::new(ExprIR::Ident {
                                name: verify_step_name,
                            }),
                            field: "pass".to_string(),
                        },
                        then_body: vec![],
                        else_body: vec![], // loop continues
                    }),
                ];
                let loop_stmt = GraphStmtIR::Loop(LoopIR {
                    max: ExprIR::LitInt {
                        value: max_retries as i64 + 1,
                    },
                    while_cond: ExprIR::LitBool { value: true },
                    body: loop_body,
                });
                (vec![loop_stmt], true)
            } else {
                (vec![stmt.clone(), verify_step], true)
            }
        }
        _ => (vec![stmt.clone()], false),
    }
}

fn apply_wrap_retry(
    graph: &GraphIR,
    step_name: &str,
    verify_node: &str,
    max_retries: u32,
) -> MutationResult {
    let mut new_body = Vec::new();
    let mut found = false;

    for stmt in &graph.body {
        if let GraphStmtIR::Step(s) = stmt {
            if s.name == step_name {
                found = true;
                let verify_step_name = format!("{}_check", s.name);
                let loop_body = vec![
                    stmt.clone(),
                    GraphStmtIR::Step(StepIR {
                        name: verify_step_name.clone(),
                        node: verify_node.to_string(),
                        args: vec![StepArgIR::Positional {
                            value: ExprIR::Ident {
                                name: s.name.clone(),
                            },
                        }],
                    }),
                ];
                new_body.push(GraphStmtIR::Loop(LoopIR {
                    max: ExprIR::LitInt {
                        value: max_retries as i64 + 1,
                    },
                    while_cond: ExprIR::UnaryNot {
                        operand: Box::new(ExprIR::FieldAccess {
                            base: Box::new(ExprIR::Ident {
                                name: verify_step_name,
                            }),
                            field: "pass".to_string(),
                        }),
                    },
                    body: loop_body,
                }));
                continue;
            }
        }
        new_body.push(stmt.clone());
    }

    if !found {
        return MutationResult::Skipped(format!("step '{}' not found", step_name));
    }

    MutationResult::Ok(GraphIR {
        body: new_body,
        ..graph.clone()
    })
}

fn apply_insert_step(
    graph: &GraphIR,
    after_step: &str,
    new_name: &str,
    node: &str,
) -> MutationResult {
    let mut new_body = Vec::new();
    let mut found = false;

    for stmt in &graph.body {
        new_body.push(stmt.clone());
        if let GraphStmtIR::Step(s) = stmt {
            if s.name == after_step {
                found = true;
                new_body.push(GraphStmtIR::Step(StepIR {
                    name: new_name.to_string(),
                    node: node.to_string(),
                    args: vec![StepArgIR::Positional {
                        value: ExprIR::Ident {
                            name: s.name.clone(),
                        },
                    }],
                }));
            }
        }
    }

    if !found {
        return MutationResult::Skipped(format!("step '{}' not found", after_step));
    }

    MutationResult::Ok(GraphIR {
        body: new_body,
        ..graph.clone()
    })
}

fn apply_remove_step(graph: &GraphIR, step_name: &str) -> MutationResult {
    let new_body: Vec<GraphStmtIR> = graph
        .body
        .iter()
        .filter(|stmt| {
            if let GraphStmtIR::Step(s) = stmt {
                s.name != step_name
            } else {
                true
            }
        })
        .cloned()
        .collect();

    if new_body.len() == graph.body.len() {
        return MutationResult::Skipped(format!("step '{}' not found", step_name));
    }

    MutationResult::Ok(GraphIR {
        body: new_body,
        ..graph.clone()
    })
}

fn apply_replace_component(
    graph: &GraphIR,
    step_name: &str,
    new_node: &str,
) -> MutationResult {
    let mut new_body = graph.body.clone();
    let mut found = false;

    for stmt in &mut new_body {
        if let GraphStmtIR::Step(s) = stmt {
            if s.name == step_name {
                s.node = new_node.to_string();
                found = true;
            }
        }
    }

    if !found {
        return MutationResult::Skipped(format!("step '{}' not found", step_name));
    }

    MutationResult::Ok(GraphIR {
        body: new_body,
        ..graph.clone()
    })
}

fn apply_fan_out(
    graph: &GraphIR,
    step_name: &str,
    split_node: &str,
    reduce_node: &str,
) -> MutationResult {
    let mut new_body = Vec::new();
    let mut found = false;

    for stmt in &graph.body {
        if let GraphStmtIR::Step(s) = stmt {
            if s.name == step_name {
                found = true;
                // Add split step
                let split_step_name = format!("{}_split", s.name);
                new_body.push(GraphStmtIR::Step(StepIR {
                    name: split_step_name.clone(),
                    node: split_node.to_string(),
                    args: s.args.clone(),
                }));
                // Add parallel with the original step inside
                new_body.push(GraphStmtIR::Parallel(ParallelIR {
                    var: "item".to_string(),
                    collection: ExprIR::Ident {
                        name: split_step_name,
                    },
                    reduce: Some(reduce_node.to_string()),
                    body: vec![GraphStmtIR::Step(StepIR {
                        name: s.name.clone(),
                        node: s.node.clone(),
                        args: vec![StepArgIR::Positional {
                            value: ExprIR::Ident {
                                name: "item".to_string(),
                            },
                        }],
                    })],
                }));
                continue;
            }
        }
        new_body.push(stmt.clone());
    }

    if !found {
        return MutationResult::Skipped(format!("step '{}' not found", step_name));
    }

    MutationResult::Ok(GraphIR {
        body: new_body,
        ..graph.clone()
    })
}

fn apply_replace_with_subgraph(
    graph: &GraphIR,
    step_name: &str,
    subgraph: &str,
) -> MutationResult {
    let mut new_body = graph.body.clone();
    let mut found = false;

    for stmt in &mut new_body {
        if let GraphStmtIR::Step(s) = stmt {
            if s.name == step_name {
                s.node = subgraph.to_string();
                found = true;
            }
        }
    }

    if !found {
        return MutationResult::Skipped(format!("step '{}' not found", step_name));
    }

    MutationResult::Ok(GraphIR {
        body: new_body,
        ..graph.clone()
    })
}

fn apply_set_config(
    _graph: &GraphIR,
    ir: &ScaffoldIR,
    node_name: &str,
    field: &str,
    _value: &serde_json::Value,
) -> MutationResult {
    // SetConfig doesn't modify the graph structure — it modifies the node config.
    // The executor uses overrides to apply these at runtime.
    // We verify the node exists.
    if !ir.nodes.iter().any(|n| n.name == node_name) {
        return MutationResult::Skipped(format!("node '{}' not found", node_name));
    }

    let valid_fields = [
        "model",
        "temperature",
        "max_tokens",
        "template",
        "system",
        "max_turns",
        "timeout",
    ];
    if !valid_fields.contains(&field) {
        return MutationResult::Skipped(format!("unknown config field '{}'", field));
    }

    // SetConfig produces overrides, not a new graph.
    // Return the graph unchanged — the optimizer handles overrides separately.
    MutationResult::Skipped("set_config produces overrides, not graph mutations".into())
}

// ── Helpers ──

fn count_steps(stmts: &[GraphStmtIR]) -> u64 {
    let mut count = 0;
    for stmt in stmts {
        match stmt {
            GraphStmtIR::Step(_) => count += 1,
            GraphStmtIR::Loop(l) => count += count_steps(&l.body),
            GraphStmtIR::If(i) => {
                count += count_steps(&i.then_body);
                count += count_steps(&i.else_body);
            }
            GraphStmtIR::Parallel(p) => count += count_steps(&p.body),
            _ => {}
        }
    }
    count
}

fn measure_depth(stmts: &[GraphStmtIR]) -> u64 {
    let mut max_depth = 0;
    for stmt in stmts {
        let d = match stmt {
            GraphStmtIR::Loop(l) => 1 + measure_depth(&l.body),
            GraphStmtIR::If(i) => {
                let then_d = measure_depth(&i.then_body);
                let else_d = measure_depth(&i.else_body);
                1 + then_d.max(else_d)
            }
            GraphStmtIR::Parallel(p) => 1 + measure_depth(&p.body),
            _ => 0,
        };
        max_depth = max_depth.max(d);
    }
    max_depth
}

fn step_exists(stmts: &[GraphStmtIR], name: &str) -> bool {
    for stmt in stmts {
        match stmt {
            GraphStmtIR::Step(s) if s.name == name => return true,
            GraphStmtIR::Loop(l) => {
                if step_exists(&l.body, name) {
                    return true;
                }
            }
            GraphStmtIR::If(i) => {
                if step_exists(&i.then_body, name) || step_exists(&i.else_body, name) {
                    return true;
                }
            }
            GraphStmtIR::Parallel(p) => {
                if step_exists(&p.body, name) {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
}

/// Collect all step names in a graph body.
pub fn collect_step_names(stmts: &[GraphStmtIR]) -> Vec<String> {
    let mut names = Vec::new();
    for stmt in stmts {
        match stmt {
            GraphStmtIR::Step(s) => names.push(s.name.clone()),
            GraphStmtIR::Loop(l) => names.extend(collect_step_names(&l.body)),
            GraphStmtIR::If(i) => {
                names.extend(collect_step_names(&i.then_body));
                names.extend(collect_step_names(&i.else_body));
            }
            GraphStmtIR::Parallel(p) => names.extend(collect_step_names(&p.body)),
            _ => {}
        }
    }
    names
}

/// Compute a structural fingerprint of a graph (for novelty search).
pub fn topology_hash(graph: &GraphIR) -> u64 {
    use std::hash::Hasher;
    use std::collections::hash_map::DefaultHasher;
    let mut hasher = DefaultHasher::new();
    hash_stmts(&graph.body, &mut hasher);
    hasher.finish()
}

fn hash_stmts(stmts: &[GraphStmtIR], hasher: &mut impl std::hash::Hasher) {
    use std::hash::Hash;
    stmts.len().hash(hasher);
    for stmt in stmts {
        match stmt {
            GraphStmtIR::Step(s) => {
                "step".hash(hasher);
                s.node.hash(hasher);
            }
            GraphStmtIR::Loop(l) => {
                "loop".hash(hasher);
                hash_stmts(&l.body, hasher);
            }
            GraphStmtIR::If(i) => {
                "if".hash(hasher);
                hash_stmts(&i.then_body, hasher);
                hash_stmts(&i.else_body, hasher);
            }
            GraphStmtIR::Parallel(p) => {
                "parallel".hash(hasher);
                hash_stmts(&p.body, hasher);
            }
            GraphStmtIR::Choose(c) => {
                "choose".hash(hasher);
                c.alternatives.hash(hasher);
            }
            GraphStmtIR::Emit(_) => {
                "emit".hash(hasher);
            }
            GraphStmtIR::Carry(c) => {
                "carry".hash(hasher);
                c.name.hash(hasher);
            }
        }
    }
}

/// Graph descriptor for the archive (novelty search).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GraphDescriptor {
    pub topology_hash: u64,
    pub node_count: u64,
    pub verify_count: u64,
    pub max_depth: u64,
}

impl GraphDescriptor {
    pub fn from_graph(graph: &GraphIR, ir: &ScaffoldIR) -> Self {
        let node_count = count_steps(&graph.body);
        let verify_count = count_verify_steps(&graph.body, ir);
        let max_depth = measure_depth(&graph.body);
        let topo_hash = topology_hash(graph);
        Self {
            topology_hash: topo_hash,
            node_count,
            verify_count,
            max_depth,
        }
    }
}

fn count_verify_steps(stmts: &[GraphStmtIR], ir: &ScaffoldIR) -> u64 {
    let mut count = 0;
    for stmt in stmts {
        match stmt {
            GraphStmtIR::Step(s) => {
                if ir
                    .nodes
                    .iter()
                    .any(|n| n.name == s.node && n.kind == NodeKindIR::Verify)
                {
                    count += 1;
                }
            }
            GraphStmtIR::Loop(l) => count += count_verify_steps(&l.body, ir),
            GraphStmtIR::If(i) => {
                count += count_verify_steps(&i.then_body, ir);
                count += count_verify_steps(&i.else_body, ir);
            }
            GraphStmtIR::Parallel(p) => count += count_verify_steps(&p.body, ir),
            _ => {}
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_graph() -> GraphIR {
        GraphIR {
            name: "solve".to_string(),
            input: TypeIR::String,
            output: TypeIR::String,
            body: vec![
                GraphStmtIR::Step(StepIR {
                    name: "s1".to_string(),
                    node: "solver".to_string(),
                    args: vec![StepArgIR::Positional {
                        value: ExprIR::Ident {
                            name: "input".to_string(),
                        },
                    }],
                }),
                GraphStmtIR::Emit(EmitIR::Direct {
                    value: ExprIR::Ident {
                        name: "s1".to_string(),
                    },
                }),
            ],
        }
    }

    fn make_test_ir() -> ScaffoldIR {
        ScaffoldIR {
            version: "2.0.0".to_string(),
            types: vec![],
            nodes: vec![
                NodeIR {
                    name: "solver".to_string(),
                    kind: NodeKindIR::Prompt,
                    input: TypeIR::String,
                    output: TypeIR::String,
                    config: NodeConfigIR::default(),
                },
                NodeIR {
                    name: "checker".to_string(),
                    kind: NodeKindIR::Verify,
                    input: TypeIR::String,
                    output: TypeIR::String,
                    config: NodeConfigIR::default(),
                },
            ],
            graphs: vec![make_test_graph()],
            objectives: vec![],
        }
    }

    #[test]
    fn test_insert_step() {
        let graph = make_test_graph();
        match apply_insert_step(&graph, "s1", "s2", "solver") {
            MutationResult::Ok(new_graph) => {
                assert_eq!(new_graph.body.len(), 3); // s1, s2, emit
                let names = collect_step_names(&new_graph.body);
                assert!(names.contains(&"s1".to_string()));
                assert!(names.contains(&"s2".to_string()));
            }
            MutationResult::Skipped(msg) => panic!("unexpected skip: {}", msg),
        }
    }

    #[test]
    fn test_remove_step() {
        let graph = make_test_graph();
        match apply_remove_step(&graph, "s1") {
            MutationResult::Ok(new_graph) => {
                assert_eq!(new_graph.body.len(), 1); // just emit
                let names = collect_step_names(&new_graph.body);
                assert!(!names.contains(&"s1".to_string()));
            }
            MutationResult::Skipped(msg) => panic!("unexpected skip: {}", msg),
        }
    }

    #[test]
    fn test_replace_component() {
        let graph = make_test_graph();
        match apply_replace_component(&graph, "s1", "checker") {
            MutationResult::Ok(new_graph) => {
                if let GraphStmtIR::Step(s) = &new_graph.body[0] {
                    assert_eq!(s.node, "checker");
                } else {
                    panic!("expected step");
                }
            }
            MutationResult::Skipped(msg) => panic!("unexpected skip: {}", msg),
        }
    }

    #[test]
    fn test_wrap_retry() {
        let graph = make_test_graph();
        match apply_wrap_retry(&graph, "s1", "checker", 3) {
            MutationResult::Ok(new_graph) => {
                // First stmt should be a loop
                assert!(matches!(&new_graph.body[0], GraphStmtIR::Loop(_)));
            }
            MutationResult::Skipped(msg) => panic!("unexpected skip: {}", msg),
        }
    }

    #[test]
    fn test_constraints() {
        let graph = make_test_graph();
        let topo = TopologyIR {
            mutations: vec![],
            max_nodes: Some(1),
            max_depth: None,
            preserve: vec!["s1".to_string()],
        };
        let violations = check_constraints(&graph, &topo);
        assert!(violations.is_empty()); // 1 step <= max_nodes=1

        let topo2 = TopologyIR {
            mutations: vec![],
            max_nodes: Some(0),
            max_depth: None,
            preserve: vec!["s1".to_string()],
        };
        let violations2 = check_constraints(&graph, &topo2);
        assert!(!violations2.is_empty()); // 1 step > max_nodes=0
    }

    #[test]
    fn test_topology_hash() {
        let g1 = make_test_graph();
        let mut g2 = make_test_graph();
        g2.body.insert(
            1,
            GraphStmtIR::Step(StepIR {
                name: "s2".to_string(),
                node: "checker".to_string(),
                args: vec![],
            }),
        );
        // Different structures should produce different hashes
        assert_ne!(topology_hash(&g1), topology_hash(&g2));
    }

    #[test]
    fn test_graph_descriptor() {
        let ir = make_test_ir();
        let graph = make_test_graph();
        let desc = GraphDescriptor::from_graph(&graph, &ir);
        assert_eq!(desc.node_count, 1);
        assert_eq!(desc.verify_count, 0);
        assert_eq!(desc.max_depth, 0);
    }
}
