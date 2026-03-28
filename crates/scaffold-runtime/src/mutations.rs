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

    /// Rewrite a node's prompt template.
    RewritePrompt { node: String, new_template: String },

    /// Rewrite a node's system prompt.
    RewriteSystem { node: String, new_system: String },

    /// Rewrite a tool node's shell command.
    RewriteShell { node: String, new_shell: String },

    /// Create a new prompt node and insert it as a step after an existing step.
    /// The new step's output is available by name to all downstream steps via DSL scoping.
    AddPromptStep {
        after_step: String,
        new_step_name: String,
        template: String,
        system: Option<String>,
        model: Option<String>,
    },

    /// Replace the entire graph with a new one written as .scaffold DSL source.
    /// Subsumes all structural mutations — the meta-agent gets full DSL expressiveness.
    EditGraph {
        new_graph: GraphIR,
        new_nodes: Vec<NodeIR>,
        description: String,
    },
}

impl Mutation {
    /// Human-readable short label for TUI display.
    pub fn short_label(&self) -> String {
        match self {
            Mutation::InsertVerify {
                after_step,
                verify_node,
                ..
            } => {
                format!("+verify({})@{}", verify_node, after_step)
            }
            Mutation::WrapRetry {
                step,
                verify_node,
                max_retries,
            } => {
                format!("retry({}x{})@{}", verify_node, max_retries, step)
            }
            Mutation::InsertStep {
                new_step_name: _,
                node,
                ..
            } => {
                format!("+step({})", node)
            }
            Mutation::RemoveStep { step } => format!("-step({})", step),
            Mutation::ReplaceComponent { step, new_node } => {
                format!("swap({}->{})", step, new_node)
            }
            Mutation::FanOutParallel { step, .. } => format!("fanout({})", step),
            Mutation::ReplaceWithSubgraph { step, graph } => {
                format!("subgraph({}->{})", step, graph)
            }
            Mutation::SetConfig { node, field, value } => {
                format!("cfg({}.{}={})", node, field, format_json_value(value))
            }
            Mutation::RewritePrompt { node, .. } => format!("rewrite_prompt({})", node),
            Mutation::RewriteSystem { node, .. } => format!("rewrite_system({})", node),
            Mutation::RewriteShell { node, .. } => format!("rewrite_shell({})", node),
            Mutation::AddPromptStep { new_step_name, .. } => format!("+prompt({})", new_step_name),
            Mutation::EditGraph { description, .. } => format!("edit: {}", description),
        }
    }
}

fn format_json_value(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => format!("{:?}", s),
        _ => value.to_string(),
    }
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

        Mutation::RewritePrompt { node, .. } => {
            // Prompt rewrites don't change graph structure — overrides are applied separately.
            // Accept both IR-defined nodes and synthetic nodes (steps referencing same-named node).
            if !ir.nodes.iter().any(|n| n.name == *node) && !node_used_by_step(&graph.body, node) {
                return MutationResult::Skipped(format!("node '{}' not found", node));
            }
            MutationResult::Ok(graph.clone())
        }

        Mutation::RewriteSystem { node, .. } => {
            if !ir.nodes.iter().any(|n| n.name == *node) && !node_used_by_step(&graph.body, node) {
                return MutationResult::Skipped(format!("node '{}' not found", node));
            }
            MutationResult::Ok(graph.clone())
        }

        Mutation::RewriteShell { node, .. } => match ir.nodes.iter().find(|n| n.name == *node) {
            None => MutationResult::Skipped(format!("node '{}' not found", node)),
            Some(n) if n.kind != NodeKindIR::Tool => {
                MutationResult::Skipped(format!("node '{}' is not a tool node", node))
            }
            _ => MutationResult::Ok(graph.clone()),
        },

        Mutation::AddPromptStep {
            after_step,
            new_step_name,
            ..
        } => {
            let input_fields = resolve_graph_input_fields(graph, ir);
            apply_add_prompt_step(graph, after_step, new_step_name, &input_fields)
        }

        Mutation::EditGraph { new_graph, .. } => {
            // Validation already done during construction in parse_and_validate_graph_edit.
            MutationResult::Ok(new_graph.clone())
        }
    }
}

/// Check if a mutation respects topology constraints.
pub fn check_constraints(graph: &GraphIR, topology: &TopologyIR) -> Vec<String> {
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
        GraphStmtIR::If(i) => {
            let mut then_body = Vec::new();
            let mut then_found = false;
            for s in &i.then_body {
                let (new_stmts, f) =
                    insert_verify_in_stmts(s, after_step, verify_node, max_retries);
                then_body.extend(new_stmts);
                then_found = then_found || f;
            }
            let mut else_body = Vec::new();
            let mut else_found = false;
            for s in &i.else_body {
                let (new_stmts, f) =
                    insert_verify_in_stmts(s, after_step, verify_node, max_retries);
                else_body.extend(new_stmts);
                else_found = else_found || f;
            }
            if then_found || else_found {
                (
                    vec![GraphStmtIR::If(IfIR {
                        cond: i.cond.clone(),
                        then_body,
                        else_body,
                    })],
                    true,
                )
            } else {
                (vec![stmt.clone()], false)
            }
        }
        GraphStmtIR::Loop(l) => {
            let mut body = Vec::new();
            let mut found = false;
            for s in &l.body {
                let (new_stmts, f) =
                    insert_verify_in_stmts(s, after_step, verify_node, max_retries);
                body.extend(new_stmts);
                found = found || f;
            }
            if found {
                (
                    vec![GraphStmtIR::Loop(LoopIR {
                        max: l.max.clone(),
                        while_cond: l.while_cond.clone(),
                        body,
                    })],
                    true,
                )
            } else {
                (vec![stmt.clone()], false)
            }
        }
        GraphStmtIR::Parallel(p) => {
            let mut body = Vec::new();
            let mut found = false;
            for s in &p.body {
                let (new_stmts, f) =
                    insert_verify_in_stmts(s, after_step, verify_node, max_retries);
                body.extend(new_stmts);
                found = found || f;
            }
            if found {
                (
                    vec![GraphStmtIR::Parallel(ParallelIR {
                        var: p.var.clone(),
                        collection: p.collection.clone(),
                        reduce: p.reduce.clone(),
                        body,
                    })],
                    true,
                )
            } else {
                (vec![stmt.clone()], false)
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
    let (new_body, found) = wrap_retry_in_stmts(&graph.body, step_name, verify_node, max_retries);

    if !found {
        return MutationResult::Skipped(format!("step '{}' not found", step_name));
    }

    MutationResult::Ok(GraphIR {
        body: new_body,
        ..graph.clone()
    })
}

fn wrap_retry_in_stmts(
    stmts: &[GraphStmtIR],
    step_name: &str,
    verify_node: &str,
    max_retries: u32,
) -> (Vec<GraphStmtIR>, bool) {
    let mut result = Vec::new();
    let mut found = false;
    for stmt in stmts {
        match stmt {
            GraphStmtIR::Step(s) if s.name == step_name => {
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
                result.push(GraphStmtIR::Loop(LoopIR {
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
            }
            GraphStmtIR::If(i) => {
                let (then_body, f1) =
                    wrap_retry_in_stmts(&i.then_body, step_name, verify_node, max_retries);
                let (else_body, f2) =
                    wrap_retry_in_stmts(&i.else_body, step_name, verify_node, max_retries);
                result.push(GraphStmtIR::If(IfIR {
                    cond: i.cond.clone(),
                    then_body,
                    else_body,
                }));
                found = found || f1 || f2;
            }
            GraphStmtIR::Loop(l) => {
                let (body, f) = wrap_retry_in_stmts(&l.body, step_name, verify_node, max_retries);
                result.push(GraphStmtIR::Loop(LoopIR {
                    max: l.max.clone(),
                    while_cond: l.while_cond.clone(),
                    body,
                }));
                found = found || f;
            }
            GraphStmtIR::Parallel(p) => {
                let (body, f) = wrap_retry_in_stmts(&p.body, step_name, verify_node, max_retries);
                result.push(GraphStmtIR::Parallel(ParallelIR {
                    var: p.var.clone(),
                    collection: p.collection.clone(),
                    reduce: p.reduce.clone(),
                    body,
                }));
                found = found || f;
            }
            _ => result.push(stmt.clone()),
        }
    }
    (result, found)
}

fn apply_insert_step(
    graph: &GraphIR,
    after_step: &str,
    new_name: &str,
    node: &str,
) -> MutationResult {
    let (new_body, found) = insert_step_in_stmts(&graph.body, after_step, new_name, node);

    if !found {
        return MutationResult::Skipped(format!("step '{}' not found", after_step));
    }

    MutationResult::Ok(GraphIR {
        body: new_body,
        ..graph.clone()
    })
}

fn insert_step_in_stmts(
    stmts: &[GraphStmtIR],
    after_step: &str,
    new_name: &str,
    node: &str,
) -> (Vec<GraphStmtIR>, bool) {
    let mut result = Vec::new();
    let mut found = false;
    for stmt in stmts {
        match stmt {
            GraphStmtIR::Step(s) if s.name == after_step => {
                result.push(stmt.clone());
                result.push(GraphStmtIR::Step(StepIR {
                    name: new_name.to_string(),
                    node: node.to_string(),
                    args: vec![StepArgIR::Positional {
                        value: ExprIR::Ident {
                            name: s.name.clone(),
                        },
                    }],
                }));
                found = true;
            }
            GraphStmtIR::If(i) => {
                let (then_body, f1) =
                    insert_step_in_stmts(&i.then_body, after_step, new_name, node);
                let (else_body, f2) =
                    insert_step_in_stmts(&i.else_body, after_step, new_name, node);
                result.push(GraphStmtIR::If(IfIR {
                    cond: i.cond.clone(),
                    then_body,
                    else_body,
                }));
                found = found || f1 || f2;
            }
            GraphStmtIR::Loop(l) => {
                let (body, f) = insert_step_in_stmts(&l.body, after_step, new_name, node);
                result.push(GraphStmtIR::Loop(LoopIR {
                    max: l.max.clone(),
                    while_cond: l.while_cond.clone(),
                    body,
                }));
                found = found || f;
            }
            GraphStmtIR::Parallel(p) => {
                let (body, f) = insert_step_in_stmts(&p.body, after_step, new_name, node);
                result.push(GraphStmtIR::Parallel(ParallelIR {
                    var: p.var.clone(),
                    collection: p.collection.clone(),
                    reduce: p.reduce.clone(),
                    body,
                }));
                found = found || f;
            }
            _ => result.push(stmt.clone()),
        }
    }
    (result, found)
}

fn apply_remove_step(graph: &GraphIR, step_name: &str) -> MutationResult {
    let (new_body, found) = remove_step_in_stmts(&graph.body, step_name);

    if !found {
        return MutationResult::Skipped(format!("step '{}' not found", step_name));
    }

    MutationResult::Ok(GraphIR {
        body: new_body,
        ..graph.clone()
    })
}

fn remove_step_in_stmts(stmts: &[GraphStmtIR], step_name: &str) -> (Vec<GraphStmtIR>, bool) {
    let mut result = Vec::new();
    let mut found = false;
    for stmt in stmts {
        match stmt {
            GraphStmtIR::Step(s) if s.name == step_name => {
                found = true;
                // skip — remove it
            }
            GraphStmtIR::If(i) => {
                let (then_body, f1) = remove_step_in_stmts(&i.then_body, step_name);
                let (else_body, f2) = remove_step_in_stmts(&i.else_body, step_name);
                result.push(GraphStmtIR::If(IfIR {
                    cond: i.cond.clone(),
                    then_body,
                    else_body,
                }));
                found = found || f1 || f2;
            }
            GraphStmtIR::Loop(l) => {
                let (body, f) = remove_step_in_stmts(&l.body, step_name);
                result.push(GraphStmtIR::Loop(LoopIR {
                    max: l.max.clone(),
                    while_cond: l.while_cond.clone(),
                    body,
                }));
                found = found || f;
            }
            GraphStmtIR::Parallel(p) => {
                let (body, f) = remove_step_in_stmts(&p.body, step_name);
                result.push(GraphStmtIR::Parallel(ParallelIR {
                    var: p.var.clone(),
                    collection: p.collection.clone(),
                    reduce: p.reduce.clone(),
                    body,
                }));
                found = found || f;
            }
            _ => result.push(stmt.clone()),
        }
    }
    (result, found)
}

fn apply_replace_component(graph: &GraphIR, step_name: &str, new_node: &str) -> MutationResult {
    let (new_body, found) = replace_node_in_stmts(&graph.body, step_name, new_node);

    if !found {
        return MutationResult::Skipped(format!("step '{}' not found", step_name));
    }

    MutationResult::Ok(GraphIR {
        body: new_body,
        ..graph.clone()
    })
}

/// Recursively replace the node a step invokes. Used by both ReplaceComponent and ReplaceWithSubgraph.
fn replace_node_in_stmts(
    stmts: &[GraphStmtIR],
    step_name: &str,
    new_node: &str,
) -> (Vec<GraphStmtIR>, bool) {
    let mut result = Vec::new();
    let mut found = false;
    for stmt in stmts {
        match stmt {
            GraphStmtIR::Step(s) if s.name == step_name => {
                result.push(GraphStmtIR::Step(StepIR {
                    name: s.name.clone(),
                    node: new_node.to_string(),
                    args: s.args.clone(),
                }));
                found = true;
            }
            GraphStmtIR::If(i) => {
                let (then_body, f1) = replace_node_in_stmts(&i.then_body, step_name, new_node);
                let (else_body, f2) = replace_node_in_stmts(&i.else_body, step_name, new_node);
                result.push(GraphStmtIR::If(IfIR {
                    cond: i.cond.clone(),
                    then_body,
                    else_body,
                }));
                found = found || f1 || f2;
            }
            GraphStmtIR::Loop(l) => {
                let (body, f) = replace_node_in_stmts(&l.body, step_name, new_node);
                result.push(GraphStmtIR::Loop(LoopIR {
                    max: l.max.clone(),
                    while_cond: l.while_cond.clone(),
                    body,
                }));
                found = found || f;
            }
            GraphStmtIR::Parallel(p) => {
                let (body, f) = replace_node_in_stmts(&p.body, step_name, new_node);
                result.push(GraphStmtIR::Parallel(ParallelIR {
                    var: p.var.clone(),
                    collection: p.collection.clone(),
                    reduce: p.reduce.clone(),
                    body,
                }));
                found = found || f;
            }
            _ => result.push(stmt.clone()),
        }
    }
    (result, found)
}

fn apply_fan_out(
    graph: &GraphIR,
    step_name: &str,
    split_node: &str,
    reduce_node: &str,
) -> MutationResult {
    let (new_body, found) = fan_out_in_stmts(&graph.body, step_name, split_node, reduce_node);

    if !found {
        return MutationResult::Skipped(format!("step '{}' not found", step_name));
    }

    MutationResult::Ok(GraphIR {
        body: new_body,
        ..graph.clone()
    })
}

fn fan_out_in_stmts(
    stmts: &[GraphStmtIR],
    step_name: &str,
    split_node: &str,
    reduce_node: &str,
) -> (Vec<GraphStmtIR>, bool) {
    let mut result = Vec::new();
    let mut found = false;
    for stmt in stmts {
        match stmt {
            GraphStmtIR::Step(s) if s.name == step_name => {
                found = true;
                let split_step_name = format!("{}_split", s.name);
                result.push(GraphStmtIR::Step(StepIR {
                    name: split_step_name.clone(),
                    node: split_node.to_string(),
                    args: s.args.clone(),
                }));
                result.push(GraphStmtIR::Parallel(ParallelIR {
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
            }
            GraphStmtIR::If(i) => {
                let (then_body, f1) =
                    fan_out_in_stmts(&i.then_body, step_name, split_node, reduce_node);
                let (else_body, f2) =
                    fan_out_in_stmts(&i.else_body, step_name, split_node, reduce_node);
                result.push(GraphStmtIR::If(IfIR {
                    cond: i.cond.clone(),
                    then_body,
                    else_body,
                }));
                found = found || f1 || f2;
            }
            GraphStmtIR::Loop(l) => {
                let (body, f) = fan_out_in_stmts(&l.body, step_name, split_node, reduce_node);
                result.push(GraphStmtIR::Loop(LoopIR {
                    max: l.max.clone(),
                    while_cond: l.while_cond.clone(),
                    body,
                }));
                found = found || f;
            }
            GraphStmtIR::Parallel(p) => {
                let (body, f) = fan_out_in_stmts(&p.body, step_name, split_node, reduce_node);
                result.push(GraphStmtIR::Parallel(ParallelIR {
                    var: p.var.clone(),
                    collection: p.collection.clone(),
                    reduce: p.reduce.clone(),
                    body,
                }));
                found = found || f;
            }
            _ => result.push(stmt.clone()),
        }
    }
    (result, found)
}

fn apply_replace_with_subgraph(graph: &GraphIR, step_name: &str, subgraph: &str) -> MutationResult {
    let (new_body, found) = replace_node_in_stmts(&graph.body, step_name, subgraph);

    if !found {
        return MutationResult::Skipped(format!("step '{}' not found", step_name));
    }

    MutationResult::Ok(GraphIR {
        body: new_body,
        ..graph.clone()
    })
}

fn apply_set_config(
    graph: &GraphIR,
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

    // Graph unchanged — the optimizer stores the override separately.
    MutationResult::Ok(graph.clone())
}

/// Resolve the field names of a graph's input type.
/// Returns field names for struct types (inline or via Named reference), empty for scalars.
pub fn resolve_graph_input_fields(graph: &GraphIR, ir: &ScaffoldIR) -> Vec<String> {
    resolve_type_fields_for_graph(&graph.input, ir)
}

fn resolve_type_fields_for_graph(ty: &TypeIR, ir: &ScaffoldIR) -> Vec<String> {
    match ty {
        TypeIR::Struct { fields } => fields.iter().map(|f| f.name.clone()).collect(),
        TypeIR::Named { name } => ir
            .types
            .iter()
            .find(|t| t.name == *name)
            .map(|td| resolve_type_fields_for_graph(&td.ty, ir))
            .unwrap_or_default(),
        _ => vec![],
    }
}

fn apply_add_prompt_step(
    graph: &GraphIR,
    after_step: &str,
    new_step_name: &str,
    graph_input_fields: &[String],
) -> MutationResult {
    let (new_body, found) =
        add_prompt_step_in_stmts(&graph.body, after_step, new_step_name, graph_input_fields);

    if !found {
        return MutationResult::Skipped(format!("step '{}' not found", after_step));
    }

    MutationResult::Ok(GraphIR {
        body: new_body,
        ..graph.clone()
    })
}

/// Insert a synthetic prompt step after `after_step`.
/// Recurses into If/Loop/Parallel to find the target step.
///
/// The new step's output is bound to `new_step_name` in scope and available to all
/// downstream steps — no rewiring is performed. The DSL's scoping rules mean any
/// subsequent step (or a future `rewrite_prompt` mutation) can reference the new step's
/// output by name. This keeps the insertion safe regardless of how `after_step`'s output
/// is used in expressions, conditions, or function calls.
///
/// When `graph_input_fields` is non-empty, the new step receives named args that forward
/// all graph input fields (e.g. `instructions: input.instructions`) in addition to the
/// `input` arg carrying the previous step's output. This lets the template reference
/// `{{ instructions }}`, `{{ stub_content }}`, etc.
fn add_prompt_step_in_stmts(
    stmts: &[GraphStmtIR],
    after_step: &str,
    new_step_name: &str,
    graph_input_fields: &[String],
) -> (Vec<GraphStmtIR>, bool) {
    let mut result = Vec::new();
    let mut found = false;

    for stmt in stmts {
        match stmt {
            GraphStmtIR::Step(s) if s.name == after_step && !found => {
                result.push(stmt.clone());

                // Build args: always include `input` (previous step output),
                // plus named args forwarding each graph input field.
                let mut args = vec![StepArgIR::Named {
                    name: "input".to_string(),
                    value: ExprIR::Ident {
                        name: s.name.clone(),
                    },
                }];
                for field in graph_input_fields {
                    args.push(StepArgIR::Named {
                        name: field.clone(),
                        value: ExprIR::FieldAccess {
                            base: Box::new(ExprIR::Ident {
                                name: "input".to_string(),
                            }),
                            field: field.clone(),
                        },
                    });
                }

                result.push(GraphStmtIR::Step(StepIR {
                    name: new_step_name.to_string(),
                    node: new_step_name.to_string(),
                    args,
                }));
                found = true;
            }
            GraphStmtIR::If(i) if !found => {
                let (then_body, f1) =
                    add_prompt_step_in_stmts(&i.then_body, after_step, new_step_name, graph_input_fields);
                let (else_body, f2) =
                    add_prompt_step_in_stmts(&i.else_body, after_step, new_step_name, graph_input_fields);
                result.push(GraphStmtIR::If(IfIR {
                    cond: i.cond.clone(),
                    then_body,
                    else_body,
                }));
                found = f1 || f2;
            }
            GraphStmtIR::Loop(l) if !found => {
                let (body, f) =
                    add_prompt_step_in_stmts(&l.body, after_step, new_step_name, graph_input_fields);
                result.push(GraphStmtIR::Loop(LoopIR {
                    max: l.max.clone(),
                    while_cond: l.while_cond.clone(),
                    body,
                }));
                found = f;
            }
            GraphStmtIR::Parallel(p) if !found => {
                let (body, f) =
                    add_prompt_step_in_stmts(&p.body, after_step, new_step_name, graph_input_fields);
                result.push(GraphStmtIR::Parallel(ParallelIR {
                    var: p.var.clone(),
                    collection: p.collection.clone(),
                    reduce: p.reduce.clone(),
                    body,
                }));
                found = f;
            }
            _ => result.push(stmt.clone()),
        }
    }
    (result, found)
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

/// Public accessor for `node_used_by_step` (used by meta_agent for synthetic node validation).
pub fn node_used_by_step_pub(stmts: &[GraphStmtIR], node_name: &str) -> bool {
    node_used_by_step(stmts, node_name)
}

/// Public accessor for `step_exists` (used by meta_agent for preserve validation).
pub fn step_exists_pub(stmts: &[GraphStmtIR], name: &str) -> bool {
    step_exists(stmts, name)
}

/// Check if any step in the graph body references a node by name.
/// Used to detect synthetic nodes (from AddPromptStep) that aren't in ir.nodes.
fn node_used_by_step(stmts: &[GraphStmtIR], node_name: &str) -> bool {
    for stmt in stmts {
        match stmt {
            GraphStmtIR::Step(s) if s.node == node_name => return true,
            GraphStmtIR::Loop(l) => {
                if node_used_by_step(&l.body, node_name) {
                    return true;
                }
            }
            GraphStmtIR::If(i) => {
                if node_used_by_step(&i.then_body, node_name)
                    || node_used_by_step(&i.else_body, node_name)
                {
                    return true;
                }
            }
            GraphStmtIR::Parallel(p) => {
                if node_used_by_step(&p.body, node_name) {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
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

/// Collect graph names referenced by step calls (where node name is in `known_graphs`).
pub fn collect_graph_refs_from_body(
    stmts: &[GraphStmtIR],
    known_graphs: &std::collections::HashSet<String>,
    refs: &mut std::collections::HashSet<String>,
) {
    for stmt in stmts {
        match stmt {
            GraphStmtIR::Step(s) => {
                if known_graphs.contains(&s.node) {
                    refs.insert(s.node.clone());
                }
            }
            GraphStmtIR::Loop(l) => collect_graph_refs_from_body(&l.body, known_graphs, refs),
            GraphStmtIR::If(i) => {
                collect_graph_refs_from_body(&i.then_body, known_graphs, refs);
                collect_graph_refs_from_body(&i.else_body, known_graphs, refs);
            }
            GraphStmtIR::Parallel(p) => collect_graph_refs_from_body(&p.body, known_graphs, refs),
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

/// Compute a structural fingerprint of a graph (for novelty search).
pub fn topology_hash(graph: &GraphIR) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hasher;
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
            target_score: None,
        };
        let violations = check_constraints(&graph, &topo);
        assert!(violations.is_empty()); // 1 step <= max_nodes=1

        let topo2 = TopologyIR {
            mutations: vec![],
            max_nodes: Some(0),
            max_depth: None,
            preserve: vec!["s1".to_string()],
            target_score: None,
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

    #[test]
    fn test_set_config_short_label_includes_value() {
        let mutation = Mutation::SetConfig {
            node: "fix_code".to_string(),
            field: "temperature".to_string(),
            value: serde_json::json!(0.2),
        };
        assert_eq!(mutation.short_label(), "cfg(fix_code.temperature=0.2)");
    }

    /// Helper: graph with a step inside an if-block (the bug scenario).
    fn make_nested_graph() -> GraphIR {
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
                GraphStmtIR::If(IfIR {
                    cond: ExprIR::LitBool { value: true },
                    then_body: vec![GraphStmtIR::Step(StepIR {
                        name: "s2".to_string(),
                        node: "solver".to_string(),
                        args: vec![StepArgIR::Positional {
                            value: ExprIR::Ident {
                                name: "s1".to_string(),
                            },
                        }],
                    })],
                    else_body: vec![],
                }),
                GraphStmtIR::Emit(EmitIR::Direct {
                    value: ExprIR::Ident {
                        name: "s2".to_string(),
                    },
                }),
            ],
        }
    }

    #[test]
    fn test_replace_component_nested() {
        let graph = make_nested_graph();
        match apply_replace_component(&graph, "s2", "checker") {
            MutationResult::Ok(new_graph) => {
                // s2 is inside the if-then; verify it was replaced
                if let GraphStmtIR::If(i) = &new_graph.body[1] {
                    if let GraphStmtIR::Step(s) = &i.then_body[0] {
                        assert_eq!(s.name, "s2");
                        assert_eq!(s.node, "checker");
                    } else {
                        panic!("expected step in then_body");
                    }
                } else {
                    panic!("expected if");
                }
            }
            MutationResult::Skipped(msg) => panic!("unexpected skip: {}", msg),
        }
    }

    #[test]
    fn test_remove_step_nested() {
        let graph = make_nested_graph();
        match apply_remove_step(&graph, "s2") {
            MutationResult::Ok(new_graph) => {
                let names = collect_step_names(&new_graph.body);
                assert!(names.contains(&"s1".to_string()));
                assert!(!names.contains(&"s2".to_string()));
            }
            MutationResult::Skipped(msg) => panic!("unexpected skip: {}", msg),
        }
    }

    #[test]
    fn test_insert_step_nested() {
        let graph = make_nested_graph();
        match apply_insert_step(&graph, "s2", "s3", "solver") {
            MutationResult::Ok(new_graph) => {
                let names = collect_step_names(&new_graph.body);
                assert!(names.contains(&"s3".to_string()));
                // s3 should be inside the if-then body
                if let GraphStmtIR::If(i) = &new_graph.body[1] {
                    assert_eq!(i.then_body.len(), 2); // s2, s3
                } else {
                    panic!("expected if");
                }
            }
            MutationResult::Skipped(msg) => panic!("unexpected skip: {}", msg),
        }
    }

    #[test]
    fn test_wrap_retry_nested() {
        let graph = make_nested_graph();
        match apply_wrap_retry(&graph, "s2", "checker", 2) {
            MutationResult::Ok(new_graph) => {
                // s2 should be wrapped in a loop inside the if-then body
                if let GraphStmtIR::If(i) = &new_graph.body[1] {
                    assert!(matches!(&i.then_body[0], GraphStmtIR::Loop(_)));
                } else {
                    panic!("expected if");
                }
            }
            MutationResult::Skipped(msg) => panic!("unexpected skip: {}", msg),
        }
    }

    #[test]
    fn test_add_prompt_step_nested_no_rewire() {
        // s2 is inside an if-then block, emit at outer level references s2.
        // add_prompt_step after s2 should:
        //   1. Insert "review" step inside the if-then after s2
        //   2. Leave outer emit unchanged (no rewiring — DSL scoping handles references)
        let graph = make_nested_graph();
        // Graph has scalar (String) input, so no graph input fields forwarded.
        match apply_add_prompt_step(&graph, "s2", "review", &[]) {
            MutationResult::Ok(new_graph) => {
                // New step should be inside the if-then body
                if let GraphStmtIR::If(i) = &new_graph.body[1] {
                    assert_eq!(i.then_body.len(), 2); // s2, review
                    if let GraphStmtIR::Step(s) = &i.then_body[1] {
                        assert_eq!(s.name, "review");
                        assert_eq!(s.node, "review");
                        // Should have named `input` arg
                        assert_eq!(s.args.len(), 1);
                        match &s.args[0] {
                            StepArgIR::Named { name, value } => {
                                assert_eq!(name, "input");
                                assert!(matches!(value, ExprIR::Ident { name } if name == "s2"));
                            }
                            _ => panic!("expected named arg 'input'"),
                        }
                    } else {
                        panic!("expected review step");
                    }
                } else {
                    panic!("expected if");
                }
                // Outer emit should still reference "s2" (no rewiring)
                if let GraphStmtIR::Emit(EmitIR::Direct { value }) = &new_graph.body[2] {
                    if let ExprIR::Ident { name } = value {
                        assert_eq!(name, "s2", "outer emit should NOT be rewired");
                    } else {
                        panic!("expected ident in emit");
                    }
                } else {
                    panic!("expected emit");
                }
            }
            MutationResult::Skipped(msg) => panic!("unexpected skip: {}", msg),
        }
    }

    #[test]
    fn test_add_prompt_step_top_level() {
        // Verify top-level case: insert after s1, emit stays unchanged (no rewiring).
        let graph = make_test_graph();
        // Graph has scalar (String) input, so no graph input fields forwarded.
        match apply_add_prompt_step(&graph, "s1", "review", &[]) {
            MutationResult::Ok(new_graph) => {
                assert_eq!(new_graph.body.len(), 3); // s1, review, emit
                // Check the new step has named `input` arg
                if let GraphStmtIR::Step(s) = &new_graph.body[1] {
                    assert_eq!(s.name, "review");
                    assert_eq!(s.args.len(), 1);
                    match &s.args[0] {
                        StepArgIR::Named { name, value } => {
                            assert_eq!(name, "input");
                            assert!(matches!(value, ExprIR::Ident { name } if name == "s1"));
                        }
                        _ => panic!("expected named arg 'input'"),
                    }
                } else {
                    panic!("expected step");
                }
                // Emit should still reference "s1" (no rewiring — DSL scoping handles references)
                if let GraphStmtIR::Emit(EmitIR::Direct { value }) = &new_graph.body[2] {
                    if let ExprIR::Ident { name } = value {
                        assert_eq!(name, "s1", "emit should NOT be rewired");
                    } else {
                        panic!("expected ident in emit");
                    }
                } else {
                    panic!("expected emit");
                }
            }
            MutationResult::Skipped(msg) => panic!("unexpected skip: {}", msg),
        }
    }

    #[test]
    fn test_add_prompt_step_forwards_graph_inputs() {
        // Graph with struct input — verify field forwarding.
        let graph = GraphIR {
            name: "solve".to_string(),
            input: TypeIR::Struct {
                fields: vec![
                    FieldIR {
                        name: "instructions".to_string(),
                        ty: TypeIR::String,
                    },
                    FieldIR {
                        name: "stub_content".to_string(),
                        ty: TypeIR::String,
                    },
                ],
            },
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
        };
        let fields = vec!["instructions".to_string(), "stub_content".to_string()];
        match apply_add_prompt_step(&graph, "s1", "review", &fields) {
            MutationResult::Ok(new_graph) => {
                assert_eq!(new_graph.body.len(), 3); // s1, review, emit
                if let GraphStmtIR::Step(s) = &new_graph.body[1] {
                    assert_eq!(s.name, "review");
                    // Should have 3 named args: input, instructions, stub_content
                    assert_eq!(s.args.len(), 3);
                    // First: input = s1
                    match &s.args[0] {
                        StepArgIR::Named { name, value } => {
                            assert_eq!(name, "input");
                            assert!(matches!(value, ExprIR::Ident { name } if name == "s1"));
                        }
                        _ => panic!("expected named arg 'input'"),
                    }
                    // Second: instructions = input.instructions
                    match &s.args[1] {
                        StepArgIR::Named { name, value } => {
                            assert_eq!(name, "instructions");
                            match value {
                                ExprIR::FieldAccess { base, field } => {
                                    assert!(matches!(base.as_ref(), ExprIR::Ident { name } if name == "input"));
                                    assert_eq!(field, "instructions");
                                }
                                _ => panic!("expected field access for instructions"),
                            }
                        }
                        _ => panic!("expected named arg 'instructions'"),
                    }
                    // Third: stub_content = input.stub_content
                    match &s.args[2] {
                        StepArgIR::Named { name, value } => {
                            assert_eq!(name, "stub_content");
                            match value {
                                ExprIR::FieldAccess { base, field } => {
                                    assert!(matches!(base.as_ref(), ExprIR::Ident { name } if name == "input"));
                                    assert_eq!(field, "stub_content");
                                }
                                _ => panic!("expected field access for stub_content"),
                            }
                        }
                        _ => panic!("expected named arg 'stub_content'"),
                    }
                } else {
                    panic!("expected step");
                }
            }
            MutationResult::Skipped(msg) => panic!("unexpected skip: {}", msg),
        }
    }

    #[test]
    fn test_resolve_graph_input_fields_named_type() {
        let ir = ScaffoldIR {
            version: "2.0.0".to_string(),
            types: vec![TypeDefIR {
                name: "ExerciseInput".to_string(),
                ty: TypeIR::Struct {
                    fields: vec![
                        FieldIR {
                            name: "instructions".to_string(),
                            ty: TypeIR::String,
                        },
                        FieldIR {
                            name: "language".to_string(),
                            ty: TypeIR::String,
                        },
                    ],
                },
            }],
            nodes: vec![],
            graphs: vec![],
            objectives: vec![],
        };
        let graph = GraphIR {
            name: "solve".to_string(),
            input: TypeIR::Named {
                name: "ExerciseInput".to_string(),
            },
            output: TypeIR::String,
            body: vec![],
        };
        let fields = resolve_graph_input_fields(&graph, &ir);
        assert_eq!(fields, vec!["instructions", "language"]);
    }

    #[test]
    fn test_resolve_graph_input_fields_scalar() {
        let ir = make_test_ir();
        let graph = make_test_graph(); // has TypeIR::String input
        let fields = resolve_graph_input_fields(&graph, &ir);
        assert!(fields.is_empty());
    }

    #[test]
    fn test_edit_graph_apply_returns_new_graph() {
        let ir = make_test_ir();
        let graph = make_test_graph();
        let new_graph = GraphIR {
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
                GraphStmtIR::Step(StepIR {
                    name: "s2".to_string(),
                    node: "checker".to_string(),
                    args: vec![StepArgIR::Positional {
                        value: ExprIR::Ident {
                            name: "s1".to_string(),
                        },
                    }],
                }),
                GraphStmtIR::Emit(EmitIR::Direct {
                    value: ExprIR::Ident {
                        name: "s2".to_string(),
                    },
                }),
            ],
        };
        let mutation = Mutation::EditGraph {
            new_graph: new_graph.clone(),
            new_nodes: vec![],
            description: "add checker step".to_string(),
        };
        match apply_mutation(&graph, &mutation, &ir) {
            MutationResult::Ok(result) => {
                assert_eq!(result.body.len(), 3); // s1, s2, emit
                let names = collect_step_names(&result.body);
                assert!(names.contains(&"s1".to_string()));
                assert!(names.contains(&"s2".to_string()));
            }
            MutationResult::Skipped(msg) => panic!("unexpected skip: {}", msg),
        }
    }

    #[test]
    fn test_edit_graph_short_label() {
        let mutation = Mutation::EditGraph {
            new_graph: make_test_graph(),
            new_nodes: vec![],
            description: "add planning step".to_string(),
        };
        assert_eq!(mutation.short_label(), "edit: add planning step");
    }
}
