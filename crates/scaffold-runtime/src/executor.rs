//! Graph execution engine for scaffold v2
//!
//! Walks `GraphStmtIR` lists, resolving steps, loops, conditionals,
//! parallel fan-out, choose, emit, and carry statements.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

use scaffold_ir::ir::*;

use crate::error::{Error, Result};
use crate::node_runner;
use crate::prompt::PromptManager;
use crate::scope::Scope;
use crate::value::Value;

/// Configuration overrides for tunables (node.field → value).
pub type TunableOverrides = HashMap<String, serde_json::Value>;

/// The graph executor: holds the full IR, prompt manager, and tunable overrides.
pub struct GraphExecutor {
    ir: ScaffoldIR,
    prompt_mgr: PromptManager,
    overrides: TunableOverrides,
    /// Synthetic nodes created by `AddPromptStep` mutations (stored via `_node.*` overrides).
    synthetic_nodes: Vec<NodeIR>,
}

impl GraphExecutor {
    /// Create a new executor from an IR.
    pub fn new(ir: ScaffoldIR) -> Self {
        Self {
            ir,
            prompt_mgr: PromptManager::new(),
            overrides: HashMap::new(),
            synthetic_nodes: Vec::new(),
        }
    }

    /// Set the prompt manager (for loading templates from a directory).
    pub fn with_prompt_manager(mut self, pm: PromptManager) -> Self {
        self.prompt_mgr = pm;
        self
    }

    /// Set tunable overrides.
    ///
    /// Also scans for `_node.<name>` keys to create synthetic nodes
    /// (used by `AddPromptStep` mutations).
    pub fn with_overrides(mut self, overrides: TunableOverrides) -> Self {
        // Parse synthetic nodes from `_node.<name>` override keys.
        let mut synthetic = Vec::new();
        for key in overrides.keys() {
            if let Some(name) = key.strip_prefix("_node.") {
                synthetic.push(NodeIR {
                    name: name.to_string(),
                    kind: NodeKindIR::Prompt,
                    input: TypeIR::String,
                    output: TypeIR::String,
                    config: NodeConfigIR::default(),
                });
            }
        }
        self.synthetic_nodes = synthetic;
        self.overrides = overrides;
        self
    }

    /// Execute a named graph with the given input.
    pub fn execute_graph<'a>(
        &'a self,
        graph_name: &'a str,
        input: Value,
    ) -> Pin<Box<dyn Future<Output = Result<Value>> + 'a>> {
        Box::pin(async move {
            let graph = self
                .ir
                .graphs
                .iter()
                .find(|g| g.name == graph_name)
                .ok_or_else(|| Error::UnknownNode(graph_name.to_string()))?;

            let mut scope = Scope::root(input);
            self.execute_stmts(&graph.body, &mut scope, graph_name)
                .await
        })
    }

    /// Execute a named graph with step-level output tracing.
    ///
    /// Returns the emitted value plus a Vec of (step_name, truncated_output)
    /// for every step executed (including inside if/parallel blocks).
    pub fn execute_graph_traced<'a>(
        &'a self,
        graph_name: &'a str,
        input: Value,
    ) -> Pin<Box<dyn Future<Output = Result<(Value, Vec<(String, String)>)>> + 'a>> {
        Box::pin(async move {
            let graph = self
                .ir
                .graphs
                .iter()
                .find(|g| g.name == graph_name)
                .ok_or_else(|| Error::UnknownNode(graph_name.to_string()))?;

            let trace = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
            let mut scope = Scope::root_traced(input, trace.clone());
            let result = self
                .execute_stmts(&graph.body, &mut scope, graph_name)
                .await?;
            let steps = match std::sync::Arc::try_unwrap(trace) {
                Ok(mutex) => mutex.into_inner().unwrap_or_default(),
                Err(arc) => arc.lock().unwrap().clone(),
            };
            Ok((result, steps))
        })
    }

    /// Execute a list of graph statements, returning the emitted value.
    fn execute_stmts<'a>(
        &'a self,
        stmts: &'a [GraphStmtIR],
        scope: &'a mut Scope,
        graph_name: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Value>> + 'a>> {
        Box::pin(async move {
            let mut emitted: Option<Value> = None;

            for stmt in stmts {
                match stmt {
                    GraphStmtIR::Step(step) => {
                        let value = self.execute_step(step, scope, graph_name).await?;
                        scope.bind(&step.name, value);
                    }
                    GraphStmtIR::Loop(loop_ir) => {
                        let result = self.execute_loop(loop_ir, scope, graph_name).await?;
                        if result.is_some() {
                            emitted = result;
                        }
                    }
                    GraphStmtIR::If(if_ir) => {
                        let result = self.execute_if(if_ir, scope, graph_name).await?;
                        if result.is_some() {
                            emitted = result;
                        }
                    }
                    GraphStmtIR::Choose(choose_ir) => {
                        let result = self.execute_choose(choose_ir, scope, graph_name).await?;
                        if result.is_some() {
                            emitted = result;
                        }
                    }
                    GraphStmtIR::Parallel(par_ir) => {
                        let result = self.execute_parallel(par_ir, scope, graph_name).await?;
                        if result.is_some() {
                            emitted = result;
                        }
                    }
                    GraphStmtIR::Emit(emit_ir) => {
                        let value = self.evaluate_emit(emit_ir, scope)?;
                        emitted = Some(value);
                    }
                    GraphStmtIR::Carry(carry_ir) => {
                        let value = self.eval_expr(&carry_ir.value, scope)?;
                        scope.bind(&carry_ir.name, value);
                    }
                }
            }

            emitted.ok_or(Error::NoEmit)
        })
    }

    // ── Step ──

    fn execute_step<'a>(
        &'a self,
        step: &'a StepIR,
        scope: &'a Scope,
        graph_name: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Value>> + 'a>> {
        Box::pin(async move {
            // Check if the step references a graph (subgraph call) or a node
            if self.ir.graphs.iter().any(|g| g.name == step.node) {
                // Subgraph call: resolve args into input
                let input = self.resolve_step_input(&step.args, scope)?;
                return self.execute_graph(&step.node, input).await;
            }

            // Node call (check IR nodes first, then synthetic nodes from AddPromptStep)
            let node = self
                .ir
                .nodes
                .iter()
                .find(|n| n.name == step.node)
                .or_else(|| self.synthetic_nodes.iter().find(|n| n.name == step.node))
                .ok_or_else(|| Error::UnknownNode(step.node.clone()))?;

            let input = self.resolve_step_input(&step.args, scope)?;

            // Collect overrides for this node
            let node_overrides = self.get_node_overrides(&node.name);

            // Handle error strategy
            let strategy = node.config.on_error.as_ref();
            match strategy {
                Some(ErrorStrategyIR::Retry { max }) => {
                    let max_retries = *max;
                    let mut last_err = None;
                    for attempt in 0..=max_retries {
                        match node_runner::run_node(
                            node,
                            input.clone(),
                            &node_overrides,
                            &self.prompt_mgr,
                        )
                        .await
                        {
                            Ok(val) => return Ok(val),
                            Err(e) => {
                                last_err = Some(e);
                                if attempt < max_retries {
                                    eprintln!(
                                        "[warn] graph={} step={} attempt={}: step failed, retrying",
                                        graph_name,
                                        step.name,
                                        attempt + 1
                                    );
                                }
                            }
                        }
                    }
                    Err(last_err.unwrap_or_else(|| Error::MaxRetriesExceeded {
                        step: step.name.clone(),
                        count: max_retries,
                    }))
                }
                Some(ErrorStrategyIR::Abort) | None => {
                    node_runner::run_node(node, input, &node_overrides, &self.prompt_mgr)
                        .await
                        .map_err(|e| Error::StepFailed {
                            step: step.name.clone(),
                            message: e.to_string(),
                        })
                }
            }
        })
    }

    /// Resolve step arguments into a single input Value.
    fn resolve_step_input(&self, args: &[StepArgIR], scope: &Scope) -> Result<Value> {
        if args.is_empty() {
            // Pass the current scope's input
            return Ok(scope.get("input").cloned().unwrap_or(Value::Null));
        }

        if args.len() == 1 {
            match &args[0] {
                StepArgIR::Positional { value } => return self.eval_expr(value, scope),
                StepArgIR::Named { name, value } => {
                    let mut map = HashMap::new();
                    map.insert(name.clone(), self.eval_expr(value, scope)?);
                    return Ok(Value::Map(map));
                }
            }
        }

        // Multiple args → build a map
        let mut map = HashMap::new();
        for (i, arg) in args.iter().enumerate() {
            match arg {
                StepArgIR::Positional { value } => {
                    let val = self.eval_expr(value, scope)?;
                    map.insert(format!("arg{}", i), val);
                }
                StepArgIR::Named { name, value } => {
                    map.insert(name.clone(), self.eval_expr(value, scope)?);
                }
            }
        }
        Ok(Value::Map(map))
    }

    // ── Loop ──

    fn execute_loop<'a>(
        &'a self,
        loop_ir: &'a LoopIR,
        scope: &'a mut Scope,
        graph_name: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Option<Value>>> + 'a>> {
        Box::pin(async move {
            let max = self.eval_expr_as_int(&loop_ir.max, scope)?;
            let mut emitted: Option<Value> = None;

            for iteration in 0..max {
                // Check while condition
                let cond = self.eval_expr_as_bool(&loop_ir.while_cond, scope)?;
                if !cond {
                    break;
                }

                // Bind iteration number
                scope.bind("iteration", Value::Int(iteration));

                // Execute loop body in the same scope (carries update the scope)
                match self.execute_stmts(&loop_ir.body, scope, graph_name).await {
                    Ok(val) => emitted = Some(val),
                    Err(Error::NoEmit) => {}
                    Err(e) => return Err(e),
                }
            }

            Ok(emitted)
        })
    }

    // ── If ──

    fn execute_if<'a>(
        &'a self,
        if_ir: &'a IfIR,
        scope: &'a mut Scope,
        graph_name: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Option<Value>>> + 'a>> {
        Box::pin(async move {
            let cond = self.eval_expr_as_bool(&if_ir.cond, scope)?;

            let body = if cond {
                &if_ir.then_body
            } else {
                &if_ir.else_body
            };

            if body.is_empty() {
                return Ok(None);
            }

            let mut child_scope = scope.child();
            match self.execute_stmts(body, &mut child_scope, graph_name).await {
                Ok(val) => Ok(Some(val)),
                Err(Error::NoEmit) => Ok(None),
                Err(e) => Err(e),
            }
        })
    }

    // ── Choose ──

    fn execute_choose<'a>(
        &'a self,
        choose_ir: &'a ChooseIR,
        scope: &'a mut Scope,
        _graph_name: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Option<Value>>> + 'a>> {
        Box::pin(async move {
            // Choose selects from alternatives. In optimization mode, the optimizer
            // picks the alternative. In execution mode, use the first one.
            let selected = choose_ir
                .alternatives
                .first()
                .ok_or_else(|| Error::Runtime("choose has no alternatives".into()))?;

            let input = scope.get("input").cloned().unwrap_or(Value::Null);

            // Try as graph first, then as node
            if self.ir.graphs.iter().any(|g| g.name == *selected) {
                let result = self.execute_graph(selected, input).await?;
                return Ok(Some(result));
            }

            if let Some(node) = self.ir.nodes.iter().find(|n| n.name == *selected) {
                let overrides = self.get_node_overrides(&node.name);
                let result =
                    node_runner::run_node(node, input, &overrides, &self.prompt_mgr).await?;
                return Ok(Some(result));
            }

            Err(Error::UnknownNode(selected.clone()))
        })
    }

    // ── Parallel ──

    fn execute_parallel<'a>(
        &'a self,
        par_ir: &'a ParallelIR,
        scope: &'a mut Scope,
        graph_name: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Option<Value>>> + 'a>> {
        Box::pin(async move {
            let collection = self.eval_expr(&par_ir.collection, scope)?;
            let items = match collection {
                Value::List(items) => items,
                other => {
                    return Err(Error::ExprError(format!(
                        "parallel collection must be a list, got {}",
                        other.type_name()
                    )))
                }
            };

            let mut results = Vec::with_capacity(items.len());

            // Execute body for each item (sequentially for now; async parallelism later)
            for item in items {
                let mut child_scope = scope.child();
                child_scope.bind(&par_ir.var, item);

                match self
                    .execute_stmts(&par_ir.body, &mut child_scope, graph_name)
                    .await
                {
                    Ok(val) => results.push(val),
                    Err(Error::NoEmit) => {}
                    Err(e) => return Err(e),
                }
            }

            let collected = Value::List(results);

            // Apply reduce if specified
            if let Some(ref reduce_name) = par_ir.reduce {
                if let Some(node) = self.ir.nodes.iter().find(|n| n.name == *reduce_name) {
                    let overrides = self.get_node_overrides(&node.name);
                    let reduced =
                        node_runner::run_node(node, collected, &overrides, &self.prompt_mgr)
                            .await?;
                    scope.bind("parallel_result", reduced.clone());
                    return Ok(Some(reduced));
                } else if self.ir.graphs.iter().any(|g| g.name == *reduce_name) {
                    let reduced = self.execute_graph(reduce_name, collected).await?;
                    scope.bind("parallel_result", reduced.clone());
                    return Ok(Some(reduced));
                } else {
                    return Err(Error::UnknownNode(reduce_name.clone()));
                }
            }

            scope.bind("parallel_result", collected.clone());
            Ok(Some(collected))
        })
    }

    // ── Emit ──

    fn evaluate_emit(&self, emit_ir: &EmitIR, scope: &Scope) -> Result<Value> {
        match emit_ir {
            EmitIR::Direct { value } => self.eval_expr(value, scope),
            EmitIR::Record { fields } => {
                let mut map = HashMap::new();
                for field in fields {
                    map.insert(field.name.clone(), self.eval_expr(&field.value, scope)?);
                }
                Ok(Value::Map(map))
            }
        }
    }

    // ── Expression evaluation ──

    /// Evaluate an IR expression against a scope.
    pub fn eval_expr(&self, expr: &ExprIR, scope: &Scope) -> Result<Value> {
        match expr {
            ExprIR::LitInt { value } => Ok(Value::Int(*value)),
            ExprIR::LitFloat { value } => Ok(Value::Float(*value)),
            ExprIR::LitString { value } => Ok(Value::String(value.clone())),
            ExprIR::LitBool { value } => Ok(Value::Bool(*value)),
            ExprIR::LitNull => Ok(Value::Null),

            ExprIR::Ident { name } => scope
                .get(name)
                .cloned()
                .ok_or_else(|| Error::ScopeError(format!("undefined variable: {}", name))),

            ExprIR::FieldAccess { base, field } => {
                let base_val = self.eval_expr(base, scope)?;
                base_val.field(field).cloned().ok_or_else(|| {
                    Error::ExprError(format!(
                        "field '{}' not found on {}",
                        field,
                        base_val.type_name()
                    ))
                })
            }

            ExprIR::Index { base, index } => {
                let base_val = self.eval_expr(base, scope)?;
                let idx_val = self.eval_expr(index, scope)?;
                match (&base_val, &idx_val) {
                    (Value::List(list), Value::Int(i)) => {
                        let idx = *i as usize;
                        list.get(idx)
                            .cloned()
                            .ok_or_else(|| Error::ExprError(format!("index {} out of bounds", i)))
                    }
                    (Value::Map(map), Value::String(key)) => map
                        .get(key)
                        .cloned()
                        .ok_or_else(|| Error::ExprError(format!("key '{}' not found", key))),
                    _ => Err(Error::ExprError(format!(
                        "cannot index {} with {}",
                        base_val.type_name(),
                        idx_val.type_name()
                    ))),
                }
            }

            ExprIR::UnaryNot { operand } => {
                let val = self.eval_expr(operand, scope)?;
                match val {
                    Value::Bool(b) => Ok(Value::Bool(!b)),
                    _ => Err(Error::ExprError(format!(
                        "cannot negate {}",
                        val.type_name()
                    ))),
                }
            }

            ExprIR::UnaryNeg { operand } => {
                let val = self.eval_expr(operand, scope)?;
                match val {
                    Value::Int(i) => Ok(Value::Int(-i)),
                    Value::Float(f) => Ok(Value::Float(-f)),
                    _ => Err(Error::ExprError(format!(
                        "cannot negate {}",
                        val.type_name()
                    ))),
                }
            }

            ExprIR::Binary { left, op, right } => {
                let lhs = self.eval_expr(left, scope)?;
                // Short-circuit for && and ||
                if op == "&&" {
                    return match lhs.as_bool() {
                        Some(false) => Ok(Value::Bool(false)),
                        Some(true) => {
                            let rhs = self.eval_expr(right, scope)?;
                            Ok(Value::Bool(rhs.as_bool().unwrap_or(false)))
                        }
                        None => Err(Error::ExprError("&& requires bool operands".into())),
                    };
                }
                if op == "||" {
                    return match lhs.as_bool() {
                        Some(true) => Ok(Value::Bool(true)),
                        Some(false) => {
                            let rhs = self.eval_expr(right, scope)?;
                            Ok(Value::Bool(rhs.as_bool().unwrap_or(false)))
                        }
                        None => Err(Error::ExprError("|| requires bool operands".into())),
                    };
                }

                let rhs = self.eval_expr(right, scope)?;
                eval_binary_op(&lhs, op, &rhs)
            }

            ExprIR::Call { name, args } => {
                let evaluated_args: Result<Vec<Value>> =
                    args.iter().map(|a| self.eval_expr(a, scope)).collect();
                let args = evaluated_args?;
                eval_builtin_call(name, &args)
            }

            ExprIR::List { elements } => {
                let items: Result<Vec<Value>> =
                    elements.iter().map(|e| self.eval_expr(e, scope)).collect();
                Ok(Value::List(items?))
            }

            ExprIR::Record { fields } => {
                let mut map = HashMap::new();
                for f in fields {
                    map.insert(f.key.clone(), self.eval_expr(&f.value, scope)?);
                }
                Ok(Value::Map(map))
            }
        }
    }

    fn eval_expr_as_bool(&self, expr: &ExprIR, scope: &Scope) -> Result<bool> {
        let val = self.eval_expr(expr, scope)?;
        match val {
            Value::Bool(b) => Ok(b),
            Value::Null => Ok(false),
            Value::Int(i) => Ok(i != 0),
            Value::String(ref s) => Ok(!s.is_empty()),
            _ => Ok(true), // truthy
        }
    }

    fn eval_expr_as_int(&self, expr: &ExprIR, scope: &Scope) -> Result<i64> {
        let val = self.eval_expr(expr, scope)?;
        match val {
            Value::Int(i) => Ok(i),
            Value::Float(f) => Ok(f as i64),
            _ => Err(Error::ExprError(format!(
                "expected integer, got {}",
                val.type_name()
            ))),
        }
    }

    /// Get overrides for a specific node.
    fn get_node_overrides(&self, node_name: &str) -> HashMap<String, serde_json::Value> {
        let prefix = format!("{}.", node_name);
        let mut result = HashMap::new();
        for (key, value) in &self.overrides {
            if let Some(field) = key.strip_prefix(&prefix) {
                result.insert(field.to_string(), value.clone());
            }
        }
        result
    }
}

/// Evaluate a binary operation.
fn eval_binary_op(lhs: &Value, op: &str, rhs: &Value) -> Result<Value> {
    match op {
        "==" => Ok(Value::Bool(values_equal(lhs, rhs))),
        "!=" => Ok(Value::Bool(!values_equal(lhs, rhs))),
        "<" => Ok(Value::Bool(compare_values(lhs, rhs)? < 0)),
        ">" => Ok(Value::Bool(compare_values(lhs, rhs)? > 0)),
        "<=" => Ok(Value::Bool(compare_values(lhs, rhs)? <= 0)),
        ">=" => Ok(Value::Bool(compare_values(lhs, rhs)? >= 0)),
        "+" => eval_add(lhs, rhs),
        "-" => eval_sub(lhs, rhs),
        "*" => eval_mul(lhs, rhs),
        "/" => eval_div(lhs, rhs),
        _ => Err(Error::ExprError(format!("unknown operator: {}", op))),
    }
}

fn values_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Null, Value::Null) => true,
        (Value::Bool(a), Value::Bool(b)) => a == b,
        (Value::Int(a), Value::Int(b)) => a == b,
        (Value::Float(a), Value::Float(b)) => (a - b).abs() < f64::EPSILON,
        (Value::Int(a), Value::Float(b)) | (Value::Float(b), Value::Int(a)) => {
            (*a as f64 - b).abs() < f64::EPSILON
        }
        (Value::String(a), Value::String(b)) => a == b,
        _ => false,
    }
}

fn compare_values(a: &Value, b: &Value) -> Result<i8> {
    match (a, b) {
        (Value::Int(a), Value::Int(b)) => Ok(a.cmp(b) as i8),
        (Value::Float(a), Value::Float(b)) => {
            Ok(a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal) as i8)
        }
        (Value::Int(a), Value::Float(b)) => {
            let a = *a as f64;
            Ok(a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal) as i8)
        }
        (Value::Float(a), Value::Int(b)) => {
            let b = *b as f64;
            Ok(a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal) as i8)
        }
        (Value::String(a), Value::String(b)) => Ok(a.cmp(b) as i8),
        _ => Err(Error::ExprError(format!(
            "cannot compare {} and {}",
            a.type_name(),
            b.type_name()
        ))),
    }
}

fn eval_add(a: &Value, b: &Value) -> Result<Value> {
    match (a, b) {
        (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a + b)),
        (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a + b)),
        (Value::Int(a), Value::Float(b)) => Ok(Value::Float(*a as f64 + b)),
        (Value::Float(a), Value::Int(b)) => Ok(Value::Float(a + *b as f64)),
        (Value::String(a), Value::String(b)) => Ok(Value::String(format!("{}{}", a, b))),
        (Value::List(a), Value::List(b)) => {
            let mut result = a.clone();
            result.extend(b.iter().cloned());
            Ok(Value::List(result))
        }
        _ => Err(Error::ExprError(format!(
            "cannot add {} and {}",
            a.type_name(),
            b.type_name()
        ))),
    }
}

fn eval_sub(a: &Value, b: &Value) -> Result<Value> {
    match (a, b) {
        (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a - b)),
        (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a - b)),
        (Value::Int(a), Value::Float(b)) => Ok(Value::Float(*a as f64 - b)),
        (Value::Float(a), Value::Int(b)) => Ok(Value::Float(a - *b as f64)),
        _ => Err(Error::ExprError(format!(
            "cannot subtract {} from {}",
            b.type_name(),
            a.type_name()
        ))),
    }
}

fn eval_mul(a: &Value, b: &Value) -> Result<Value> {
    match (a, b) {
        (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a * b)),
        (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a * b)),
        (Value::Int(a), Value::Float(b)) => Ok(Value::Float(*a as f64 * b)),
        (Value::Float(a), Value::Int(b)) => Ok(Value::Float(a * *b as f64)),
        _ => Err(Error::ExprError(format!(
            "cannot multiply {} and {}",
            a.type_name(),
            b.type_name()
        ))),
    }
}

fn eval_div(a: &Value, b: &Value) -> Result<Value> {
    match (a, b) {
        (Value::Int(_), Value::Int(0)) | (Value::Float(_), Value::Int(0)) => {
            Err(Error::ExprError("division by zero".into()))
        }
        (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a / b)),
        (Value::Float(a), Value::Float(b)) => {
            if *b == 0.0 {
                Err(Error::ExprError("division by zero".into()))
            } else {
                Ok(Value::Float(a / b))
            }
        }
        (Value::Int(a), Value::Float(b)) => {
            if *b == 0.0 {
                Err(Error::ExprError("division by zero".into()))
            } else {
                Ok(Value::Float(*a as f64 / b))
            }
        }
        (Value::Float(a), Value::Int(b)) => Ok(Value::Float(a / *b as f64)),
        _ => Err(Error::ExprError(format!(
            "cannot divide {} by {}",
            a.type_name(),
            b.type_name()
        ))),
    }
}

/// Evaluate built-in function calls (len, contains, str, int, float, etc.)
fn eval_builtin_call(name: &str, args: &[Value]) -> Result<Value> {
    match name {
        "len" => {
            if args.len() != 1 {
                return Err(Error::ExprError("len() takes exactly 1 argument".into()));
            }
            match &args[0] {
                Value::String(s) => Ok(Value::Int(s.len() as i64)),
                Value::List(l) => Ok(Value::Int(l.len() as i64)),
                Value::Map(m) => Ok(Value::Int(m.len() as i64)),
                _ => Err(Error::ExprError(format!(
                    "len() not supported for {}",
                    args[0].type_name()
                ))),
            }
        }
        "contains" => {
            if args.len() != 2 {
                return Err(Error::ExprError(
                    "contains() takes exactly 2 arguments".into(),
                ));
            }
            match (&args[0], &args[1]) {
                (Value::String(haystack), Value::String(needle)) => {
                    Ok(Value::Bool(haystack.contains(needle.as_str())))
                }
                (Value::List(list), val) => Ok(Value::Bool(list.contains(val))),
                _ => Ok(Value::Bool(false)),
            }
        }
        "str" => {
            if args.len() != 1 {
                return Err(Error::ExprError("str() takes exactly 1 argument".into()));
            }
            Ok(Value::String(args[0].to_string()))
        }
        "int" => {
            if args.len() != 1 {
                return Err(Error::ExprError("int() takes exactly 1 argument".into()));
            }
            match &args[0] {
                Value::Int(i) => Ok(Value::Int(*i)),
                Value::Float(f) => Ok(Value::Int(*f as i64)),
                Value::String(s) => s
                    .parse::<i64>()
                    .map(Value::Int)
                    .map_err(|_| Error::ExprError(format!("cannot convert '{}' to int", s))),
                Value::Bool(b) => Ok(Value::Int(if *b { 1 } else { 0 })),
                _ => Err(Error::ExprError(format!(
                    "cannot convert {} to int",
                    args[0].type_name()
                ))),
            }
        }
        "float" => {
            if args.len() != 1 {
                return Err(Error::ExprError("float() takes exactly 1 argument".into()));
            }
            match &args[0] {
                Value::Float(f) => Ok(Value::Float(*f)),
                Value::Int(i) => Ok(Value::Float(*i as f64)),
                Value::String(s) => s
                    .parse::<f64>()
                    .map(Value::Float)
                    .map_err(|_| Error::ExprError(format!("cannot convert '{}' to float", s))),
                _ => Err(Error::ExprError(format!(
                    "cannot convert {} to float",
                    args[0].type_name()
                ))),
            }
        }
        "lower" => {
            if args.len() != 1 {
                return Err(Error::ExprError("lower() takes exactly 1 argument".into()));
            }
            match &args[0] {
                Value::String(s) => Ok(Value::String(s.to_lowercase())),
                _ => Err(Error::ExprError("lower() requires a string".into())),
            }
        }
        "upper" => {
            if args.len() != 1 {
                return Err(Error::ExprError("upper() takes exactly 1 argument".into()));
            }
            match &args[0] {
                Value::String(s) => Ok(Value::String(s.to_uppercase())),
                _ => Err(Error::ExprError("upper() requires a string".into())),
            }
        }
        "trim" => {
            if args.len() != 1 {
                return Err(Error::ExprError("trim() takes exactly 1 argument".into()));
            }
            match &args[0] {
                Value::String(s) => Ok(Value::String(s.trim().to_string())),
                _ => Err(Error::ExprError("trim() requires a string".into())),
            }
        }
        "split" => {
            if args.len() != 2 {
                return Err(Error::ExprError("split() takes exactly 2 arguments".into()));
            }
            match (&args[0], &args[1]) {
                (Value::String(s), Value::String(sep)) => {
                    let parts: Vec<Value> = s
                        .split(sep.as_str())
                        .map(|p| Value::String(p.to_string()))
                        .collect();
                    Ok(Value::List(parts))
                }
                _ => Err(Error::ExprError("split() requires (string, string)".into())),
            }
        }
        "join" => {
            if args.len() != 2 {
                return Err(Error::ExprError("join() takes exactly 2 arguments".into()));
            }
            match (&args[0], &args[1]) {
                (Value::List(items), Value::String(sep)) => {
                    let strs: Vec<String> = items.iter().map(|v| v.to_string()).collect();
                    Ok(Value::String(strs.join(sep)))
                }
                _ => Err(Error::ExprError("join() requires (list, string)".into())),
            }
        }
        "keys" => {
            if args.len() != 1 {
                return Err(Error::ExprError("keys() takes exactly 1 argument".into()));
            }
            match &args[0] {
                Value::Map(m) => {
                    let keys: Vec<Value> = m.keys().map(|k| Value::String(k.clone())).collect();
                    Ok(Value::List(keys))
                }
                _ => Err(Error::ExprError("keys() requires a map".into())),
            }
        }
        "values" => {
            if args.len() != 1 {
                return Err(Error::ExprError("values() takes exactly 1 argument".into()));
            }
            match &args[0] {
                Value::Map(m) => {
                    let vals: Vec<Value> = m.values().cloned().collect();
                    Ok(Value::List(vals))
                }
                _ => Err(Error::ExprError("values() requires a map".into())),
            }
        }
        "json_parse" => {
            if args.len() != 1 {
                return Err(Error::ExprError(
                    "json_parse() takes exactly 1 argument".into(),
                ));
            }
            match &args[0] {
                Value::String(s) => match serde_json::from_str::<serde_json::Value>(s) {
                    Ok(json) => Ok(Value::from(json)),
                    Err(e) => Err(Error::ExprError(format!("json_parse: {}", e))),
                },
                _ => Err(Error::ExprError(
                    "json_parse() requires a string argument".into(),
                )),
            }
        }
        _ => Err(Error::ExprError(format!("unknown function: {}", name))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_simple_ir() -> ScaffoldIR {
        ScaffoldIR {
            version: "2.0.0".to_string(),
            types: vec![],
            nodes: vec![],
            graphs: vec![GraphIR {
                name: "test".to_string(),
                input: TypeIR::String,
                output: TypeIR::String,
                body: vec![GraphStmtIR::Emit(EmitIR::Direct {
                    value: ExprIR::Ident {
                        name: "input".to_string(),
                    },
                })],
            }],
            objectives: vec![],
        }
    }

    #[tokio::test]
    async fn test_passthrough_graph() {
        let ir = make_simple_ir();
        let executor = GraphExecutor::new(ir);
        let result = executor
            .execute_graph("test", Value::String("hello".into()))
            .await
            .unwrap();
        assert_eq!(result, Value::String("hello".into()));
    }

    #[test]
    fn test_eval_expr_literals() {
        let ir = make_simple_ir();
        let executor = GraphExecutor::new(ir);
        let scope = Scope::root(Value::Null);

        assert_eq!(
            executor
                .eval_expr(&ExprIR::LitInt { value: 42 }, &scope)
                .unwrap(),
            Value::Int(42)
        );
        assert_eq!(
            executor
                .eval_expr(&ExprIR::LitString { value: "hi".into() }, &scope)
                .unwrap(),
            Value::String("hi".into())
        );
        assert_eq!(
            executor
                .eval_expr(&ExprIR::LitBool { value: true }, &scope)
                .unwrap(),
            Value::Bool(true)
        );
    }

    #[test]
    fn test_eval_binary_ops() {
        let ir = make_simple_ir();
        let executor = GraphExecutor::new(ir);
        let scope = Scope::root(Value::Null);

        let add = ExprIR::Binary {
            left: Box::new(ExprIR::LitInt { value: 2 }),
            op: "+".to_string(),
            right: Box::new(ExprIR::LitInt { value: 3 }),
        };
        assert_eq!(executor.eval_expr(&add, &scope).unwrap(), Value::Int(5));

        let eq = ExprIR::Binary {
            left: Box::new(ExprIR::LitString { value: "a".into() }),
            op: "==".to_string(),
            right: Box::new(ExprIR::LitString { value: "a".into() }),
        };
        assert_eq!(executor.eval_expr(&eq, &scope).unwrap(), Value::Bool(true));
    }

    #[test]
    fn test_eval_builtin_calls() {
        let ir = make_simple_ir();
        let executor = GraphExecutor::new(ir);
        let scope = Scope::root(Value::Null);

        let len_call = ExprIR::Call {
            name: "len".to_string(),
            args: vec![ExprIR::LitString {
                value: "hello".into(),
            }],
        };
        assert_eq!(
            executor.eval_expr(&len_call, &scope).unwrap(),
            Value::Int(5)
        );
    }

    #[tokio::test]
    async fn test_unknown_graph() {
        let ir = make_simple_ir();
        let executor = GraphExecutor::new(ir);
        let result = executor.execute_graph("nonexistent", Value::Null).await;
        assert!(result.is_err());
    }
}
