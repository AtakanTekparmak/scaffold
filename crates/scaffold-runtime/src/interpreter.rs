//! Interpreter-style execution for task/harness IR.

use crate::error::{Error, Result};
use crate::llm::{query_structured_with_config, LlmConfig};
use crate::prompt::PromptManager;
use crate::value::{ResultValue, Value};
use scaffold_ir::{
    types_to_json_schema_document, AgentIR, ExprIR, LiteralIR, PromptIR, ScaffoldIR, StageIR,
    StageKindIR, StringOrFileIR, TaskIR, TaskNodeIR, TypeIR,
};
use std::collections::{BTreeSet, HashMap};
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;

/// Execute a task directly from IR, optionally applying a concrete harness.
pub fn execute_task(
    ir: &ScaffoldIR,
    task_name: &str,
    harness_name: Option<&str>,
    input: Value,
    base_dir: &Path,
) -> Result<Value> {
    let interpreter = TaskInterpreter::new(ir, base_dir);
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| Error::Runtime(format!("failed to create task runtime: {}", e)))?;
    runtime.block_on(interpreter.execute(task_name, harness_name, input))
}

struct TaskInterpreter<'a> {
    ir: &'a ScaffoldIR,
    base_dir: PathBuf,
}

#[derive(Default)]
struct ExecutionContext {
    input: Value,
    artifacts: HashMap<String, Value>,
}

#[derive(Debug, Clone, Default)]
struct ResolvedHarness {
    defaults: HashMap<String, Value>,
    bindings: HashMap<String, HashMap<String, Value>>,
}

impl ResolvedHarness {
    fn field_value(&self, target: &str, field: &str) -> Option<&Value> {
        self.bindings
            .get(target)
            .and_then(|fields| fields.get(field))
            .or_else(|| self.defaults.get(field))
    }

    fn field_names_for_target(&self, target: &str) -> BTreeSet<String> {
        let mut names = BTreeSet::new();
        names.extend(self.defaults.keys().cloned());
        if let Some(fields) = self.bindings.get(target) {
            names.extend(fields.keys().cloned());
        }
        names
    }
}

impl<'a> TaskInterpreter<'a> {
    fn new(ir: &'a ScaffoldIR, base_dir: &Path) -> Self {
        Self {
            ir,
            base_dir: base_dir.to_path_buf(),
        }
    }

    async fn execute(
        &self,
        task_name: &str,
        harness_name: Option<&str>,
        input: Value,
    ) -> Result<Value> {
        let task = self.find_task(task_name)?;
        let harness = self.resolve_harness(task_name, harness_name)?;
        let input = self.validate_value(input, &task.input)?;
        let mut ctx = ExecutionContext {
            input,
            artifacts: HashMap::new(),
        };
        self.execute_nodes(&task.body, &mut ctx, &harness).await?;
        let output = self.build_output(task, &ctx)?;
        self.validate_value(output, &task.output)
    }

    fn execute_nodes<'b>(
        &'b self,
        nodes: &'b [TaskNodeIR],
        ctx: &'b mut ExecutionContext,
        harness: &'b ResolvedHarness,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + 'b>> {
        Box::pin(async move {
            for node in nodes {
                match node {
                    TaskNodeIR::Stage(stage) => self.execute_stage(stage, ctx, harness).await?,
                    TaskNodeIR::Loop(loop_decl) => {
                        let max_iters = self.eval_max_iters(&loop_decl.max_iters, ctx)?;
                        for _ in 0..max_iters {
                            self.execute_nodes(&loop_decl.body, ctx, harness).await?;
                            if self.eval_bool(&loop_decl.until, ctx)? {
                                break;
                            }
                        }
                    }
                    TaskNodeIR::Branch(branch) => {
                        if self.eval_bool(&branch.condition, ctx)? {
                            self.execute_nodes(&branch.then_body, ctx, harness).await?;
                        } else {
                            self.execute_nodes(&branch.else_body, ctx, harness).await?;
                        }
                    }
                }
            }
            Ok(())
        })
    }

    async fn execute_stage(
        &self,
        stage: &StageIR,
        ctx: &mut ExecutionContext,
        harness: &ResolvedHarness,
    ) -> Result<()> {
        if let Some(when) = &stage.when {
            if !self.eval_bool(when, ctx)? {
                return Ok(());
            }
        }

        let stage_input = self.eval_expr(&stage.input, ctx)?;
        let output = match stage.stage_kind {
            StageKindIR::Prompt => {
                self.execute_prompt_stage(stage, stage_input, harness)
                    .await?
            }
            StageKindIR::Agent => {
                self.execute_agent_stage(stage, stage_input, harness)
                    .await?
            }
            StageKindIR::Tool => {
                return Err(Error::Runtime(format!(
                    "task interpreter does not yet support tool stage '{}' (component '{}'); use codegen for tool-backed execution for now",
                    stage.name, stage.component
                )));
            }
        };
        ctx.artifacts.insert(stage.output.clone(), output);
        Ok(())
    }

    async fn execute_prompt_stage(
        &self,
        stage: &StageIR,
        input: Value,
        harness: &ResolvedHarness,
    ) -> Result<Value> {
        self.ensure_supported_fields(
            &stage.name,
            harness,
            &["model", "temperature", "system_prompt"],
        )?;
        let prompt = self.find_prompt(&stage.component)?;
        let input = self.validate_value(input, &prompt.input)?;
        let rendered = self.render_prompt(prompt, &input)?;
        let schema = self.output_schema_string(&prompt.output)?;
        let mut config = LlmConfig::new();
        if let Some(model) = self.harness_string(harness, &stage.name, "model")? {
            config = config.with_model(model);
        }
        if let Some(temperature) = self.harness_float(harness, &stage.name, "temperature")? {
            config = config.with_temperature(temperature as f32);
        }
        if let Some(system_prompt) =
            self.effective_system_prompt(harness, &stage.name, prompt.system.as_ref())?
        {
            config = config.with_system_prompt(system_prompt);
        }
        let output = query_structured_with_config(&rendered, &schema, &config).await?;
        self.validate_value(output, &prompt.output)
    }

    async fn execute_agent_stage(
        &self,
        stage: &StageIR,
        input: Value,
        harness: &ResolvedHarness,
    ) -> Result<Value> {
        self.ensure_supported_fields(
            &stage.name,
            harness,
            &["model", "temperature", "system_prompt"],
        )?;
        let agent = self.find_agent(&stage.component)?;
        if !agent.tools.is_empty() {
            return Err(Error::Runtime(format!(
                "task interpreter does not yet support tool-enabled agent stage '{}' (component '{}'); use codegen for full agentic tool execution for now",
                stage.name, stage.component
            )));
        }

        let input = self.validate_value(input, &agent.input)?;
        let input_json = serde_json::to_string_pretty(&serde_json::Value::from(input.clone()))
            .map_err(|e| Error::SerializationError(e.to_string()))?;
        let schema = self.output_schema_string(&agent.output)?;
        let mut config = LlmConfig::new();
        if let Some(model) = self
            .harness_string(harness, &stage.name, "model")?
            .or_else(|| agent.model.clone())
        {
            config = config.with_model(model);
        }
        if let Some(temperature) = self.harness_float(harness, &stage.name, "temperature")? {
            config = config.with_temperature(temperature as f32);
        }
        if let Some(system_prompt) =
            self.effective_system_prompt(harness, &stage.name, Some(&agent.system))?
        {
            config = config.with_system_prompt(system_prompt);
        }
        let user_prompt = format!("Input:\n{}", input_json);
        let output = query_structured_with_config(&user_prompt, &schema, &config).await?;
        self.validate_value(output, &agent.output)
    }

    fn render_prompt(&self, prompt: &PromptIR, input: &Value) -> Result<String> {
        let template = self.read_string_or_file(&prompt.template)?;
        let manager = PromptManager::new();
        manager.interpolate(&template, &self.prompt_context(input))
    }

    fn prompt_context(&self, input: &Value) -> Value {
        match input {
            Value::Map(fields) => {
                let mut ctx = fields.clone();
                ctx.entry("input".to_string())
                    .or_insert_with(|| input.clone());
                Value::Map(ctx)
            }
            Value::Struct { fields, .. } => {
                let mut ctx = fields.clone();
                ctx.entry("input".to_string())
                    .or_insert_with(|| input.clone());
                Value::Map(ctx)
            }
            other => {
                let mut ctx = HashMap::new();
                ctx.insert("input".to_string(), other.clone());
                Value::Map(ctx)
            }
        }
    }

    fn output_schema_string(&self, output: &TypeIR) -> Result<String> {
        let definitions = self
            .ir
            .types
            .iter()
            .map(|def| (def.name.clone(), def.definition.clone()))
            .collect::<Vec<_>>();
        serde_json::to_string_pretty(&types_to_json_schema_document(output, &definitions))
            .map_err(|e| Error::SerializationError(e.to_string()))
    }

    fn effective_system_prompt(
        &self,
        harness: &ResolvedHarness,
        target: &str,
        fallback: Option<&StringOrFileIR>,
    ) -> Result<Option<String>> {
        if let Some(system_prompt) = self.harness_string(harness, target, "system_prompt")? {
            return Ok(Some(system_prompt));
        }
        fallback
            .map(|value| self.read_string_or_file(value))
            .transpose()
    }

    fn ensure_supported_fields(
        &self,
        target: &str,
        harness: &ResolvedHarness,
        supported: &[&str],
    ) -> Result<()> {
        let supported = supported.iter().copied().collect::<BTreeSet<_>>();
        let unsupported = harness
            .field_names_for_target(target)
            .into_iter()
            .filter(|field| !supported.contains(field.as_str()))
            .collect::<Vec<_>>();
        if unsupported.is_empty() {
            Ok(())
        } else {
            Err(Error::Runtime(format!(
                "task interpreter does not yet support harness fields [{}] for target '{}'",
                unsupported.join(", "),
                target
            )))
        }
    }

    fn harness_string(
        &self,
        harness: &ResolvedHarness,
        target: &str,
        field: &str,
    ) -> Result<Option<String>> {
        match harness.field_value(target, field) {
            Some(Value::String(value)) => Ok(Some(value.clone())),
            Some(value) => Err(Error::TypeError {
                expected: "string".to_string(),
                actual: value.type_name().to_string(),
            }),
            None => Ok(None),
        }
    }

    fn harness_float(
        &self,
        harness: &ResolvedHarness,
        target: &str,
        field: &str,
    ) -> Result<Option<f64>> {
        match harness.field_value(target, field) {
            Some(Value::Float(value)) => Ok(Some(*value)),
            Some(Value::Int(value)) => Ok(Some(*value as f64)),
            Some(value) => Err(Error::TypeError {
                expected: "float".to_string(),
                actual: value.type_name().to_string(),
            }),
            None => Ok(None),
        }
    }

    fn build_output(&self, task: &TaskIR, ctx: &ExecutionContext) -> Result<Value> {
        let mut fields = HashMap::new();
        for field in &task.emit {
            fields.insert(field.name.clone(), self.eval_expr(&field.value, ctx)?);
        }
        Ok(self.wrap_struct_like(&task.output, fields))
    }

    fn wrap_struct_like(&self, ty: &TypeIR, fields: HashMap<String, Value>) -> Value {
        match ty {
            TypeIR::Named { name } => Value::Struct {
                type_name: name.clone(),
                fields,
            },
            _ => Value::Map(fields),
        }
    }

    fn resolve_harness(
        &self,
        task_name: &str,
        harness_name: Option<&str>,
    ) -> Result<ResolvedHarness> {
        let harness = if let Some(harness_name) = harness_name {
            let harness = self
                .ir
                .harnesses
                .iter()
                .find(|candidate| candidate.name == harness_name)
                .ok_or_else(|| Error::Runtime(format!("harness '{}' not found", harness_name)))?;
            if harness.task != task_name {
                return Err(Error::Runtime(format!(
                    "harness '{}' targets task '{}', not '{}'",
                    harness.name, harness.task, task_name
                )));
            }
            Some(harness)
        } else {
            let candidates = self
                .ir
                .harnesses
                .iter()
                .filter(|candidate| candidate.task == task_name)
                .collect::<Vec<_>>();
            match candidates.as_slice() {
                [] => None,
                [only] => Some(*only),
                _ => {
                    return Err(Error::Runtime(format!(
                        "multiple harnesses target task '{}'; specify --harness",
                        task_name
                    )));
                }
            }
        };

        let mut resolved = ResolvedHarness::default();
        if let Some(harness) = harness {
            for binding in &harness.defaults {
                resolved.defaults.insert(
                    binding.key.segments.join("."),
                    self.eval_static_expr(&binding.value)?,
                );
            }
            for target in &harness.bindings {
                let mut fields = HashMap::new();
                for binding in &target.bindings {
                    fields.insert(
                        binding.key.segments.join("."),
                        self.eval_static_expr(&binding.value)?,
                    );
                }
                resolved.bindings.insert(target.target.clone(), fields);
            }
        }
        Ok(resolved)
    }

    fn eval_static_expr(&self, expr: &ExprIR) -> Result<Value> {
        let ctx = ExecutionContext::default();
        self.eval_expr(expr, &ctx)
    }

    fn eval_max_iters(&self, expr: &ExprIR, ctx: &ExecutionContext) -> Result<usize> {
        match self.eval_expr(expr, ctx)? {
            Value::Int(value) if value >= 0 => Ok(value as usize),
            Value::Float(value) if value.is_finite() && value >= 0.0 && value.fract() == 0.0 => {
                Ok(value as usize)
            }
            other => Err(Error::TypeError {
                expected: "non-negative integer".to_string(),
                actual: other.type_name().to_string(),
            }),
        }
    }

    fn eval_bool(&self, expr: &ExprIR, ctx: &ExecutionContext) -> Result<bool> {
        match self.eval_expr(expr, ctx)? {
            Value::Bool(value) => Ok(value),
            other => Err(Error::TypeError {
                expected: "bool".to_string(),
                actual: other.type_name().to_string(),
            }),
        }
    }

    fn eval_expr(&self, expr: &ExprIR, ctx: &ExecutionContext) -> Result<Value> {
        match expr {
            ExprIR::Literal { value } => Ok(match value {
                LiteralIR::Int { value } => Value::Int(*value),
                LiteralIR::Float { value } => Value::Float(*value),
                LiteralIR::String { value } => Value::String(value.clone()),
                LiteralIR::Bool { value } => Value::Bool(*value),
                LiteralIR::Null => Value::Null,
            }),
            ExprIR::Ident { name } => {
                if name == "input" {
                    return Ok(ctx.input.clone());
                }
                ctx.artifacts
                    .get(name)
                    .cloned()
                    .ok_or_else(|| Error::Runtime(format!("unknown identifier '{}'", name)))
            }
            ExprIR::FieldAccess { base, field } => {
                let base = self.eval_expr(base, ctx)?;
                base.field(field).cloned().ok_or_else(|| {
                    Error::Runtime(format!(
                        "field '{}' not found on {}",
                        field,
                        self.type_label_from_value(&base)
                    ))
                })
            }
            ExprIR::Binary { left, op, right } => {
                let left = self.eval_expr(left, ctx)?;
                let right = self.eval_expr(right, ctx)?;
                self.eval_binary(&left, op, &right)
            }
            ExprIR::Call { function, args } => {
                let args = args
                    .iter()
                    .map(|arg| self.eval_expr(arg, ctx))
                    .collect::<Result<Vec<_>>>()?;
                self.eval_call(function, &args)
            }
            ExprIR::ForeignCall {
                module, function, ..
            } => Err(Error::Runtime(format!(
                "task interpreter does not support foreign call '{}::{}'",
                module, function
            ))),
            ExprIR::List { elements } => Ok(Value::List(
                elements
                    .iter()
                    .map(|element| self.eval_expr(element, ctx))
                    .collect::<Result<Vec<_>>>()?,
            )),
            ExprIR::Record { fields } => {
                let mut out = HashMap::new();
                for field in fields {
                    out.insert(field.key.clone(), self.eval_expr(&field.value, ctx)?);
                }
                Ok(Value::Map(out))
            }
        }
    }

    fn eval_binary(&self, left: &Value, op: &str, right: &Value) -> Result<Value> {
        match op {
            "==" => Ok(Value::Bool(left == right)),
            "!=" => Ok(Value::Bool(left != right)),
            "&&" => Ok(Value::Bool(
                self.expect_bool(left)? && self.expect_bool(right)?,
            )),
            "||" => Ok(Value::Bool(
                self.expect_bool(left)? || self.expect_bool(right)?,
            )),
            "+" => self.numeric_binary(left, right, |a, b| a + b, |a, b| a + b),
            "-" => self.numeric_binary(left, right, |a, b| a - b, |a, b| a - b),
            "*" => self.numeric_binary(left, right, |a, b| a * b, |a, b| a * b),
            "/" => {
                let denominator = self.expect_number(right)?;
                if denominator == 0.0 {
                    return Err(Error::Runtime("division by zero".to_string()));
                }
                Ok(Value::Float(self.expect_number(left)? / denominator))
            }
            "<" | "<=" | ">" | ">=" => Ok(Value::Bool(self.compare_values(left, op, right)?)),
            other => Err(Error::Runtime(format!("unsupported operator '{}'", other))),
        }
    }

    fn eval_call(&self, function: &str, args: &[Value]) -> Result<Value> {
        match function {
            "len" => {
                let value = self.expect_arity(function, args, 1)?;
                Ok(Value::Int(self.length_of(value)? as i64))
            }
            "is_empty" => {
                let value = self.expect_arity(function, args, 1)?;
                Ok(Value::Bool(self.length_of(value)? == 0))
            }
            "contains" => {
                let haystack = self.expect_arity(function, args, 2)?;
                let needle = &args[1];
                Ok(Value::Bool(match haystack {
                    Value::String(text) => needle
                        .as_str()
                        .map(|substr| text.contains(substr))
                        .unwrap_or(false),
                    Value::List(items) => items.contains(needle),
                    Value::Map(map) => needle
                        .as_str()
                        .map(|key| map.contains_key(key))
                        .unwrap_or(false),
                    Value::Struct { fields, .. } => needle
                        .as_str()
                        .map(|key| fields.contains_key(key))
                        .unwrap_or(false),
                    _ => false,
                }))
            }
            "is_some" => {
                let value = self.expect_arity(function, args, 1)?;
                Ok(Value::Bool(!value.is_null()))
            }
            "is_none" => {
                let value = self.expect_arity(function, args, 1)?;
                Ok(Value::Bool(value.is_null()))
            }
            "not" => {
                let value = self.expect_arity(function, args, 1)?;
                Ok(Value::Bool(!self.expect_bool(value)?))
            }
            "unwrap" => {
                let value = self.expect_arity(function, args, 1)?;
                self.unwrap_value(value)
            }
            "unwrap_or" => {
                self.expect_arity(function, args, 2)?;
                if args[0].is_null() {
                    Ok(args[1].clone())
                } else {
                    match &args[0] {
                        Value::Result(result) => match &**result {
                            ResultValue::Ok(value) => Ok(value.clone()),
                            ResultValue::Err(_) => Ok(args[1].clone()),
                        },
                        other => Ok(other.clone()),
                    }
                }
            }
            "abs" => {
                let value = self.expect_arity(function, args, 1)?;
                match value {
                    Value::Int(value) => Ok(Value::Int(value.abs())),
                    Value::Float(value) => Ok(Value::Float(value.abs())),
                    other => Err(Error::TypeError {
                        expected: "numeric".to_string(),
                        actual: other.type_name().to_string(),
                    }),
                }
            }
            "min" => self.numeric_extreme(function, args, f64::min),
            "max" => self.numeric_extreme(function, args, f64::max),
            "variant" => {
                self.expect_arity(function, args, 2)?;
                let group = self.expect_string(&args[0])?;
                let name = self.expect_string(&args[1])?;
                Ok(Value::String(format!("{}::{}", group, name)))
            }
            other => Err(Error::Runtime(format!(
                "task interpreter does not support function '{}'",
                other
            ))),
        }
    }

    fn numeric_extreme(
        &self,
        function: &str,
        args: &[Value],
        combine: fn(f64, f64) -> f64,
    ) -> Result<Value> {
        if args.is_empty() {
            return Err(Error::Runtime(format!(
                "function '{}' expects at least one argument",
                function
            )));
        }
        let mut out = self.expect_number(&args[0])?;
        let mut any_float = matches!(args[0], Value::Float(_));
        for value in &args[1..] {
            any_float |= matches!(value, Value::Float(_));
            out = combine(out, self.expect_number(value)?);
        }
        if any_float {
            Ok(Value::Float(out))
        } else {
            Ok(Value::Int(out as i64))
        }
    }

    fn numeric_binary(
        &self,
        left: &Value,
        right: &Value,
        int_op: fn(i64, i64) -> i64,
        float_op: fn(f64, f64) -> f64,
    ) -> Result<Value> {
        match (left, right) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Int(int_op(*a, *b))),
            _ => Ok(Value::Float(float_op(
                self.expect_number(left)?,
                self.expect_number(right)?,
            ))),
        }
    }

    fn compare_values(&self, left: &Value, op: &str, right: &Value) -> Result<bool> {
        if let (Ok(left), Ok(right)) = (self.try_number(left), self.try_number(right)) {
            return Ok(match op {
                "<" => left < right,
                "<=" => left <= right,
                ">" => left > right,
                ">=" => left >= right,
                _ => false,
            });
        }
        match (left, right) {
            (Value::String(a), Value::String(b)) => Ok(match op {
                "<" => a < b,
                "<=" => a <= b,
                ">" => a > b,
                ">=" => a >= b,
                _ => false,
            }),
            _ => Err(Error::Runtime(format!(
                "cannot compare {} and {} with '{}'",
                left.type_name(),
                right.type_name(),
                op
            ))),
        }
    }

    fn length_of(&self, value: &Value) -> Result<usize> {
        match value {
            Value::Null => Ok(0),
            Value::String(text) => Ok(text.len()),
            Value::Bytes(bytes) => Ok(bytes.len()),
            Value::List(items) => Ok(items.len()),
            Value::Map(map) => Ok(map.len()),
            Value::Struct { fields, .. } => Ok(fields.len()),
            other => Err(Error::TypeError {
                expected: "string, bytes, list, or map".to_string(),
                actual: other.type_name().to_string(),
            }),
        }
    }

    fn unwrap_value(&self, value: &Value) -> Result<Value> {
        match value {
            Value::Null => Err(Error::Runtime("called unwrap() on null".to_string())),
            Value::Result(result) => match &**result {
                ResultValue::Ok(value) => Ok(value.clone()),
                ResultValue::Err(error) => Err(Error::Runtime(format!(
                    "called unwrap() on error value: {}",
                    error
                ))),
            },
            other => Ok(other.clone()),
        }
    }

    fn expect_arity<'b>(
        &self,
        function: &str,
        args: &'b [Value],
        expected: usize,
    ) -> Result<&'b Value> {
        if args.len() != expected {
            return Err(Error::Runtime(format!(
                "function '{}' expects {} argument(s), got {}",
                function,
                expected,
                args.len()
            )));
        }
        Ok(&args[0])
    }

    fn expect_bool(&self, value: &Value) -> Result<bool> {
        value.as_bool().ok_or_else(|| Error::TypeError {
            expected: "bool".to_string(),
            actual: value.type_name().to_string(),
        })
    }

    fn expect_number(&self, value: &Value) -> Result<f64> {
        self.try_number(value).map_err(|_| Error::TypeError {
            expected: "numeric".to_string(),
            actual: value.type_name().to_string(),
        })
    }

    fn try_number(&self, value: &Value) -> std::result::Result<f64, ()> {
        match value {
            Value::Int(value) => Ok(*value as f64),
            Value::Float(value) => Ok(*value),
            _ => Err(()),
        }
    }

    fn expect_string<'b>(&self, value: &'b Value) -> Result<&'b str> {
        value.as_str().ok_or_else(|| Error::TypeError {
            expected: "string".to_string(),
            actual: value.type_name().to_string(),
        })
    }

    fn read_string_or_file(&self, value: &StringOrFileIR) -> Result<String> {
        match value {
            StringOrFileIR::Literal { value } => Ok(value.clone()),
            StringOrFileIR::File { path } => {
                let path = Path::new(path);
                let resolved = if path.is_absolute() {
                    path.to_path_buf()
                } else {
                    self.base_dir.join(path)
                };
                std::fs::read_to_string(&resolved).map_err(|e| {
                    Error::Runtime(format!(
                        "failed to read template/system file {}: {}",
                        resolved.display(),
                        e
                    ))
                })
            }
        }
    }

    fn validate_value(&self, value: Value, ty: &TypeIR) -> Result<Value> {
        match ty {
            TypeIR::Any => Ok(value),
            TypeIR::Bool => match value {
                Value::Bool(_) => Ok(value),
                other => self.type_mismatch("bool", &other),
            },
            TypeIR::Int => match value {
                Value::Int(_) => Ok(value),
                other => self.type_mismatch("int", &other),
            },
            TypeIR::Float => match value {
                Value::Float(_) => Ok(value),
                Value::Int(value) => Ok(Value::Float(value as f64)),
                other => self.type_mismatch("float", &other),
            },
            TypeIR::String => match value {
                Value::String(_) => Ok(value),
                other => self.type_mismatch("string", &other),
            },
            TypeIR::Bytes => match value {
                Value::Bytes(_) => Ok(value),
                other => self.type_mismatch("bytes", &other),
            },
            TypeIR::Option { inner } => {
                if value.is_null() {
                    Ok(Value::Null)
                } else {
                    self.validate_value(value, inner)
                }
            }
            TypeIR::List { element } => match value {
                Value::List(items) => Ok(Value::List(
                    items
                        .into_iter()
                        .map(|item| self.validate_value(item, element))
                        .collect::<Result<Vec<_>>>()?,
                )),
                other => self.type_mismatch("list", &other),
            },
            TypeIR::Map { value: item_ty, .. } => match value {
                Value::Map(map) => {
                    let mut out = HashMap::new();
                    for (key, value) in map {
                        out.insert(key, self.validate_value(value, item_ty)?);
                    }
                    Ok(Value::Map(out))
                }
                other => self.type_mismatch("map", &other),
            },
            TypeIR::Struct { fields } => {
                let mut actual = match value {
                    Value::Map(map) => map,
                    Value::Struct { fields, .. } => fields,
                    other => return self.type_mismatch("struct", &other),
                };

                let mut out = HashMap::new();
                for (name, field_ty) in fields {
                    let field_value = actual.remove(name).ok_or_else(|| {
                        Error::Runtime(format!("missing field '{}' in struct value", name))
                    })?;
                    out.insert(name.clone(), self.validate_value(field_value, field_ty)?);
                }
                if let Some(extra) = actual.keys().next() {
                    return Err(Error::Runtime(format!(
                        "unexpected field '{}' in struct value",
                        extra
                    )));
                }
                Ok(Value::Map(out))
            }
            TypeIR::Named { name } => {
                let definition = self.resolve_named_type(name)?;
                let validated = self.validate_value(value, definition)?;
                match (definition, validated) {
                    (TypeIR::Struct { .. }, Value::Map(fields)) => Ok(Value::Struct {
                        type_name: name.clone(),
                        fields,
                    }),
                    (_, other) => Ok(other),
                }
            }
            TypeIR::Result { ok, err } => match value {
                Value::Result(result) => match *result {
                    ResultValue::Ok(value) => Ok(Value::Result(Box::new(ResultValue::Ok(
                        self.validate_value(value, ok)?,
                    )))),
                    ResultValue::Err(value) => Ok(Value::Result(Box::new(ResultValue::Err(
                        self.validate_value(value, err)?,
                    )))),
                },
                other => self.type_mismatch("result", &other),
            },
        }
    }

    fn type_mismatch<T>(&self, expected: &str, actual: &Value) -> Result<T> {
        Err(Error::TypeError {
            expected: expected.to_string(),
            actual: actual.type_name().to_string(),
        })
    }

    fn resolve_named_type(&self, name: &str) -> Result<&TypeIR> {
        self.ir
            .types
            .iter()
            .find(|def| def.name == name)
            .map(|def| &def.definition)
            .ok_or_else(|| Error::Runtime(format!("unknown named type '{}'", name)))
    }

    fn find_task(&self, name: &str) -> Result<&TaskIR> {
        self.ir
            .tasks
            .iter()
            .find(|task| task.name == name)
            .ok_or_else(|| Error::Runtime(format!("task '{}' not found", name)))
    }

    fn find_prompt(&self, name: &str) -> Result<&PromptIR> {
        self.ir
            .prompts
            .iter()
            .find(|prompt| prompt.name == name)
            .ok_or_else(|| Error::Runtime(format!("prompt '{}' not found", name)))
    }

    fn find_agent(&self, name: &str) -> Result<&AgentIR> {
        self.ir
            .agents
            .iter()
            .find(|agent| agent.name == name)
            .ok_or_else(|| Error::Runtime(format!("agent '{}' not found", name)))
    }

    fn type_label_from_value(&self, value: &Value) -> String {
        match value {
            Value::Struct { type_name, .. } => format!("struct {}", type_name),
            other => other.type_name().to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scaffold_ir::{
        ArtifactSlotIR, BindingIR, BindingPathIR, EmitFieldIR, HarnessIR, MetricIR, StageIR,
        TypeDefIR, TypeDefKindIR,
    };

    fn string_type() -> TypeIR {
        TypeIR::String
    }

    fn answer_type() -> TypeIR {
        TypeIR::Struct {
            fields: HashMap::from([("text".to_string(), TypeIR::String)]),
        }
    }

    #[test]
    fn eval_expr_supports_field_access_and_builtins() {
        let empty_ir = ScaffoldIR {
            version: scaffold_ir::IR_VERSION.to_string(),
            extern_crates: Vec::new(),
            foreign_modules: Vec::new(),
            types: Vec::new(),
            tools: Vec::new(),
            prompts: Vec::new(),
            agents: Vec::new(),
            pipelines: Vec::new(),
            tasks: Vec::new(),
            harnesses: Vec::new(),
            objectives: Vec::new(),
        };
        let interpreter = TaskInterpreter::new(&empty_ir, Path::new("."));
        let ctx = ExecutionContext {
            input: Value::Map(HashMap::from([(
                "query".to_string(),
                Value::String("hello".into()),
            )])),
            artifacts: HashMap::from([(
                "notes".to_string(),
                Value::Map(HashMap::from([(
                    "items".to_string(),
                    Value::List(vec![Value::String("a".into()), Value::String("b".into())]),
                )])),
            )]),
        };
        let expr = ExprIR::Binary {
            left: Box::new(ExprIR::Call {
                function: "len".to_string(),
                args: vec![ExprIR::FieldAccess {
                    base: Box::new(ExprIR::Ident {
                        name: "notes".to_string(),
                    }),
                    field: "items".to_string(),
                }],
            }),
            op: "==".to_string(),
            right: Box::new(ExprIR::Literal {
                value: LiteralIR::Int { value: 2 },
            }),
        };
        let result = interpreter.eval_expr(&expr, &ctx).unwrap();
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn execute_task_runs_prompt_stage_with_harness_overrides() {
        std::env::set_var("SCAFFOLD_LLM_MOCK_JSON", r#"{"text":"hello world"}"#);

        let ir = ScaffoldIR {
            version: scaffold_ir::IR_VERSION.to_string(),
            extern_crates: Vec::new(),
            foreign_modules: Vec::new(),
            types: vec![
                TypeDefIR {
                    name: "Draft".to_string(),
                    kind: TypeDefKindIR::Artifact,
                    definition: answer_type(),
                },
                TypeDefIR {
                    name: "TaskOutput".to_string(),
                    kind: TypeDefKindIR::Type,
                    definition: TypeIR::Struct {
                        fields: HashMap::from([("answer".to_string(), TypeIR::String)]),
                    },
                },
            ],
            tools: Vec::new(),
            prompts: vec![PromptIR {
                name: "writer".to_string(),
                input: TypeIR::Struct {
                    fields: HashMap::from([("query".to_string(), string_type())]),
                },
                output: TypeIR::Named {
                    name: "Draft".to_string(),
                },
                template: StringOrFileIR::Literal {
                    value: "Answer this: {query}".to_string(),
                },
                system: Some(StringOrFileIR::Literal {
                    value: "You are careful.".to_string(),
                }),
            }],
            agents: Vec::new(),
            pipelines: Vec::new(),
            tasks: vec![TaskIR {
                name: "answer_question".to_string(),
                input: TypeIR::Struct {
                    fields: HashMap::from([("query".to_string(), string_type())]),
                },
                output: TypeIR::Named {
                    name: "TaskOutput".to_string(),
                },
                artifacts: vec![ArtifactSlotIR {
                    name: "draft".to_string(),
                    ty: TypeIR::Named {
                        name: "Draft".to_string(),
                    },
                }],
                body: vec![TaskNodeIR::Stage(StageIR {
                    name: "write".to_string(),
                    stage_kind: StageKindIR::Prompt,
                    component: "writer".to_string(),
                    input: ExprIR::Ident {
                        name: "input".to_string(),
                    },
                    output: "draft".to_string(),
                    when: None,
                })],
                emit: vec![EmitFieldIR {
                    name: "answer".to_string(),
                    value: ExprIR::FieldAccess {
                        base: Box::new(ExprIR::Ident {
                            name: "draft".to_string(),
                        }),
                        field: "text".to_string(),
                    },
                }],
            }],
            harnesses: vec![HarnessIR {
                name: "baseline".to_string(),
                task: "answer_question".to_string(),
                defaults: vec![BindingIR {
                    key: BindingPathIR {
                        segments: vec!["model".to_string()],
                    },
                    value: ExprIR::Literal {
                        value: LiteralIR::String {
                            value: "gpt-4o-mini".to_string(),
                        },
                    },
                }],
                bindings: vec![scaffold_ir::TargetBindingIR {
                    target: "write".to_string(),
                    bindings: vec![BindingIR {
                        key: BindingPathIR {
                            segments: vec!["system_prompt".to_string()],
                        },
                        value: ExprIR::Literal {
                            value: LiteralIR::String {
                                value: "Be concise.".to_string(),
                            },
                        },
                    }],
                }],
                tunables: Vec::new(),
            }],
            objectives: vec![scaffold_ir::ObjectiveIR {
                name: "quality".to_string(),
                task: "answer_question".to_string(),
                harness: "baseline".to_string(),
                dataset: scaffold_ir::DatasetSpecIR::Inline { cases: Vec::new() },
                repeats: Some(1),
                metrics: vec![MetricIR {
                    name: "dummy".to_string(),
                    expr: ExprIR::Literal {
                        value: LiteralIR::Bool { value: true },
                    },
                }],
                score: ExprIR::Literal {
                    value: LiteralIR::Int { value: 1 },
                },
                split: None,
                select: None,
            }],
        };

        let output = execute_task(
            &ir,
            "answer_question",
            Some("baseline"),
            Value::Map(HashMap::from([(
                "query".to_string(),
                Value::String("What is Scaffold?".to_string()),
            )])),
            Path::new("."),
        )
        .unwrap();

        match output {
            Value::Struct { type_name, fields } => {
                assert_eq!(type_name, "TaskOutput");
                assert_eq!(
                    fields.get("answer"),
                    Some(&Value::String("hello world".into()))
                );
            }
            other => panic!("expected struct output, got {:?}", other),
        }

        std::env::remove_var("SCAFFOLD_LLM_MOCK_JSON");
    }
}
