//! Execution engine for scaffold IR
//!
//! Executes tasks and tools by interpreting the IR.

use crate::error::{InterpreterError, Result};
use crate::foreign::ForeignRegistry;
use scaffold_ir::{
    ToolIR, ToolImplIR, ToolExprIR, ExprIR, LiteralIR,
    PromptIR, AgentIR, PipelineIR, PipelineCallIR, StringOrFileIR, TypeIR,
};
use scaffold_runtime::{PromptManager, Value};
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::path::Path;

/// Tool executor - executes tool implementations
pub struct ToolExecutor {
    /// Cached tool results (for pure tools)
    #[allow(dead_code)]
    cache: HashMap<String, Value>,
    /// Registered tools for ToolCall lookups
    tools: HashMap<String, ToolIR>,
    /// Registered prompts for prompt calls
    prompts_ir: HashMap<String, PromptIR>,
    /// Registered agents
    agents: HashMap<String, AgentIR>,
    /// Registered pipelines
    pipelines: HashMap<String, PipelineIR>,
    /// Base path for file() references
    base_path: Option<std::path::PathBuf>,
    /// Foreign function registry
    foreign_registry: ForeignRegistry,
}

impl Default for ToolExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolExecutor {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            tools: HashMap::new(),
            prompts_ir: HashMap::new(),
            agents: HashMap::new(),
            pipelines: HashMap::new(),
            base_path: None,
            foreign_registry: ForeignRegistry::new(),
        }
    }

    /// Set base path for file() references
    pub fn with_base_path(mut self, path: std::path::PathBuf) -> Self {
        self.base_path = Some(path);
        self
    }

    /// Set the foreign function registry
    pub fn with_foreign_registry(mut self, registry: ForeignRegistry) -> Self {
        self.foreign_registry = registry;
        self
    }

    /// Get a mutable reference to the foreign registry for registration
    pub fn foreign_registry_mut(&mut self) -> &mut ForeignRegistry {
        &mut self.foreign_registry
    }

    /// Get a reference to the foreign registry
    pub fn foreign_registry(&self) -> &ForeignRegistry {
        &self.foreign_registry
    }

    /// Register tools for ToolCall lookups
    pub fn register_tools(&mut self, tools: &[ToolIR]) {
        for tool in tools {
            self.tools.insert(tool.name.clone(), tool.clone());
        }
    }

    /// Register prompts
    pub fn register_prompts(&mut self, prompts: &[PromptIR]) {
        for prompt in prompts {
            self.prompts_ir.insert(prompt.name.clone(), prompt.clone());
        }
    }

    /// Register agents
    pub fn register_agents(&mut self, agents: &[AgentIR]) {
        for agent in agents {
            self.agents.insert(agent.name.clone(), agent.clone());
        }
    }

    /// Register pipelines
    pub fn register_pipelines(&mut self, pipelines: &[PipelineIR]) {
        for pipeline in pipelines {
            self.pipelines.insert(pipeline.name.clone(), pipeline.clone());
        }
    }

    /// Get a registered tool by name
    pub fn get_tool(&self, name: &str) -> Option<&ToolIR> {
        self.tools.get(name)
    }

    /// Get a registered prompt by name
    pub fn get_prompt(&self, name: &str) -> Option<&PromptIR> {
        self.prompts_ir.get(name)
    }

    /// Get a registered agent by name
    pub fn get_agent(&self, name: &str) -> Option<&AgentIR> {
        self.agents.get(name)
    }

    /// Get a registered pipeline by name
    pub fn get_pipeline(&self, name: &str) -> Option<&PipelineIR> {
        self.pipelines.get(name)
    }

    /// Resolve a StringOrFileIR to actual content
    fn resolve_string_or_file(&self, sof: &StringOrFileIR) -> Result<String> {
        match sof {
            StringOrFileIR::Literal { value } => Ok(value.clone()),
            StringOrFileIR::File { path } => {
                let full_path = if let Some(ref base) = self.base_path {
                    base.join(path)
                } else {
                    Path::new(path).to_path_buf()
                };
                std::fs::read_to_string(&full_path)
                    .map_err(|e| InterpreterError::Runtime(format!(
                        "Failed to read file '{}': {}",
                        full_path.display(), e
                    )))
            }
        }
    }

    /// Generate JSON schema from TypeIR
    fn type_to_json_schema(&self, ty: &TypeIR) -> String {
        match ty {
            TypeIR::Bool => "boolean".to_string(),
            TypeIR::Int => "integer".to_string(),
            TypeIR::Float => "number".to_string(),
            TypeIR::String => "\"string\"".to_string(),
            TypeIR::Any => "\"any\"".to_string(),
            TypeIR::Bytes => "\"string\"".to_string(),
            TypeIR::List { element } => {
                format!("[{}]", self.type_to_json_schema(element))
            }
            TypeIR::Map { key: _, value } => {
                format!("{{ \"key\": {} }}", self.type_to_json_schema(value))
            }
            TypeIR::Option { inner } => self.type_to_json_schema(inner),
            TypeIR::Result { ok, .. } => self.type_to_json_schema(ok),
            TypeIR::Struct { fields } => {
                let field_strs: Vec<String> = fields.iter()
                    .map(|(k, v)| format!("\"{}\": {}", k, self.type_to_json_schema(v)))
                    .collect();
                format!("{{ {} }}", field_strs.join(", "))
            }
            TypeIR::Named { name } => format!("\"{}\"", name),
        }
    }

    /// Execute a tool with the given input
    pub async fn execute(
        &mut self,
        tool: &ToolIR,
        input: Value,
        prompts: &PromptManager,
    ) -> Result<Value> {
        // Check preconditions if spec exists
        if let Some(ref spec) = tool.spec {
            for pre in &spec.preconditions {
                let result = self.eval_expr(pre, &input)?;
                if !result.as_bool().unwrap_or(false) {
                    return Err(InterpreterError::PreconditionFailed(format!("{:?}", pre)));
                }
            }
        }

        // Execute implementation with expected output type
        let result = match &tool.implementation {
            Some(impl_) => self.execute_impl(impl_, &input, prompts, Some(&tool.output)).await?,
            None => {
                return Err(InterpreterError::Runtime(format!(
                    "Tool '{}' has no implementation",
                    tool.name
                )));
            }
        };

        // Check postconditions
        if let Some(ref spec) = tool.spec {
            for post in &spec.postconditions {
                // Create context with output available
                let mut ctx = HashMap::new();
                ctx.insert("output".to_string(), result.clone());
                ctx.insert("input".to_string(), input.clone());
                let ctx_value = Value::Map(ctx);

                let check = self.eval_expr(post, &ctx_value)?;
                if !check.as_bool().unwrap_or(false) {
                    return Err(InterpreterError::PostconditionFailed(format!("{:?}", post)));
                }
            }
        }

        Ok(result)
    }

    /// Execute a prompt with the given input
    pub async fn execute_prompt(
        &mut self,
        prompt: &PromptIR,
        input: Value,
        prompts: &PromptManager,
    ) -> Result<Value> {
        // Resolve system prompt if present
        let system = if let Some(ref sys) = prompt.system {
            Some(self.resolve_string_or_file(sys)?)
        } else {
            None
        };

        // Resolve template
        let template = self.resolve_string_or_file(&prompt.template)?;

        // Interpolate template with input
        let rendered = prompts.interpolate(&template, &input)
            .map_err(|e| InterpreterError::Runtime(e.to_string()))?;

        // Generate JSON schema from output type
        let schema = self.type_to_json_schema(&prompt.output);

        // Build full prompt with system message if present
        let full_prompt = if let Some(sys) = system {
            format!("{}\n\n{}", sys, rendered)
        } else {
            rendered
        };

        // Query LLM with structured output
        let result = scaffold_runtime::llm::query_structured(&full_prompt, &schema)
            .await
            .map_err(|e| InterpreterError::LlmError(e.to_string()))?;

        Ok(result)
    }

    /// Execute an agent with the given input
    pub async fn execute_agent(
        &mut self,
        agent: &AgentIR,
        input: Value,
        prompts: &PromptManager,
    ) -> Result<Value> {
        // Resolve system prompt
        let system = self.resolve_string_or_file(&agent.system)?;

        let max_turns = agent.max_turns.unwrap_or(10);
        let mut turn = 0;
        let mut conversation_history = Vec::new();

        // Add initial user input to history
        let input_str = serde_json::to_string_pretty(&input)
            .unwrap_or_else(|_| format!("{:?}", input));
        conversation_history.push(format!("User input: {}", input_str));

        // Build tool descriptions for the system prompt
        let tool_descriptions: Vec<String> = agent.tools.iter()
            .filter_map(|tool_name| {
                self.tools.get(tool_name).map(|tool| {
                    let input_schema = self.type_to_json_schema(&tool.input);
                    let output_schema = self.type_to_json_schema(&tool.output);
                    format!("- {}: input={}, output={}", tool.name, input_schema, output_schema)
                })
            })
            .collect();

        let tools_prompt = if tool_descriptions.is_empty() {
            String::new()
        } else {
            format!("\n\nAvailable tools:\n{}\n\nTo call a tool, respond with: TOOL_CALL: tool_name({{\"arg\": \"value\"}})\nTo finish, respond with: DONE: {{your final answer as JSON}}", tool_descriptions.join("\n"))
        };

        let output_schema = self.type_to_json_schema(&agent.output);

        loop {
            if turn >= max_turns {
                return Err(InterpreterError::Runtime(format!(
                    "Agent exceeded max_turns ({})", max_turns
                )));
            }
            turn += 1;

            // Build prompt for this turn
            let history_str = conversation_history.join("\n\n");
            let turn_prompt = format!(
                "{}{}\n\nExpected output format: {}\n\nConversation so far:\n{}\n\nWhat would you like to do next?",
                system, tools_prompt, output_schema, history_str
            );

            // Query LLM
            let response = scaffold_runtime::llm_query(&turn_prompt)
                .await
                .map_err(|e| InterpreterError::LlmError(e.to_string()))?;

            // Parse response for tool calls or done signal
            if response.contains("DONE:") {
                // Extract final answer
                if let Some(done_idx) = response.find("DONE:") {
                    let json_str = response[done_idx + 5..].trim();
                    // Try to extract JSON from the response
                    let json_str = extract_json(json_str);
                    let json_value: serde_json::Value = serde_json::from_str(json_str)
                        .map_err(|e| InterpreterError::Runtime(format!(
                            "Failed to parse agent final answer as JSON: {}. Response was: {}",
                            e, json_str
                        )))?;
                    return Ok(json_to_value(json_value));
                }
            } else if response.contains("TOOL_CALL:") {
                // Extract and execute tool call
                if let Some(call_idx) = response.find("TOOL_CALL:") {
                    let call_str = response[call_idx + 10..].trim();

                    // Parse tool name and args: tool_name({"arg": "value"})
                    if let Some(paren_idx) = call_str.find('(') {
                        let tool_name = call_str[..paren_idx].trim();
                        let args_str = &call_str[paren_idx..];

                        // Find matching closing paren
                        let args_json = if args_str.starts_with('(') && args_str.contains(')') {
                            let end_idx = args_str.rfind(')').unwrap_or(args_str.len());
                            &args_str[1..end_idx]
                        } else {
                            "{}"
                        };

                        // Parse args
                        let args_value: serde_json::Value = serde_json::from_str(args_json)
                            .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
                        let args = json_to_value(args_value);

                        // Execute tool
                        if let Some(tool) = self.tools.get(tool_name).cloned() {
                            match self.execute(&tool, args, prompts).await {
                                Ok(result) => {
                                    let result_str = serde_json::to_string_pretty(&result)
                                        .unwrap_or_else(|_| format!("{:?}", result));
                                    conversation_history.push(format!(
                                        "Assistant: TOOL_CALL: {}({})\n\nTool result: {}",
                                        tool_name, args_json, result_str
                                    ));
                                }
                                Err(e) => {
                                    conversation_history.push(format!(
                                        "Assistant: TOOL_CALL: {}({})\n\nTool error: {}",
                                        tool_name, args_json, e
                                    ));
                                }
                            }
                        } else {
                            conversation_history.push(format!(
                                "Assistant: TOOL_CALL: {}({})\n\nError: Tool '{}' not found",
                                tool_name, args_json, tool_name
                            ));
                        }
                    }
                }
            } else {
                // Regular response - add to history
                conversation_history.push(format!("Assistant: {}", response));
            }
        }
    }

    /// Execute a pipeline with the given input
    pub async fn execute_pipeline(
        &mut self,
        pipeline: &PipelineIR,
        input: Value,
        prompts: &PromptManager,
    ) -> Result<Value> {
        let mut bindings: HashMap<String, Value> = HashMap::new();

        // Add input to bindings
        if let Value::Map(m) = &input {
            bindings.extend(m.clone());
        }
        bindings.insert("input".to_string(), input.clone());

        let mut last_value = Value::Null;

        for step in &pipeline.steps {
            let ctx = Value::Map(bindings.clone());

            let result = match &step.call {
                PipelineCallIR::Prompt { name, args } => {
                    // Get the prompt
                    let prompt = self.prompts_ir.get(name).cloned()
                        .ok_or_else(|| InterpreterError::Runtime(format!(
                            "Prompt '{}' not found", name
                        )))?;

                    // Build input from args
                    let mut input_map = HashMap::new();
                    let field_names: Vec<String> = match &prompt.input {
                        TypeIR::Struct { fields } => fields.keys().cloned().collect(),
                        _ => Vec::new(),
                    };

                    for (i, arg) in args.iter().enumerate() {
                        let val = self.execute_tool_expr(arg, &ctx, prompts, None).await?;
                        let key = field_names.get(i)
                            .cloned()
                            .unwrap_or_else(|| format!("arg{}", i));
                        input_map.insert(key, val);
                    }

                    self.execute_prompt(&prompt, Value::Map(input_map), prompts).await?
                }
                PipelineCallIR::Tool { name, args } => {
                    // Get the tool
                    let tool = self.tools.get(name).cloned()
                        .ok_or_else(|| InterpreterError::ToolNotFound(name.clone()))?;

                    // Build input from args
                    let mut input_map = HashMap::new();
                    let field_names: Vec<String> = match &tool.input {
                        TypeIR::Struct { fields } => fields.keys().cloned().collect(),
                        _ => Vec::new(),
                    };

                    for (i, arg) in args.iter().enumerate() {
                        let val = self.execute_tool_expr(arg, &ctx, prompts, None).await?;
                        let key = field_names.get(i)
                            .cloned()
                            .unwrap_or_else(|| format!("arg{}", i));
                        input_map.insert(key, val);
                    }

                    self.execute(&tool, Value::Map(input_map), prompts).await?
                }
                PipelineCallIR::Expr { expr } => {
                    // Evaluate expression in current bindings context
                    self.execute_tool_expr(expr, &ctx, prompts, None).await?
                }
            };

            if let Some(ref binding_name) = step.binding {
                bindings.insert(binding_name.clone(), result.clone());
            }
            last_value = result;
        }

        // If pipeline declares a struct output, synthesize from bindings
        if let TypeIR::Struct { fields } = &pipeline.output {
            let mut out_map: HashMap<String, Value> = HashMap::new();
            for (k, _) in fields {
                if let Some(v) = bindings.get(k).cloned() {
                    out_map.insert(k.clone(), v);
                }
            }
            if !out_map.is_empty() {
                return Ok(Value::Map(out_map));
            }
        }
        Ok(last_value)
    }

    /// Execute a tool implementation
    ///
    /// Returns a boxed future to handle recursive async calls
    fn execute_impl<'a>(
        &'a mut self,
        impl_: &'a ToolImplIR,
        input: &'a Value,
        prompts: &'a PromptManager,
        expected: Option<&'a TypeIR>,
    ) -> Pin<Box<dyn Future<Output = Result<Value>> + Send + 'a>> {
        Box::pin(async move {
            match impl_ {
                ToolImplIR::Expr { expr } => {
                    self.execute_tool_expr(expr, input, prompts, expected).await
                }
                ToolImplIR::Sequence { statements } => {
                    let mut last_value = Value::Null;
                    let mut bindings: HashMap<String, Value> = HashMap::new();

                    // Add input to bindings
                    if let Value::Map(m) = input {
                        bindings.extend(m.clone());
                    }
                    bindings.insert("input".to_string(), input.clone());

                    for stmt in statements {
                        let ctx = Value::Map(bindings.clone());
                        let value = self.execute_tool_expr(&stmt.expr, &ctx, prompts, None).await?;

                        if let Some(ref name) = stmt.binding {
                            bindings.insert(name.clone(), value.clone());
                        }
                        last_value = value;
                    }

                    // Synthesize struct output if expected
                    if let Some(TypeIR::Struct { fields }) = expected {
                        let mut out_map: HashMap<String, Value> = HashMap::new();
                        for (k, _) in fields {
                            if let Some(v) = bindings.get(k).cloned() {
                                out_map.insert(k.clone(), v);
                            }
                        }
                        if !out_map.is_empty() {
                            return Ok(Value::Map(out_map));
                        }
                    }
                    Ok(last_value)
                }
                ToolImplIR::Parallel { statements } => {
                    // For now, execute sequentially (TODO: true parallel with tokio::join!)
                    let mut last_value = Value::Null;
                    let mut bindings: HashMap<String, Value> = HashMap::new();

                    // Add input to bindings
                    if let Value::Map(m) = input {
                        bindings.extend(m.clone());
                    }
                    bindings.insert("input".to_string(), input.clone());

                    for stmt in statements {
                        let ctx = Value::Map(bindings.clone());
                        let value = self.execute_tool_expr(&stmt.expr, &ctx, prompts, None).await?;

                        if let Some(ref name) = stmt.binding {
                            bindings.insert(name.clone(), value.clone());
                        }
                        last_value = value;
                    }

                    if let Some(TypeIR::Struct { fields }) = expected {
                        let mut out_map: HashMap<String, Value> = HashMap::new();
                        for (k, _) in fields {
                            if let Some(v) = bindings.get(k).cloned() {
                                out_map.insert(k.clone(), v);
                            }
                        }
                        if !out_map.is_empty() {
                            return Ok(Value::Map(out_map));
                        }
                    }
                    Ok(last_value)
                }
            }
        })
    }

    /// Execute a tool expression
    ///
    /// Returns a boxed future to handle recursive async calls
    fn execute_tool_expr<'a>(
        &'a mut self,
        expr: &'a ToolExprIR,
        ctx: &'a Value,
        prompts: &'a PromptManager,
        expected: Option<&'a TypeIR>,
    ) -> Pin<Box<dyn Future<Output = Result<Value>> + Send + 'a>> {
        Box::pin(async move {
            match expr {
                ToolExprIR::Ident { name } => {
                    self.get_value(ctx, name)
                }
                ToolExprIR::FieldAccess { base, field } => {
                    let base_val = self.execute_tool_expr(base, ctx, prompts, None).await?;
                    self.get_field(&base_val, field)
                }
                ToolExprIR::Literal { value } => {
                    Ok(self.literal_to_value(value))
                }
                ToolExprIR::Shell { command } => {
                    // Interpolate variables in command
                    let cmd = prompts.interpolate(command, ctx)
                        .map_err(|e| InterpreterError::Runtime(e.to_string()))?;
                    if let Some(exp) = expected {
                        match exp {
                            TypeIR::Bytes => {
                                let bytes = scaffold_runtime::shell::execute_bytes(&cmd)
                                    .map_err(|e| InterpreterError::ShellError(e.to_string()))?;
                                Ok(Value::Bytes(bytes))
                            }
                            TypeIR::Int => {
                                let s = scaffold_runtime::shell::execute(&cmd)
                                    .map_err(|e| InterpreterError::ShellError(e.to_string()))?;
                                let i = scaffold_runtime::parse::parse_i64(s.trim())
                                    .map_err(|e| InterpreterError::Runtime(e.to_string()))?;
                                Ok(Value::Int(i))
                            }
                            TypeIR::Float => {
                                let s = scaffold_runtime::shell::execute(&cmd)
                                    .map_err(|e| InterpreterError::ShellError(e.to_string()))?;
                                let f = scaffold_runtime::parse::parse_f64(s.trim())
                                    .map_err(|e| InterpreterError::Runtime(e.to_string()))?;
                                Ok(Value::Float(f))
                            }
                            TypeIR::Bool => {
                                let s = scaffold_runtime::shell::execute(&cmd)
                                    .map_err(|e| InterpreterError::ShellError(e.to_string()))?;
                                let b = scaffold_runtime::parse::parse_bool(s.trim())
                                    .map_err(|e| InterpreterError::Runtime(e.to_string()))?;
                                Ok(Value::Bool(b))
                            }
                            TypeIR::String => {
                                let s = scaffold_runtime::shell::execute(&cmd)
                                    .map_err(|e| InterpreterError::ShellError(e.to_string()))?;
                                Ok(Value::String(s.trim().to_string()))
                            }
                            TypeIR::Struct { fields } => {
                                let s = scaffold_runtime::shell::execute(&cmd)
                                    .map_err(|e| InterpreterError::ShellError(e.to_string()))?;
                                let st = s.trim();
                                if fields.len() == 1 {
                                    let (fname, fty) = fields.iter().next().unwrap();
                                    let inner = match fty {
                                        TypeIR::Int => Value::Int(scaffold_runtime::parse::parse_i64(st)
                                            .map_err(|e| InterpreterError::Runtime(e.to_string()))?),
                                        TypeIR::Float => Value::Float(scaffold_runtime::parse::parse_f64(st)
                                            .map_err(|e| InterpreterError::Runtime(e.to_string()))?),
                                        TypeIR::Bool => Value::Bool(scaffold_runtime::parse::parse_bool(st)
                                            .map_err(|e| InterpreterError::Runtime(e.to_string()))?),
                                        TypeIR::String => Value::String(st.to_string()),
                                        _ => match serde_json::from_str::<serde_json::Value>(st) {
                                            Ok(j) => json_to_value(j),
                                            Err(_) => Value::String(st.to_string()),
                                        },
                                    };
                                    let mut m = HashMap::new();
                                    m.insert(fname.clone(), inner);
                                    Ok(Value::Map(m))
                                } else {
                                    match serde_json::from_str::<serde_json::Value>(st) {
                                        Ok(j) => Ok(json_to_value(j)),
                                        Err(_) => Ok(Value::String(s)),
                                    }
                                }
                            }
                            _ => {
                                let s = scaffold_runtime::shell::execute(&cmd)
                                    .map_err(|e| InterpreterError::ShellError(e.to_string()))?;
                                Ok(Value::String(s))
                            }
                        }
                    } else {
                        let output = scaffold_runtime::shell::execute(&cmd)
                            .map_err(|e| InterpreterError::ShellError(e.to_string()))?;
                        Ok(Value::String(output))
                    }
                }
                ToolExprIR::ForeignCall { module, function, args } => {
                    // Evaluate arguments
                    let mut arg_values = Vec::new();
                    for arg in args {
                        let val = self.execute_tool_expr(arg, ctx, prompts, None).await?;
                        arg_values.push(val);
                    }

                    // Call the foreign function through the registry
                    self.foreign_registry.call(module, function, arg_values)
                }
                ToolExprIR::ToolCall { tool, args } => {
                    // Look up the tool first to get input field names
                    let tool_ir = self.get_tool(tool).cloned()
                        .ok_or_else(|| InterpreterError::ToolNotFound(tool.clone()))?;

                    // Get field names from tool's input type
                    let field_names: Vec<String> = match &tool_ir.input {
                        scaffold_ir::TypeIR::Struct { fields } => {
                            fields.keys().cloned().collect()
                        }
                        _ => Vec::new(),
                    };

                    // Build input from args, using field names if available
                    let mut input_map = HashMap::new();
                    for (i, arg) in args.iter().enumerate() {
                        let val = self.execute_tool_expr(arg, ctx, prompts, None).await?;
                        let key = field_names.get(i)
                            .cloned()
                            .unwrap_or_else(|| format!("arg{}", i));
                        input_map.insert(key, val);
                    }
                    let input = Value::Map(input_map);

                    // Execute the tool
                    self.execute(&tool_ir, input, prompts).await
                }
                ToolExprIR::Pipe { left, right } => {
                    let left_val = self.execute_tool_expr(left, ctx, prompts, None).await?;
                    // Use left result as input to right
                    self.execute_tool_expr(right, &left_val, prompts, expected).await
                }
                ToolExprIR::If { condition, then_branch, else_branch } => {
                    let cond = self.eval_expr(condition, ctx)?;
                    if cond.as_bool().unwrap_or(false) {
                        self.execute_impl(then_branch, ctx, prompts, expected).await
                    } else if let Some(else_) = else_branch {
                        self.execute_impl(else_, ctx, prompts, expected).await
                    } else {
                        Ok(Value::Null)
                    }
                }
                ToolExprIR::Match { scrutinee, arms } => {
                    let scrutinee_val = self.execute_tool_expr(scrutinee, ctx, prompts, None).await?;

                    for arm in arms {
                        // Simple pattern matching - check equality
                        let pattern_val = self.eval_expr(&arm.pattern, ctx)?;
                        if scrutinee_val == pattern_val {
                            return self.execute_impl(&arm.body, ctx, prompts, expected).await;
                        }
                    }

                    // No match - return null
                    Ok(Value::Null)
                }
                ToolExprIR::For { variable, iterable, body } => {
                    let iterable_val = self.execute_tool_expr(iterable, ctx, prompts, None).await?;

                    // Get list to iterate over
                    let items = match &iterable_val {
                        Value::List(items) => items.clone(),
                        _ => return Err(InterpreterError::TypeMismatch {
                            expected: "list".to_string(),
                            actual: iterable_val.type_name().to_string(),
                        }),
                    };

                    let mut last_result = Value::Null;

                    // Create a mutable context with the loop variable
                    for item in items {
                        let loop_ctx = match ctx {
                            Value::Map(m) => {
                                let mut new_map = m.clone();
                                new_map.insert(variable.clone(), item);
                                Value::Map(new_map)
                            }
                            _ => {
                                let mut new_map = HashMap::new();
                                new_map.insert(variable.clone(), item);
                                Value::Map(new_map)
                            }
                        };

                        match self.execute_impl(body, &loop_ctx, prompts, expected).await {
                            Ok(val) => last_result = val,
                            Err(InterpreterError::Break) => break,
                            Err(InterpreterError::Continue) => continue,
                            Err(e) => return Err(e),
                        }
                    }

                    Ok(last_result)
                }
                ToolExprIR::While { condition, body } => {
                    let mut last_result = Value::Null;

                    loop {
                        let cond = self.eval_expr(condition, ctx)?;
                        if !cond.as_bool().unwrap_or(false) {
                            break;
                        }

                        match self.execute_impl(body, ctx, prompts, expected).await {
                            Ok(val) => last_result = val,
                            Err(InterpreterError::Break) => break,
                            Err(InterpreterError::Continue) => continue,
                            Err(e) => return Err(e),
                        }
                    }

                    Ok(last_result)
                }
                ToolExprIR::Loop { body } => {
                    let mut last_result = Value::Null;

                    loop {
                        match self.execute_impl(body, ctx, prompts, expected).await {
                            Ok(val) => last_result = val,
                            Err(InterpreterError::Break) => break,
                            Err(InterpreterError::Continue) => continue,
                            Err(e) => return Err(e),
                        }
                    }

                    Ok(last_result)
                }
                ToolExprIR::Break => Err(InterpreterError::Break),
                ToolExprIR::Continue => Err(InterpreterError::Continue),
                ToolExprIR::Expr { expr } => {
                    // Evaluate a general expression (arithmetic, comparisons, etc.)
                    self.eval_expr(expr, &ctx)
                }
            }
        })
    }

    /// Evaluate a simple expression
    fn eval_expr(&self, expr: &ExprIR, ctx: &Value) -> Result<Value> {
        match expr {
            ExprIR::Literal { value } => Ok(self.literal_to_value(value)),
            ExprIR::Ident { name } => self.get_value(ctx, name),
            ExprIR::FieldAccess { base, field } => {
                let base_val = self.eval_expr(base, ctx)?;
                self.get_field(&base_val, field)
            }
            ExprIR::Binary { left, op, right } => {
                let l = self.eval_expr(left, ctx)?;
                let r = self.eval_expr(right, ctx)?;
                self.eval_binary_op(&l, op, &r)
            }
            ExprIR::Call { function, args } => {
                // Built-in functions
                let arg_vals: Vec<Value> = args
                    .iter()
                    .map(|a| self.eval_expr(a, ctx))
                    .collect::<Result<Vec<_>>>()?;

                self.eval_builtin(function, &arg_vals)
            }
        }
    }

    /// Convert literal to value
    fn literal_to_value(&self, lit: &LiteralIR) -> Value {
        match lit {
            LiteralIR::Int { value } => Value::Int(*value),
            LiteralIR::Float { value } => Value::Float(*value),
            LiteralIR::String { value } => Value::String(value.clone()),
            LiteralIR::Bool { value } => Value::Bool(*value),
            LiteralIR::Null => Value::Null,
        }
    }

    /// Get a value from context
    fn get_value(&self, ctx: &Value, name: &str) -> Result<Value> {
        // Special identifier: 'input' refers to the entire current context
        if name == "input" {
            return Ok(ctx.clone());
        }
        match ctx {
            Value::Map(m) => m.get(name).cloned()
                .ok_or_else(|| InterpreterError::VariableNotFound(name.to_string())),
            Value::Struct { fields, .. } => fields.get(name).cloned()
                .ok_or_else(|| InterpreterError::VariableNotFound(name.to_string())),
            _ => Err(InterpreterError::VariableNotFound(name.to_string())),
        }
    }

    /// Get a field from a value
    fn get_field(&self, val: &Value, field: &str) -> Result<Value> {
        match val {
            Value::Map(m) => m.get(field).cloned()
                .ok_or_else(|| InterpreterError::FieldNotFound {
                    type_name: "map".to_string(),
                    field: field.to_string(),
                }),
            Value::Struct { type_name, fields } => fields.get(field).cloned()
                .ok_or_else(|| InterpreterError::FieldNotFound {
                    type_name: type_name.clone(),
                    field: field.to_string(),
                }),
            _ => Err(InterpreterError::FieldNotFound {
                type_name: val.type_name().to_string(),
                field: field.to_string(),
            }),
        }
    }

    /// Evaluate binary operation
    fn eval_binary_op(&self, left: &Value, op: &str, right: &Value) -> Result<Value> {
        match op {
            "==" => Ok(Value::Bool(left == right)),
            "!=" => Ok(Value::Bool(left != right)),
            "<" => {
                match (left.as_int(), right.as_int()) {
                    (Some(l), Some(r)) => Ok(Value::Bool(l < r)),
                    _ => match (left.as_float(), right.as_float()) {
                        (Some(l), Some(r)) => Ok(Value::Bool(l < r)),
                        _ => Err(InterpreterError::TypeMismatch {
                            expected: "number".to_string(),
                            actual: format!("{}, {}", left.type_name(), right.type_name()),
                        }),
                    }
                }
            }
            "<=" => {
                match (left.as_int(), right.as_int()) {
                    (Some(l), Some(r)) => Ok(Value::Bool(l <= r)),
                    _ => match (left.as_float(), right.as_float()) {
                        (Some(l), Some(r)) => Ok(Value::Bool(l <= r)),
                        _ => Err(InterpreterError::TypeMismatch {
                            expected: "number".to_string(),
                            actual: format!("{}, {}", left.type_name(), right.type_name()),
                        }),
                    }
                }
            }
            ">" => {
                match (left.as_int(), right.as_int()) {
                    (Some(l), Some(r)) => Ok(Value::Bool(l > r)),
                    _ => match (left.as_float(), right.as_float()) {
                        (Some(l), Some(r)) => Ok(Value::Bool(l > r)),
                        _ => Err(InterpreterError::TypeMismatch {
                            expected: "number".to_string(),
                            actual: format!("{}, {}", left.type_name(), right.type_name()),
                        }),
                    }
                }
            }
            ">=" => {
                match (left.as_int(), right.as_int()) {
                    (Some(l), Some(r)) => Ok(Value::Bool(l >= r)),
                    _ => match (left.as_float(), right.as_float()) {
                        (Some(l), Some(r)) => Ok(Value::Bool(l >= r)),
                        _ => Err(InterpreterError::TypeMismatch {
                            expected: "number".to_string(),
                            actual: format!("{}, {}", left.type_name(), right.type_name()),
                        }),
                    }
                }
            }
            "+" => {
                match (left, right) {
                    (Value::Int(l), Value::Int(r)) => Ok(Value::Int(l + r)),
                    (Value::Float(l), Value::Float(r)) => Ok(Value::Float(l + r)),
                    (Value::Int(l), Value::Float(r)) => Ok(Value::Float(*l as f64 + r)),
                    (Value::Float(l), Value::Int(r)) => Ok(Value::Float(l + *r as f64)),
                    (Value::String(l), Value::String(r)) => Ok(Value::String(format!("{}{}", l, r))),
                    _ => Err(InterpreterError::TypeMismatch {
                        expected: "number or string".to_string(),
                        actual: format!("{}, {}", left.type_name(), right.type_name()),
                    }),
                }
            }
            "-" => {
                match (left, right) {
                    (Value::Int(l), Value::Int(r)) => Ok(Value::Int(l - r)),
                    (Value::Float(l), Value::Float(r)) => Ok(Value::Float(l - r)),
                    (Value::Int(l), Value::Float(r)) => Ok(Value::Float(*l as f64 - r)),
                    (Value::Float(l), Value::Int(r)) => Ok(Value::Float(l - *r as f64)),
                    _ => Err(InterpreterError::TypeMismatch {
                        expected: "number".to_string(),
                        actual: format!("{}, {}", left.type_name(), right.type_name()),
                    }),
                }
            }
            "*" => {
                match (left, right) {
                    (Value::Int(l), Value::Int(r)) => Ok(Value::Int(l * r)),
                    (Value::Float(l), Value::Float(r)) => Ok(Value::Float(l * r)),
                    (Value::Int(l), Value::Float(r)) => Ok(Value::Float(*l as f64 * r)),
                    (Value::Float(l), Value::Int(r)) => Ok(Value::Float(l * *r as f64)),
                    _ => Err(InterpreterError::TypeMismatch {
                        expected: "number".to_string(),
                        actual: format!("{}, {}", left.type_name(), right.type_name()),
                    }),
                }
            }
            "/" => {
                match (left, right) {
                    (Value::Int(l), Value::Int(r)) if *r != 0 => Ok(Value::Int(l / r)),
                    (Value::Float(l), Value::Float(r)) => Ok(Value::Float(l / r)),
                    (Value::Int(l), Value::Float(r)) => Ok(Value::Float(*l as f64 / r)),
                    (Value::Float(l), Value::Int(r)) => Ok(Value::Float(l / *r as f64)),
                    _ => Err(InterpreterError::Runtime("Division error".to_string())),
                }
            }
            "&&" | "and" => {
                let l = left.as_bool().unwrap_or(false);
                let r = right.as_bool().unwrap_or(false);
                Ok(Value::Bool(l && r))
            }
            "||" | "or" => {
                let l = left.as_bool().unwrap_or(false);
                let r = right.as_bool().unwrap_or(false);
                Ok(Value::Bool(l || r))
            }
            _ => Err(InterpreterError::Runtime(format!("Unknown operator: {}", op))),
        }
    }

    /// Evaluate built-in function
    fn eval_builtin(&self, name: &str, args: &[Value]) -> Result<Value> {
        match name {
            "len" | "length" => {
                if let Some(v) = args.first() {
                    match v {
                        Value::String(s) => Ok(Value::Int(s.len() as i64)),
                        Value::List(l) => Ok(Value::Int(l.len() as i64)),
                        Value::Bytes(b) => Ok(Value::Int(b.len() as i64)),
                        _ => Ok(Value::Int(0)),
                    }
                } else {
                    Ok(Value::Int(0))
                }
            }
            "is_some" => {
                Ok(Value::Bool(!matches!(args.first(), Some(Value::Null) | None)))
            }
            "is_none" => {
                Ok(Value::Bool(matches!(args.first(), Some(Value::Null) | None)))
            }
            "not" => {
                let b = args.first().and_then(|v| v.as_bool()).unwrap_or(false);
                Ok(Value::Bool(!b))
            }
            "abs" => {
                if let Some(v) = args.first() {
                    match v {
                        Value::Int(i) => Ok(Value::Int(i.abs())),
                        Value::Float(f) => Ok(Value::Float(f.abs())),
                        _ => Ok(Value::Int(0)),
                    }
                } else {
                    Ok(Value::Int(0))
                }
            }
            _ => Err(InterpreterError::Runtime(format!("Unknown function: {}", name))),
        }
    }
}

/// Extract JSON from a string that might have markdown code blocks
fn extract_json(s: &str) -> &str {
    let s = s.trim();
    if s.starts_with("```json") {
        let start = s.find('\n').map(|i| i + 1).unwrap_or(7);
        let end = s.rfind("```").unwrap_or(s.len());
        return s[start..end].trim();
    }
    if s.starts_with("```") {
        let start = s.find('\n').map(|i| i + 1).unwrap_or(3);
        let end = s.rfind("```").unwrap_or(s.len());
        return s[start..end].trim();
    }
    s
}

/// Convert serde_json::Value to scaffold_runtime::Value
fn json_to_value(json: serde_json::Value) -> Value {
    match json {
        serde_json::Value::Null => Value::Null,
        serde_json::Value::Bool(b) => Value::Bool(b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::Int(i)
            } else if let Some(f) = n.as_f64() {
                Value::Float(f)
            } else {
                Value::Null
            }
        }
        serde_json::Value::String(s) => Value::String(s),
        serde_json::Value::Array(arr) => {
            Value::List(arr.into_iter().map(json_to_value).collect())
        }
        serde_json::Value::Object(obj) => {
            let map: HashMap<String, Value> = obj.into_iter()
                .map(|(k, v)| (k, json_to_value(v)))
                .collect();
            Value::Map(map)
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eval_literal() {
        let executor = ToolExecutor::new();
        let ctx = Value::Map(HashMap::new());

        let expr = ExprIR::Literal {
            value: LiteralIR::Int { value: 42 },
        };
        let result = executor.eval_expr(&expr, &ctx).unwrap();
        assert_eq!(result, Value::Int(42));
    }

    #[test]
    fn test_eval_binary() {
        let executor = ToolExecutor::new();
        let ctx = Value::Map(HashMap::new());

        let expr = ExprIR::Binary {
            left: Box::new(ExprIR::Literal { value: LiteralIR::Int { value: 10 } }),
            op: "+".to_string(),
            right: Box::new(ExprIR::Literal { value: LiteralIR::Int { value: 5 } }),
        };
        let result = executor.eval_expr(&expr, &ctx).unwrap();
        assert_eq!(result, Value::Int(15));
    }

    #[test]
    fn test_eval_comparison() {
        let executor = ToolExecutor::new();
        let ctx = Value::Map(HashMap::new());

        let expr = ExprIR::Binary {
            left: Box::new(ExprIR::Literal { value: LiteralIR::Int { value: 10 } }),
            op: ">".to_string(),
            right: Box::new(ExprIR::Literal { value: LiteralIR::Int { value: 5 } }),
        };
        let result = executor.eval_expr(&expr, &ctx).unwrap();
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn test_get_field() {
        let executor = ToolExecutor::new();

        let mut fields = HashMap::new();
        fields.insert("x".to_string(), Value::Int(10));
        fields.insert("y".to_string(), Value::Int(20));
        let ctx = Value::Struct {
            type_name: "Point".to_string(),
            fields,
        };

        let expr = ExprIR::FieldAccess {
            base: Box::new(ExprIR::Ident { name: "input".to_string() }),
            field: "x".to_string(),
        };

        // Wrap in map for lookup
        let mut wrapper = HashMap::new();
        wrapper.insert("input".to_string(), ctx);
        let result = executor.eval_expr(&expr, &Value::Map(wrapper)).unwrap();
        assert_eq!(result, Value::Int(10));
    }
}
