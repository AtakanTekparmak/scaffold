//! Interpreter-style execution for task/harness IR.

use crate::error::{Error, Result};
use crate::llm::{query_structured_with_config, LlmConfig};
use crate::prompt::PromptManager;
use crate::trace::{tracer, TaskStatus, TraceEvent};
use crate::value::{ResultValue, Value};
use crate::{builtins, shell};
use scaffold_ir::{
    type_to_json_schema, types_to_json_schema_document, AgentIR, DatasetSpecIR, ExprIR,
    FiniteDomainIR, HarnessIR, InlineDatasetCaseIR, LiteralIR, ObjectiveIR, PromptIR, ScaffoldIR,
    SelectIR, StageIR, StageKindIR, StringOrFileIR, TaskIR, TaskNodeIR, ToolExprIR, ToolIR,
    ToolImplIR, ToolStatementIR, TunableIR, TuneOperatorIR, TypeIR,
};
use serde::Serialize;
use serde_json::json;
use std::cell::RefCell;
use std::collections::{hash_map::DefaultHasher, BTreeSet, HashMap};
use std::future::Future;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::rc::Rc;
use std::time::Instant;
use tokio::time::{timeout, Duration};

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

/// Optimize an objective by searching the declared finite harness space.
pub fn optimize_objective(
    ir: &ScaffoldIR,
    objective_name: &str,
    base_dir: &Path,
    max_candidates: usize,
) -> Result<ObjectiveOptimizationReport> {
    let interpreter = TaskInterpreter::new(ir, base_dir);
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| Error::Runtime(format!("failed to create optimization runtime: {}", e)))?;
    runtime.block_on(interpreter.optimize(objective_name, max_candidates))
}

struct TaskInterpreter<'a> {
    ir: &'a ScaffoldIR,
    base_dir: PathBuf,
}

#[derive(Default)]
struct ExecutionContext {
    input: Value,
    artifacts: HashMap<String, Value>,
    telemetry: Rc<RefCell<RolloutTelemetry>>,
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
        self.bindings
            .get(target)
            .map(|fields| fields.keys().cloned().collect())
            .unwrap_or_default()
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ObjectiveOptimizationReport {
    pub objective: String,
    pub task: String,
    pub harness: String,
    pub evaluated_candidates: usize,
    pub truncated: bool,
    pub best: CandidateOptimizationReport,
}

#[derive(Debug, Clone, Serialize)]
pub struct CandidateOptimizationReport {
    pub assignments: HashMap<String, Value>,
    pub train: SplitEvaluationSummary,
    pub val: Option<SplitEvaluationSummary>,
    pub test: Option<SplitEvaluationSummary>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SplitEvaluationSummary {
    pub rollouts: usize,
    pub metrics: HashMap<String, f64>,
    pub score: f64,
    pub primary: f64,
    pub tie_breakers: Vec<f64>,
}

#[derive(Debug, Clone)]
struct CandidateEvaluation {
    assignments: HashMap<String, Value>,
    train: SplitEvaluationSummary,
    val: Option<SplitEvaluationSummary>,
    test: Option<SplitEvaluationSummary>,
}

#[derive(Debug, Clone)]
struct RolloutEvaluation {
    metrics: HashMap<String, f64>,
    score: f64,
    primary: f64,
    tie_breakers: Vec<f64>,
}

#[derive(Debug, Clone)]
struct DatasetCase {
    id: Option<String>,
    input: Value,
    expected: Value,
}

#[derive(Debug, Clone, Default, Serialize)]
struct RolloutTelemetry {
    stages: Vec<StageTelemetry>,
    tool_calls: Vec<ToolCallTelemetry>,
    prompt_calls: Vec<PromptCallTelemetry>,
    agent_turns: Vec<AgentTurnTelemetry>,
    loop_iterations: Vec<LoopIterationTelemetry>,
}

#[derive(Debug, Clone, Serialize)]
struct StageTelemetry {
    stage_name: String,
    stage_kind: String,
    component: String,
    output_artifact: String,
    status: String,
    duration_ms: f64,
    input: serde_json::Value,
    output: Option<serde_json::Value>,
    error: Option<String>,
    model: Option<String>,
    variant: Option<String>,
    timeout_secs: Option<u64>,
    retries: u64,
}

#[derive(Debug, Clone, Serialize)]
struct ToolCallTelemetry {
    scope: String,
    tool_name: String,
    duration_ms: f64,
    input: serde_json::Value,
    output: Option<serde_json::Value>,
    error: Option<String>,
    variant: Option<String>,
    timeout_secs: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
struct PromptCallTelemetry {
    scope: String,
    prompt_name: String,
    duration_ms: f64,
    input: serde_json::Value,
    output: Option<serde_json::Value>,
    error: Option<String>,
    model: Option<String>,
    variant: Option<String>,
    timeout_secs: Option<u64>,
    retries: u64,
    prompt_hash: String,
    system_prompt_hash: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct AgentTurnTelemetry {
    scope: String,
    agent_name: String,
    turn_number: u64,
    duration_ms: f64,
    action: String,
    tool: Option<String>,
    completed: bool,
    error: Option<String>,
    model: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct LoopIterationTelemetry {
    loop_name: String,
    iteration: u64,
    max_iters: usize,
    duration_ms: f64,
    terminated: bool,
    error: Option<String>,
}

#[derive(Debug)]
struct TaskRunResult {
    output: Result<Value>,
    telemetry: RolloutTelemetry,
}

#[derive(Debug, Clone, Copy)]
enum TaskTargetKindRuntime<'a> {
    Stage {
        kind: StageKindIR,
        component: &'a str,
    },
    Loop,
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
        self.execute_with_resolved_harness(task, &harness, input)
            .await
    }

    async fn execute_with_resolved_harness(
        &self,
        task: &TaskIR,
        harness: &ResolvedHarness,
        input: Value,
    ) -> Result<Value> {
        self.execute_with_resolved_harness_trace(task, harness, input)
            .await
            .output
    }

    async fn execute_with_resolved_harness_trace(
        &self,
        task: &TaskIR,
        harness: &ResolvedHarness,
        input: Value,
    ) -> TaskRunResult {
        let telemetry = self.new_telemetry_handle();
        let output = match self.validate_value(input, &task.input) {
            Ok(input) => {
                let mut ctx = ExecutionContext {
                    input,
                    artifacts: HashMap::new(),
                    telemetry: telemetry.clone(),
                };
                match self.execute_nodes(&task.body, &mut ctx, &harness).await {
                    Ok(()) => {
                        let output = self.build_output(task, &ctx);
                        output.and_then(|value| self.validate_value(value, &task.output))
                    }
                    Err(error) => Err(error),
                }
            }
            Err(error) => Err(error),
        };

        tracer().record(TraceEvent::TaskComplete {
            task_name: task.name.clone(),
            status: if output.is_ok() {
                TaskStatus::Completed
            } else {
                TaskStatus::Failed
            },
            reward: None,
            error: output.as_ref().err().map(ToString::to_string),
        });

        let telemetry_snapshot = telemetry.borrow().clone();
        TaskRunResult {
            output,
            telemetry: telemetry_snapshot,
        }
    }

    fn new_telemetry_handle(&self) -> Rc<RefCell<RolloutTelemetry>> {
        Rc::new(RefCell::new(RolloutTelemetry::default()))
    }

    fn hash_text(&self, text: &str) -> String {
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        format!("{:016x}", hasher.finish())
    }

    fn record_stage_telemetry(
        &self,
        telemetry: &Rc<RefCell<RolloutTelemetry>>,
        entry: StageTelemetry,
    ) {
        telemetry.borrow_mut().stages.push(entry);
    }

    fn record_tool_call_telemetry(
        &self,
        telemetry: &Rc<RefCell<RolloutTelemetry>>,
        entry: ToolCallTelemetry,
    ) {
        telemetry.borrow_mut().tool_calls.push(entry);
    }

    fn record_prompt_call_telemetry(
        &self,
        telemetry: &Rc<RefCell<RolloutTelemetry>>,
        entry: PromptCallTelemetry,
    ) {
        telemetry.borrow_mut().prompt_calls.push(entry);
    }

    fn record_agent_turn_telemetry(
        &self,
        telemetry: &Rc<RefCell<RolloutTelemetry>>,
        entry: AgentTurnTelemetry,
    ) {
        telemetry.borrow_mut().agent_turns.push(entry);
    }

    fn record_loop_iteration_telemetry(
        &self,
        telemetry: &Rc<RefCell<RolloutTelemetry>>,
        entry: LoopIterationTelemetry,
    ) {
        telemetry.borrow_mut().loop_iterations.push(entry);
    }

    fn telemetry_to_value(&self, telemetry: &RolloutTelemetry) -> Result<Value> {
        serde_json::to_value(telemetry)
            .map(Value::from)
            .map_err(|e| Error::SerializationError(e.to_string()))
    }

    fn rollout_value(
        &self,
        telemetry: &RolloutTelemetry,
        success: bool,
        duration_ms: f64,
        repeat: u64,
        case_id: Option<&str>,
        error: Option<&str>,
    ) -> Result<Value> {
        Ok(Value::Map(HashMap::from([
            ("success".to_string(), Value::Bool(success)),
            ("duration_ms".to_string(), Value::Float(duration_ms)),
            ("latency".to_string(), Value::Float(duration_ms)),
            ("token_cost".to_string(), Value::Float(0.0)),
            ("cost".to_string(), Value::Float(0.0)),
            ("repeat".to_string(), Value::Int(repeat as i64)),
            (
                "case_id".to_string(),
                case_id
                    .map(|id| Value::String(id.to_string()))
                    .unwrap_or(Value::Null),
            ),
            (
                "error".to_string(),
                error
                    .map(|value| Value::String(value.to_string()))
                    .unwrap_or(Value::Null),
            ),
            (
                "stage_count".to_string(),
                Value::Int(telemetry.stages.len() as i64),
            ),
            (
                "tool_call_count".to_string(),
                Value::Int(telemetry.tool_calls.len() as i64),
            ),
            (
                "prompt_call_count".to_string(),
                Value::Int(telemetry.prompt_calls.len() as i64),
            ),
            (
                "agent_turn_count".to_string(),
                Value::Int(telemetry.agent_turns.len() as i64),
            ),
            (
                "loop_iteration_count".to_string(),
                Value::Int(telemetry.loop_iterations.len() as i64),
            ),
            ("trace".to_string(), self.telemetry_to_value(telemetry)?),
        ])))
    }

    fn refresh_rollout_artifact(
        &self,
        ctx: &mut ExecutionContext,
        success: bool,
        duration_ms: f64,
        repeat: u64,
        case_id: Option<&str>,
        error: Option<&str>,
    ) -> Result<()> {
        let telemetry = ctx.telemetry.borrow().clone();
        let rollout =
            self.rollout_value(&telemetry, success, duration_ms, repeat, case_id, error)?;
        ctx.artifacts.insert("rollout".to_string(), rollout);
        Ok(())
    }

    fn stage_metadata(
        &self,
        stage: &StageIR,
        harness: &ResolvedHarness,
    ) -> Result<(Option<String>, Option<String>, Option<u64>, u64)> {
        Ok((
            self.harness_string(harness, &stage.name, "model")?,
            self.harness_string(harness, &stage.name, "variant")?,
            self.harness_u64(harness, &stage.name, "timeout_secs")?,
            self.harness_u64(harness, &stage.name, "retries")?
                .unwrap_or(0),
        ))
    }

    fn stage_kind_label(&self, kind: StageKindIR) -> &'static str {
        match kind {
            StageKindIR::Tool => "tool",
            StageKindIR::Prompt => "prompt",
            StageKindIR::Agent => "agent",
        }
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
                        self.ensure_supported_fields(&loop_decl.name, harness, &["max_iters"])?;
                        let max_iters = self.eval_loop_max_iters(loop_decl, ctx, harness)?;
                        for iteration in 0..max_iters {
                            if let Some(condition) = &loop_decl.while_condition {
                                if !self.eval_bool(condition, ctx)? {
                                    break;
                                }
                            }
                            let started = Instant::now();
                            let body_result =
                                self.execute_nodes(&loop_decl.body, ctx, harness).await;
                            let duration_ms = started.elapsed().as_secs_f64() * 1000.0;
                            let (terminated, error) = match body_result {
                                Ok(()) => match &loop_decl.until {
                                    Some(until) => match self.eval_bool(until, ctx) {
                                        Ok(terminated) => (terminated, None),
                                        Err(error) => {
                                            let error_text = error.to_string();
                                            self.record_loop_iteration_telemetry(
                                                &ctx.telemetry,
                                                LoopIterationTelemetry {
                                                    loop_name: loop_decl.name.clone(),
                                                    iteration: iteration as u64 + 1,
                                                    max_iters,
                                                    duration_ms,
                                                    terminated: false,
                                                    error: Some(error_text.clone()),
                                                },
                                            );
                                            return Err(error);
                                        }
                                    },
                                    None => (false, None),
                                },
                                Err(error) => {
                                    let error_text = error.to_string();
                                    self.record_loop_iteration_telemetry(
                                        &ctx.telemetry,
                                        LoopIterationTelemetry {
                                            loop_name: loop_decl.name.clone(),
                                            iteration: iteration as u64 + 1,
                                            max_iters,
                                            duration_ms,
                                            terminated: false,
                                            error: Some(error_text),
                                        },
                                    );
                                    return Err(error);
                                }
                            };
                            self.record_loop_iteration_telemetry(
                                &ctx.telemetry,
                                LoopIterationTelemetry {
                                    loop_name: loop_decl.name.clone(),
                                    iteration: iteration as u64 + 1,
                                    max_iters,
                                    duration_ms,
                                    terminated,
                                    error,
                                },
                            );
                            if terminated {
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
        let started = Instant::now();
        let (model, variant, timeout_secs, retries) = self.stage_metadata(stage, harness)?;
        if let Some(when) = &stage.when {
            if !self.eval_bool(when, ctx)? {
                self.record_stage_telemetry(
                    &ctx.telemetry,
                    StageTelemetry {
                        stage_name: stage.name.clone(),
                        stage_kind: self.stage_kind_label(stage.stage_kind).to_string(),
                        component: stage.component.clone(),
                        output_artifact: stage.output.clone(),
                        status: "skipped".to_string(),
                        duration_ms: 0.0,
                        input: serde_json::Value::Null,
                        output: None,
                        error: None,
                        model,
                        variant,
                        timeout_secs,
                        retries,
                    },
                );
                return Ok(());
            }
        }

        let stage_input = self.eval_expr(&stage.input, ctx)?;
        let stage_input_json = serde_json::Value::from(stage_input.clone());
        let result = match stage.stage_kind {
            StageKindIR::Prompt => {
                self.execute_prompt_stage(stage, stage_input, harness, &ctx.telemetry)
                    .await
            }
            StageKindIR::Agent => {
                self.execute_agent_stage(stage, stage_input, harness, &ctx.telemetry)
                    .await
            }
            StageKindIR::Tool => {
                self.execute_tool_stage(stage, stage_input, harness, &ctx.telemetry)
            }
        };
        let duration_ms = started.elapsed().as_secs_f64() * 1000.0;
        match result {
            Ok(output) => {
                let output_json = serde_json::Value::from(output.clone());
                self.record_stage_telemetry(
                    &ctx.telemetry,
                    StageTelemetry {
                        stage_name: stage.name.clone(),
                        stage_kind: self.stage_kind_label(stage.stage_kind).to_string(),
                        component: stage.component.clone(),
                        output_artifact: stage.output.clone(),
                        status: "ok".to_string(),
                        duration_ms,
                        input: stage_input_json,
                        output: Some(output_json),
                        error: None,
                        model,
                        variant,
                        timeout_secs,
                        retries,
                    },
                );
                ctx.artifacts.insert(stage.output.clone(), output);
                Ok(())
            }
            Err(error) => {
                self.record_stage_telemetry(
                    &ctx.telemetry,
                    StageTelemetry {
                        stage_name: stage.name.clone(),
                        stage_kind: self.stage_kind_label(stage.stage_kind).to_string(),
                        component: stage.component.clone(),
                        output_artifact: stage.output.clone(),
                        status: "error".to_string(),
                        duration_ms,
                        input: stage_input_json,
                        output: None,
                        error: Some(error.to_string()),
                        model,
                        variant,
                        timeout_secs,
                        retries,
                    },
                );
                Err(error)
            }
        }
    }

    async fn execute_prompt_stage(
        &self,
        stage: &StageIR,
        input: Value,
        harness: &ResolvedHarness,
        telemetry: &Rc<RefCell<RolloutTelemetry>>,
    ) -> Result<Value> {
        self.ensure_supported_fields(
            &stage.name,
            harness,
            &[
                "model",
                "temperature",
                "timeout_secs",
                "retries",
                "system_prompt",
                "variant",
            ],
        )?;
        let prompt = self.find_prompt(&stage.component)?;
        let input = self.validate_value(input, &prompt.input)?;
        let schema = self.output_schema_string(&prompt.output)?;
        let rendered = self.render_prompt(prompt, &input, harness, &stage.name)?;
        let timeout_secs = self.harness_u64(harness, &stage.name, "timeout_secs")?;
        let retries = self
            .harness_u64(harness, &stage.name, "retries")?
            .unwrap_or(0);
        let config = self.prompt_llm_config(prompt, harness, &stage.name)?;
        let variant = self.harness_string(harness, &stage.name, "variant")?;
        let input_json = serde_json::Value::from(input.clone());
        let span_id = tracer().record(TraceEvent::PromptExecution {
            prompt_name: prompt.name.clone(),
            input: Some(input_json.clone()),
            output: None,
            error: None,
        });
        let started = Instant::now();
        let result = self
            .query_structured_with_policy(&rendered, &schema, &config, timeout_secs, retries)
            .await
            .and_then(|output| self.validate_value(output, &prompt.output));
        let duration = started.elapsed();
        let duration_ms = duration.as_secs_f64() * 1000.0;
        let (output_json, error) = match &result {
            Ok(output) => (Some(serde_json::Value::from(output.clone())), None),
            Err(error) => (None, Some(error.to_string())),
        };
        self.record_prompt_call_telemetry(
            telemetry,
            PromptCallTelemetry {
                scope: stage.name.clone(),
                prompt_name: prompt.name.clone(),
                duration_ms,
                input: input_json.clone(),
                output: output_json.clone(),
                error: error.clone(),
                model: config.model.clone(),
                variant,
                timeout_secs,
                retries,
                prompt_hash: self.hash_text(&rendered),
                system_prompt_hash: config
                    .system_prompt
                    .as_deref()
                    .map(|prompt| self.hash_text(prompt)),
            },
        );
        tracer().record_completed(
            &span_id,
            TraceEvent::PromptExecution {
                prompt_name: prompt.name.clone(),
                input: Some(input_json),
                output: output_json,
                error,
            },
            duration,
        );
        result
    }

    async fn execute_agent_stage(
        &self,
        stage: &StageIR,
        input: Value,
        harness: &ResolvedHarness,
        telemetry: &Rc<RefCell<RolloutTelemetry>>,
    ) -> Result<Value> {
        self.ensure_supported_fields(
            &stage.name,
            harness,
            &[
                "model",
                "temperature",
                "timeout_secs",
                "retries",
                "max_turns",
                "system_prompt",
                "variant",
                "tools",
            ],
        )?;
        let agent = self.find_agent(&stage.component)?;
        let input = self.validate_value(input, &agent.input)?;
        let retries = self
            .harness_u64(harness, &stage.name, "retries")?
            .or_else(|| match agent.on_error {
                scaffold_ir::ErrorStrategyIR::Retry { count } => Some(count),
                scaffold_ir::ErrorStrategyIR::Abort => None,
            })
            .unwrap_or(0);
        let timeout_secs = self
            .harness_u64(harness, &stage.name, "timeout_secs")?
            .or(agent.timeout);
        let effective_tools = self.effective_agent_tools(agent, harness, &stage.name)?;
        let max_turns = self
            .harness_u64(harness, &stage.name, "max_turns")?
            .or(agent.max_turns)
            .unwrap_or(if effective_tools.is_empty() { 1 } else { 4 });
        let config = self.agent_llm_config(agent, harness, &stage.name)?;

        let mut last_error = None;
        for _ in 0..=retries {
            match self
                .run_agent_turn_loop(
                    &stage.name,
                    agent,
                    &input,
                    &config,
                    timeout_secs,
                    max_turns,
                    &effective_tools,
                    telemetry,
                )
                .await
            {
                Ok(output) => return Ok(output),
                Err(error) => last_error = Some(error),
            }
        }

        Err(last_error.unwrap_or_else(|| {
            Error::Runtime(format!(
                "agent stage '{}' failed without returning an error",
                stage.name
            ))
        }))
    }

    fn execute_tool_stage(
        &self,
        stage: &StageIR,
        input: Value,
        harness: &ResolvedHarness,
        telemetry: &Rc<RefCell<RolloutTelemetry>>,
    ) -> Result<Value> {
        self.ensure_supported_fields(
            &stage.name,
            harness,
            &["timeout_secs", "retries", "variant"],
        )?;
        let tool = self.find_tool(&stage.component)?;
        let input = self.validate_value(input, &tool.input)?;
        let retries = self
            .harness_u64(harness, &stage.name, "retries")?
            .unwrap_or(0);
        let timeout_secs = self.harness_u64(harness, &stage.name, "timeout_secs")?;
        let variant = self.harness_string(harness, &stage.name, "variant")?;

        let mut last_error = None;
        for _ in 0..=retries {
            match self.execute_tool(
                tool,
                input.clone(),
                variant.as_deref(),
                timeout_secs,
                &stage.name,
                telemetry,
            ) {
                Ok(output) => return Ok(output),
                Err(err) => last_error = Some(err),
            }
        }

        Err(last_error.unwrap_or_else(|| {
            Error::Runtime(format!(
                "tool stage '{}' failed without returning an error",
                stage.name
            ))
        }))
    }

    fn execute_tool(
        &self,
        tool: &ToolIR,
        input: Value,
        variant: Option<&str>,
        timeout_secs: Option<u64>,
        scope: &str,
        telemetry: &Rc<RefCell<RolloutTelemetry>>,
    ) -> Result<Value> {
        let input = self.validate_value(input, &tool.input)?;
        let input_json = serde_json::Value::from(input.clone());
        let span_id = tracer().record(TraceEvent::ToolCall {
            tool_name: tool.name.clone(),
            input: Some(input_json.clone()),
            output: None,
            error: None,
        });
        let started = Instant::now();
        let empty_locals = HashMap::new();
        let result = (|| -> Result<Value> {
            if let Some(spec) = &tool.spec {
                for condition in &spec.preconditions {
                    if !self.eval_tool_bool_expr(
                        condition,
                        &input,
                        &empty_locals,
                        timeout_secs,
                        scope,
                        telemetry,
                    )? {
                        return Err(Error::PreconditionFailed(format!("{:?}", condition)));
                    }
                }
            }

            let implementation = self.select_tool_implementation(tool, variant)?;
            let mut locals = HashMap::new();
            let output = self.execute_tool_impl(
                implementation,
                Some(&tool.output),
                &input,
                &mut locals,
                timeout_secs,
                scope,
                telemetry,
            )?;
            let output = self.validate_value(output, &tool.output)?;

            if let Some(spec) = &tool.spec {
                let mut post_locals = HashMap::new();
                post_locals.insert("output".to_string(), output.clone());
                for condition in &spec.postconditions {
                    if !self.eval_tool_bool_expr(
                        condition,
                        &input,
                        &post_locals,
                        timeout_secs,
                        scope,
                        telemetry,
                    )? {
                        return Err(Error::PostconditionFailed(format!("{:?}", condition)));
                    }
                }
            }

            Ok(output)
        })();
        let duration = started.elapsed();
        let duration_ms = duration.as_secs_f64() * 1000.0;
        let (output_json, error) = match &result {
            Ok(output) => (Some(serde_json::Value::from(output.clone())), None),
            Err(error) => (None, Some(error.to_string())),
        };
        self.record_tool_call_telemetry(
            telemetry,
            ToolCallTelemetry {
                scope: scope.to_string(),
                tool_name: tool.name.clone(),
                duration_ms,
                input: input_json.clone(),
                output: output_json.clone(),
                error: error.clone(),
                variant: variant.map(str::to_string),
                timeout_secs,
            },
        );
        tracer().record_completed(
            &span_id,
            TraceEvent::ToolCall {
                tool_name: tool.name.clone(),
                input: Some(input_json),
                output: output_json,
                error,
            },
            duration,
        );
        result
    }

    fn select_tool_implementation<'b>(
        &self,
        tool: &'b ToolIR,
        variant: Option<&str>,
    ) -> Result<&'b ToolImplIR> {
        if let Some(variant) = variant {
            let selected = variant.rsplit("::").next().unwrap_or(variant);
            return tool
                .variants
                .iter()
                .find(|candidate| candidate.name == selected)
                .map(|candidate| &candidate.implementation)
                .ok_or_else(|| {
                    Error::Runtime(format!(
                        "tool '{}' does not define variant '{}'",
                        tool.name, variant
                    ))
                });
        }

        tool.implementation.as_ref().ok_or_else(|| {
            Error::Runtime(format!(
                "tool '{}' does not define a default implementation",
                tool.name
            ))
        })
    }

    fn execute_tool_impl(
        &self,
        implementation: &ToolImplIR,
        expected: Option<&TypeIR>,
        input: &Value,
        locals: &mut HashMap<String, Value>,
        timeout_secs: Option<u64>,
        scope: &str,
        telemetry: &Rc<RefCell<RolloutTelemetry>>,
    ) -> Result<Value> {
        match implementation {
            ToolImplIR::Expr { expr } => self.eval_tool_expr(
                expr,
                expected,
                input,
                locals,
                timeout_secs,
                scope,
                telemetry,
            ),
            ToolImplIR::Sequence { statements } | ToolImplIR::Parallel { statements } => self
                .execute_tool_block(
                    statements,
                    expected,
                    input,
                    locals,
                    timeout_secs,
                    scope,
                    telemetry,
                ),
        }
    }

    fn execute_tool_block(
        &self,
        statements: &[ToolStatementIR],
        expected: Option<&TypeIR>,
        input: &Value,
        locals: &mut HashMap<String, Value>,
        timeout_secs: Option<u64>,
        scope: &str,
        telemetry: &Rc<RefCell<RolloutTelemetry>>,
    ) -> Result<Value> {
        let binding_names = statements
            .iter()
            .filter_map(|stmt| stmt.binding.as_deref())
            .collect::<BTreeSet<_>>();
        let mut last_value = None;
        let mut last_binding = None;

        for statement in statements {
            let value = self.eval_tool_expr(
                &statement.expr,
                None,
                input,
                locals,
                timeout_secs,
                scope,
                telemetry,
            )?;
            if let Some(binding) = &statement.binding {
                locals.insert(binding.clone(), value.clone());
                last_binding = Some(binding.clone());
            }
            last_value = Some(value);
        }

        if let Some(expected_ty) = expected {
            if let Some(fields) = self.struct_fields_for_type(expected_ty)? {
                if !fields.is_empty()
                    && fields
                        .keys()
                        .all(|field| binding_names.contains(field.as_str()))
                {
                    let mut out = HashMap::new();
                    for field in fields.keys() {
                        if let Some(value) = locals.get(field) {
                            out.insert(field.clone(), value.clone());
                        }
                    }
                    return Ok(Value::Map(out));
                }
            }
        }

        if let Some(binding) = last_binding {
            if let Some(value) = locals.get(&binding) {
                return Ok(value.clone());
            }
        }

        if let Some(value) = last_value {
            return Ok(value);
        }

        self.default_value_for_type(expected)
    }

    fn eval_tool_expr(
        &self,
        expr: &ToolExprIR,
        expected: Option<&TypeIR>,
        input: &Value,
        locals: &HashMap<String, Value>,
        timeout_secs: Option<u64>,
        scope: &str,
        telemetry: &Rc<RefCell<RolloutTelemetry>>,
    ) -> Result<Value> {
        match expr {
            ToolExprIR::Ident { name } => self.lookup_tool_value(name, input, locals),
            ToolExprIR::FieldAccess { base, field } => {
                let base =
                    self.eval_tool_expr(base, None, input, locals, timeout_secs, scope, telemetry)?;
                base.field(field).cloned().ok_or_else(|| {
                    Error::Runtime(format!(
                        "field '{}' not found on {}",
                        field,
                        self.type_label_from_value(&base)
                    ))
                })
            }
            ToolExprIR::ForeignCall {
                module, function, ..
            } => Err(Error::Runtime(format!(
                "task interpreter does not support foreign call '{}::{}'",
                module, function
            ))),
            ToolExprIR::ToolCall { tool, args } => {
                let values = args
                    .iter()
                    .map(|arg| {
                        self.eval_tool_expr(
                            arg,
                            None,
                            input,
                            locals,
                            timeout_secs,
                            scope,
                            telemetry,
                        )
                    })
                    .collect::<Result<Vec<_>>>()?;

                if builtins::BUILTIN_NAMES.contains(&tool.as_str()) {
                    self.eval_builtin_call(tool, &values)
                } else {
                    let callee = self.find_tool(tool)?;
                    let call_input =
                        self.prepare_tool_call_input_from_tool_args(callee, args, &values)?;
                    let child_scope = format!("{}::{}", scope, tool);
                    self.execute_tool(
                        callee,
                        call_input,
                        None,
                        timeout_secs,
                        &child_scope,
                        telemetry,
                    )
                }
            }
            ToolExprIR::Shell { command } => {
                self.execute_shell_tool_expr(command, expected, input, locals, timeout_secs)
            }
            ToolExprIR::Pipe { .. } => Err(Error::Runtime(
                "task interpreter does not yet support tool pipe expressions".to_string(),
            )),
            ToolExprIR::If {
                condition,
                then_branch,
                else_branch,
            } => {
                if self.eval_tool_bool_expr(
                    condition,
                    input,
                    locals,
                    timeout_secs,
                    scope,
                    telemetry,
                )? {
                    let mut branch_locals = locals.clone();
                    self.execute_tool_impl(
                        then_branch,
                        expected,
                        input,
                        &mut branch_locals,
                        timeout_secs,
                        scope,
                        telemetry,
                    )
                } else if let Some(branch) = else_branch {
                    let mut branch_locals = locals.clone();
                    self.execute_tool_impl(
                        branch,
                        expected,
                        input,
                        &mut branch_locals,
                        timeout_secs,
                        scope,
                        telemetry,
                    )
                } else {
                    self.default_value_for_type(expected)
                }
            }
            ToolExprIR::Match { scrutinee, arms } => {
                let value = self.eval_tool_expr(
                    scrutinee,
                    None,
                    input,
                    locals,
                    timeout_secs,
                    scope,
                    telemetry,
                )?;
                for arm in arms {
                    let pattern = self.eval_tool_logic_expr(
                        &arm.pattern,
                        input,
                        locals,
                        timeout_secs,
                        scope,
                        telemetry,
                    )?;
                    if value == pattern {
                        let mut arm_locals = locals.clone();
                        return self.execute_tool_impl(
                            &arm.body,
                            expected,
                            input,
                            &mut arm_locals,
                            timeout_secs,
                            scope,
                            telemetry,
                        );
                    }
                }
                self.default_value_for_type(expected)
            }
            ToolExprIR::For {
                variable,
                iterable,
                body,
            } => {
                let iterable = self.eval_tool_expr(
                    iterable,
                    None,
                    input,
                    locals,
                    timeout_secs,
                    scope,
                    telemetry,
                )?;
                let items = iterable
                    .as_list()
                    .cloned()
                    .ok_or_else(|| Error::TypeError {
                        expected: "list".to_string(),
                        actual: iterable.type_name().to_string(),
                    })?;
                let mut result = self.default_value_for_type(None)?;
                for item in items {
                    let mut body_locals = locals.clone();
                    body_locals.insert(variable.clone(), item);
                    match self.execute_tool_impl(
                        body,
                        None,
                        input,
                        &mut body_locals,
                        timeout_secs,
                        scope,
                        telemetry,
                    ) {
                        Ok(value) => result = value,
                        Err(Error::LoopBreak) => break,
                        Err(Error::LoopContinue) => continue,
                        Err(err) => return Err(err),
                    }
                }
                Ok(result)
            }
            ToolExprIR::While { condition, body } => {
                let mut result = self.default_value_for_type(None)?;
                while self.eval_tool_bool_expr(
                    condition,
                    input,
                    locals,
                    timeout_secs,
                    scope,
                    telemetry,
                )? {
                    let mut body_locals = locals.clone();
                    match self.execute_tool_impl(
                        body,
                        None,
                        input,
                        &mut body_locals,
                        timeout_secs,
                        scope,
                        telemetry,
                    ) {
                        Ok(value) => result = value,
                        Err(Error::LoopBreak) => break,
                        Err(Error::LoopContinue) => continue,
                        Err(err) => return Err(err),
                    }
                }
                Ok(result)
            }
            ToolExprIR::Loop { body } => {
                let mut result = self.default_value_for_type(None)?;
                loop {
                    let mut body_locals = locals.clone();
                    match self.execute_tool_impl(
                        body,
                        None,
                        input,
                        &mut body_locals,
                        timeout_secs,
                        scope,
                        telemetry,
                    ) {
                        Ok(value) => result = value,
                        Err(Error::LoopBreak) => break,
                        Err(Error::LoopContinue) => continue,
                        Err(err) => return Err(err),
                    }
                }
                Ok(result)
            }
            ToolExprIR::Break => Err(Error::LoopBreak),
            ToolExprIR::Continue => Err(Error::LoopContinue),
            ToolExprIR::Literal { value } => Ok(self.literal_to_value(value)),
            ToolExprIR::MapLiteral { entries } => {
                let mut out = HashMap::new();
                for entry in entries {
                    out.insert(
                        entry.key.clone(),
                        self.eval_tool_expr(
                            &entry.value,
                            None,
                            input,
                            locals,
                            timeout_secs,
                            scope,
                            telemetry,
                        )?,
                    );
                }
                Ok(Value::Map(out))
            }
            ToolExprIR::Expr { expr } => {
                self.eval_tool_logic_expr(expr, input, locals, timeout_secs, scope, telemetry)
            }
        }
    }

    fn eval_tool_logic_expr(
        &self,
        expr: &ExprIR,
        input: &Value,
        locals: &HashMap<String, Value>,
        timeout_secs: Option<u64>,
        scope: &str,
        telemetry: &Rc<RefCell<RolloutTelemetry>>,
    ) -> Result<Value> {
        match expr {
            ExprIR::Literal { value } => Ok(self.literal_to_value(value)),
            ExprIR::Ident { name } => self.lookup_tool_value(name, input, locals),
            ExprIR::FieldAccess { base, field } => {
                let base =
                    self.eval_tool_logic_expr(base, input, locals, timeout_secs, scope, telemetry)?;
                base.field(field).cloned().ok_or_else(|| {
                    Error::Runtime(format!(
                        "field '{}' not found on {}",
                        field,
                        self.type_label_from_value(&base)
                    ))
                })
            }
            ExprIR::Binary { left, op, right } => {
                let left =
                    self.eval_tool_logic_expr(left, input, locals, timeout_secs, scope, telemetry)?;
                let right = self.eval_tool_logic_expr(
                    right,
                    input,
                    locals,
                    timeout_secs,
                    scope,
                    telemetry,
                )?;
                self.eval_binary(&left, op, &right)
            }
            ExprIR::Call { function, args } => {
                let values = args
                    .iter()
                    .map(|arg| {
                        self.eval_tool_logic_expr(
                            arg,
                            input,
                            locals,
                            timeout_secs,
                            scope,
                            telemetry,
                        )
                    })
                    .collect::<Result<Vec<_>>>()?;

                if builtins::BUILTIN_NAMES.contains(&function.as_str()) {
                    self.eval_builtin_call(function, &values)
                } else if self.ir.tools.iter().any(|tool| tool.name == *function) {
                    let callee = self.find_tool(function)?;
                    let call_input =
                        self.prepare_tool_call_input_from_expr_args(callee, args, &values)?;
                    let child_scope = format!("{}::{}", scope, function);
                    self.execute_tool(
                        callee,
                        call_input,
                        None,
                        timeout_secs,
                        &child_scope,
                        telemetry,
                    )
                } else {
                    self.eval_call(function, &values)
                }
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
                    .map(|element| {
                        self.eval_tool_logic_expr(
                            element,
                            input,
                            locals,
                            timeout_secs,
                            scope,
                            telemetry,
                        )
                    })
                    .collect::<Result<Vec<_>>>()?,
            )),
            ExprIR::Record { fields } => {
                let mut out = HashMap::new();
                for field in fields {
                    out.insert(
                        field.key.clone(),
                        self.eval_tool_logic_expr(
                            &field.value,
                            input,
                            locals,
                            timeout_secs,
                            scope,
                            telemetry,
                        )?,
                    );
                }
                Ok(Value::Map(out))
            }
        }
    }

    fn eval_tool_bool_expr(
        &self,
        expr: &ExprIR,
        input: &Value,
        locals: &HashMap<String, Value>,
        timeout_secs: Option<u64>,
        scope: &str,
        telemetry: &Rc<RefCell<RolloutTelemetry>>,
    ) -> Result<bool> {
        match self.eval_tool_logic_expr(expr, input, locals, timeout_secs, scope, telemetry)? {
            Value::Bool(value) => Ok(value),
            other => Err(Error::TypeError {
                expected: "bool".to_string(),
                actual: other.type_name().to_string(),
            }),
        }
    }

    fn lookup_tool_value(
        &self,
        name: &str,
        input: &Value,
        locals: &HashMap<String, Value>,
    ) -> Result<Value> {
        if name == "input" {
            return Ok(input.clone());
        }
        if let Some(value) = locals.get(name) {
            return Ok(value.clone());
        }
        if let Some(value) = input.field(name) {
            return Ok(value.clone());
        }
        Err(Error::Runtime(format!("unknown identifier '{}'", name)))
    }

    fn prepare_tool_call_input_from_tool_args(
        &self,
        tool: &ToolIR,
        args: &[ToolExprIR],
        values: &[Value],
    ) -> Result<Value> {
        let named_positions = args
            .iter()
            .enumerate()
            .map(|(idx, arg)| match arg {
                ToolExprIR::Ident { name } => Ok((name.clone(), idx)),
                _ => Err(()),
            })
            .collect::<std::result::Result<HashMap<_, _>, _>>()
            .ok();
        self.prepare_tool_call_input(tool, values, named_positions.as_ref())
    }

    fn prepare_tool_call_input_from_expr_args(
        &self,
        tool: &ToolIR,
        args: &[ExprIR],
        values: &[Value],
    ) -> Result<Value> {
        let named_positions = args
            .iter()
            .enumerate()
            .map(|(idx, arg)| match arg {
                ExprIR::Ident { name } => Ok((name.clone(), idx)),
                _ => Err(()),
            })
            .collect::<std::result::Result<HashMap<_, _>, _>>()
            .ok();
        self.prepare_component_call_input(&tool.input, values, named_positions.as_ref())
    }

    fn prepare_tool_call_input(
        &self,
        tool: &ToolIR,
        values: &[Value],
        named_positions: Option<&HashMap<String, usize>>,
    ) -> Result<Value> {
        self.prepare_component_call_input(&tool.input, values, named_positions)
    }

    fn prepare_component_call_input_from_expr_args(
        &self,
        input_ty: &TypeIR,
        args: &[ExprIR],
        values: &[Value],
    ) -> Result<Value> {
        let named_positions = args
            .iter()
            .enumerate()
            .map(|(idx, arg)| match arg {
                ExprIR::Ident { name } => Ok((name.clone(), idx)),
                _ => Err(()),
            })
            .collect::<std::result::Result<HashMap<_, _>, _>>()
            .ok();
        self.prepare_component_call_input(input_ty, values, named_positions.as_ref())
    }

    fn prepare_component_call_input(
        &self,
        input_ty: &TypeIR,
        values: &[Value],
        named_positions: Option<&HashMap<String, usize>>,
    ) -> Result<Value> {
        if let [single] = values {
            if let Ok(validated) = self.validate_value(single.clone(), input_ty) {
                return Ok(validated);
            }
        }

        if let Some(fields) = self.struct_fields_for_type(input_ty)? {
            let use_named_mapping = named_positions
                .map(|positions| fields.keys().all(|field| positions.contains_key(field)))
                .unwrap_or(false);
            let mut out = HashMap::new();
            for (index, (field, field_ty)) in fields.iter().enumerate() {
                let value = if use_named_mapping {
                    named_positions
                        .and_then(|positions| positions.get(field))
                        .and_then(|position| values.get(*position))
                        .cloned()
                } else {
                    values.get(index).cloned()
                }
                .unwrap_or(self.default_value_for_type(Some(field_ty))?);
                out.insert(field.clone(), value);
            }
            return self.validate_value(Value::Map(out), input_ty);
        }

        match values {
            [] => self.validate_value(Value::Null, input_ty),
            [value] => self.validate_value(value.clone(), input_ty),
            many => self.validate_value(Value::List(many.to_vec()), input_ty),
        }
    }

    fn struct_fields_for_type<'b>(
        &'b self,
        ty: &'b TypeIR,
    ) -> Result<Option<&'b HashMap<String, TypeIR>>> {
        match ty {
            TypeIR::Struct { fields } => Ok(Some(fields)),
            TypeIR::Named { name } => match self.resolve_named_type(name)? {
                TypeIR::Struct { fields } => Ok(Some(fields)),
                other => self.struct_fields_for_type(other),
            },
            _ => Ok(None),
        }
    }

    fn execute_shell_tool_expr(
        &self,
        command: &str,
        expected: Option<&TypeIR>,
        input: &Value,
        locals: &HashMap<String, Value>,
        timeout_secs: Option<u64>,
    ) -> Result<Value> {
        let command = self.interpolate_shell_command(command, input, locals)?;
        let output = match expected {
            Some(TypeIR::Bytes) => Value::Bytes(shell::execute_bytes(&command)?),
            _ => {
                let stdout = if let Some(timeout_secs) = timeout_secs {
                    shell::execute_with_timeout(&command, timeout_secs * 1000)?
                } else {
                    shell::execute(&command)?
                };
                self.parse_shell_output(stdout, expected)?
            }
        };
        Ok(output)
    }

    fn interpolate_shell_command(
        &self,
        command: &str,
        input: &Value,
        locals: &HashMap<String, Value>,
    ) -> Result<String> {
        let mut rendered = String::with_capacity(command.len());
        let chars = command.chars().collect::<Vec<_>>();
        let mut index = 0usize;

        while index < chars.len() {
            if chars[index] == '{' {
                let start = index + 1;
                let mut end = start;
                while end < chars.len() && (chars[end].is_ascii_alphanumeric() || chars[end] == '_')
                {
                    end += 1;
                }
                if end > start && end < chars.len() && chars[end] == '}' {
                    let name = chars[start..end].iter().collect::<String>();
                    let value = self.lookup_tool_value(&name, input, locals)?;
                    rendered.push_str(&value.to_string());
                    index = end + 1;
                    continue;
                }
            }
            rendered.push(chars[index]);
            index += 1;
        }

        Ok(rendered)
    }

    fn parse_shell_output(&self, stdout: String, expected: Option<&TypeIR>) -> Result<Value> {
        let trimmed = stdout.trim().to_string();
        match expected {
            Some(TypeIR::String) => Ok(Value::String(trimmed)),
            Some(TypeIR::Int) => Ok(Value::Int(crate::parse::parse_i64(&trimmed)?)),
            Some(TypeIR::Float) => Ok(Value::Float(crate::parse::parse_f64(&trimmed)?)),
            Some(TypeIR::Bool) => Ok(Value::Bool(crate::parse::parse_bool(&trimmed)?)),
            Some(TypeIR::Any) | None => Ok(Value::String(stdout)),
            _ => {
                let json = serde_json::from_str::<serde_json::Value>(&trimmed)
                    .map_err(|e| Error::ParseError(e.to_string()))?;
                Ok(Value::from(json))
            }
        }
    }

    fn eval_builtin_call(&self, function: &str, args: &[Value]) -> Result<Value> {
        match function {
            "http_get" => {
                let url = self.expect_string(self.expect_arity(function, args, 1)?)?;
                Ok(Value::String(builtins::http_get(url)?))
            }
            "http_get_with_headers" => {
                self.expect_arity(function, args, 2)?;
                Ok(Value::String(builtins::http_get_with_headers(
                    self.expect_string(&args[0])?,
                    self.expect_string(&args[1])?,
                )?))
            }
            "http_post" => {
                self.expect_arity(function, args, 2)?;
                Ok(Value::String(builtins::http_post(
                    self.expect_string(&args[0])?,
                    self.expect_string(&args[1])?,
                )?))
            }
            "json_parse" => {
                let text = self.expect_string(self.expect_arity(function, args, 1)?)?;
                Ok(Value::from(builtins::json_parse(text)?))
            }
            "json_get" => {
                self.expect_arity(function, args, 2)?;
                let json = serde_json::Value::from(args[0].clone());
                Ok(Value::String(builtins::json_get(
                    &json,
                    self.expect_string(&args[1])?,
                )?))
            }
            "json_stringify" => {
                let value = self.expect_arity(function, args, 1)?;
                Ok(Value::String(builtins::json_stringify(
                    &serde_json::Value::from(value.clone()),
                )))
            }
            "regex_extract" => {
                self.expect_arity(function, args, 2)?;
                Ok(Value::String(builtins::regex_extract(
                    self.expect_string(&args[0])?,
                    self.expect_string(&args[1])?,
                )?))
            }
            "regex_extract_all" => {
                self.expect_arity(function, args, 2)?;
                Ok(Value::List(
                    builtins::regex_extract_all(
                        self.expect_string(&args[0])?,
                        self.expect_string(&args[1])?,
                    )?
                    .into_iter()
                    .map(Value::String)
                    .collect(),
                ))
            }
            "regex_matches" => {
                self.expect_arity(function, args, 2)?;
                Ok(Value::Bool(builtins::regex_matches(
                    self.expect_string(&args[0])?,
                    self.expect_string(&args[1])?,
                )?))
            }
            "regex_replace" => {
                self.expect_arity(function, args, 3)?;
                Ok(Value::String(builtins::regex_replace(
                    self.expect_string(&args[0])?,
                    self.expect_string(&args[1])?,
                    self.expect_string(&args[2])?,
                )?))
            }
            "text_split" => {
                self.expect_arity(function, args, 2)?;
                Ok(Value::List(
                    builtins::text_split(
                        self.expect_string(&args[0])?,
                        self.expect_string(&args[1])?,
                    )
                    .into_iter()
                    .map(Value::String)
                    .collect(),
                ))
            }
            "text_join" => {
                self.expect_arity(function, args, 2)?;
                Ok(Value::String(builtins::text_join(
                    &self.expect_string_list(&args[0])?,
                    self.expect_string(&args[1])?,
                )))
            }
            "text_truncate" => {
                self.expect_arity(function, args, 2)?;
                Ok(Value::String(builtins::text_truncate(
                    self.expect_string(&args[0])?,
                    self.expect_i64(&args[1])?,
                )))
            }
            "html_strip" => {
                let text = self.expect_string(self.expect_arity(function, args, 1)?)?;
                Ok(Value::String(builtins::html_strip(text)))
            }
            "text_count" => {
                self.expect_arity(function, args, 2)?;
                Ok(Value::Int(builtins::text_count(
                    self.expect_string(&args[0])?,
                    self.expect_string(&args[1])?,
                )))
            }
            "text_contains_ci" => {
                self.expect_arity(function, args, 2)?;
                Ok(Value::Bool(builtins::text_contains_ci(
                    self.expect_string(&args[0])?,
                    self.expect_string(&args[1])?,
                )))
            }
            "url_encode" => {
                let text = self.expect_string(self.expect_arity(function, args, 1)?)?;
                Ok(Value::String(builtins::url_encode(text)))
            }
            "url_decode" => {
                let text = self.expect_string(self.expect_arity(function, args, 1)?)?;
                Ok(Value::String(builtins::url_decode(text)))
            }
            other => Err(Error::Runtime(format!(
                "task interpreter does not support builtin '{}'",
                other
            ))),
        }
    }

    fn expect_string_list(&self, value: &Value) -> Result<Vec<String>> {
        let values = value.as_list().ok_or_else(|| Error::TypeError {
            expected: "list".to_string(),
            actual: value.type_name().to_string(),
        })?;
        values
            .iter()
            .map(|value| self.expect_string(value).map(str::to_string))
            .collect()
    }

    fn expect_i64(&self, value: &Value) -> Result<i64> {
        value.as_int().ok_or_else(|| Error::TypeError {
            expected: "int".to_string(),
            actual: value.type_name().to_string(),
        })
    }

    fn default_value_for_type(&self, ty: Option<&TypeIR>) -> Result<Value> {
        match ty {
            None => Ok(Value::Null),
            Some(TypeIR::Any) => Ok(Value::Null),
            Some(TypeIR::Bool) => Ok(Value::Bool(false)),
            Some(TypeIR::Int) => Ok(Value::Int(0)),
            Some(TypeIR::Float) => Ok(Value::Float(0.0)),
            Some(TypeIR::String) => Ok(Value::String(String::new())),
            Some(TypeIR::Bytes) => Ok(Value::Bytes(Vec::new())),
            Some(TypeIR::Option { .. }) => Ok(Value::Null),
            Some(TypeIR::List { .. }) => Ok(Value::List(Vec::new())),
            Some(TypeIR::Map { .. }) => Ok(Value::Map(HashMap::new())),
            Some(TypeIR::Struct { fields }) => {
                let mut out = HashMap::new();
                for (name, field_ty) in fields {
                    out.insert(name.clone(), self.default_value_for_type(Some(field_ty))?);
                }
                Ok(Value::Map(out))
            }
            Some(TypeIR::Named { name }) => {
                let definition = self.resolve_named_type(name)?;
                let value = self.default_value_for_type(Some(definition))?;
                match value {
                    Value::Map(fields) if matches!(definition, TypeIR::Struct { .. }) => {
                        Ok(Value::Struct {
                            type_name: name.clone(),
                            fields,
                        })
                    }
                    other => Ok(other),
                }
            }
            Some(TypeIR::Result { ok, .. }) => Ok(Value::Result(Box::new(ResultValue::Ok(
                self.default_value_for_type(Some(ok))?,
            )))),
        }
    }

    fn literal_to_value(&self, literal: &LiteralIR) -> Value {
        match literal {
            LiteralIR::Int { value } => Value::Int(*value),
            LiteralIR::Float { value } => Value::Float(*value),
            LiteralIR::String { value } => Value::String(value.clone()),
            LiteralIR::Bool { value } => Value::Bool(*value),
            LiteralIR::Null => Value::Null,
        }
    }

    fn render_prompt(
        &self,
        prompt: &PromptIR,
        input: &Value,
        harness: &ResolvedHarness,
        target: &str,
    ) -> Result<String> {
        let template = self.effective_prompt_template(harness, target, prompt)?;
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

    fn prompt_llm_config(
        &self,
        prompt: &PromptIR,
        harness: &ResolvedHarness,
        target: &str,
    ) -> Result<LlmConfig> {
        let mut config = LlmConfig::new();
        if let Some(model) = self.harness_string(harness, target, "model")? {
            config = config.with_model(model);
        }
        if let Some(temperature) = self.harness_float(harness, target, "temperature")? {
            config = config.with_temperature(temperature as f32);
        }
        if let Some(system_prompt) =
            self.effective_system_prompt(harness, target, prompt.system.as_ref(), false)?
        {
            config = config.with_system_prompt(system_prompt);
        }
        Ok(config)
    }

    fn agent_llm_config(
        &self,
        agent: &AgentIR,
        harness: &ResolvedHarness,
        target: &str,
    ) -> Result<LlmConfig> {
        let mut config = LlmConfig::new();
        if let Some(model) = self
            .harness_string(harness, target, "model")?
            .or_else(|| agent.model.clone())
        {
            config = config.with_model(model);
        }
        if let Some(temperature) = self.harness_float(harness, target, "temperature")? {
            config = config.with_temperature(temperature as f32);
        }
        if let Some(system_prompt) =
            self.effective_system_prompt(harness, target, Some(&agent.system), true)?
        {
            config = config.with_system_prompt(system_prompt);
        }
        Ok(config)
    }

    async fn execute_prompt_component(
        &self,
        prompt: &PromptIR,
        input: Value,
        scope: &str,
        telemetry: &Rc<RefCell<RolloutTelemetry>>,
    ) -> Result<Value> {
        let harness = ResolvedHarness::default();
        let input = self.validate_value(input, &prompt.input)?;
        let rendered = self.render_prompt(prompt, &input, &harness, &prompt.name)?;
        let schema = self.output_schema_string(&prompt.output)?;
        let config = self.prompt_llm_config(prompt, &harness, &prompt.name)?;
        let input_json = serde_json::Value::from(input.clone());
        let span_id = tracer().record(TraceEvent::PromptExecution {
            prompt_name: prompt.name.clone(),
            input: Some(input_json.clone()),
            output: None,
            error: None,
        });
        let started = Instant::now();
        let result = self
            .query_structured_with_policy(&rendered, &schema, &config, None, 0)
            .await
            .and_then(|output| self.validate_value(output, &prompt.output));
        let duration = started.elapsed();
        let duration_ms = duration.as_secs_f64() * 1000.0;
        let (output_json, error) = match &result {
            Ok(output) => (Some(serde_json::Value::from(output.clone())), None),
            Err(error) => (None, Some(error.to_string())),
        };
        self.record_prompt_call_telemetry(
            telemetry,
            PromptCallTelemetry {
                scope: scope.to_string(),
                prompt_name: prompt.name.clone(),
                duration_ms,
                input: input_json.clone(),
                output: output_json.clone(),
                error: error.clone(),
                model: config.model.clone(),
                variant: None,
                timeout_secs: None,
                retries: 0,
                prompt_hash: self.hash_text(&rendered),
                system_prompt_hash: config
                    .system_prompt
                    .as_deref()
                    .map(|value| self.hash_text(value)),
            },
        );
        tracer().record_completed(
            &span_id,
            TraceEvent::PromptExecution {
                prompt_name: prompt.name.clone(),
                input: Some(input_json),
                output: output_json,
                error,
            },
            duration,
        );
        result
    }

    async fn execute_agent_component(
        &self,
        agent: &AgentIR,
        input: Value,
        scope: &str,
        telemetry: &Rc<RefCell<RolloutTelemetry>>,
    ) -> Result<Value> {
        let harness = ResolvedHarness::default();
        let input = self.validate_value(input, &agent.input)?;
        let retries = match agent.on_error {
            scaffold_ir::ErrorStrategyIR::Retry { count } => count,
            scaffold_ir::ErrorStrategyIR::Abort => 0,
        };
        let timeout_secs = agent.timeout;
        let effective_tools = agent.tools.clone();
        let max_turns = agent
            .max_turns
            .unwrap_or(if effective_tools.is_empty() { 1 } else { 4 });
        let config = self.agent_llm_config(agent, &harness, &agent.name)?;

        let mut last_error = None;
        for _ in 0..=retries {
            match self
                .run_agent_turn_loop(
                    scope,
                    agent,
                    &input,
                    &config,
                    timeout_secs,
                    max_turns,
                    &effective_tools,
                    telemetry,
                )
                .await
            {
                Ok(output) => return Ok(output),
                Err(error) => last_error = Some(error),
            }
        }

        Err(last_error.unwrap_or_else(|| {
            Error::Runtime(format!(
                "agent '{}' failed without returning an error",
                agent.name
            ))
        }))
    }

    async fn query_structured_with_policy(
        &self,
        prompt: &str,
        schema: &str,
        config: &LlmConfig,
        timeout_secs: Option<u64>,
        retries: u64,
    ) -> Result<Value> {
        let mut last_error = None;
        for _ in 0..=retries {
            let model = config
                .model
                .clone()
                .unwrap_or_else(|| "default".to_string());
            let span_id = tracer().record(TraceEvent::LlmCall {
                model: model.clone(),
                prompt: Some(prompt.to_string()),
                response: None,
                input_tokens: None,
                output_tokens: None,
                error: None,
            });
            let started = Instant::now();
            let future = query_structured_with_config(prompt, schema, config);
            let outcome = if let Some(timeout_secs) = timeout_secs {
                match timeout(Duration::from_secs(timeout_secs), future).await {
                    Ok(result) => result,
                    Err(_) => Err(Error::Runtime(format!(
                        "LLM call timed out after {}s",
                        timeout_secs
                    ))),
                }
            } else {
                future.await
            };

            match outcome {
                Ok(value) => {
                    tracer().record_completed(
                        &span_id,
                        TraceEvent::LlmCall {
                            model,
                            prompt: Some(prompt.to_string()),
                            response: Some(serde_json::Value::from(value.clone()).to_string()),
                            input_tokens: None,
                            output_tokens: None,
                            error: None,
                        },
                        started.elapsed(),
                    );
                    return Ok(value);
                }
                Err(error) => {
                    tracer().record_completed(
                        &span_id,
                        TraceEvent::LlmCall {
                            model,
                            prompt: Some(prompt.to_string()),
                            response: None,
                            input_tokens: None,
                            output_tokens: None,
                            error: Some(error.to_string()),
                        },
                        started.elapsed(),
                    );
                    last_error = Some(error);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            Error::Runtime("LLM call failed without returning an error".to_string())
        }))
    }

    async fn run_agent_turn_loop(
        &self,
        stage_name: &str,
        agent: &AgentIR,
        input: &Value,
        config: &LlmConfig,
        timeout_secs: Option<u64>,
        max_turns: u64,
        effective_tools: &[String],
        telemetry: &Rc<RefCell<RolloutTelemetry>>,
    ) -> Result<Value> {
        if max_turns == 0 {
            return Err(Error::Runtime(format!(
                "agent stage '{}' has max_turns = 0",
                stage_name
            )));
        }

        if effective_tools.is_empty() {
            let input_json = serde_json::to_string_pretty(&serde_json::Value::from(input.clone()))
                .map_err(|e| Error::SerializationError(e.to_string()))?;
            let user_prompt = format!("Input:\n{}", input_json);
            let schema = self.output_schema_string(&agent.output)?;
            let started = Instant::now();
            let result = self
                .query_structured_with_policy(&user_prompt, &schema, config, timeout_secs, 0)
                .await
                .and_then(|output| self.validate_value(output, &agent.output));
            let duration = started.elapsed();
            let duration_ms = duration.as_secs_f64() * 1000.0;
            let (completed, error) = match &result {
                Ok(_) => (true, None),
                Err(error) => (false, Some(error.to_string())),
            };
            self.record_agent_turn_telemetry(
                telemetry,
                AgentTurnTelemetry {
                    scope: stage_name.to_string(),
                    agent_name: agent.name.clone(),
                    turn_number: 1,
                    duration_ms,
                    action: "final".to_string(),
                    tool: None,
                    completed,
                    error: error.clone(),
                    model: config.model.clone(),
                },
            );
            let span_id = tracer().record(TraceEvent::AgentTurn {
                agent_name: agent.name.clone(),
                turn_number: 1,
                tool_calls: None,
                completed: None,
            });
            tracer().record_completed(
                &span_id,
                TraceEvent::AgentTurn {
                    agent_name: agent.name.clone(),
                    turn_number: 1,
                    tool_calls: None,
                    completed: Some(completed),
                },
                duration,
            );
            return result;
        }

        let input_json = serde_json::to_string_pretty(&serde_json::Value::from(input.clone()))
            .map_err(|e| Error::SerializationError(e.to_string()))?;
        let mut history = Vec::new();
        let action_schema = self.agent_action_schema(&agent.output)?;

        for turn in 0..max_turns {
            let tool_specs = effective_tools
                .iter()
                .map(|name| {
                    let tool = self.find_tool(name)?;
                    Ok(json!({
                        "name": tool.name,
                        "input_schema": type_to_json_schema(&tool.input),
                    }))
                })
                .collect::<Result<Vec<_>>>()?;
            let history_json = serde_json::to_string_pretty(&history)
                .map_err(|e| Error::SerializationError(e.to_string()))?;
            let user_prompt = format!(
                concat!(
                    "You are executing agent stage '{stage}'.\n",
                    "Turn {turn} of {max_turns}.\n\n",
                    "Return JSON with action='tool' to invoke one allowed tool, or action='final' to finish.\n",
                    "When action='tool', set 'tool' and 'tool_input'.\n",
                    "When action='final', set 'output' to match the final output schema.\n\n",
                    "Original input:\n{input}\n\n",
                    "Available tools:\n{tools}\n\n",
                    "History:\n{history}"
                ),
                stage = stage_name,
                turn = turn + 1,
                max_turns = max_turns,
                input = input_json,
                tools = serde_json::to_string_pretty(&tool_specs)
                    .map_err(|e| Error::SerializationError(e.to_string()))?,
                history = history_json,
            );
            let started = Instant::now();
            let action = self
                .query_structured_with_policy(&user_prompt, &action_schema, config, timeout_secs, 0)
                .await?;
            let turn_duration = started.elapsed();
            let turn_duration_ms = turn_duration.as_secs_f64() * 1000.0;
            let action_fields = match action {
                Value::Map(fields) | Value::Struct { fields, .. } => fields,
                other => {
                    self.record_agent_turn_telemetry(
                        telemetry,
                        AgentTurnTelemetry {
                            scope: stage_name.to_string(),
                            agent_name: agent.name.clone(),
                            turn_number: turn + 1,
                            duration_ms: turn_duration_ms,
                            action: "invalid".to_string(),
                            tool: None,
                            completed: false,
                            error: Some(format!(
                                "expected object action payload, found {}",
                                other.type_name()
                            )),
                            model: config.model.clone(),
                        },
                    );
                    return Err(Error::TypeError {
                        expected: "object".to_string(),
                        actual: other.type_name().to_string(),
                    });
                }
            };
            let action_name = action_fields
                .get("action")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    Error::Runtime(format!(
                        "agent stage '{}' produced an action without an 'action' field",
                        stage_name
                    ))
                })?;

            match action_name {
                "final" => {
                    let output = action_fields.get("output").cloned().ok_or_else(|| {
                        Error::Runtime(format!(
                            "agent stage '{}' returned action='final' without an 'output' field",
                            stage_name
                        ))
                    })?;
                    let result = self.validate_value(output, &agent.output);
                    let completed = result.is_ok();
                    self.record_agent_turn_telemetry(
                        telemetry,
                        AgentTurnTelemetry {
                            scope: stage_name.to_string(),
                            agent_name: agent.name.clone(),
                            turn_number: turn + 1,
                            duration_ms: turn_duration_ms,
                            action: "final".to_string(),
                            tool: None,
                            completed,
                            error: result.as_ref().err().map(ToString::to_string),
                            model: config.model.clone(),
                        },
                    );
                    let span_id = tracer().record(TraceEvent::AgentTurn {
                        agent_name: agent.name.clone(),
                        turn_number: turn + 1,
                        tool_calls: None,
                        completed: None,
                    });
                    tracer().record_completed(
                        &span_id,
                        TraceEvent::AgentTurn {
                            agent_name: agent.name.clone(),
                            turn_number: turn + 1,
                            tool_calls: None,
                            completed: Some(completed),
                        },
                        turn_duration,
                    );
                    return result;
                }
                "tool" => {
                    let tool_name = action_fields
                        .get("tool")
                        .and_then(Value::as_str)
                        .ok_or_else(|| {
                            Error::Runtime(format!(
                                "agent stage '{}' returned action='tool' without a 'tool' field",
                                stage_name
                            ))
                        })?;
                    if !effective_tools
                        .iter()
                        .any(|candidate| candidate == tool_name)
                    {
                        return Err(Error::Runtime(format!(
                            "agent stage '{}' attempted to call undeclared tool '{}'",
                            stage_name, tool_name
                        )));
                    }
                    let tool = self.find_tool(tool_name)?;
                    let tool_input = match action_fields.get("tool_input").cloned() {
                        Some(value) => self.validate_value(value, &tool.input)?,
                        None => self.default_value_for_type(Some(&tool.input))?,
                    };
                    let tool_scope = format!("{}.turn{}", stage_name, turn + 1);
                    match self.execute_tool(
                        tool,
                        tool_input.clone(),
                        None,
                        timeout_secs,
                        &tool_scope,
                        telemetry,
                    ) {
                        Ok(output) => {
                            history.push(json!({
                                "turn": turn + 1,
                                "action": "tool",
                                "tool": tool_name,
                                "tool_input": serde_json::Value::from(tool_input),
                                "tool_output": serde_json::Value::from(output),
                            }));
                            self.record_agent_turn_telemetry(
                                telemetry,
                                AgentTurnTelemetry {
                                    scope: stage_name.to_string(),
                                    agent_name: agent.name.clone(),
                                    turn_number: turn + 1,
                                    duration_ms: turn_duration_ms,
                                    action: "tool".to_string(),
                                    tool: Some(tool_name.to_string()),
                                    completed: false,
                                    error: None,
                                    model: config.model.clone(),
                                },
                            );
                            let span_id = tracer().record(TraceEvent::AgentTurn {
                                agent_name: agent.name.clone(),
                                turn_number: turn + 1,
                                tool_calls: Some(vec![tool_name.to_string()]),
                                completed: None,
                            });
                            tracer().record_completed(
                                &span_id,
                                TraceEvent::AgentTurn {
                                    agent_name: agent.name.clone(),
                                    turn_number: turn + 1,
                                    tool_calls: Some(vec![tool_name.to_string()]),
                                    completed: Some(false),
                                },
                                turn_duration,
                            );
                        }
                        Err(error) => {
                            let error_text = error.to_string();
                            history.push(json!({
                                "turn": turn + 1,
                                "action": "tool",
                                "tool": tool_name,
                                "tool_input": serde_json::Value::from(tool_input),
                                "tool_error": error_text,
                            }));
                            self.record_agent_turn_telemetry(
                                telemetry,
                                AgentTurnTelemetry {
                                    scope: stage_name.to_string(),
                                    agent_name: agent.name.clone(),
                                    turn_number: turn + 1,
                                    duration_ms: turn_duration_ms,
                                    action: "tool".to_string(),
                                    tool: Some(tool_name.to_string()),
                                    completed: false,
                                    error: Some(error_text),
                                    model: config.model.clone(),
                                },
                            );
                            let span_id = tracer().record(TraceEvent::AgentTurn {
                                agent_name: agent.name.clone(),
                                turn_number: turn + 1,
                                tool_calls: Some(vec![tool_name.to_string()]),
                                completed: None,
                            });
                            tracer().record_completed(
                                &span_id,
                                TraceEvent::AgentTurn {
                                    agent_name: agent.name.clone(),
                                    turn_number: turn + 1,
                                    tool_calls: Some(vec![tool_name.to_string()]),
                                    completed: Some(false),
                                },
                                turn_duration,
                            );
                        }
                    }
                }
                other => {
                    self.record_agent_turn_telemetry(
                        telemetry,
                        AgentTurnTelemetry {
                            scope: stage_name.to_string(),
                            agent_name: agent.name.clone(),
                            turn_number: turn + 1,
                            duration_ms: turn_duration_ms,
                            action: other.to_string(),
                            tool: None,
                            completed: false,
                            error: Some(format!("unsupported action '{}'", other)),
                            model: config.model.clone(),
                        },
                    );
                    return Err(Error::Runtime(format!(
                        "agent stage '{}' returned unsupported action '{}'",
                        stage_name, other
                    )));
                }
            }
        }

        Err(Error::Runtime(format!(
            "agent stage '{}' exhausted {} turn(s) without producing final output",
            stage_name, max_turns
        )))
    }

    fn agent_action_schema(&self, output: &TypeIR) -> Result<String> {
        serde_json::to_string_pretty(&json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["tool", "final"]
                },
                "tool": { "type": "string" },
                "tool_input": {},
                "output": type_to_json_schema(output),
            },
            "required": ["action"],
            "additionalProperties": false
        }))
        .map_err(|e| Error::SerializationError(e.to_string()))
    }

    fn effective_prompt_template(
        &self,
        harness: &ResolvedHarness,
        target: &str,
        prompt: &PromptIR,
    ) -> Result<String> {
        if let Some(variant) = self.harness_string(harness, target, "variant")? {
            return self.resolve_text_surface(&variant, true);
        }
        self.read_string_or_file(&prompt.template)
    }

    fn effective_system_prompt(
        &self,
        harness: &ResolvedHarness,
        target: &str,
        fallback: Option<&StringOrFileIR>,
        allow_variant_fallback: bool,
    ) -> Result<Option<String>> {
        if let Some(system_prompt) = self.harness_string(harness, target, "system_prompt")? {
            return Ok(Some(self.resolve_text_surface(&system_prompt, false)?));
        }
        if allow_variant_fallback {
            if let Some(variant) = self.harness_string(harness, target, "variant")? {
                return Ok(Some(self.resolve_text_surface(&variant, true)?));
            }
        }
        fallback
            .map(|value| self.read_string_or_file(value))
            .transpose()
    }

    fn effective_agent_tools(
        &self,
        agent: &AgentIR,
        harness: &ResolvedHarness,
        target: &str,
    ) -> Result<Vec<String>> {
        match harness.field_value(target, "tools") {
            Some(value) => self.harness_string_list(value),
            None => Ok(agent.tools.clone()),
        }
    }

    fn harness_string_list(&self, value: &Value) -> Result<Vec<String>> {
        self.expect_string_list(value)
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

    fn harness_u64(
        &self,
        harness: &ResolvedHarness,
        target: &str,
        field: &str,
    ) -> Result<Option<u64>> {
        match harness.field_value(target, field) {
            Some(Value::Int(value)) if *value >= 0 => Ok(Some(*value as u64)),
            Some(value) => Err(Error::TypeError {
                expected: "non-negative int".to_string(),
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

    fn resolve_harness_with_assignments(
        &self,
        task_name: &str,
        harness_name: &str,
        assignments: &HashMap<String, Value>,
    ) -> Result<ResolvedHarness> {
        let mut harness = self.resolve_harness(task_name, Some(harness_name))?;
        for (path, value) in assignments {
            let mut segments = path.split('.');
            let target = segments.next().ok_or_else(|| {
                Error::Runtime(format!("invalid harness assignment path '{}'", path))
            })?;
            let field = segments.next().ok_or_else(|| {
                Error::Runtime(format!("invalid harness assignment path '{}'", path))
            })?;
            if segments.next().is_some() {
                return Err(Error::Runtime(format!(
                    "invalid harness assignment path '{}'",
                    path
                )));
            }
            harness
                .bindings
                .entry(target.to_string())
                .or_default()
                .insert(field.to_string(), value.clone());
        }
        Ok(harness)
    }

    async fn optimize(
        &self,
        objective_name: &str,
        max_candidates: usize,
    ) -> Result<ObjectiveOptimizationReport> {
        if max_candidates == 0 {
            return Err(Error::Runtime(
                "optimize requires max_candidates >= 1".to_string(),
            ));
        }

        let objective = self.find_objective(objective_name)?;
        let task = self.find_task(&objective.task)?;
        let harness = self.find_harness(&objective.harness)?;
        let dataset = self.load_dataset_cases(objective, task)?;
        if dataset.is_empty() {
            return Err(Error::Runtime(format!(
                "objective '{}' has an empty dataset",
                objective.name
            )));
        }

        let (assignments, truncated) =
            self.enumerate_candidate_assignments(task, harness, max_candidates)?;
        let evaluated_candidates = assignments.len();
        let (train_cases, val_cases, test_cases) = self.partition_dataset(&dataset, objective);

        let mut best: Option<CandidateEvaluation> = None;
        for assignment in assignments {
            let resolved = self.resolve_harness_with_assignments(
                &objective.task,
                &objective.harness,
                &assignment,
            )?;
            let candidate = self
                .evaluate_candidate(
                    objective,
                    task,
                    &resolved,
                    &assignment,
                    &train_cases,
                    &val_cases,
                    &test_cases,
                )
                .await?;
            if best
                .as_ref()
                .map(|current| self.candidate_beats(&candidate, current))
                .unwrap_or(true)
            {
                best = Some(candidate);
            }
        }

        let best = best.ok_or_else(|| {
            Error::Runtime(format!(
                "objective '{}' did not yield any candidate evaluations",
                objective.name
            ))
        })?;

        Ok(ObjectiveOptimizationReport {
            objective: objective.name.clone(),
            task: objective.task.clone(),
            harness: objective.harness.clone(),
            evaluated_candidates,
            truncated,
            best: CandidateOptimizationReport {
                assignments: best.assignments,
                train: best.train,
                val: best.val,
                test: best.test,
            },
        })
    }

    fn load_dataset_cases(
        &self,
        objective: &ObjectiveIR,
        task: &TaskIR,
    ) -> Result<Vec<DatasetCase>> {
        match &objective.dataset {
            DatasetSpecIR::Inline { cases } => cases
                .iter()
                .map(|case| self.inline_dataset_case(task, case))
                .collect(),
            DatasetSpecIR::File { path } => self.file_dataset_cases(task, path),
        }
    }

    fn inline_dataset_case(
        &self,
        task: &TaskIR,
        case: &InlineDatasetCaseIR,
    ) -> Result<DatasetCase> {
        let input = self.validate_value(self.eval_static_expr(&case.input)?, &task.input)?;
        let expected = match &case.expected {
            Some(expr) => self.validate_value(self.eval_static_expr(expr)?, &task.output)?,
            None => self.default_value_for_type(Some(&task.output))?,
        };
        Ok(DatasetCase {
            id: case.id.clone(),
            input,
            expected,
        })
    }

    fn file_dataset_cases(&self, task: &TaskIR, path: &str) -> Result<Vec<DatasetCase>> {
        let path = Path::new(path);
        let resolved = if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.base_dir.join(path)
        };
        let content = std::fs::read_to_string(&resolved).map_err(|e| {
            Error::Runtime(format!(
                "failed to read dataset file {}: {}",
                resolved.display(),
                e
            ))
        })?;

        let trimmed = content.trim();
        if trimmed.is_empty() {
            return Ok(Vec::new());
        }

        if trimmed.starts_with('[') {
            let value: serde_json::Value = serde_json::from_str(trimmed)
                .map_err(|e| Error::ParseError(format!("failed to parse dataset JSON: {}", e)))?;
            let items = value.as_array().ok_or_else(|| {
                Error::ParseError("dataset JSON must be an array of cases".to_string())
            })?;
            return items
                .iter()
                .map(|item| self.json_dataset_case(task, item))
                .collect();
        }

        content
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| {
                let value: serde_json::Value = serde_json::from_str(line).map_err(|e| {
                    Error::ParseError(format!("failed to parse dataset JSONL line: {}", e))
                })?;
                self.json_dataset_case(task, &value)
            })
            .collect()
    }

    fn json_dataset_case(&self, task: &TaskIR, value: &serde_json::Value) -> Result<DatasetCase> {
        let object = value
            .as_object()
            .ok_or_else(|| Error::ParseError("dataset case must be a JSON object".to_string()))?;
        let input = object.get("input").cloned().ok_or_else(|| {
            Error::ParseError("dataset case is missing required 'input' field".to_string())
        })?;
        let expected = object
            .get("expected")
            .cloned()
            .map(Value::from)
            .map(|value| self.validate_value(value, &task.output))
            .transpose()?
            .unwrap_or(self.default_value_for_type(Some(&task.output))?);
        Ok(DatasetCase {
            id: object
                .get("id")
                .and_then(serde_json::Value::as_str)
                .map(str::to_string),
            input: self.validate_value(Value::from(input), &task.input)?,
            expected,
        })
    }

    fn partition_dataset(
        &self,
        dataset: &[DatasetCase],
        objective: &ObjectiveIR,
    ) -> (Vec<DatasetCase>, Vec<DatasetCase>, Vec<DatasetCase>) {
        if let Some(split) = &objective.split {
            let total = dataset.len();
            let train_count = ((total as f64) * split.train).round() as usize;
            let val_count = ((total as f64) * split.val).round() as usize;
            let train_end = train_count.min(total);
            let val_end = (train_end + val_count).min(total);
            (
                dataset[..train_end].to_vec(),
                dataset[train_end..val_end].to_vec(),
                dataset[val_end..].to_vec(),
            )
        } else {
            (dataset.to_vec(), Vec::new(), Vec::new())
        }
    }

    fn enumerate_candidate_assignments(
        &self,
        task: &TaskIR,
        harness: &HarnessIR,
        max_candidates: usize,
    ) -> Result<(Vec<HashMap<String, Value>>, bool)> {
        if harness.tunables.is_empty() {
            return Ok((vec![HashMap::new()], false));
        }

        let mut candidates = vec![HashMap::new()];
        let mut truncated = false;

        for tunable in &harness.tunables {
            let path = tunable.path.segments.join(".");
            let options = self.expand_tunable_domain(task, harness, tunable)?;
            let mut next = Vec::new();
            'outer: for candidate in &candidates {
                for option in &options {
                    let mut updated = candidate.clone();
                    updated.insert(path.clone(), option.clone());
                    next.push(updated);
                    if next.len() >= max_candidates {
                        if candidate != candidates.last().unwrap()
                            || option != options.last().unwrap()
                        {
                            truncated = true;
                        }
                        break 'outer;
                    }
                }
            }
            candidates = next;
            if candidates.is_empty() {
                break;
            }
        }

        if candidates.is_empty() {
            candidates.push(HashMap::new());
        }

        Ok((candidates, truncated))
    }

    fn expand_tunable_domain(
        &self,
        task: &TaskIR,
        harness: &HarnessIR,
        tunable: &TunableIR,
    ) -> Result<Vec<Value>> {
        let target = tunable
            .path
            .segments
            .first()
            .ok_or_else(|| Error::Runtime("empty tunable path".to_string()))?;
        let field = tunable
            .path
            .segments
            .get(1)
            .ok_or_else(|| Error::Runtime("incomplete tunable path".to_string()))?;

        match (&tunable.operator, &tunable.domain) {
            (TuneOperatorIR::In, FiniteDomainIR::List { values }) => values
                .iter()
                .map(|value| self.eval_static_expr(value))
                .collect(),
            (TuneOperatorIR::SubsetOf, FiniteDomainIR::List { values }) => {
                let items = values
                    .iter()
                    .map(|value| self.eval_static_expr(value))
                    .collect::<Result<Vec<_>>>()?;
                Ok(self
                    .power_set(&items)
                    .into_iter()
                    .map(Value::List)
                    .collect::<Vec<_>>())
            }
            (TuneOperatorIR::In, FiniteDomainIR::Variants { name }) => {
                self.expand_variants_domain(task, harness, target, field, name)
            }
            (TuneOperatorIR::SubsetOf, FiniteDomainIR::Variants { .. }) => Err(Error::Runtime(
                "subset_of variants(...) is not supported".to_string(),
            )),
        }
    }

    fn expand_variants_domain(
        &self,
        task: &TaskIR,
        harness: &HarnessIR,
        target: &str,
        field: &str,
        group: &str,
    ) -> Result<Vec<Value>> {
        let _ = harness;
        match self.find_task_target(task, target)? {
            TaskTargetKindRuntime::Stage {
                kind: StageKindIR::Tool,
                component,
            } if field == "variant" => {
                let tool = self.find_tool(component)?;
                let values = tool
                    .variants
                    .iter()
                    .map(|variant| Value::String(format!("{}::{}", group, variant.name)))
                    .collect::<Vec<_>>();
                if values.is_empty() {
                    return Err(Error::Runtime(format!(
                        "tool '{}' has no named variants to tune",
                        tool.name
                    )));
                }
                Ok(values)
            }
            TaskTargetKindRuntime::Stage { .. } => {
                let names = self.variant_group_names(group)?;
                if names.is_empty() {
                    return Err(Error::Runtime(format!(
                        "variants('{}') did not resolve any text variants under {}",
                        group,
                        self.base_dir.display()
                    )));
                }
                Ok(names
                    .into_iter()
                    .map(|name| Value::String(format!("{}::{}", group, name)))
                    .collect())
            }
            TaskTargetKindRuntime::Loop if field == "max_iters" => Err(Error::Runtime(format!(
                "variants('{}') is not valid for loop field '{}'",
                group, field
            ))),
            TaskTargetKindRuntime::Loop => Err(Error::Runtime(format!(
                "unknown variant usage for loop target '{}.{}'",
                target, field
            ))),
        }
    }

    fn variant_group_names(&self, group: &str) -> Result<Vec<String>> {
        let mut names = BTreeSet::new();
        for dir in [
            self.base_dir.join("variants").join(group),
            self.base_dir.join("prompts").join(group),
        ] {
            if !dir.is_dir() {
                continue;
            }
            for entry in std::fs::read_dir(&dir).map_err(|e| {
                Error::Runtime(format!(
                    "failed to read variant dir {}: {}",
                    dir.display(),
                    e
                ))
            })? {
                let entry = entry
                    .map_err(|e| Error::Runtime(format!("failed to read variant entry: {}", e)))?;
                let path = entry.path();
                if !path.is_file() {
                    continue;
                }
                let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) else {
                    continue;
                };
                names.insert(stem.to_string());
            }
        }
        Ok(names.into_iter().collect())
    }

    async fn evaluate_candidate(
        &self,
        objective: &ObjectiveIR,
        task: &TaskIR,
        harness: &ResolvedHarness,
        assignments: &HashMap<String, Value>,
        train_cases: &[DatasetCase],
        val_cases: &[DatasetCase],
        test_cases: &[DatasetCase],
    ) -> Result<CandidateEvaluation> {
        let train = self
            .evaluate_split(objective, task, harness, train_cases)
            .await?;
        let val = if val_cases.is_empty() {
            None
        } else {
            Some(
                self.evaluate_split(objective, task, harness, val_cases)
                    .await?,
            )
        };
        let test = if test_cases.is_empty() {
            None
        } else {
            Some(
                self.evaluate_split(objective, task, harness, test_cases)
                    .await?,
            )
        };

        Ok(CandidateEvaluation {
            assignments: assignments.clone(),
            train,
            val,
            test,
        })
    }

    async fn evaluate_split(
        &self,
        objective: &ObjectiveIR,
        task: &TaskIR,
        harness: &ResolvedHarness,
        cases: &[DatasetCase],
    ) -> Result<SplitEvaluationSummary> {
        let repeats = objective.repeats.unwrap_or(1).max(1);
        let mut rollouts = Vec::new();
        for case in cases {
            for repeat in 0..repeats {
                rollouts.push(
                    self.evaluate_rollout(objective, task, harness, case, repeat)
                        .await?,
                );
            }
        }
        self.summarize_rollouts(&rollouts)
    }

    async fn evaluate_rollout(
        &self,
        objective: &ObjectiveIR,
        task: &TaskIR,
        harness: &ResolvedHarness,
        case: &DatasetCase,
        repeat: u64,
    ) -> Result<RolloutEvaluation> {
        let started = Instant::now();
        let run = self
            .execute_with_resolved_harness_trace(task, harness, case.input.clone())
            .await;
        let duration_ms = started.elapsed().as_secs_f64() * 1000.0;

        let telemetry = self.new_telemetry_handle();
        *telemetry.borrow_mut() = run.telemetry.clone();

        let (success, output, error) = match run.output {
            Ok(output) => (true, output, None),
            Err(error) => (
                false,
                self.default_value_for_type(Some(&task.output))?,
                Some(error.to_string()),
            ),
        };

        let mut artifacts = HashMap::new();
        artifacts.insert("expected".to_string(), case.expected.clone());
        artifacts.insert("output".to_string(), output);
        let mut ctx = ExecutionContext {
            input: case.input.clone(),
            artifacts,
            telemetry,
        };
        self.refresh_rollout_artifact(
            &mut ctx,
            success,
            duration_ms,
            repeat,
            case.id.as_deref(),
            error.as_deref(),
        )?;

        let mut metrics = HashMap::new();
        let constraints_ok = self
            .evaluate_named_rollout_signals(
                &objective.constraints,
                &mut ctx,
                &mut metrics,
                "constraint",
                true,
                success,
                duration_ms,
                repeat,
                case.id.as_deref(),
                error.as_deref(),
            )
            .await?;
        self.evaluate_named_rollout_signals(
            &objective.checkers,
            &mut ctx,
            &mut metrics,
            "checker",
            false,
            success,
            duration_ms,
            repeat,
            case.id.as_deref(),
            error.as_deref(),
        )
        .await?;
        self.evaluate_named_rollout_signals(
            &objective.judges,
            &mut ctx,
            &mut metrics,
            "judge",
            false,
            success,
            duration_ms,
            repeat,
            case.id.as_deref(),
            error.as_deref(),
        )
        .await?;
        self.evaluate_named_rollout_signals(
            &objective.metrics,
            &mut ctx,
            &mut metrics,
            "metric",
            false,
            success,
            duration_ms,
            repeat,
            case.id.as_deref(),
            error.as_deref(),
        )
        .await?;

        self.refresh_rollout_artifact(
            &mut ctx,
            success,
            duration_ms,
            repeat,
            case.id.as_deref(),
            error.as_deref(),
        )?;

        let mut score = self.numeric_objective_value(
            &self.eval_objective_expr(&objective.score, &ctx).await?,
            "objective score",
        )?;

        let (mut primary, mut tie_breakers) = if let Some(select) = &objective.select {
            self.evaluate_select(select, &ctx).await?
        } else {
            (score, Vec::new())
        };

        if !constraints_ok {
            let penalty = -1_000_000_000.0;
            score = penalty;
            primary = penalty;
            tie_breakers = vec![penalty; tie_breakers.len()];
        }

        Ok(RolloutEvaluation {
            metrics,
            score,
            primary,
            tie_breakers,
        })
    }

    async fn evaluate_named_rollout_signals(
        &self,
        decls: &[scaffold_ir::MetricIR],
        ctx: &mut ExecutionContext,
        metrics: &mut HashMap<String, f64>,
        label: &str,
        require_bool: bool,
        success: bool,
        duration_ms: f64,
        repeat: u64,
        case_id: Option<&str>,
        error: Option<&str>,
    ) -> Result<bool> {
        let mut all_passed = true;
        for decl in decls {
            self.refresh_rollout_artifact(ctx, success, duration_ms, repeat, case_id, error)?;
            let value = self.eval_objective_expr(&decl.expr, ctx).await?;
            if require_bool {
                match value {
                    Value::Bool(passed) => {
                        all_passed &= passed;
                        metrics.insert(decl.name.clone(), if passed { 1.0 } else { 0.0 });
                        ctx.artifacts.insert(decl.name.clone(), Value::Bool(passed));
                    }
                    other => {
                        return Err(Error::Runtime(format!(
                            "{} '{}' must evaluate to bool, found {}",
                            label,
                            decl.name,
                            other.type_name()
                        )))
                    }
                }
            } else {
                let numeric =
                    self.numeric_objective_value(&value, &format!("{} '{}'", label, decl.name))?;
                metrics.insert(decl.name.clone(), numeric);
                ctx.artifacts.insert(decl.name.clone(), value);
            }
        }
        Ok(all_passed)
    }

    async fn evaluate_select(
        &self,
        select: &SelectIR,
        ctx: &ExecutionContext,
    ) -> Result<(f64, Vec<f64>)> {
        let primary = self.numeric_objective_value(
            &self.eval_objective_expr(&select.primary, ctx).await?,
            "select.primary",
        )?;
        let mut tie_breakers = Vec::with_capacity(select.tie_breakers.len());
        for expr in &select.tie_breakers {
            tie_breakers.push(self.numeric_objective_value(
                &self.eval_objective_expr(expr, ctx).await?,
                "select.tie_breaker",
            )?);
        }
        Ok((primary, tie_breakers))
    }

    fn summarize_rollouts(&self, rollouts: &[RolloutEvaluation]) -> Result<SplitEvaluationSummary> {
        if rollouts.is_empty() {
            return Ok(SplitEvaluationSummary {
                rollouts: 0,
                metrics: HashMap::new(),
                score: 0.0,
                primary: 0.0,
                tie_breakers: Vec::new(),
            });
        }

        let count = rollouts.len() as f64;
        let mut metrics = HashMap::new();
        for rollout in rollouts {
            for (name, value) in &rollout.metrics {
                *metrics.entry(name.clone()).or_insert(0.0) += value;
            }
        }
        for value in metrics.values_mut() {
            *value /= count;
        }

        let tie_breaker_len = rollouts
            .iter()
            .map(|rollout| rollout.tie_breakers.len())
            .max()
            .unwrap_or(0);
        let mut tie_breakers = vec![0.0; tie_breaker_len];
        let mut score = 0.0;
        let mut primary = 0.0;
        for rollout in rollouts {
            score += rollout.score;
            primary += rollout.primary;
            for (index, value) in rollout.tie_breakers.iter().enumerate() {
                tie_breakers[index] += value;
            }
        }
        score /= count;
        primary /= count;
        for value in &mut tie_breakers {
            *value /= count;
        }

        Ok(SplitEvaluationSummary {
            rollouts: rollouts.len(),
            metrics,
            score,
            primary,
            tie_breakers,
        })
    }

    fn numeric_objective_value(&self, value: &Value, label: &str) -> Result<f64> {
        match value {
            Value::Bool(value) => Ok(if *value { 1.0 } else { 0.0 }),
            Value::Int(value) => Ok(*value as f64),
            Value::Float(value) if value.is_finite() => Ok(*value),
            other => Err(Error::Runtime(format!(
                "{} must evaluate to bool or numeric, found {}",
                label,
                other.type_name()
            ))),
        }
    }

    fn candidate_beats(
        &self,
        candidate: &CandidateEvaluation,
        current: &CandidateEvaluation,
    ) -> bool {
        self.compare_summary(&candidate.train, &current.train)
            .then_with(|| self.compare_assignments(&candidate.assignments, &current.assignments))
            .is_gt()
    }

    fn compare_summary(
        &self,
        left: &SplitEvaluationSummary,
        right: &SplitEvaluationSummary,
    ) -> std::cmp::Ordering {
        left.primary
            .total_cmp(&right.primary)
            .then_with(|| {
                for (left_value, right_value) in
                    left.tie_breakers.iter().zip(right.tie_breakers.iter())
                {
                    let ordering = left_value.total_cmp(right_value);
                    if !ordering.is_eq() {
                        return ordering;
                    }
                }
                left.tie_breakers.len().cmp(&right.tie_breakers.len())
            })
            .then_with(|| left.score.total_cmp(&right.score))
    }

    fn compare_assignments(
        &self,
        left: &HashMap<String, Value>,
        right: &HashMap<String, Value>,
    ) -> std::cmp::Ordering {
        let mut left_items = left
            .iter()
            .map(|(key, value)| (key.clone(), value.to_string()))
            .collect::<Vec<_>>();
        let mut right_items = right
            .iter()
            .map(|(key, value)| (key.clone(), value.to_string()))
            .collect::<Vec<_>>();
        left_items.sort();
        right_items.sort();
        left_items.cmp(&right_items)
    }

    fn power_set(&self, values: &[Value]) -> Vec<Vec<Value>> {
        let mut subsets = Vec::new();
        let total = 1usize << values.len();
        for mask in 0..total {
            let mut subset = Vec::new();
            for (index, value) in values.iter().enumerate() {
                if (mask & (1usize << index)) != 0 {
                    subset.push(value.clone());
                }
            }
            subsets.push(subset);
        }
        subsets
    }

    fn eval_static_expr(&self, expr: &ExprIR) -> Result<Value> {
        self.eval_static_expr_inner(expr)
    }

    fn eval_static_expr_inner(&self, expr: &ExprIR) -> Result<Value> {
        match expr {
            ExprIR::Ident { name } => Ok(Value::String(name.clone())),
            ExprIR::FieldAccess { .. } => Err(Error::Runtime(
                "static harness expressions do not support field access".to_string(),
            )),
            ExprIR::Call { function, args } => {
                let values = args
                    .iter()
                    .map(|arg| self.eval_static_expr_inner(arg))
                    .collect::<Result<Vec<_>>>()?;
                self.eval_call(function, &values)
            }
            ExprIR::List { elements } => Ok(Value::List(
                elements
                    .iter()
                    .map(|element| self.eval_static_expr_inner(element))
                    .collect::<Result<Vec<_>>>()?,
            )),
            ExprIR::Record { fields } => {
                let mut out = HashMap::new();
                for field in fields {
                    out.insert(
                        field.key.clone(),
                        self.eval_static_expr_inner(&field.value)?,
                    );
                }
                Ok(Value::Map(out))
            }
            other => {
                let ctx = ExecutionContext::default();
                self.eval_expr(other, &ctx)
            }
        }
    }

    fn eval_loop_max_iters(
        &self,
        loop_decl: &scaffold_ir::LoopIR,
        ctx: &ExecutionContext,
        harness: &ResolvedHarness,
    ) -> Result<usize> {
        if let Some(value) = self.harness_u64(harness, &loop_decl.name, "max_iters")? {
            return Ok(value as usize);
        }
        match self.eval_expr(&loop_decl.max_iters, ctx)? {
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

    fn resolve_text_surface(&self, value: &str, require_variant: bool) -> Result<String> {
        if let Some((group, name)) = value.split_once("::") {
            if let Some(path) = self.find_variant_file(group, name) {
                return std::fs::read_to_string(&path).map_err(|e| {
                    Error::Runtime(format!(
                        "failed to read variant file {}: {}",
                        path.display(),
                        e
                    ))
                });
            }
            if require_variant {
                return Err(Error::Runtime(format!(
                    "variant '{}' could not be resolved under {}",
                    value,
                    self.base_dir.display()
                )));
            }
        }
        Ok(value.to_string())
    }

    fn find_variant_file(&self, group: &str, name: &str) -> Option<PathBuf> {
        let candidates = [
            self.base_dir
                .join("variants")
                .join(group)
                .join(format!("{}.md", name)),
            self.base_dir
                .join("variants")
                .join(group)
                .join(format!("{}.txt", name)),
            self.base_dir
                .join("variants")
                .join(group)
                .join(format!("{}.prompt", name)),
            self.base_dir
                .join("prompts")
                .join(group)
                .join(format!("{}.md", name)),
            self.base_dir
                .join("prompts")
                .join(group)
                .join(format!("{}.txt", name)),
            self.base_dir.join("prompts").join(format!("{}.md", name)),
            self.base_dir.join("prompts").join(format!("{}.txt", name)),
        ];
        candidates.into_iter().find(|path| path.is_file())
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

    fn eval_objective_expr<'b>(
        &'b self,
        expr: &'b ExprIR,
        ctx: &'b ExecutionContext,
    ) -> Pin<Box<dyn Future<Output = Result<Value>> + 'b>> {
        Box::pin(async move {
            match expr {
                ExprIR::Literal { value } => Ok(self.literal_to_value(value)),
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
                    let base = self.eval_objective_expr(base, ctx).await?;
                    base.field(field).cloned().ok_or_else(|| {
                        Error::Runtime(format!(
                            "field '{}' not found on {}",
                            field,
                            self.type_label_from_value(&base)
                        ))
                    })
                }
                ExprIR::Binary { left, op, right } => {
                    let left = self.eval_objective_expr(left, ctx).await?;
                    let right = self.eval_objective_expr(right, ctx).await?;
                    self.eval_binary(&left, op, &right)
                }
                ExprIR::Call { function, args } => {
                    let mut values = Vec::with_capacity(args.len());
                    for arg in args {
                        values.push(self.eval_objective_expr(arg, ctx).await?);
                    }
                    self.eval_objective_call(function, args, &values, ctx).await
                }
                ExprIR::ForeignCall {
                    module, function, ..
                } => Err(Error::Runtime(format!(
                    "task interpreter does not support foreign call '{}::{}'",
                    module, function
                ))),
                ExprIR::List { elements } => {
                    let mut out = Vec::with_capacity(elements.len());
                    for element in elements {
                        out.push(self.eval_objective_expr(element, ctx).await?);
                    }
                    Ok(Value::List(out))
                }
                ExprIR::Record { fields } => {
                    let mut out = HashMap::new();
                    for field in fields {
                        out.insert(
                            field.key.clone(),
                            self.eval_objective_expr(&field.value, ctx).await?,
                        );
                    }
                    Ok(Value::Map(out))
                }
            }
        })
    }

    async fn eval_objective_call(
        &self,
        function: &str,
        arg_exprs: &[ExprIR],
        args: &[Value],
        ctx: &ExecutionContext,
    ) -> Result<Value> {
        if builtins::BUILTIN_NAMES.contains(&function) || function == "variant" {
            return self.eval_call(function, args);
        }

        if let Ok(tool) = self.find_tool(function) {
            let input =
                self.prepare_component_call_input_from_expr_args(&tool.input, arg_exprs, args)?;
            return self.execute_tool(tool, input, None, None, function, &ctx.telemetry);
        }

        if let Ok(prompt) = self.find_prompt(function) {
            let input =
                self.prepare_component_call_input_from_expr_args(&prompt.input, arg_exprs, args)?;
            return self
                .execute_prompt_component(prompt, input, function, &ctx.telemetry)
                .await;
        }

        if let Ok(agent) = self.find_agent(function) {
            let input =
                self.prepare_component_call_input_from_expr_args(&agent.input, arg_exprs, args)?;
            return self
                .execute_agent_component(agent, input, function, &ctx.telemetry)
                .await;
        }

        self.eval_call(function, args)
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

    fn find_harness(&self, name: &str) -> Result<&HarnessIR> {
        self.ir
            .harnesses
            .iter()
            .find(|harness| harness.name == name)
            .ok_or_else(|| Error::Runtime(format!("harness '{}' not found", name)))
    }

    fn find_objective(&self, name: &str) -> Result<&ObjectiveIR> {
        self.ir
            .objectives
            .iter()
            .find(|objective| objective.name == name)
            .ok_or_else(|| Error::Runtime(format!("objective '{}' not found", name)))
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

    fn find_tool(&self, name: &str) -> Result<&ToolIR> {
        self.ir
            .tools
            .iter()
            .find(|tool| tool.name == name)
            .ok_or_else(|| Error::Runtime(format!("tool '{}' not found", name)))
    }

    fn find_task_target<'b>(
        &self,
        task: &'b TaskIR,
        target: &str,
    ) -> Result<TaskTargetKindRuntime<'b>> {
        fn visit<'a>(nodes: &'a [TaskNodeIR], target: &str) -> Option<TaskTargetKindRuntime<'a>> {
            for node in nodes {
                match node {
                    TaskNodeIR::Stage(stage) if stage.name == target => {
                        return Some(TaskTargetKindRuntime::Stage {
                            kind: stage.stage_kind,
                            component: &stage.component,
                        });
                    }
                    TaskNodeIR::Loop(loop_decl) if loop_decl.name == target => {
                        return Some(TaskTargetKindRuntime::Loop);
                    }
                    TaskNodeIR::Loop(loop_decl) => {
                        if let Some(found) = visit(&loop_decl.body, target) {
                            return Some(found);
                        }
                    }
                    TaskNodeIR::Branch(branch) => {
                        if let Some(found) = visit(&branch.then_body, target) {
                            return Some(found);
                        }
                        if let Some(found) = visit(&branch.else_body, target) {
                            return Some(found);
                        }
                    }
                    _ => {}
                }
            }
            None
        }

        visit(&task.body, target).ok_or_else(|| {
            Error::Runtime(format!(
                "task '{}' does not define target '{}'",
                task.name, target
            ))
        })
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
    use crate::llm::{clear_mock_structured_sequence, set_mock_structured_sequence};
    use scaffold_ir::{
        ArtifactSlotIR, BindingIR, BindingPathIR, EmitFieldIR, ExprFieldIR, FiniteDomainIR,
        HarnessIR, MetricIR, ObjectiveIR, SelectIR, StageIR, ToolExprIR, ToolIR, ToolImplIR,
        ToolStatementIR, ToolVariantIR, TunableIR, TuneOperatorIR, TypeDefIR, TypeDefKindIR,
    };
    use serde_json::json;
    use std::sync::{Mutex, OnceLock};

    fn llm_mock_guard() -> std::sync::MutexGuard<'static, ()> {
        static GUARD: OnceLock<Mutex<()>> = OnceLock::new();
        GUARD
            .get_or_init(|| Mutex::new(()))
            .lock()
            .expect("llm mock test guard poisoned")
    }

    fn string_type() -> TypeIR {
        TypeIR::String
    }

    fn answer_type() -> TypeIR {
        TypeIR::Struct {
            fields: HashMap::from([("text".to_string(), TypeIR::String)]),
        }
    }

    fn empty_ir() -> ScaffoldIR {
        ScaffoldIR {
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
        }
    }

    #[test]
    fn eval_expr_supports_field_access_and_builtins() {
        let empty_ir = empty_ir();
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
            telemetry: Rc::new(RefCell::new(RolloutTelemetry::default())),
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
    fn execute_task_runs_tool_stage_with_variant_override() {
        let ir = ScaffoldIR {
            tools: vec![ToolIR {
                name: "formatter".to_string(),
                input: TypeIR::Struct {
                    fields: HashMap::from([("text".to_string(), string_type())]),
                },
                output: TypeIR::String,
                implementation: Some(ToolImplIR::Expr {
                    expr: ToolExprIR::Ident {
                        name: "text".to_string(),
                    },
                }),
                spec: None,
                variants: vec![ToolVariantIR {
                    name: "shout".to_string(),
                    implementation: ToolImplIR::Expr {
                        expr: ToolExprIR::Shell {
                            command: "printf '{text}' | tr '[:lower:]' '[:upper:]'".to_string(),
                        },
                    },
                }],
            }],
            types: vec![TypeDefIR {
                name: "TaskOutput".to_string(),
                kind: TypeDefKindIR::Type,
                definition: TypeIR::Struct {
                    fields: HashMap::from([("answer".to_string(), TypeIR::String)]),
                },
            }],
            tasks: vec![TaskIR {
                name: "format_text".to_string(),
                input: TypeIR::Struct {
                    fields: HashMap::from([("text".to_string(), string_type())]),
                },
                output: TypeIR::Named {
                    name: "TaskOutput".to_string(),
                },
                artifacts: vec![ArtifactSlotIR {
                    name: "formatted".to_string(),
                    ty: TypeIR::String,
                }],
                body: vec![TaskNodeIR::Stage(StageIR {
                    name: "format".to_string(),
                    stage_kind: StageKindIR::Tool,
                    component: "formatter".to_string(),
                    input: ExprIR::Ident {
                        name: "input".to_string(),
                    },
                    output: "formatted".to_string(),
                    when: None,
                })],
                emit: vec![EmitFieldIR {
                    name: "answer".to_string(),
                    value: ExprIR::Ident {
                        name: "formatted".to_string(),
                    },
                }],
            }],
            harnesses: vec![HarnessIR {
                name: "shouty".to_string(),
                task: "format_text".to_string(),
                defaults: Vec::new(),
                bindings: vec![scaffold_ir::TargetBindingIR {
                    target: "format".to_string(),
                    bindings: vec![BindingIR {
                        key: BindingPathIR {
                            segments: vec!["variant".to_string()],
                        },
                        value: ExprIR::Literal {
                            value: LiteralIR::String {
                                value: "formatter::shout".to_string(),
                            },
                        },
                    }],
                }],
                tunables: Vec::new(),
            }],
            ..empty_ir()
        };

        let output = execute_task(
            &ir,
            "format_text",
            Some("shouty"),
            Value::Map(HashMap::from([(
                "text".to_string(),
                Value::String("hello".to_string()),
            )])),
            Path::new("."),
        )
        .unwrap();

        match output {
            Value::Struct { type_name, fields } => {
                assert_eq!(type_name, "TaskOutput");
                assert_eq!(fields.get("answer"), Some(&Value::String("HELLO".into())));
            }
            other => panic!("expected struct output, got {:?}", other),
        }
    }

    #[test]
    fn execute_task_runs_nested_tool_sequence() {
        let analysis_type = TypeIR::Struct {
            fields: HashMap::from([
                ("words".to_string(), TypeIR::Int),
                ("original".to_string(), TypeIR::String),
            ]),
        };
        let ir = ScaffoldIR {
            types: vec![
                TypeDefIR {
                    name: "Analysis".to_string(),
                    kind: TypeDefKindIR::Artifact,
                    definition: analysis_type.clone(),
                },
                TypeDefIR {
                    name: "TaskOutput".to_string(),
                    kind: TypeDefKindIR::Type,
                    definition: analysis_type.clone(),
                },
            ],
            tools: vec![
                ToolIR {
                    name: "count_words".to_string(),
                    input: TypeIR::Struct {
                        fields: HashMap::from([("text".to_string(), string_type())]),
                    },
                    output: TypeIR::Int,
                    implementation: Some(ToolImplIR::Expr {
                        expr: ToolExprIR::Shell {
                            command: "echo '{text}' | wc -w | tr -d ' '".to_string(),
                        },
                    }),
                    spec: None,
                    variants: Vec::new(),
                },
                ToolIR {
                    name: "analyze_text".to_string(),
                    input: TypeIR::Struct {
                        fields: HashMap::from([("text".to_string(), string_type())]),
                    },
                    output: TypeIR::Named {
                        name: "Analysis".to_string(),
                    },
                    implementation: Some(ToolImplIR::Sequence {
                        statements: vec![
                            ToolStatementIR {
                                binding: Some("words".to_string()),
                                expr: ToolExprIR::ToolCall {
                                    tool: "count_words".to_string(),
                                    args: vec![ToolExprIR::Ident {
                                        name: "text".to_string(),
                                    }],
                                },
                            },
                            ToolStatementIR {
                                binding: Some("original".to_string()),
                                expr: ToolExprIR::Ident {
                                    name: "text".to_string(),
                                },
                            },
                        ],
                    }),
                    spec: None,
                    variants: Vec::new(),
                },
            ],
            tasks: vec![TaskIR {
                name: "analyze".to_string(),
                input: TypeIR::Struct {
                    fields: HashMap::from([("text".to_string(), string_type())]),
                },
                output: TypeIR::Named {
                    name: "TaskOutput".to_string(),
                },
                artifacts: vec![ArtifactSlotIR {
                    name: "analysis".to_string(),
                    ty: TypeIR::Named {
                        name: "Analysis".to_string(),
                    },
                }],
                body: vec![TaskNodeIR::Stage(StageIR {
                    name: "analyze_text".to_string(),
                    stage_kind: StageKindIR::Tool,
                    component: "analyze_text".to_string(),
                    input: ExprIR::Ident {
                        name: "input".to_string(),
                    },
                    output: "analysis".to_string(),
                    when: None,
                })],
                emit: vec![
                    EmitFieldIR {
                        name: "words".to_string(),
                        value: ExprIR::FieldAccess {
                            base: Box::new(ExprIR::Ident {
                                name: "analysis".to_string(),
                            }),
                            field: "words".to_string(),
                        },
                    },
                    EmitFieldIR {
                        name: "original".to_string(),
                        value: ExprIR::FieldAccess {
                            base: Box::new(ExprIR::Ident {
                                name: "analysis".to_string(),
                            }),
                            field: "original".to_string(),
                        },
                    },
                ],
            }],
            ..empty_ir()
        };

        let output = execute_task(
            &ir,
            "analyze",
            None,
            Value::Map(HashMap::from([(
                "text".to_string(),
                Value::String("two words".to_string()),
            )])),
            Path::new("."),
        )
        .unwrap();

        match output {
            Value::Struct { type_name, fields } => {
                assert_eq!(type_name, "TaskOutput");
                assert_eq!(fields.get("words"), Some(&Value::Int(2)));
                assert_eq!(
                    fields.get("original"),
                    Some(&Value::String("two words".into()))
                );
            }
            other => panic!("expected struct output, got {:?}", other),
        }
    }

    #[test]
    fn execute_task_runs_prompt_stage_with_harness_overrides() {
        let _guard = llm_mock_guard();
        clear_mock_structured_sequence();
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
                constraints: Vec::new(),
                checkers: Vec::new(),
                judges: Vec::new(),
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

    #[test]
    fn execute_task_applies_loop_max_iters_harness_override() {
        let ir = ScaffoldIR {
            tools: vec![ToolIR {
                name: "increment".to_string(),
                input: TypeIR::Struct {
                    fields: HashMap::from([("value".to_string(), TypeIR::Int)]),
                },
                output: TypeIR::Int,
                implementation: Some(ToolImplIR::Expr {
                    expr: ToolExprIR::Expr {
                        expr: Box::new(ExprIR::Binary {
                            left: Box::new(ExprIR::Ident {
                                name: "value".to_string(),
                            }),
                            op: "+".to_string(),
                            right: Box::new(ExprIR::Literal {
                                value: LiteralIR::Int { value: 1 },
                            }),
                        }),
                    },
                }),
                spec: None,
                variants: Vec::new(),
            }],
            types: vec![TypeDefIR {
                name: "CounterOutput".to_string(),
                kind: TypeDefKindIR::Type,
                definition: TypeIR::Struct {
                    fields: HashMap::from([("count".to_string(), TypeIR::Int)]),
                },
            }],
            tasks: vec![TaskIR {
                name: "count_twice".to_string(),
                input: TypeIR::Struct {
                    fields: HashMap::new(),
                },
                output: TypeIR::Named {
                    name: "CounterOutput".to_string(),
                },
                artifacts: vec![ArtifactSlotIR {
                    name: "counter".to_string(),
                    ty: TypeIR::Int,
                }],
                body: vec![
                    TaskNodeIR::Stage(StageIR {
                        name: "seed".to_string(),
                        stage_kind: StageKindIR::Tool,
                        component: "increment".to_string(),
                        input: ExprIR::Record {
                            fields: vec![ExprFieldIR {
                                key: "value".to_string(),
                                value: ExprIR::Literal {
                                    value: LiteralIR::Int { value: -1 },
                                },
                            }],
                        },
                        output: "counter".to_string(),
                        when: None,
                    }),
                    TaskNodeIR::Loop(scaffold_ir::LoopIR {
                        name: "refine".to_string(),
                        max_iters: ExprIR::Literal {
                            value: LiteralIR::Int { value: 5 },
                        },
                        carry: vec!["counter".to_string()],
                        while_condition: Some(ExprIR::Literal {
                            value: LiteralIR::Bool { value: true },
                        }),
                        until: None,
                        body: vec![TaskNodeIR::Stage(StageIR {
                            name: "step".to_string(),
                            stage_kind: StageKindIR::Tool,
                            component: "increment".to_string(),
                            input: ExprIR::Record {
                                fields: vec![ExprFieldIR {
                                    key: "value".to_string(),
                                    value: ExprIR::Ident {
                                        name: "counter".to_string(),
                                    },
                                }],
                            },
                            output: "counter".to_string(),
                            when: None,
                        })],
                    }),
                ],
                emit: vec![EmitFieldIR {
                    name: "count".to_string(),
                    value: ExprIR::Ident {
                        name: "counter".to_string(),
                    },
                }],
            }],
            harnesses: vec![HarnessIR {
                name: "bounded".to_string(),
                task: "count_twice".to_string(),
                defaults: Vec::new(),
                bindings: vec![scaffold_ir::TargetBindingIR {
                    target: "refine".to_string(),
                    bindings: vec![BindingIR {
                        key: BindingPathIR {
                            segments: vec!["max_iters".to_string()],
                        },
                        value: ExprIR::Literal {
                            value: LiteralIR::Int { value: 2 },
                        },
                    }],
                }],
                tunables: Vec::new(),
            }],
            ..empty_ir()
        };

        let output = execute_task(
            &ir,
            "count_twice",
            Some("bounded"),
            Value::Map(HashMap::new()),
            Path::new("."),
        )
        .unwrap();

        match output {
            Value::Struct { fields, .. } => {
                assert_eq!(fields.get("count"), Some(&Value::Int(2)));
            }
            other => panic!("expected struct output, got {:?}", other),
        }
    }

    #[test]
    fn execute_task_skips_loop_when_while_condition_is_false_before_first_iteration() {
        let ir = ScaffoldIR {
            tools: vec![ToolIR {
                name: "increment".to_string(),
                input: TypeIR::Struct {
                    fields: HashMap::from([("value".to_string(), TypeIR::Int)]),
                },
                output: TypeIR::Int,
                implementation: Some(ToolImplIR::Expr {
                    expr: ToolExprIR::Expr {
                        expr: Box::new(ExprIR::Binary {
                            left: Box::new(ExprIR::Ident {
                                name: "value".to_string(),
                            }),
                            op: "+".to_string(),
                            right: Box::new(ExprIR::Literal {
                                value: LiteralIR::Int { value: 1 },
                            }),
                        }),
                    },
                }),
                spec: None,
                variants: Vec::new(),
            }],
            types: vec![TypeDefIR {
                name: "CounterOutput".to_string(),
                kind: TypeDefKindIR::Type,
                definition: TypeIR::Struct {
                    fields: HashMap::from([("count".to_string(), TypeIR::Int)]),
                },
            }],
            tasks: vec![TaskIR {
                name: "count_once".to_string(),
                input: TypeIR::Struct {
                    fields: HashMap::new(),
                },
                output: TypeIR::Named {
                    name: "CounterOutput".to_string(),
                },
                artifacts: vec![ArtifactSlotIR {
                    name: "counter".to_string(),
                    ty: TypeIR::Int,
                }],
                body: vec![
                    TaskNodeIR::Stage(StageIR {
                        name: "seed".to_string(),
                        stage_kind: StageKindIR::Tool,
                        component: "increment".to_string(),
                        input: ExprIR::Record {
                            fields: vec![ExprFieldIR {
                                key: "value".to_string(),
                                value: ExprIR::Literal {
                                    value: LiteralIR::Int { value: -1 },
                                },
                            }],
                        },
                        output: "counter".to_string(),
                        when: None,
                    }),
                    TaskNodeIR::Loop(scaffold_ir::LoopIR {
                        name: "refine".to_string(),
                        max_iters: ExprIR::Literal {
                            value: LiteralIR::Int { value: 5 },
                        },
                        carry: vec!["counter".to_string()],
                        while_condition: Some(ExprIR::Binary {
                            left: Box::new(ExprIR::Ident {
                                name: "counter".to_string(),
                            }),
                            op: "<".to_string(),
                            right: Box::new(ExprIR::Literal {
                                value: LiteralIR::Int { value: 0 },
                            }),
                        }),
                        until: None,
                        body: vec![TaskNodeIR::Stage(StageIR {
                            name: "step".to_string(),
                            stage_kind: StageKindIR::Tool,
                            component: "increment".to_string(),
                            input: ExprIR::Record {
                                fields: vec![ExprFieldIR {
                                    key: "value".to_string(),
                                    value: ExprIR::Ident {
                                        name: "counter".to_string(),
                                    },
                                }],
                            },
                            output: "counter".to_string(),
                            when: None,
                        })],
                    }),
                ],
                emit: vec![EmitFieldIR {
                    name: "count".to_string(),
                    value: ExprIR::Ident {
                        name: "counter".to_string(),
                    },
                }],
            }],
            ..empty_ir()
        };

        let output = execute_task(
            &ir,
            "count_once",
            None,
            Value::Map(HashMap::new()),
            Path::new("."),
        )
        .unwrap();

        match output {
            Value::Struct { fields, .. } => {
                assert_eq!(fields.get("count"), Some(&Value::Int(0)));
            }
            other => panic!("expected struct output, got {:?}", other),
        }
    }

    #[test]
    fn execute_task_runs_tool_enabled_agent_stage() {
        let _guard = llm_mock_guard();
        clear_mock_structured_sequence();
        set_mock_structured_sequence(vec![
            json!({
                "action": "tool",
                "tool": "lookup",
                "tool_input": { "country": "France" }
            }),
            json!({
                "action": "final",
                "output": { "text": "Paris" }
            }),
        ]);

        let ir = ScaffoldIR {
            tools: vec![ToolIR {
                name: "lookup".to_string(),
                input: TypeIR::Struct {
                    fields: HashMap::from([("country".to_string(), TypeIR::String)]),
                },
                output: TypeIR::String,
                implementation: Some(ToolImplIR::Expr {
                    expr: ToolExprIR::Literal {
                        value: LiteralIR::String {
                            value: "Paris".to_string(),
                        },
                    },
                }),
                spec: None,
                variants: Vec::new(),
            }],
            types: vec![
                TypeDefIR {
                    name: "Draft".to_string(),
                    kind: TypeDefKindIR::Artifact,
                    definition: answer_type(),
                },
                TypeDefIR {
                    name: "AgentOutput".to_string(),
                    kind: TypeDefKindIR::Type,
                    definition: answer_type(),
                },
            ],
            agents: vec![AgentIR {
                name: "researcher".to_string(),
                input: TypeIR::Struct {
                    fields: HashMap::from([("question".to_string(), TypeIR::String)]),
                },
                output: TypeIR::Named {
                    name: "Draft".to_string(),
                },
                tools: vec!["lookup".to_string()],
                system: StringOrFileIR::Literal {
                    value: "Use tools when needed.".to_string(),
                },
                model: Some("gpt-5-mini".to_string()),
                max_turns: Some(3),
                reward: None,
                done: None,
                on_error: scaffold_ir::ErrorStrategyIR::Abort,
                timeout: None,
            }],
            tasks: vec![TaskIR {
                name: "ask".to_string(),
                input: TypeIR::Struct {
                    fields: HashMap::from([("question".to_string(), TypeIR::String)]),
                },
                output: TypeIR::Named {
                    name: "AgentOutput".to_string(),
                },
                artifacts: vec![ArtifactSlotIR {
                    name: "draft".to_string(),
                    ty: TypeIR::Named {
                        name: "Draft".to_string(),
                    },
                }],
                body: vec![TaskNodeIR::Stage(StageIR {
                    name: "research".to_string(),
                    stage_kind: StageKindIR::Agent,
                    component: "researcher".to_string(),
                    input: ExprIR::Ident {
                        name: "input".to_string(),
                    },
                    output: "draft".to_string(),
                    when: None,
                })],
                emit: vec![EmitFieldIR {
                    name: "text".to_string(),
                    value: ExprIR::FieldAccess {
                        base: Box::new(ExprIR::Ident {
                            name: "draft".to_string(),
                        }),
                        field: "text".to_string(),
                    },
                }],
            }],
            ..empty_ir()
        };

        let output = execute_task(
            &ir,
            "ask",
            None,
            Value::Map(HashMap::from([(
                "question".to_string(),
                Value::String("What is the capital of France?".to_string()),
            )])),
            Path::new("."),
        )
        .unwrap();

        clear_mock_structured_sequence();

        match output {
            Value::Struct { fields, .. } => {
                assert_eq!(fields.get("text"), Some(&Value::String("Paris".into())));
            }
            other => panic!("expected struct output, got {:?}", other),
        }
    }

    #[test]
    fn optimize_objective_selects_best_tool_variant() {
        let ir = ScaffoldIR {
            tools: vec![ToolIR {
                name: "formatter".to_string(),
                input: TypeIR::Struct {
                    fields: HashMap::from([("text".to_string(), TypeIR::String)]),
                },
                output: TypeIR::String,
                implementation: Some(ToolImplIR::Expr {
                    expr: ToolExprIR::Ident {
                        name: "text".to_string(),
                    },
                }),
                spec: None,
                variants: vec![
                    ToolVariantIR {
                        name: "plain".to_string(),
                        implementation: ToolImplIR::Expr {
                            expr: ToolExprIR::Ident {
                                name: "text".to_string(),
                            },
                        },
                    },
                    ToolVariantIR {
                        name: "shout".to_string(),
                        implementation: ToolImplIR::Expr {
                            expr: ToolExprIR::Shell {
                                command: "printf '{text}' | tr '[:lower:]' '[:upper:]'".to_string(),
                            },
                        },
                    },
                ],
            }],
            types: vec![TypeDefIR {
                name: "TaskOutput".to_string(),
                kind: TypeDefKindIR::Type,
                definition: TypeIR::Struct {
                    fields: HashMap::from([("answer".to_string(), TypeIR::String)]),
                },
            }],
            tasks: vec![TaskIR {
                name: "format_text".to_string(),
                input: TypeIR::Struct {
                    fields: HashMap::from([("text".to_string(), TypeIR::String)]),
                },
                output: TypeIR::Named {
                    name: "TaskOutput".to_string(),
                },
                artifacts: vec![ArtifactSlotIR {
                    name: "formatted".to_string(),
                    ty: TypeIR::String,
                }],
                body: vec![TaskNodeIR::Stage(StageIR {
                    name: "format".to_string(),
                    stage_kind: StageKindIR::Tool,
                    component: "formatter".to_string(),
                    input: ExprIR::Ident {
                        name: "input".to_string(),
                    },
                    output: "formatted".to_string(),
                    when: None,
                })],
                emit: vec![EmitFieldIR {
                    name: "answer".to_string(),
                    value: ExprIR::Ident {
                        name: "formatted".to_string(),
                    },
                }],
            }],
            harnesses: vec![HarnessIR {
                name: "search".to_string(),
                task: "format_text".to_string(),
                defaults: Vec::new(),
                bindings: Vec::new(),
                tunables: vec![TunableIR {
                    path: BindingPathIR {
                        segments: vec!["format".to_string(), "variant".to_string()],
                    },
                    operator: TuneOperatorIR::In,
                    domain: FiniteDomainIR::Variants {
                        name: "formatter".to_string(),
                    },
                }],
            }],
            objectives: vec![ObjectiveIR {
                name: "quality".to_string(),
                task: "format_text".to_string(),
                harness: "search".to_string(),
                dataset: scaffold_ir::DatasetSpecIR::Inline {
                    cases: vec![scaffold_ir::InlineDatasetCaseIR {
                        id: Some("one".to_string()),
                        input: ExprIR::Record {
                            fields: vec![ExprFieldIR {
                                key: "text".to_string(),
                                value: ExprIR::Literal {
                                    value: LiteralIR::String {
                                        value: "hello".to_string(),
                                    },
                                },
                            }],
                        },
                        expected: Some(ExprIR::Record {
                            fields: vec![ExprFieldIR {
                                key: "answer".to_string(),
                                value: ExprIR::Literal {
                                    value: LiteralIR::String {
                                        value: "HELLO".to_string(),
                                    },
                                },
                            }],
                        }),
                    }],
                },
                repeats: Some(1),
                constraints: Vec::new(),
                checkers: Vec::new(),
                judges: Vec::new(),
                metrics: vec![MetricIR {
                    name: "exact".to_string(),
                    expr: ExprIR::Binary {
                        left: Box::new(ExprIR::FieldAccess {
                            base: Box::new(ExprIR::Ident {
                                name: "output".to_string(),
                            }),
                            field: "answer".to_string(),
                        }),
                        op: "==".to_string(),
                        right: Box::new(ExprIR::FieldAccess {
                            base: Box::new(ExprIR::Ident {
                                name: "expected".to_string(),
                            }),
                            field: "answer".to_string(),
                        }),
                    },
                }],
                score: ExprIR::Ident {
                    name: "exact".to_string(),
                },
                split: None,
                select: Some(SelectIR {
                    primary: ExprIR::Ident {
                        name: "exact".to_string(),
                    },
                    tie_breakers: Vec::new(),
                }),
            }],
            ..empty_ir()
        };

        let report = optimize_objective(&ir, "quality", Path::new("."), 8).unwrap();

        assert_eq!(report.evaluated_candidates, 2);
        assert_eq!(
            report.best.assignments.get("format.variant"),
            Some(&Value::String("formatter::shout".to_string()))
        );
        assert_eq!(report.best.train.primary, 1.0);
    }

    #[test]
    fn optimize_objective_supports_tool_prompt_and_agent_calls_in_signals() {
        let _guard = llm_mock_guard();
        clear_mock_structured_sequence();
        set_mock_structured_sequence(vec![json!(0.75), json!(0.5)]);

        let ir = ScaffoldIR {
            tools: vec![
                ToolIR {
                    name: "echo_answer".to_string(),
                    input: TypeIR::Struct {
                        fields: HashMap::from([("text".to_string(), TypeIR::String)]),
                    },
                    output: TypeIR::String,
                    implementation: Some(ToolImplIR::Expr {
                        expr: ToolExprIR::Ident {
                            name: "text".to_string(),
                        },
                    }),
                    spec: None,
                    variants: Vec::new(),
                },
                ToolIR {
                    name: "exact_match".to_string(),
                    input: TypeIR::Struct {
                        fields: HashMap::from([
                            ("actual".to_string(), TypeIR::String),
                            ("expected".to_string(), TypeIR::String),
                        ]),
                    },
                    output: TypeIR::Bool,
                    implementation: Some(ToolImplIR::Expr {
                        expr: ToolExprIR::Expr {
                            expr: Box::new(ExprIR::Binary {
                                left: Box::new(ExprIR::Ident {
                                    name: "actual".to_string(),
                                }),
                                op: "==".to_string(),
                                right: Box::new(ExprIR::Ident {
                                    name: "expected".to_string(),
                                }),
                            }),
                        },
                    }),
                    spec: None,
                    variants: Vec::new(),
                },
            ],
            prompts: vec![PromptIR {
                name: "fluency_judge".to_string(),
                input: TypeIR::Struct {
                    fields: HashMap::from([("answer".to_string(), TypeIR::String)]),
                },
                output: TypeIR::Float,
                template: StringOrFileIR::Literal {
                    value: "Score this answer for fluency: {answer}".to_string(),
                },
                system: None,
            }],
            agents: vec![AgentIR {
                name: "utility_judge".to_string(),
                input: TypeIR::Struct {
                    fields: HashMap::from([("answer".to_string(), TypeIR::String)]),
                },
                output: TypeIR::Float,
                tools: Vec::new(),
                system: StringOrFileIR::Literal {
                    value: "Return a numeric usefulness score.".to_string(),
                },
                model: Some("gpt-5-mini".to_string()),
                max_turns: Some(1),
                reward: None,
                done: None,
                on_error: scaffold_ir::ErrorStrategyIR::Abort,
                timeout: None,
            }],
            types: vec![TypeDefIR {
                name: "AnswerOut".to_string(),
                kind: TypeDefKindIR::Type,
                definition: TypeIR::Struct {
                    fields: HashMap::from([("answer".to_string(), TypeIR::String)]),
                },
            }],
            tasks: vec![TaskIR {
                name: "echo".to_string(),
                input: TypeIR::Struct {
                    fields: HashMap::from([("text".to_string(), TypeIR::String)]),
                },
                output: TypeIR::Named {
                    name: "AnswerOut".to_string(),
                },
                artifacts: vec![ArtifactSlotIR {
                    name: "answer".to_string(),
                    ty: TypeIR::String,
                }],
                body: vec![TaskNodeIR::Stage(StageIR {
                    name: "echo".to_string(),
                    stage_kind: StageKindIR::Tool,
                    component: "echo_answer".to_string(),
                    input: ExprIR::Ident {
                        name: "input".to_string(),
                    },
                    output: "answer".to_string(),
                    when: None,
                })],
                emit: vec![EmitFieldIR {
                    name: "answer".to_string(),
                    value: ExprIR::Ident {
                        name: "answer".to_string(),
                    },
                }],
            }],
            harnesses: vec![HarnessIR {
                name: "baseline".to_string(),
                task: "echo".to_string(),
                defaults: Vec::new(),
                bindings: Vec::new(),
                tunables: Vec::new(),
            }],
            objectives: vec![ObjectiveIR {
                name: "quality".to_string(),
                task: "echo".to_string(),
                harness: "baseline".to_string(),
                dataset: scaffold_ir::DatasetSpecIR::Inline {
                    cases: vec![scaffold_ir::InlineDatasetCaseIR {
                        id: Some("one".to_string()),
                        input: ExprIR::Record {
                            fields: vec![ExprFieldIR {
                                key: "text".to_string(),
                                value: ExprIR::Literal {
                                    value: LiteralIR::String {
                                        value: "hello".to_string(),
                                    },
                                },
                            }],
                        },
                        expected: Some(ExprIR::Record {
                            fields: vec![ExprFieldIR {
                                key: "answer".to_string(),
                                value: ExprIR::Literal {
                                    value: LiteralIR::String {
                                        value: "hello".to_string(),
                                    },
                                },
                            }],
                        }),
                    }],
                },
                repeats: Some(1),
                constraints: vec![MetricIR {
                    name: "present".to_string(),
                    expr: ExprIR::Binary {
                        left: Box::new(ExprIR::Call {
                            function: "len".to_string(),
                            args: vec![ExprIR::FieldAccess {
                                base: Box::new(ExprIR::Ident {
                                    name: "output".to_string(),
                                }),
                                field: "answer".to_string(),
                            }],
                        }),
                        op: ">".to_string(),
                        right: Box::new(ExprIR::Literal {
                            value: LiteralIR::Int { value: 0 },
                        }),
                    },
                }],
                checkers: vec![MetricIR {
                    name: "exact".to_string(),
                    expr: ExprIR::Call {
                        function: "exact_match".to_string(),
                        args: vec![ExprIR::Record {
                            fields: vec![
                                ExprFieldIR {
                                    key: "actual".to_string(),
                                    value: ExprIR::FieldAccess {
                                        base: Box::new(ExprIR::Ident {
                                            name: "output".to_string(),
                                        }),
                                        field: "answer".to_string(),
                                    },
                                },
                                ExprFieldIR {
                                    key: "expected".to_string(),
                                    value: ExprIR::FieldAccess {
                                        base: Box::new(ExprIR::Ident {
                                            name: "expected".to_string(),
                                        }),
                                        field: "answer".to_string(),
                                    },
                                },
                            ],
                        }],
                    },
                }],
                judges: vec![
                    MetricIR {
                        name: "fluency".to_string(),
                        expr: ExprIR::Call {
                            function: "fluency_judge".to_string(),
                            args: vec![ExprIR::Record {
                                fields: vec![ExprFieldIR {
                                    key: "answer".to_string(),
                                    value: ExprIR::FieldAccess {
                                        base: Box::new(ExprIR::Ident {
                                            name: "output".to_string(),
                                        }),
                                        field: "answer".to_string(),
                                    },
                                }],
                            }],
                        },
                    },
                    MetricIR {
                        name: "utility".to_string(),
                        expr: ExprIR::Call {
                            function: "utility_judge".to_string(),
                            args: vec![ExprIR::Record {
                                fields: vec![ExprFieldIR {
                                    key: "answer".to_string(),
                                    value: ExprIR::FieldAccess {
                                        base: Box::new(ExprIR::Ident {
                                            name: "output".to_string(),
                                        }),
                                        field: "answer".to_string(),
                                    },
                                }],
                            }],
                        },
                    },
                ],
                metrics: vec![
                    MetricIR {
                        name: "combined".to_string(),
                        expr: ExprIR::Binary {
                            left: Box::new(ExprIR::Ident {
                                name: "fluency".to_string(),
                            }),
                            op: "+".to_string(),
                            right: Box::new(ExprIR::Ident {
                                name: "utility".to_string(),
                            }),
                        },
                    },
                    MetricIR {
                        name: "stage_count".to_string(),
                        expr: ExprIR::FieldAccess {
                            base: Box::new(ExprIR::Ident {
                                name: "rollout".to_string(),
                            }),
                            field: "stage_count".to_string(),
                        },
                    },
                    MetricIR {
                        name: "tool_call_count".to_string(),
                        expr: ExprIR::FieldAccess {
                            base: Box::new(ExprIR::Ident {
                                name: "rollout".to_string(),
                            }),
                            field: "tool_call_count".to_string(),
                        },
                    },
                    MetricIR {
                        name: "prompt_call_count".to_string(),
                        expr: ExprIR::FieldAccess {
                            base: Box::new(ExprIR::Ident {
                                name: "rollout".to_string(),
                            }),
                            field: "prompt_call_count".to_string(),
                        },
                    },
                    MetricIR {
                        name: "agent_turn_count".to_string(),
                        expr: ExprIR::FieldAccess {
                            base: Box::new(ExprIR::Ident {
                                name: "rollout".to_string(),
                            }),
                            field: "agent_turn_count".to_string(),
                        },
                    },
                    MetricIR {
                        name: "trace_stage_entries".to_string(),
                        expr: ExprIR::Call {
                            function: "len".to_string(),
                            args: vec![ExprIR::FieldAccess {
                                base: Box::new(ExprIR::FieldAccess {
                                    base: Box::new(ExprIR::Ident {
                                        name: "rollout".to_string(),
                                    }),
                                    field: "trace".to_string(),
                                }),
                                field: "stages".to_string(),
                            }],
                        },
                    },
                ],
                score: ExprIR::Ident {
                    name: "combined".to_string(),
                },
                split: None,
                select: Some(SelectIR {
                    primary: ExprIR::Ident {
                        name: "combined".to_string(),
                    },
                    tie_breakers: vec![ExprIR::Ident {
                        name: "exact".to_string(),
                    }],
                }),
            }],
            ..empty_ir()
        };

        let report = optimize_objective(&ir, "quality", Path::new("."), 1).unwrap();
        clear_mock_structured_sequence();

        assert_eq!(report.evaluated_candidates, 1);
        assert_eq!(report.best.train.metrics.get("present"), Some(&1.0));
        assert_eq!(report.best.train.metrics.get("exact"), Some(&1.0));
        assert_eq!(report.best.train.metrics.get("fluency"), Some(&0.75));
        assert_eq!(report.best.train.metrics.get("utility"), Some(&0.5));
        assert_eq!(report.best.train.metrics.get("combined"), Some(&1.25));
        assert_eq!(report.best.train.metrics.get("stage_count"), Some(&1.0));
        assert_eq!(report.best.train.metrics.get("tool_call_count"), Some(&2.0));
        assert_eq!(
            report.best.train.metrics.get("prompt_call_count"),
            Some(&1.0)
        );
        assert_eq!(
            report.best.train.metrics.get("agent_turn_count"),
            Some(&1.0)
        );
        assert_eq!(
            report.best.train.metrics.get("trace_stage_entries"),
            Some(&1.0)
        );
        assert_eq!(report.best.train.score, 1.25);
    }
}
