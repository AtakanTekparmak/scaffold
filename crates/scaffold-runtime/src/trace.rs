//! Tracing and metrics for scaffold execution
//!
//! This module provides structured tracing for:
//! - LLM calls (prompts, responses, token usage)
//! - Tool executions (inputs, outputs, duration)
//! - Agent turns (tool calls, intermediate results)
//! - Task/subgoal completion (rewards, status)
//!
//! # Usage
//!
//! Enable tracing with `SCAFFOLD_TRACE=1` for JSONL, or `SCAFFOLD_TRACE_PRETTY=1`
//! / `--live` for human-readable progress logs.
//! Traces are emitted to stderr or a file.
//!
//! ```ignore
//! use scaffold_runtime::trace::{Tracer, TraceEvent};
//!
//! let tracer = Tracer::new();
//! tracer.record(TraceEvent::LlmCall { ... });
//! ```

use serde::{Deserialize, Serialize};
use std::fmt::Write as _;
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Global tracer state
static TRACER: std::sync::OnceLock<Tracer> = std::sync::OnceLock::new();

/// Get the global tracer instance
pub fn tracer() -> &'static Tracer {
    TRACER.get_or_init(Tracer::new)
}

/// Initialize the global tracer with custom configuration
pub fn init_tracer(config: TracerConfig) {
    let _ = TRACER.set(Tracer::with_config(config));
}

/// Tracer configuration
#[derive(Clone, Debug)]
pub struct TracerConfig {
    /// Whether tracing is enabled
    pub enabled: bool,
    /// Output destination (stderr, file path, or callback)
    pub output: TraceOutput,
    /// Whether to include full request/response bodies
    pub include_bodies: bool,
    /// Minimum event level to record
    pub min_level: TraceLevel,
    /// Output format
    pub format: TraceFormat,
}

impl Default for TracerConfig {
    fn default() -> Self {
        let trace_format = match std::env::var("SCAFFOLD_TRACE_FORMAT") {
            Ok(value) if value.eq_ignore_ascii_case("pretty") => TraceFormat::Pretty,
            _ if std::env::var("SCAFFOLD_TRACE_PRETTY").is_ok() => TraceFormat::Pretty,
            _ => TraceFormat::Json,
        };
        Self {
            enabled: std::env::var("SCAFFOLD_TRACE").is_ok()
                || matches!(trace_format, TraceFormat::Pretty),
            output: TraceOutput::Stderr,
            include_bodies: true,
            min_level: TraceLevel::Info,
            format: trace_format,
        }
    }
}

/// Trace output formatting
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TraceFormat {
    Json,
    Pretty,
}

/// Trace output destination
#[derive(Clone, Debug)]
pub enum TraceOutput {
    /// Write to stderr
    Stderr,
    /// Write to a file
    File(String),
    /// Collect in memory (for testing)
    Memory,
}

/// Trace event severity level
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TraceLevel {
    Debug,
    Info,
    Warn,
    Error,
}

/// A recorded trace event
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TraceRecord {
    /// Unix timestamp in milliseconds
    pub timestamp_ms: u64,
    /// Event level
    pub level: TraceLevel,
    /// Span ID for correlation
    pub span_id: String,
    /// Parent span ID (for nested events)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_span_id: Option<String>,
    /// The event data
    pub event: TraceEvent,
    /// Duration in milliseconds (for completed events)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
}

/// Types of trace events
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TraceEvent {
    /// LLM call started/completed
    LlmCall {
        model: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        graph_name: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        step_name: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        prompt_name: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        split: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        case_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        repeat: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        prompt: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        response: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        input_tokens: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        output_tokens: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    /// LLM call heartbeat while waiting for a response
    LlmHeartbeat {
        model: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        graph_name: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        step_name: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        prompt_name: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        split: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        case_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        repeat: Option<u64>,
        elapsed_ms: u64,
        #[serde(skip_serializing_if = "Option::is_none")]
        timeout_secs: Option<u64>,
    },
    /// Tool execution
    ToolCall {
        tool_name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        input: Option<serde_json::Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        output: Option<serde_json::Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    /// Agent turn (one iteration of the agent loop)
    AgentTurn {
        agent_name: String,
        turn_number: u64,
        #[serde(skip_serializing_if = "Option::is_none")]
        tool_calls: Option<Vec<String>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        completed: Option<bool>,
    },
    /// Prompt execution
    PromptExecution {
        #[serde(skip_serializing_if = "Option::is_none")]
        graph_name: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        step_name: Option<String>,
        prompt_name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        split: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        case_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        repeat: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        input: Option<serde_json::Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        output: Option<serde_json::Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    /// Task/subgoal completion
    TaskComplete {
        graph_name: String,
        status: TaskStatus,
        #[serde(skip_serializing_if = "Option::is_none")]
        reward: Option<f64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    /// Custom metric
    Metric {
        name: String,
        value: MetricValue,
        #[serde(skip_serializing_if = "Option::is_none")]
        tags: Option<std::collections::HashMap<String, String>>,
    },
    /// Objective evaluation or optimization progress
    ObjectiveProgress {
        objective_name: String,
        phase: ObjectiveProgressPhase,
        #[serde(skip_serializing_if = "Option::is_none")]
        split: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        case_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        repeat: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        candidate_index: Option<usize>,
        #[serde(skip_serializing_if = "Option::is_none")]
        candidate_total: Option<usize>,
        #[serde(skip_serializing_if = "Option::is_none")]
        assignments: Option<serde_json::Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        success: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        score: Option<f64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        train_primary: Option<f64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        val_primary: Option<f64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        test_primary: Option<f64>,
    },
}

/// Objective progress phase
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObjectiveProgressPhase {
    CandidateStarted,
    CandidateCompleted,
    EarlyStopped,
    RolloutStarted,
    RolloutCompleted,
}

/// Task completion status
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    Completed,
    Failed,
    Timeout,
    Skipped,
}

/// Metric value types
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MetricValue {
    Counter(i64),
    Gauge(f64),
    Histogram(Vec<f64>),
}

/// The main tracer struct
#[derive(Debug)]
pub struct Tracer {
    config: TracerConfig,
    /// In-memory trace storage (for testing/inspection)
    records: Mutex<Vec<TraceRecord>>,
}

impl Tracer {
    /// Create a new tracer with default configuration
    pub fn new() -> Self {
        Self::with_config(TracerConfig::default())
    }

    /// Create a new tracer with custom configuration
    pub fn with_config(config: TracerConfig) -> Self {
        Self {
            config,
            records: Mutex::new(Vec::new()),
        }
    }

    /// Check if tracing is enabled
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Record a trace event
    pub fn record(&self, event: TraceEvent) -> String {
        self.record_with_level(TraceLevel::Info, event)
    }

    /// Record a trace event with a specific level
    pub fn record_with_level(&self, level: TraceLevel, event: TraceEvent) -> String {
        if !self.config.enabled || level < self.config.min_level {
            return String::new();
        }

        let span_id = generate_span_id();
        let record = TraceRecord {
            timestamp_ms: current_timestamp_ms(),
            level,
            span_id: span_id.clone(),
            parent_span_id: None,
            event,
            duration_ms: None,
        };

        self.emit(&record);
        span_id
    }

    /// Record a trace event with duration
    pub fn record_completed(&self, span_id: &str, event: TraceEvent, duration: Duration) {
        if !self.config.enabled {
            return;
        }

        let record = TraceRecord {
            timestamp_ms: current_timestamp_ms(),
            level: TraceLevel::Info,
            span_id: span_id.to_string(),
            parent_span_id: None,
            event,
            duration_ms: Some(duration.as_millis() as u64),
        };

        self.emit(&record);
    }

    /// Start a span and return a guard that records completion
    pub fn span(&self, event: TraceEvent) -> SpanGuard<'_> {
        let span_id = self.record(event);
        SpanGuard {
            tracer: self,
            span_id,
            start: Instant::now(),
        }
    }

    /// Emit a trace record to the configured output
    fn emit(&self, record: &TraceRecord) {
        // Store in memory if configured
        if matches!(self.config.output, TraceOutput::Memory) {
            if let Ok(mut records) = self.records.lock() {
                records.push(record.clone());
            }
        }

        let rendered = match self.config.format {
            TraceFormat::Json => match serde_json::to_string(record) {
                Ok(json) => json,
                Err(_) => return,
            },
            TraceFormat::Pretty => self.format_pretty(record),
        };

        match &self.config.output {
            TraceOutput::Stderr => {
                eprintln!("{}", rendered);
            }
            TraceOutput::File(path) => {
                use std::io::Write;
                if let Ok(mut file) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                {
                    let _ = writeln!(file, "{}", rendered);
                }
            }
            TraceOutput::Memory => {
                // Already stored above
            }
        }
    }

    fn format_pretty(&self, record: &TraceRecord) -> String {
        let level = match record.level {
            TraceLevel::Debug => "debug",
            TraceLevel::Info => "info",
            TraceLevel::Warn => "warn",
            TraceLevel::Error => "error",
        };
        let duration = record
            .duration_ms
            .map(|ms| format!(" in {}ms", ms))
            .unwrap_or_default();

        match &record.event {
            TraceEvent::PromptExecution {
                graph_name,
                step_name,
                prompt_name,
                split,
                case_id,
                repeat,
                input,
                output,
                error,
            } => {
                if let Some(error) = error {
                    format!(
                        "[{level}] prompt {prompt_name} failed{duration}{}: {error}",
                        summarize_execution_context(
                            graph_name.as_deref(),
                            step_name.as_deref(),
                            Some(prompt_name),
                            split.as_deref(),
                            case_id.as_deref(),
                            *repeat,
                        )
                    )
                } else if output.is_some() {
                    let mut line = format!(
                        "[{level}] prompt {prompt_name} completed{duration}{}",
                        summarize_execution_context(
                            graph_name.as_deref(),
                            step_name.as_deref(),
                            Some(prompt_name),
                            split.as_deref(),
                            case_id.as_deref(),
                            *repeat,
                        )
                    );
                    if self.config.include_bodies {
                        if let Some(output) = output {
                            write!(&mut line, " output={}", summarize_json(output, 160)).ok();
                        }
                    }
                    line
                } else {
                    let mut line = format!(
                        "[{level}] prompt {prompt_name} started{}",
                        summarize_execution_context(
                            graph_name.as_deref(),
                            step_name.as_deref(),
                            Some(prompt_name),
                            split.as_deref(),
                            case_id.as_deref(),
                            *repeat,
                        )
                    );
                    if self.config.include_bodies {
                        if let Some(input) = input {
                            write!(&mut line, " input={}", summarize_json(input, 160)).ok();
                        }
                    }
                    line
                }
            }
            TraceEvent::LlmCall {
                model,
                graph_name,
                step_name,
                prompt_name,
                split,
                case_id,
                repeat,
                prompt,
                response,
                error,
                ..
            } => {
                if let Some(error) = error {
                    format!(
                        "[{level}] llm {model} failed{duration}{}: {error}",
                        summarize_execution_context(
                            graph_name.as_deref(),
                            step_name.as_deref(),
                            prompt_name.as_deref(),
                            split.as_deref(),
                            case_id.as_deref(),
                            *repeat,
                        )
                    )
                } else if response.is_some() {
                    let mut line = format!(
                        "[{level}] llm {model} completed{duration}{}",
                        summarize_execution_context(
                            graph_name.as_deref(),
                            step_name.as_deref(),
                            prompt_name.as_deref(),
                            split.as_deref(),
                            case_id.as_deref(),
                            *repeat,
                        )
                    );
                    if self.config.include_bodies {
                        if let Some(response) = response {
                            write!(&mut line, " response={}", summarize_text(response, 160)).ok();
                        }
                    }
                    line
                } else {
                    let mut line = format!(
                        "[{level}] llm {model} started{}",
                        summarize_execution_context(
                            graph_name.as_deref(),
                            step_name.as_deref(),
                            prompt_name.as_deref(),
                            split.as_deref(),
                            case_id.as_deref(),
                            *repeat,
                        )
                    );
                    if self.config.include_bodies {
                        if let Some(prompt) = prompt {
                            write!(&mut line, " prompt={}", summarize_text(prompt, 160)).ok();
                        }
                    }
                    line
                }
            }
            TraceEvent::LlmHeartbeat {
                model,
                graph_name,
                step_name,
                prompt_name,
                split,
                case_id,
                repeat,
                elapsed_ms,
                timeout_secs,
            } => {
                let mut line = format!(
                    "[{level}] llm {model} still running after {}s{}",
                    elapsed_ms / 1000,
                    summarize_execution_context(
                        graph_name.as_deref(),
                        step_name.as_deref(),
                        prompt_name.as_deref(),
                        split.as_deref(),
                        case_id.as_deref(),
                        *repeat,
                    )
                );
                if let Some(timeout_secs) = timeout_secs {
                    write!(&mut line, " (timeout={}s)", timeout_secs).ok();
                }
                line
            }
            TraceEvent::ToolCall {
                tool_name,
                input,
                output,
                error,
            } => {
                if let Some(error) = error {
                    format!("[{level}] tool {tool_name} failed{duration}: {error}")
                } else if output.is_some() {
                    let mut line = format!("[{level}] tool {tool_name} completed{duration}");
                    if self.config.include_bodies {
                        if let Some(output) = output {
                            write!(&mut line, " output={}", summarize_json(output, 160)).ok();
                        }
                    }
                    line
                } else {
                    let mut line = format!("[{level}] tool {tool_name} started");
                    if self.config.include_bodies {
                        if let Some(input) = input {
                            write!(&mut line, " input={}", summarize_json(input, 160)).ok();
                        }
                    }
                    line
                }
            }
            TraceEvent::AgentTurn {
                agent_name,
                turn_number,
                completed,
                ..
            } => {
                if completed.is_some() || record.duration_ms.is_some() {
                    format!("[{level}] agent {agent_name} turn {turn_number} completed{duration}")
                } else {
                    format!("[{level}] agent {agent_name} turn {turn_number} started")
                }
            }
            TraceEvent::TaskComplete {
                graph_name,
                status,
                reward,
                error,
            } => {
                let status = match status {
                    TaskStatus::Completed => "completed",
                    TaskStatus::Failed => "failed",
                    TaskStatus::Timeout => "timed_out",
                    TaskStatus::Skipped => "skipped",
                };
                let mut line = format!("[{level}] task {graph_name} {status}");
                if let Some(reward) = reward {
                    write!(&mut line, " reward={reward:.3}").ok();
                }
                if let Some(error) = error {
                    write!(&mut line, ": {error}").ok();
                }
                line
            }
            TraceEvent::Metric { name, value, .. } => {
                format!("[{level}] metric {name}={}", summarize_metric(value))
            }
            TraceEvent::ObjectiveProgress {
                objective_name,
                phase,
                split,
                case_id,
                repeat,
                candidate_index,
                candidate_total,
                assignments,
                success,
                score,
                train_primary,
                val_primary,
                test_primary,
            } => match phase {
                ObjectiveProgressPhase::CandidateStarted => {
                    let label = candidate_label(*candidate_index, *candidate_total);
                    let assignments = assignments
                        .as_ref()
                        .map(summarize_assignments)
                        .unwrap_or_else(|| "{}".to_string());
                    format!(
                        "[{level}] optimize {objective_name} {label} started assignments={assignments}"
                    )
                }
                ObjectiveProgressPhase::CandidateCompleted => {
                    let label = candidate_label(*candidate_index, *candidate_total);
                    let mut line = format!("[{level}] optimize {objective_name} {label} completed");
                    if let Some(score) = score {
                        write!(&mut line, " score={score:.3}").ok();
                    }
                    if let Some(train) = train_primary {
                        write!(&mut line, " train={train:.3}").ok();
                    }
                    if let Some(val) = val_primary {
                        write!(&mut line, " val={val:.3}").ok();
                    }
                    if let Some(test) = test_primary {
                        write!(&mut line, " test={test:.3}").ok();
                    }
                    line
                }
                ObjectiveProgressPhase::EarlyStopped => {
                    let label = candidate_label(*candidate_index, *candidate_total);
                    let mut line =
                        format!("[{level}] optimize {objective_name} {label} early-stopped");
                    if let Some(score) = score {
                        write!(&mut line, " score={score:.3}").ok();
                    }
                    if let Some(train) = train_primary {
                        write!(&mut line, " train={train:.3}").ok();
                    }
                    if let Some(val) = val_primary {
                        write!(&mut line, " val={val:.3}").ok();
                    }
                    if let Some(test) = test_primary {
                        write!(&mut line, " test={test:.3}").ok();
                    }
                    line
                }
                ObjectiveProgressPhase::RolloutStarted => format!(
                    "[{level}] rollout {objective_name} split={} case={} repeat={}",
                    split.as_deref().unwrap_or("unknown"),
                    case_id.as_deref().unwrap_or("<none>"),
                    repeat.unwrap_or(0)
                ),
                ObjectiveProgressPhase::RolloutCompleted => {
                    let mut line = format!(
                        "[{level}] rollout {objective_name} split={} case={} repeat={}",
                        split.as_deref().unwrap_or("unknown"),
                        case_id.as_deref().unwrap_or("<none>"),
                        repeat.unwrap_or(0)
                    );
                    if let Some(success) = success {
                        write!(&mut line, " success={success}").ok();
                    }
                    if let Some(score) = score {
                        write!(&mut line, " primary={score:.3}").ok();
                    }
                    line
                }
            },
        }
    }

    /// Get all recorded traces (for testing)
    pub fn get_records(&self) -> Vec<TraceRecord> {
        self.records.lock().map(|r| r.clone()).unwrap_or_default()
    }

    /// Clear recorded traces
    pub fn clear(&self) {
        if let Ok(mut records) = self.records.lock() {
            records.clear();
        }
    }
}

fn summarize_execution_context(
    graph_name: Option<&str>,
    step_name: Option<&str>,
    prompt_name: Option<&str>,
    split: Option<&str>,
    case_id: Option<&str>,
    repeat: Option<u64>,
) -> String {
    let mut parts = Vec::new();
    if let Some(graph_name) = graph_name {
        if !graph_name.is_empty() {
            parts.push(format!("graph={graph_name}"));
        }
    }
    if let Some(step_name) = step_name {
        if !step_name.is_empty() {
            parts.push(format!("step={step_name}"));
        }
    }
    if let Some(prompt_name) = prompt_name {
        if !prompt_name.is_empty() {
            parts.push(format!("prompt={prompt_name}"));
        }
    }
    if let Some(split) = split {
        parts.push(format!("split={split}"));
    }
    if let Some(case_id) = case_id {
        parts.push(format!("case={case_id}"));
    }
    if let Some(repeat) = repeat {
        parts.push(format!("repeat={repeat}"));
    }
    if parts.is_empty() {
        String::new()
    } else {
        format!(" {}", parts.join(" "))
    }
}

impl Default for Tracer {
    fn default() -> Self {
        Self::new()
    }
}

/// RAII guard for span timing
pub struct SpanGuard<'a> {
    tracer: &'a Tracer,
    span_id: String,
    start: Instant,
}

impl<'a> SpanGuard<'a> {
    /// Get the span ID
    pub fn span_id(&self) -> &str {
        &self.span_id
    }

    /// Complete the span with an event
    pub fn complete(self, event: TraceEvent) {
        let duration = self.start.elapsed();
        self.tracer.record_completed(&self.span_id, event, duration);
    }
}

/// Generate a unique span ID
fn generate_span_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);

    let count = COUNTER.fetch_add(1, Ordering::SeqCst);
    let timestamp = current_timestamp_ms();
    format!("{:x}-{:04x}", timestamp, count & 0xFFFF)
}

/// Get current timestamp in milliseconds
fn current_timestamp_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn summarize_text(text: &str, max_len: usize) -> String {
    let squashed = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if squashed.len() <= max_len {
        squashed
    } else {
        let mut end = max_len;
        while end > 0 && !squashed.is_char_boundary(end) {
            end -= 1;
        }
        format!("{}...", &squashed[..end])
    }
}

fn summarize_json(value: &serde_json::Value, max_len: usize) -> String {
    summarize_text(&value.to_string(), max_len)
}

fn summarize_metric(value: &MetricValue) -> String {
    match value {
        MetricValue::Counter(v) => v.to_string(),
        MetricValue::Gauge(v) => format!("{v:.3}"),
        MetricValue::Histogram(values) => format!("hist[{}]", values.len()),
    }
}

fn candidate_label(index: Option<usize>, total: Option<usize>) -> String {
    match (index, total) {
        (Some(index), Some(total)) => format!("candidate {index}/{total}"),
        (Some(index), None) => format!("candidate {index}"),
        _ => "candidate".to_string(),
    }
}

fn summarize_assignments(value: &serde_json::Value) -> String {
    if let serde_json::Value::Object(map) = value {
        let mut parts = map
            .iter()
            .map(|(key, value)| format!("{key}={}", summarize_json(value, 48)))
            .collect::<Vec<_>>();
        parts.sort();
        if parts.is_empty() {
            "{}".to_string()
        } else {
            parts.join(", ")
        }
    } else {
        summarize_json(value, 120)
    }
}

/// Convenience macro for tracing LLM calls
#[macro_export]
macro_rules! trace_llm {
    ($model:expr, $prompt:expr) => {
        $crate::trace::tracer().record($crate::trace::TraceEvent::LlmCall {
            model: $model.to_string(),
            graph_name: None,
            step_name: None,
            prompt_name: None,
            split: None,
            case_id: None,
            repeat: None,
            prompt: Some($prompt.to_string()),
            response: None,
            input_tokens: None,
            output_tokens: None,
            error: None,
        })
    };
}

/// Convenience macro for tracing tool calls
#[macro_export]
macro_rules! trace_tool {
    ($name:expr, $input:expr) => {
        $crate::trace::tracer().record($crate::trace::TraceEvent::ToolCall {
            tool_name: $name.to_string(),
            input: Some(serde_json::to_value($input).unwrap_or_default()),
            output: None,
            error: None,
        })
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tracer_disabled_by_default() {
        let _tracer = Tracer::new();
        // Without SCAFFOLD_TRACE env var, tracing should be disabled
        // (depends on env during test)
    }

    #[test]
    fn test_tracer_memory_output() {
        let tracer = Tracer::with_config(TracerConfig {
            enabled: true,
            output: TraceOutput::Memory,
            include_bodies: true,
            min_level: TraceLevel::Debug,
            format: TraceFormat::Json,
        });

        tracer.record(TraceEvent::ToolCall {
            tool_name: "test_tool".to_string(),
            input: Some(serde_json::json!({"x": 1})),
            output: None,
            error: None,
        });

        let records = tracer.get_records();
        assert_eq!(records.len(), 1);
        assert!(
            matches!(&records[0].event, TraceEvent::ToolCall { tool_name, .. } if tool_name == "test_tool")
        );
    }

    #[test]
    fn test_trace_record_serialization() {
        let record = TraceRecord {
            timestamp_ms: 1234567890,
            level: TraceLevel::Info,
            span_id: "abc-123".to_string(),
            parent_span_id: None,
            event: TraceEvent::LlmCall {
                model: "gpt-4".to_string(),
                graph_name: None,
                step_name: None,
                prompt_name: None,
                split: None,
                case_id: None,
                repeat: None,
                prompt: Some("Hello".to_string()),
                response: Some("Hi there!".to_string()),
                input_tokens: Some(10),
                output_tokens: Some(5),
                error: None,
            },
            duration_ms: Some(150),
        };

        let json = serde_json::to_string(&record).unwrap();
        assert!(json.contains("llm_call"));
        assert!(json.contains("gpt-4"));
    }
}
