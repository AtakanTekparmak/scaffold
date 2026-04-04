//! Scaffold v2 Runtime Library
//!
//! This crate provides the runtime support for executing scaffold v2 programs.
//!
//! # Architecture
//!
//! - [`executor::GraphExecutor`] — walks IR graph statements, dispatches nodes
//! - [`node_runner`] — runs individual nodes (prompt, tool, agent, verify)
//! - [`scope::Scope`] — variable binding management with parent chain
//! - [`value::Value`] — dynamic value type for runtime data
//!
//! # Configuration
//!
//! API keys and settings can be configured via:
//! - Environment variables (OPENAI_API_KEY, ANTHROPIC_API_KEY)
//! - Config file (~/.scaffold/config.toml)

pub mod agent_convos;
pub mod builtins;
pub mod config;
pub mod error;
pub mod example_bank;
pub mod executor;
pub mod llm;
pub mod meta_agent;
pub mod motifs;
pub mod mutations;
pub mod node_runner;
pub mod optimizer;
pub mod parse;
pub mod prompt;
pub mod scope;
pub mod shell;
pub mod trace;
pub mod value;

// Re-exports for convenience
pub use agent_convos::maybe_log_agent_conversation;
pub use config::{config, Config};
pub use error::{Error, Result};
pub use executor::GraphExecutor;
pub use llm::{
    query as llm_query, query_structured, query_structured_with_config, query_with_config,
    query_with_model, Agent, AgentBuilder, LlmBackend, LlmConfig,
};
pub use mutations::Mutation;
pub use optimizer::{
    eval_checker_expr, optimize, optimize_hierarchical, HierarchicalReport, OptEvent, OptPhase,
    OptimizationBackend, OptimizationOptions, OptimizationReport, SubOptimizationReport,
};
pub use prompt::PromptManager;
pub use scope::Scope;
pub use trace::{
    tracer, ObjectiveProgressPhase, TraceEvent, TraceFormat, TraceLevel, TraceOutput, TraceRecord,
    Tracer, TracerConfig,
};
pub use value::{ResultValue, Value};

/// Prelude module for common imports
pub mod prelude {
    pub use crate::error::{Error, Result};
    pub use crate::executor::GraphExecutor;
    pub use crate::scope::Scope;
    pub use crate::value::Value;
}
