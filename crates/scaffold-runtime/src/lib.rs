//! Scaffold Runtime Library
//!
//! This crate provides the runtime support for scaffold programs.
//!
//! # Types
//!
//! - [`Value`] - Dynamic value type for runtime data
//! - [`ExecutionState`] - State management during task execution
//! - [`TaskContext`] - Context passed to task execution
//!
//! # Configuration
//!
//! API keys and settings can be configured via:
//! - Environment variables (OPENAI_API_KEY, ANTHROPIC_API_KEY)
//! - Config file (~/.scaffold/config.toml)
//!
//! # Example
//!
//! ```ignore
//! use scaffold_runtime::prelude::*;
//!
//! // Execute a shell command
//! let output = scaffold_runtime::shell::execute("echo hello")?;
//!
//! // Query an LLM
//! let response = scaffold_runtime::llm_query("What is 2+2?").await?;
//! ```

pub mod config;
pub mod error;
pub mod llm;
pub mod prompt;
pub mod parse;
pub mod shell;
pub mod state;
pub mod task;
pub mod tool;
pub mod trace;
pub mod value;

// Re-exports for convenience
pub use config::{config, Config};
pub use error::{Error, Result};
pub use llm::{query as llm_query, query_with_model, query_with_config, query_structured, Agent, AgentBuilder, LlmConfig, LlmBackend};
pub use prompt::PromptManager;
pub use state::{ExecutionState, StateCheckpoint};
pub use task::{Deadline, FailureStrategy, SubgoalResult, TaskContext};
pub use tool::ToolError;
pub use rig::completion::request::ToolDefinition;
pub use trace::{tracer, Tracer, TracerConfig, TraceEvent, TraceLevel, TraceOutput, TraceRecord};
pub use value::{Value, ResultValue};

// Re-export rig types for generated code
pub use rig;

/// Prelude module for common imports
pub mod prelude {
    pub use crate::error::{Error, Result};
    pub use crate::state::ExecutionState;
    pub use crate::task::{Deadline, FailureStrategy, SubgoalResult, TaskContext};
    pub use crate::tool::ToolError;
    pub use rig::completion::request::ToolDefinition;
    pub use crate::value::Value;

    // Re-export rig tool trait for implementations
    pub use rig::tool::Tool;
}
