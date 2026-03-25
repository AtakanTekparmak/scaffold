//! Error types for the scaffold v2 runtime

use thiserror::Error;

/// Runtime error types
#[derive(Debug, Error)]
pub enum Error {
    /// Timeout expired during execution
    #[error("timeout expired in '{0}'")]
    Timeout(String),

    /// Unknown node or graph
    #[error("unknown node or graph: {0}")]
    UnknownNode(String),

    /// Node execution failed
    #[error("node '{node}' failed: {message}")]
    NodeFailed { node: String, message: String },

    /// Step execution failed
    #[error("step '{step}' failed: {message}")]
    StepFailed { step: String, message: String },

    /// Graph execution aborted
    #[error("graph aborted: {0}")]
    Aborted(String),

    /// Maximum retries exceeded
    #[error("max retries ({count}) exceeded for step '{step}'")]
    MaxRetriesExceeded { step: String, count: u64 },

    /// Scope/variable access error
    #[error("scope error: {0}")]
    ScopeError(String),

    /// Type conversion error
    #[error("type error: expected {expected}, got {actual}")]
    TypeError { expected: String, actual: String },

    /// Invalid configuration
    #[error("configuration error: {0}")]
    ConfigError(String),

    /// IO error wrapper
    #[error("io error: {0}")]
    IoError(#[from] std::io::Error),

    /// Generic runtime error
    #[error("{0}")]
    Runtime(String),

    /// Serialization error
    #[error("serialization error: {0}")]
    SerializationError(String),

    /// Parse error
    #[error("parse error: {0}")]
    ParseError(String),

    /// Template rendering error
    #[error("template error: {0}")]
    TemplateError(String),

    /// Verify gate failed
    #[error("verification failed: {0}")]
    VerifyFailed(String),

    /// Expression evaluation error
    #[error("expression error: {0}")]
    ExprError(String),

    /// Shell command failed
    #[error("shell command failed: {0}")]
    ShellFailed(String),

    /// Action failed (used by builtins and shell)
    #[error("{action} failed: {message}")]
    ActionFailed { action: String, message: String },

    /// Emit missing (graph completed without emit)
    #[error("graph completed without emit")]
    NoEmit,
}

/// Result type alias using runtime Error
pub type Result<T> = std::result::Result<T, Error>;
