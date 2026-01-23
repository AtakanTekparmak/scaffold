//! Error types for the scaffold runtime

use thiserror::Error;

/// Runtime error types
#[derive(Debug, Error)]
pub enum Error {
    /// Timeout expired during subgoal execution
    #[error("timeout expired in subgoal '{0}'")]
    Timeout(String),

    /// Unknown action requested
    #[error("unknown action: {0}")]
    UnknownAction(String),

    /// Action execution failed
    #[error("action '{action}' failed: {message}")]
    ActionFailed { action: String, message: String },

    /// Precondition not satisfied
    #[error("precondition failed for subgoal '{0}'")]
    PreconditionFailed(String),

    /// Postcondition not satisfied
    #[error("postcondition failed for subgoal '{0}'")]
    PostconditionFailed(String),

    /// Task execution aborted
    #[error("task aborted: {0}")]
    Aborted(String),

    /// Maximum retries exceeded
    #[error("max retries ({count}) exceeded for subgoal '{subgoal}'")]
    MaxRetriesExceeded { subgoal: String, count: u64 },

    /// State access error
    #[error("state error: {0}")]
    StateError(String),

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

    /// Loop break control flow (not a real error)
    #[error("break")]
    LoopBreak,

    /// Loop continue control flow (not a real error)
    #[error("continue")]
    LoopContinue,
}

/// Result type alias using runtime Error
pub type Result<T> = std::result::Result<T, Error>;
