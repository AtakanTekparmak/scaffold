//! Error types for the interpreter

use thiserror::Error;

/// Result type for interpreter operations
pub type Result<T> = std::result::Result<T, InterpreterError>;

/// Interpreter error types
#[derive(Debug, Error)]
pub enum InterpreterError {
    /// File loading error
    #[error("Failed to load file: {0}")]
    LoadError(String),

    /// Parse error
    #[error("Parse error: {0}")]
    ParseError(String),

    /// Type error
    #[error("Type error: {0}")]
    TypeError(String),

    /// Task not found
    #[error("Task not found: {0}")]
    TaskNotFound(String),

    /// Tool not found
    #[error("Tool not found: {0}")]
    ToolNotFound(String),

    /// Runtime error
    #[error("Runtime error: {0}")]
    Runtime(String),

    /// Type mismatch during execution
    #[error("Type mismatch: expected {expected}, got {actual}")]
    TypeMismatch { expected: String, actual: String },

    /// Precondition failed
    #[error("Precondition failed: {0}")]
    PreconditionFailed(String),

    /// Postcondition failed
    #[error("Postcondition failed: {0}")]
    PostconditionFailed(String),

    /// Timeout
    #[error("Timeout: {0}")]
    Timeout(String),

    /// Shell command failed
    #[error("Shell command failed: {0}")]
    ShellError(String),

    /// LLM error
    #[error("LLM error: {0}")]
    LlmError(String),

    /// Foreign function error
    #[error("Foreign function error: {0}")]
    ForeignError(String),

    /// Variable not found
    #[error("Variable not found: {0}")]
    VariableNotFound(String),

    /// Field not found
    #[error("Field '{field}' not found in type '{type_name}'")]
    FieldNotFound { type_name: String, field: String },

    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Break control flow (not a real error)
    #[error("break")]
    Break,

    /// Continue control flow (not a real error)
    #[error("continue")]
    Continue,
}

impl From<scaffold_runtime::Error> for InterpreterError {
    fn from(e: scaffold_runtime::Error) -> Self {
        InterpreterError::Runtime(e.to_string())
    }
}
