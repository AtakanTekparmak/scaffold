//! Tool support for rig integration
//!
//! This module provides types and helpers for implementing rig tools
//! in scaffold-generated code.

use std::fmt;

/// Error type for tool execution that implements std::error::Error
#[derive(Debug)]
pub struct ToolError {
    message: String,
}

impl ToolError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for ToolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for ToolError {}

impl From<crate::Error> for ToolError {
    fn from(err: crate::Error) -> Self {
        Self::new(err.to_string())
    }
}

impl From<std::io::Error> for ToolError {
    fn from(err: std::io::Error) -> Self {
        Self::new(err.to_string())
    }
}

impl From<serde_json::Error> for ToolError {
    fn from(err: serde_json::Error) -> Self {
        Self::new(err.to_string())
    }
}

/// Helper trait for creating rig-compatible tool definitions
pub trait ToolSchema {
    /// Get the JSON schema for the input type
    fn input_schema() -> serde_json::Value;
}
