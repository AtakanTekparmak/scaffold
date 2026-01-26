//! File loader for scaffold files
//!
//! Handles parsing, type checking, and IR lowering.

use crate::error::{InterpreterError, Result};
use scaffold_ir::{Lowerer, ScaffoldIR};
use scaffold_syntax::parse;
use scaffold_types::check;
use scaffold_verify::verify;
use std::path::Path;

/// Loader for scaffold files
pub struct Loader {
    /// Whether to perform verification
    verify_enabled: bool,
}

impl Default for Loader {
    fn default() -> Self {
        Self::new()
    }
}

impl Loader {
    /// Create a new loader
    pub fn new() -> Self {
        Self {
            verify_enabled: false,
        }
    }

    /// Enable verification
    pub fn with_verify(mut self, verify: bool) -> Self {
        self.verify_enabled = verify;
        self
    }

    /// Load a scaffold file and return IR
    pub fn load(&self, path: impl AsRef<Path>) -> Result<ScaffoldIR> {
        let path = path.as_ref();

        // Read file
        let source = std::fs::read_to_string(path)
            .map_err(|e| InterpreterError::LoadError(format!("{}: {}", path.display(), e)))?;

        self.load_source(&source, path.to_string_lossy().as_ref())
    }

    /// Load from source string
    pub fn load_source(&self, source: &str, filename: &str) -> Result<ScaffoldIR> {
        // Parse
        let ast = parse(source).map_err(|e| InterpreterError::ParseError(e.message))?;

        // Type check
        let type_env = check(&ast).map_err(|errors| {
            let error_msgs: Vec<String> = errors.iter().map(|e| e.message.clone()).collect();
            InterpreterError::TypeError(error_msgs.join("\n"))
        })?;

        // Run verification
        let verify_result = verify(&ast, &type_env);

        // Check for verification errors
        if self.verify_enabled && verify_result.has_errors() {
            let error_msgs: Vec<String> = verify_result
                .errors
                .iter()
                .map(|e| e.message.clone())
                .collect();
            return Err(InterpreterError::Runtime(format!(
                "Verification failed:\n{}",
                error_msgs.join("\n")
            )));
        }

        // Lower to IR
        let lowerer = Lowerer::new().with_source_file(filename.to_string());
        let ir = lowerer
            .lower(&ast, &type_env)
            .map_err(|e| InterpreterError::Runtime(format!("IR lowering failed: {}", e.message)))?;

        Ok(ir)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_simple() {
        // A complete scaffold program with tools
        let source = r#"
            type Position = { x: int, y: int }

            tool echo {
                input: string
                output: string
                impl: shell("echo {input}")
            }

            agent test_agent {
                input: Position
                output: bool
                tools: [echo]
                system: "Test agent"
            }
        "#;

        let loader = Loader::new();
        let ir = loader.load_source(source, "test.scaffold").unwrap();

        assert_eq!(ir.types.len(), 1);
        assert_eq!(ir.tools.len(), 1);
        assert_eq!(ir.tools[0].name, "echo");
        assert_eq!(ir.agents.len(), 1);
        assert_eq!(ir.agents[0].name, "test_agent");
    }
}
