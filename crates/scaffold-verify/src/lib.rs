//! Verification layer for the Scaffold DSL
//!
//! This module provides static analysis capabilities.
//! Note: The task-based verification (deadlock detection, reachability, bounds)
//! has been removed as part of the task system simplification.
//! Verification is now a no-op but the module is kept for API compatibility.

pub mod bounds;
pub mod deadlock;
pub mod reachability;

use scaffold_syntax::ast::*;
use scaffold_syntax::Span;
use scaffold_types::TypeEnv;

pub use bounds::{BoundsAnalyzer, BoundsResult, TerminationChecker};
pub use deadlock::{DeadlockAnalyzer, DeadlockResult};
pub use reachability::{ReachabilityAnalyzer, ReachabilityResult};

/// Verification error
#[derive(Debug, Clone)]
pub struct VerifyError {
    pub message: String,
    pub span: Span,
    pub severity: Severity,
}

impl VerifyError {
    pub fn new(message: impl Into<String>, span: Span) -> Self {
        Self {
            message: message.into(),
            span,
            severity: Severity::Error,
        }
    }

    pub fn warning(message: impl Into<String>, span: Span) -> Self {
        Self {
            message: message.into(),
            span,
            severity: Severity::Warning,
        }
    }
}

/// Severity level for verification messages
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
}

/// Result of verification including all checks
#[derive(Debug, Clone)]
pub struct VerifyResult {
    /// All verification errors found
    pub errors: Vec<VerifyError>,
    /// Reachability results (kept for API compatibility)
    pub reachability: Vec<(String, ReachabilityResult)>,
    /// Deadlock results (kept for API compatibility)
    pub deadlock: Vec<(String, DeadlockResult)>,
    /// Bounds results (kept for API compatibility)
    pub bounds: Vec<(String, BoundsResult)>,
}

impl VerifyResult {
    pub fn new() -> Self {
        Self {
            errors: Vec::new(),
            reachability: Vec::new(),
            deadlock: Vec::new(),
            bounds: Vec::new(),
        }
    }

    pub fn has_errors(&self) -> bool {
        self.errors.iter().any(|e| e.severity == Severity::Error)
    }

    pub fn has_warnings(&self) -> bool {
        self.errors.iter().any(|e| e.severity == Severity::Warning)
    }
}

impl Default for VerifyResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Main verifier that runs all checks
pub struct Verifier;

impl Verifier {
    pub fn new() -> Self {
        Self
    }

    /// Verify a program with the given type environment
    /// Currently a no-op since task-based verification was removed
    pub fn verify(&mut self, _program: &Program, _type_env: &TypeEnv) -> VerifyResult {
        VerifyResult::new()
    }
}

impl Default for Verifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Verify a program and return the result
pub fn verify(program: &Program, type_env: &TypeEnv) -> VerifyResult {
    let mut verifier = Verifier::new();
    verifier.verify(program, type_env)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scaffold_syntax::parse;
    use scaffold_types::check;

    #[test]
    fn test_verify_simple_program() {
        let source = r#"
            type Position = { x: int, y: int }

            tool get_pos {
                input: { id: int }
                output: Position
            }
        "#;

        let program = parse(source).unwrap();
        let type_env = check(&program).unwrap();
        let result = verify(&program, &type_env);

        // Should have no errors
        assert!(!result.has_errors());
    }
}
