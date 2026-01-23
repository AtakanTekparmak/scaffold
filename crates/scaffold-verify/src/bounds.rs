//! Bounds analysis (stub - task system removed)
//!
//! This module is kept for API compatibility but is now a no-op.

/// Bounds analyzer (stub)
pub struct BoundsAnalyzer;

impl BoundsAnalyzer {
    pub fn new() -> Self {
        Self
    }
}

impl Default for BoundsAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of bounds analysis
#[derive(Debug, Clone)]
pub struct BoundsResult {
    pub bounded: bool,
    pub reason: Option<String>,
}

impl BoundsResult {
    pub fn new() -> Self {
        Self {
            bounded: true,
            reason: None,
        }
    }
}

impl Default for BoundsResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Termination checker (stub)
pub struct TerminationChecker;

impl TerminationChecker {
    pub fn new() -> Self {
        Self
    }
}

impl Default for TerminationChecker {
    fn default() -> Self {
        Self::new()
    }
}
