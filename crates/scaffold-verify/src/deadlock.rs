//! Deadlock detection (stub - task system removed)
//!
//! This module is kept for API compatibility but is now a no-op.

/// Result of deadlock analysis
#[derive(Debug, Clone)]
pub struct DeadlockResult {
    /// Whether the decomposition has potential deadlocks
    pub has_deadlock: bool,
    /// List of cycles found (empty if no deadlock)
    pub cycles: Vec<Vec<String>>,
}

impl DeadlockResult {
    pub fn new() -> Self {
        Self {
            has_deadlock: false,
            cycles: Vec::new(),
        }
    }
}

impl Default for DeadlockResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Deadlock analyzer (stub)
pub struct DeadlockAnalyzer;

impl DeadlockAnalyzer {
    pub fn new() -> Self {
        Self
    }

    /// Detect deadlocks in the decomposition graph
    pub fn detect_deadlocks(&self) -> DeadlockResult {
        DeadlockResult::new()
    }
}

impl Default for DeadlockAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}
