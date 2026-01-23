//! Reachability analysis (stub - task system removed)
//!
//! This module is kept for API compatibility but is now a no-op.

/// Result of reachability analysis
#[derive(Debug, Clone)]
pub struct ReachabilityResult {
    /// Whether the target is reachable
    pub reachable: bool,
    /// Path to reach the target (if reachable)
    pub path: Vec<String>,
    /// Reason for unreachability (if not reachable)
    pub reason: Option<String>,
}

impl ReachabilityResult {
    pub fn new() -> Self {
        Self {
            reachable: true,
            path: Vec::new(),
            reason: None,
        }
    }
}

impl Default for ReachabilityResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Reachability analyzer (stub)
pub struct ReachabilityAnalyzer;

impl ReachabilityAnalyzer {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ReachabilityAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}
