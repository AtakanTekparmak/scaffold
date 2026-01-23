//! Task execution context and related types

use crate::error::Result;
use crate::state::ExecutionState;
use std::time::{Duration, Instant};

/// Result of subgoal execution
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubgoalResult {
    /// Subgoal completed successfully
    Completed,
    /// Subgoal was skipped (precondition not met)
    Skipped,
    /// Subgoal timed out
    TimedOut,
    /// Subgoal failed with error
    Failed(String),
}

impl SubgoalResult {
    /// Check if result is successful (Completed or Skipped)
    pub fn is_success(&self) -> bool {
        matches!(self, SubgoalResult::Completed | SubgoalResult::Skipped)
    }

    /// Check if result is Completed
    pub fn is_completed(&self) -> bool {
        matches!(self, SubgoalResult::Completed)
    }
}

/// Strategy for handling task failures
#[derive(Debug, Clone)]
pub enum FailureStrategy {
    /// Retry the failed subgoal N times
    Retry { count: u64 },
    /// Rollback to a previous subgoal
    Rollback { to: String },
    /// Abort the task immediately
    Abort,
    /// Request replanning from the orchestrator
    Replan,
}

impl Default for FailureStrategy {
    fn default() -> Self {
        FailureStrategy::Abort
    }
}

/// Deadline for timeout tracking
#[derive(Debug, Clone, Copy)]
pub struct Deadline {
    instant: Instant,
}

impl Deadline {
    /// Create a deadline from now plus duration
    pub fn from_duration(duration: Duration) -> Self {
        Self {
            instant: Instant::now() + duration,
        }
    }

    /// Create a deadline from milliseconds
    pub fn from_ms(ms: u64) -> Self {
        Self::from_duration(Duration::from_millis(ms))
    }

    /// Check if deadline has passed
    pub fn is_expired(&self) -> bool {
        Instant::now() >= self.instant
    }

    /// Get remaining time
    pub fn remaining(&self) -> Duration {
        self.instant.saturating_duration_since(Instant::now())
    }
}

/// Execution context passed to tasks
pub struct TaskContext {
    /// Current execution state
    pub state: ExecutionState,
    /// Accumulated reward
    pub total_reward: f64,
    /// Start time of task execution
    start_time: Instant,
    /// Available actions for current subgoal
    available_actions: Vec<String>,
    /// Action selection callback (set by runtime)
    action_selector: Option<Box<dyn Fn(&[&str]) -> Result<String>>>,
}

impl TaskContext {
    /// Create a new task context
    pub fn new() -> Self {
        Self {
            state: ExecutionState::new(),
            total_reward: 0.0,
            start_time: Instant::now(),
            available_actions: Vec::new(),
            action_selector: None,
        }
    }

    /// Create a new task context with initial state
    pub fn with_state(state: ExecutionState) -> Self {
        Self {
            state,
            total_reward: 0.0,
            start_time: Instant::now(),
            available_actions: Vec::new(),
            action_selector: None,
        }
    }

    /// Create deadline from milliseconds
    pub fn deadline_from_ms(&self, ms: u64) -> Deadline {
        Deadline::from_ms(ms)
    }

    /// Check if deadline has passed
    pub fn is_past_deadline(&self, deadline: Deadline) -> bool {
        deadline.is_expired()
    }

    /// Get elapsed time since task start
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Set available actions for current subgoal
    pub fn set_available_actions(&mut self, actions: Vec<String>) {
        self.available_actions = actions;
    }

    /// Get available actions
    pub fn available_actions(&self) -> &[String] {
        &self.available_actions
    }

    /// Set action selector callback
    pub fn set_action_selector<F>(&mut self, selector: F)
    where
        F: Fn(&[&str]) -> Result<String> + 'static,
    {
        self.action_selector = Some(Box::new(selector));
    }

    /// Select an action from available actions
    /// Uses the action selector if set, otherwise returns first action
    pub fn select_action(&self, actions: &[&str]) -> Result<String> {
        if let Some(ref selector) = self.action_selector {
            selector(actions)
        } else if !actions.is_empty() {
            Ok(actions[0].to_string())
        } else {
            Err(crate::error::Error::Runtime("no actions available".into()))
        }
    }

    /// Add reward
    pub fn add_reward(&mut self, reward: f64) {
        self.total_reward += reward;
    }

    /// Reset for new execution
    pub fn reset(&mut self) {
        self.state.reset();
        self.total_reward = 0.0;
        self.start_time = Instant::now();
        self.available_actions.clear();
    }
}

impl Default for TaskContext {
    fn default() -> Self {
        Self::new()
    }
}
