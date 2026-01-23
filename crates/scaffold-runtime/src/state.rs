//! Execution state management

use crate::value::Value;
use std::collections::HashMap;

/// Execution state for a task
#[derive(Debug, Clone, Default)]
pub struct ExecutionState {
    /// Named state variables
    variables: HashMap<String, Value>,
    /// Completed subgoals
    completed_subgoals: Vec<String>,
    /// Current subgoal being executed
    current_subgoal: Option<String>,
    /// Rollback checkpoints
    checkpoints: HashMap<String, StateCheckpoint>,
}

/// A checkpoint of state for rollback
#[derive(Debug, Clone)]
pub struct StateCheckpoint {
    /// Snapshot of variables at checkpoint time
    pub variables: HashMap<String, Value>,
    /// Subgoals completed at checkpoint time
    pub completed_subgoals: Vec<String>,
}

impl ExecutionState {
    /// Create new empty state
    pub fn new() -> Self {
        Self::default()
    }

    /// Get a state variable
    pub fn get(&self, name: &str) -> Option<&Value> {
        self.variables.get(name)
    }

    /// Set a state variable
    pub fn set(&mut self, name: impl Into<String>, value: impl Into<Value>) {
        self.variables.insert(name.into(), value.into());
    }

    /// Remove a state variable
    pub fn remove(&mut self, name: &str) -> Option<Value> {
        self.variables.remove(name)
    }

    /// Check if variable exists
    pub fn contains(&self, name: &str) -> bool {
        self.variables.contains_key(name)
    }

    /// Get all variable names
    pub fn variable_names(&self) -> impl Iterator<Item = &String> {
        self.variables.keys()
    }

    /// Mark a subgoal as completed
    pub fn mark_completed(&mut self, subgoal: impl Into<String>) {
        let name = subgoal.into();
        if !self.completed_subgoals.contains(&name) {
            self.completed_subgoals.push(name);
        }
    }

    /// Check if a subgoal is completed
    pub fn is_completed(&self, subgoal: &str) -> bool {
        self.completed_subgoals.contains(&subgoal.to_string())
    }

    /// Get completed subgoals
    pub fn completed_subgoals(&self) -> &[String] {
        &self.completed_subgoals
    }

    /// Set current subgoal
    pub fn set_current_subgoal(&mut self, subgoal: Option<String>) {
        self.current_subgoal = subgoal;
    }

    /// Get current subgoal
    pub fn current_subgoal(&self) -> Option<&str> {
        self.current_subgoal.as_deref()
    }

    /// Create a checkpoint for potential rollback
    pub fn create_checkpoint(&mut self, name: impl Into<String>) {
        let checkpoint = StateCheckpoint {
            variables: self.variables.clone(),
            completed_subgoals: self.completed_subgoals.clone(),
        };
        self.checkpoints.insert(name.into(), checkpoint);
    }

    /// Rollback to a checkpoint
    pub fn rollback_to(&mut self, name: &str) -> bool {
        if let Some(checkpoint) = self.checkpoints.get(name) {
            self.variables = checkpoint.variables.clone();
            self.completed_subgoals = checkpoint.completed_subgoals.clone();
            true
        } else {
            false
        }
    }

    /// Clear all checkpoints
    pub fn clear_checkpoints(&mut self) {
        self.checkpoints.clear();
    }

    /// Reset state to initial empty state
    pub fn reset(&mut self) {
        self.variables.clear();
        self.completed_subgoals.clear();
        self.current_subgoal = None;
        self.checkpoints.clear();
    }
}
