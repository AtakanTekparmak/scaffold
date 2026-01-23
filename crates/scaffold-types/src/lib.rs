//! Type system and type checker for the Scaffold DSL

pub mod checker;
pub mod types;

pub use checker::{check, TypeChecker, TypeError};
pub use types::*;
