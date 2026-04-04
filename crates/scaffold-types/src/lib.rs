//! Scaffold v2 type system and checker

pub mod checker;
pub mod types;

pub use checker::{check, TypeError};
pub use types::{GraphSig, NodeSig, NodeSigKind, StructType, Type, TypeEnv};
