//! Scaffold v2 Intermediate Representation
//!
//! - IR structure definitions
//! - AST-to-IR lowering
//! - JSON serialization/deserialization
//! - Pretty printing (IR → scaffold source)

pub mod ir;
pub mod pretty;
pub mod serialize;

pub use ir::*;
pub use pretty::pretty_print;
pub use serialize::{from_json, lower, to_json, to_json_compact, LowerError, Lowerer};
