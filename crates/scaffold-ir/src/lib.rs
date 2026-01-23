//! Intermediate Representation (IR) for the Scaffold DSL
//!
//! This crate provides:
//! - IR structure definitions
//! - AST-to-IR lowering
//! - JSON serialization/deserialization
//! - JSON Schema generation

pub mod ir;
pub mod schema;
pub mod serialize;

pub use ir::*;
pub use schema::{
    type_to_json_schema, type_to_json_schema_compact, type_to_json_schema_string,
    types_to_json_schema_document,
};
pub use serialize::{from_json, to_json, to_json_compact, LowerError, Lowerer};
