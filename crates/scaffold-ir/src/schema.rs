//! JSON Schema generation from TypeIR
//!
//! Generates JSON Schema (draft-07) from scaffold type definitions.

use crate::ir::TypeIR;
use serde_json::{json, Value as JsonValue};

/// Generate a JSON Schema from a TypeIR
pub fn type_to_json_schema(ty: &TypeIR) -> JsonValue {
    match ty {
        TypeIR::Bool => json!({ "type": "boolean" }),
        TypeIR::Int => json!({ "type": "integer" }),
        TypeIR::Float => json!({ "type": "number" }),
        TypeIR::String => json!({ "type": "string" }),
        TypeIR::Bytes => json!({ "type": "string", "contentEncoding": "base64" }),
        TypeIR::Any => json!({}), // Any type accepts anything

        TypeIR::List { element } => {
            json!({
                "type": "array",
                "items": type_to_json_schema(element)
            })
        }

        TypeIR::Map { key: _, value } => {
            // JSON Schema doesn't directly support typed keys, use additionalProperties
            json!({
                "type": "object",
                "additionalProperties": type_to_json_schema(value)
            })
        }

        TypeIR::Option { inner } => {
            // Option is represented as oneOf with null
            json!({
                "oneOf": [
                    type_to_json_schema(inner),
                    { "type": "null" }
                ]
            })
        }

        TypeIR::Result { ok, err: _ } => {
            // For LLM output, we typically want the Ok type
            // Could be extended to support error variant
            type_to_json_schema(ok)
        }

        TypeIR::Struct { fields } => {
            let mut properties = serde_json::Map::new();
            let mut required = Vec::new();

            for (name, field_type) in fields {
                properties.insert(name.clone(), type_to_json_schema(field_type));
                required.push(JsonValue::String(name.clone()));
            }

            json!({
                "type": "object",
                "properties": JsonValue::Object(properties),
                "required": required,
                "additionalProperties": false
            })
        }

        TypeIR::Named { name } => {
            // Reference to a named type - use $ref in full schema context
            // For standalone use, just indicate the type name
            json!({ "$ref": format!("#/definitions/{}", name) })
        }
    }
}

/// Generate a JSON Schema string from a TypeIR
pub fn type_to_json_schema_string(ty: &TypeIR) -> String {
    serde_json::to_string_pretty(&type_to_json_schema(ty))
        .unwrap_or_else(|_| "{}".to_string())
}

/// Generate a compact JSON Schema string from a TypeIR
pub fn type_to_json_schema_compact(ty: &TypeIR) -> String {
    serde_json::to_string(&type_to_json_schema(ty))
        .unwrap_or_else(|_| "{}".to_string())
}

/// Generate a complete JSON Schema document with definitions
pub fn types_to_json_schema_document(
    root_type: &TypeIR,
    definitions: &[(String, TypeIR)],
) -> JsonValue {
    let mut defs = serde_json::Map::new();

    for (name, ty) in definitions {
        defs.insert(name.clone(), type_to_json_schema(ty));
    }

    let mut schema = type_to_json_schema(root_type);

    if let JsonValue::Object(ref mut obj) = schema {
        obj.insert("$schema".to_string(), json!("http://json-schema.org/draft-07/schema#"));
        if !defs.is_empty() {
            obj.insert("definitions".to_string(), JsonValue::Object(defs));
        }
    }

    schema
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_primitive_schemas() {
        assert_eq!(
            type_to_json_schema(&TypeIR::Bool),
            json!({ "type": "boolean" })
        );
        assert_eq!(
            type_to_json_schema(&TypeIR::Int),
            json!({ "type": "integer" })
        );
        assert_eq!(
            type_to_json_schema(&TypeIR::Float),
            json!({ "type": "number" })
        );
        assert_eq!(
            type_to_json_schema(&TypeIR::String),
            json!({ "type": "string" })
        );
    }

    #[test]
    fn test_list_schema() {
        let list_type = TypeIR::List {
            element: Box::new(TypeIR::String),
        };
        assert_eq!(
            type_to_json_schema(&list_type),
            json!({
                "type": "array",
                "items": { "type": "string" }
            })
        );
    }

    #[test]
    fn test_struct_schema() {
        let mut fields = HashMap::new();
        fields.insert("name".to_string(), TypeIR::String);
        fields.insert("age".to_string(), TypeIR::Int);

        let struct_type = TypeIR::Struct { fields };
        let schema = type_to_json_schema(&struct_type);

        assert_eq!(schema["type"], "object");
        assert_eq!(schema["properties"]["name"]["type"], "string");
        assert_eq!(schema["properties"]["age"]["type"], "integer");
        assert_eq!(schema["additionalProperties"], false);
    }

    #[test]
    fn test_nested_struct_schema() {
        let mut inner_fields = HashMap::new();
        inner_fields.insert("x".to_string(), TypeIR::Int);
        inner_fields.insert("y".to_string(), TypeIR::Int);

        let mut outer_fields = HashMap::new();
        outer_fields.insert("position".to_string(), TypeIR::Struct { fields: inner_fields });
        outer_fields.insert("label".to_string(), TypeIR::String);

        let struct_type = TypeIR::Struct { fields: outer_fields };
        let schema = type_to_json_schema(&struct_type);

        assert_eq!(schema["type"], "object");
        assert_eq!(schema["properties"]["position"]["type"], "object");
        assert_eq!(schema["properties"]["label"]["type"], "string");
    }

    #[test]
    fn test_option_schema() {
        let option_type = TypeIR::Option {
            inner: Box::new(TypeIR::String),
        };
        let schema = type_to_json_schema(&option_type);

        assert!(schema["oneOf"].is_array());
    }
}
