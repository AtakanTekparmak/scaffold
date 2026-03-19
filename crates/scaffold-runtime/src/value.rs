//! Dynamic value type for runtime data

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// Dynamic value type for runtime data interchange
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Value {
    /// Null/None value
    Null,
    /// Boolean value
    Bool(bool),
    /// Integer value
    Int(i64),
    /// Floating point value
    Float(f64),
    /// String value
    String(String),
    /// Bytes value
    Bytes(Vec<u8>),
    /// List of values
    List(Vec<Value>),
    /// Map of string keys to values
    Map(HashMap<String, Value>),
    /// Named struct with type info
    Struct {
        type_name: String,
        fields: HashMap<String, Value>,
    },
    /// Result type (Ok or Err)
    Result(Box<ResultValue>),
}

/// Result value wrapper
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ResultValue {
    Ok(Value),
    Err(Value),
}

impl Value {
    /// Check if value is null
    pub fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }

    /// Try to get as bool
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Try to get as i64
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Value::Int(i) => Some(*i),
            _ => None,
        }
    }

    /// Try to get as f64
    pub fn as_float(&self) -> Option<f64> {
        match self {
            Value::Float(f) => Some(*f),
            Value::Int(i) => Some(*i as f64),
            _ => None,
        }
    }

    /// Try to get as string reference
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    /// Try to get as list reference
    pub fn as_list(&self) -> Option<&Vec<Value>> {
        match self {
            Value::List(l) => Some(l),
            _ => None,
        }
    }

    /// Try to get as map reference
    pub fn as_map(&self) -> Option<&HashMap<String, Value>> {
        match self {
            Value::Map(m) => Some(m),
            _ => None,
        }
    }

    /// Get a field from a map value
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Map(m) => m.get(key),
            _ => None,
        }
    }

    /// Try to get as bytes reference
    pub fn as_bytes(&self) -> Option<&Vec<u8>> {
        match self {
            Value::Bytes(b) => Some(b),
            _ => None,
        }
    }

    /// Check if this is a struct of a specific type
    pub fn is_struct(&self, type_name: &str) -> bool {
        matches!(self, Value::Struct { type_name: tn, .. } if tn == type_name)
    }

    /// Get struct type name
    pub fn struct_type(&self) -> Option<&str> {
        match self {
            Value::Struct { type_name, .. } => Some(type_name),
            _ => None,
        }
    }

    /// Get struct field
    pub fn field(&self, name: &str) -> Option<&Value> {
        match self {
            Value::Struct { fields, .. } => fields.get(name),
            Value::Map(m) => m.get(name),
            _ => None,
        }
    }

    /// Check if this is an Ok result
    pub fn is_ok(&self) -> bool {
        matches!(self, Value::Result(r) if matches!(**r, ResultValue::Ok(_)))
    }

    /// Check if this is an Err result
    pub fn is_err(&self) -> bool {
        matches!(self, Value::Result(r) if matches!(**r, ResultValue::Err(_)))
    }

    /// Unwrap Ok value
    pub fn unwrap_ok(self) -> Option<Value> {
        match self {
            Value::Result(r) => match *r {
                ResultValue::Ok(v) => Some(v),
                ResultValue::Err(_) => None,
            },
            _ => None,
        }
    }

    /// Unwrap Err value
    pub fn unwrap_err(self) -> Option<Value> {
        match self {
            Value::Result(r) => match *r {
                ResultValue::Err(v) => Some(v),
                ResultValue::Ok(_) => None,
            },
            _ => None,
        }
    }

    /// Get type name for error messages
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Null => "null",
            Value::Bool(_) => "bool",
            Value::Int(_) => "int",
            Value::Float(_) => "float",
            Value::String(_) => "string",
            Value::Bytes(_) => "bytes",
            Value::List(_) => "list",
            Value::Map(_) => "map",
            Value::Struct { .. } => "struct",
            Value::Result(_) => "result",
        }
    }

    /// Convert to minijinja-compatible value
    pub fn to_template_value(&self) -> minijinja::Value {
        match self {
            Value::Null => minijinja::Value::UNDEFINED,
            Value::Bool(b) => minijinja::Value::from(*b),
            Value::Int(i) => minijinja::Value::from(*i),
            Value::Float(f) => minijinja::Value::from(*f),
            Value::String(s) => minijinja::Value::from(s.clone()),
            Value::Bytes(b) => minijinja::Value::from(format!("<{} bytes>", b.len())),
            Value::List(l) => {
                let items: Vec<minijinja::Value> =
                    l.iter().map(|v| v.to_template_value()).collect();
                minijinja::Value::from(items)
            }
            Value::Map(m) => {
                let map: std::collections::BTreeMap<String, minijinja::Value> = m
                    .iter()
                    .map(|(k, v)| (k.clone(), v.to_template_value()))
                    .collect();
                minijinja::Value::from_object(map)
            }
            Value::Struct { fields, .. } => {
                let map: std::collections::BTreeMap<String, minijinja::Value> = fields
                    .iter()
                    .map(|(k, v)| (k.clone(), v.to_template_value()))
                    .collect();
                minijinja::Value::from_object(map)
            }
            Value::Result(r) => match &**r {
                ResultValue::Ok(v) => v.to_template_value(),
                ResultValue::Err(e) => minijinja::Value::from(format!("Error: {:?}", e)),
            },
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Null => write!(f, "null"),
            Value::Bool(b) => write!(f, "{}", b),
            Value::Int(i) => write!(f, "{}", i),
            Value::Float(fl) => write!(f, "{}", fl),
            Value::String(s) => write!(f, "{}", s),
            Value::Bytes(b) => write!(f, "<{} bytes>", b.len()),
            Value::List(l) => {
                write!(f, "[")?;
                for (i, v) in l.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", v)?;
                }
                write!(f, "]")
            }
            Value::Map(m) => {
                write!(f, "{{")?;
                for (i, (k, v)) in m.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}: {}", k, v)?;
                }
                write!(f, "}}")
            }
            Value::Struct { type_name, fields } => {
                write!(f, "{}{{", type_name)?;
                for (i, (k, v)) in fields.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}: {}", k, v)?;
                }
                write!(f, "}}")
            }
            Value::Result(r) => match &**r {
                ResultValue::Ok(v) => write!(f, "Ok({})", v),
                ResultValue::Err(e) => write!(f, "Err({})", e),
            },
        }
    }
}

impl From<serde_json::Value> for Value {
    fn from(value: serde_json::Value) -> Self {
        match value {
            serde_json::Value::Null => Value::Null,
            serde_json::Value::Bool(b) => Value::Bool(b),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Value::Int(i)
                } else if let Some(f) = n.as_f64() {
                    Value::Float(f)
                } else {
                    Value::Null
                }
            }
            serde_json::Value::String(s) => Value::String(s),
            serde_json::Value::Array(items) => {
                Value::List(items.into_iter().map(Value::from).collect())
            }
            serde_json::Value::Object(map) => Value::Map(
                map.into_iter()
                    .map(|(key, value)| (key, Value::from(value)))
                    .collect(),
            ),
        }
    }
}

impl From<Value> for serde_json::Value {
    fn from(value: Value) -> Self {
        match value {
            Value::Null => serde_json::Value::Null,
            Value::Bool(b) => serde_json::Value::Bool(b),
            Value::Int(i) => serde_json::Value::Number(i.into()),
            Value::Float(f) => serde_json::Number::from_f64(f)
                .map(serde_json::Value::Number)
                .unwrap_or(serde_json::Value::Null),
            Value::String(s) => serde_json::Value::String(s),
            Value::Bytes(bytes) => serde_json::Value::Array(
                bytes
                    .into_iter()
                    .map(|byte| serde_json::Value::Number(byte.into()))
                    .collect(),
            ),
            Value::List(items) => {
                serde_json::Value::Array(items.into_iter().map(serde_json::Value::from).collect())
            }
            Value::Map(map) => serde_json::Value::Object(
                map.into_iter()
                    .map(|(key, value)| (key, serde_json::Value::from(value)))
                    .collect(),
            ),
            Value::Struct { fields, .. } => serde_json::Value::Object(
                fields
                    .into_iter()
                    .map(|(key, value)| (key, serde_json::Value::from(value)))
                    .collect(),
            ),
            Value::Result(result) => match *result {
                ResultValue::Ok(value) => serde_json::Value::from(value),
                ResultValue::Err(value) => serde_json::json!({
                    "err": serde_json::Value::from(value),
                }),
            },
        }
    }
}

impl Default for Value {
    fn default() -> Self {
        Value::Null
    }
}

impl From<bool> for Value {
    fn from(v: bool) -> Self {
        Value::Bool(v)
    }
}

impl From<i64> for Value {
    fn from(v: i64) -> Self {
        Value::Int(v)
    }
}

impl From<i32> for Value {
    fn from(v: i32) -> Self {
        Value::Int(v as i64)
    }
}

impl From<f64> for Value {
    fn from(v: f64) -> Self {
        Value::Float(v)
    }
}

impl From<String> for Value {
    fn from(v: String) -> Self {
        Value::String(v)
    }
}

impl From<&str> for Value {
    fn from(v: &str) -> Self {
        Value::String(v.to_string())
    }
}

impl<T: Into<Value>> From<Vec<T>> for Value {
    fn from(v: Vec<T>) -> Self {
        Value::List(v.into_iter().map(Into::into).collect())
    }
}

impl<T: Into<Value>> From<Option<T>> for Value {
    fn from(v: Option<T>) -> Self {
        match v {
            Some(val) => val.into(),
            None => Value::Null,
        }
    }
}

impl From<Vec<u8>> for Value {
    fn from(v: Vec<u8>) -> Self {
        Value::Bytes(v)
    }
}

impl<T: Into<Value>, E: Into<Value>> From<Result<T, E>> for Value {
    fn from(v: Result<T, E>) -> Self {
        match v {
            Ok(val) => Value::Result(Box::new(ResultValue::Ok(val.into()))),
            Err(err) => Value::Result(Box::new(ResultValue::Err(err.into()))),
        }
    }
}

impl Value {
    /// Create a struct value
    pub fn new_struct(type_name: impl Into<String>, fields: HashMap<String, Value>) -> Self {
        Value::Struct {
            type_name: type_name.into(),
            fields,
        }
    }

    /// Create an Ok result
    pub fn ok(value: impl Into<Value>) -> Self {
        Value::Result(Box::new(ResultValue::Ok(value.into())))
    }

    /// Create an Err result
    pub fn err(value: impl Into<Value>) -> Self {
        Value::Result(Box::new(ResultValue::Err(value.into())))
    }
}
