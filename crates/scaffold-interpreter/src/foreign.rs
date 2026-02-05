//! Foreign function registry for the interpreter
//!
//! Allows registering Rust functions that can be called from scaffold code.

use scaffold_runtime::Value;
use std::collections::HashMap;
use std::sync::Arc;

use crate::error::{InterpreterError, Result};

/// Type alias for foreign functions
pub type ForeignFn = Arc<dyn Fn(Vec<Value>) -> Result<Value> + Send + Sync>;

/// Registry for foreign functions
#[derive(Default, Clone)]
pub struct ForeignRegistry {
    /// Functions indexed by "module::function" key
    functions: HashMap<String, ForeignFn>,
}

impl ForeignRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a foreign function
    ///
    /// # Example
    /// ```ignore
    /// let mut registry = ForeignRegistry::new();
    /// registry.register("math", "add", |args| {
    ///     let a = args.get(0).and_then(|v| v.as_int()).unwrap_or(0);
    ///     let b = args.get(1).and_then(|v| v.as_int()).unwrap_or(0);
    ///     Ok(Value::Int(a + b))
    /// });
    /// ```
    pub fn register<F>(&mut self, module: &str, function: &str, f: F)
    where
        F: Fn(Vec<Value>) -> Result<Value> + Send + Sync + 'static,
    {
        let key = format!("{}::{}", module, function);
        self.functions.insert(key, Arc::new(f));
    }

    /// Call a foreign function
    pub fn call(&self, module: &str, function: &str, args: Vec<Value>) -> Result<Value> {
        let key = format!("{}::{}", module, function);
        match self.functions.get(&key) {
            Some(f) => f(args),
            None => Err(InterpreterError::ForeignError(format!(
                "Foreign function '{}::{}' not registered",
                module, function
            ))),
        }
    }

    /// Check if a function is registered
    pub fn has_function(&self, module: &str, function: &str) -> bool {
        let key = format!("{}::{}", module, function);
        self.functions.contains_key(&key)
    }

    /// List all registered functions
    pub fn list_functions(&self) -> Vec<String> {
        self.functions.keys().cloned().collect()
    }
}

impl std::fmt::Debug for ForeignRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ForeignRegistry")
            .field("functions", &self.functions.keys().collect::<Vec<_>>())
            .finish()
    }
}

/// Builder for creating a foreign registry with common functions
pub struct ForeignRegistryBuilder {
    registry: ForeignRegistry,
}

impl ForeignRegistryBuilder {
    pub fn new() -> Self {
        Self {
            registry: ForeignRegistry::new(),
        }
    }

    /// Add standard library functions
    pub fn with_stdlib(mut self) -> Self {
        // String functions
        self.registry.register("string", "length", |args| {
            let s = args.first().and_then(|v| v.as_str()).unwrap_or("");
            Ok(Value::Int(s.len() as i64))
        });

        self.registry.register("string", "to_upper", |args| {
            let s = args.first().and_then(|v| v.as_str()).unwrap_or("");
            Ok(Value::String(s.to_uppercase()))
        });

        self.registry.register("string", "to_lower", |args| {
            let s = args.first().and_then(|v| v.as_str()).unwrap_or("");
            Ok(Value::String(s.to_lowercase()))
        });

        self.registry.register("string", "trim", |args| {
            let s = args.first().and_then(|v| v.as_str()).unwrap_or("");
            Ok(Value::String(s.trim().to_string()))
        });

        self.registry.register("string", "split", |args| {
            let s = args.first().and_then(|v| v.as_str()).unwrap_or("");
            let sep = args.get(1).and_then(|v| v.as_str()).unwrap_or(" ");
            let parts: Vec<Value> = s.split(sep).map(|p| Value::String(p.to_string())).collect();
            Ok(Value::List(parts))
        });

        self.registry.register("string", "join", |args| {
            let list = args
                .first()
                .and_then(|v| match v {
                    Value::List(l) => Some(l.clone()),
                    _ => None,
                })
                .unwrap_or_default();
            let sep = args.get(1).and_then(|v| v.as_str()).unwrap_or("");
            let parts: Vec<&str> = list.iter().filter_map(|v| v.as_str()).collect();
            Ok(Value::String(parts.join(sep)))
        });

        // Math functions
        self.registry
            .register("math", "abs", |args| match args.first() {
                Some(Value::Int(n)) => Ok(Value::Int(n.abs())),
                Some(Value::Float(n)) => Ok(Value::Float(n.abs())),
                _ => Ok(Value::Int(0)),
            });

        self.registry.register("math", "max", |args| {
            let a = args.first().and_then(|v| v.as_int()).unwrap_or(0);
            let b = args.get(1).and_then(|v| v.as_int()).unwrap_or(0);
            Ok(Value::Int(a.max(b)))
        });

        self.registry.register("math", "min", |args| {
            let a = args.first().and_then(|v| v.as_int()).unwrap_or(0);
            let b = args.get(1).and_then(|v| v.as_int()).unwrap_or(0);
            Ok(Value::Int(a.min(b)))
        });

        self.registry.register("math", "sqrt", |args| {
            let n = args
                .first()
                .and_then(|v| v.as_float())
                .or_else(|| args.first().and_then(|v| v.as_int()).map(|i| i as f64))
                .unwrap_or(0.0);
            Ok(Value::Float(n.sqrt()))
        });

        self.registry.register("math", "pow", |args| {
            let base = args
                .first()
                .and_then(|v| v.as_float())
                .or_else(|| args.first().and_then(|v| v.as_int()).map(|i| i as f64))
                .unwrap_or(0.0);
            let exp = args
                .get(1)
                .and_then(|v| v.as_float())
                .or_else(|| args.get(1).and_then(|v| v.as_int()).map(|i| i as f64))
                .unwrap_or(1.0);
            Ok(Value::Float(base.powf(exp)))
        });

        // List functions
        self.registry.register("list", "length", |args| {
            let len = match args.first() {
                Some(Value::List(l)) => l.len(),
                Some(Value::String(s)) => s.len(),
                Some(Value::Bytes(b)) => b.len(),
                _ => 0,
            };
            Ok(Value::Int(len as i64))
        });

        self.registry
            .register("list", "first", |args| match args.first() {
                Some(Value::List(l)) => Ok(l.first().cloned().unwrap_or(Value::Null)),
                _ => Ok(Value::Null),
            });

        self.registry
            .register("list", "last", |args| match args.first() {
                Some(Value::List(l)) => Ok(l.last().cloned().unwrap_or(Value::Null)),
                _ => Ok(Value::Null),
            });

        self.registry
            .register("list", "reverse", |args| match args.first() {
                Some(Value::List(l)) => {
                    let mut reversed = l.clone();
                    reversed.reverse();
                    Ok(Value::List(reversed))
                }
                _ => Ok(Value::List(vec![])),
            });

        self.registry.register("list", "contains", |args| {
            let list = match args.first() {
                Some(Value::List(l)) => l,
                _ => return Ok(Value::Bool(false)),
            };
            let item = args.get(1).cloned().unwrap_or(Value::Null);
            Ok(Value::Bool(list.contains(&item)))
        });

        // JSON functions
        self.registry.register("json", "parse", |args| {
            let s = args.first().and_then(|v| v.as_str()).unwrap_or("");
            match serde_json::from_str::<serde_json::Value>(s) {
                Ok(json) => Ok(json_to_value(json)),
                Err(e) => Err(InterpreterError::Runtime(format!(
                    "JSON parse error: {}",
                    e
                ))),
            }
        });

        self.registry.register("json", "stringify", |args| {
            let val = args.first().cloned().unwrap_or(Value::Null);
            let json = value_to_json(&val);
            Ok(Value::String(
                serde_json::to_string(&json).unwrap_or_default(),
            ))
        });

        self.registry.register("json", "stringify_pretty", |args| {
            let val = args.first().cloned().unwrap_or(Value::Null);
            let json = value_to_json(&val);
            Ok(Value::String(
                serde_json::to_string_pretty(&json).unwrap_or_default(),
            ))
        });

        self
    }

    /// Build the registry
    pub fn build(self) -> ForeignRegistry {
        self.registry
    }
}

impl Default for ForeignRegistryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert serde_json::Value to scaffold Value
fn json_to_value(json: serde_json::Value) -> Value {
    match json {
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
        serde_json::Value::Array(arr) => Value::List(arr.into_iter().map(json_to_value).collect()),
        serde_json::Value::Object(obj) => {
            let map: HashMap<String, Value> = obj
                .into_iter()
                .map(|(k, v)| (k, json_to_value(v)))
                .collect();
            Value::Map(map)
        }
    }
}

/// Convert scaffold Value to serde_json::Value
fn value_to_json(val: &Value) -> serde_json::Value {
    use base64::Engine;

    match val {
        Value::Null => serde_json::Value::Null,
        Value::Bool(b) => serde_json::Value::Bool(*b),
        Value::Int(i) => serde_json::json!(*i),
        Value::Float(f) => serde_json::json!(*f),
        Value::String(s) => serde_json::Value::String(s.clone()),
        Value::Bytes(b) => {
            serde_json::Value::String(base64::engine::general_purpose::STANDARD.encode(b))
        }
        Value::List(l) => serde_json::Value::Array(l.iter().map(value_to_json).collect()),
        Value::Map(m) => {
            let obj: serde_json::Map<String, serde_json::Value> = m
                .iter()
                .map(|(k, v)| (k.clone(), value_to_json(v)))
                .collect();
            serde_json::Value::Object(obj)
        }
        Value::Struct { fields, .. } => {
            let obj: serde_json::Map<String, serde_json::Value> = fields
                .iter()
                .map(|(k, v)| (k.clone(), value_to_json(v)))
                .collect();
            serde_json::Value::Object(obj)
        }
        Value::Result(res) => {
            // Serialize Result as an object with "ok" or "err" field
            match res.as_ref() {
                scaffold_runtime::ResultValue::Ok(v) => {
                    serde_json::json!({ "ok": value_to_json(v) })
                }
                scaffold_runtime::ResultValue::Err(e) => {
                    serde_json::json!({ "err": value_to_json(e) })
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_and_call() {
        let mut registry = ForeignRegistry::new();
        registry.register("math", "double", |args| {
            let n = args.first().and_then(|v| v.as_int()).unwrap_or(0);
            Ok(Value::Int(n * 2))
        });

        let result = registry
            .call("math", "double", vec![Value::Int(21)])
            .unwrap();
        assert_eq!(result, Value::Int(42));
    }

    #[test]
    fn test_stdlib_string_functions() {
        let registry = ForeignRegistryBuilder::new().with_stdlib().build();

        let result = registry
            .call(
                "string",
                "to_upper",
                vec![Value::String("hello".to_string())],
            )
            .unwrap();
        assert_eq!(result, Value::String("HELLO".to_string()));

        let result = registry
            .call("string", "length", vec![Value::String("hello".to_string())])
            .unwrap();
        assert_eq!(result, Value::Int(5));
    }

    #[test]
    fn test_stdlib_math_functions() {
        let registry = ForeignRegistryBuilder::new().with_stdlib().build();

        let result = registry
            .call("math", "max", vec![Value::Int(5), Value::Int(10)])
            .unwrap();
        assert_eq!(result, Value::Int(10));

        let result = registry
            .call("math", "sqrt", vec![Value::Float(16.0)])
            .unwrap();
        assert_eq!(result, Value::Float(4.0));
    }

    #[test]
    fn test_not_registered() {
        let registry = ForeignRegistry::new();
        let result = registry.call("unknown", "function", vec![]);
        assert!(result.is_err());
    }
}
