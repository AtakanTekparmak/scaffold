//! Scope/binding management for graph execution
//!
//! Scopes form a parent chain. Each scope level holds variable bindings
//! created by steps, carries, and loop variables.

use std::collections::HashMap;

use crate::value::Value;

/// A scope holds variable bindings during graph execution.
///
/// Scopes are chained: a child scope can read from its parent,
/// but writes only affect the current scope.
#[derive(Debug, Clone)]
pub struct Scope {
    bindings: HashMap<String, Value>,
    parent: Option<Box<Scope>>,
}

impl Scope {
    /// Create a root scope with an initial "input" binding.
    pub fn root(input: Value) -> Self {
        let mut bindings = HashMap::new();
        bindings.insert("input".to_string(), input);
        Self {
            bindings,
            parent: None,
        }
    }

    /// Create a root scope with arbitrary named bindings (no "input" key).
    pub fn with_bindings(bindings: Vec<(&str, Value)>) -> Self {
        let map = bindings
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();
        Self {
            bindings: map,
            parent: None,
        }
    }

    /// Create a child scope that can read from this scope.
    pub fn child(&self) -> Self {
        Self {
            bindings: HashMap::new(),
            parent: Some(Box::new(self.clone())),
        }
    }

    /// Bind a name to a value in this scope.
    pub fn bind(&mut self, name: impl Into<String>, value: Value) {
        self.bindings.insert(name.into(), value);
    }

    /// Look up a name, searching this scope then parents.
    pub fn get(&self, name: &str) -> Option<&Value> {
        self.bindings
            .get(name)
            .or_else(|| self.parent.as_ref().and_then(|p| p.get(name)))
    }

    /// Check if a name is bound in this scope or any parent.
    pub fn contains(&self, name: &str) -> bool {
        self.get(name).is_some()
    }

    /// Convert all visible bindings to a flat map (for template rendering).
    /// Child bindings shadow parent bindings.
    pub fn to_template_value(&self) -> Value {
        let mut all = HashMap::new();
        self.collect_bindings(&mut all);
        Value::Map(all)
    }

    fn collect_bindings(&self, out: &mut HashMap<String, Value>) {
        if let Some(ref parent) = self.parent {
            parent.collect_bindings(out);
        }
        // Child overwrites parent
        for (k, v) in &self.bindings {
            out.insert(k.clone(), v.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_root_scope() {
        let scope = Scope::root(Value::String("hello".into()));
        assert_eq!(scope.get("input"), Some(&Value::String("hello".into())));
        assert_eq!(scope.get("missing"), None);
    }

    #[test]
    fn test_child_scope() {
        let mut parent = Scope::root(Value::String("hello".into()));
        parent.bind("x", Value::Int(42));

        let mut child = parent.child();
        child.bind("y", Value::Int(100));

        // Child can see parent bindings
        assert_eq!(child.get("x"), Some(&Value::Int(42)));
        assert_eq!(child.get("y"), Some(&Value::Int(100)));
        assert_eq!(child.get("input"), Some(&Value::String("hello".into())));

        // Parent can't see child bindings
        assert_eq!(parent.get("y"), None);
    }

    #[test]
    fn test_shadowing() {
        let mut parent = Scope::root(Value::Int(1));
        parent.bind("x", Value::Int(10));

        let mut child = parent.child();
        child.bind("x", Value::Int(20));

        assert_eq!(child.get("x"), Some(&Value::Int(20)));
        assert_eq!(parent.get("x"), Some(&Value::Int(10)));
    }

    #[test]
    fn test_to_template_value() {
        let mut parent = Scope::root(Value::String("hello".into()));
        parent.bind("a", Value::Int(1));

        let mut child = parent.child();
        child.bind("b", Value::Int(2));
        child.bind("a", Value::Int(99)); // shadow

        let val = child.to_template_value();
        let map = val.as_map().unwrap();
        assert_eq!(map.get("a"), Some(&Value::Int(99)));
        assert_eq!(map.get("b"), Some(&Value::Int(2)));
        assert_eq!(map.get("input"), Some(&Value::String("hello".into())));
    }
}
