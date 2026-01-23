//! Type system definitions for the Scaffold DSL

use std::collections::HashMap;
use std::fmt;

/// Resolved type representation
#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    /// Primitive types
    Bool,
    Int,
    Float,
    String,
    /// Binary data
    Bytes,
    /// Any type (escape hatch)
    Any,
    /// List type
    List(Box<Type>),
    /// Map type
    Map(Box<Type>, Box<Type>),
    /// Optional type
    Option(Box<Type>),
    /// Result type (success, error)
    Result(Box<Type>, Box<Type>),
    /// Struct type with named fields
    Struct(StructType),
    /// Named type (unresolved reference)
    Named(String),
    /// Error type for propagating type errors
    Error,
    /// Unit type (for expressions with no value)
    Unit,
}

impl Type {
    /// Check if this type is numeric (int or float)
    pub fn is_numeric(&self) -> bool {
        matches!(self, Type::Int | Type::Float)
    }

    /// Check if this type is comparable
    pub fn is_comparable(&self) -> bool {
        matches!(
            self,
            Type::Int | Type::Float | Type::String | Type::Bool
        )
    }

    /// Check if this type is the error type
    pub fn is_error(&self) -> bool {
        matches!(self, Type::Error)
    }

    /// Check if two types are compatible (for assignment/comparison)
    pub fn is_compatible_with(&self, other: &Type) -> bool {
        if self.is_error() || other.is_error() {
            return true; // Don't cascade errors
        }
        if matches!(self, Type::Any) || matches!(other, Type::Any) {
            return true;
        }
        match (self, other) {
            (Type::Bool, Type::Bool)
            | (Type::Int, Type::Int)
            | (Type::Float, Type::Float)
            | (Type::String, Type::String)
            | (Type::Bytes, Type::Bytes)
            | (Type::Unit, Type::Unit) => true,
            // Int can be promoted to float
            (Type::Int, Type::Float) | (Type::Float, Type::Int) => true,
            (Type::List(a), Type::List(b)) => a.is_compatible_with(b),
            (Type::Map(k1, v1), Type::Map(k2, v2)) => {
                k1.is_compatible_with(k2) && v1.is_compatible_with(v2)
            }
            (Type::Option(a), Type::Option(b)) => a.is_compatible_with(b),
            (Type::Result(ok1, err1), Type::Result(ok2, err2)) => {
                ok1.is_compatible_with(ok2) && err1.is_compatible_with(err2)
            }
            // None (null) is compatible with any option type
            (Type::Option(_), Type::Unit) | (Type::Unit, Type::Option(_)) => true,
            (Type::Struct(a), Type::Struct(b)) => a.is_compatible_with(b),
            (Type::Named(a), Type::Named(b)) => a == b,
            _ => false,
        }
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Bool => write!(f, "bool"),
            Type::Int => write!(f, "int"),
            Type::Float => write!(f, "float"),
            Type::String => write!(f, "string"),
            Type::Bytes => write!(f, "bytes"),
            Type::Any => write!(f, "any"),
            Type::List(inner) => write!(f, "list<{}>", inner),
            Type::Map(k, v) => write!(f, "map<{}, {}>", k, v),
            Type::Option(inner) => write!(f, "option<{}>", inner),
            Type::Result(ok, err) => write!(f, "result<{}, {}>", ok, err),
            Type::Struct(s) => write!(f, "{}", s),
            Type::Named(name) => write!(f, "{}", name),
            Type::Error => write!(f, "<error>"),
            Type::Unit => write!(f, "()"),
        }
    }
}

/// Struct type with named fields
#[derive(Debug, Clone, PartialEq)]
pub struct StructType {
    pub fields: HashMap<String, Type>,
}

impl StructType {
    pub fn new() -> Self {
        Self {
            fields: HashMap::new(),
        }
    }

    pub fn with_fields(fields: HashMap<String, Type>) -> Self {
        Self { fields }
    }

    pub fn get_field(&self, name: &str) -> Option<&Type> {
        self.fields.get(name)
    }

    pub fn is_compatible_with(&self, other: &StructType) -> bool {
        // Structural compatibility: other must have all fields of self
        for (name, ty) in &self.fields {
            match other.fields.get(name) {
                Some(other_ty) if ty.is_compatible_with(other_ty) => {}
                _ => return false,
            }
        }
        true
    }
}

impl Default for StructType {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for StructType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{ ")?;
        let mut first = true;
        for (name, ty) in &self.fields {
            if !first {
                write!(f, ", ")?;
            }
            write!(f, "{}: {}", name, ty)?;
            first = false;
        }
        write!(f, " }}")
    }
}

/// Type environment for tracking defined types and variables
#[derive(Debug, Clone)]
pub struct TypeEnv {
    /// Named type definitions
    pub types: HashMap<String, Type>,
    /// Variable bindings in the current scope
    pub variables: HashMap<String, Type>,
    /// Declared tool names
    pub tools: std::collections::HashSet<String>,
    /// Declared prompt names
    pub prompts: std::collections::HashSet<String>,
}

impl TypeEnv {
    pub fn new() -> Self {
        Self {
            types: HashMap::new(),
            variables: HashMap::new(),
            tools: std::collections::HashSet::new(),
            prompts: std::collections::HashSet::new(),
        }
    }

    pub fn define_type(&mut self, name: String, ty: Type) {
        self.types.insert(name, ty);
    }

    pub fn lookup_type(&self, name: &str) -> Option<&Type> {
        self.types.get(name)
    }

    pub fn define_variable(&mut self, name: String, ty: Type) {
        self.variables.insert(name, ty);
    }

    pub fn lookup_variable(&self, name: &str) -> Option<&Type> {
        self.variables.get(name)
    }

    pub fn define_tool(&mut self, name: String) {
        self.tools.insert(name);
    }

    pub fn has_tool(&self, name: &str) -> bool {
        self.tools.contains(name)
    }

    pub fn define_prompt(&mut self, name: String) {
        self.prompts.insert(name);
    }

    pub fn has_prompt(&self, name: &str) -> bool {
        self.prompts.contains(name)
    }

    /// Resolve a named type to its definition
    pub fn resolve_type(&self, ty: &Type) -> Type {
        match ty {
            Type::Named(name) => self
                .lookup_type(name)
                .cloned()
                .unwrap_or(Type::Error),
            Type::List(inner) => Type::List(Box::new(self.resolve_type(inner))),
            Type::Map(k, v) => Type::Map(
                Box::new(self.resolve_type(k)),
                Box::new(self.resolve_type(v)),
            ),
            Type::Option(inner) => Type::Option(Box::new(self.resolve_type(inner))),
            Type::Result(ok, err) => Type::Result(
                Box::new(self.resolve_type(ok)),
                Box::new(self.resolve_type(err)),
            ),
            Type::Struct(s) => {
                let mut resolved_fields = HashMap::new();
                for (name, field_ty) in &s.fields {
                    resolved_fields.insert(name.clone(), self.resolve_type(field_ty));
                }
                Type::Struct(StructType::with_fields(resolved_fields))
            }
            _ => ty.clone(),
        }
    }
}

impl Default for TypeEnv {
    fn default() -> Self {
        Self::new()
    }
}

