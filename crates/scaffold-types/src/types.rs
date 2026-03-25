//! Type system definitions for Scaffold v2

use std::collections::HashMap;
use std::fmt;

/// Resolved type representation
#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Bool,
    Int,
    Float,
    String,
    Bytes,
    Any,
    List(Box<Type>),
    Map(Box<Type>, Box<Type>),
    Option(Box<Type>),
    Struct(StructType),
    Named(String),
    Error,
    Unit,
}

impl Type {
    pub fn is_numeric(&self) -> bool {
        matches!(self, Type::Int | Type::Float)
    }

    pub fn is_comparable(&self) -> bool {
        matches!(self, Type::Int | Type::Float | Type::String | Type::Bool)
    }

    pub fn is_error(&self) -> bool {
        matches!(self, Type::Error)
    }

    pub fn is_compatible_with(&self, other: &Type) -> bool {
        if self.is_error() || other.is_error() {
            return true;
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
            (Type::Int, Type::Float) | (Type::Float, Type::Int) => true,
            (Type::List(a), Type::List(b)) => a.is_compatible_with(b),
            (Type::Map(k1, v1), Type::Map(k2, v2)) => {
                k1.is_compatible_with(k2) && v1.is_compatible_with(v2)
            }
            (Type::Option(a), Type::Option(b)) => a.is_compatible_with(b),
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
            Type::Struct(s) => write!(f, "{}", s),
            Type::Named(name) => write!(f, "{}", name),
            Type::Error => write!(f, "<error>"),
            Type::Unit => write!(f, "()"),
        }
    }
}

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
        let mut fields: Vec<_> = self.fields.iter().collect();
        fields.sort_by_key(|(k, _)| k.clone());
        for (name, ty) in fields {
            if !first {
                write!(f, ", ")?;
            }
            write!(f, "{}: {}", name, ty)?;
            first = false;
        }
        write!(f, " }}")
    }
}

/// Node signature for type checking
#[derive(Debug, Clone)]
pub struct NodeSig {
    pub kind: NodeSigKind,
    pub input: Type,
    pub output: Type,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeSigKind {
    Prompt,
    Tool,
    Agent,
    Verify,
}

/// Type environment for v2
#[derive(Debug, Clone, Default)]
pub struct TypeEnv {
    pub types: HashMap<String, Type>,
    pub nodes: HashMap<String, NodeSig>,
    pub graphs: HashMap<String, GraphSig>,
}

/// Graph signature
#[derive(Debug, Clone)]
pub struct GraphSig {
    pub input: Type,
    pub output: Type,
}

impl TypeEnv {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn define_type(&mut self, name: String, ty: Type) {
        self.types.insert(name, ty);
    }

    pub fn lookup_type(&self, name: &str) -> Option<&Type> {
        self.types.get(name)
    }

    pub fn resolve_type(&self, ty: &Type) -> Type {
        match ty {
            Type::Named(name) => self.lookup_type(name).cloned().unwrap_or(Type::Error),
            Type::List(inner) => Type::List(Box::new(self.resolve_type(inner))),
            Type::Map(k, v) => Type::Map(
                Box::new(self.resolve_type(k)),
                Box::new(self.resolve_type(v)),
            ),
            Type::Option(inner) => Type::Option(Box::new(self.resolve_type(inner))),
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
