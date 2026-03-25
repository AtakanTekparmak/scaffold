//! Type checker for Scaffold v2
//!
//! Three-pass architecture:
//! Pass 1: Collect type declarations → TypeEnv.types
//! Pass 2: Collect node signatures → TypeEnv.nodes
//! Pass 3: Check graph and objective declarations

use std::collections::HashMap;

use scaffold_syntax::ast::*;

use crate::types::{GraphSig, NodeSig, NodeSigKind, StructType, Type, TypeEnv};

/// A type error with source location
#[derive(Debug, Clone)]
pub struct TypeError {
    pub message: String,
    pub span: Span,
}

impl std::fmt::Display for TypeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "type error at {:?}: {}", self.span, self.message)
    }
}

/// Type check a program and return errors
pub fn check(program: &Program) -> (TypeEnv, Vec<TypeError>) {
    let mut checker = Checker::new();
    checker.check_program(program);
    (checker.env, checker.errors)
}

struct Checker {
    env: TypeEnv,
    errors: Vec<TypeError>,
}

impl Checker {
    fn new() -> Self {
        Self {
            env: TypeEnv::new(),
            errors: Vec::new(),
        }
    }

    fn error(&mut self, span: &Span, message: impl Into<String>) {
        self.errors.push(TypeError {
            message: message.into(),
            span: span.clone(),
        });
    }

    fn check_program(&mut self, program: &Program) {
        // Pass 1: collect type declarations
        for decl in &program.declarations {
            if let Declaration::Type(td) = decl {
                let ty = self.resolve_type_expr(&td.ty);
                self.env.define_type(td.name.name.clone(), ty);
            }
        }

        // Pass 2: collect node signatures
        for decl in &program.declarations {
            if let Declaration::Node(nd) = decl {
                let input = self.resolve_type_expr(&nd.input);
                let output = self.resolve_type_expr(&nd.output);
                let kind = match nd.kind.node {
                    NodeKind::Prompt => NodeSigKind::Prompt,
                    NodeKind::Tool => NodeSigKind::Tool,
                    NodeKind::Agent => NodeSigKind::Agent,
                    NodeKind::Verify => NodeSigKind::Verify,
                };
                self.env
                    .nodes
                    .insert(nd.name.name.clone(), NodeSig { kind, input, output });
            }
        }

        // Also collect graph signatures (needed for objective checking)
        for decl in &program.declarations {
            if let Declaration::Graph(gd) = decl {
                let input = self.resolve_type_expr(&gd.input);
                let output = self.resolve_type_expr(&gd.output);
                self.env
                    .graphs
                    .insert(gd.name.name.clone(), GraphSig { input, output });
            }
        }

        // Pass 3: check graphs and objectives
        for decl in &program.declarations {
            match decl {
                Declaration::Graph(gd) => self.check_graph(gd),
                Declaration::Objective(od) => self.check_objective(od),
                Declaration::Node(nd) => self.check_node_config(nd),
                Declaration::Type(_) => {} // already processed
            }
        }
    }

    fn resolve_type_expr(&mut self, texpr: &Spanned<TypeExpr>) -> Type {
        match &texpr.node {
            TypeExpr::Primitive(p) => match p {
                PrimitiveType::Bool => Type::Bool,
                PrimitiveType::Int => Type::Int,
                PrimitiveType::Float => Type::Float,
                PrimitiveType::String => Type::String,
                PrimitiveType::Bytes => Type::Bytes,
                PrimitiveType::Any => Type::Any,
            },
            TypeExpr::Named(name) => {
                if self.env.lookup_type(name).is_some() {
                    Type::Named(name.clone())
                } else {
                    self.error(&texpr.span, format!("undefined type '{}'", name));
                    Type::Error
                }
            }
            TypeExpr::List(inner) => {
                let inner_ty = self.resolve_type_expr(inner);
                Type::List(Box::new(inner_ty))
            }
            TypeExpr::Map(key, val) => {
                let key_ty = self.resolve_type_expr(key);
                let val_ty = self.resolve_type_expr(val);
                Type::Map(Box::new(key_ty), Box::new(val_ty))
            }
            TypeExpr::Option(inner) => {
                let inner_ty = self.resolve_type_expr(inner);
                Type::Option(Box::new(inner_ty))
            }
            TypeExpr::Struct(fields) => {
                let mut field_types = HashMap::new();
                for field in fields {
                    let ty = self.resolve_type_expr(&field.ty);
                    if field_types.contains_key(&field.name.name) {
                        self.error(
                            &field.name.span,
                            format!("duplicate field '{}'", field.name.name),
                        );
                    }
                    field_types.insert(field.name.name.clone(), ty);
                }
                Type::Struct(StructType::with_fields(field_types))
            }
        }
    }

    fn check_node_config(&mut self, nd: &NodeDecl) {
        match nd.kind.node {
            NodeKind::Prompt | NodeKind::Verify => {
                if nd.config.template.is_none() {
                    self.error(
                        &nd.span,
                        format!(
                            "{} node '{}' should have a template",
                            if nd.kind.node == NodeKind::Prompt {
                                "prompt"
                            } else {
                                "verify"
                            },
                            nd.name.name
                        ),
                    );
                }
            }
            NodeKind::Tool => {
                if nd.config.shell.is_none() && nd.config.json.is_none() {
                    self.error(
                        &nd.span,
                        format!(
                            "tool node '{}' must have either 'shell' or 'json' field",
                            nd.name.name
                        ),
                    );
                }
            }
            NodeKind::Agent => {
                // Agents need tools
            }
        }
    }

    fn check_graph(&mut self, gd: &GraphDecl) {
        let input_ty = self.resolve_type_expr(&gd.input);
        let output_ty = self.resolve_type_expr(&gd.output);

        let mut scope = Scope::new();
        scope.define("input".to_string(), input_ty);

        self.check_graph_body(&gd.body, &mut scope, &output_ty);
    }

    fn check_graph_body(&mut self, stmts: &[GraphStmt], scope: &mut Scope, _output_ty: &Type) {
        for stmt in stmts {
            self.check_graph_stmt(stmt, scope);
        }
    }

    fn check_graph_stmt(&mut self, stmt: &GraphStmt, scope: &mut Scope) {
        match stmt {
            GraphStmt::Step(step) => {
                // Look up node signature
                if let Some(sig) = self.env.nodes.get(&step.node.name).cloned() {
                    // Bind step output to scope
                    scope.define(step.name.name.clone(), sig.output.clone());
                } else if self.env.graphs.get(&step.node.name).is_some() {
                    let gsig = self.env.graphs.get(&step.node.name).unwrap().clone();
                    scope.define(step.name.name.clone(), gsig.output.clone());
                } else {
                    self.error(
                        &step.node.span,
                        format!(
                            "undefined node or graph '{}'",
                            step.node.name
                        ),
                    );
                    scope.define(step.name.name.clone(), Type::Error);
                }
            }
            GraphStmt::Loop(loop_stmt) => {
                // Check max is numeric
                let max_ty = self.infer_expr(&loop_stmt.max, scope);
                if !max_ty.is_numeric() && !max_ty.is_error() {
                    self.error(
                        &loop_stmt.max.span,
                        format!("loop max must be numeric, got {}", max_ty),
                    );
                }
                // Check while is bool
                let cond_ty = self.infer_expr(&loop_stmt.while_cond, scope);
                if !matches!(cond_ty, Type::Bool | Type::Error | Type::Any) {
                    self.error(
                        &loop_stmt.while_cond.span,
                        format!("loop while condition must be bool, got {}", cond_ty),
                    );
                }
                // Check body in same scope (carries update scope)
                for s in &loop_stmt.body {
                    self.check_graph_stmt(s, scope);
                }
            }
            GraphStmt::If(if_stmt) => {
                let cond_ty = self.infer_expr(&if_stmt.cond, scope);
                if !matches!(cond_ty, Type::Bool | Type::Error | Type::Any) {
                    self.error(
                        &if_stmt.cond.span,
                        format!("if condition must be bool, got {}", cond_ty),
                    );
                }
                let mut then_scope = scope.child();
                for s in &if_stmt.then_body {
                    self.check_graph_stmt(s, &mut then_scope);
                }
                let mut else_scope = scope.child();
                for s in &if_stmt.else_body {
                    self.check_graph_stmt(s, &mut else_scope);
                }
            }
            GraphStmt::Choose(choose) => {
                for alt in &choose.alternatives {
                    if !self.env.graphs.contains_key(&alt.name)
                        && !self.env.nodes.contains_key(&alt.name)
                    {
                        self.error(
                            &alt.span,
                            format!("choose alternative '{}' not found", alt.name),
                        );
                    }
                }
            }
            GraphStmt::Parallel(par) => {
                let coll_ty = self.infer_expr(&par.collection, scope);
                let elem_ty = match &coll_ty {
                    Type::List(inner) => (**inner).clone(),
                    Type::Any => Type::Any,
                    Type::Error => Type::Error,
                    _ => {
                        self.error(
                            &par.collection.span,
                            format!("parallel collection must be list, got {}", coll_ty),
                        );
                        Type::Error
                    }
                };
                let mut body_scope = scope.child();
                body_scope.define(par.var.name.clone(), elem_ty);
                for s in &par.body {
                    self.check_graph_stmt(s, &mut body_scope);
                }
                if let Some(ref reduce) = par.reduce {
                    if !self.env.nodes.contains_key(&reduce.name)
                        && !self.env.graphs.contains_key(&reduce.name)
                    {
                        self.error(
                            &reduce.span,
                            format!("reduce function '{}' not found", reduce.name),
                        );
                    }
                }
            }
            GraphStmt::Emit(emit) => {
                match emit {
                    EmitStmt::Direct { value, .. } => {
                        let _ty = self.infer_expr(value, scope);
                    }
                    EmitStmt::Record { fields, .. } => {
                        for field in fields {
                            let _ty = self.infer_expr(&field.value, scope);
                        }
                    }
                }
            }
            GraphStmt::Carry(carry) => {
                let _ty = self.infer_expr(&carry.value, scope);
                // Carry should target something in enclosing scope
                if scope.lookup(&carry.name.name).is_none() {
                    // It's OK to create new bindings via carry
                    scope.define(carry.name.name.clone(), Type::Any);
                }
            }
        }
    }

    fn check_objective(&mut self, od: &ObjectiveDecl) {
        // Check graph reference
        if !self.env.graphs.contains_key(&od.graph.name) {
            self.error(
                &od.graph.span,
                format!("objective references undefined graph '{}'", od.graph.name),
            );
        }

        // Check score expression
        let mut scope = Scope::new();
        // Metrics are available in score expression
        for metric in &od.metrics {
            scope.define(metric.name.name.clone(), Type::Float);
        }
        let _score_ty = self.infer_expr(&od.score, &scope);

        // Check tune paths resolve
        for tune in &od.tunables {
            if tune.path.is_empty() {
                continue;
            }
            let node_name = &tune.path[0].name;
            if !self.env.nodes.contains_key(node_name) {
                self.error(
                    &tune.path[0].span,
                    format!("tune path references undefined node '{}'", node_name),
                );
            }
        }

        // Check topology constraints
        if let Some(ref topo) = od.topology {
            for preserved in &topo.preserve {
                // Check that preserved steps exist (we'd need the graph body here,
                // so just check nodes/graphs for now)
                if !self.env.nodes.contains_key(preserved)
                    && !self.env.graphs.contains_key(preserved)
                {
                    // This is a warning-level check, not an error
                }
            }
        }
    }

    /// Infer the type of an expression
    fn infer_expr(&mut self, expr: &Spanned<Expr>, scope: &Scope) -> Type {
        match &expr.node {
            Expr::Literal(lit) => match lit {
                Literal::Int(_) => Type::Int,
                Literal::Float(_) => Type::Float,
                Literal::String(_) => Type::String,
                Literal::Bool(_) => Type::Bool,
                Literal::Null => Type::Option(Box::new(Type::Any)),
            },
            Expr::Ident(name) => {
                if let Some(ty) = scope.lookup(name) {
                    ty.clone()
                } else {
                    self.error(&expr.span, format!("undefined variable '{}'", name));
                    Type::Error
                }
            }
            Expr::FieldAccess(base, field) => {
                let base_ty = self.infer_expr(base, scope);
                let resolved = self.env.resolve_type(&base_ty);
                match &resolved {
                    Type::Struct(s) => {
                        if let Some(field_ty) = s.get_field(&field.name) {
                            field_ty.clone()
                        } else {
                            self.error(
                                &field.span,
                                format!(
                                    "field '{}' not found in {}",
                                    field.name, resolved
                                ),
                            );
                            Type::Error
                        }
                    }
                    Type::Any => Type::Any,
                    Type::Error => Type::Error,
                    _ => {
                        // Allow field access on any type for flexibility
                        Type::Any
                    }
                }
            }
            Expr::Index(base, _index) => {
                let base_ty = self.infer_expr(base, scope);
                match &base_ty {
                    Type::List(inner) => (**inner).clone(),
                    Type::Map(_, val) => (**val).clone(),
                    Type::Any => Type::Any,
                    Type::Error => Type::Error,
                    _ => {
                        self.error(
                            &expr.span,
                            format!("cannot index into {}", base_ty),
                        );
                        Type::Error
                    }
                }
            }
            Expr::UnaryNot(operand) => {
                let ty = self.infer_expr(operand, scope);
                if !matches!(ty, Type::Bool | Type::Error | Type::Any) {
                    self.error(
                        &expr.span,
                        format!("cannot apply ! to {}", ty),
                    );
                }
                Type::Bool
            }
            Expr::UnaryNeg(operand) => {
                let ty = self.infer_expr(operand, scope);
                if !ty.is_numeric() && !ty.is_error() && !matches!(ty, Type::Any) {
                    self.error(
                        &expr.span,
                        format!("cannot negate {}", ty),
                    );
                }
                ty
            }
            Expr::Binary(left, op, right) => {
                let left_ty = self.infer_expr(left, scope);
                let right_ty = self.infer_expr(right, scope);

                match op {
                    BinOp::Eq | BinOp::Ne => Type::Bool,
                    BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge => {
                        if !left_ty.is_comparable()
                            && !left_ty.is_error()
                            && !matches!(left_ty, Type::Any)
                        {
                            self.error(
                                &expr.span,
                                format!("cannot compare {} with {}", left_ty, right_ty),
                            );
                        }
                        Type::Bool
                    }
                    BinOp::And | BinOp::Or => Type::Bool,
                    BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div => {
                        if left_ty.is_error() || right_ty.is_error() {
                            Type::Error
                        } else if matches!(left_ty, Type::Any) || matches!(right_ty, Type::Any) {
                            Type::Any
                        } else if matches!((&left_ty, &right_ty), (Type::String, _) | (_, Type::String))
                            && matches!(op, BinOp::Add)
                        {
                            Type::String
                        } else if matches!(left_ty, Type::Float) || matches!(right_ty, Type::Float) {
                            Type::Float
                        } else {
                            Type::Int
                        }
                    }
                }
            }
            Expr::Call(name, args) => {
                // Type-check arguments
                for arg in args {
                    let _ty = self.infer_expr(arg, scope);
                }
                // Built-in functions return Any for now
                let _ = name;
                Type::Any
            }
            Expr::List(elements) => {
                if elements.is_empty() {
                    Type::List(Box::new(Type::Any))
                } else {
                    let first_ty = self.infer_expr(&elements[0], scope);
                    for elem in &elements[1..] {
                        let _ty = self.infer_expr(elem, scope);
                    }
                    Type::List(Box::new(first_ty))
                }
            }
            Expr::Record(fields) => {
                let mut field_types = HashMap::new();
                for field in fields {
                    let ty = self.infer_expr(&field.value, scope);
                    field_types.insert(field.key.name.clone(), ty);
                }
                Type::Struct(StructType::with_fields(field_types))
            }
            Expr::Paren(inner) => self.infer_expr(inner, scope),
        }
    }
}

/// Scope for variable bindings with parent chain
struct Scope {
    bindings: HashMap<String, Type>,
    parent: Option<Box<Scope>>,
}

impl Scope {
    fn new() -> Self {
        Self {
            bindings: HashMap::new(),
            parent: None,
        }
    }

    fn child(&self) -> Self {
        // Clone current scope as parent bindings (simplified parent chain)
        let mut child = Self::new();
        // Copy parent bindings into child
        if let Some(ref parent) = self.parent {
            for (k, v) in &parent.bindings {
                child.bindings.insert(k.clone(), v.clone());
            }
        }
        for (k, v) in &self.bindings {
            child.bindings.insert(k.clone(), v.clone());
        }
        child
    }

    fn define(&mut self, name: String, ty: Type) {
        self.bindings.insert(name, ty);
    }

    fn lookup(&self, name: &str) -> Option<&Type> {
        self.bindings
            .get(name)
            .or_else(|| self.parent.as_ref().and_then(|p| p.lookup(name)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scaffold_syntax::parser::parse;

    #[test]
    fn check_valid_program() {
        let src = r#"
            type Question = { text: string }

            node solver: prompt {
                in: Question
                out: { answer: string }
                model: "gpt-4o"
                template: "Solve: {{ text }}"
            }

            graph solve {
                in: Question
                out: string
                step s = solver(input)
                emit s.answer
            }
        "#;
        let program = parse(src).unwrap();
        let (_, errors) = check(&program);
        assert!(errors.is_empty(), "unexpected errors: {:?}", errors);
    }

    #[test]
    fn check_undefined_node() {
        let src = r#"
            graph test {
                in: string
                out: string
                step s = nonexistent(input)
                emit s
            }
        "#;
        let program = parse(src).unwrap();
        let (_, errors) = check(&program);
        assert!(!errors.is_empty());
        assert!(errors[0].message.contains("undefined"));
    }

    #[test]
    fn check_undefined_type() {
        let src = r#"
            node solver: prompt {
                in: NonExistent
                out: string
                template: "test"
            }
        "#;
        let program = parse(src).unwrap();
        let (_, errors) = check(&program);
        assert!(!errors.is_empty());
        assert!(errors[0].message.contains("undefined type"));
    }
}
