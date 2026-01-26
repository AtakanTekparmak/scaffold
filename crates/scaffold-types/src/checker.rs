//! Type checker for the Scaffold DSL

use std::collections::HashMap;

use scaffold_syntax::ast::*;
use scaffold_syntax::Span;

use crate::types::*;

/// Type checking error
#[derive(Debug, Clone)]
pub struct TypeError {
    pub message: String,
    pub span: Span,
}

impl TypeError {
    pub fn new(message: impl Into<String>, span: Span) -> Self {
        Self {
            message: message.into(),
            span,
        }
    }
}

pub type TypeResult<T> = Result<T, TypeError>;

/// Type checker for Scaffold programs
pub struct TypeChecker {
    /// Global type environment
    env: TypeEnv,
    /// Collected errors (for recovery)
    errors: Vec<TypeError>,
}

impl TypeChecker {
    pub fn new() -> Self {
        Self {
            env: TypeEnv::new(),
            errors: Vec::new(),
        }
    }

    /// Check a complete program
    pub fn check_program(&mut self, program: &Program) -> Result<TypeEnv, Vec<TypeError>> {
        // First pass: collect all type declarations
        for decl in &program.declarations {
            if let Declaration::Type(type_decl) = decl {
                self.collect_type_decl(type_decl);
            }
        }

        // Second pass: collect tool and prompt names
        for decl in &program.declarations {
            match decl {
                Declaration::Tool(tool_decl) => {
                    self.env.define_tool(tool_decl.name.node.clone());
                }
                Declaration::Prompt(prompt_decl) => {
                    self.env.define_prompt(prompt_decl.name.node.clone());
                }
                _ => {}
            }
        }

        // Third pass: check all declarations
        for decl in &program.declarations {
            self.check_declaration(decl);
        }

        if self.errors.is_empty() {
            Ok(self.env.clone())
        } else {
            Err(std::mem::take(&mut self.errors))
        }
    }

    fn collect_type_decl(&mut self, decl: &TypeDecl) {
        let ty = self.resolve_type_expr(&decl.ty);
        self.env.define_type(decl.name.node.clone(), ty);
    }

    fn check_declaration(&mut self, decl: &Declaration) {
        match decl {
            Declaration::Type(type_decl) => self.check_type_decl(type_decl),
            Declaration::ExternCrate(crate_decl) => self.check_extern_crate(crate_decl),
            Declaration::Foreign(foreign_decl) => self.check_foreign(foreign_decl),
            Declaration::Tool(tool_decl) => self.check_tool(tool_decl),
            Declaration::Prompt(prompt_decl) => self.check_prompt(prompt_decl),
            Declaration::Agent(agent_decl) => self.check_agent(agent_decl),
            Declaration::Pipeline(pipeline_decl) => self.check_pipeline(pipeline_decl),
        }
    }

    fn check_extern_crate(&mut self, _crate_decl: &scaffold_syntax::ast::ExternCrateDecl) {
        // External crate declarations are validated at code generation time
        // Just store for later use
    }

    fn check_foreign(&mut self, foreign_decl: &scaffold_syntax::ast::ForeignDecl) {
        // Validate foreign function signatures
        for func in &foreign_decl.functions {
            for param in &func.params {
                self.validate_type_expr(&param.ty);
            }
            self.validate_type_expr(&func.return_type);
        }
    }

    fn check_tool(&mut self, tool_decl: &scaffold_syntax::ast::ToolDecl) {
        // Validate tool input/output types
        self.validate_type_expr(&tool_decl.input);
        self.validate_type_expr(&tool_decl.output);

        // Validate spec expressions if present
        if let Some(ref spec) = tool_decl.spec {
            for pre in &spec.preconditions {
                self.check_expr(pre);
            }
            for post in &spec.postconditions {
                self.check_expr(post);
            }
        }
    }

    fn check_prompt(&mut self, prompt_decl: &scaffold_syntax::ast::PromptDecl) {
        // Validate prompt input/output types
        self.validate_type_expr(&prompt_decl.input);
        self.validate_type_expr(&prompt_decl.output);
    }

    fn check_agent(&mut self, agent_decl: &scaffold_syntax::ast::AgentDecl) {
        // Validate agent input/output types
        self.validate_type_expr(&agent_decl.input);
        self.validate_type_expr(&agent_decl.output);

        // Validate tool references exist
        for tool_ref in &agent_decl.tools {
            if !self.env.has_tool(&tool_ref.node) {
                self.errors.push(TypeError::new(
                    format!(
                        "undefined tool '{}' in agent '{}'",
                        tool_ref.node, agent_decl.name.node
                    ),
                    tool_ref.span,
                ));
            }
        }

        // Validate reward expression if present
        if let Some(ref reward) = agent_decl.reward {
            self.check_expr(reward);
        }

        // Validate done expression if present
        if let Some(ref done) = agent_decl.done {
            let done_ty = self.check_expr(done);
            if !done_ty.is_compatible_with(&Type::Bool) && !done_ty.is_error() {
                self.errors.push(TypeError::new(
                    format!("'done' condition must be bool, found {}", done_ty),
                    done.span,
                ));
            }
        }
    }

    fn check_pipeline(&mut self, pipeline_decl: &scaffold_syntax::ast::PipelineDecl) {
        // Validate pipeline input/output types
        self.validate_type_expr(&pipeline_decl.input);
        self.validate_type_expr(&pipeline_decl.output);

        // Validate step references exist
        // Note: Parser currently treats all calls as PipelineCall::Tool,
        // so we check if the name exists as either a tool OR a prompt
        for step in &pipeline_decl.steps {
            match &step.call {
                scaffold_syntax::ast::PipelineCall::Prompt { name, .. } => {
                    if !self.env.has_prompt(name) {
                        self.errors.push(TypeError::new(
                            format!(
                                "undefined prompt '{}' in pipeline '{}'",
                                name, pipeline_decl.name.node
                            ),
                            step.span,
                        ));
                    }
                }
                scaffold_syntax::ast::PipelineCall::Tool { name, .. } => {
                    // Check if it's a tool OR a prompt (parser doesn't distinguish)
                    if !self.env.has_tool(name) && !self.env.has_prompt(name) {
                        self.errors.push(TypeError::new(
                            format!(
                                "undefined tool or prompt '{}' in pipeline '{}'",
                                name, pipeline_decl.name.node
                            ),
                            step.span,
                        ));
                    }
                }
                scaffold_syntax::ast::PipelineCall::Expr(_expr) => {
                    // General expressions are allowed in pipeline steps; no name to validate here.
                }
            }
        }

        // Validate reward expression if present
        if let Some(ref reward) = pipeline_decl.reward {
            self.check_expr(reward);
        }
    }

    fn check_type_decl(&mut self, decl: &TypeDecl) {
        // Ensure all referenced types exist
        self.validate_type_expr(&decl.ty);
    }

    fn check_expr(&mut self, expr: &Spanned<Expr>) -> Type {
        match &expr.node {
            Expr::Literal(lit) => match lit {
                Literal::Int(_) => Type::Int,
                Literal::Float(_) => Type::Float,
                Literal::String(_) => Type::String,
                Literal::Bool(_) => Type::Bool,
                Literal::Null => Type::Unit,
            },
            Expr::Ident(name) => {
                // Check variables first
                if let Some(ty) = self.env.lookup_variable(name) {
                    return ty.clone();
                }
                // Unknown identifier - could be a forward reference or external
                // For now, return Any to allow flexibility
                Type::Any
            }
            Expr::FieldAccess(base, field) => {
                let base_ty = self.check_expr(base);
                let resolved = self.env.resolve_type(&base_ty);

                match &resolved {
                    Type::Struct(s) => {
                        if let Some(field_ty) = s.get_field(&field.node) {
                            field_ty.clone()
                        } else {
                            self.errors.push(TypeError::new(
                                format!("no field '{}' on type {}", field.node, resolved),
                                field.span,
                            ));
                            Type::Error
                        }
                    }
                    Type::Any | Type::Error => Type::Any,
                    _ => {
                        self.errors.push(TypeError::new(
                            format!("cannot access field '{}' on type {}", field.node, resolved),
                            field.span,
                        ));
                        Type::Error
                    }
                }
            }
            Expr::Binary(left, op, right) => {
                let left_ty = self.check_expr(left);
                let right_ty = self.check_expr(right);

                match op {
                    // Comparison operators
                    BinOp::Eq | BinOp::Ne => {
                        // Allow comparing anything with null (Unit)
                        let null_comparison =
                            matches!(left_ty, Type::Unit) || matches!(right_ty, Type::Unit);
                        if !null_comparison && !left_ty.is_compatible_with(&right_ty) {
                            self.errors.push(TypeError::new(
                                format!(
                                    "cannot compare {} with {} using {}",
                                    left_ty, right_ty, op
                                ),
                                expr.span,
                            ));
                        }
                        Type::Bool
                    }
                    BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge => {
                        if !left_ty.is_comparable() && !matches!(left_ty, Type::Any | Type::Error) {
                            self.errors.push(TypeError::new(
                                format!("cannot compare type {} with {}", left_ty, op),
                                left.span,
                            ));
                        }
                        if !right_ty.is_comparable() && !matches!(right_ty, Type::Any | Type::Error)
                        {
                            self.errors.push(TypeError::new(
                                format!("cannot compare type {} with {}", right_ty, op),
                                right.span,
                            ));
                        }
                        Type::Bool
                    }
                    // Logical operators
                    BinOp::And | BinOp::Or => {
                        if !left_ty.is_compatible_with(&Type::Bool) && !left_ty.is_error() {
                            self.errors.push(TypeError::new(
                                format!("expected bool for {}, found {}", op, left_ty),
                                left.span,
                            ));
                        }
                        if !right_ty.is_compatible_with(&Type::Bool) && !right_ty.is_error() {
                            self.errors.push(TypeError::new(
                                format!("expected bool for {}, found {}", op, right_ty),
                                right.span,
                            ));
                        }
                        Type::Bool
                    }
                    // Arithmetic operators
                    BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div => {
                        if !left_ty.is_numeric() && !matches!(left_ty, Type::Any | Type::Error) {
                            self.errors.push(TypeError::new(
                                format!("expected numeric type for {}, found {}", op, left_ty),
                                left.span,
                            ));
                            return Type::Error;
                        }
                        if !right_ty.is_numeric() && !matches!(right_ty, Type::Any | Type::Error) {
                            self.errors.push(TypeError::new(
                                format!("expected numeric type for {}, found {}", op, right_ty),
                                right.span,
                            ));
                            return Type::Error;
                        }
                        // Promote to float if either operand is float
                        if matches!(left_ty, Type::Float) || matches!(right_ty, Type::Float) {
                            Type::Float
                        } else {
                            Type::Int
                        }
                    }
                }
            }
            Expr::Call(_name, args) => {
                // Built-in functions or user-defined
                // For now, treat all function calls as returning Any
                // A more complete implementation would have a function signature registry
                for arg in args {
                    self.check_expr(arg);
                }
                Type::Any
            }
            Expr::ForeignCall {
                module: _,
                function: _,
                args,
            } => {
                // Foreign function calls - type check arguments
                // Return type is determined by the foreign function signature
                // For now, treat as returning Any
                for arg in args {
                    self.check_expr(arg);
                }
                Type::Any
            }
            Expr::Paren(inner) => self.check_expr(inner),
        }
    }

    fn resolve_type_expr(&self, ty: &Spanned<TypeExpr>) -> Type {
        match &ty.node {
            TypeExpr::Primitive(p) => match p {
                PrimitiveType::Bool => Type::Bool,
                PrimitiveType::Int => Type::Int,
                PrimitiveType::Float => Type::Float,
                PrimitiveType::String => Type::String,
                PrimitiveType::Any => Type::Any,
                PrimitiveType::Bytes => Type::Bytes,
            },
            TypeExpr::Named(name) => {
                if let Some(resolved) = self.env.lookup_type(name) {
                    resolved.clone()
                } else {
                    Type::Named(name.clone())
                }
            }
            TypeExpr::List(inner) => Type::List(Box::new(self.resolve_type_expr(inner))),
            TypeExpr::Map(key, value) => Type::Map(
                Box::new(self.resolve_type_expr(key)),
                Box::new(self.resolve_type_expr(value)),
            ),
            TypeExpr::Option(inner) => Type::Option(Box::new(self.resolve_type_expr(inner))),
            TypeExpr::Result(ok, err) => Type::Result(
                Box::new(self.resolve_type_expr(ok)),
                Box::new(self.resolve_type_expr(err)),
            ),
            TypeExpr::Struct(fields) => {
                let mut type_fields = HashMap::new();
                for field in fields {
                    type_fields.insert(field.name.node.clone(), self.resolve_type_expr(&field.ty));
                }
                Type::Struct(StructType::with_fields(type_fields))
            }
        }
    }

    fn validate_type_expr(&mut self, ty: &Spanned<TypeExpr>) {
        match &ty.node {
            TypeExpr::Named(name) => {
                if self.env.lookup_type(name).is_none() {
                    self.errors.push(TypeError::new(
                        format!("undefined type '{}'", name),
                        ty.span,
                    ));
                }
            }
            TypeExpr::List(inner) => self.validate_type_expr(inner),
            TypeExpr::Map(key, value) => {
                self.validate_type_expr(key);
                self.validate_type_expr(value);
            }
            TypeExpr::Option(inner) => self.validate_type_expr(inner),
            TypeExpr::Result(ok, err) => {
                self.validate_type_expr(ok);
                self.validate_type_expr(err);
            }
            TypeExpr::Struct(fields) => {
                for field in fields {
                    self.validate_type_expr(&field.ty);
                }
            }
            TypeExpr::Primitive(_) => {
                // Primitives are always valid
            }
        }
    }
}

impl Default for TypeChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// Check a program and return the type environment or errors
pub fn check(program: &Program) -> Result<TypeEnv, Vec<TypeError>> {
    let mut checker = TypeChecker::new();
    checker.check_program(program)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scaffold_syntax::parse;

    #[test]
    fn test_type_check_simple() {
        let source = r#"
            type Position = { x: int, y: int }

            tool get_pos {
                input: Position
                output: bool
            }

            agent test_agent {
                input: Position
                output: bool
                tools: [get_pos]
                system: "Test agent"
            }
        "#;
        let program = parse(source).unwrap();
        let result = check(&program);
        assert!(result.is_ok());
    }

    #[test]
    fn test_undefined_type() {
        let source = r#"
            tool test {
                input: UndefinedType
                output: bool
            }
        "#;
        let program = parse(source).unwrap();
        let result = check(&program);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| e.message.contains("undefined type")));
    }

    #[test]
    fn test_type_definitions() {
        let source = r#"
            type Position = { x: int, y: int }
            type Result = { success: bool, pos: Position }

            tool check_result {
                input: Result
                output: bool
            }
        "#;
        let program = parse(source).unwrap();
        let result = check(&program);
        assert!(result.is_ok());
    }

    #[test]
    fn test_undefined_tool_in_agent() {
        let source = r#"
            agent test_agent {
                input: { x: int }
                output: bool
                tools: [nonexistent_tool]
                system: "Test"
            }
        "#;
        let program = parse(source).unwrap();
        let result = check(&program);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors
            .iter()
            .any(|e| e.message.contains("undefined tool 'nonexistent_tool'")));
    }

    #[test]
    fn test_undefined_call_in_pipeline() {
        let source = r#"
            pipeline test_pipeline {
                input: { x: int }
                output: { y: int }
                steps {
                    let r = nonexistent_call(x)
                }
            }
        "#;
        let program = parse(source).unwrap();
        let result = check(&program);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| e
            .message
            .contains("undefined tool or prompt 'nonexistent_call'")));
    }

    #[test]
    fn test_valid_pipeline_references() {
        let source = r#"
            tool my_tool {
                input: { x: int }
                output: { y: int }
            }

            prompt my_prompt {
                input: { x: int }
                output: { y: int }
                template: "test"
            }

            pipeline test_pipeline {
                input: { x: int }
                output: { y: int }
                steps {
                    let a = my_tool(x)
                    let b = my_prompt(a)
                }
            }
        "#;
        let program = parse(source).unwrap();
        let result = check(&program);
        assert!(result.is_ok());
    }
}
