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

#[derive(Debug, Clone)]
struct ForeignFnSig {
    params: Vec<Type>,
    return_type: Type,
}

#[derive(Debug, Clone)]
struct ToolSig {
    input: Type,
    output: Type,
}

/// Type checker for Scaffold programs
pub struct TypeChecker {
    /// Global type environment
    env: TypeEnv,
    /// Collected errors (for recovery)
    errors: Vec<TypeError>,
    /// Foreign function signatures by (module, function)
    foreign_sigs: HashMap<(String, String), ForeignFnSig>,
    /// Tool signatures by name
    tool_sigs: HashMap<String, ToolSig>,
    /// Prompt signatures by name
    prompt_sigs: HashMap<String, ToolSig>,
    /// Agent signatures by name
    agent_sigs: HashMap<String, ToolSig>,
}

impl TypeChecker {
    pub fn new() -> Self {
        Self {
            env: TypeEnv::new(),
            errors: Vec::new(),
            foreign_sigs: HashMap::new(),
            tool_sigs: HashMap::new(),
            prompt_sigs: HashMap::new(),
            agent_sigs: HashMap::new(),
        }
    }

    /// Check a complete program
    pub fn check_program(&mut self, program: &Program) -> Result<TypeEnv, Vec<TypeError>> {
        // First pass: collect all type declarations
        for decl in &program.declarations {
            match decl {
                Declaration::Type(type_decl) => self.collect_type_decl(type_decl),
                Declaration::Artifact(artifact_decl) => self.collect_artifact_decl(artifact_decl),
                _ => {}
            }
        }

        // Second pass: collect names and signatures
        for decl in &program.declarations {
            match decl {
                Declaration::Foreign(foreign_decl) => {
                    self.collect_foreign_signatures(foreign_decl);
                }
                Declaration::Tool(tool_decl) => {
                    self.env.define_tool(tool_decl.name.node.clone());
                    self.collect_tool_signature(tool_decl);
                }
                Declaration::Prompt(prompt_decl) => {
                    self.env.define_prompt(prompt_decl.name.node.clone());
                    self.collect_prompt_signature(prompt_decl);
                }
                Declaration::Agent(agent_decl) => {
                    self.env.define_agent(agent_decl.name.node.clone());
                    self.collect_agent_signature(agent_decl);
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

    fn collect_artifact_decl(&mut self, decl: &ArtifactDecl) {
        let ty = self.resolve_type_expr(&decl.ty);
        self.env.define_type(decl.name.node.clone(), ty);
    }

    fn collect_foreign_signatures(&mut self, decl: &ForeignDecl) {
        for func in &decl.functions {
            let key = (decl.name.node.clone(), func.name.node.clone());
            let params = func
                .params
                .iter()
                .map(|p| self.resolve_type_expr(&p.ty))
                .collect::<Vec<_>>();
            let return_type = self.resolve_type_expr(&func.return_type);
            self.foreign_sigs.insert(
                key,
                ForeignFnSig {
                    params,
                    return_type,
                },
            );
        }
    }

    fn collect_tool_signature(&mut self, decl: &ToolDecl) {
        let input = self.resolve_type_expr(&decl.input);
        let output = self.resolve_type_expr(&decl.output);
        self.tool_sigs
            .insert(decl.name.node.clone(), ToolSig { input, output });
    }

    fn collect_prompt_signature(&mut self, decl: &PromptDecl) {
        let input = self.resolve_type_expr(&decl.input);
        let output = self.resolve_type_expr(&decl.output);
        self.prompt_sigs
            .insert(decl.name.node.clone(), ToolSig { input, output });
    }

    fn collect_agent_signature(&mut self, decl: &AgentDecl) {
        let input = self.resolve_type_expr(&decl.input);
        let output = self.resolve_type_expr(&decl.output);
        self.agent_sigs
            .insert(decl.name.node.clone(), ToolSig { input, output });
    }

    fn check_declaration(&mut self, decl: &Declaration) {
        match decl {
            Declaration::Type(type_decl) => self.check_type_decl(type_decl),
            Declaration::Artifact(artifact_decl) => self.check_artifact_decl(artifact_decl),
            Declaration::ExternCrate(crate_decl) => self.check_extern_crate(crate_decl),
            Declaration::Foreign(foreign_decl) => self.check_foreign(foreign_decl),
            Declaration::Tool(tool_decl) => self.check_tool(tool_decl),
            Declaration::Prompt(prompt_decl) => self.check_prompt(prompt_decl),
            Declaration::Agent(agent_decl) => self.check_agent(agent_decl),
            Declaration::Pipeline(pipeline_decl) => self.check_pipeline(pipeline_decl),
            Declaration::Task(task_decl) => self.check_task(task_decl),
            Declaration::Harness(harness_decl) => self.check_harness(harness_decl),
            Declaration::Objective(objective_decl) => self.check_objective(objective_decl),
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

        let input_ty = self.resolve_type_expr(&tool_decl.input);
        let mut scope = HashMap::new();
        scope.insert("input".to_string(), input_ty.clone());
        if let Type::Struct(s) = self.env.resolve_type(&input_ty) {
            for (field, ty) in s.fields {
                scope.insert(field, ty);
            }
        }

        if let Some(impl_) = &tool_decl.implementation {
            self.check_tool_impl(impl_, &mut scope);
        }

        // Validate spec expressions if present
        if let Some(ref spec) = tool_decl.spec {
            for pre in &spec.preconditions {
                self.check_expr_with_scope(pre, &scope);
            }
            for post in &spec.postconditions {
                self.check_expr_with_scope(post, &scope);
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

        // Validate step references and argument types (recursively)
        let mut scope = HashMap::new();
        let input_ty = self.resolve_type_expr(&pipeline_decl.input);
        scope.insert("input".to_string(), input_ty.clone());
        if let Type::Struct(s) = self.env.resolve_type(&input_ty) {
            for (field, ty) in s.fields {
                scope.insert(field, ty);
            }
        }
        self.check_pipeline_steps(&pipeline_decl.steps, &pipeline_decl.name.node, &mut scope);

        // Validate reward expression if present
        if let Some(ref reward) = pipeline_decl.reward {
            self.check_expr_with_scope(reward, &scope);
        }
    }

    fn check_pipeline_steps(
        &mut self,
        steps: &[PipelineStep],
        pipeline_name: &str,
        scope: &mut HashMap<String, Type>,
    ) {
        for step in steps {
            let step_output_ty = match &step.call {
                scaffold_syntax::ast::PipelineCall::Prompt { name, args } => {
                    if !self.env.has_prompt(name) {
                        self.errors.push(TypeError::new(
                            format!(
                                "undefined prompt '{}' in pipeline '{}'",
                                name, pipeline_name
                            ),
                            step.span,
                        ));
                        Type::Error
                    } else if let Some(sig) = self.prompt_sigs.get(name).cloned() {
                        let arg_tys = args
                            .iter()
                            .map(|a| self.check_tool_expr(a, scope))
                            .collect::<Vec<_>>();
                        let arg_names = args
                            .iter()
                            .map(|a| match &a.node {
                                ToolExpr::Ident(n) => Some(n.clone()),
                                _ => None,
                            })
                            .collect::<Vec<_>>();
                        self.check_call_against_input_type(
                            name, &sig.input, &arg_tys, &arg_names, step.span,
                        );
                        sig.output
                    } else {
                        Type::Any
                    }
                }
                scaffold_syntax::ast::PipelineCall::Tool { name, args } => {
                    // Parser doesn't distinguish tool/prompt/agent for call-like syntax in steps.
                    let arg_tys = args
                        .iter()
                        .map(|a| self.check_tool_expr(a, scope))
                        .collect::<Vec<_>>();
                    let arg_names = args
                        .iter()
                        .map(|a| match &a.node {
                            ToolExpr::Ident(n) => Some(n.clone()),
                            _ => None,
                        })
                        .collect::<Vec<_>>();

                    if let Some(sig) = self.tool_sigs.get(name).cloned() {
                        self.check_call_against_input_type(
                            name, &sig.input, &arg_tys, &arg_names, step.span,
                        );
                        sig.output
                    } else if let Some(sig) = self.prompt_sigs.get(name).cloned() {
                        self.check_call_against_input_type(
                            name, &sig.input, &arg_tys, &arg_names, step.span,
                        );
                        sig.output
                    } else if let Some(sig) = self.agent_sigs.get(name).cloned() {
                        self.check_call_against_input_type(
                            name, &sig.input, &arg_tys, &arg_names, step.span,
                        );
                        sig.output
                    } else {
                        self.errors.push(TypeError::new(
                            format!(
                                "undefined tool, prompt, or agent '{}' in pipeline '{}'",
                                name, pipeline_name
                            ),
                            step.span,
                        ));
                        Type::Error
                    }
                }
                scaffold_syntax::ast::PipelineCall::Agent { name, args } => {
                    if !self.env.has_agent(name) {
                        self.errors.push(TypeError::new(
                            format!("undefined agent '{}' in pipeline '{}'", name, pipeline_name),
                            step.span,
                        ));
                        Type::Error
                    } else if let Some(sig) = self.agent_sigs.get(name).cloned() {
                        let arg_tys = args
                            .iter()
                            .map(|a| self.check_tool_expr(a, scope))
                            .collect::<Vec<_>>();
                        let arg_names = args
                            .iter()
                            .map(|a| match &a.node {
                                ToolExpr::Ident(n) => Some(n.clone()),
                                _ => None,
                            })
                            .collect::<Vec<_>>();
                        self.check_call_against_input_type(
                            name, &sig.input, &arg_tys, &arg_names, step.span,
                        );
                        sig.output
                    } else {
                        Type::Any
                    }
                }
                scaffold_syntax::ast::PipelineCall::Expr(expr) => self.check_tool_expr(expr, scope),
                scaffold_syntax::ast::PipelineCall::Parallel { branches } => {
                    for branch in branches {
                        let mut branch_scope = scope.clone();
                        self.check_pipeline_steps(branch, pipeline_name, &mut branch_scope);
                    }
                    Type::Unit
                }
                scaffold_syntax::ast::PipelineCall::If {
                    condition,
                    then_steps,
                    else_steps,
                } => {
                    let cond_ty = self.check_expr_with_scope(condition, scope);
                    if !cond_ty.is_compatible_with(&Type::Bool) && !cond_ty.is_error() {
                        self.errors.push(TypeError::new(
                            format!("'if' condition must be bool, found {}", cond_ty),
                            condition.span,
                        ));
                    }
                    let mut then_scope = scope.clone();
                    let mut else_scope = scope.clone();
                    self.check_pipeline_steps(then_steps, pipeline_name, &mut then_scope);
                    self.check_pipeline_steps(else_steps, pipeline_name, &mut else_scope);
                    Type::Any
                }
                scaffold_syntax::ast::PipelineCall::Match { scrutinee, arms } => {
                    self.check_expr_with_scope(scrutinee, scope);
                    for arm in arms {
                        self.check_expr_with_scope(&arm.pattern, scope);
                        let mut arm_scope = scope.clone();
                        self.check_pipeline_steps(&arm.steps, pipeline_name, &mut arm_scope);
                    }
                    Type::Any
                }
            };

            if let Some(binding) = &step.binding {
                scope.insert(binding.node.clone(), step_output_ty);
            }
        }
    }

    fn check_tool_impl(&mut self, impl_: &ToolImpl, scope: &mut HashMap<String, Type>) -> Type {
        match impl_ {
            ToolImpl::Expr(expr) => self.check_tool_expr(expr, scope),
            ToolImpl::Sequence(statements) | ToolImpl::Parallel(statements) => {
                let mut last_ty = Type::Unit;
                for stmt in statements {
                    last_ty = self.check_tool_statement(stmt, scope);
                }
                last_ty
            }
        }
    }

    fn check_tool_statement(
        &mut self,
        stmt: &ToolStatement,
        scope: &mut HashMap<String, Type>,
    ) -> Type {
        let ty = self.check_tool_expr(&stmt.expr, scope);
        if let Some(binding) = &stmt.binding {
            scope.insert(binding.node.clone(), ty.clone());
        }
        ty
    }

    fn check_tool_expr(
        &mut self,
        expr: &Spanned<ToolExpr>,
        scope: &mut HashMap<String, Type>,
    ) -> Type {
        match &expr.node {
            ToolExpr::Ident(name) => scope.get(name).cloned().unwrap_or(Type::Any),
            ToolExpr::FieldAccess(base, field) => {
                let base_ty = self.check_tool_expr(base, scope);
                let resolved = self.env.resolve_type(&base_ty);
                match resolved {
                    Type::Struct(s) => s.get_field(&field.node).cloned().unwrap_or_else(|| {
                        self.errors.push(TypeError::new(
                            format!("no field '{}' on type {}", field.node, Type::Struct(s)),
                            field.span,
                        ));
                        Type::Error
                    }),
                    Type::Any | Type::Error => Type::Any,
                    other => {
                        self.errors.push(TypeError::new(
                            format!("cannot access field '{}' on type {}", field.node, other),
                            field.span,
                        ));
                        Type::Error
                    }
                }
            }
            ToolExpr::ForeignCall {
                module,
                function,
                args,
            } => {
                let arg_tys = args
                    .iter()
                    .map(|a| self.check_tool_expr(a, scope))
                    .collect::<Vec<_>>();
                self.check_foreign_call(module, function, &arg_tys, expr.span)
            }
            ToolExpr::ToolCall { tool, args } => {
                let arg_tys = args
                    .iter()
                    .map(|a| self.check_tool_expr(a, scope))
                    .collect::<Vec<_>>();
                let arg_names = args
                    .iter()
                    .map(|a| match &a.node {
                        ToolExpr::Ident(n) => Some(n.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                if let Some(sig) = self.tool_sigs.get(tool).cloned() {
                    self.check_call_against_input_type(
                        tool, &sig.input, &arg_tys, &arg_names, expr.span,
                    );
                    sig.output
                } else {
                    self.errors.push(TypeError::new(
                        format!("undefined tool '{}' in tool implementation", tool),
                        expr.span,
                    ));
                    Type::Error
                }
            }
            ToolExpr::Shell(_) => Type::Any,
            ToolExpr::Pipe(left, right) => {
                self.check_tool_expr(left, scope);
                self.check_tool_expr(right, scope);
                Type::Any
            }
            ToolExpr::If {
                condition,
                then_branch,
                else_branch,
            } => {
                let cond_ty = self.check_expr_with_scope(condition, scope);
                if !cond_ty.is_compatible_with(&Type::Bool) && !cond_ty.is_error() {
                    self.errors.push(TypeError::new(
                        format!("'if' condition must be bool, found {}", cond_ty),
                        condition.span,
                    ));
                }

                let mut then_scope = scope.clone();
                let then_ty = self.check_tool_impl(then_branch, &mut then_scope);
                let else_ty = if let Some(branch) = else_branch {
                    let mut else_scope = scope.clone();
                    self.check_tool_impl(branch, &mut else_scope)
                } else {
                    Type::Unit
                };

                if then_ty.is_compatible_with(&else_ty) {
                    then_ty
                } else if else_ty.is_compatible_with(&then_ty) {
                    else_ty
                } else {
                    Type::Any
                }
            }
            ToolExpr::Match { scrutinee, arms } => {
                self.check_tool_expr(scrutinee, scope);
                let mut out_ty = Type::Unit;
                for arm in arms {
                    self.check_expr_with_scope(&arm.pattern, scope);
                    let mut arm_scope = scope.clone();
                    let ty = self.check_tool_impl(&arm.body, &mut arm_scope);
                    if matches!(out_ty, Type::Unit) {
                        out_ty = ty;
                    } else if !out_ty.is_compatible_with(&ty) && !ty.is_compatible_with(&out_ty) {
                        out_ty = Type::Any;
                    }
                }
                out_ty
            }
            ToolExpr::For {
                variable,
                iterable,
                body,
            } => {
                let iter_ty = self.check_tool_expr(iterable, scope);
                let element_ty = match self.env.resolve_type(&iter_ty) {
                    Type::List(inner) => *inner,
                    Type::Any | Type::Error => Type::Any,
                    other => {
                        self.errors.push(TypeError::new(
                            format!("'for' iterable must be list, found {}", other),
                            iterable.span,
                        ));
                        Type::Error
                    }
                };
                let mut body_scope = scope.clone();
                body_scope.insert(variable.node.clone(), element_ty);
                self.check_tool_impl(body, &mut body_scope)
            }
            ToolExpr::While { condition, body } => {
                let cond_ty = self.check_expr_with_scope(condition, scope);
                if !cond_ty.is_compatible_with(&Type::Bool) && !cond_ty.is_error() {
                    self.errors.push(TypeError::new(
                        format!("'while' condition must be bool, found {}", cond_ty),
                        condition.span,
                    ));
                }
                let mut body_scope = scope.clone();
                self.check_tool_impl(body, &mut body_scope)
            }
            ToolExpr::Loop { body } => {
                let mut body_scope = scope.clone();
                self.check_tool_impl(body, &mut body_scope)
            }
            ToolExpr::Break | ToolExpr::Continue => Type::Unit,
            ToolExpr::Literal(lit) => match lit {
                Literal::Int(_) => Type::Int,
                Literal::Float(_) => Type::Float,
                Literal::String(_) => Type::String,
                Literal::Bool(_) => Type::Bool,
                Literal::Null => Type::Unit,
            },
            ToolExpr::MapLiteral { entries } => {
                let mut value_ty: Option<Type> = None;
                for entry in entries {
                    let ty = self.check_tool_expr(&entry.value, scope);
                    value_ty = match value_ty {
                        None => Some(ty),
                        Some(prev) => {
                            if prev.is_compatible_with(&ty) {
                                Some(prev)
                            } else if ty.is_compatible_with(&prev) {
                                Some(ty)
                            } else {
                                Some(Type::Any)
                            }
                        }
                    };
                }
                Type::Map(
                    Box::new(Type::String),
                    Box::new(value_ty.unwrap_or(Type::Any)),
                )
            }
            ToolExpr::Expr(inner) => self.check_expr_with_scope(inner, scope),
        }
    }

    fn check_call_against_input_type(
        &mut self,
        function: &str,
        input_ty: &Type,
        args: &[Type],
        arg_names: &[Option<String>],
        span: Span,
    ) {
        let resolved_input = self.env.resolve_type(input_ty);
        match resolved_input {
            Type::Struct(s) if s.fields.is_empty() => {
                if !args.is_empty() {
                    self.errors.push(TypeError::new(
                        format!("'{}' expects no arguments", function),
                        span,
                    ));
                }
            }
            Type::Struct(s) => {
                // Strict struct-input matching:
                // - Preferred: one argument compatible with the whole struct
                // - One-field shorthand: tool(value) for input { field: T }
                // - Deterministic named-field form:
                //   tool(field_a, field_b, ...) where each arg is an identifier
                //   matching a struct field name exactly; type checked per-field.
                if args.len() == 1 {
                    let actual = &args[0];
                    if actual.is_error() {
                        return;
                    }

                    if actual.is_compatible_with(input_ty) {
                        return;
                    }

                    // One-field shorthand
                    if s.fields.len() == 1 {
                        let field_ty = s.fields.values().next().expect("single-field struct");
                        if actual.is_compatible_with(field_ty) {
                            return;
                        }
                    }

                    self.errors.push(TypeError::new(
                        format!(
                            "'{}' expected argument compatible with struct input {}, found {}",
                            function, input_ty, actual
                        ),
                        span,
                    ));
                    return;
                }

                if args.len() == s.fields.len() && arg_names.len() == args.len() {
                    let mut seen = std::collections::HashSet::new();
                    let mut ok = true;
                    for (idx, (actual_ty, maybe_name)) in
                        args.iter().zip(arg_names.iter()).enumerate()
                    {
                        let Some(name) = maybe_name else {
                            ok = false;
                            break;
                        };
                        if !seen.insert(name.clone()) {
                            ok = false;
                            break;
                        }
                        let Some(expected_ty) = s.fields.get(name) else {
                            ok = false;
                            break;
                        };
                        if !actual_ty.is_compatible_with(expected_ty) && !actual_ty.is_error() {
                            self.errors.push(TypeError::new(
                                format!(
                                    "'{}' field argument '{}' expected {}, found {}",
                                    function, name, expected_ty, actual_ty
                                ),
                                span,
                            ));
                            // Keep checking remaining args for full diagnostics.
                        }
                        if idx == args.len() - 1 && seen.len() != s.fields.len() {
                            ok = false;
                        }
                    }
                    if ok {
                        return;
                    }
                }

                self.errors.push(TypeError::new(
                    format!(
                        "'{}' expects either 1 struct argument of type {} or named field arguments matching {} fields",
                        function, input_ty, s.fields.len()
                    ),
                    span,
                ));
            }
            other => {
                if args.len() != 1 {
                    self.errors.push(TypeError::new(
                        format!(
                            "'{}' expects 1 argument of type {}, got {}",
                            function,
                            other,
                            args.len()
                        ),
                        span,
                    ));
                    return;
                }
                if !args[0].is_compatible_with(&other) && !args[0].is_error() {
                    self.errors.push(TypeError::new(
                        format!(
                            "'{}' expected argument of type {}, found {}",
                            function, other, args[0]
                        ),
                        span,
                    ));
                }
            }
        }
    }

    fn check_foreign_call(
        &mut self,
        module: &str,
        function: &str,
        arg_tys: &[Type],
        span: Span,
    ) -> Type {
        let key = (module.to_string(), function.to_string());
        if let Some(sig) = self.foreign_sigs.get(&key).cloned() {
            if sig.params.len() != arg_tys.len() {
                self.errors.push(TypeError::new(
                    format!(
                        "foreign call '{}::{}' expects {} arguments, got {}",
                        module,
                        function,
                        sig.params.len(),
                        arg_tys.len()
                    ),
                    span,
                ));
                return Type::Error;
            }

            for (idx, (expected, actual)) in sig.params.iter().zip(arg_tys.iter()).enumerate() {
                if !actual.is_compatible_with(expected) && !actual.is_error() {
                    self.errors.push(TypeError::new(
                        format!(
                            "foreign call '{}::{}' argument {} expected {}, found {}",
                            module,
                            function,
                            idx + 1,
                            expected,
                            actual
                        ),
                        span,
                    ));
                }
            }
            sig.return_type
        } else {
            self.errors.push(TypeError::new(
                format!("undefined foreign function '{}::{}'", module, function),
                span,
            ));
            Type::Error
        }
    }

    fn check_type_decl(&mut self, decl: &TypeDecl) {
        // Ensure all referenced types exist
        self.validate_type_expr(&decl.ty);
    }

    fn check_artifact_decl(&mut self, decl: &ArtifactDecl) {
        self.validate_type_expr(&decl.ty);
    }

    fn check_task(&mut self, task_decl: &TaskDecl) {
        self.validate_type_expr(&task_decl.input);
        self.validate_type_expr(&task_decl.output);
        for artifact in &task_decl.artifacts {
            self.validate_type_expr(&artifact.ty);
        }

        let mut scope = HashMap::new();
        let input_ty = self.resolve_type_expr(&task_decl.input);
        scope.insert("input".to_string(), input_ty.clone());
        if let Type::Struct(s) = self.env.resolve_type(&input_ty) {
            for (field, ty) in s.fields {
                scope.insert(field, ty);
            }
        }
        for artifact in &task_decl.artifacts {
            scope.insert(
                artifact.name.node.clone(),
                self.resolve_type_expr(&artifact.ty),
            );
        }

        for node in &task_decl.nodes {
            self.check_task_node(node, &scope);
        }
        for emit in &task_decl.emit {
            self.check_expr_with_scope(&emit.value, &scope);
        }
    }

    fn check_task_node(&mut self, node: &TaskNode, scope: &HashMap<String, Type>) {
        match node {
            TaskNode::Stage(stage) => {
                self.check_expr_with_scope(&stage.input, scope);
                if let Some(when) = &stage.when {
                    let when_ty = self.check_expr_with_scope(when, scope);
                    if !when_ty.is_compatible_with(&Type::Bool) && !when_ty.is_error() {
                        self.errors.push(TypeError::new(
                            format!("stage 'when' condition must be bool, found {}", when_ty),
                            when.span,
                        ));
                    }
                }
            }
            TaskNode::Loop(loop_decl) => {
                let max_iters_ty = self.check_expr_with_scope(&loop_decl.max_iters, scope);
                if !max_iters_ty.is_compatible_with(&Type::Int)
                    && !max_iters_ty.is_compatible_with(&Type::Float)
                    && !max_iters_ty.is_error()
                {
                    self.errors.push(TypeError::new(
                        format!("loop 'max_iters' must be numeric, found {}", max_iters_ty),
                        loop_decl.max_iters.span,
                    ));
                }

                let until_ty = self.check_expr_with_scope(&loop_decl.until, scope);
                if !until_ty.is_compatible_with(&Type::Bool) && !until_ty.is_error() {
                    self.errors.push(TypeError::new(
                        format!("loop 'until' condition must be bool, found {}", until_ty),
                        loop_decl.until.span,
                    ));
                }

                for node in &loop_decl.nodes {
                    self.check_task_node(node, scope);
                }
            }
            TaskNode::Branch(branch) => {
                let cond_ty = self.check_expr_with_scope(&branch.condition, scope);
                if !cond_ty.is_compatible_with(&Type::Bool) && !cond_ty.is_error() {
                    self.errors.push(TypeError::new(
                        format!("branch condition must be bool, found {}", cond_ty),
                        branch.condition.span,
                    ));
                }
                for node in &branch.then_nodes {
                    self.check_task_node(node, scope);
                }
                for node in &branch.else_nodes {
                    self.check_task_node(node, scope);
                }
            }
        }
    }

    fn check_harness(&mut self, harness_decl: &HarnessDecl) {
        for binding in &harness_decl.defaults {
            self.check_expr(&binding.value);
        }
        for bind in &harness_decl.binds {
            for binding in &bind.bindings {
                self.check_expr(&binding.value);
            }
        }
        for tune in &harness_decl.tune {
            if let FiniteDomain::List(values) = &tune.domain {
                for value in values {
                    self.check_expr(value);
                }
            }
        }
    }

    fn check_objective(&mut self, objective_decl: &ObjectiveDecl) {
        match &objective_decl.dataset {
            DatasetSpec::File(_) => {}
            DatasetSpec::Inline(cases) => {
                for case in cases {
                    self.check_expr(&case.input);
                    if let Some(expected) = &case.expected {
                        self.check_expr(expected);
                    }
                }
            }
        }

        let mut scope = HashMap::new();
        scope.insert("input".to_string(), Type::Any);
        scope.insert("expected".to_string(), Type::Any);
        scope.insert("output".to_string(), Type::Any);
        scope.insert("rollout".to_string(), Type::Any);

        for metric in &objective_decl.metrics {
            self.check_expr_with_scope(&metric.expr, &scope);
            scope.insert(metric.name.node.clone(), Type::Any);
        }
        self.check_expr_with_scope(&objective_decl.score, &scope);

        if let Some(select) = &objective_decl.select {
            self.check_expr_with_scope(&select.primary, &scope);
            for expr in &select.tie_breakers {
                self.check_expr_with_scope(expr, &scope);
            }
        }
    }

    fn check_expr(&mut self, expr: &Spanned<Expr>) -> Type {
        let scope = HashMap::new();
        self.check_expr_with_scope(expr, &scope)
    }

    fn merge_literal_types(&mut self, current: Type, next: Type, span: Span) -> Type {
        let current = self.env.resolve_type(&current);
        let next = self.env.resolve_type(&next);

        if current.is_error() || next.is_error() {
            return Type::Error;
        }

        if current.is_compatible_with(&next) {
            if matches!(current, Type::Any) {
                return next;
            }
            if matches!(next, Type::Any) {
                return current;
            }
            if (matches!(current, Type::Int) && matches!(next, Type::Float))
                || (matches!(current, Type::Float) && matches!(next, Type::Int))
            {
                return Type::Float;
            }
            return current;
        }

        self.errors.push(TypeError::new(
            format!(
                "incompatible literal element types {} and {}",
                current, next
            ),
            span,
        ));
        Type::Error
    }

    fn check_expr_with_scope(
        &mut self,
        expr: &Spanned<Expr>,
        scope: &HashMap<String, Type>,
    ) -> Type {
        match &expr.node {
            Expr::Literal(lit) => match lit {
                Literal::Int(_) => Type::Int,
                Literal::Float(_) => Type::Float,
                Literal::String(_) => Type::String,
                Literal::Bool(_) => Type::Bool,
                Literal::Null => Type::Unit,
            },
            Expr::Ident(name) => {
                if let Some(ty) = scope.get(name) {
                    return ty.clone();
                }
                // Check variables first
                if let Some(ty) = self.env.lookup_variable(name) {
                    return ty.clone();
                }
                // Unknown identifier - could be a forward reference or external
                // For now, return Any to allow flexibility
                Type::Any
            }
            Expr::FieldAccess(base, field) => {
                let base_ty = self.check_expr_with_scope(base, scope);
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
                let left_ty = self.check_expr_with_scope(left, scope);
                let right_ty = self.check_expr_with_scope(right, scope);

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
            Expr::Call(name, args) => {
                let arg_tys = args
                    .iter()
                    .map(|a| self.check_expr_with_scope(a, scope))
                    .collect::<Vec<_>>();

                // Built-ins
                match name.as_str() {
                    "len" => return Type::Int,
                    "is_empty" | "contains" | "is_some" | "is_none" | "not" => return Type::Bool,
                    "unwrap" => {
                        if let Some(first) = arg_tys.first() {
                            return match self.env.resolve_type(first) {
                                Type::Option(inner) => *inner,
                                Type::Result(ok, _err) => *ok,
                                _ => Type::Any,
                            };
                        }
                        return Type::Any;
                    }
                    "unwrap_or" => {
                        if let Some(first) = arg_tys.first() {
                            return match self.env.resolve_type(first) {
                                Type::Option(inner) => *inner,
                                Type::Result(ok, _err) => *ok,
                                other => other,
                            };
                        }
                        return Type::Any;
                    }
                    "abs" | "min" | "max" => {
                        return if arg_tys
                            .iter()
                            .any(|t| matches!(self.env.resolve_type(t), Type::Float))
                        {
                            Type::Float
                        } else {
                            Type::Int
                        };
                    }
                    _ => {}
                }

                if let Some(sig) = self.tool_sigs.get(name).cloned() {
                    let arg_names = args
                        .iter()
                        .map(|a| match &a.node {
                            Expr::Ident(n) => Some(n.clone()),
                            _ => None,
                        })
                        .collect::<Vec<_>>();
                    self.check_call_against_input_type(
                        name, &sig.input, &arg_tys, &arg_names, expr.span,
                    );
                    return sig.output;
                }

                // Unknown call target - keep flexible.
                for arg in args {
                    self.check_expr_with_scope(arg, scope);
                }
                Type::Any
            }
            Expr::ForeignCall {
                module,
                function,
                args,
            } => {
                let arg_tys = args
                    .iter()
                    .map(|a| self.check_expr_with_scope(a, scope))
                    .collect::<Vec<_>>();
                self.check_foreign_call(module, function, &arg_tys, expr.span)
            }
            Expr::ListLiteral(items) => {
                if items.is_empty() {
                    Type::List(Box::new(Type::Any))
                } else {
                    let mut item_ty = self.check_expr_with_scope(&items[0], scope);
                    for item in &items[1..] {
                        let next_ty = self.check_expr_with_scope(item, scope);
                        item_ty = self.merge_literal_types(item_ty, next_ty, item.span);
                    }

                    if item_ty.is_error() {
                        Type::Error
                    } else {
                        Type::List(Box::new(item_ty))
                    }
                }
            }
            Expr::RecordLiteral(fields) => {
                let mut out = HashMap::new();
                for field in fields {
                    let value_ty = self.check_expr_with_scope(&field.value, scope);
                    if out.insert(field.key.node.clone(), value_ty).is_some() {
                        self.errors.push(TypeError::new(
                            format!("duplicate field '{}' in record literal", field.key.node),
                            field.key.span,
                        ));
                    }
                }
                Type::Struct(StructType::with_fields(out))
            }
            Expr::Paren(inner) => self.check_expr_with_scope(inner, scope),
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
            tool demo_tool {
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
            .contains("undefined tool, prompt, or agent 'nonexistent_call'")));
    }

    #[test]
    fn test_valid_pipeline_references() {
        let source = r#"
            tool my_tool {
                input: { x: int }
                output: { y: int }
            }

            prompt my_prompt {
                input: { y: int }
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

    #[test]
    fn test_foreign_call_accepts_named_scaffold_type() {
        let source = r#"
            type SearchInput = { query: string, max_results: int }
            type SearchPayload = { source_urls: string, source_snippets: string, error: string }

            foreign rust web_fetch {
                fn wiki_search(req: SearchInput) -> SearchPayload
            }

            tool wiki_search {
                input: SearchInput
                output: SearchPayload
                impl: web_fetch::wiki_search(input)
            }
        "#;

        let program = parse(source).unwrap();
        let result = check(&program);
        assert!(result.is_ok());
    }

    #[test]
    fn test_foreign_call_rejects_wrong_named_type_argument() {
        let source = r#"
            type SearchInput = { query: string, max_results: int }
            type SearchPayload = { source_urls: string, source_snippets: string, error: string }

            foreign rust web_fetch {
                fn wiki_search(req: SearchInput) -> SearchPayload
            }

            tool wiki_search {
                input: SearchInput
                output: SearchPayload
                impl: web_fetch::wiki_search(max_results)
            }
        "#;

        let program = parse(source).unwrap();
        let result = check(&program);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| e
            .message
            .contains("foreign call 'web_fetch::wiki_search' argument 1 expected")));
    }

    #[test]
    fn test_struct_tool_call_rejects_multi_arg_positional_form() {
        let source = r#"
            type SearchInput = { query: string, max_results: int }
            type SearchPayload = { source_urls: string, source_snippets: string, error: string }

            tool inner {
                input: SearchInput
                output: SearchPayload
                impl: json { source_urls: "", source_snippets: "", error: "" }
            }

            tool caller {
                input: SearchInput
                output: SearchPayload
                impl: inner(query, 7)
            }
        "#;

        let program = parse(source).unwrap();
        let result = check(&program);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors
            .iter()
            .any(|e| e.message.contains("expects either 1 struct argument")));
    }

    #[test]
    fn test_struct_tool_call_accepts_single_field_shorthand() {
        let source = r#"
            type One = { x: int }

            tool inner {
                input: One
                output: int
                impl: x
            }

            tool caller {
                input: One
                output: int
                impl: inner(x)
            }
        "#;

        let program = parse(source).unwrap();
        let result = check(&program);
        assert!(result.is_ok());
    }
}
