//! Serialization and AST-to-IR lowering

use std::collections::HashMap;

use scaffold_syntax::ast::*;
use scaffold_types::TypeEnv;

use crate::ir::*;

/// Error during IR lowering
#[derive(Debug, Clone)]
pub struct LowerError {
    pub message: String,
    pub span: Span,
}

impl LowerError {
    pub fn new(message: impl Into<String>, span: Span) -> Self {
        Self {
            message: message.into(),
            span,
        }
    }
}

pub type LowerResult<T> = Result<T, LowerError>;

/// Lower AST to IR
pub struct Lowerer {
    /// Source file name (for debugging)
    source_file: Option<String>,
}

impl Lowerer {
    pub fn new() -> Self {
        Self { source_file: None }
    }

    pub fn with_source_file(mut self, file: String) -> Self {
        self.source_file = Some(file);
        self
    }

    /// Lower a complete program to IR
    pub fn lower(
        &self,
        program: &Program,
        _type_env: &TypeEnv,
    ) -> LowerResult<ScaffoldIR> {
        let mut ir = ScaffoldIR::new();

        for decl in &program.declarations {
            match decl {
                Declaration::Type(type_decl) => {
                    ir.types.push(self.lower_type_decl(type_decl)?);
                }
                Declaration::ExternCrate(extern_crate) => {
                    ir.extern_crates.push(self.lower_extern_crate(extern_crate)?);
                }
                Declaration::Foreign(foreign) => {
                    ir.foreign_modules.push(self.lower_foreign(foreign)?);
                }
                Declaration::Tool(tool) => {
                    ir.tools.push(self.lower_tool(tool)?);
                }
                Declaration::Prompt(prompt) => {
                    ir.prompts.push(self.lower_prompt(prompt)?);
                }
                Declaration::Agent(agent) => {
                    ir.agents.push(self.lower_agent(agent)?);
                }
                Declaration::Pipeline(pipeline) => {
                    ir.pipelines.push(self.lower_pipeline(pipeline)?);
                }
            }
        }

        Ok(ir)
    }

    fn lower_type_decl(&self, type_decl: &TypeDecl) -> LowerResult<TypeDefIR> {
        Ok(TypeDefIR {
            name: type_decl.name.node.clone(),
            definition: self.lower_type_expr(&type_decl.ty.node)?,
        })
    }

    fn lower_type_ref(&self, ty: &Spanned<TypeExpr>) -> LowerResult<TypeRefIR> {
        match &ty.node {
            TypeExpr::Named(name) => Ok(TypeRefIR::Named { ref_name: name.clone() }),
            _ => Ok(TypeRefIR::Inline(self.lower_type_expr(&ty.node)?)),
        }
    }

    fn lower_type_expr(&self, ty: &TypeExpr) -> LowerResult<TypeIR> {
        match ty {
            TypeExpr::Primitive(p) => match p {
                PrimitiveType::Bool => Ok(TypeIR::Bool),
                PrimitiveType::Int => Ok(TypeIR::Int),
                PrimitiveType::Float => Ok(TypeIR::Float),
                PrimitiveType::String => Ok(TypeIR::String),
                PrimitiveType::Bytes => Ok(TypeIR::Bytes),
                PrimitiveType::Any => Ok(TypeIR::Any),
            },
            TypeExpr::Named(name) => Ok(TypeIR::Named { name: name.clone() }),
            TypeExpr::List(inner) => Ok(TypeIR::List {
                element: Box::new(self.lower_type_expr(&inner.node)?),
            }),
            TypeExpr::Map(key, value) => Ok(TypeIR::Map {
                key: Box::new(self.lower_type_expr(&key.node)?),
                value: Box::new(self.lower_type_expr(&value.node)?),
            }),
            TypeExpr::Option(inner) => Ok(TypeIR::Option {
                inner: Box::new(self.lower_type_expr(&inner.node)?),
            }),
            TypeExpr::Result(ok, err) => Ok(TypeIR::Result {
                ok: Box::new(self.lower_type_expr(&ok.node)?),
                err: Box::new(self.lower_type_expr(&err.node)?),
            }),
            TypeExpr::Struct(fields) => {
                let mut ir_fields = HashMap::new();
                for field in fields {
                    ir_fields.insert(
                        field.name.node.clone(),
                        self.lower_type_expr(&field.ty.node)?,
                    );
                }
                Ok(TypeIR::Struct { fields: ir_fields })
            }
        }
    }

    fn lower_expr(&self, expr: &Expr) -> LowerResult<ExprIR> {
        match expr {
            Expr::Literal(lit) => Ok(ExprIR::Literal {
                value: self.lower_literal(lit),
            }),
            Expr::Ident(name) => Ok(ExprIR::Ident { name: name.clone() }),
            Expr::FieldAccess(base, field) => Ok(ExprIR::FieldAccess {
                base: Box::new(self.lower_expr(&base.node)?),
                field: field.node.clone(),
            }),
            Expr::Binary(left, op, right) => Ok(ExprIR::Binary {
                left: Box::new(self.lower_expr(&left.node)?),
                op: op.to_string(),
                right: Box::new(self.lower_expr(&right.node)?),
            }),
            Expr::Call(name, args) => {
                let mut ir_args = Vec::new();
                for arg in args {
                    ir_args.push(self.lower_expr(&arg.node)?);
                }
                Ok(ExprIR::Call {
                    function: name.clone(),
                    args: ir_args,
                })
            }
            Expr::Paren(inner) => self.lower_expr(&inner.node),
        }
    }

    fn lower_literal(&self, lit: &Literal) -> LiteralIR {
        match lit {
            Literal::Int(n) => LiteralIR::Int { value: *n },
            Literal::Float(n) => LiteralIR::Float { value: *n },
            Literal::String(s) => LiteralIR::String { value: s.clone() },
            Literal::Bool(b) => LiteralIR::Bool { value: *b },
            Literal::Null => LiteralIR::Null,
        }
    }

    #[allow(dead_code)]
    fn make_span(&self, span: Span) -> SourceSpanIR {
        let mut ir_span = SourceSpanIR::new(span.start, span.end);
        if let Some(ref file) = self.source_file {
            ir_span = ir_span.with_file(file.clone());
        }
        ir_span
    }

    // ============================================
    // Foreign and Tool Lowering
    // ============================================

    fn lower_extern_crate(&self, decl: &ExternCrateDecl) -> LowerResult<ExternCrateIR> {
        Ok(ExternCrateIR {
            name: decl.name.node.clone(),
            version: decl.version.clone(),
            features: decl.features.clone(),
        })
    }

    fn lower_foreign(&self, decl: &ForeignDecl) -> LowerResult<ForeignModuleIR> {
        let mut type_aliases = Vec::new();
        for alias in &decl.type_aliases {
            type_aliases.push(ForeignTypeAliasIR {
                name: alias.name.node.clone(),
                external_type: alias.external_type.clone(),
            });
        }

        let mut functions = Vec::new();
        for func in &decl.functions {
            let mut params = Vec::new();
            for param in &func.params {
                params.push(ForeignParamIR {
                    name: param.name.node.clone(),
                    ty: self.lower_type_expr(&param.ty.node)?,
                });
            }
            functions.push(ForeignFnIR {
                name: func.name.node.clone(),
                params,
                return_type: self.lower_type_expr(&func.return_type.node)?,
            });
        }

        Ok(ForeignModuleIR {
            language: decl.language.node.clone(),
            name: decl.name.node.clone(),
            type_aliases,
            functions,
        })
    }

    fn lower_tool(&self, decl: &ToolDecl) -> LowerResult<ToolIR> {
        let implementation = match &decl.implementation {
            Some(impl_) => Some(self.lower_tool_impl(impl_)?),
            None => None,
        };

        let spec = match &decl.spec {
            Some(spec) => Some(self.lower_tool_spec(spec)?),
            None => None,
        };

        let mut variants = Vec::new();
        for variant in &decl.variants {
            variants.push(ToolVariantIR {
                name: variant.name.node.clone(),
                implementation: self.lower_tool_impl(&variant.implementation)?,
            });
        }

        Ok(ToolIR {
            name: decl.name.node.clone(),
            input: self.lower_type_expr(&decl.input.node)?,
            output: self.lower_type_expr(&decl.output.node)?,
            implementation,
            spec,
            variants,
        })
    }

    fn lower_tool_impl(&self, impl_: &ToolImpl) -> LowerResult<ToolImplIR> {
        match impl_ {
            ToolImpl::Expr(expr) => Ok(ToolImplIR::Expr {
                expr: self.lower_tool_expr(&expr.node)?,
            }),
            ToolImpl::Sequence(stmts) => {
                let mut ir_stmts = Vec::new();
                for stmt in stmts {
                    ir_stmts.push(self.lower_tool_statement(stmt)?);
                }
                Ok(ToolImplIR::Sequence { statements: ir_stmts })
            }
            ToolImpl::Parallel(stmts) => {
                let mut ir_stmts = Vec::new();
                for stmt in stmts {
                    ir_stmts.push(self.lower_tool_statement(stmt)?);
                }
                Ok(ToolImplIR::Parallel { statements: ir_stmts })
            }
        }
    }

    fn lower_tool_expr(&self, expr: &ToolExpr) -> LowerResult<ToolExprIR> {
        match expr {
            ToolExpr::Ident(name) => Ok(ToolExprIR::Ident { name: name.clone() }),
            ToolExpr::FieldAccess(base, field) => Ok(ToolExprIR::FieldAccess {
                base: Box::new(self.lower_tool_expr(&base.node)?),
                field: field.node.clone(),
            }),
            ToolExpr::ForeignCall { module, function, args } => {
                let mut ir_args = Vec::new();
                for arg in args {
                    ir_args.push(self.lower_tool_expr(&arg.node)?);
                }
                Ok(ToolExprIR::ForeignCall {
                    module: module.clone(),
                    function: function.clone(),
                    args: ir_args,
                })
            }
            ToolExpr::ToolCall { tool, args } => {
                let mut ir_args = Vec::new();
                for arg in args {
                    ir_args.push(self.lower_tool_expr(&arg.node)?);
                }
                Ok(ToolExprIR::ToolCall {
                    tool: tool.clone(),
                    args: ir_args,
                })
            }
            ToolExpr::Shell(cmd) => Ok(ToolExprIR::Shell { command: cmd.clone() }),
            ToolExpr::Pipe(left, right) => Ok(ToolExprIR::Pipe {
                left: Box::new(self.lower_tool_expr(&left.node)?),
                right: Box::new(self.lower_tool_expr(&right.node)?),
            }),
            ToolExpr::If { condition, then_branch, else_branch } => Ok(ToolExprIR::If {
                condition: self.lower_expr(&condition.node)?,
                then_branch: Box::new(self.lower_tool_impl(then_branch)?),
                else_branch: match else_branch {
                    Some(branch) => Some(Box::new(self.lower_tool_impl(branch)?)),
                    None => None,
                },
            }),
            ToolExpr::Match { scrutinee, arms } => {
                let mut ir_arms = Vec::new();
                for arm in arms {
                    ir_arms.push(MatchArmIR {
                        pattern: self.lower_expr(&arm.pattern.node)?,
                        body: self.lower_tool_impl(&arm.body)?,
                    });
                }
                Ok(ToolExprIR::Match {
                    scrutinee: Box::new(self.lower_tool_expr(&scrutinee.node)?),
                    arms: ir_arms,
                })
            }
            ToolExpr::For { variable, iterable, body } => Ok(ToolExprIR::For {
                variable: variable.node.clone(),
                iterable: Box::new(self.lower_tool_expr(&iterable.node)?),
                body: Box::new(self.lower_tool_impl(body)?),
            }),
            ToolExpr::While { condition, body } => Ok(ToolExprIR::While {
                condition: self.lower_expr(&condition.node)?,
                body: Box::new(self.lower_tool_impl(body)?),
            }),
            ToolExpr::Loop { body } => Ok(ToolExprIR::Loop {
                body: Box::new(self.lower_tool_impl(body)?),
            }),
            ToolExpr::Break => Ok(ToolExprIR::Break),
            ToolExpr::Continue => Ok(ToolExprIR::Continue),
            ToolExpr::Literal(lit) => Ok(ToolExprIR::Literal {
                value: self.lower_literal(lit),
            }),
            ToolExpr::Expr(expr) => Ok(ToolExprIR::Expr {
                expr: Box::new(self.lower_expr(&expr.node)?),
            }),
        }
    }

    fn lower_tool_statement(&self, stmt: &ToolStatement) -> LowerResult<ToolStatementIR> {
        Ok(ToolStatementIR {
            binding: stmt.binding.as_ref().map(|b| b.node.clone()),
            expr: self.lower_tool_expr(&stmt.expr.node)?,
        })
    }

    fn lower_tool_spec(&self, spec: &ToolSpec) -> LowerResult<ToolSpecIR> {
        let mut preconditions = Vec::new();
        for pre in &spec.preconditions {
            preconditions.push(self.lower_expr(&pre.node)?);
        }

        let mut postconditions = Vec::new();
        for post in &spec.postconditions {
            postconditions.push(self.lower_expr(&post.node)?);
        }

        Ok(ToolSpecIR {
            preconditions,
            postconditions,
            pure: spec.pure,
        })
    }

    // ============================================
    // Prompt, Agent, Pipeline Lowering
    // ============================================

    fn lower_prompt(&self, decl: &PromptDecl) -> LowerResult<PromptIR> {
        Ok(PromptIR {
            name: decl.name.node.clone(),
            input: self.lower_type_expr(&decl.input.node)?,
            output: self.lower_type_expr(&decl.output.node)?,
            template: self.lower_string_or_file(&decl.template),
            system: decl.system.as_ref().map(|s| self.lower_string_or_file(s)),
        })
    }

    fn lower_agent(&self, decl: &AgentDecl) -> LowerResult<AgentIR> {
        let reward = match &decl.reward {
            Some(expr) => Some(self.lower_expr(&expr.node)?),
            None => None,
        };
        let done = match &decl.done {
            Some(expr) => Some(self.lower_expr(&expr.node)?),
            None => None,
        };
        let on_error = self.lower_error_strategy(&decl.on_error);

        Ok(AgentIR {
            name: decl.name.node.clone(),
            input: self.lower_type_expr(&decl.input.node)?,
            output: self.lower_type_expr(&decl.output.node)?,
            tools: decl.tools.iter().map(|t| t.node.clone()).collect(),
            system: self.lower_string_or_file(&decl.system),
            max_turns: decl.max_turns,
            reward,
            done,
            on_error,
            timeout: decl.timeout,
        })
    }

    fn lower_error_strategy(&self, strategy: &scaffold_syntax::ast::ErrorStrategy) -> ErrorStrategyIR {
        match strategy {
            scaffold_syntax::ast::ErrorStrategy::Abort => ErrorStrategyIR::Abort,
            scaffold_syntax::ast::ErrorStrategy::Retry(count) => ErrorStrategyIR::Retry { count: *count },
        }
    }

    fn lower_pipeline(&self, decl: &PipelineDecl) -> LowerResult<PipelineIR> {
        let mut steps = Vec::new();
        for step in &decl.steps {
            steps.push(self.lower_pipeline_step(step)?);
        }

        let reward = match &decl.reward {
            Some(expr) => Some(self.lower_expr(&expr.node)?),
            None => None,
        };

        Ok(PipelineIR {
            name: decl.name.node.clone(),
            input: self.lower_type_expr(&decl.input.node)?,
            output: self.lower_type_expr(&decl.output.node)?,
            steps,
            reward,
        })
    }

    fn lower_pipeline_step(&self, step: &PipelineStep) -> LowerResult<PipelineStepIR> {
        let call = match &step.call {
            PipelineCall::Prompt { name, args } => {
                let mut ir_args = Vec::new();
                for arg in args {
                    ir_args.push(self.lower_tool_expr(&arg.node)?);
                }
                PipelineCallIR::Prompt {
                    name: name.clone(),
                    args: ir_args,
                }
            }
            PipelineCall::Tool { name, args } => {
                let mut ir_args = Vec::new();
                for arg in args {
                    ir_args.push(self.lower_tool_expr(&arg.node)?);
                }
                PipelineCallIR::Tool {
                    name: name.clone(),
                    args: ir_args,
                }
            }
            PipelineCall::Expr(expr) => {
                let e = self.lower_tool_expr(&expr.node)?;
                PipelineCallIR::Expr { expr: e }
            }
        };

        Ok(PipelineStepIR {
            binding: step.binding.as_ref().map(|b| b.node.clone()),
            call,
        })
    }

    fn lower_string_or_file(&self, sof: &StringOrFile) -> StringOrFileIR {
        match sof {
            StringOrFile::Literal(s) => StringOrFileIR::Literal { value: s.clone() },
            StringOrFile::File(path) => StringOrFileIR::File { path: path.clone() },
        }
    }
}

impl Default for Lowerer {
    fn default() -> Self {
        Self::new()
    }
}

/// Serialize IR to JSON
pub fn to_json(ir: &ScaffoldIR) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(ir)
}

/// Serialize IR to compact JSON
pub fn to_json_compact(ir: &ScaffoldIR) -> Result<String, serde_json::Error> {
    serde_json::to_string(ir)
}

/// Deserialize IR from JSON
pub fn from_json(json: &str) -> Result<ScaffoldIR, serde_json::Error> {
    serde_json::from_str(json)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scaffold_syntax::parse;
    use scaffold_types::check;

    #[test]
    fn test_lower_and_serialize() {
        let source = r#"
            type Position = { x: int, y: int }

            tool get_position {
                input: { id: int }
                output: Position
            }

            prompt summarize {
                input: { text: string }
                output: { summary: string }
                template: "Summarize: {text}"
            }

            agent researcher {
                input: { query: string }
                output: { answer: string }
                tools: [get_position]
                system: "You are a researcher."
            }

            pipeline main_pipeline {
                input: { data: string }
                output: { response: string }
                steps {
                    let summary = summarize(data)
                }
            }
        "#;

        let program = parse(source).unwrap();
        let type_env = check(&program).unwrap();

        let lowerer = Lowerer::new().with_source_file("test.scaffold".to_string());
        let ir = lowerer.lower(&program, &type_env).unwrap();

        // Check basic structure
        assert_eq!(ir.version, IR_VERSION);
        assert_eq!(ir.types.len(), 1);
        assert_eq!(ir.tools.len(), 1);
        assert_eq!(ir.prompts.len(), 1);
        assert_eq!(ir.agents.len(), 1);
        assert_eq!(ir.pipelines.len(), 1);

        // Serialize and deserialize
        let json = to_json(&ir).unwrap();
        let deserialized = from_json(&json).unwrap();

        assert_eq!(deserialized.types.len(), ir.types.len());
        assert_eq!(deserialized.tools[0].name, "get_position");
        assert_eq!(deserialized.prompts[0].name, "summarize");
        assert_eq!(deserialized.agents[0].name, "researcher");
        assert_eq!(deserialized.pipelines[0].name, "main_pipeline");
    }
}
