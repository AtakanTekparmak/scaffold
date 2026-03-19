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
    /// Known prompt names (for classifying pipeline calls)
    prompt_names: std::collections::HashSet<String>,
    /// Known agent names (for classifying pipeline calls)
    agent_names: std::collections::HashSet<String>,
}

impl Lowerer {
    pub fn new() -> Self {
        Self {
            source_file: None,
            prompt_names: std::collections::HashSet::new(),
            agent_names: std::collections::HashSet::new(),
        }
    }

    pub fn with_source_file(mut self, file: String) -> Self {
        self.source_file = Some(file);
        self
    }

    /// Lower a complete program to IR
    pub fn lower(&self, program: &Program, _type_env: &TypeEnv) -> LowerResult<ScaffoldIR> {
        // First pass: collect tool, prompt, and agent names for classifying pipeline calls
        let mut prompt_names = std::collections::HashSet::new();
        let mut agent_names = std::collections::HashSet::new();

        for decl in &program.declarations {
            match decl {
                Declaration::Tool(_) => {}
                Declaration::Prompt(prompt) => {
                    prompt_names.insert(prompt.name.node.clone());
                }
                Declaration::Agent(agent) => {
                    agent_names.insert(agent.name.node.clone());
                }
                _ => {}
            }
        }

        // Create a new lowerer with the collected names
        let lowerer = Lowerer {
            source_file: self.source_file.clone(),
            prompt_names,
            agent_names,
        };

        // Second pass: lower all declarations
        let mut ir = ScaffoldIR::new();

        for decl in &program.declarations {
            match decl {
                Declaration::Type(type_decl) => {
                    ir.types.push(lowerer.lower_type_decl(type_decl)?);
                }
                Declaration::Artifact(artifact_decl) => {
                    ir.types.push(lowerer.lower_artifact_decl(artifact_decl)?);
                }
                Declaration::ExternCrate(extern_crate) => {
                    ir.extern_crates
                        .push(lowerer.lower_extern_crate(extern_crate)?);
                }
                Declaration::Foreign(foreign) => {
                    ir.foreign_modules.push(lowerer.lower_foreign(foreign)?);
                }
                Declaration::Tool(tool) => {
                    ir.tools.push(lowerer.lower_tool(tool)?);
                }
                Declaration::Prompt(prompt) => {
                    ir.prompts.push(lowerer.lower_prompt(prompt)?);
                }
                Declaration::Agent(agent) => {
                    ir.agents.push(lowerer.lower_agent(agent)?);
                }
                Declaration::Pipeline(pipeline) => {
                    ir.pipelines.push(lowerer.lower_pipeline(pipeline)?);
                }
                Declaration::Task(task) => {
                    ir.tasks.push(lowerer.lower_task(task)?);
                }
                Declaration::Harness(harness) => {
                    ir.harnesses.push(lowerer.lower_harness(harness)?);
                }
                Declaration::Objective(objective) => {
                    ir.objectives.push(lowerer.lower_objective(objective)?);
                }
            }
        }

        Ok(ir)
    }

    fn lower_type_decl(&self, type_decl: &TypeDecl) -> LowerResult<TypeDefIR> {
        Ok(TypeDefIR {
            kind: TypeDefKindIR::Type,
            name: type_decl.name.node.clone(),
            definition: self.lower_type_expr(&type_decl.ty.node)?,
        })
    }

    fn lower_artifact_decl(&self, artifact_decl: &ArtifactDecl) -> LowerResult<TypeDefIR> {
        Ok(TypeDefIR {
            kind: TypeDefKindIR::Artifact,
            name: artifact_decl.name.node.clone(),
            definition: self.lower_type_expr(&artifact_decl.ty.node)?,
        })
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
            Expr::ForeignCall {
                module,
                function,
                args,
            } => {
                let mut ir_args = Vec::new();
                for arg in args {
                    ir_args.push(self.lower_expr(&arg.node)?);
                }
                Ok(ExprIR::ForeignCall {
                    module: module.clone(),
                    function: function.clone(),
                    args: ir_args,
                })
            }
            Expr::ListLiteral(items) => {
                let mut elements = Vec::with_capacity(items.len());
                for item in items {
                    elements.push(self.lower_expr(&item.node)?);
                }
                Ok(ExprIR::List { elements })
            }
            Expr::RecordLiteral(fields) => {
                let mut ir_fields = Vec::with_capacity(fields.len());
                for field in fields {
                    ir_fields.push(ExprFieldIR {
                        key: field.key.node.clone(),
                        value: self.lower_expr(&field.value.node)?,
                    });
                }
                Ok(ExprIR::Record { fields: ir_fields })
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
                Ok(ToolImplIR::Sequence {
                    statements: ir_stmts,
                })
            }
            ToolImpl::Parallel(stmts) => {
                let mut ir_stmts = Vec::new();
                for stmt in stmts {
                    ir_stmts.push(self.lower_tool_statement(stmt)?);
                }
                Ok(ToolImplIR::Parallel {
                    statements: ir_stmts,
                })
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
            ToolExpr::ForeignCall {
                module,
                function,
                args,
            } => {
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
            ToolExpr::Shell(cmd) => Ok(ToolExprIR::Shell {
                command: cmd.clone(),
            }),
            ToolExpr::Pipe(left, right) => Ok(ToolExprIR::Pipe {
                left: Box::new(self.lower_tool_expr(&left.node)?),
                right: Box::new(self.lower_tool_expr(&right.node)?),
            }),
            ToolExpr::If {
                condition,
                then_branch,
                else_branch,
            } => Ok(ToolExprIR::If {
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
            ToolExpr::For {
                variable,
                iterable,
                body,
            } => Ok(ToolExprIR::For {
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
            ToolExpr::MapLiteral { entries } => {
                let mut ir_entries = Vec::new();
                for entry in entries {
                    ir_entries.push(MapEntryIR {
                        key: entry.key.clone(),
                        value: self.lower_tool_expr(&entry.value.node)?,
                    });
                }
                Ok(ToolExprIR::MapLiteral {
                    entries: ir_entries,
                })
            }
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
            model: decl.model.clone(),
            max_turns: decl.max_turns,
            reward,
            done,
            on_error,
            timeout: decl.timeout,
        })
    }

    fn lower_error_strategy(
        &self,
        strategy: &scaffold_syntax::ast::ErrorStrategy,
    ) -> ErrorStrategyIR {
        match strategy {
            scaffold_syntax::ast::ErrorStrategy::Abort => ErrorStrategyIR::Abort,
            scaffold_syntax::ast::ErrorStrategy::Retry(count) => {
                ErrorStrategyIR::Retry { count: *count }
            }
        }
    }

    fn lower_pipeline(&self, decl: &PipelineDecl) -> LowerResult<PipelineIR> {
        let steps = self.lower_pipeline_steps(&decl.steps)?;

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

    fn lower_pipeline_steps(&self, steps: &[PipelineStep]) -> LowerResult<Vec<PipelineStepIR>> {
        let mut out = Vec::new();
        for step in steps {
            out.push(self.lower_pipeline_step(step)?);
        }
        Ok(out)
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
                // Check if this is actually a prompt call (parser doesn't distinguish)
                if self.prompt_names.contains(name) {
                    PipelineCallIR::Prompt {
                        name: name.clone(),
                        args: ir_args,
                    }
                } else if self.agent_names.contains(name) {
                    PipelineCallIR::Agent {
                        name: name.clone(),
                        args: ir_args,
                    }
                } else {
                    PipelineCallIR::Tool {
                        name: name.clone(),
                        args: ir_args,
                    }
                }
            }
            PipelineCall::Agent { name, args } => {
                let mut ir_args = Vec::new();
                for arg in args {
                    ir_args.push(self.lower_tool_expr(&arg.node)?);
                }
                PipelineCallIR::Agent {
                    name: name.clone(),
                    args: ir_args,
                }
            }
            PipelineCall::Expr(expr) => {
                let e = self.lower_tool_expr(&expr.node)?;
                PipelineCallIR::Expr { expr: e }
            }
            PipelineCall::Parallel { branches } => {
                let mut ir_branches = Vec::new();
                for branch in branches {
                    ir_branches.push(self.lower_pipeline_steps(branch)?);
                }
                PipelineCallIR::Parallel {
                    branches: ir_branches,
                }
            }
            PipelineCall::If {
                condition,
                then_steps,
                else_steps,
            } => {
                let cond = self.lower_expr(&condition.node)?;
                let then_ir = self.lower_pipeline_steps(then_steps)?;
                let else_ir = self.lower_pipeline_steps(else_steps)?;
                PipelineCallIR::If {
                    condition: cond,
                    then_steps: then_ir,
                    else_steps: else_ir,
                }
            }
            PipelineCall::Match { scrutinee, arms } => {
                let scrut = self.lower_expr(&scrutinee.node)?;
                let mut ir_arms = Vec::new();
                for arm in arms {
                    let pattern = self.lower_expr(&arm.pattern.node)?;
                    let steps = self.lower_pipeline_steps(&arm.steps)?;
                    ir_arms.push(PipelineMatchArmIR { pattern, steps });
                }
                PipelineCallIR::Match {
                    scrutinee: scrut,
                    arms: ir_arms,
                }
            }
        };

        Ok(PipelineStepIR {
            binding: step.binding.as_ref().map(|b| b.node.clone()),
            call,
        })
    }

    // ============================================
    // Task / Harness / Objective Lowering
    // ============================================

    fn lower_task(&self, decl: &TaskDecl) -> LowerResult<TaskIR> {
        let mut artifacts = Vec::with_capacity(decl.artifacts.len());
        for artifact in &decl.artifacts {
            artifacts.push(ArtifactSlotIR {
                name: artifact.name.node.clone(),
                ty: self.lower_type_expr(&artifact.ty.node)?,
            });
        }

        let mut body = Vec::with_capacity(decl.nodes.len());
        for node in &decl.nodes {
            body.push(self.lower_task_node(node)?);
        }

        let mut emit = Vec::with_capacity(decl.emit.len());
        for field in &decl.emit {
            emit.push(EmitFieldIR {
                name: field.name.node.clone(),
                value: self.lower_expr(&field.value.node)?,
            });
        }

        Ok(TaskIR {
            name: decl.name.node.clone(),
            input: self.lower_type_expr(&decl.input.node)?,
            output: self.lower_type_expr(&decl.output.node)?,
            artifacts,
            body,
            emit,
        })
    }

    fn lower_task_node(&self, node: &TaskNode) -> LowerResult<TaskNodeIR> {
        match node {
            TaskNode::Stage(stage) => Ok(TaskNodeIR::Stage(self.lower_stage(stage)?)),
            TaskNode::Loop(loop_decl) => Ok(TaskNodeIR::Loop(self.lower_loop(loop_decl)?)),
            TaskNode::Branch(branch) => Ok(TaskNodeIR::Branch(self.lower_branch(branch)?)),
        }
    }

    fn lower_stage(&self, stage: &StageDecl) -> LowerResult<StageIR> {
        Ok(StageIR {
            name: stage.name.node.clone(),
            stage_kind: self.lower_stage_kind(stage.kind),
            component: stage.component.node.clone(),
            input: self.lower_expr(&stage.input.node)?,
            output: stage.output.node.clone(),
            when: stage
                .when
                .as_ref()
                .map(|expr| self.lower_expr(&expr.node))
                .transpose()?,
        })
    }

    fn lower_loop(&self, loop_decl: &TaskLoopDecl) -> LowerResult<LoopIR> {
        let mut body = Vec::with_capacity(loop_decl.nodes.len());
        for node in &loop_decl.nodes {
            body.push(self.lower_task_node(node)?);
        }

        Ok(LoopIR {
            name: loop_decl.name.node.clone(),
            max_iters: self.lower_expr(&loop_decl.max_iters.node)?,
            carry: loop_decl
                .carry
                .iter()
                .map(|ident| ident.node.clone())
                .collect(),
            until: self.lower_expr(&loop_decl.until.node)?,
            body,
        })
    }

    fn lower_branch(&self, branch: &TaskBranchDecl) -> LowerResult<BranchIR> {
        let mut then_body = Vec::with_capacity(branch.then_nodes.len());
        for node in &branch.then_nodes {
            then_body.push(self.lower_task_node(node)?);
        }

        let mut else_body = Vec::with_capacity(branch.else_nodes.len());
        for node in &branch.else_nodes {
            else_body.push(self.lower_task_node(node)?);
        }

        Ok(BranchIR {
            condition: self.lower_expr(&branch.condition.node)?,
            then_body,
            else_body,
        })
    }

    fn lower_stage_kind(&self, kind: StageKind) -> StageKindIR {
        match kind {
            StageKind::Tool => StageKindIR::Tool,
            StageKind::Prompt => StageKindIR::Prompt,
            StageKind::Agent => StageKindIR::Agent,
        }
    }

    fn lower_harness(&self, decl: &HarnessDecl) -> LowerResult<HarnessIR> {
        let defaults = self.lower_bindings(&decl.defaults)?;

        let mut bindings = Vec::with_capacity(decl.binds.len());
        for bind in &decl.binds {
            bindings.push(TargetBindingIR {
                target: bind.target.node.clone(),
                bindings: self.lower_bindings(&bind.bindings)?,
            });
        }

        let mut tunables = Vec::with_capacity(decl.tune.len());
        for tune in &decl.tune {
            tunables.push(TunableIR {
                path: self.lower_binding_path(&tune.path),
                operator: self.lower_tune_operator(tune.operator),
                domain: self.lower_finite_domain(&tune.domain)?,
            });
        }

        Ok(HarnessIR {
            name: decl.name.node.clone(),
            task: decl.task.node.clone(),
            defaults,
            bindings,
            tunables,
        })
    }

    fn lower_bindings(&self, bindings: &[BindingStmt]) -> LowerResult<Vec<BindingIR>> {
        let mut out = Vec::with_capacity(bindings.len());
        for binding in bindings {
            out.push(BindingIR {
                key: self.lower_binding_path(&binding.key),
                value: self.lower_expr(&binding.value.node)?,
            });
        }
        Ok(out)
    }

    fn lower_binding_path(&self, path: &BindingPath) -> BindingPathIR {
        BindingPathIR {
            segments: path
                .segments
                .iter()
                .map(|segment| segment.node.clone())
                .collect(),
        }
    }

    fn lower_tune_operator(&self, operator: TuneOperator) -> TuneOperatorIR {
        match operator {
            TuneOperator::In => TuneOperatorIR::In,
            TuneOperator::SubsetOf => TuneOperatorIR::SubsetOf,
        }
    }

    fn lower_finite_domain(&self, domain: &FiniteDomain) -> LowerResult<FiniteDomainIR> {
        match domain {
            FiniteDomain::List(values) => {
                let mut ir_values = Vec::with_capacity(values.len());
                for value in values {
                    ir_values.push(self.lower_expr(&value.node)?);
                }
                Ok(FiniteDomainIR::List { values: ir_values })
            }
            FiniteDomain::Variants(name) => Ok(FiniteDomainIR::Variants { name: name.clone() }),
        }
    }

    fn lower_objective(&self, decl: &ObjectiveDecl) -> LowerResult<ObjectiveIR> {
        let mut metrics = Vec::with_capacity(decl.metrics.len());
        for metric in &decl.metrics {
            metrics.push(MetricIR {
                name: metric.name.node.clone(),
                expr: self.lower_expr(&metric.expr.node)?,
            });
        }

        Ok(ObjectiveIR {
            name: decl.name.node.clone(),
            task: decl.task.node.clone(),
            harness: decl.harness.node.clone(),
            dataset: self.lower_dataset_spec(&decl.dataset)?,
            repeats: decl.repeats,
            metrics,
            score: self.lower_expr(&decl.score.node)?,
            split: decl.split.as_ref().map(|split| SplitIR {
                train: split.train,
                val: split.val,
                test: split.test,
            }),
            select: decl
                .select
                .as_ref()
                .map(|select| {
                    let mut tie_breakers = Vec::with_capacity(select.tie_breakers.len());
                    for expr in &select.tie_breakers {
                        tie_breakers.push(self.lower_expr(&expr.node)?);
                    }
                    Ok(SelectIR {
                        primary: self.lower_expr(&select.primary.node)?,
                        tie_breakers,
                    })
                })
                .transpose()?,
        })
    }

    fn lower_dataset_spec(&self, dataset: &DatasetSpec) -> LowerResult<DatasetSpecIR> {
        match dataset {
            DatasetSpec::File(path) => Ok(DatasetSpecIR::File { path: path.clone() }),
            DatasetSpec::Inline(cases) => {
                let mut ir_cases = Vec::with_capacity(cases.len());
                for case in cases {
                    ir_cases.push(InlineDatasetCaseIR {
                        input: self.lower_expr(&case.input.node)?,
                        expected: case
                            .expected
                            .as_ref()
                            .map(|expr| self.lower_expr(&expr.node))
                            .transpose()?,
                        id: case.id.clone(),
                    });
                }
                Ok(DatasetSpecIR::Inline { cases: ir_cases })
            }
        }
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
        assert!(ir.tasks.is_empty());
        assert!(ir.harnesses.is_empty());
        assert!(ir.objectives.is_empty());

        // Serialize and deserialize
        let json = to_json(&ir).unwrap();
        let deserialized = from_json(&json).unwrap();

        assert_eq!(deserialized.types.len(), ir.types.len());
        assert_eq!(deserialized.tools[0].name, "get_position");
        assert_eq!(deserialized.prompts[0].name, "summarize");
        assert_eq!(deserialized.agents[0].name, "researcher");
        assert_eq!(deserialized.pipelines[0].name, "main_pipeline");
    }

    #[test]
    fn test_lower_task_harness_and_objective() {
        let source = r#"
            artifact ResearchNotes = { summary: string, score: float }

            prompt writer {
                input: { question: string }
                output: ResearchNotes
                template: "Write: {question}"
            }

            agent reviewer {
                input: ResearchNotes
                output: ResearchNotes
                tools: []
                system: "Review"
            }

            task answer_question {
                input: { question: string }
                output: { answer: string }
                artifacts {
                    notes: ResearchNotes
                }
                stage draft using prompt writer {
                    in: { question: input.question }
                    out: notes
                }
                loop refine {
                    max_iters: 2
                    carry: [notes]
                    until: notes.score > 0.9
                    stage revise using agent reviewer {
                        in: notes
                        out: notes
                    }
                }
                emit {
                    answer: notes.summary
                }
            }

            harness baseline for task answer_question {
                defaults {
                    model: "gpt-5"
                }
                bind draft {
                    temperature: 0.2
                }
                tune {
                    draft.model in ["gpt-5-mini", "gpt-5"]
                    refine.max_iters in [1, 2, 3]
                }
            }

            objective quality for task answer_question {
                dataset: [
                    { input: { question: "hello" }, expected: { answer: "hi" }, id: "case-1" }
                ]
                harness: baseline
                metric accuracy = output.answer == expected.answer
                score = accuracy
            }
        "#;

        let program = parse(source).unwrap();
        let type_env = check(&program).unwrap();
        let ir = Lowerer::new().lower(&program, &type_env).unwrap();

        assert_eq!(ir.types.len(), 1);
        assert_eq!(ir.types[0].kind, TypeDefKindIR::Artifact);
        assert_eq!(ir.tasks.len(), 1);
        assert_eq!(ir.harnesses.len(), 1);
        assert_eq!(ir.objectives.len(), 1);

        assert_eq!(ir.tasks[0].name, "answer_question");
        assert_eq!(ir.tasks[0].artifacts.len(), 1);
        assert_eq!(ir.tasks[0].body.len(), 2);
        match &ir.tasks[0].body[0] {
            TaskNodeIR::Stage(stage) => {
                assert_eq!(stage.name, "draft");
                assert_eq!(stage.stage_kind, StageKindIR::Prompt);
                assert_eq!(stage.output, "notes");
            }
            _ => panic!("expected first task node to be a stage"),
        }

        assert_eq!(ir.harnesses[0].defaults.len(), 1);
        assert_eq!(ir.harnesses[0].bindings.len(), 1);
        assert_eq!(ir.harnesses[0].tunables.len(), 2);
        assert_eq!(ir.objectives[0].metrics.len(), 1);

        let json = to_json(&ir).unwrap();
        let round_trip = from_json(&json).unwrap();
        assert_eq!(round_trip.tasks.len(), 1);
        assert_eq!(round_trip.harnesses.len(), 1);
        assert_eq!(round_trip.objectives.len(), 1);
    }
}
