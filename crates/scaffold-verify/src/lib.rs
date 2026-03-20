//! Verification layer for the Scaffold DSL
//!
//! This module provides static analysis capabilities.
//! Note: The task-based verification (deadlock detection, reachability, bounds)
//! has been removed as part of the task system simplification.
//! Verification is now a no-op but the module is kept for API compatibility.

pub mod bounds;
pub mod deadlock;
pub mod reachability;

use std::collections::{HashMap, HashSet};

use scaffold_syntax::ast::*;
use scaffold_syntax::Span;
use scaffold_types::{StructType, Type, TypeEnv};

pub use bounds::{BoundsAnalyzer, BoundsResult, TerminationChecker};
pub use deadlock::{DeadlockAnalyzer, DeadlockResult};
pub use reachability::{ReachabilityAnalyzer, ReachabilityResult};

/// Verification error
#[derive(Debug, Clone)]
pub struct VerifyError {
    pub message: String,
    pub span: Span,
    pub severity: Severity,
}

impl VerifyError {
    pub fn new(message: impl Into<String>, span: Span) -> Self {
        Self {
            message: message.into(),
            span,
            severity: Severity::Error,
        }
    }

    pub fn warning(message: impl Into<String>, span: Span) -> Self {
        Self {
            message: message.into(),
            span,
            severity: Severity::Warning,
        }
    }
}

/// Severity level for verification messages
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
}

/// Result of verification including all checks
#[derive(Debug, Clone)]
pub struct VerifyResult {
    /// All verification errors found
    pub errors: Vec<VerifyError>,
    /// Reachability results (kept for API compatibility)
    pub reachability: Vec<(String, ReachabilityResult)>,
    /// Deadlock results (kept for API compatibility)
    pub deadlock: Vec<(String, DeadlockResult)>,
    /// Bounds results (kept for API compatibility)
    pub bounds: Vec<(String, BoundsResult)>,
}

impl VerifyResult {
    pub fn new() -> Self {
        Self {
            errors: Vec::new(),
            reachability: Vec::new(),
            deadlock: Vec::new(),
            bounds: Vec::new(),
        }
    }

    pub fn has_errors(&self) -> bool {
        self.errors.iter().any(|e| e.severity == Severity::Error)
    }

    pub fn has_warnings(&self) -> bool {
        self.errors.iter().any(|e| e.severity == Severity::Warning)
    }
}

impl Default for VerifyResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Main verifier that runs all checks
pub struct Verifier;

impl Verifier {
    pub fn new() -> Self {
        Self
    }

    /// Verify a program with the given type environment
    pub fn verify(&mut self, program: &Program, type_env: &TypeEnv) -> VerifyResult {
        let mut verifier = ProgramVerifier::new(program, type_env);
        verifier.run();
        verifier.result
    }
}

impl Default for Verifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Verify a program and return the result
pub fn verify(program: &Program, type_env: &TypeEnv) -> VerifyResult {
    let mut verifier = Verifier::new();
    verifier.verify(program, type_env)
}

#[derive(Debug, Clone)]
struct ComponentSig {
    input: Type,
    output: Type,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TaskTargetKind {
    Stage(StageKind),
    Loop,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HarnessFieldValueKind {
    Bool,
    Int,
    Float,
    String,
    TextSurface,
    ToolList,
}

#[derive(Debug, Clone, Copy)]
struct HarnessFieldSpec {
    value_kind: HarnessFieldValueKind,
}

#[derive(Debug, Clone)]
struct TaskSummary {
    output_type: Type,
    scope: HashMap<String, Type>,
    artifacts: HashMap<String, Type>,
    targets: HashMap<String, TaskTargetKind>,
}

struct ProgramVerifier<'a> {
    program: &'a Program,
    type_env: &'a TypeEnv,
    result: VerifyResult,
    tool_sigs: HashMap<String, ComponentSig>,
    prompt_sigs: HashMap<String, ComponentSig>,
    agent_sigs: HashMap<String, ComponentSig>,
    tasks: HashMap<String, &'a TaskDecl>,
    harnesses: HashMap<String, &'a HarnessDecl>,
    objectives: HashMap<String, &'a ObjectiveDecl>,
}

impl<'a> ProgramVerifier<'a> {
    fn new(program: &'a Program, type_env: &'a TypeEnv) -> Self {
        Self {
            program,
            type_env,
            result: VerifyResult::new(),
            tool_sigs: HashMap::new(),
            prompt_sigs: HashMap::new(),
            agent_sigs: HashMap::new(),
            tasks: HashMap::new(),
            harnesses: HashMap::new(),
            objectives: HashMap::new(),
        }
    }

    fn run(&mut self) {
        self.collect_symbols();

        for task in self.tasks.values().copied().collect::<Vec<_>>() {
            self.verify_task(task);
        }
        for harness in self.harnesses.values().copied().collect::<Vec<_>>() {
            self.verify_harness(harness);
        }
        for objective in self.objectives.values().copied().collect::<Vec<_>>() {
            self.verify_objective(objective);
        }
    }

    fn collect_symbols(&mut self) {
        for decl in &self.program.declarations {
            match decl {
                Declaration::Tool(tool) => {
                    let sig = ComponentSig {
                        input: self.resolve_type_expr(&tool.input),
                        output: self.resolve_type_expr(&tool.output),
                    };
                    if self.tool_sigs.contains_key(&tool.name.node) {
                        self.error(
                            format!("duplicate tool '{}'", tool.name.node),
                            tool.name.span,
                        );
                    } else {
                        self.tool_sigs.insert(tool.name.node.clone(), sig);
                    }
                }
                Declaration::Prompt(prompt) => {
                    let sig = ComponentSig {
                        input: self.resolve_type_expr(&prompt.input),
                        output: self.resolve_type_expr(&prompt.output),
                    };
                    if self.prompt_sigs.contains_key(&prompt.name.node) {
                        self.error(
                            format!("duplicate prompt '{}'", prompt.name.node),
                            prompt.name.span,
                        );
                    } else {
                        self.prompt_sigs.insert(prompt.name.node.clone(), sig);
                    }
                }
                Declaration::Agent(agent) => {
                    let sig = ComponentSig {
                        input: self.resolve_type_expr(&agent.input),
                        output: self.resolve_type_expr(&agent.output),
                    };
                    if self.agent_sigs.contains_key(&agent.name.node) {
                        self.error(
                            format!("duplicate agent '{}'", agent.name.node),
                            agent.name.span,
                        );
                    } else {
                        self.agent_sigs.insert(agent.name.node.clone(), sig);
                    }
                }
                Declaration::Task(task) => {
                    if self.tasks.contains_key(&task.name.node) {
                        self.error(
                            format!("duplicate task '{}'", task.name.node),
                            task.name.span,
                        );
                    } else {
                        self.tasks.insert(task.name.node.clone(), task);
                    }
                }
                Declaration::Harness(harness) => {
                    if self.harnesses.contains_key(&harness.name.node) {
                        self.error(
                            format!("duplicate harness '{}'", harness.name.node),
                            harness.name.span,
                        );
                    } else {
                        self.harnesses.insert(harness.name.node.clone(), harness);
                    }
                }
                Declaration::Objective(objective) => {
                    if self.objectives.contains_key(&objective.name.node) {
                        self.error(
                            format!("duplicate objective '{}'", objective.name.node),
                            objective.name.span,
                        );
                    } else {
                        self.objectives
                            .insert(objective.name.node.clone(), objective);
                    }
                }
                _ => {}
            }
        }
    }

    fn verify_task(&mut self, task: &TaskDecl) {
        let summary = self.build_task_summary(task);

        let assigned_after = self.verify_task_nodes(&task.nodes, &summary, &HashSet::new(), None);

        let output_struct = match self.type_env.resolve_type(&summary.output_type) {
            Type::Struct(s) => Some(s),
            Type::Any | Type::Error => None,
            other => {
                self.error(
                    format!(
                        "task '{}' output must be a struct type because emit maps named fields, found {}",
                        task.name.node, other
                    ),
                    task.output.span,
                );
                None
            }
        };

        let mut seen_emit_fields = HashSet::new();
        for emit in &task.emit {
            self.verify_artifact_reads(
                &emit.value,
                &assigned_after,
                &summary.artifacts,
                "emit expression",
            );
            let value_ty = self.infer_expr_type(&emit.value, &summary.scope);

            if !seen_emit_fields.insert(emit.name.node.clone()) {
                self.error(
                    format!("duplicate emit field '{}'", emit.name.node),
                    emit.name.span,
                );
            }

            if let Some(output_struct) = &output_struct {
                if let Some(expected_ty) = output_struct.get_field(&emit.name.node) {
                    if !expected_ty.is_compatible_with(&value_ty) && !value_ty.is_error() {
                        self.error(
                            format!(
                                "emit field '{}' expects {}, found {}",
                                emit.name.node, expected_ty, value_ty
                            ),
                            emit.value.span,
                        );
                    }
                } else {
                    self.error(
                        format!(
                            "emit field '{}' is not declared in task output type",
                            emit.name.node
                        ),
                        emit.name.span,
                    );
                }
            }
        }

        if let Some(output_struct) = output_struct {
            for field_name in output_struct.fields.keys() {
                if !seen_emit_fields.contains(field_name) {
                    self.error(
                        format!(
                            "task '{}' emit is missing output field '{}'",
                            task.name.node, field_name
                        ),
                        task.span,
                    );
                }
            }
        }
    }

    fn build_task_summary(&mut self, task: &TaskDecl) -> TaskSummary {
        let input_type = self.resolve_type_expr(&task.input);
        let output_type = self.resolve_type_expr(&task.output);

        let mut scope = HashMap::new();
        scope.insert("input".to_string(), input_type.clone());
        if let Type::Struct(input_struct) = self.type_env.resolve_type(&input_type) {
            for (field, ty) in input_struct.fields {
                scope.insert(field, ty);
            }
        }

        let mut artifacts = HashMap::new();
        for artifact in &task.artifacts {
            let artifact_ty = self.resolve_type_expr(&artifact.ty);
            if artifacts
                .insert(artifact.name.node.clone(), artifact_ty.clone())
                .is_some()
            {
                self.error(
                    format!(
                        "duplicate artifact slot '{}' in task '{}'",
                        artifact.name.node, task.name.node
                    ),
                    artifact.name.span,
                );
            }
            scope.insert(artifact.name.node.clone(), artifact_ty);
        }

        let mut targets = HashMap::new();
        self.collect_task_targets(&task.nodes, &mut targets, &task.name.node);

        TaskSummary {
            output_type,
            scope,
            artifacts,
            targets,
        }
    }

    fn collect_task_targets(
        &mut self,
        nodes: &[TaskNode],
        targets: &mut HashMap<String, TaskTargetKind>,
        task_name: &str,
    ) {
        for node in nodes {
            match node {
                TaskNode::Stage(stage) => {
                    if targets
                        .insert(stage.name.node.clone(), TaskTargetKind::Stage(stage.kind))
                        .is_some()
                    {
                        self.error(
                            format!(
                                "duplicate task target '{}' in task '{}'",
                                stage.name.node, task_name
                            ),
                            stage.name.span,
                        );
                    }
                }
                TaskNode::Loop(loop_decl) => {
                    if targets
                        .insert(loop_decl.name.node.clone(), TaskTargetKind::Loop)
                        .is_some()
                    {
                        self.error(
                            format!(
                                "duplicate task target '{}' in task '{}'",
                                loop_decl.name.node, task_name
                            ),
                            loop_decl.name.span,
                        );
                    }
                    self.collect_task_targets(&loop_decl.nodes, targets, task_name);
                }
                TaskNode::Branch(branch) => {
                    self.collect_task_targets(&branch.then_nodes, targets, task_name);
                    self.collect_task_targets(&branch.else_nodes, targets, task_name);
                }
            }
        }
    }

    fn verify_task_nodes(
        &mut self,
        nodes: &[TaskNode],
        summary: &TaskSummary,
        assigned: &HashSet<String>,
        loop_carry: Option<&HashSet<String>>,
    ) -> HashSet<String> {
        let mut current = assigned.clone();
        for node in nodes {
            current = self.verify_task_node(node, summary, &current, loop_carry);
        }
        current
    }

    fn verify_task_node(
        &mut self,
        node: &TaskNode,
        summary: &TaskSummary,
        assigned: &HashSet<String>,
        loop_carry: Option<&HashSet<String>>,
    ) -> HashSet<String> {
        match node {
            TaskNode::Stage(stage) => self.verify_stage(stage, summary, assigned, loop_carry),
            TaskNode::Loop(loop_decl) => self.verify_loop(loop_decl, summary, assigned),
            TaskNode::Branch(branch) => self.verify_branch(branch, summary, assigned, loop_carry),
        }
    }

    fn verify_stage(
        &mut self,
        stage: &StageDecl,
        summary: &TaskSummary,
        assigned: &HashSet<String>,
        loop_carry: Option<&HashSet<String>>,
    ) -> HashSet<String> {
        self.verify_artifact_reads(&stage.input, assigned, &summary.artifacts, "stage input");
        if let Some(when) = &stage.when {
            self.verify_artifact_reads(when, assigned, &summary.artifacts, "stage when condition");
            let when_ty = self.infer_expr_type(when, &summary.scope);
            if !when_ty.is_compatible_with(&Type::Bool) && !when_ty.is_error() {
                self.error(
                    format!("stage 'when' condition must be bool, found {}", when_ty),
                    when.span,
                );
            }
        }

        let component_sig = match stage.kind {
            StageKind::Tool => self.tool_sigs.get(&stage.component.node).cloned(),
            StageKind::Prompt => self.prompt_sigs.get(&stage.component.node).cloned(),
            StageKind::Agent => self.agent_sigs.get(&stage.component.node).cloned(),
        };

        if component_sig.is_none() {
            self.error(
                format!(
                    "stage '{}' references undefined {} '{}'",
                    stage.name.node,
                    self.stage_kind_name(stage.kind),
                    stage.component.node
                ),
                stage.component.span,
            );
        }

        if let Some(slot_ty) = summary.artifacts.get(&stage.output.node) {
            if let Some(carry) = loop_carry {
                if !carry.contains(&stage.output.node) {
                    self.error(
                        format!(
                            "loop stages may only write to carried artifacts; '{}' is not in carry",
                            stage.output.node
                        ),
                        stage.output.span,
                    );
                }
            }

            if let Some(sig) = component_sig {
                let input_ty = self.infer_expr_type(&stage.input, &summary.scope);
                if !sig.input.is_compatible_with(&input_ty) && !input_ty.is_error() {
                    self.error(
                        format!(
                            "stage '{}' input expects {}, found {}",
                            stage.name.node, sig.input, input_ty
                        ),
                        stage.input.span,
                    );
                }
                if !slot_ty.is_compatible_with(&sig.output) && !sig.output.is_error() {
                    self.error(
                        format!(
                            "stage '{}' writes {} to artifact '{}' of type {}",
                            stage.name.node, sig.output, stage.output.node, slot_ty
                        ),
                        stage.output.span,
                    );
                }
            }
        } else {
            self.error(
                format!(
                    "stage '{}' outputs to undeclared artifact slot '{}'",
                    stage.name.node, stage.output.node
                ),
                stage.output.span,
            );
        }

        let mut next = assigned.clone();
        if stage.when.is_none() {
            next.insert(stage.output.node.clone());
        }
        next
    }

    fn verify_loop(
        &mut self,
        loop_decl: &TaskLoopDecl,
        summary: &TaskSummary,
        assigned: &HashSet<String>,
    ) -> HashSet<String> {
        self.verify_artifact_reads(
            &loop_decl.max_iters,
            assigned,
            &summary.artifacts,
            "loop max_iters",
        );
        let max_iters_ty = self.infer_expr_type(&loop_decl.max_iters, &summary.scope);
        if !max_iters_ty.is_compatible_with(&Type::Int)
            && !max_iters_ty.is_compatible_with(&Type::Float)
            && !max_iters_ty.is_error()
        {
            self.error(
                format!("loop 'max_iters' must be numeric, found {}", max_iters_ty),
                loop_decl.max_iters.span,
            );
        }

        let mut carry = HashSet::new();
        for ident in &loop_decl.carry {
            if !summary.artifacts.contains_key(&ident.node) {
                self.error(
                    format!(
                        "loop '{}' carries undeclared artifact '{}'",
                        loop_decl.name.node, ident.node
                    ),
                    ident.span,
                );
            }
            if !carry.insert(ident.node.clone()) {
                self.error(
                    format!(
                        "loop '{}' carries duplicate artifact '{}'",
                        loop_decl.name.node, ident.node
                    ),
                    ident.span,
                );
            }
        }

        let body_assigned =
            self.verify_task_nodes(&loop_decl.nodes, summary, assigned, Some(&carry));
        for artifact in &carry {
            if !body_assigned.contains(artifact) {
                self.error(
                    format!(
                        "loop '{}' carry artifact '{}' is not definitely assigned by the loop body",
                        loop_decl.name.node, artifact
                    ),
                    loop_decl.span,
                );
            }
        }

        if loop_decl.while_condition.is_none() && loop_decl.until.is_none() {
            self.error(
                "loop requires at least one termination condition: 'while' or 'until'".to_string(),
                loop_decl.span,
            );
        }

        let mut condition_assigned = assigned.clone();
        condition_assigned.extend(carry.iter().cloned());
        if let Some(while_condition) = &loop_decl.while_condition {
            self.verify_artifact_reads(
                while_condition,
                &condition_assigned,
                &summary.artifacts,
                "loop while condition",
            );
            let while_ty = self.infer_expr_type(while_condition, &summary.scope);
            if !while_ty.is_compatible_with(&Type::Bool) && !while_ty.is_error() {
                self.error(
                    format!("loop 'while' condition must be bool, found {}", while_ty),
                    while_condition.span,
                );
            }
        }

        if let Some(until) = &loop_decl.until {
            self.verify_artifact_reads(
                until,
                &condition_assigned,
                &summary.artifacts,
                "loop until condition",
            );
            let until_ty = self.infer_expr_type(until, &summary.scope);
            if !until_ty.is_compatible_with(&Type::Bool) && !until_ty.is_error() {
                self.error(
                    format!("loop 'until' condition must be bool, found {}", until_ty),
                    until.span,
                );
            }
        }

        assigned.clone()
    }

    fn verify_branch(
        &mut self,
        branch: &TaskBranchDecl,
        summary: &TaskSummary,
        assigned: &HashSet<String>,
        loop_carry: Option<&HashSet<String>>,
    ) -> HashSet<String> {
        self.verify_artifact_reads(
            &branch.condition,
            assigned,
            &summary.artifacts,
            "branch condition",
        );
        let condition_ty = self.infer_expr_type(&branch.condition, &summary.scope);
        if !condition_ty.is_compatible_with(&Type::Bool) && !condition_ty.is_error() {
            self.error(
                format!("branch condition must be bool, found {}", condition_ty),
                branch.condition.span,
            );
        }

        let then_assigned =
            self.verify_task_nodes(&branch.then_nodes, summary, assigned, loop_carry);
        let else_assigned = if branch.else_nodes.is_empty() {
            assigned.clone()
        } else {
            self.verify_task_nodes(&branch.else_nodes, summary, assigned, loop_carry)
        };

        then_assigned
            .intersection(&else_assigned)
            .cloned()
            .collect::<HashSet<_>>()
    }

    fn verify_harness(&mut self, harness: &HarnessDecl) {
        let Some(task) = self.tasks.get(&harness.task.node).copied() else {
            self.error(
                format!(
                    "harness '{}' references undefined task '{}'",
                    harness.name.node, harness.task.node
                ),
                harness.task.span,
            );
            return;
        };

        let summary = self.build_task_summary(task);

        let mut seen_default_fields = HashSet::new();
        for binding in &harness.defaults {
            self.verify_default_binding(harness, binding, &mut seen_default_fields);
        }

        let mut seen_bind_targets = HashSet::new();
        for bind in &harness.binds {
            let target_kind = match summary.targets.get(&bind.target.node).copied() {
                Some(kind) => Some(kind),
                None => {
                    self.error(
                        format!(
                            "harness '{}' binds unknown task target '{}'",
                            harness.name.node, bind.target.node
                        ),
                        bind.target.span,
                    );
                    None
                }
            };

            if !seen_bind_targets.insert(bind.target.node.clone()) {
                self.error(
                    format!(
                        "harness '{}' binds '{}' more than once",
                        harness.name.node, bind.target.node
                    ),
                    bind.target.span,
                );
            }

            if let Some(target_kind) = target_kind {
                let mut seen_fields = HashSet::new();
                for binding in &bind.bindings {
                    self.verify_target_binding(
                        task,
                        &summary,
                        harness,
                        &bind.target,
                        target_kind,
                        binding,
                        &mut seen_fields,
                    );
                }
            }
        }

        let mut seen_tune_paths = HashSet::new();
        for tune in &harness.tune {
            let path_string = self.binding_path_string(&tune.path);
            if !seen_tune_paths.insert(path_string.clone()) {
                self.error(
                    format!(
                        "harness '{}' tunes '{}' more than once",
                        harness.name.node, path_string
                    ),
                    tune.path.span,
                );
            }

            if tune.path.segments.len() != 2 {
                self.error(
                    format!(
                        "tune path '{}' must be of the form target.field in V1",
                        path_string
                    ),
                    tune.path.span,
                );
                continue;
            }

            let target = &tune.path.segments[0];
            let field = &tune.path.segments[1];

            let Some(target_kind) = summary.targets.get(&target.node).copied() else {
                self.error(
                    format!(
                        "tune path '{}' references unknown task target '{}'",
                        path_string, target.node
                    ),
                    target.span,
                );
                continue;
            };

            let Some(spec) = self.target_field_spec(target_kind, &field.node) else {
                self.error(
                    format!(
                        "tune path '{}' is not a mutable field for {} '{}'",
                        path_string,
                        self.target_kind_name(target_kind),
                        target.node
                    ),
                    field.span,
                );
                continue;
            };

            self.verify_tune_stmt(task, &summary, harness, tune, target_kind, spec);
        }
    }

    fn verify_default_binding(
        &mut self,
        harness: &HarnessDecl,
        binding: &BindingStmt,
        seen_fields: &mut HashSet<String>,
    ) {
        let path = self.binding_path_string(&binding.key);
        if binding.key.segments.len() != 1 {
            self.error(
                format!(
                    "defaults binding '{}' must be a single field name in V1",
                    path
                ),
                binding.key.span,
            );
            return;
        }

        let field = &binding.key.segments[0];
        let Some(spec) = self.default_field_spec(&field.node) else {
            self.error(
                format!(
                    "defaults field '{}' is not a supported harness default",
                    field.node
                ),
                field.span,
            );
            return;
        };

        if !seen_fields.insert(field.node.clone()) {
            self.error(
                format!(
                    "harness '{}' sets default '{}' more than once",
                    harness.name.node, field.node
                ),
                field.span,
            );
        }

        self.verify_binding_value(
            &format!("defaults.{}", field.node),
            spec,
            &binding.value,
            binding.span,
        );
    }

    fn verify_target_binding(
        &mut self,
        task: &TaskDecl,
        summary: &TaskSummary,
        harness: &HarnessDecl,
        target: &Ident,
        target_kind: TaskTargetKind,
        binding: &BindingStmt,
        seen_fields: &mut HashSet<String>,
    ) {
        let path = self.binding_path_string(&binding.key);
        if binding.key.segments.len() != 1 {
            self.error(
                format!(
                    "binding '{}' for target '{}' must be a single field name in V1",
                    path, target.node
                ),
                binding.key.span,
            );
            return;
        }

        let field = &binding.key.segments[0];
        let Some(spec) = self.target_field_spec(target_kind, &field.node) else {
            self.error(
                format!(
                    "field '{}' is not mutable for {} '{}'",
                    field.node,
                    self.target_kind_name(target_kind),
                    target.node
                ),
                field.span,
            );
            return;
        };

        if !seen_fields.insert(field.node.clone()) {
            self.error(
                format!(
                    "harness '{}' sets '{}' twice for target '{}'",
                    harness.name.node, field.node, target.node
                ),
                field.span,
            );
        }

        let path = format!("{}.{}", target.node, field.node);
        if field.node == "component" {
            self.verify_component_binding_value(
                task,
                summary,
                target,
                target_kind,
                &path,
                &binding.value,
                binding.span,
            );
        } else {
            self.verify_binding_value(&path, spec, &binding.value, binding.span);
        }
    }

    fn verify_tune_stmt(
        &mut self,
        task: &TaskDecl,
        summary: &TaskSummary,
        _harness: &HarnessDecl,
        tune: &TuneStmt,
        target_kind: TaskTargetKind,
        spec: HarnessFieldSpec,
    ) {
        let path = self.binding_path_string(&tune.path);
        match (&tune.operator, &tune.domain) {
            (TuneOperator::SubsetOf, FiniteDomain::Variants(_)) => self.error(
                format!(
                    "tune path '{}' uses subset_of with variants(...), but subset_of requires a finite list domain",
                    path
                ),
                tune.span,
            ),
            (_, FiniteDomain::List(values)) if values.is_empty() => self.error(
                format!("tune path '{}' has an empty search domain", path),
                tune.span,
            ),
            (_, FiniteDomain::Variants(name)) if name.is_empty() => self.error(
                "variants domain name must not be empty".to_string(),
                tune.span,
            ),
            (TuneOperator::SubsetOf, _) if spec.value_kind != HarnessFieldValueKind::ToolList => {
                self.error(
                    format!(
                        "tune path '{}' uses subset_of, but {} '{}' is not list-like",
                        path,
                        self.target_kind_name(target_kind),
                        tune.path.segments[0].node
                    ),
                    tune.span,
                );
            }
            (TuneOperator::In, FiniteDomain::Variants(_))
                if !matches!(spec.value_kind, HarnessFieldValueKind::TextSurface) =>
            {
                self.error(
                    format!(
                        "tune path '{}' uses variants(...), but '{}' does not support prompt variants",
                        path, tune.path.segments[1].node
                    ),
                    tune.span,
                );
            }
            _ => {}
        }

        match (&tune.operator, &tune.domain, spec.value_kind) {
            (
                TuneOperator::SubsetOf,
                FiniteDomain::List(values),
                HarnessFieldValueKind::ToolList,
            ) => {
                self.verify_tool_ref_domain(&path, values, true);
            }
            (TuneOperator::In, FiniteDomain::List(values), HarnessFieldValueKind::ToolList) => {
                for value in values {
                    self.verify_tool_list_expr(&path, value);
                }
            }
            (_, FiniteDomain::List(values), _) => {
                if tune.path.segments[1].node == "component" {
                    let target = &tune.path.segments[0];
                    for value in values {
                        self.verify_component_binding_value(
                            task,
                            summary,
                            target,
                            target_kind,
                            &path,
                            value,
                            value.span,
                        );
                    }
                } else {
                    for value in values {
                        self.verify_binding_value(&path, spec, value, value.span);
                    }
                }
            }
            (TuneOperator::In, FiniteDomain::Variants(_), HarnessFieldValueKind::TextSurface) => {}
            _ => {}
        }
    }

    fn default_field_spec(&self, field: &str) -> Option<HarnessFieldSpec> {
        match field {
            "model" => Some(HarnessFieldSpec {
                value_kind: HarnessFieldValueKind::String,
            }),
            "temperature" => Some(HarnessFieldSpec {
                value_kind: HarnessFieldValueKind::Float,
            }),
            "timeout_secs" | "retries" | "max_turns" => Some(HarnessFieldSpec {
                value_kind: HarnessFieldValueKind::Int,
            }),
            "system_prompt" | "variant" => Some(HarnessFieldSpec {
                value_kind: HarnessFieldValueKind::TextSurface,
            }),
            "tools" => Some(HarnessFieldSpec {
                value_kind: HarnessFieldValueKind::ToolList,
            }),
            _ => None,
        }
    }

    fn target_field_spec(
        &self,
        target_kind: TaskTargetKind,
        field: &str,
    ) -> Option<HarnessFieldSpec> {
        match target_kind {
            TaskTargetKind::Stage(StageKind::Tool) => match field {
                "enabled" => Some(HarnessFieldSpec {
                    value_kind: HarnessFieldValueKind::Bool,
                }),
                "component" => Some(HarnessFieldSpec {
                    value_kind: HarnessFieldValueKind::String,
                }),
                "timeout_secs" | "retries" => Some(HarnessFieldSpec {
                    value_kind: HarnessFieldValueKind::Int,
                }),
                "variant" => Some(HarnessFieldSpec {
                    value_kind: HarnessFieldValueKind::TextSurface,
                }),
                _ => None,
            },
            TaskTargetKind::Stage(StageKind::Prompt) => match field {
                "enabled" => Some(HarnessFieldSpec {
                    value_kind: HarnessFieldValueKind::Bool,
                }),
                "component" => Some(HarnessFieldSpec {
                    value_kind: HarnessFieldValueKind::String,
                }),
                "model" => Some(HarnessFieldSpec {
                    value_kind: HarnessFieldValueKind::String,
                }),
                "temperature" => Some(HarnessFieldSpec {
                    value_kind: HarnessFieldValueKind::Float,
                }),
                "timeout_secs" | "retries" => Some(HarnessFieldSpec {
                    value_kind: HarnessFieldValueKind::Int,
                }),
                "system_prompt" | "variant" => Some(HarnessFieldSpec {
                    value_kind: HarnessFieldValueKind::TextSurface,
                }),
                _ => None,
            },
            TaskTargetKind::Stage(StageKind::Agent) => match field {
                "enabled" => Some(HarnessFieldSpec {
                    value_kind: HarnessFieldValueKind::Bool,
                }),
                "component" => Some(HarnessFieldSpec {
                    value_kind: HarnessFieldValueKind::String,
                }),
                "model" => Some(HarnessFieldSpec {
                    value_kind: HarnessFieldValueKind::String,
                }),
                "temperature" => Some(HarnessFieldSpec {
                    value_kind: HarnessFieldValueKind::Float,
                }),
                "timeout_secs" | "retries" | "max_turns" => Some(HarnessFieldSpec {
                    value_kind: HarnessFieldValueKind::Int,
                }),
                "system_prompt" | "variant" => Some(HarnessFieldSpec {
                    value_kind: HarnessFieldValueKind::TextSurface,
                }),
                "tools" => Some(HarnessFieldSpec {
                    value_kind: HarnessFieldValueKind::ToolList,
                }),
                _ => None,
            },
            TaskTargetKind::Loop => match field {
                "max_iters" => Some(HarnessFieldSpec {
                    value_kind: HarnessFieldValueKind::Int,
                }),
                _ => None,
            },
        }
    }

    fn verify_binding_value(
        &mut self,
        path: &str,
        spec: HarnessFieldSpec,
        expr: &Spanned<Expr>,
        error_span: Span,
    ) {
        match spec.value_kind {
            HarnessFieldValueKind::Bool => {
                if !matches!(&expr.node, Expr::Literal(Literal::Bool(_))) {
                    self.error(format!("'{}' must be a boolean literal", path), error_span);
                }
            }
            HarnessFieldValueKind::Int => {
                if !self.is_numeric_config_expr(expr) {
                    self.error(
                        format!("'{}' must be a static integer expression", path),
                        error_span,
                    );
                    return;
                }
                let ty = self.infer_expr_type(expr, &HashMap::new());
                if !matches!(self.type_env.resolve_type(&ty), Type::Int) && !ty.is_error() {
                    self.error(
                        format!("'{}' must be an integer value, found {}", path, ty),
                        error_span,
                    );
                }
            }
            HarnessFieldValueKind::Float => {
                if !self.is_numeric_config_expr(expr) {
                    self.error(
                        format!("'{}' must be a static numeric expression", path),
                        error_span,
                    );
                    return;
                }
                let ty = self.infer_expr_type(expr, &HashMap::new());
                if !ty.is_compatible_with(&Type::Float) && !ty.is_error() {
                    self.error(
                        format!("'{}' must be numeric, found {}", path, ty),
                        error_span,
                    );
                }
            }
            HarnessFieldValueKind::String => {
                if !matches!(&expr.node, Expr::Literal(Literal::String(_))) {
                    self.error(format!("'{}' must be a string literal", path), error_span);
                }
            }
            HarnessFieldValueKind::TextSurface => {
                if !(matches!(&expr.node, Expr::Literal(Literal::String(_)))
                    || self.is_variant_call(expr))
                {
                    self.error(
                        format!(
                            "'{}' must be a string literal or variant(\"group\", \"name\")",
                            path
                        ),
                        error_span,
                    );
                }
            }
            HarnessFieldValueKind::ToolList => self.verify_tool_list_expr(path, expr),
        }
    }

    fn verify_tool_list_expr(&mut self, path: &str, expr: &Spanned<Expr>) {
        let Expr::ListLiteral(items) = &expr.node else {
            self.error(
                format!("'{}' must be a list of tool references", path),
                expr.span,
            );
            return;
        };

        for item in items {
            self.verify_tool_ref_expr(path, item);
        }
    }

    fn verify_tool_ref_domain(&mut self, path: &str, values: &[Spanned<Expr>], allow_empty: bool) {
        if values.is_empty() && !allow_empty {
            self.error(
                format!("'{}' must list at least one tool reference", path),
                Span::new(0, 0),
            );
            return;
        }
        for value in values {
            self.verify_tool_ref_expr(path, value);
        }
    }

    fn verify_tool_ref_expr(&mut self, path: &str, expr: &Spanned<Expr>) {
        let tool_name = match &expr.node {
            Expr::Ident(name) => name.clone(),
            Expr::Literal(Literal::String(name)) => name.clone(),
            _ => {
                self.error(
                    format!("'{}' expects tool references like [web_search]", path),
                    expr.span,
                );
                return;
            }
        };

        if !self.tool_sigs.contains_key(&tool_name) {
            self.error(
                format!("'{}' references unknown tool '{}'", path, tool_name),
                expr.span,
            );
        }
    }

    fn verify_component_binding_value(
        &mut self,
        task: &TaskDecl,
        summary: &TaskSummary,
        target: &Ident,
        target_kind: TaskTargetKind,
        path: &str,
        expr: &Spanned<Expr>,
        error_span: Span,
    ) {
        let TaskTargetKind::Stage(stage_kind) = target_kind else {
            self.error(
                format!("'{}' may only target stage components", path),
                error_span,
            );
            return;
        };

        let Expr::Literal(Literal::String(component_name)) = &expr.node else {
            self.error(
                format!(
                    "'{}' must be a string literal naming a compatible {}",
                    path,
                    self.stage_kind_component_name(stage_kind)
                ),
                error_span,
            );
            return;
        };

        let Some(stage) = self.find_stage_decl(&task.nodes, &target.node) else {
            self.error(
                format!(
                    "task '{}' does not define stage target '{}'",
                    task.name.node, target.node
                ),
                target.span,
            );
            return;
        };

        let candidate_sig = match stage_kind {
            StageKind::Tool => self.tool_sigs.get(component_name),
            StageKind::Prompt => self.prompt_sigs.get(component_name),
            StageKind::Agent => self.agent_sigs.get(component_name),
        };

        let Some(candidate_sig) = candidate_sig else {
            self.error(
                format!(
                    "'{}' references unknown {} '{}'",
                    path,
                    self.stage_kind_component_name(stage_kind),
                    component_name
                ),
                error_span,
            );
            return;
        };

        let expected_input = self.infer_expr_type(&stage.input, &summary.scope);
        let Some(expected_output) = summary.artifacts.get(&stage.output.node) else {
            self.error(
                format!(
                    "stage '{}' writes to unknown artifact '{}'",
                    stage.name.node, stage.output.node
                ),
                stage.output.span,
            );
            return;
        };

        if !candidate_sig.input.is_compatible_with(&expected_input)
            || !candidate_sig.output.is_compatible_with(expected_output)
        {
            self.error(
                format!(
                    "'{}' swaps stage '{}' to {} '{}', but it expects {} -> {} while the stage requires {} -> {}",
                    path,
                    stage.name.node,
                    self.stage_kind_component_name(stage_kind),
                    component_name,
                    candidate_sig.input,
                    candidate_sig.output,
                    expected_input,
                    expected_output
                ),
                error_span,
            );
        }
    }

    fn find_stage_decl<'b>(&self, nodes: &'b [TaskNode], target: &str) -> Option<&'b StageDecl> {
        for node in nodes {
            match node {
                TaskNode::Stage(stage) if stage.name.node == target => return Some(stage),
                TaskNode::Loop(loop_decl) => {
                    if let Some(stage) = self.find_stage_decl(&loop_decl.nodes, target) {
                        return Some(stage);
                    }
                }
                TaskNode::Branch(branch) => {
                    if let Some(stage) = self.find_stage_decl(&branch.then_nodes, target) {
                        return Some(stage);
                    }
                    if let Some(stage) = self.find_stage_decl(&branch.else_nodes, target) {
                        return Some(stage);
                    }
                }
                _ => {}
            }
        }
        None
    }

    fn stage_kind_component_name(&self, kind: StageKind) -> &'static str {
        match kind {
            StageKind::Tool => "tool",
            StageKind::Prompt => "prompt",
            StageKind::Agent => "agent",
        }
    }

    fn is_numeric_config_expr(&self, expr: &Spanned<Expr>) -> bool {
        match &expr.node {
            Expr::Literal(Literal::Int(_) | Literal::Float(_)) => true,
            Expr::Binary(left, op, right) => {
                matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div)
                    && self.is_numeric_config_expr(left)
                    && self.is_numeric_config_expr(right)
            }
            Expr::Paren(inner) => self.is_numeric_config_expr(inner),
            _ => false,
        }
    }

    fn is_variant_call(&self, expr: &Spanned<Expr>) -> bool {
        match &expr.node {
            Expr::Call(name, args) if name == "variant" => {
                (args.len() == 1 || args.len() == 2)
                    && args
                        .iter()
                        .all(|arg| matches!(&arg.node, Expr::Literal(Literal::String(_))))
            }
            Expr::Paren(inner) => self.is_variant_call(inner),
            _ => false,
        }
    }

    fn verify_objective(&mut self, objective: &ObjectiveDecl) {
        let task = self.tasks.get(&objective.task.node).copied();
        if task.is_none() {
            self.error(
                format!(
                    "objective '{}' references undefined task '{}'",
                    objective.name.node, objective.task.node
                ),
                objective.task.span,
            );
        }

        let harness = self.harnesses.get(&objective.harness.node).copied();
        if harness.is_none() {
            self.error(
                format!(
                    "objective '{}' references undefined harness '{}'",
                    objective.name.node, objective.harness.node
                ),
                objective.harness.span,
            );
        }

        if let (Some(task), Some(harness)) = (task, harness) {
            if harness.task.node != task.name.node {
                self.error(
                    format!(
                        "objective '{}' targets task '{}' but harness '{}' is for task '{}'",
                        objective.name.node, task.name.node, harness.name.node, harness.task.node
                    ),
                    objective.harness.span,
                );
            }
        }

        if matches!(objective.repeats, Some(0)) {
            self.error(
                format!(
                    "objective '{}' repeats must be greater than zero",
                    objective.name.node
                ),
                objective.span,
            );
        }

        if let Some(split) = &objective.split {
            for (label, value) in [
                ("train", split.train),
                ("val", split.val),
                ("test", split.test),
            ] {
                if !(0.0..=1.0).contains(&value) {
                    self.error(
                        format!("objective split '{}' must be between 0.0 and 1.0", label),
                        split.span,
                    );
                }
            }
            let total = split.train + split.val + split.test;
            if (total - 1.0).abs() > 1e-6 {
                self.error(
                    format!("objective split must sum to 1.0, found {}", total),
                    split.span,
                );
            }
        }

        if let Some(task) = task {
            let task_input = self.resolve_type_expr(&task.input);
            let task_output = self.resolve_type_expr(&task.output);
            match &objective.dataset {
                DatasetSpec::File(_) => {}
                DatasetSpec::Inline(cases) => {
                    let mut seen_ids = HashSet::new();
                    for case in cases {
                        let input_ty = self.infer_expr_type(&case.input, &HashMap::new());
                        if !task_input.is_compatible_with(&input_ty) && !input_ty.is_error() {
                            self.error(
                                format!(
                                    "objective '{}' dataset input expects {}, found {}",
                                    objective.name.node, task_input, input_ty
                                ),
                                case.input.span,
                            );
                        }
                        if let Some(expected) = &case.expected {
                            let expected_ty = self.infer_expr_type(expected, &HashMap::new());
                            if !task_output.is_compatible_with(&expected_ty)
                                && !expected_ty.is_error()
                            {
                                self.error(
                                    format!(
                                        "objective '{}' expected output expects {}, found {}",
                                        objective.name.node, task_output, expected_ty
                                    ),
                                    expected.span,
                                );
                            }
                        }
                        if let Some(id) = &case.id {
                            if !seen_ids.insert(id.clone()) {
                                self.error(
                                    format!(
                                        "objective '{}' dataset contains duplicate case id '{}'",
                                        objective.name.node, id
                                    ),
                                    case.span,
                                );
                            }
                        }
                    }
                }
            }
        }

        let base_roots = HashSet::from([
            "input".to_string(),
            "expected".to_string(),
            "output".to_string(),
            "rollout".to_string(),
        ]);
        let mut signal_scope = HashMap::from([
            ("input".to_string(), Type::Any),
            ("expected".to_string(), Type::Any),
            ("output".to_string(), Type::Any),
            ("rollout".to_string(), Type::Any),
        ]);
        let mut signal_names = HashSet::new();
        let mut score_roots = base_roots.clone();
        for constraint in &objective.constraints {
            self.verify_objective_signal(
                objective,
                constraint,
                &score_roots,
                &signal_scope,
                &mut signal_names,
                "constraint",
                true,
            );
            score_roots.insert(constraint.name.node.clone());
            signal_scope.insert(constraint.name.node.clone(), Type::Any);
        }
        for checker in &objective.checkers {
            self.verify_objective_signal(
                objective,
                checker,
                &score_roots,
                &signal_scope,
                &mut signal_names,
                "checker",
                false,
            );
            score_roots.insert(checker.name.node.clone());
            signal_scope.insert(checker.name.node.clone(), Type::Any);
        }
        for judge in &objective.judges {
            self.verify_objective_signal(
                objective,
                judge,
                &score_roots,
                &signal_scope,
                &mut signal_names,
                "judge",
                false,
            );
            score_roots.insert(judge.name.node.clone());
            signal_scope.insert(judge.name.node.clone(), Type::Any);
        }
        for metric in &objective.metrics {
            self.verify_objective_signal(
                objective,
                metric,
                &score_roots,
                &signal_scope,
                &mut signal_names,
                "metric",
                false,
            );
            score_roots.insert(metric.name.node.clone());
            signal_scope.insert(metric.name.node.clone(), Type::Any);
        }

        self.verify_allowed_roots(&objective.score, &score_roots, "score expression");
        if let Some(select) = &objective.select {
            self.verify_allowed_roots(&select.primary, &score_roots, "select primary expression");
            for expr in &select.tie_breakers {
                self.verify_allowed_roots(expr, &score_roots, "select tie_breaker expression");
            }
        }
    }

    fn verify_objective_signal(
        &mut self,
        objective: &ObjectiveDecl,
        decl: &MetricDecl,
        allowed_roots: &HashSet<String>,
        scope: &HashMap<String, Type>,
        seen_names: &mut HashSet<String>,
        label: &str,
        require_bool: bool,
    ) {
        if !seen_names.insert(decl.name.node.clone()) {
            self.error(
                format!(
                    "objective '{}' declares {} '{}' more than once",
                    objective.name.node, label, decl.name.node
                ),
                decl.name.span,
            );
        }

        self.verify_allowed_roots(&decl.expr, allowed_roots, &format!("{} expression", label));

        let ty = self.infer_expr_type(&decl.expr, scope);
        let resolved = self.type_env.resolve_type(&ty);
        if require_bool {
            if !matches!(resolved, Type::Bool) && !ty.is_error() {
                self.error(
                    format!(
                        "{} '{}' must evaluate to bool, found {}",
                        label, decl.name.node, ty
                    ),
                    decl.expr.span,
                );
            }
        } else if !matches!(resolved, Type::Bool | Type::Int | Type::Float) && !ty.is_error() {
            self.error(
                format!(
                    "{} '{}' must evaluate to bool or numeric, found {}",
                    label, decl.name.node, ty
                ),
                decl.expr.span,
            );
        }
    }

    fn resolve_type_expr(&self, ty: &Spanned<TypeExpr>) -> Type {
        let resolved = match &ty.node {
            TypeExpr::Primitive(p) => match p {
                PrimitiveType::Bool => Type::Bool,
                PrimitiveType::Int => Type::Int,
                PrimitiveType::Float => Type::Float,
                PrimitiveType::String => Type::String,
                PrimitiveType::Any => Type::Any,
                PrimitiveType::Bytes => Type::Bytes,
            },
            TypeExpr::Named(name) => self
                .type_env
                .lookup_type(name)
                .cloned()
                .unwrap_or(Type::Named(name.clone())),
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
        };
        self.type_env.resolve_type(&resolved)
    }

    fn infer_expr_type(&self, expr: &Spanned<Expr>, scope: &HashMap<String, Type>) -> Type {
        match &expr.node {
            Expr::Literal(lit) => match lit {
                Literal::Int(_) => Type::Int,
                Literal::Float(_) => Type::Float,
                Literal::String(_) => Type::String,
                Literal::Bool(_) => Type::Bool,
                Literal::Null => Type::Unit,
            },
            Expr::Ident(name) => scope
                .get(name)
                .cloned()
                .or_else(|| self.type_env.lookup_variable(name).cloned())
                .unwrap_or(Type::Any),
            Expr::FieldAccess(base, field) => {
                let base_ty = self.infer_expr_type(base, scope);
                match self.type_env.resolve_type(&base_ty) {
                    Type::Struct(s) => s.get_field(&field.node).cloned().unwrap_or(Type::Error),
                    Type::Any | Type::Error => Type::Any,
                    _ => Type::Error,
                }
            }
            Expr::Binary(left, op, right) => {
                let left_ty = self.infer_expr_type(left, scope);
                let right_ty = self.infer_expr_type(right, scope);
                match op {
                    BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge => {
                        Type::Bool
                    }
                    BinOp::And | BinOp::Or => Type::Bool,
                    BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div => {
                        if left_ty.is_error() || right_ty.is_error() {
                            Type::Error
                        } else if matches!(self.type_env.resolve_type(&left_ty), Type::Float)
                            || matches!(self.type_env.resolve_type(&right_ty), Type::Float)
                        {
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
                    .map(|arg| self.infer_expr_type(arg, scope))
                    .collect::<Vec<_>>();
                match name.as_str() {
                    "len" => Type::Int,
                    "is_empty" | "contains" | "is_some" | "is_none" | "not" => Type::Bool,
                    "unwrap" | "unwrap_or" => arg_tys
                        .first()
                        .map(|ty| match self.type_env.resolve_type(ty) {
                            Type::Option(inner) => *inner,
                            Type::Result(ok, _) => *ok,
                            other => other,
                        })
                        .unwrap_or(Type::Any),
                    "abs" | "min" | "max" => {
                        if arg_tys
                            .iter()
                            .any(|ty| matches!(self.type_env.resolve_type(ty), Type::Float))
                        {
                            Type::Float
                        } else {
                            Type::Int
                        }
                    }
                    _ => {
                        if let Some(sig) = self.tool_sigs.get(name) {
                            sig.output.clone()
                        } else if let Some(sig) = self.prompt_sigs.get(name) {
                            sig.output.clone()
                        } else if let Some(sig) = self.agent_sigs.get(name) {
                            sig.output.clone()
                        } else {
                            Type::Any
                        }
                    }
                }
            }
            Expr::ForeignCall { .. } => Type::Any,
            Expr::ListLiteral(items) => {
                if items.is_empty() {
                    Type::List(Box::new(Type::Any))
                } else {
                    let mut item_ty = self.infer_expr_type(&items[0], scope);
                    for item in &items[1..] {
                        let next_ty = self.infer_expr_type(item, scope);
                        item_ty = self.merge_literal_types(item_ty, next_ty);
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
                    out.insert(
                        field.key.node.clone(),
                        self.infer_expr_type(&field.value, scope),
                    );
                }
                Type::Struct(StructType::with_fields(out))
            }
            Expr::Paren(inner) => self.infer_expr_type(inner, scope),
        }
    }

    fn merge_literal_types(&self, current: Type, next: Type) -> Type {
        let current = self.type_env.resolve_type(&current);
        let next = self.type_env.resolve_type(&next);

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
        Type::Error
    }

    fn verify_artifact_reads(
        &mut self,
        expr: &Spanned<Expr>,
        assigned: &HashSet<String>,
        artifacts: &HashMap<String, Type>,
        context: &str,
    ) {
        let mut roots = Vec::new();
        self.collect_expr_roots(expr, &mut roots);
        let mut seen = HashSet::new();
        for (root, span) in roots {
            if artifacts.contains_key(&root)
                && !assigned.contains(&root)
                && seen.insert(root.clone())
            {
                self.error(
                    format!(
                        "{} may read artifact '{}' before it is assigned",
                        context, root
                    ),
                    span,
                );
            }
        }
    }

    fn verify_allowed_roots(
        &mut self,
        expr: &Spanned<Expr>,
        allowed_roots: &HashSet<String>,
        context: &str,
    ) {
        let mut roots = Vec::new();
        self.collect_expr_roots(expr, &mut roots);
        let mut seen = HashSet::new();
        for (root, span) in roots {
            if !allowed_roots.contains(&root) && seen.insert(root.clone()) {
                self.error(
                    format!("{} references unknown root identifier '{}'", context, root),
                    span,
                );
            }
        }
    }

    fn collect_expr_roots(&self, expr: &Spanned<Expr>, out: &mut Vec<(String, Span)>) {
        match &expr.node {
            Expr::Literal(_) => {}
            Expr::Ident(name) => out.push((name.clone(), expr.span)),
            Expr::FieldAccess(base, _) => self.collect_expr_roots(base, out),
            Expr::Binary(left, _, right) => {
                self.collect_expr_roots(left, out);
                self.collect_expr_roots(right, out);
            }
            Expr::Call(_, args) | Expr::ForeignCall { args, .. } => {
                for arg in args {
                    self.collect_expr_roots(arg, out);
                }
            }
            Expr::ListLiteral(items) => {
                for item in items {
                    self.collect_expr_roots(item, out);
                }
            }
            Expr::RecordLiteral(fields) => {
                for field in fields {
                    self.collect_expr_roots(&field.value, out);
                }
            }
            Expr::Paren(inner) => self.collect_expr_roots(inner, out),
        }
    }

    fn stage_kind_name(&self, kind: StageKind) -> &'static str {
        match kind {
            StageKind::Tool => "tool",
            StageKind::Prompt => "prompt",
            StageKind::Agent => "agent",
        }
    }

    fn target_kind_name(&self, kind: TaskTargetKind) -> String {
        match kind {
            TaskTargetKind::Stage(stage_kind) => {
                format!("{} stage", self.stage_kind_name(stage_kind))
            }
            TaskTargetKind::Loop => "loop".to_string(),
        }
    }

    fn binding_path_string(&self, path: &BindingPath) -> String {
        path.segments
            .iter()
            .map(|segment| segment.node.as_str())
            .collect::<Vec<_>>()
            .join(".")
    }

    fn error(&mut self, message: impl Into<String>, span: Span) {
        self.result.errors.push(VerifyError::new(message, span));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scaffold_syntax::parse;
    use scaffold_types::check;

    #[test]
    fn test_verify_simple_program() {
        let source = r#"
            type Position = { x: int, y: int }

            tool get_pos {
                input: { id: int }
                output: Position
            }
        "#;

        let program = parse(source).unwrap();
        let type_env = check(&program).unwrap();
        let result = verify(&program, &type_env);

        // Should have no errors
        assert!(!result.has_errors());
    }

    #[test]
    fn test_verify_valid_task_harness_objective() {
        let source = r#"
            artifact Notes = { text: string, score: float }

            prompt write_notes {
                input: { question: string }
                output: Notes
                template: "write"
            }

            agent revise_notes {
                input: Notes
                output: Notes
                tools: []
                system: "revise"
            }

            task answer_question {
                input: { question: string }
                output: { text: string }
                artifacts {
                    notes: Notes
                }
                stage draft using prompt write_notes {
                    in: { question: input.question }
                    out: notes
                }
                loop refine {
                    max_iters: 2
                    carry: [notes]
                    until: notes.score >= 0.9
                    stage revise using agent revise_notes {
                        in: notes
                        out: notes
                    }
                }
                emit {
                    text: notes.text
                }
            }

            harness baseline for task answer_question {
                bind draft {
                    model: "gpt-5"
                }
                tune {
                    draft.model in ["gpt-5", "gpt-5-mini"]
                }
            }

            objective quality for task answer_question {
                dataset: [
                    { input: { question: "hi" }, expected: { text: "hello" }, id: "one" }
                ]
                harness: baseline
                metric accuracy = output.text == expected.text
                score = accuracy
                split {
                    train: 0.7
                    val: 0.2
                    test: 0.1
                }
            }
        "#;

        let program = parse(source).unwrap();
        let type_env = check(&program).unwrap();
        let result = verify(&program, &type_env);
        assert!(!result.has_errors(), "{:#?}", result.errors);
    }

    #[test]
    fn test_verify_unassigned_artifact_read() {
        let source = r#"
            artifact Notes = { text: string }

            tool echo_notes {
                input: Notes
                output: Notes
            }

            task answer_question {
                input: { question: string }
                output: { text: string }
                artifacts {
                    notes: Notes
                }
                stage draft using tool echo_notes {
                    in: notes
                    out: notes
                }
                emit {
                    text: notes.text
                }
            }
        "#;

        let program = parse(source).unwrap();
        let type_env = check(&program).unwrap();
        let result = verify(&program, &type_env);
        assert!(result.errors.iter().any(|error| error
            .message
            .contains("may read artifact 'notes' before it is assigned")));
    }

    #[test]
    fn test_verify_loop_noncarry_write() {
        let source = r#"
            artifact Draft = { text: string }

            tool make_draft {
                input: { question: string }
                output: Draft
            }

            task answer_question {
                input: { question: string }
                output: { text: string }
                artifacts {
                    draft: Draft
                    revised: Draft
                }
                stage first using tool make_draft {
                    in: { question: input.question }
                    out: draft
                }
                loop refine {
                    max_iters: 2
                    carry: [draft]
                    until: draft.text == "done"
                    stage revise using tool make_draft {
                        in: { question: input.question }
                        out: revised
                    }
                }
                emit {
                    text: draft.text
                }
            }
        "#;

        let program = parse(source).unwrap();
        let type_env = check(&program).unwrap();
        let result = verify(&program, &type_env);
        assert!(result.errors.iter().any(|error| error
            .message
            .contains("may only write to carried artifacts")));
    }

    #[test]
    fn test_verify_stage_output_type_mismatch() {
        let source = r#"
            artifact Draft = { text: string }
            artifact Review = { score: float }

            prompt write_draft {
                input: { question: string }
                output: Draft
                template: "write"
            }

            task answer_question {
                input: { question: string }
                output: { text: string }
                artifacts {
                    review: Review
                }
                stage draft using prompt write_draft {
                    in: { question: input.question }
                    out: review
                }
                emit {
                    text: "ok"
                }
            }
        "#;

        let program = parse(source).unwrap();
        let type_env = check(&program).unwrap();
        let result = verify(&program, &type_env);
        assert!(result.errors.iter().any(|error| error
            .message
            .contains("writes { text: string } to artifact 'review'")));
    }

    #[test]
    fn test_verify_objective_harness_mismatch_and_bad_split() {
        let source = r#"
            task first_task {
                input: { question: string }
                output: { text: string }
                stage noop using prompt writer {
                    in: { question: input.question }
                    out: draft
                }
                emit {
                    text: "ok"
                }
            }

            prompt writer {
                input: { question: string }
                output: { text: string }
                template: "write"
            }

            harness baseline for task first_task {}

            task second_task {
                input: { question: string }
                output: { text: string }
                artifacts {
                    draft: { text: string }
                }
                stage write using prompt writer {
                    in: { question: input.question }
                    out: draft
                }
                emit {
                    text: draft.text
                }
            }

            objective quality for task second_task {
                dataset: [
                    { input: { question: "hi" }, expected: { text: "hello" }, id: "one" }
                ]
                harness: baseline
                metric accuracy = output.text == expected.text
                score = accuracy
                split {
                    train: 0.8
                    val: 0.3
                    test: 0.1
                }
            }
        "#;

        let program = parse(source).unwrap();
        let type_env = check(&program).unwrap();
        let result = verify(&program, &type_env);
        assert!(result.errors.iter().any(|error| error
            .message
            .contains("harness 'baseline' is for task 'first_task'")));
        assert!(result
            .errors
            .iter()
            .any(|error| error.message.contains("split must sum to 1.0")));
    }

    #[test]
    fn test_verify_invalid_stage_field_binding() {
        let source = r#"
            artifact Notes = { text: string }

            tool collect_notes {
                input: { question: string }
                output: Notes
            }

            task answer_question {
                input: { question: string }
                output: { text: string }
                artifacts {
                    notes: Notes
                }
                stage gather using tool collect_notes {
                    in: { question: input.question }
                    out: notes
                }
                emit {
                    text: notes.text
                }
            }

            harness baseline for task answer_question {
                bind gather {
                    model: "gpt-5"
                }
            }
        "#;

        let program = parse(source).unwrap();
        let type_env = check(&program).unwrap();
        let result = verify(&program, &type_env);
        assert!(result.errors.iter().any(|error| {
            error
                .message
                .contains("field 'model' is not mutable for tool stage 'gather'")
        }));
    }

    #[test]
    fn test_verify_stage_enabled_binding_and_tune() {
        let source = r#"
            artifact Notes = { text: string }

            prompt write_notes {
                input: { question: string }
                output: Notes
                template: "write"
            }

            task answer_question {
                input: { question: string }
                output: { text: string }
                artifacts {
                    notes: Notes
                }
                stage write using prompt write_notes {
                    in: { question: input.question }
                    out: notes
                }
                emit {
                    text: notes.text
                }
            }

            harness baseline for task answer_question {
                bind write {
                    enabled: false
                }
                tune {
                    write.enabled in [true, false]
                }
            }
        "#;

        let program = parse(source).unwrap();
        let type_env = check(&program).unwrap();
        let result = verify(&program, &type_env);
        assert!(result.errors.is_empty(), "{:?}", result.errors);
    }

    #[test]
    fn test_verify_invalid_component_binding_type_mismatch() {
        let source = r#"
            artifact Draft = { text: string }
            artifact Review = { score: float }

            prompt write_draft {
                input: { question: string }
                output: Draft
                template: "write"
            }

            prompt review_draft {
                input: { question: string }
                output: Review
                template: "review"
            }

            task answer_question {
                input: { question: string }
                output: { text: string }
                artifacts {
                    draft: Draft
                }
                stage write using prompt write_draft {
                    in: { question: input.question }
                    out: draft
                }
                emit {
                    text: draft.text
                }
            }

            harness baseline for task answer_question {
                bind write {
                    component: "review_draft"
                }
            }
        "#;

        let program = parse(source).unwrap();
        let type_env = check(&program).unwrap();
        let result = verify(&program, &type_env);
        assert!(result.errors.iter().any(|error| error
            .message
            .contains("swaps stage 'write' to prompt 'review_draft'")));
    }

    #[test]
    fn test_verify_invalid_tune_field_domain_and_unknown_tool() {
        let source = r#"
            artifact Notes = { text: string }

            prompt write_notes {
                input: { question: string }
                output: Notes
                template: "write"
            }

            agent review_notes {
                input: Notes
                output: Notes
                tools: []
                system: "review"
            }

            task answer_question {
                input: { question: string }
                output: { text: string }
                artifacts {
                    notes: Notes
                }
                stage write using prompt write_notes {
                    in: { question: input.question }
                    out: notes
                }
                stage review using agent review_notes {
                    in: notes
                    out: notes
                }
                emit {
                    text: notes.text
                }
            }

            harness baseline for task answer_question {
                bind review {
                    tools: [missing_tool]
                }
                tune {
                    write.temperature in ["hot"]
                    write.model in variants("writer")
                }
            }
        "#;

        let program = parse(source).unwrap();
        let type_env = check(&program).unwrap();
        let result = verify(&program, &type_env);
        assert!(result.errors.iter().any(|error| error
            .message
            .contains("references unknown tool 'missing_tool'")));
        assert!(result.errors.iter().any(|error| error
            .message
            .contains("'write.temperature' must be a static numeric expression")));
        assert!(result.errors.iter().any(|error| {
            error
                .message
                .contains("uses variants(...), but 'model' does not support prompt variants")
        }));
    }
}
