//! Pretty-printer: IR → .scaffold source
//!
//! Converts a `ScaffoldIR` back to human-readable `.scaffold` syntax.
//! This completes the round-trip needed for LLM-driven optimization:
//!
//! ```text
//! .scaffold → parse → IR (JSON) → LLM mutates → pretty-print → .scaffold
//! ```

use crate::ir::*;

/// Pretty-print a complete `ScaffoldIR` to `.scaffold` source.
pub fn pretty_print(ir: &ScaffoldIR) -> String {
    let mut out = String::new();

    // Extern crates
    for ec in &ir.extern_crates {
        pp_extern_crate(&mut out, ec);
        out.push('\n');
    }

    // Foreign modules
    for fm in &ir.foreign_modules {
        pp_foreign_module(&mut out, fm);
        out.push('\n');
    }

    // Type definitions
    for td in &ir.types {
        pp_type_def(&mut out, td);
        out.push('\n');
    }

    // Tools
    for tool in &ir.tools {
        pp_tool(&mut out, tool);
        out.push('\n');
    }

    // Prompts
    for prompt in &ir.prompts {
        pp_prompt(&mut out, prompt);
        out.push('\n');
    }

    // Agents
    for agent in &ir.agents {
        pp_agent(&mut out, agent);
        out.push('\n');
    }

    // Pipelines
    for pipeline in &ir.pipelines {
        pp_pipeline(&mut out, pipeline);
        out.push('\n');
    }

    // Tasks
    for task in &ir.tasks {
        pp_task(&mut out, task);
        out.push('\n');
    }

    // Harnesses
    for harness in &ir.harnesses {
        pp_harness(&mut out, harness);
        out.push('\n');
    }

    // Objectives
    for objective in &ir.objectives {
        pp_objective(&mut out, objective);
        out.push('\n');
    }

    out
}

// ─── Types ───────────────────────────────────────────────────────────────────

fn pp_type_ir(ty: &TypeIR) -> String {
    match ty {
        TypeIR::Bool => "bool".into(),
        TypeIR::Int => "int".into(),
        TypeIR::Float => "float".into(),
        TypeIR::String => "string".into(),
        TypeIR::Bytes => "bytes".into(),
        TypeIR::Any => "any".into(),
        TypeIR::List { element } => format!("list<{}>", pp_type_ir(element)),
        TypeIR::Map { key, value } => {
            format!("map<{}, {}>", pp_type_ir(key), pp_type_ir(value))
        }
        TypeIR::Option { inner } => format!("option<{}>", pp_type_ir(inner)),
        TypeIR::Result { ok, err } => {
            format!("result<{}, {}>", pp_type_ir(ok), pp_type_ir(err))
        }
        TypeIR::Struct { fields } => {
            if fields.is_empty() {
                "{}".into()
            } else {
                let mut parts = Vec::new();
                // Sort fields for deterministic output
                let mut sorted: Vec<_> = fields.iter().collect();
                sorted.sort_by_key(|(k, _)| (*k).clone());
                for (name, ty) in sorted {
                    parts.push(format!("{}: {}", name, pp_type_ir(ty)));
                }
                format!("{{ {} }}", parts.join(", "))
            }
        }
        TypeIR::Named { name } => name.clone(),
    }
}

fn pp_type_def(out: &mut String, td: &TypeDefIR) {
    out.push_str(&format!(
        "{} {} = {}\n",
        match td.kind {
            TypeDefKindIR::Type => "type",
            TypeDefKindIR::Artifact => "artifact",
        },
        td.name,
        pp_type_ir(&td.definition),
    ));
}

// ─── Extern / Foreign ────────────────────────────────────────────────────────

fn pp_extern_crate(out: &mut String, ec: &ExternCrateIR) {
    out.push_str(&format!("extern crate {} = \"{}\"", ec.name, ec.version));
    if !ec.features.is_empty() {
        out.push_str(&format!(
            " features [{}]",
            ec.features
                .iter()
                .map(|f| format!("\"{}\"", f))
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    out.push('\n');
}

fn pp_foreign_module(out: &mut String, fm: &ForeignModuleIR) {
    out.push_str(&format!("foreign {} {} {{\n", fm.language, fm.name));
    for ta in &fm.type_aliases {
        out.push_str(&format!("    type {} = {}\n", ta.name, ta.external_type));
    }
    for func in &fm.functions {
        let params: Vec<String> = func
            .params
            .iter()
            .map(|p| format!("{}: {}", p.name, pp_type_ir(&p.ty)))
            .collect();
        out.push_str(&format!(
            "    fn {}({}) -> {}\n",
            func.name,
            params.join(", "),
            pp_type_ir(&func.return_type)
        ));
    }
    out.push_str("}\n");
}

// ─── Tools ───────────────────────────────────────────────────────────────────

fn pp_tool(out: &mut String, tool: &ToolIR) {
    out.push_str(&format!("tool {} {{\n", tool.name));
    out.push_str(&format!("    input: {}\n", pp_type_ir(&tool.input)));
    out.push_str(&format!("    output: {}\n", pp_type_ir(&tool.output)));

    if let Some(spec) = &tool.spec {
        pp_tool_spec(out, spec);
    }

    if let Some(impl_) = &tool.implementation {
        out.push_str("    impl: ");
        pp_tool_impl(out, impl_, 1);
        out.push('\n');
    }

    for variant in &tool.variants {
        out.push_str(&format!("    variant {} {{\n", variant.name));
        out.push_str("        ");
        pp_tool_impl(out, &variant.implementation, 2);
        out.push('\n');
        out.push_str("    }\n");
    }

    out.push_str("}\n");
}

fn pp_tool_spec(out: &mut String, spec: &ToolSpecIR) {
    if spec.pure {
        out.push_str("    spec: pure\n");
    }
    for pre in &spec.preconditions {
        out.push_str(&format!("    pre: {}\n", pp_expr(pre)));
    }
    for post in &spec.postconditions {
        out.push_str(&format!("    post: {}\n", pp_expr(post)));
    }
}

fn pp_tool_impl(out: &mut String, impl_: &ToolImplIR, indent: usize) {
    match impl_ {
        ToolImplIR::Expr { expr } => {
            pp_tool_expr(out, expr, indent);
        }
        ToolImplIR::Sequence { statements } => {
            out.push_str("sequence {\n");
            for stmt in statements {
                pp_indent(out, indent + 1);
                pp_tool_statement(out, stmt, indent + 1);
                out.push('\n');
            }
            pp_indent(out, indent);
            out.push('}');
        }
        ToolImplIR::Parallel { statements } => {
            out.push_str("parallel {\n");
            for stmt in statements {
                pp_indent(out, indent + 1);
                pp_tool_statement(out, stmt, indent + 1);
                out.push('\n');
            }
            pp_indent(out, indent);
            out.push('}');
        }
    }
}

fn pp_tool_statement(out: &mut String, stmt: &ToolStatementIR, indent: usize) {
    if let Some(name) = &stmt.binding {
        out.push_str(&format!("let {} = ", name));
    }
    pp_tool_expr(out, &stmt.expr, indent);
}

fn pp_tool_expr(out: &mut String, expr: &ToolExprIR, indent: usize) {
    match expr {
        ToolExprIR::Ident { name } => out.push_str(name),
        ToolExprIR::FieldAccess { base, field } => {
            pp_tool_expr(out, base, indent);
            out.push('.');
            out.push_str(field);
        }
        ToolExprIR::ForeignCall {
            module,
            function,
            args,
        } => {
            out.push_str(&format!("{}::{}", module, function));
            out.push('(');
            for (i, arg) in args.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                pp_tool_expr(out, arg, indent);
            }
            out.push(')');
        }
        ToolExprIR::ToolCall { tool, args } => {
            out.push_str(tool);
            out.push('(');
            for (i, arg) in args.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                pp_tool_expr(out, arg, indent);
            }
            out.push(')');
        }
        ToolExprIR::Shell { command } => {
            out.push_str(&format!("shell(\"{}\")", escape_str(command)));
        }
        ToolExprIR::Pipe { left, right } => {
            pp_tool_expr(out, left, indent);
            out.push_str(" |> ");
            pp_tool_expr(out, right, indent);
        }
        ToolExprIR::If {
            condition,
            then_branch,
            else_branch,
        } => {
            out.push_str(&format!("if {} {{\n", pp_expr(condition)));
            pp_indent(out, indent + 1);
            pp_tool_impl(out, then_branch, indent + 1);
            out.push('\n');
            pp_indent(out, indent);
            out.push('}');
            if let Some(else_) = else_branch {
                out.push_str(" else {\n");
                pp_indent(out, indent + 1);
                pp_tool_impl(out, else_, indent + 1);
                out.push('\n');
                pp_indent(out, indent);
                out.push('}');
            }
        }
        ToolExprIR::Match { scrutinee, arms } => {
            out.push_str("match ");
            pp_tool_expr(out, scrutinee, indent);
            out.push_str(" {\n");
            for arm in arms {
                pp_indent(out, indent + 1);
                out.push_str(&pp_expr(&arm.pattern));
                out.push_str(" => {\n");
                pp_indent(out, indent + 2);
                pp_tool_impl(out, &arm.body, indent + 2);
                out.push('\n');
                pp_indent(out, indent + 1);
                out.push_str("}\n");
            }
            pp_indent(out, indent);
            out.push('}');
        }
        ToolExprIR::For {
            variable,
            iterable,
            body,
        } => {
            out.push_str(&format!("for {} in ", variable));
            pp_tool_expr(out, iterable, indent);
            out.push_str(" {\n");
            pp_indent(out, indent + 1);
            pp_tool_impl(out, body, indent + 1);
            out.push('\n');
            pp_indent(out, indent);
            out.push('}');
        }
        ToolExprIR::While { condition, body } => {
            out.push_str(&format!("while {} {{\n", pp_expr(condition)));
            pp_indent(out, indent + 1);
            pp_tool_impl(out, body, indent + 1);
            out.push('\n');
            pp_indent(out, indent);
            out.push('}');
        }
        ToolExprIR::Loop { body } => {
            out.push_str("loop {\n");
            pp_indent(out, indent + 1);
            pp_tool_impl(out, body, indent + 1);
            out.push('\n');
            pp_indent(out, indent);
            out.push('}');
        }
        ToolExprIR::Break => out.push_str("break"),
        ToolExprIR::Continue => out.push_str("continue"),
        ToolExprIR::Literal { value } => out.push_str(&pp_literal(value)),
        ToolExprIR::MapLiteral { entries } => {
            out.push_str("{\n");
            for entry in entries {
                pp_indent(out, indent + 1);
                out.push_str(&format!("\"{}\": ", escape_str(&entry.key)));
                pp_tool_expr(out, &entry.value, indent + 1);
                out.push('\n');
            }
            pp_indent(out, indent);
            out.push('}');
        }
        ToolExprIR::Expr { expr } => {
            out.push_str(&pp_expr(expr));
        }
    }
}

// ─── Expressions ─────────────────────────────────────────────────────────────

fn pp_expr(expr: &ExprIR) -> String {
    match expr {
        ExprIR::Literal { value } => pp_literal(value),
        ExprIR::Ident { name } => name.clone(),
        ExprIR::FieldAccess { base, field } => {
            format!("{}.{}", pp_expr(base), field)
        }
        ExprIR::Binary { left, op, right } => {
            format!("{} {} {}", pp_expr(left), op, pp_expr(right))
        }
        ExprIR::Call { function, args } => {
            let arg_strs: Vec<String> = args.iter().map(pp_expr).collect();
            format!("{}({})", function, arg_strs.join(", "))
        }
        ExprIR::ForeignCall {
            module,
            function,
            args,
        } => {
            let arg_strs: Vec<String> = args.iter().map(pp_expr).collect();
            format!("{}::{}({})", module, function, arg_strs.join(", "))
        }
        ExprIR::List { elements } => {
            let item_strs: Vec<String> = elements.iter().map(pp_expr).collect();
            format!("[{}]", item_strs.join(", "))
        }
        ExprIR::Record { fields } => {
            let field_strs: Vec<String> = fields
                .iter()
                .map(|f| format!("{}: {}", f.key, pp_expr(&f.value)))
                .collect();
            format!("{{{}}}", field_strs.join(", "))
        }
    }
}

fn pp_literal(lit: &LiteralIR) -> String {
    match lit {
        LiteralIR::Int { value } => value.to_string(),
        LiteralIR::Float { value } => {
            let s = value.to_string();
            if s.contains('.') {
                s
            } else {
                format!("{}.0", s)
            }
        }
        LiteralIR::String { value } => format!("\"{}\"", escape_str(value)),
        LiteralIR::Bool { value } => value.to_string(),
        LiteralIR::Null => "null".into(),
    }
}

// ─── Prompts ─────────────────────────────────────────────────────────────────

fn pp_prompt(out: &mut String, prompt: &PromptIR) {
    out.push_str(&format!("prompt {} {{\n", prompt.name));
    out.push_str(&format!("    input: {}\n", pp_type_ir(&prompt.input)));
    out.push_str(&format!("    output: {}\n", pp_type_ir(&prompt.output)));

    if let Some(sys) = &prompt.system {
        out.push_str(&format!("    system: {}\n", pp_string_or_file(sys)));
    }

    out.push_str(&format!(
        "    template: {}\n",
        pp_string_or_file(&prompt.template)
    ));

    out.push_str("}\n");
}

fn pp_string_or_file(s: &StringOrFileIR) -> String {
    match s {
        StringOrFileIR::Literal { value } => format!("\"{}\"", escape_str(value)),
        StringOrFileIR::File { path } => format!("file(\"{}\")", escape_str(path)),
    }
}

// ─── Agents ──────────────────────────────────────────────────────────────────

fn pp_agent(out: &mut String, agent: &AgentIR) {
    out.push_str(&format!("agent {} {{\n", agent.name));
    out.push_str(&format!("    input: {}\n", pp_type_ir(&agent.input)));
    out.push_str(&format!("    output: {}\n", pp_type_ir(&agent.output)));

    let tools_str: Vec<String> = agent.tools.iter().cloned().collect();
    out.push_str(&format!("    tools: [{}]\n", tools_str.join(", ")));

    out.push_str(&format!(
        "    system: {}\n",
        pp_string_or_file(&agent.system)
    ));

    if let Some(model) = &agent.model {
        out.push_str(&format!("    model: \"{}\"\n", model));
    }

    if let Some(max_turns) = agent.max_turns {
        out.push_str(&format!("    max_turns: {}\n", max_turns));
    }

    if let Some(reward) = &agent.reward {
        out.push_str(&format!("    reward: {}\n", pp_expr(reward)));
    }

    if let Some(done) = &agent.done {
        out.push_str(&format!("    done: {}\n", pp_expr(done)));
    }

    match &agent.on_error {
        ErrorStrategyIR::Abort => {} // default, don't print
        ErrorStrategyIR::Retry { count } => {
            out.push_str(&format!("    on_error: retry({})\n", count));
        }
    }

    if let Some(timeout) = agent.timeout {
        out.push_str(&format!("    timeout: {}\n", timeout));
    }

    out.push_str("}\n");
}

// ─── Pipelines ───────────────────────────────────────────────────────────────

fn pp_pipeline(out: &mut String, pipeline: &PipelineIR) {
    out.push_str(&format!("pipeline {} {{\n", pipeline.name));
    out.push_str(&format!("    input: {}\n", pp_type_ir(&pipeline.input)));
    out.push_str(&format!("    output: {}\n", pp_type_ir(&pipeline.output)));

    out.push_str("    steps {\n");
    for step in &pipeline.steps {
        pp_pipeline_step(out, step, 2);
    }
    out.push_str("    }\n");

    if let Some(reward) = &pipeline.reward {
        out.push_str(&format!("    reward: {}\n", pp_expr(reward)));
    }

    out.push_str("}\n");
}

fn pp_pipeline_step(out: &mut String, step: &PipelineStepIR, indent: usize) {
    pp_indent(out, indent);
    if let Some(name) = &step.binding {
        out.push_str(&format!("let {} = ", name));
    }
    pp_pipeline_call(out, &step.call, indent);
    out.push('\n');
}

fn pp_pipeline_call(out: &mut String, call: &PipelineCallIR, indent: usize) {
    match call {
        PipelineCallIR::Prompt { name, args }
        | PipelineCallIR::Tool { name, args }
        | PipelineCallIR::Agent { name, args } => {
            out.push_str(name);
            out.push('(');
            for (i, arg) in args.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                pp_tool_expr(out, arg, indent);
            }
            out.push(')');
        }
        PipelineCallIR::Expr { expr } => {
            pp_tool_expr(out, expr, indent);
        }
        PipelineCallIR::Parallel { branches } => {
            out.push_str("parallel {\n");
            for branch in branches {
                pp_indent(out, indent + 1);
                out.push_str("{\n");
                for step in branch {
                    pp_pipeline_step(out, step, indent + 2);
                }
                pp_indent(out, indent + 1);
                out.push_str("}\n");
            }
            pp_indent(out, indent);
            out.push('}');
        }
        PipelineCallIR::If {
            condition,
            then_steps,
            else_steps,
        } => {
            out.push_str(&format!("if {} {{\n", pp_expr(condition)));
            for step in then_steps {
                pp_pipeline_step(out, step, indent + 1);
            }
            pp_indent(out, indent);
            out.push('}');
            if !else_steps.is_empty() {
                out.push_str(" else {\n");
                for step in else_steps {
                    pp_pipeline_step(out, step, indent + 1);
                }
                pp_indent(out, indent);
                out.push('}');
            }
        }
        PipelineCallIR::Match { scrutinee, arms } => {
            out.push_str(&format!("match {} {{\n", pp_expr(scrutinee)));
            for arm in arms {
                pp_indent(out, indent + 1);
                out.push_str(&format!("{} => {{\n", pp_expr(&arm.pattern)));
                for step in &arm.steps {
                    pp_pipeline_step(out, step, indent + 2);
                }
                pp_indent(out, indent + 1);
                out.push_str("}\n");
            }
            pp_indent(out, indent);
            out.push('}');
        }
    }
}

// ─── Tasks / Harnesses / Objectives ─────────────────────────────────────────

fn pp_task(out: &mut String, task: &TaskIR) {
    out.push_str(&format!("task {} {{\n", task.name));
    out.push_str(&format!("    input: {}\n", pp_type_ir(&task.input)));
    out.push_str(&format!("    output: {}\n", pp_type_ir(&task.output)));

    if !task.artifacts.is_empty() {
        out.push_str("    artifacts {\n");
        for artifact in &task.artifacts {
            out.push_str(&format!(
                "        {}: {}\n",
                artifact.name,
                pp_type_ir(&artifact.ty)
            ));
        }
        out.push_str("    }\n");
    }

    for node in &task.body {
        pp_task_node(out, node, 1);
    }

    out.push_str("    emit {\n");
    for field in &task.emit {
        out.push_str(&format!(
            "        {}: {}\n",
            field.name,
            pp_expr(&field.value)
        ));
    }
    out.push_str("    }\n");
    out.push_str("}\n");
}

fn pp_task_node(out: &mut String, node: &TaskNodeIR, indent: usize) {
    match node {
        TaskNodeIR::Stage(stage) => pp_stage(out, stage, indent),
        TaskNodeIR::Loop(loop_ir) => pp_loop(out, loop_ir, indent),
        TaskNodeIR::Branch(branch) => pp_branch(out, branch, indent),
    }
}

fn pp_stage(out: &mut String, stage: &StageIR, indent: usize) {
    pp_indent(out, indent);
    out.push_str(&format!(
        "stage {} using {} {} {{\n",
        stage.name,
        pp_stage_kind(stage.stage_kind),
        stage.component
    ));
    pp_indent(out, indent + 1);
    out.push_str(&format!("in: {}\n", pp_expr(&stage.input)));
    pp_indent(out, indent + 1);
    out.push_str(&format!("out: {}\n", stage.output));
    if let Some(when) = &stage.when {
        pp_indent(out, indent + 1);
        out.push_str(&format!("when: {}\n", pp_expr(when)));
    }
    pp_indent(out, indent);
    out.push_str("}\n");
}

fn pp_loop(out: &mut String, loop_ir: &LoopIR, indent: usize) {
    pp_indent(out, indent);
    out.push_str(&format!("loop {} {{\n", loop_ir.name));
    pp_indent(out, indent + 1);
    out.push_str(&format!("max_iters: {}\n", pp_expr(&loop_ir.max_iters)));
    pp_indent(out, indent + 1);
    out.push_str(&format!("carry: [{}]\n", loop_ir.carry.join(", ")));
    pp_indent(out, indent + 1);
    out.push_str(&format!("until: {}\n", pp_expr(&loop_ir.until)));
    for node in &loop_ir.body {
        pp_task_node(out, node, indent + 1);
    }
    pp_indent(out, indent);
    out.push_str("}\n");
}

fn pp_branch(out: &mut String, branch: &BranchIR, indent: usize) {
    pp_indent(out, indent);
    out.push_str(&format!("if {} {{\n", pp_expr(&branch.condition)));
    for node in &branch.then_body {
        pp_task_node(out, node, indent + 1);
    }
    pp_indent(out, indent);
    out.push('}');
    if !branch.else_body.is_empty() {
        out.push_str(" else {\n");
        for node in &branch.else_body {
            pp_task_node(out, node, indent + 1);
        }
        pp_indent(out, indent);
        out.push('}');
    }
    out.push('\n');
}

fn pp_harness(out: &mut String, harness: &HarnessIR) {
    out.push_str(&format!(
        "harness {} for task {} {{\n",
        harness.name, harness.task
    ));
    if !harness.defaults.is_empty() {
        out.push_str("    defaults {\n");
        for binding in &harness.defaults {
            out.push_str(&format!(
                "        {}: {}\n",
                pp_binding_path(&binding.key),
                pp_expr(&binding.value)
            ));
        }
        out.push_str("    }\n");
    }

    for binding in &harness.bindings {
        out.push_str(&format!("    bind {} {{\n", binding.target));
        for item in &binding.bindings {
            out.push_str(&format!(
                "        {}: {}\n",
                pp_binding_path(&item.key),
                pp_expr(&item.value)
            ));
        }
        out.push_str("    }\n");
    }

    if !harness.tunables.is_empty() {
        out.push_str("    tune {\n");
        for tunable in &harness.tunables {
            out.push_str(&format!(
                "        {} {} {}\n",
                pp_binding_path(&tunable.path),
                pp_tune_operator(tunable.operator),
                pp_finite_domain(&tunable.domain)
            ));
        }
        out.push_str("    }\n");
    }
    out.push_str("}\n");
}

fn pp_objective(out: &mut String, objective: &ObjectiveIR) {
    out.push_str(&format!(
        "objective {} for task {} {{\n",
        objective.name, objective.task
    ));
    out.push_str(&format!(
        "    dataset: {}\n",
        pp_dataset_spec(&objective.dataset)
    ));
    out.push_str(&format!("    harness: {}\n", objective.harness));
    if let Some(repeats) = objective.repeats {
        out.push_str(&format!("    repeats: {}\n", repeats));
    }
    for metric in &objective.metrics {
        out.push_str(&format!(
            "    metric {} = {}\n",
            metric.name,
            pp_expr(&metric.expr)
        ));
    }
    out.push_str(&format!("    score = {}\n", pp_expr(&objective.score)));
    if let Some(split) = &objective.split {
        out.push_str("    split {\n");
        out.push_str(&format!("        train: {}\n", split.train));
        out.push_str(&format!("        val: {}\n", split.val));
        out.push_str(&format!("        test: {}\n", split.test));
        out.push_str("    }\n");
    }
    if let Some(select) = &objective.select {
        out.push_str("    select {\n");
        out.push_str(&format!("        primary: {}\n", pp_expr(&select.primary)));
        if !select.tie_breakers.is_empty() {
            let ties: Vec<String> = select.tie_breakers.iter().map(pp_expr).collect();
            out.push_str(&format!("        tie_breakers: [{}]\n", ties.join(", ")));
        }
        out.push_str("    }\n");
    }
    out.push_str("}\n");
}

fn pp_stage_kind(kind: StageKindIR) -> &'static str {
    match kind {
        StageKindIR::Tool => "tool",
        StageKindIR::Prompt => "prompt",
        StageKindIR::Agent => "agent",
    }
}

fn pp_binding_path(path: &BindingPathIR) -> String {
    path.segments.join(".")
}

fn pp_tune_operator(op: TuneOperatorIR) -> &'static str {
    match op {
        TuneOperatorIR::In => "in",
        TuneOperatorIR::SubsetOf => "subset_of",
    }
}

fn pp_finite_domain(domain: &FiniteDomainIR) -> String {
    match domain {
        FiniteDomainIR::List { values } => {
            let items: Vec<String> = values.iter().map(pp_expr).collect();
            format!("[{}]", items.join(", "))
        }
        FiniteDomainIR::Variants { name } => format!("variants(\"{}\")", escape_str(name)),
    }
}

fn pp_dataset_spec(dataset: &DatasetSpecIR) -> String {
    match dataset {
        DatasetSpecIR::File { path } => format!("file(\"{}\")", escape_str(path)),
        DatasetSpecIR::Inline { cases } => {
            let cases: Vec<String> = cases.iter().map(pp_inline_dataset_case).collect();
            format!("[{}]", cases.join(", "))
        }
    }
}

fn pp_inline_dataset_case(case: &InlineDatasetCaseIR) -> String {
    let mut parts = vec![format!("input: {}", pp_expr(&case.input))];
    if let Some(expected) = &case.expected {
        parts.push(format!("expected: {}", pp_expr(expected)));
    }
    if let Some(id) = &case.id {
        parts.push(format!("id: \"{}\"", escape_str(id)));
    }
    format!("{{ {} }}", parts.join(", "))
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn pp_indent(out: &mut String, level: usize) {
    for _ in 0..level {
        out.push_str("    ");
    }
}

fn escape_str(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_round_trip_type_def() {
        let td = TypeDefIR {
            kind: TypeDefKindIR::Type,
            name: "SearchPayload".into(),
            definition: TypeIR::Struct {
                fields: {
                    let mut f = HashMap::new();
                    f.insert("urls".into(), TypeIR::String);
                    f.insert(
                        "items".into(),
                        TypeIR::List {
                            element: Box::new(TypeIR::String),
                        },
                    );
                    f
                },
            },
        };
        let out = {
            let mut s = String::new();
            pp_type_def(&mut s, &td);
            s
        };
        assert!(out.contains("type SearchPayload ="));
        assert!(out.contains("urls: string"));
        assert!(out.contains("items: list<string>"));
    }

    #[test]
    fn test_round_trip_prompt() {
        let prompt = PromptIR {
            name: "draft_answer".into(),
            input: TypeIR::Struct {
                fields: {
                    let mut f = HashMap::new();
                    f.insert("query".into(), TypeIR::String);
                    f
                },
            },
            output: TypeIR::Struct {
                fields: {
                    let mut f = HashMap::new();
                    f.insert("answer".into(), TypeIR::String);
                    f
                },
            },
            template: StringOrFileIR::Literal {
                value: "Answer: {query}".into(),
            },
            system: None,
        };
        let out = {
            let mut s = String::new();
            pp_prompt(&mut s, &prompt);
            s
        };
        assert!(out.contains("prompt draft_answer"));
        assert!(out.contains("template:"));
        assert!(out.contains("Answer: {query}"));
    }

    #[test]
    fn test_round_trip_agent() {
        let agent = AgentIR {
            name: "reviewer".into(),
            input: TypeIR::Struct {
                fields: {
                    let mut f = HashMap::new();
                    f.insert("query".into(), TypeIR::String);
                    f
                },
            },
            output: TypeIR::Struct {
                fields: {
                    let mut f = HashMap::new();
                    f.insert("verdict".into(), TypeIR::String);
                    f
                },
            },
            tools: vec![],
            system: StringOrFileIR::Literal {
                value: "You are a reviewer.".into(),
            },
            model: None,
            max_turns: Some(4),
            reward: Some(ExprIR::Binary {
                left: Box::new(ExprIR::Ident {
                    name: "verdict".into(),
                }),
                op: "!=".into(),
                right: Box::new(ExprIR::Literal {
                    value: LiteralIR::String { value: "".into() },
                }),
            }),
            done: None,
            on_error: ErrorStrategyIR::Abort,
            timeout: None,
        };
        let out = {
            let mut s = String::new();
            pp_agent(&mut s, &agent);
            s
        };
        assert!(out.contains("agent reviewer"));
        assert!(out.contains("max_turns: 4"));
        assert!(out.contains("reward:"));
        assert!(out.contains("verdict != \"\""));
    }

    #[test]
    fn test_round_trip_pipeline() {
        let pipeline = PipelineIR {
            name: "analyze".into(),
            input: TypeIR::Struct {
                fields: {
                    let mut f = HashMap::new();
                    f.insert("text".into(), TypeIR::String);
                    f
                },
            },
            output: TypeIR::Struct {
                fields: {
                    let mut f = HashMap::new();
                    f.insert("summary".into(), TypeIR::String);
                    f
                },
            },
            steps: vec![
                PipelineStepIR {
                    binding: Some("result".into()),
                    call: PipelineCallIR::Prompt {
                        name: "summarize".into(),
                        args: vec![ToolExprIR::Ident {
                            name: "text".into(),
                        }],
                    },
                },
                PipelineStepIR {
                    binding: Some("summary".into()),
                    call: PipelineCallIR::Expr {
                        expr: ToolExprIR::FieldAccess {
                            base: Box::new(ToolExprIR::Ident {
                                name: "result".into(),
                            }),
                            field: "summary".into(),
                        },
                    },
                },
            ],
            reward: Some(ExprIR::Binary {
                left: Box::new(ExprIR::Call {
                    function: "length".into(),
                    args: vec![ExprIR::Ident {
                        name: "summary".into(),
                    }],
                }),
                op: ">".into(),
                right: Box::new(ExprIR::Literal {
                    value: LiteralIR::Int { value: 10 },
                }),
            }),
        };
        let out = {
            let mut s = String::new();
            pp_pipeline(&mut s, &pipeline);
            s
        };
        assert!(out.contains("pipeline analyze"));
        assert!(out.contains("let result = summarize(text)"));
        assert!(out.contains("let summary = result.summary"));
        assert!(out.contains("reward: length(summary) > 10"));
    }

    #[test]
    fn test_full_ir() {
        let ir = ScaffoldIR {
            version: "0.1.0".into(),
            types: vec![TypeDefIR {
                kind: TypeDefKindIR::Type,
                name: "Output".into(),
                definition: TypeIR::Struct {
                    fields: {
                        let mut f = HashMap::new();
                        f.insert("answer".into(), TypeIR::String);
                        f
                    },
                },
            }],
            extern_crates: vec![],
            foreign_modules: vec![],
            tools: vec![ToolIR {
                name: "search".into(),
                input: TypeIR::Struct {
                    fields: {
                        let mut f = HashMap::new();
                        f.insert("query".into(), TypeIR::String);
                        f
                    },
                },
                output: TypeIR::String,
                implementation: Some(ToolImplIR::Expr {
                    expr: ToolExprIR::Shell {
                        command: "echo {query}".into(),
                    },
                }),
                spec: None,
                variants: vec![],
            }],
            prompts: vec![],
            agents: vec![],
            pipelines: vec![],
            tasks: vec![],
            harnesses: vec![],
            objectives: vec![],
        };
        let out = pretty_print(&ir);
        assert!(out.contains("type Output ="));
        assert!(out.contains("tool search"));
        assert!(out.contains("shell(\"echo {query}\")"));
    }

    #[test]
    fn test_round_trip_artifact_task_and_harness() {
        let ir = ScaffoldIR {
            version: "0.1.0".into(),
            types: vec![TypeDefIR {
                kind: TypeDefKindIR::Artifact,
                name: "ResearchNotes".into(),
                definition: TypeIR::Struct {
                    fields: {
                        let mut f = HashMap::new();
                        f.insert("summary".into(), TypeIR::String);
                        f.insert("score".into(), TypeIR::Float);
                        f
                    },
                },
            }],
            extern_crates: vec![],
            foreign_modules: vec![],
            tools: vec![],
            prompts: vec![],
            agents: vec![],
            pipelines: vec![],
            tasks: vec![TaskIR {
                name: "answer_question".into(),
                input: TypeIR::Struct {
                    fields: {
                        let mut f = HashMap::new();
                        f.insert("question".into(), TypeIR::String);
                        f
                    },
                },
                output: TypeIR::Struct {
                    fields: {
                        let mut f = HashMap::new();
                        f.insert("answer".into(), TypeIR::String);
                        f
                    },
                },
                artifacts: vec![ArtifactSlotIR {
                    name: "notes".into(),
                    ty: TypeIR::Named {
                        name: "ResearchNotes".into(),
                    },
                }],
                body: vec![TaskNodeIR::Stage(StageIR {
                    name: "draft".into(),
                    stage_kind: StageKindIR::Prompt,
                    component: "writer".into(),
                    input: ExprIR::Record {
                        fields: vec![ExprFieldIR {
                            key: "question".into(),
                            value: ExprIR::FieldAccess {
                                base: Box::new(ExprIR::Ident {
                                    name: "input".into(),
                                }),
                                field: "question".into(),
                            },
                        }],
                    },
                    output: "notes".into(),
                    when: None,
                })],
                emit: vec![EmitFieldIR {
                    name: "answer".into(),
                    value: ExprIR::FieldAccess {
                        base: Box::new(ExprIR::Ident {
                            name: "notes".into(),
                        }),
                        field: "summary".into(),
                    },
                }],
            }],
            harnesses: vec![HarnessIR {
                name: "baseline".into(),
                task: "answer_question".into(),
                defaults: vec![BindingIR {
                    key: BindingPathIR {
                        segments: vec!["model".into()],
                    },
                    value: ExprIR::Literal {
                        value: LiteralIR::String {
                            value: "gpt-5".into(),
                        },
                    },
                }],
                bindings: vec![TargetBindingIR {
                    target: "draft".into(),
                    bindings: vec![BindingIR {
                        key: BindingPathIR {
                            segments: vec!["temperature".into()],
                        },
                        value: ExprIR::Literal {
                            value: LiteralIR::Float { value: 0.2 },
                        },
                    }],
                }],
                tunables: vec![TunableIR {
                    path: BindingPathIR {
                        segments: vec!["draft".into(), "model".into()],
                    },
                    operator: TuneOperatorIR::In,
                    domain: FiniteDomainIR::List {
                        values: vec![
                            ExprIR::Literal {
                                value: LiteralIR::String {
                                    value: "gpt-5-mini".into(),
                                },
                            },
                            ExprIR::Literal {
                                value: LiteralIR::String {
                                    value: "gpt-5".into(),
                                },
                            },
                        ],
                    },
                }],
            }],
            objectives: vec![ObjectiveIR {
                name: "quality".into(),
                task: "answer_question".into(),
                harness: "baseline".into(),
                dataset: DatasetSpecIR::File {
                    path: "dataset.jsonl".into(),
                },
                repeats: Some(2),
                metrics: vec![MetricIR {
                    name: "accuracy".into(),
                    expr: ExprIR::Ident {
                        name: "output".into(),
                    },
                }],
                score: ExprIR::Ident {
                    name: "accuracy".into(),
                },
                split: Some(SplitIR {
                    train: 0.7,
                    val: 0.2,
                    test: 0.1,
                }),
                select: Some(SelectIR {
                    primary: ExprIR::Ident {
                        name: "accuracy".into(),
                    },
                    tie_breakers: vec![ExprIR::FieldAccess {
                        base: Box::new(ExprIR::Ident {
                            name: "rollout".into(),
                        }),
                        field: "cost".into(),
                    }],
                }),
            }],
        };

        let out = pretty_print(&ir);
        assert!(out.contains("artifact ResearchNotes ="));
        assert!(out.contains("task answer_question"));
        assert!(out.contains("stage draft using prompt writer"));
        assert!(out.contains("harness baseline for task answer_question"));
        assert!(out.contains("objective quality for task answer_question"));
    }
}
