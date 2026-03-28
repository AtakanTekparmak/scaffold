//! LLM-guided meta-agent for proposing targeted mutations (Algorithm 2, DGM-H).
//!
//! Instead of random dice rolls, the meta-agent receives rich context about the
//! parent candidate, archive history, and available nodes, then proposes a
//! structured mutation via an LLM call.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::path::PathBuf;

use scaffold_ir::ir::*;

use crate::error::{Error, Result};
use crate::llm::{self, LlmConfig};
use crate::mutations::Mutation;

use crate::optimizer::{Archive, Candidate};

/// A mutation proposal from the meta-agent, with reasoning.
#[derive(Debug, Clone)]
pub struct MutationProposal {
    pub mutation: Mutation,
    pub reasoning: String,
    /// Size of the accepted context payload sent to the meta-agent (chars).
    pub context_chars: usize,
    pub system_chars: usize,
}

/// The meta-agent: an LLM-based mutation proposer.
pub struct MetaAgent {
    model: String,
    /// Optional path to a debug log file. When set, every LLM interaction
    /// (system prompt, context, raw response, parse result) is appended.
    log_path: Option<PathBuf>,
}

impl MetaAgent {
    pub fn new(model: &str) -> Self {
        Self {
            model: model.to_string(),
            log_path: None,
        }
    }

    pub fn with_log(mut self, path: Option<PathBuf>) -> Self {
        self.log_path = path;
        self
    }

    /// Propose a mutation given the parent candidate, archive, IR, and objective.
    ///
    /// Retries up to `MAX_RETRIES` times, feeding validation errors and
    /// deduplication warnings back into the context so the meta-agent can
    /// self-correct. Never falls back to random mutations.
    pub async fn propose_mutation(
        &self,
        parent: &Candidate,
        archive: &Archive,
        ir: &ScaffoldIR,
        objective: &ObjectiveIR,
        allowed_mutations: &[String],
    ) -> Result<MutationProposal> {
        const MAX_RETRIES: usize = 10;

        let base_context = build_context(parent, archive, ir, objective, allowed_mutations);
        let system_chars = SYSTEM_PROMPT.len();
        let llm_config = LlmConfig::new()
            .with_model(&self.model)
            .with_temperature(0.7)
            .with_system_prompt(SYSTEM_PROMPT);

        // Log base context on first call (system prompt is constant, log it once)
        self.log_section("SYSTEM PROMPT", SYSTEM_PROMPT);
        self.log_section("BASE CONTEXT", &base_context);

        let mut errors: Vec<String> = Vec::new();
        let mut seen_labels: Vec<String> = Vec::new();

        for attempt in 0..=MAX_RETRIES {
            let context = if errors.is_empty() {
                base_context.clone()
            } else {
                let mut ctx = base_context.clone();
                ctx.push_str("\n\n## Previous Proposal Errors (FIX THESE)\n");
                ctx.push_str("Your previous proposals were REJECTED. You MUST propose something different.\n");
                for (i, err) in errors.iter().enumerate() {
                    ctx.push_str(&format!("{}. {}\n", i + 1, err));
                }
                ctx.push_str("\nDo NOT repeat any of the rejected proposals. Choose a completely different mutation.\n");
                ctx
            };

            if attempt > 0 {
                self.log_section(&format!("RETRY CONTEXT (attempt {})", attempt), &context);
            }

            let response = llm::query_with_config(&context, &llm_config).await?;
            self.log_section(&format!("LLM RESPONSE (attempt {})", attempt), &response);

            match parse_proposal(&response, ir, &parent.graph, objective, allowed_mutations) {
                Ok(mut proposal) => {
                    // Deduplication: reject if same label was already proposed in this retry sequence
                    let label = proposal.mutation.short_label();
                    if seen_labels.contains(&label) {
                        let msg = format!(
                            "DUPLICATE: you already proposed '{}' which was rejected. Try a DIFFERENT mutation kind or target.",
                            label,
                        );
                        self.log_section("REJECTED (duplicate)", &msg);
                        errors.push(msg);
                        continue;
                    }
                    seen_labels.push(label.clone());

                    // Validate mutation can be applied
                    match crate::mutations::apply_mutation(&parent.graph, &proposal.mutation, ir) {
                        crate::mutations::MutationResult::Ok(_) => {
                            proposal.context_chars = context.len();
                            proposal.system_chars = system_chars;
                            self.log_section(
                                "ACCEPTED",
                                &format!("mutation={} reasoning={}", label, proposal.reasoning,),
                            );
                            return Ok(proposal);
                        }
                        crate::mutations::MutationResult::Skipped(reason) => {
                            let msg = format!("Mutation '{}' failed to apply: {}", label, reason,);
                            self.log_section("REJECTED (apply failed)", &msg);
                            errors.push(msg);
                            continue;
                        }
                    }
                }
                Err(e) => {
                    let msg = format!("Parse/validation error: {}", e);
                    self.log_section("REJECTED (parse error)", &msg);
                    if attempt == MAX_RETRIES {
                        return Err(e);
                    }
                    errors.push(msg);
                    continue;
                }
            }
        }

        Err(Error::Runtime(format!(
            "meta-agent failed after {} retries: {}",
            MAX_RETRIES,
            errors.last().unwrap_or(&"unknown".to_string()),
        )))
    }

    /// Append a labeled section to the debug log file (if configured).
    fn log_section(&self, label: &str, content: &str) {
        if let Some(ref path) = self.log_path {
            let elapsed = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default();
            let secs = elapsed.as_secs();
            let millis = elapsed.subsec_millis();
            let separator = "=".repeat(80);
            let mut file = match std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
            {
                Ok(f) => f,
                Err(_) => return,
            };
            let _ = writeln!(
                file,
                "\n{}\n[{}.{:03}] {}\n{}\n{}",
                separator, secs, millis, label, separator, content
            );
        }
    }
}

/// Structural mutation kinds that edit_graph subsumes.
const STRUCTURAL_MUTATIONS: &[&str] = &[
    "insert_verify",
    "wrap_retry",
    "insert_step",
    "remove_step",
    "replace_component",
    "fan_out_parallel",
    "replace_with_subgraph",
    "add_prompt_step",
];

const SYSTEM_PROMPT: &str = r#"You are a meta-optimizer for computational graphs. You analyze an optimization archive and propose targeted mutations to improve a graph's score on a dataset evaluation.

You can propose exactly ONE mutation as a JSON object with these fields:
- "kind": one of the mutation types listed below
- "reasoning": a brief explanation of why this mutation should help
- Plus kind-specific fields

The available mutation types are listed in the context under "Mutation Reference".

CRITICAL RULES:
1. Study the "Pass/Fail Matrix" first. Each row is a case, each column is a candidate. Columns are chronological (seed, #1, #2, ...). Look for patterns: cases that are always F, cases that flip between P/F, cases that a specific mutation fixed or broke.
2. Study "Repair Effectiveness" next. If most failed cases show REPAIR NO-OP or REPAIR SAME-ERROR, the repair node is the bottleneck — rewrite its prompt BEFORE further changes to the initial-attempt node. Fixing the repair path is higher-leverage than re-rolling the first attempt.
3. Study "Failure Decomposition Hints". Infer 2-5 latent subproblems behind the failures (for example: exact output contract, public API shape, parser/validation semantics, algorithm/search logic, stateful edge rules, repair-step weakness).
4. Study the "Selected Candidate Lineage" and "Lineage Mutation Evidence". Use the exact mutation values, active overrides, and raw step-level outputs to diagnose what the task agent is doing wrong locally.
5. Check "Mutation Effects" to see what was tried. NEVER repeat a mutation type+target+value that consistently regressed. If a family previously improved but the recent deltas are flat/regressive, treat that family as saturated and try a different lever.
6. Choose ONE subproblem to target. Prefer the smallest tunable change with the best expected gain and lowest regression risk. Do not try to solve every failure cluster with one giant rewrite.
7. Preserve what works — compare the parent's column in the matrix with other columns. Avoid changes that risk flipping P→F on stable cases.
8. Node type constraints — rewrite_prompt and rewrite_system apply ONLY to prompt/agent nodes. rewrite_shell applies ONLY to tool nodes. Check "Available Nodes" for each node's type and valid mutations.
9. For rewrite_prompt, provide the FULL template in "new_template" including all {{ variable }} references. You MUST keep every {{ variable }} from the original so data binding works at runtime. You can change the instructional prose, add formatting guidance, reorder sections, etc.
10. Study "Failed cases" and raw execution evidence closely — the checker output and task-agent step outputs show WHY cases fail and whether the repair step helped, no-op'd, or merely shifted the symptom. The [REPAIR NO-OP] / [REPAIR SAME-ERROR] / [REPAIR SHIFTED] tags on each case tell you directly.
11. For structural changes, use `edit_graph` — write the complete modified graph as .scaffold DSL source. This gives you full language power: loops, conditionals, parallel blocks, new tool/verify nodes, complex data flow. For content-only changes, use `rewrite_prompt` / `rewrite_system` / `rewrite_shell` / `set_config`.

Strategy:
- Use the Pass/Fail Matrix to identify which cases to target: always-failing cases are high-value targets, flip-flopping cases suggest fragility
- Check "Repair Effectiveness" BEFORE choosing which node to rewrite. If the repair node is mostly NO-OP/SAME-ERROR, rewriting the repair node's prompt is almost always higher-leverage than another rewrite of the initial-attempt node
- Decompose residual failures into small tunable subproblems before choosing a mutation
- Pick one failure cluster or contract to target, not the whole task at once
- Use the selected lineage to understand local cause/effect: exact mutation values, active overrides, and raw step outputs matter more than broad guesses
- Read "Mutation Effects" for what has been tried and its score impact
- Compare attempts of the same mutation type from different parents — if a family repeatedly regressed or recently saturated, it's likely not the right next move
- Study the checker output and task-agent outputs in "Failed cases" / lineage evidence to understand specific failure patterns and attribution
- Match mutation scope to subproblem size:
  * use set_config for a narrow tunable behavior change
  * use rewrite_prompt/rewrite_system/rewrite_shell for a single-node contract mismatch
  * use edit_graph when you need to change the graph structure: add/remove steps, add loops/conditionals, insert new tool or verify nodes, reorder execution flow
- When repair behavior is mostly no-op or symptom-shifting, rewrite the repair node's prompt first — the repair step has access to the error output and should be able to fix the code, but a weak prompt causes it to repeat the same mistake
- Avoid full rewrites unless the evidence says the current design is fundamentally wrong

Return ONLY a single JSON object. No markdown, no explanation outside the JSON."#;

/// Condensed DSL syntax reference for edit_graph proposals.
/// Included in context only when edit_graph is available.
const DSL_SYNTAX_REFERENCE: &str = r#"## Scaffold DSL Syntax Reference (for edit_graph)

### Types
Primitives: `bool`, `int`, `float`, `string`, `bytes`, `any`
Containers: `list<T>`, `map<K,V>`, `option<T>`
Structs: `{ field1: Type1, field2: Type2 }`
Named: any previously declared `type Name = ...`

### Node Definition
```
node <name>: <kind> {
    in: <TypeExpr>
    out: <TypeExpr>
    <config fields...>
}
```
Kinds: `prompt` (LLM call), `tool` (shell command), `agent` (multi-turn LLM+tools), `verify` (LLM verification)

Config fields:
- `template`: `"text"` or `file("path")` — Jinja2 template (prompt/agent/verify)
- `system`: `"text"` or `file("path")` — system prompt (prompt/agent/verify)
- `model`: `"model-id"` (prompt/agent/verify)
- `temperature`: float (prompt/agent/verify)
- `max_tokens`: int (prompt/agent/verify)
- `max_turns`: int (agent only)
- `tools`: `[node1, node2]` (agent only)
- `shell`: `"command with {{var}}"` (tool only)
- `timeout`: int in seconds (tool only)
- `on_error`: `abort` or `retry(N)` (all)

### Graph Body Statements
```
graph <name> {
    in: <TypeExpr>
    out: <TypeExpr>
    <statements...>
}
```

**Step** — call a node or subgraph:
```
step <var> = <node>(<args>)
step x = solver(input)                     // positional (entire value)
step x = solver(task: input, ctx: memory)  // named arguments (builds struct)
step x = eval(code: attempt, dir: input.exercise_dir)  // field access
```

**Emit** — return value from graph:
```
emit <expr>
emit { field1: expr1, field2: expr2 }
```
Every execution path must emit a value matching the graph's `out` type.

**If/Else** — conditional:
```
if <condition> {
    <statements...>
} else {
    <statements...>
}
```

**Loop** — bounded iteration:
```
loop (max: <N>, while: <condition>) {
    <statements...>
    carry <var> = <expr>   // persist value for next iteration
}
```

**Parallel** — fan-out:
```
parallel (<var> in <collection>, reduce: <node>) {
    <statements...>
}
```

**Choose** — structural alternatives:
```
choose [alt1, alt2, alt3]
```

### Expressions
Operators: `+`, `-`, `*`, `/`, `==`, `!=`, `<`, `>`, `<=`, `>=`, `&&`, `||`, `!`
Field access: `input.field`, `step_name.field`
Index: `list[0]`, `map["key"]`
Builtins: `len(x)`, `contains(h,n)`, `str(x)`, `int(x)`, `float(x)`, `lower(s)`, `upper(s)`, `trim(s)`, `split(s,sep)`, `join(list,sep)`, `keys(m)`, `values(m)`, `json_parse(s)`

### Template Syntax (Jinja2)
Variables: `{{ field_name }}`, `{{ input.nested }}`
Loops: `{% for item in list %}...{% endfor %}`
Raw blocks: `{% raw %}...{% endraw %}`
"#;

/// Per-mutation-kind documentation. Keys are the kind strings used in proposals.
fn mutation_doc(kind: &str) -> Option<&'static str> {
    match kind {
        "edit_graph" => Some(
            r#"{"kind":"edit_graph","graph":"<complete graph as .scaffold source>","new_nodes":[<optional array of new/modified node definitions as .scaffold source>],"description":"<short label>","reasoning":"..."}
  - "graph" is REQUIRED: the complete modified graph as .scaffold DSL source (ALL steps, not just changed ones). Must use the same graph name, input type, and output type as the parent.
  - "new_nodes" is OPTIONAL: array of new or modified node definitions as .scaffold source strings. Only include nodes you are adding or changing — existing unchanged nodes are inherited.
  - "description" is REQUIRED: a short label (e.g. "add planning step", "wrap with retry loop") shown in the TUI.
  - The graph source is parsed and validated. Preserved steps (from topology.preserve) must still exist. All referenced nodes must be either in the original IR or in new_nodes.
  - Use this for ANY structural change: adding/removing steps, loops, conditionals, parallel blocks, new tool/verify nodes, complex rewiring."#,
        ),
        "insert_verify" => Some(
            r#"{"kind":"insert_verify","after_step":"<step>","verify_node":"<node>","max_retries":<n>,"reasoning":"..."} (verify_node MUST be a node of kind "verify", NOT prompt/tool/agent)"#,
        ),
        "wrap_retry" => Some(
            r#"{"kind":"wrap_retry","step":"<step>","verify_node":"<node>","max_retries":<n>,"reasoning":"..."} (verify_node MUST be a node of kind "verify", NOT prompt/tool/agent)"#,
        ),
        "insert_step" => Some(
            r#"{"kind":"insert_step","after_step":"<step>","new_step_name":"<name>","node":"<node>","reasoning":"..."} — the new step receives the output of after_step as its single positional argument. Best for nodes that accept a single input (e.g. tool nodes). For multi-input prompt nodes, use replace_component instead."#,
        ),
        "remove_step" => Some(
            r#"{"kind":"remove_step","step":"<step>","reasoning":"..."} — removes a step; downstream steps referencing it will break, so only remove steps whose output is not used by other steps."#,
        ),
        "replace_component" => Some(
            r#"{"kind":"replace_component","step":"<step>","new_node":"<node>","reasoning":"..."} — swaps which node a step invokes while keeping the same wiring. The new node must accept the same argument names."#,
        ),
        "set_config" => Some(
            r#"{"kind":"set_config","node":"<node>","field":"<field>","value":<json_value>,"reasoning":"..."}"#,
        ),
        "add_prompt_step" => Some(
            r#"{"kind":"add_prompt_step","after_step":"<step>","new_step_name":"<name>","template":"<jinja template>","system":"<optional system prompt>","model":"<optional model>","reasoning":"..."} — creates a new prompt node and inserts it as a step after after_step. The new step receives {{ input }} (the output of after_step) PLUS all graph input fields as named template variables. The new step's output is available by name to all downstream steps via DSL scoping (no rewiring). Use this to add planning, analysis, or review steps that need both the previous step's output and the original problem context."#,
        ),
        "rewrite_prompt" => Some(
            r#"{"kind":"rewrite_prompt","node":"<node>","new_template":"<full template text>","reasoning":"..."} — provide the COMPLETE template including both instructions and {{ variable }} references. You MUST preserve all {{ variable }} references from the original template so runtime data binding still works. You may rearrange, add context around them, or change the instructional prose freely."#,
        ),
        "rewrite_system" => Some(
            r#"{"kind":"rewrite_system","node":"<node>","new_system":"<full system prompt text>","reasoning":"..."}"#,
        ),
        "rewrite_shell" => Some(
            r#"{"kind":"rewrite_shell","node":"<tool_node>","new_shell":"<shell command template>","reasoning":"..."}"#,
        ),
        _ => None,
    }
}

/// Estimate the total context size (in chars) that would be sent to the meta-agent.
pub fn estimate_context_chars(
    parent: &Candidate,
    archive: &Archive,
    ir: &ScaffoldIR,
    objective: &ObjectiveIR,
    allowed_mutations: &[String],
) -> usize {
    // Building the full string is cheap (no I/O), so just measure it directly.
    let ctx = build_context(parent, archive, ir, objective, allowed_mutations);
    ctx.len() + SYSTEM_PROMPT.len()
}

/// Build the context string sent to the meta-agent LLM.
fn build_context(
    parent: &Candidate,
    archive: &Archive,
    ir: &ScaffoldIR,
    objective: &ObjectiveIR,
    allowed_mutations: &[String],
) -> String {
    let mut ctx = String::with_capacity(4096);

    // 1. Objective summary
    ctx.push_str("## Objective\n");
    ctx.push_str(&format!("Name: {}\n", objective.name));
    ctx.push_str(&format!("Graph: {}\n", objective.graph));
    ctx.push_str("Checkers:\n");
    for c in &objective.checkers {
        ctx.push_str(&format!(
            "- {}: {}\n",
            c.name,
            scaffold_ir::pretty::format_expr(&c.expr)
        ));
    }
    ctx.push_str(&format!(
        "Metrics: {}\n",
        objective
            .metrics
            .iter()
            .map(|m| format!("{} (from {})", m.name, m.checker))
            .collect::<Vec<_>>()
            .join(", ")
    ));
    // Show structural mutations from topology + content mutations (always available).
    // When meta-model is active, replace individual structural mutations with edit_graph.
    let has_tool_nodes = ir.nodes.iter().any(|n| n.kind == NodeKindIR::Tool);
    let has_prompt_nodes = ir
        .nodes
        .iter()
        .any(|n| matches!(n.kind, NodeKindIR::Prompt | NodeKindIR::Agent));
    let has_any_structural = allowed_mutations
        .iter()
        .any(|m| STRUCTURAL_MUTATIONS.contains(&m.as_str()));
    let mut all_available: Vec<String> = if has_any_structural {
        // When structural mutations are available, show edit_graph instead of individuals
        let mut v = vec!["edit_graph".to_string()];
        // Keep non-structural mutations (set_config)
        for m in allowed_mutations {
            if !STRUCTURAL_MUTATIONS.contains(&m.as_str()) {
                if !v.contains(m) {
                    v.push(m.clone());
                }
            }
        }
        v
    } else {
        allowed_mutations.to_vec()
    };
    if has_prompt_nodes {
        for content in &["rewrite_prompt", "rewrite_system"] {
            if !all_available.iter().any(|m| m == content) {
                all_available.push(content.to_string());
            }
        }
    }
    if has_tool_nodes {
        if !all_available.iter().any(|m| m == "rewrite_shell") {
            all_available.push("rewrite_shell".to_string());
        }
    }
    ctx.push_str(&format!(
        "Allowed mutations: {}\n",
        all_available.join(", ")
    ));
    if !has_tool_nodes {
        ctx.push_str("Note: No tool nodes exist — rewrite_shell is NOT available.\n");
    }

    // Mutation reference: only show docs for mutations actually available
    ctx.push_str("\n## Mutation Reference\n");
    for kind in &all_available {
        if let Some(doc) = mutation_doc(kind) {
            ctx.push_str(&format!("- {}\n", doc));
        }
    }

    // DSL syntax reference: include when edit_graph is available so the LLM
    // knows the exact .scaffold syntax for writing graph/node definitions.
    if all_available.iter().any(|m| m == "edit_graph") {
        ctx.push_str("\n");
        ctx.push_str(DSL_SYNTAX_REFERENCE);
    }

    // Tunables (set_config is limited to these)
    if !objective.tunables.is_empty() {
        ctx.push_str("\nTunable parameters (set_config MUST use one of these):\n");
        for t in &objective.tunables {
            let path = t.path.join(".");
            let values: Vec<String> = t.domain.iter().map(|e| format!("{:?}", e)).collect();
            ctx.push_str(&format!("- {} in [{}]\n", path, values.join(", ")));
        }
    } else {
        ctx.push_str("\nNo tunables declared — set_config is not available.\n");
    }
    ctx.push('\n');

    // 2. Parent graph (pretty-printed)
    ctx.push_str("## Parent Graph (selected candidate to mutate)\n");
    ctx.push_str(&format!("Score: {:.4}\n", parent.score.unwrap_or(0.0)));
    if !parent.mutations.is_empty() {
        ctx.push_str(&format!(
            "Mutations from seed: {}\n",
            parent
                .mutations
                .iter()
                .map(|m| m.short_label())
                .collect::<Vec<_>>()
                .join(" → ")
        ));
    }
    ctx.push_str("```\n");
    ctx.push_str(&scaffold_ir::pretty::pretty_print_graph(&parent.graph));
    ctx.push_str("```\n\n");
    push_candidate_overrides(&mut ctx, "Selected Candidate Active Overrides", parent);
    ctx.push('\n');

    let lineage = candidate_lineage(archive, parent);
    if !lineage.is_empty() {
        ctx.push_str("## Selected Candidate Lineage\n");
        ctx.push_str("Exact causal path from seed to the selected candidate. Use this to reason about what changed locally.\n\n");
        for cand in &lineage {
            ctx.push_str(&format!(
                "- {} score={:.4} parent={}\n",
                candidate_label(cand),
                cand.score.unwrap_or(0.0),
                cand.parent_id
                    .and_then(|pid| archive.candidates.iter().find(|p| p.id == pid))
                    .map(candidate_label)
                    .unwrap_or_else(|| "none".to_string())
            ));
            if is_seed_candidate(cand) {
                ctx.push_str("  exact mutation: seed\n");
            } else {
                for mutation in &cand.mutations {
                    ctx.push_str(&format!(
                        "  exact mutation: {}\n",
                        format_mutation_for_meta(mutation)
                    ));
                }
            }
            let active = format_overrides_inline(&cand.overrides);
            ctx.push_str(&format!("  active overrides: {}\n", active));
            if let Some(parent_cand) = cand
                .parent_id
                .and_then(|pid| archive.candidates.iter().find(|p| p.id == pid))
            {
                let delta = candidate_case_delta(parent_cand, cand);
                ctx.push_str(&format!(
                    "  delta vs parent: fixed [{}] | broken [{}]\n",
                    join_case_ids(&delta.fixed),
                    join_case_ids(&delta.broken),
                ));
            }
        }
        ctx.push('\n');
    }

    if let Some(best) = archive.best() {
        ctx.push_str("## Best Candidate Overall\n");
        ctx.push_str(&format!(
            "Candidate: {} | Score: {:.4}\n",
            candidate_label(best),
            best.score.unwrap_or(0.0)
        ));
        push_candidate_overrides(&mut ctx, "Best Candidate Active Overrides", best);
        ctx.push_str("Graph:\n```\n");
        ctx.push_str(&scaffold_ir::pretty::pretty_print_graph(&best.graph));
        ctx.push_str("```\n");
        push_candidate_node_state(&mut ctx, "Nodes in Best Candidate", best, ir);
        push_step_details(&mut ctx, "Steps in Best Candidate", &best.graph);
        ctx.push('\n');
    }

    // 3. Archive summary (top 10 candidates)
    ctx.push_str("## Archive (top candidates by score)\n");
    let ranked = archive.ranked();
    for (i, c) in ranked.iter().take(10).enumerate() {
        let mutations_str = if is_seed_candidate(c) {
            "seed".to_string()
        } else if c.mutations.is_empty() {
            "none".to_string()
        } else {
            c.mutations
                .iter()
                .map(|m| m.short_label())
                .collect::<Vec<_>>()
                .join(", ")
        };
        ctx.push_str(&format!(
            "{}. {} score={:.4} parent={} mutations=[{}] children={}\n",
            i + 1,
            candidate_label(c),
            c.score.unwrap_or(0.0),
            c.parent_id
                .and_then(|pid| archive.candidates.iter().find(|p| p.id == pid))
                .map(candidate_label)
                .unwrap_or_else(|| "none".to_string()),
            mutations_str,
            c.children_count,
        ));
    }
    ctx.push('\n');

    // 3.4 Pass/Fail Matrix — cases (rows) × candidates (columns)
    // Extremely compact view that lets the meta-agent see patterns across all candidates at once.
    {
        let mut all_case_ids: BTreeSet<String> = BTreeSet::new();
        for c in &archive.candidates {
            if c.score.is_none() {
                continue;
            }
            for id in &c.passed_case_ids {
                all_case_ids.insert(id.clone());
            }
            for cr in &c.case_results {
                if let Some(ref id) = cr.case_id {
                    all_case_ids.insert(id.clone());
                }
            }
        }
        let all_case_ids: Vec<String> = all_case_ids.into_iter().collect();

        let mut scored: Vec<&Candidate> = archive
            .candidates
            .iter()
            .filter(|c| c.score.is_some())
            .collect();
        scored.sort_by_key(|c| c.id);

        if !all_case_ids.is_empty() && scored.len() > 1 {
            // Limit to 12 columns to prevent excessive width
            let show: Vec<&Candidate> = scored.into_iter().take(12).collect();

            ctx.push_str("## Pass/Fail Matrix\n");
            let max_id_len = all_case_ids.iter().map(|id| id.len()).max().unwrap_or(10);
            let pad = max_id_len + 2;

            // Header
            ctx.push_str(&format!("{:pad$}", "Case", pad = pad));
            for c in &show {
                let label = candidate_label(c);
                ctx.push_str(&format!("{:>6}", label));
            }
            ctx.push('\n');

            // Rows
            for case_id in &all_case_ids {
                ctx.push_str(&format!("{:pad$}", case_id, pad = pad));
                for c in &show {
                    let passed = c.passed_case_ids.iter().any(|id| id == case_id);
                    ctx.push_str(&format!("{:>6}", if passed { "P" } else { "F" }));
                }
                ctx.push('\n');
            }
            ctx.push('\n');
        }
    }

    // 3.5 Mutation Effects — grouped by mutation type+target, showing lineage and score deltas.
    // Compact format: one summary line per attempt with score delta.
    // Case-level detail is in the Pass/Fail Matrix above.
    {
        let evaluated: Vec<&Candidate> = archive
            .candidates
            .iter()
            .filter(|c| c.score.is_some() && !c.mutations.is_empty())
            .collect();

        if !evaluated.is_empty() {
            // Group candidates by mutation label (e.g. "rewrite_prompt(solve_code)")
            let mut groups: Vec<(String, Vec<&Candidate>)> = Vec::new();
            for c in &evaluated {
                let label = c
                    .mutations
                    .last()
                    .map(|m| m.short_label())
                    .unwrap_or_default();
                if let Some(entry) = groups.iter_mut().find(|(l, _)| *l == label) {
                    entry.1.push(c);
                } else {
                    groups.push((label, vec![c]));
                }
            }

            ctx.push_str("## Mutation Effects (grouped by type)\n");
            ctx.push_str("Score impact of each mutation type. Compare with the Pass/Fail Matrix above for case-level detail.\n\n");

            for (label, candidates) in &groups {
                let trend = mutation_group_trend(candidates, archive);

                ctx.push_str(&format!(
                    "### {} — {} attempt{}, best={:.4}, {}\n",
                    label,
                    candidates.len(),
                    if candidates.len() > 1 { "s" } else { "" },
                    trend.best_score,
                    trend.verdict,
                ));
                ctx.push_str(&format!(
                    "  Recent deltas: [{}] | ever improved: {} | saturated: {}\n",
                    format_delta_list(&trend.recent_deltas),
                    if trend.ever_improved { "yes" } else { "no" },
                    if trend.saturated { "yes" } else { "no" },
                ));

                for c in candidates {
                    let score = c.score.unwrap_or(0.0);
                    let parent_cand = c
                        .parent_id
                        .and_then(|pid| archive.candidates.iter().find(|p| p.id == pid));
                    let parent_score = parent_cand.and_then(|p| p.score).unwrap_or(0.0);
                    let parent_label = match parent_cand {
                        Some(p) => candidate_label(p),
                        None => "?".to_string(),
                    };
                    let delta = score - parent_score;

                    // Flag likely runtime/template errors (0 passed out of N = total crash)
                    let crash_note = if score == 0.0 && c.total_cases > 0 && c.cases_passed == 0 {
                        " ⚠ LIKELY RUNTIME ERROR — mutation idea may be valid, template was broken"
                    } else {
                        ""
                    };
                    ctx.push_str(&format!(
                        "  {} ({}@{:.4} → {:.4}, Δ{}) passed {}/{}{}\n",
                        candidate_label(c),
                        parent_label,
                        parent_score,
                        score,
                        format_delta(delta),
                        c.cases_passed,
                        c.total_cases,
                        crash_note,
                    ));

                    // Show meta-agent reasoning if available (brief)
                    if let Some(ref reasoning) = c.meta_reasoning {
                        let short: String = reasoning.chars().take(200).collect();
                        ctx.push_str(&format!("    Reasoning: \"{}\"\n", short));
                    }
                }
                ctx.push('\n');
            }

            // Stagnation warning (inline)
            let ranked = archive.ranked();
            let best_score = ranked
                .first()
                .map(|c| c.score.unwrap_or(0.0))
                .unwrap_or(0.0);
            let stagnation_count = stagnation_count_after_best(archive);
            if stagnation_count >= 2 {
                ctx.push_str(&format!(
                    "⚠ STAGNATION: Best score ({:.4}) unchanged for {} candidates. Try a fundamentally different approach.\n\n",
                    best_score, stagnation_count
                ));
            }
        }
    }

    let failure_clusters = summarize_failure_clusters(parent);
    let repair_behavior = summarize_repair_behavior(parent, ir);
    let structural_pressure =
        structural_pressure_summary(archive, &failure_clusters, &repair_behavior);

    if !failure_clusters.is_empty() || parent.total_cases > 0 {
        ctx.push_str("## Failure Decomposition Hints\n");
        ctx.push_str(
            "Use these hints to decompose the task into smaller subproblems before choosing one mutation.\n",
        );

        if failure_clusters.is_empty() {
            ctx.push_str("- Dominant failure clusters: none available\n");
        } else {
            ctx.push_str("- Dominant failure clusters:\n");
            for cluster in failure_clusters.iter().take(4) {
                ctx.push_str(&format!(
                    "  {}: {} case(s) [{}] — {}\n",
                    cluster.kind.label(),
                    cluster.count,
                    join_case_ids(&cluster.cases),
                    cluster.kind.description(),
                ));
            }
        }

        ctx.push_str(&format!(
            "- Structural pressure: {}{}\n",
            structural_pressure.level,
            if structural_pressure.reasons.is_empty() {
                String::new()
            } else {
                format!(" — {}", structural_pressure.reasons.join("; "))
            }
        ));
        ctx.push('\n');
    }

    // Repair Effectiveness section — dedicated, prominent section
    {
        let non_trivial: Vec<&CaseRepairDetail> = repair_behavior
            .per_case
            .iter()
            .filter(|c| c.outcome != RepairOutcome::SingleAttempt)
            .collect();
        let total_with_repair = non_trivial.len();
        if total_with_repair > 0 {
            let no_ops = non_trivial
                .iter()
                .filter(|c| c.outcome == RepairOutcome::NoOp)
                .count();
            let same_err = non_trivial
                .iter()
                .filter(|c| c.outcome == RepairOutcome::SameError)
                .count();
            let shifted = non_trivial
                .iter()
                .filter(|c| c.outcome == RepairOutcome::ShiftedError)
                .count();

            ctx.push_str("## Repair Effectiveness\n");
            ctx.push_str("How well the repair step (second attempt) performed on failed cases.\n");
            ctx.push_str(&format!(
                "Summary: NO-OP={}/{} | SAME-ERROR={}/{} | SHIFTED-ERROR={}/{}\n",
                no_ops, total_with_repair,
                same_err, total_with_repair,
                shifted, total_with_repair,
            ));

            // Per-case breakdown
            for detail in &non_trivial {
                let node_str = detail
                    .repair_node
                    .as_deref()
                    .unwrap_or("?");
                ctx.push_str(&format!(
                    "  {} → {} (repair node: {})\n",
                    detail.case_id,
                    detail.outcome.label(),
                    node_str,
                ));
            }

            // Bottleneck diagnosis
            let ineffective = no_ops + same_err;
            if total_with_repair > 0 && ineffective * 2 >= total_with_repair {
                // Majority of repairs are no-op or same-error
                let bottleneck_node = non_trivial
                    .iter()
                    .filter(|c| c.outcome == RepairOutcome::NoOp || c.outcome == RepairOutcome::SameError)
                    .filter_map(|c| c.repair_node.as_deref())
                    .next()
                    .unwrap_or("unknown");
                ctx.push_str(&format!(
                    "\n⚠ BOTTLENECK: The repair node '{}' is ineffective in {}/{} failed cases — \
                     it reproduces the same code or error. Consider rewriting its prompt (rewrite_prompt on '{}') \
                     before further changes to the initial-attempt node.\n",
                    bottleneck_node, ineffective, total_with_repair, bottleneck_node,
                ));
            }
            ctx.push('\n');
        }
    }

    // 4. Parent's evaluation summary + failed cases
    if parent.total_cases > 0 {
        let failed = parent.total_cases - parent.cases_passed;
        ctx.push_str("## Parent Evaluation Summary\n");
        ctx.push_str(&format!(
            "Score: {:.4} | {}/{} cases PASSED, {}/{} FAILED\n",
            parent.score.unwrap_or(0.0),
            parent.cases_passed,
            parent.total_cases,
            failed,
            parent.total_cases,
        ));
        // Per-metric breakdown
        if !parent.metric_scores.is_empty() {
            ctx.push_str("Per-metric scores: ");
            let metrics: Vec<String> = parent
                .metric_scores
                .iter()
                .map(|(k, v)| format!("{}={:.4}", k, v))
                .collect();
            ctx.push_str(&metrics.join(", "));
            ctx.push('\n');
        }
        ctx.push('\n');

        if !parent.case_results.is_empty() {
            // Show passing cases first (just IDs)
            if !parent.passed_case_ids.is_empty() {
                ctx.push_str(&format!(
                    "Passing cases ({}): [{}]\n",
                    parent.passed_case_ids.len(),
                    parent.passed_case_ids.join(", "),
                ));
            }

            // Detailed failures: show model output + test errors for the first N,
            // then one-liner summaries for the rest.
            let detailed_limit = 5;

            ctx.push_str(&format!(
                "\nFailed cases ({}):\n",
                parent.case_results.len()
            ));
            for (i, case) in parent.case_results.iter().enumerate() {
                let repair_tag = case
                    .case_id
                    .as_deref()
                    .and_then(|cid| {
                        repair_behavior.per_case.iter().find(|r| r.case_id == cid)
                    })
                    .map(|r| r.outcome.label());
                if i < detailed_limit {
                    push_case_evidence(&mut ctx, parent, case, ir, 4000, repair_tag);
                } else {
                    if i == detailed_limit {
                        ctx.push_str(&format!(
                            "\n  Other failures ({}):\n",
                            parent.case_results.len() - detailed_limit,
                        ));
                    }
                    let id = case.case_id.as_deref().unwrap_or("?");
                    let failure_summary = case
                        .output_excerpt
                        .as_deref()
                        .or(case.raw_output.as_deref())
                        .map(parse_test_failure)
                        .unwrap_or_else(|| "no output".to_string());
                    if let Some(tag) = repair_tag {
                        ctx.push_str(&format!("    {}: [{}] {}\n", id, tag, failure_summary));
                    } else {
                        ctx.push_str(&format!("    {}: {}\n", id, failure_summary));
                    }
                }
            }
        }
        ctx.push('\n');
    }

    push_lineage_evidence(&mut ctx, archive, &lineage, ir);
    ctx.push('\n');

    // 5. Available nodes with DSL source + override annotations
    ctx.push_str("## Available Nodes\n");
    for node in &ir.nodes {
        // Show the node definition as .scaffold DSL source
        ctx.push_str("```\n");
        ctx.push_str(&scaffold_ir::pretty::pretty_print_node(node));
        ctx.push_str("```\n");

        // Show override annotations
        let override_template_key = format!("{}.template", node.name);
        if let Some(override_val) = parent.overrides.get(&override_template_key) {
            if let Some(content) = override_val.as_str() {
                let excerpt: String = content.chars().take(1500).collect();
                let truncated = if content.len() > 1500 { "..." } else { "" };
                ctx.push_str(&format!(
                    "  OVERRIDDEN template: \"{}{}\"\n",
                    excerpt.replace('\n', "\\n"),
                    truncated
                ));
            }
        }

        let override_system_key = format!("{}.system", node.name);
        if let Some(override_val) = parent.overrides.get(&override_system_key) {
            if let Some(content) = override_val.as_str() {
                let excerpt: String = content.chars().take(300).collect();
                let truncated = if content.len() > 300 { "..." } else { "" };
                ctx.push_str(&format!(
                    "  OVERRIDDEN system: \"{}{}\"\n",
                    excerpt.replace('\n', "\\n"),
                    truncated
                ));
            }
        }

        let override_shell_key = format!("{}.shell", node.name);
        if let Some(override_val) = parent.overrides.get(&override_shell_key) {
            if let Some(content) = override_val.as_str() {
                let excerpt: String = content.chars().take(500).collect();
                let truncated = if content.len() > 500 { "..." } else { "" };
                ctx.push_str(&format!(
                    "  OVERRIDDEN shell: \"{}{}\"\n",
                    excerpt.replace('\n', "\\n"),
                    truncated
                ));
            }
        }

        if let Some(active_cfg) = active_node_config_summary(parent, node) {
            ctx.push_str(&format!("  active config: {}\n", active_cfg));
        }

        ctx.push('\n');
    }

    // Show synthetic nodes from AddPromptStep mutations (stored in overrides as _node.*)
    let synthetic_vars = {
        let graph_input_fields = crate::mutations::resolve_graph_input_fields(&parent.graph, ir);
        let mut vars = vec!["input".to_string()];
        vars.extend(graph_input_fields);
        vars.join(", ")
    };
    for (key, _) in &parent.overrides {
        if let Some(name) = key.strip_prefix("_node.") {
            let template_key = format!("{}.template", name);
            let system_key = format!("{}.system", name);
            let model_key = format!("{}.model", name);
            ctx.push_str(&format!("- {} (prompt, SYNTHETIC) [mutations: rewrite_prompt, rewrite_system] [vars: {}]\n  output type: string", name, synthetic_vars));
            if let Some(tmpl) = parent.overrides.get(&template_key).and_then(|v| v.as_str()) {
                let excerpt: String = tmpl.chars().take(1500).collect();
                let truncated = if tmpl.len() > 1500 { "..." } else { "" };
                ctx.push_str(&format!(
                    "\n  template (OVERRIDDEN): \"{}{}\"\n",
                    excerpt.replace('\n', "\\n"),
                    truncated
                ));
            }
            if let Some(sys) = parent.overrides.get(&system_key).and_then(|v| v.as_str()) {
                let excerpt: String = sys.chars().take(300).collect();
                let truncated = if sys.len() > 300 { "..." } else { "" };
                ctx.push_str(&format!(
                    "  system (OVERRIDDEN): \"{}{}\"\n",
                    excerpt.replace('\n', "\\n"),
                    truncated
                ));
            }
            if let Some(model) = parent.overrides.get(&model_key).and_then(|v| v.as_str()) {
                ctx.push_str(&format!("  model: {}\n", model));
            }
            ctx.push('\n');
        }
    }

    // 5. Graph steps for reference
    push_step_details(&mut ctx, "Steps in Parent Graph", &parent.graph);

    ctx
}

#[derive(Debug, Default)]
struct CaseDelta {
    fixed: Vec<String>,
    broken: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum FailureClusterKind {
    ExactOutputContract,
    ApiShapeContract,
    ParserValidationRule,
    StatefulEdgeRules,
    RuntimeException,
    AlgorithmSearchLogic,
}

impl FailureClusterKind {
    fn label(self) -> &'static str {
        match self {
            Self::ExactOutputContract => "exact-output-contract",
            Self::ApiShapeContract => "api-shape-contract",
            Self::ParserValidationRule => "parser-validation-rule",
            Self::StatefulEdgeRules => "stateful-edge-rules",
            Self::RuntimeException => "runtime-exception",
            Self::AlgorithmSearchLogic => "algorithm-search-logic",
        }
    }

    fn description(self) -> &'static str {
        match self {
            Self::ExactOutputContract => {
                "exact strings, casing, whitespace, list formatting, or newline behavior"
            }
            Self::ApiShapeContract => {
                "missing exports/constants/classes/functions or wrong return/container/signature shape"
            }
            Self::ParserValidationRule => {
                "parser/validator ordering, invalid-input semantics, or exact error-rule behavior"
            }
            Self::StatefulEdgeRules => {
                "state machine, lifecycle, frame/turn, final-case, or off-by-one edge rules"
            }
            Self::RuntimeException => {
                "crashes such as recursion, indexing, key/type/attribute errors during execution"
            }
            Self::AlgorithmSearchLogic => {
                "wrong traversal/search/optimization/domain logic despite the program running"
            }
        }
    }
}

#[derive(Debug, Clone)]
struct FailureClusterSummary {
    kind: FailureClusterKind,
    count: usize,
    cases: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RepairOutcome {
    /// Repair step produced identical or near-identical output to the initial attempt.
    NoOp,
    /// Repair step changed the output but failed with the same error signature.
    SameError,
    /// Repair step changed the output and failed with a different error.
    ShiftedError,
    /// Only one attempt was executed (no repair path taken).
    SingleAttempt,
}

impl RepairOutcome {
    fn label(&self) -> &'static str {
        match self {
            RepairOutcome::NoOp => "REPAIR NO-OP",
            RepairOutcome::SameError => "REPAIR SAME-ERROR",
            RepairOutcome::ShiftedError => "REPAIR SHIFTED",
            RepairOutcome::SingleAttempt => "SINGLE-ATTEMPT",
        }
    }
}

#[derive(Debug, Clone)]
struct CaseRepairDetail {
    case_id: String,
    outcome: RepairOutcome,
    /// The node that produced the repair attempt (e.g. "fix_code").
    repair_node: Option<String>,
}

#[derive(Debug, Default)]
struct RepairBehaviorSummary {
    no_repair_signal: usize,
    no_op: usize,
    changed_unresolved: usize,
    /// Per-case repair analysis for failed cases.
    per_case: Vec<CaseRepairDetail>,
}

#[derive(Debug, Clone)]
struct MutationTrendSummary {
    best_score: f64,
    ever_improved: bool,
    recent_deltas: Vec<f64>,
    saturated: bool,
    verdict: String,
}

#[derive(Debug, Clone)]
struct StructuralPressureSummary {
    level: &'static str,
    reasons: Vec<String>,
}

#[derive(Debug)]
struct StepDetail {
    name: String,
    node: String,
    args: Vec<String>,
    scope: String,
}

fn is_seed_candidate(candidate: &Candidate) -> bool {
    candidate.parent_id.is_none()
        && candidate.mutations.is_empty()
        && candidate.overrides.is_empty()
}

fn candidate_label(candidate: &Candidate) -> String {
    if is_seed_candidate(candidate) {
        "seed".to_string()
    } else {
        format!("#{}", candidate.id)
    }
}

fn candidate_lineage<'a>(archive: &'a Archive, candidate: &'a Candidate) -> Vec<&'a Candidate> {
    let mut lineage = Vec::new();
    let mut current = Some(candidate);
    while let Some(cand) = current {
        lineage.push(cand);
        current = cand
            .parent_id
            .and_then(|pid| archive.candidates.iter().find(|p| p.id == pid));
    }
    lineage.reverse();
    lineage
}

fn candidate_case_delta(parent: &Candidate, child: &Candidate) -> CaseDelta {
    let parent_pass: BTreeSet<&str> = parent
        .passed_case_ids
        .iter()
        .map(|id| id.as_str())
        .collect();
    let child_pass: BTreeSet<&str> = child.passed_case_ids.iter().map(|id| id.as_str()).collect();

    let fixed = child_pass
        .difference(&parent_pass)
        .map(|id| (*id).to_string())
        .collect();
    let broken = parent_pass
        .difference(&child_pass)
        .map(|id| (*id).to_string())
        .collect();

    CaseDelta { fixed, broken }
}

fn join_case_ids(ids: &[String]) -> String {
    if ids.is_empty() {
        "none".to_string()
    } else {
        ids.join(", ")
    }
}

fn contains_any(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| haystack.contains(needle))
}

fn classify_failure_cluster(case: &crate::optimizer::CaseResult) -> FailureClusterKind {
    let summary = case
        .output_excerpt
        .as_deref()
        .or(case.raw_output.as_deref())
        .map(parse_test_failure)
        .unwrap_or_else(|| "unknown failure".to_string());
    let raw = case.raw_output.as_deref().unwrap_or("");
    let haystack = format!("{}\n{}", summary.to_lowercase(), raw.to_lowercase());

    if contains_any(
        &haystack,
        &[
            "cannot import name",
            "importerror",
            "attributeerror",
            "missing 1 required positional argument",
            "missing required positional argument",
            "unexpected keyword argument",
            "is not an instance of",
            "has no attribute",
            "__init__() missing",
            "takes 0 positional arguments",
        ],
    ) {
        return FailureClusterKind::ApiShapeContract;
    }

    if contains_any(
        &haystack,
        &[
            "not raised",
            "syntax error",
            "malformed",
            "must be in uppercase",
            "punctuations not permitted",
            "properties without delimiter",
            "unknown operation",
            "cannot start with",
            "graph item incomplete",
            "graph data malformed",
            "should be smaller than",
        ],
    ) {
        return FailureClusterKind::ParserValidationRule;
    }

    if contains_any(
        &haystack,
        &[
            "tenth frame",
            "bonus roll",
            "bonus with an open tenth frame",
            "game already has ten frames",
            "fill balls",
            "cannot roll",
            "last frame",
            "counted once",
        ],
    ) {
        return FailureClusterKind::StatefulEdgeRules;
    }

    if contains_any(
        &haystack,
        &[
            "indexerror",
            "keyerror",
            "recursionerror",
            "traceback",
            "error collecting",
            "typeerror:",
            "valueerror:",
        ],
    ) {
        return FailureClusterKind::RuntimeException;
    }

    if contains_any(
        &haystack,
        &[
            "lists differ",
            "multilineequal",
            "first differing element",
            "diff is",
            "trailing newline",
            "whitespace",
            "capitalization",
            "spacing",
        ],
    ) {
        return FailureClusterKind::ExactOutputContract;
    }

    FailureClusterKind::AlgorithmSearchLogic
}

fn summarize_failure_clusters(candidate: &Candidate) -> Vec<FailureClusterSummary> {
    let mut grouped: BTreeMap<FailureClusterKind, FailureClusterSummary> = BTreeMap::new();

    for case in &candidate.case_results {
        let kind = classify_failure_cluster(case);
        let entry = grouped
            .entry(kind)
            .or_insert_with(|| FailureClusterSummary {
                kind,
                count: 0,
                cases: Vec::new(),
            });
        entry.count += 1;
        if let Some(case_id) = case.case_id.as_ref() {
            if entry.cases.len() < 3 {
                entry.cases.push(case_id.clone());
            }
        }
    }

    let mut summaries: Vec<FailureClusterSummary> = grouped.into_values().collect();
    summaries.sort_by(|a, b| {
        b.count
            .cmp(&a.count)
            .then_with(|| a.kind.label().cmp(b.kind.label()))
    });
    summaries
}

fn summarize_repair_behavior(candidate: &Candidate, ir: &ScaffoldIR) -> RepairBehaviorSummary {
    let mut summary = RepairBehaviorSummary::default();

    for case in &candidate.case_results {
        let case_id = case
            .case_id
            .as_deref()
            .unwrap_or("?")
            .to_string();

        // Collect prompt/agent outputs with their step→node mapping
        let mut prompt_steps: Vec<(String, String, &str)> = Vec::new(); // (step, node, value)
        let mut non_agent_outputs: Vec<&str> = Vec::new();

        for (step_name, value) in &case.step_trace {
            let node_name = find_step_node(&candidate.graph.body, step_name)
                .unwrap_or_else(|| "?".to_string());
            let kind = node_kind_label(ir, &node_name, &candidate.overrides);
            if kind == "prompt" || kind == "agent" {
                prompt_steps.push((step_name.clone(), node_name, value.as_str()));
            } else {
                non_agent_outputs.push(value.as_str());
            }
        }

        let has_repair_signal = prompt_steps.len() >= 2 || non_agent_outputs.len() >= 2;
        if !has_repair_signal {
            summary.no_repair_signal += 1;
            summary.per_case.push(CaseRepairDetail {
                case_id,
                outcome: RepairOutcome::SingleAttempt,
                repair_node: None,
            });
            continue;
        }

        // The repair node is the last prompt/agent step (e.g. fix_code)
        let repair_node = prompt_steps.last().map(|(_, node, _)| node.clone());

        let prompt_same = prompt_steps.len() >= 2 && {
            let a = prompt_steps[prompt_steps.len() - 2].2;
            let b = prompt_steps[prompt_steps.len() - 1].2;
            a == b
        };

        let failure_sig_first = non_agent_outputs
            .get(non_agent_outputs.len().wrapping_sub(2))
            .map(|s| parse_test_failure(s));
        let failure_sig_last = non_agent_outputs.last().map(|s| parse_test_failure(s));
        let failure_same = non_agent_outputs.len() >= 2
            && failure_sig_first == failure_sig_last;

        let outcome = if prompt_same {
            RepairOutcome::NoOp
        } else if failure_same {
            RepairOutcome::SameError
        } else {
            RepairOutcome::ShiftedError
        };

        match outcome {
            RepairOutcome::NoOp | RepairOutcome::SameError => summary.no_op += 1,
            RepairOutcome::ShiftedError => summary.changed_unresolved += 1,
            RepairOutcome::SingleAttempt => summary.no_repair_signal += 1,
        }

        summary.per_case.push(CaseRepairDetail {
            case_id,
            outcome,
            repair_node,
        });
    }

    summary
}

fn format_delta(delta: f64) -> String {
    if delta > 0.0 {
        format!("+{:.4}", delta)
    } else {
        format!("{:.4}", delta)
    }
}

fn format_delta_list(deltas: &[f64]) -> String {
    if deltas.is_empty() {
        "none".to_string()
    } else {
        deltas
            .iter()
            .map(|delta| format_delta(*delta))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

fn mutation_group_trend(group: &[&Candidate], archive: &Archive) -> MutationTrendSummary {
    let mut ordered: Vec<&Candidate> = group.to_vec();
    ordered.sort_by_key(|candidate| candidate.id);

    let deltas: Vec<f64> = ordered
        .iter()
        .map(|candidate| {
            let parent_score = candidate
                .parent_id
                .and_then(|pid| archive.candidates.iter().find(|parent| parent.id == pid))
                .and_then(|parent| parent.score)
                .unwrap_or(0.0);
            candidate.score.unwrap_or(0.0) - parent_score
        })
        .collect();

    let best_score = ordered
        .iter()
        .filter_map(|candidate| candidate.score)
        .fold(f64::NEG_INFINITY, f64::max);
    let ever_improved = deltas.iter().any(|delta| *delta > 0.0);
    let recent_deltas = if deltas.len() > 3 {
        deltas[deltas.len() - 3..].to_vec()
    } else {
        deltas.clone()
    };
    let recent_tail = if deltas.len() > 2 {
        &deltas[deltas.len() - 2..]
    } else {
        &deltas[..]
    };
    let saturated = recent_tail.len() >= 2 && recent_tail.iter().all(|delta| *delta <= 0.0);
    let verdict = if recent_tail.iter().any(|delta| *delta > 0.0) {
        "currently improving".to_string()
    } else if ever_improved && saturated {
        "previously improved, now saturated".to_string()
    } else if ever_improved {
        "mixed results".to_string()
    } else {
        "no improvement".to_string()
    };

    MutationTrendSummary {
        best_score,
        ever_improved,
        recent_deltas,
        saturated,
        verdict,
    }
}

fn stagnation_count_after_best(archive: &Archive) -> usize {
    let ranked = archive.ranked();
    let best_id = ranked.first().map(|candidate| candidate.id).unwrap_or(0);
    archive
        .candidates
        .iter()
        .filter(|candidate| candidate.score.is_some() && candidate.id > best_id)
        .count()
}

fn structural_pressure_summary(
    archive: &Archive,
    clusters: &[FailureClusterSummary],
    repair: &RepairBehaviorSummary,
) -> StructuralPressureSummary {
    let mut points = 0usize;
    let mut reasons = Vec::new();

    if clusters.len() >= 3 {
        points += 1;
        reasons.push(format!(
            "failures span {} distinct clusters",
            clusters.len()
        ));
    }

    if repair.no_op >= 2 {
        points += 1;
        reasons.push(format!(
            "repair often no-ops or repeats the same failure ({} cases)",
            repair.no_op
        ));
    } else if repair.changed_unresolved >= 3 {
        points += 1;
        reasons.push(format!(
            "repair often changes output but stays unresolved ({} cases)",
            repair.changed_unresolved
        ));
    }

    let evaluated: Vec<&Candidate> = archive
        .candidates
        .iter()
        .filter(|candidate| candidate.score.is_some() && !candidate.mutations.is_empty())
        .collect();
    let mut saturated_families = 0usize;
    let mut groups: Vec<(String, Vec<&Candidate>)> = Vec::new();
    for candidate in evaluated {
        let label = candidate
            .mutations
            .last()
            .map(|mutation| mutation.short_label())
            .unwrap_or_default();
        if let Some(entry) = groups
            .iter_mut()
            .find(|(group_label, _)| *group_label == label)
        {
            entry.1.push(candidate);
        } else {
            groups.push((label, vec![candidate]));
        }
    }
    for (_label, group) in groups {
        if mutation_group_trend(&group, archive).saturated {
            saturated_families += 1;
        }
    }
    if saturated_families > 0 {
        points += 1;
        reasons.push(format!(
            "{} mutation famil{} recently saturated",
            saturated_families,
            if saturated_families == 1 { "y" } else { "ies" }
        ));
    }

    let stagnation = stagnation_count_after_best(archive);
    if stagnation >= 2 {
        points += 1;
        reasons.push(format!(
            "best score unchanged for {} later candidates",
            stagnation
        ));
    }

    let level = match points {
        0 | 1 => "LOW",
        2 => "MEDIUM",
        _ => "HIGH",
    };

    StructuralPressureSummary { level, reasons }
}

fn text_fingerprint(text: &str) -> String {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in text.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{:08x}", hash)
}

fn summarized_string_value(label: &str, text: &str) -> String {
    format!(
        "[{} fp={} len={}]",
        label,
        text_fingerprint(text),
        text.chars().count()
    )
}

fn override_value_label(key: &str) -> &'static str {
    if key == "template" || key.ends_with(".template") {
        "template override"
    } else if key == "system" || key.ends_with(".system") {
        "system override"
    } else if key == "shell" || key.ends_with(".shell") {
        "shell override"
    } else {
        "string override"
    }
}

fn should_summarize_override_string(key: &str, text: &str) -> bool {
    key == "template"
        || key == "system"
        || key == "shell"
        || key.ends_with(".template")
        || key.ends_with(".system")
        || key.ends_with(".shell")
        || text.chars().count() > 160
}

fn format_json_value_inline(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => format!("{:?}", s),
        _ => value.to_string(),
    }
}

fn format_override_value(key: &str, value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) if should_summarize_override_string(key, s) => {
            summarized_string_value(override_value_label(key), s)
        }
        _ => format_json_value_inline(value),
    }
}

fn format_mutation_for_meta(mutation: &Mutation) -> String {
    match mutation {
        Mutation::RewritePrompt { node, new_template } => format!(
            r#"{{"kind":"rewrite_prompt","node":"{}","new_template":"{}"}}"#,
            node,
            summarized_string_value("template override", new_template)
        ),
        Mutation::RewriteSystem { node, new_system } => format!(
            r#"{{"kind":"rewrite_system","node":"{}","new_system":"{}"}}"#,
            node,
            summarized_string_value("system override", new_system)
        ),
        Mutation::RewriteShell { node, new_shell } => format!(
            r#"{{"kind":"rewrite_shell","node":"{}","new_shell":"{}"}}"#,
            node,
            summarized_string_value("shell override", new_shell)
        ),
        Mutation::AddPromptStep {
            after_step,
            new_step_name,
            template,
            system,
            model,
        } => {
            let mut fields = vec![
                r#""kind":"add_prompt_step""#.to_string(),
                format!(r#""after_step":"{}""#, after_step),
                format!(r#""new_step_name":"{}""#, new_step_name),
                format!(
                    r#""template":"{}""#,
                    summarized_string_value("template override", template)
                ),
            ];
            if let Some(system) = system {
                fields.push(format!(
                    r#""system":"{}""#,
                    summarized_string_value("system override", system)
                ));
            }
            if let Some(model) = model {
                fields.push(format!(r#""model":"{}""#, model));
            }
            format!("{{{}}}", fields.join(","))
        }
        Mutation::SetConfig { node, field, value } => {
            let rendered = match value {
                serde_json::Value::String(text)
                    if should_summarize_override_string(field, text) =>
                {
                    format!(
                        r#""{}""#,
                        summarized_string_value(override_value_label(field), text)
                    )
                }
                _ => format_json_value_inline(value),
            };
            format!(
                r#"{{"kind":"set_config","node":"{}","field":"{}","value":{}}}"#,
                node, field, rendered
            )
        }
        _ => serde_json::to_string(mutation).unwrap_or_else(|_| mutation.short_label()),
    }
}

fn format_overrides_inline(
    overrides: &std::collections::HashMap<String, serde_json::Value>,
) -> String {
    if overrides.is_empty() {
        return "none".to_string();
    }

    let sorted: BTreeMap<String, serde_json::Value> = overrides
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();

    sorted
        .iter()
        .map(|(key, value)| format!("{}={}", key, format_override_value(key, value)))
        .collect::<Vec<_>>()
        .join(", ")
}

fn push_candidate_overrides(ctx: &mut String, title: &str, candidate: &Candidate) {
    ctx.push_str(&format!("## {}\n", title));
    if candidate.overrides.is_empty() {
        ctx.push_str("none\n");
        return;
    }

    let sorted: BTreeMap<String, serde_json::Value> = candidate
        .overrides
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    for (key, value) in sorted {
        ctx.push_str(&format!(
            "- {} = {}\n",
            key,
            format_override_value(&key, &value)
        ));
    }
}

fn push_candidate_node_state(
    ctx: &mut String,
    title: &str,
    candidate: &Candidate,
    ir: &ScaffoldIR,
) {
    ctx.push_str(&format!("## {}\n", title));
    for node in &ir.nodes {
        let kind_str = match node.kind {
            NodeKindIR::Prompt => "prompt",
            NodeKindIR::Tool => "tool",
            NodeKindIR::Agent => "agent",
            NodeKindIR::Verify => "verify",
        };
        let active = active_node_config_summary(candidate, node)
            .unwrap_or_else(|| "default config only".to_string());
        ctx.push_str(&format!(
            "- {} ({}) in={} out={} | {}\n",
            node.name,
            kind_str,
            scaffold_ir::pretty::format_type(&node.input),
            scaffold_ir::pretty::format_type(&node.output),
            active,
        ));
    }

    let mut synthetic: BTreeSet<String> = BTreeSet::new();
    for key in candidate.overrides.keys() {
        if let Some(name) = key.strip_prefix("_node.") {
            synthetic.insert(name.to_string());
        }
    }
    for name in synthetic {
        let active = synthetic_node_config_summary(candidate, &name);
        ctx.push_str(&format!(
            "- {} (prompt, synthetic) in=string out=string | {}\n",
            name, active
        ));
    }
}

fn active_node_config_summary(candidate: &Candidate, node: &NodeIR) -> Option<String> {
    let mut parts = Vec::new();
    let prefix = format!("{}.", node.name);

    let override_value = |field: &str| candidate.overrides.get(&format!("{}{}", prefix, field));

    let model = override_value("model")
        .and_then(|v| v.as_str().map(|s| s.to_string()))
        .or_else(|| node.config.model.clone());
    if let Some(model) = model {
        parts.push(format!("model={}", model));
    }

    let temperature = override_value("temperature")
        .map(format_json_value_inline)
        .or_else(|| node.config.temperature.map(|v| v.to_string()));
    if let Some(temp) = temperature {
        parts.push(format!("temperature={}", temp));
    }

    let max_tokens = override_value("max_tokens")
        .map(format_json_value_inline)
        .or_else(|| node.config.max_tokens.map(|v| v.to_string()));
    if let Some(v) = max_tokens {
        parts.push(format!("max_tokens={}", v));
    }

    let max_turns = override_value("max_turns")
        .map(format_json_value_inline)
        .or_else(|| node.config.max_turns.map(|v| v.to_string()));
    if let Some(v) = max_turns {
        parts.push(format!("max_turns={}", v));
    }

    let timeout = override_value("timeout")
        .map(format_json_value_inline)
        .or_else(|| node.config.timeout.map(|v| v.to_string()));
    if let Some(v) = timeout {
        parts.push(format!("timeout={}", v));
    }

    if override_value("template").is_some() {
        parts.push("template=override".to_string());
    } else if node.config.template.is_some() {
        parts.push("template=base".to_string());
    }

    if override_value("system").is_some() {
        parts.push("system=override".to_string());
    } else if node.config.system.is_some() {
        parts.push("system=base".to_string());
    }

    if override_value("shell").is_some() {
        parts.push("shell=override".to_string());
    } else if node.config.shell.is_some() {
        parts.push("shell=base".to_string());
    }

    if parts.is_empty() {
        None
    } else {
        Some(parts.join(", "))
    }
}

fn synthetic_node_config_summary(candidate: &Candidate, node_name: &str) -> String {
    let prefix = format!("{}.", node_name);
    let mut parts = Vec::new();

    for field in ["model", "temperature", "max_tokens", "max_turns", "timeout"] {
        if let Some(value) = candidate.overrides.get(&format!("{}{}", prefix, field)) {
            parts.push(format!("{}={}", field, format_json_value_inline(value)));
        }
    }
    if candidate
        .overrides
        .contains_key(&format!("{}template", prefix))
    {
        parts.push("template=override".to_string());
    }
    if candidate
        .overrides
        .contains_key(&format!("{}system", prefix))
    {
        parts.push("system=override".to_string());
    }
    if parts.is_empty() {
        "override metadata only".to_string()
    } else {
        parts.join(", ")
    }
}

fn push_step_details(ctx: &mut String, title: &str, graph: &GraphIR) {
    ctx.push_str(&format!("## {}\n", title));
    let mut details = Vec::new();
    collect_step_details(&graph.body, "root", &mut details);
    for detail in details {
        let args = if detail.args.is_empty() {
            "(no args)".to_string()
        } else {
            detail.args.join(", ")
        };
        ctx.push_str(&format!(
            "- step '{}' [{}] → node '{}' args: {}\n",
            detail.name, detail.scope, detail.node, args
        ));
    }
}

fn collect_step_details(stmts: &[GraphStmtIR], scope: &str, out: &mut Vec<StepDetail>) {
    for stmt in stmts {
        match stmt {
            GraphStmtIR::Step(step) => {
                let args = step.args.iter().map(format_step_arg_for_meta).collect();
                out.push(StepDetail {
                    name: step.name.clone(),
                    node: step.node.clone(),
                    args,
                    scope: scope.to_string(),
                });
            }
            GraphStmtIR::Loop(loop_stmt) => {
                let nested = nested_scope(scope, "loop");
                collect_step_details(&loop_stmt.body, &nested, out);
            }
            GraphStmtIR::If(if_stmt) => {
                let then_scope = nested_scope(scope, "if.then");
                collect_step_details(&if_stmt.then_body, &then_scope, out);
                let else_scope = nested_scope(scope, "if.else");
                collect_step_details(&if_stmt.else_body, &else_scope, out);
            }
            GraphStmtIR::Parallel(par) => {
                let nested = nested_scope(scope, &format!("parallel({})", par.var));
                collect_step_details(&par.body, &nested, out);
            }
            _ => {}
        }
    }
}

fn nested_scope(parent: &str, child: &str) -> String {
    if parent.is_empty() || parent == "root" {
        child.to_string()
    } else {
        format!("{} > {}", parent, child)
    }
}

fn format_step_arg_for_meta(arg: &StepArgIR) -> String {
    match arg {
        StepArgIR::Positional { value } => scaffold_ir::pretty::format_expr(value),
        StepArgIR::Named { name, value } => {
            format!("{}: {}", name, scaffold_ir::pretty::format_expr(value))
        }
    }
}

fn push_lineage_evidence(
    ctx: &mut String,
    archive: &Archive,
    lineage: &[&Candidate],
    ir: &ScaffoldIR,
) {
    if lineage.len() <= 1 {
        return;
    }

    ctx.push_str("## Lineage Mutation Evidence\n");
    ctx.push_str("Raw failure evidence for cases each lineage mutation fixed or broke. For fixed cases, evidence is taken from the parent failure that the child resolved.\n\n");

    for window in lineage.windows(2) {
        let parent = window[0];
        let child = window[1];
        let delta = candidate_case_delta(parent, child);
        ctx.push_str(&format!(
            "### {} from {}\n",
            candidate_label(child),
            candidate_label(parent)
        ));
        for mutation in &child.mutations {
            ctx.push_str(&format!(
                "Exact mutation: {}\n",
                format_mutation_for_meta(mutation)
            ));
        }
        ctx.push_str(&format!(
            "Delta: fixed [{}] | broken [{}]\n",
            join_case_ids(&delta.fixed),
            join_case_ids(&delta.broken),
        ));

        if delta.fixed.is_empty() && delta.broken.is_empty() {
            ctx.push_str("No case-level flips relative to parent.\n\n");
            continue;
        }

        for case_id in delta.fixed.iter().take(3) {
            if let Some(case) = find_case_result(parent, case_id) {
                ctx.push_str(&format!(
                    "Pre-fix failure evidence from {}:\n",
                    candidate_label(parent)
                ));
                ctx.push_str(&format!(
                    "Child outcome: F -> P (resolved in {})\n",
                    candidate_label(child)
                ));
                push_case_evidence(ctx, parent, case, ir, 2500, None);
            }
        }

        for case_id in delta.broken.iter().take(3) {
            if let Some(case) = find_case_result(child, case_id) {
                ctx.push_str(&format!(
                    "Regression evidence from {}:\n",
                    candidate_label(child)
                ));
                ctx.push_str(&format!(
                    "Child outcome: P -> F (regressed in {})\n",
                    candidate_label(child)
                ));
                push_case_evidence(ctx, child, case, ir, 2500, None);
            }
        }

        if let Some(best) = archive.best() {
            if best.id == child.id {
                ctx.push_str("This mutation is currently on the best-known path.\n");
            }
        }
        ctx.push('\n');
    }
}

fn find_case_result<'a>(
    candidate: &'a Candidate,
    case_id: &str,
) -> Option<&'a crate::optimizer::CaseResult> {
    candidate
        .case_results
        .iter()
        .find(|case| case.case_id.as_deref() == Some(case_id))
}

fn push_case_evidence(
    ctx: &mut String,
    candidate: &Candidate,
    case: &crate::optimizer::CaseResult,
    ir: &ScaffoldIR,
    max_chars: usize,
    repair_tag: Option<&str>,
) {
    let id = case.case_id.as_deref().unwrap_or("?");
    let failure_summary = case
        .output_excerpt
        .as_deref()
        .or(case.raw_output.as_deref())
        .map(parse_test_failure)
        .unwrap_or_else(|| "no output".to_string());
    if let Some(tag) = repair_tag {
        ctx.push_str(&format!("  [{}] [{}] {}\n", id, tag, failure_summary));
    } else {
        ctx.push_str(&format!("  [{}] {}\n", id, failure_summary));
    }

    if !case.step_trace.is_empty() {
        let executed: Vec<String> = case
            .step_trace
            .iter()
            .map(|(step_name, _)| {
                let node_name = find_step_node(&candidate.graph.body, step_name)
                    .unwrap_or_else(|| "?".to_string());
                format!(
                    "{} -> {} ({})",
                    step_name,
                    node_name,
                    node_kind_label(ir, &node_name, &candidate.overrides),
                )
            })
            .collect();
        ctx.push_str(&format!("    executed steps: {}\n", executed.join(" -> ")));

        let prompt_outputs: Vec<(String, String, String)> = case
            .step_trace
            .iter()
            .filter_map(|(step_name, value)| {
                let node_name = find_step_node(&candidate.graph.body, step_name)
                    .unwrap_or_else(|| "?".to_string());
                let kind = node_kind_label(ir, &node_name, &candidate.overrides);
                if kind == "prompt" || kind == "agent" {
                    Some((step_name.clone(), node_name, value.clone()))
                } else {
                    None
                }
            })
            .collect();

        if !prompt_outputs.is_empty() {
            ctx.push_str("    task-agent outputs:\n");
            for (step_name, node_name, value) in prompt_outputs {
                ctx.push_str(&format!(
                    "    [{} -> {}]\n```text\n{}\n```\n",
                    step_name,
                    node_name,
                    clip_for_block(&value, max_chars)
                ));
            }
        } else if let Some(response) = case.model_response.as_deref() {
            ctx.push_str(&format!(
                "    task-agent output:\n```text\n{}\n```\n",
                clip_for_block(response, max_chars)
            ));
        }

        let non_agent_steps: Vec<(String, String, String)> = case
            .step_trace
            .iter()
            .filter_map(|(step_name, value)| {
                let node_name = find_step_node(&candidate.graph.body, step_name)
                    .unwrap_or_else(|| "?".to_string());
                let kind = node_kind_label(ir, &node_name, &candidate.overrides);
                if kind == "prompt" || kind == "agent" {
                    None
                } else {
                    Some((step_name.clone(), node_name, value.clone()))
                }
            })
            .collect();

        if !non_agent_steps.is_empty() {
            ctx.push_str("    raw non-agent execution log:\n```text\n");
            for (step_name, node_name, value) in non_agent_steps {
                ctx.push_str(&format!(
                    "[{} -> {} ({})]\n{}\n\n",
                    step_name,
                    node_name,
                    node_kind_label(ir, &node_name, &candidate.overrides),
                    clip_for_block(&value, max_chars)
                ));
            }
            ctx.push_str("```\n");
        }
    }

    if let Some(raw_output) = case.raw_output.as_deref() {
        ctx.push_str(&format!(
            "    raw checker output:\n```text\n{}\n```\n",
            clip_for_block(raw_output, max_chars)
        ));
    }
}

fn clip_for_block(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        text.to_string()
    } else {
        let clipped: String = text.chars().take(max_chars).collect();
        format!("{}...", clipped)
    }
}

fn node_kind_label(
    ir: &ScaffoldIR,
    node_name: &str,
    overrides: &std::collections::HashMap<String, serde_json::Value>,
) -> &'static str {
    if overrides.contains_key(&format!("_node.{}", node_name)) {
        return "prompt";
    }

    match ir
        .nodes
        .iter()
        .find(|node| node.name == node_name)
        .map(|node| node.kind)
    {
        Some(NodeKindIR::Prompt) => "prompt",
        Some(NodeKindIR::Tool) => "tool",
        Some(NodeKindIR::Agent) => "agent",
        Some(NodeKindIR::Verify) => "verify",
        None => "unknown",
    }
}

/// Find the node name for a given step (public for optimizer use).
pub fn find_step_node_pub(stmts: &[GraphStmtIR], step_name: &str) -> Option<String> {
    find_step_node(stmts, step_name)
}

/// Find the node name for a given step.
fn find_step_node(stmts: &[GraphStmtIR], step_name: &str) -> Option<String> {
    for stmt in stmts {
        match stmt {
            GraphStmtIR::Step(s) if s.name == step_name => return Some(s.node.clone()),
            GraphStmtIR::Loop(l) => {
                if let Some(n) = find_step_node(&l.body, step_name) {
                    return Some(n);
                }
            }
            GraphStmtIR::If(i) => {
                if let Some(n) = find_step_node(&i.then_body, step_name) {
                    return Some(n);
                }
                if let Some(n) = find_step_node(&i.else_body, step_name) {
                    return Some(n);
                }
            }
            GraphStmtIR::Parallel(p) => {
                if let Some(n) = find_step_node(&p.body, step_name) {
                    return Some(n);
                }
            }
            _ => {}
        }
    }
    None
}

/// Collect the named argument names from a step in the graph.
fn find_step_named_args(stmts: &[GraphStmtIR], step_name: &str) -> Vec<String> {
    for stmt in stmts {
        match stmt {
            GraphStmtIR::Step(s) if s.name == step_name => {
                return s
                    .args
                    .iter()
                    .filter_map(|a| {
                        if let StepArgIR::Named { name, .. } = a {
                            Some(name.clone())
                        } else {
                            None
                        }
                    })
                    .collect();
            }
            GraphStmtIR::Loop(l) => {
                let r = find_step_named_args(&l.body, step_name);
                if !r.is_empty() {
                    return r;
                }
            }
            GraphStmtIR::If(i) => {
                let r = find_step_named_args(&i.then_body, step_name);
                if !r.is_empty() {
                    return r;
                }
                let r = find_step_named_args(&i.else_body, step_name);
                if !r.is_empty() {
                    return r;
                }
            }
            GraphStmtIR::Parallel(p) => {
                let r = find_step_named_args(&p.body, step_name);
                if !r.is_empty() {
                    return r;
                }
            }
            _ => {}
        }
    }
    Vec::new()
}

/// Parse the LLM response into a MutationProposal.
fn parse_proposal(
    response: &str,
    ir: &ScaffoldIR,
    parent_graph: &GraphIR,
    objective: &ObjectiveIR,
    allowed_mutations: &[String],
) -> Result<MutationProposal> {
    let json: serde_json::Value = llm::parse_json_with_repairs(response)
        .map_err(|e| Error::Runtime(format!("meta-agent returned invalid JSON: {}", e)))?;

    let kind = json
        .get("kind")
        .and_then(|v| v.as_str())
        .ok_or_else(|| Error::Runtime("meta-agent response missing 'kind' field".into()))?;

    let reasoning = json
        .get("reasoning")
        .and_then(|v| v.as_str())
        .unwrap_or("no reasoning provided")
        .to_string();

    // Content rewrites (prompt/system/shell) are always allowed — they don't change
    // graph structure. Structural mutations must be in the topology's allowed list.
    // edit_graph is allowed when any structural mutation is in the allowed list (it subsumes all).
    let content_mutations = ["rewrite_prompt", "rewrite_system", "rewrite_shell"];
    let has_any_structural = allowed_mutations
        .iter()
        .any(|m| STRUCTURAL_MUTATIONS.contains(&m.as_str()));
    if !content_mutations.contains(&kind)
        && !(kind == "edit_graph" && has_any_structural)
        && !allowed_mutations.contains(&kind.to_string())
    {
        return Err(Error::Runtime(format!(
            "meta-agent proposed '{}' which is not in allowed mutations: {:?}",
            kind, allowed_mutations
        )));
    }

    let mutation = match kind {
        "insert_verify" => {
            let after_step = require_str(&json, "after_step")?;
            let verify_node = require_str(&json, "verify_node")?;
            let max_retries = json
                .get("max_retries")
                .and_then(|v| v.as_u64())
                .unwrap_or(2) as u32;
            validate_step_exists(parent_graph, &after_step)?;
            validate_node_exists(ir, &verify_node, Some(NodeKindIR::Verify))?;
            Mutation::InsertVerify {
                after_step,
                verify_node,
                max_retries,
            }
        }
        "wrap_retry" => {
            let step = require_str(&json, "step")?;
            let verify_node = require_str(&json, "verify_node")?;
            let max_retries = json
                .get("max_retries")
                .and_then(|v| v.as_u64())
                .unwrap_or(2) as u32;
            validate_step_exists(parent_graph, &step)?;
            validate_node_exists(ir, &verify_node, Some(NodeKindIR::Verify))?;
            Mutation::WrapRetry {
                step,
                verify_node,
                max_retries,
            }
        }
        "insert_step" => {
            let after_step = require_str(&json, "after_step")?;
            let new_step_name = require_str(&json, "new_step_name")?;
            let node = require_str(&json, "node")?;
            validate_step_exists(parent_graph, &after_step)?;
            validate_node_exists(ir, &node, None)?;
            Mutation::InsertStep {
                after_step,
                new_step_name,
                node,
            }
        }
        "remove_step" => {
            let step = require_str(&json, "step")?;
            validate_step_exists(parent_graph, &step)?;
            Mutation::RemoveStep { step }
        }
        "replace_component" => {
            let step = require_str(&json, "step")?;
            let new_node = require_str(&json, "new_node")?;
            validate_step_exists(parent_graph, &step)?;
            validate_node_exists(ir, &new_node, None)?;
            // Validate input compatibility: the new node's required input fields
            // must be satisfiable by the step's existing named arguments.
            if let Some(node_ir) = ir.nodes.iter().find(|n| n.name == new_node) {
                let required_fields = resolve_input_fields(ir, node_ir);
                if !required_fields.is_empty() {
                    let step_args = find_step_named_args(&parent_graph.body, &step);
                    let missing: Vec<&String> = required_fields
                        .iter()
                        .filter(|f| !step_args.contains(f))
                        .collect();
                    if !missing.is_empty() {
                        return Err(Error::Runtime(format!(
                            "replace_component: node '{}' requires input fields [{}] but step '{}' only provides [{}]. Missing: [{}]",
                            new_node,
                            required_fields.join(", "),
                            step,
                            step_args.join(", "),
                            missing.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", "),
                        )));
                    }
                }
            }
            Mutation::ReplaceComponent { step, new_node }
        }
        "set_config" => {
            let raw_node = require_str(&json, "node")?;
            let raw_field = json
                .get("field")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            let value = json
                .get("value")
                .cloned()
                .unwrap_or(serde_json::Value::Null);

            // The meta-agent sometimes sends "node": "answer_memory.temperature"
            // instead of "node": "answer_memory", "field": "temperature".
            // Handle both formats gracefully.
            let (node, field) = if let Some(f) = raw_field {
                (raw_node, f)
            } else if let Some(dot_pos) = raw_node.find('.') {
                (
                    raw_node[..dot_pos].to_string(),
                    raw_node[dot_pos + 1..].to_string(),
                )
            } else {
                return Err(Error::Runtime(
                    "meta-agent response missing 'field' for set_config".into(),
                ));
            };

            validate_node_exists(ir, &node, None)?;
            // Validate that this node.field is a declared tunable
            let path = vec![node.clone(), field.clone()];
            if !objective.tunables.iter().any(|t| t.path == path) {
                return Err(Error::Runtime(format!(
                    "meta-agent proposed set_config for '{}.{}' which is not a declared tunable",
                    node, field
                )));
            }
            Mutation::SetConfig { node, field, value }
        }
        "add_prompt_step" => {
            let after_step = require_str(&json, "after_step")?;
            let new_step_name = require_str(&json, "new_step_name")?;
            let template = require_str(&json, "template")?;
            let system = json
                .get("system")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            let model = json
                .get("model")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            validate_step_exists(parent_graph, &after_step)?;
            // Check new_step_name doesn't collide with existing steps
            let existing_steps = crate::mutations::collect_step_names(&parent_graph.body);
            if existing_steps.iter().any(|s| s == &new_step_name) {
                return Err(Error::Runtime(format!(
                    "add_prompt_step: step name '{}' already exists in graph",
                    new_step_name
                )));
            }
            // Also check it doesn't collide with existing node names
            if ir.nodes.iter().any(|n| n.name == new_step_name) {
                return Err(Error::Runtime(format!(
                    "add_prompt_step: name '{}' collides with an existing node",
                    new_step_name
                )));
            }
            validate_template_syntax(&template, "template")?;
            // Validate template variables: allowed vars are `input` plus graph input fields.
            let graph_input_fields = crate::mutations::resolve_graph_input_fields(parent_graph, ir);
            let mut allowed_vars: Vec<String> = vec!["input".to_string()];
            allowed_vars.extend(graph_input_fields.iter().cloned());
            validate_add_prompt_step_template_vars(&template, &allowed_vars)?;
            Mutation::AddPromptStep {
                after_step,
                new_step_name,
                template,
                system,
                model,
            }
        }
        "rewrite_prompt" => {
            let node = require_str(&json, "node")?;
            // Accept "new_template" (preferred) or "new_instructions" (legacy — treated as full template)
            let new_template = json
                .get("new_template")
                .and_then(|v| v.as_str())
                .or_else(|| json.get("new_instructions").and_then(|v| v.as_str()))
                .map(|s| s.to_string())
                .ok_or_else(|| {
                    Error::Runtime(
                        "meta-agent response missing 'new_template' field for rewrite_prompt"
                            .into(),
                    )
                })?;
            // Also accept synthetic nodes from AddPromptStep
            validate_node_exists_with_graph(ir, &node, None, Some(parent_graph))?;
            validate_template_syntax(&new_template, "new_template")?;
            validate_template_variables(&new_template, ir, &node, "new_template", Some(parent_graph))?;
            Mutation::RewritePrompt { node, new_template }
        }
        "rewrite_system" => {
            let node = require_str(&json, "node")?;
            let new_system = require_str(&json, "new_system")?;
            // Also accept synthetic nodes from AddPromptStep
            validate_node_exists_with_graph(ir, &node, None, Some(parent_graph))?;
            validate_template_syntax(&new_system, "new_system")?;
            validate_template_variables(&new_system, ir, &node, "new_system", Some(parent_graph))?;
            Mutation::RewriteSystem { node, new_system }
        }
        "rewrite_shell" => {
            let node = require_str(&json, "node")?;
            let new_shell = require_str(&json, "new_shell")?;
            validate_node_exists(ir, &node, Some(NodeKindIR::Tool))?;
            Mutation::RewriteShell { node, new_shell }
        }
        "edit_graph" => {
            let graph_source = require_str(&json, "graph")?;
            let description = require_str(&json, "description")?;
            let new_node_sources: Vec<String> = json
                .get("new_nodes")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default();
            let (new_graph, new_nodes) =
                parse_and_validate_graph_edit(&graph_source, &new_node_sources, parent_graph, ir, objective)?;
            Mutation::EditGraph {
                new_graph,
                new_nodes,
                description,
            }
        }
        other => {
            return Err(Error::Runtime(format!(
                "meta-agent proposed unknown mutation kind: '{}'",
                other
            )));
        }
    };

    Ok(MutationProposal {
        mutation,
        reasoning,
        context_chars: 0,
        system_chars: 0,
    })
}

fn require_str(json: &serde_json::Value, field: &str) -> Result<String> {
    json.get(field)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| Error::Runtime(format!("meta-agent response missing '{}' field", field)))
}

/// Parse and validate an edit_graph proposal.
///
/// 1. Build combined source: type defs + new node sources + graph source
/// 2. Parse + lower via scaffold_ir::parse_and_lower()
/// 3. Extract graph by name — reject if graph name doesn't match parent's
/// 4. Validate input/output types match parent's
/// 5. Validate preserved steps exist in new graph
/// 6. Validate node references — every step references a known node
/// 7. Check topology constraints (max_nodes, max_depth)
/// 8. Extract new/modified nodes from lowered IR
fn parse_and_validate_graph_edit(
    graph_source: &str,
    new_node_sources: &[String],
    parent_graph: &GraphIR,
    ir: &ScaffoldIR,
    objective: &ObjectiveIR,
) -> Result<(GraphIR, Vec<NodeIR>)> {
    // 1. Build combined source for parsing.
    // Include type definitions so the parser can resolve named types.
    let mut combined = String::new();
    for td in &ir.types {
        combined.push_str(&format!(
            "type {} = {}\n",
            td.name,
            scaffold_ir::pretty::format_type(&td.ty)
        ));
    }
    // Include new node sources
    for ns in new_node_sources {
        combined.push_str(ns);
        combined.push('\n');
    }
    // Include graph source
    combined.push_str(graph_source);
    combined.push('\n');

    // 2. Parse + lower
    let lowered = scaffold_ir::parse_and_lower(&combined).map_err(|e| {
        Error::Runtime(format!("edit_graph: failed to parse .scaffold source: {}", e))
    })?;

    // 3. Extract graph by name
    let new_graph = lowered
        .graphs
        .iter()
        .find(|g| g.name == parent_graph.name)
        .ok_or_else(|| {
            Error::Runtime(format!(
                "edit_graph: graph '{}' not found in source (found: [{}])",
                parent_graph.name,
                lowered.graphs.iter().map(|g| g.name.as_str()).collect::<Vec<_>>().join(", ")
            ))
        })?
        .clone();

    // 4. Validate input/output types match parent's
    let parent_input = scaffold_ir::pretty::format_type(&parent_graph.input);
    let new_input = scaffold_ir::pretty::format_type(&new_graph.input);
    if parent_input != new_input {
        return Err(Error::Runtime(format!(
            "edit_graph: input type mismatch — parent has '{}', new graph has '{}'",
            parent_input, new_input
        )));
    }
    let parent_output = scaffold_ir::pretty::format_type(&parent_graph.output);
    let new_output = scaffold_ir::pretty::format_type(&new_graph.output);
    if parent_output != new_output {
        return Err(Error::Runtime(format!(
            "edit_graph: output type mismatch — parent has '{}', new graph has '{}'",
            parent_output, new_output
        )));
    }

    // 5. Validate preserved steps exist
    if let Some(ref topo) = objective.topology {
        for preserved in &topo.preserve {
            if !crate::mutations::step_exists_pub(&new_graph.body, preserved) {
                return Err(Error::Runtime(format!(
                    "edit_graph: preserved step '{}' missing from new graph",
                    preserved
                )));
            }
        }
    }

    // 6. Validate node references — every step must reference a known node
    let known_node_names: std::collections::HashSet<String> = ir
        .nodes
        .iter()
        .map(|n| n.name.clone())
        .chain(lowered.nodes.iter().map(|n| n.name.clone()))
        .chain(ir.graphs.iter().map(|g| g.name.clone()))
        .collect();
    validate_node_refs_in_stmts(&new_graph.body, &known_node_names)?;

    // 7. Check topology constraints
    if let Some(ref topo) = objective.topology {
        let violations = crate::mutations::check_constraints(&new_graph, topo);
        if !violations.is_empty() {
            return Err(Error::Runtime(format!(
                "edit_graph: topology constraint violations: {}",
                violations.join("; ")
            )));
        }
    }

    // 8. Extract new/modified nodes from lowered IR
    let new_nodes: Vec<NodeIR> = lowered
        .nodes
        .into_iter()
        .filter(|n| {
            // Only include nodes that are NOT in the original IR, or that have changed config
            !ir.nodes.iter().any(|orig| orig.name == n.name)
        })
        .collect();

    Ok((new_graph, new_nodes))
}

/// Recursively validate that all steps reference known nodes.
fn validate_node_refs_in_stmts(
    stmts: &[GraphStmtIR],
    known: &std::collections::HashSet<String>,
) -> Result<()> {
    for stmt in stmts {
        match stmt {
            GraphStmtIR::Step(s) => {
                if !known.contains(&s.node) {
                    return Err(Error::Runtime(format!(
                        "edit_graph: step '{}' references unknown node '{}'",
                        s.name, s.node
                    )));
                }
            }
            GraphStmtIR::Loop(l) => validate_node_refs_in_stmts(&l.body, known)?,
            GraphStmtIR::If(i) => {
                validate_node_refs_in_stmts(&i.then_body, known)?;
                validate_node_refs_in_stmts(&i.else_body, known)?;
            }
            GraphStmtIR::Parallel(p) => validate_node_refs_in_stmts(&p.body, known)?,
            GraphStmtIR::Choose(c) => {
                for alt in &c.alternatives {
                    if !known.contains(alt) {
                        return Err(Error::Runtime(format!(
                            "edit_graph: choose references unknown node/graph '{}'",
                            alt
                        )));
                    }
                }
            }
            _ => {}
        }
    }
    Ok(())
}

fn validate_step_exists(graph: &GraphIR, step_name: &str) -> Result<()> {
    let steps = crate::mutations::collect_step_names(&graph.body);
    if !steps.iter().any(|s| s == step_name) {
        return Err(Error::Runtime(format!(
            "meta-agent referenced non-existent step '{}'",
            step_name
        )));
    }
    Ok(())
}

/// Validate that a template string compiles as valid Jinja2.
/// Catches syntax errors before running a full evaluation that would score 0.
fn validate_template_syntax(template: &str, field_name: &str) -> Result<()> {
    let env = minijinja::Environment::new();
    if let Err(e) = env.render_str(template, minijinja::context!()) {
        // Distinguish syntax errors from missing variables:
        // Syntax errors are fatal; missing variables are expected (they'll be filled at runtime).
        let err_str = e.to_string();
        if err_str.contains("syntax error")
            || err_str.contains("unexpected end")
            || err_str.contains("expected")
        {
            return Err(Error::Runtime(format!(
                "meta-agent proposed {} with invalid template syntax: {}",
                field_name, err_str
            )));
        }
        // Missing variable errors are OK — the template is syntactically valid
    }
    Ok(())
}



/// Resolve the input field names for a node, following named type references.
fn resolve_input_fields(ir: &ScaffoldIR, node: &NodeIR) -> Vec<String> {
    resolve_type_fields(ir, &node.input)
}

fn resolve_type_fields(ir: &ScaffoldIR, ty: &scaffold_ir::ir::TypeIR) -> Vec<String> {
    match ty {
        scaffold_ir::ir::TypeIR::Struct { fields } => {
            fields.iter().map(|f| f.name.clone()).collect()
        }
        scaffold_ir::ir::TypeIR::Named { name } => {
            // Look up the named type in the IR's type definitions
            if let Some(td) = ir.types.iter().find(|t| t.name == *name) {
                resolve_type_fields(ir, &td.ty)
            } else {
                Vec::new()
            }
        }
        _ => Vec::new(),
    }
}

/// Validate that a rewritten template only references variables available in the node's input.
/// Uses minijinja's undeclared_variables to find references, then checks against known fields.
/// For synthetic nodes (from add_prompt_step), the known fields are `input` + graph input fields.
fn validate_template_variables(
    template: &str,
    ir: &ScaffoldIR,
    node_name: &str,
    field_name: &str,
    parent_graph: Option<&GraphIR>,
) -> Result<()> {
    let known_fields = match ir.nodes.iter().find(|n| n.name == node_name) {
        Some(n) => resolve_input_fields(ir, n),
        None => {
            // Synthetic node (from AddPromptStep): allowed vars = input + graph input fields
            if let Some(graph) = parent_graph {
                let mut vars = vec!["input".to_string()];
                vars.extend(crate::mutations::resolve_graph_input_fields(graph, ir));
                vars
            } else {
                return Ok(()); // can't validate without graph context
            }
        }
    };
    if known_fields.is_empty() {
        return Ok(()); // can't validate if we don't know the fields
    }

    // Render with strict undefined behavior so that referencing unknown variables errors.
    let mut env = minijinja::Environment::new();
    env.set_undefined_behavior(minijinja::UndefinedBehavior::Strict);
    let mut ctx = std::collections::BTreeMap::new();
    for field in &known_fields {
        ctx.insert(field.as_str(), minijinja::Value::from(""));
    }
    if let Err(e) = env.render_str(template, ctx) {
        let err_str = e.to_string();
        if err_str.contains("undefined value") {
            return Err(Error::Runtime(format!(
                "meta-agent proposed {} with undefined variable reference (available variables: {}): {}",
                field_name,
                known_fields.join(", "),
                err_str
            )));
        }
        // Other errors (syntax) are caught by validate_template_syntax
    }
    Ok(())
}

/// Validate that an add_prompt_step template only references allowed variables.
/// Allowed variables are `input` (previous step output) plus graph input field names.
fn validate_add_prompt_step_template_vars(template: &str, allowed_vars: &[String]) -> Result<()> {
    let mut env = minijinja::Environment::new();
    env.set_undefined_behavior(minijinja::UndefinedBehavior::Strict);
    let mut ctx = std::collections::BTreeMap::new();
    for var in allowed_vars {
        ctx.insert(var.as_str(), minijinja::Value::from(""));
    }
    if let Err(e) = env.render_str(template, ctx) {
        let err_str = e.to_string();
        if err_str.contains("undefined value") {
            return Err(Error::Runtime(format!(
                "add_prompt_step: template references undefined variable (available: {}): {}",
                allowed_vars.join(", "),
                err_str
            )));
        }
        // Other errors (syntax) are caught by validate_template_syntax
    }
    Ok(())
}

/// Parse raw test output into a concise one-liner summary.
///
/// Extracts the most useful signal: assertion errors with expected/actual,
/// exception names, failing test names. Falls back to first meaningful line.
fn parse_test_failure(raw: &str) -> String {
    // Try to extract from JSON test_output first
    if let Ok(json) = serde_json::from_str::<serde_json::Value>(raw) {
        if let Some(test_out) = json.get("test_output").and_then(|v| v.as_str()) {
            return parse_test_failure(test_out);
        }
    }

    let lines: Vec<&str> = raw
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect();

    // Look for assertion errors with expected/actual values
    for line in &lines {
        let lower = line.to_lowercase();
        if lower.contains("assertionerror") || lower.contains("assert ") {
            let short: String = line.chars().take(200).collect();
            return short;
        }
    }

    // Look for common exception patterns
    for line in &lines {
        let lower = line.to_lowercase();
        if lower.contains("error:") || lower.contains("exception") || lower.contains("traceback") {
            // Skip pure "Traceback (most recent call last):" — get the actual error
            if lower.starts_with("traceback") {
                continue;
            }
            let short: String = line.chars().take(200).collect();
            return short;
        }
    }

    // Look for FAILED test names
    for line in &lines {
        if line.starts_with("FAILED") || line.contains("FAILED") {
            let short: String = line.chars().take(200).collect();
            return short;
        }
    }

    // Look for "E " lines (pytest assertion detail)
    for line in &lines {
        if line.starts_with("E ") {
            let short: String = line.chars().take(200).collect();
            return short;
        }
    }

    // Fallback: last non-empty line (often the most informative in test output)
    lines
        .last()
        .map(|l| l.chars().take(200).collect::<String>())
        .unwrap_or_else(|| "unknown failure".to_string())
}

fn validate_node_exists(
    ir: &ScaffoldIR,
    node_name: &str,
    expected_kind: Option<NodeKindIR>,
) -> Result<()> {
    validate_node_exists_with_graph(ir, node_name, expected_kind, None)
}

/// Validate node exists — checks IR nodes first, then synthetic nodes in the graph.
fn validate_node_exists_with_graph(
    ir: &ScaffoldIR,
    node_name: &str,
    expected_kind: Option<NodeKindIR>,
    graph: Option<&GraphIR>,
) -> Result<()> {
    let node = ir.nodes.iter().find(|n| n.name == node_name);
    match node {
        Some(n) => {
            if let Some(kind) = expected_kind {
                if n.kind != kind {
                    return Err(Error::Runtime(format!(
                        "meta-agent expected node '{}' to be {:?} but it is {:?}",
                        node_name, kind, n.kind
                    )));
                }
            }
            Ok(())
        }
        None => {
            // Check if it's a synthetic node (used as a node name by a step in the graph)
            if let Some(g) = graph {
                if crate::mutations::node_used_by_step_pub(&g.body, node_name) {
                    // Synthetic nodes are always prompt type
                    if let Some(kind) = expected_kind {
                        if kind != NodeKindIR::Prompt {
                            return Err(Error::Runtime(format!(
                                "synthetic node '{}' is a prompt node, not {:?}",
                                node_name, kind
                            )));
                        }
                    }
                    return Ok(());
                }
            }
            Err(Error::Runtime(format!(
                "meta-agent referenced non-existent node '{}'",
                node_name
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    use crate::mutations::GraphDescriptor;
    use crate::optimizer::CaseResult;

    fn make_test_graph() -> GraphIR {
        GraphIR {
            name: "solve".into(),
            input: TypeIR::String,
            output: TypeIR::String,
            body: vec![
                GraphStmtIR::Step(StepIR {
                    name: "attempt1".into(),
                    node: "solve_code".into(),
                    args: vec![StepArgIR::Positional {
                        value: ExprIR::Ident {
                            name: "input".into(),
                        },
                    }],
                }),
                GraphStmtIR::Step(StepIR {
                    name: "result1".into(),
                    node: "eval_exercise".into(),
                    args: vec![StepArgIR::Positional {
                        value: ExprIR::Ident {
                            name: "attempt1".into(),
                        },
                    }],
                }),
                GraphStmtIR::Step(StepIR {
                    name: "attempt2".into(),
                    node: "fix_code".into(),
                    args: vec![StepArgIR::Positional {
                        value: ExprIR::Ident {
                            name: "attempt1".into(),
                        },
                    }],
                }),
                GraphStmtIR::Step(StepIR {
                    name: "result2".into(),
                    node: "eval_exercise".into(),
                    args: vec![StepArgIR::Positional {
                        value: ExprIR::Ident {
                            name: "attempt2".into(),
                        },
                    }],
                }),
                GraphStmtIR::Emit(EmitIR::Direct {
                    value: ExprIR::Ident {
                        name: "result2".into(),
                    },
                }),
            ],
        }
    }

    fn make_test_ir() -> ScaffoldIR {
        ScaffoldIR {
            version: "2.0.0".into(),
            types: vec![],
            nodes: vec![
                NodeIR {
                    name: "solve_code".into(),
                    kind: NodeKindIR::Prompt,
                    input: TypeIR::String,
                    output: TypeIR::String,
                    config: NodeConfigIR::default(),
                },
                NodeIR {
                    name: "fix_code".into(),
                    kind: NodeKindIR::Prompt,
                    input: TypeIR::String,
                    output: TypeIR::String,
                    config: NodeConfigIR::default(),
                },
                NodeIR {
                    name: "eval_exercise".into(),
                    kind: NodeKindIR::Tool,
                    input: TypeIR::String,
                    output: TypeIR::String,
                    config: NodeConfigIR::default(),
                },
            ],
            graphs: vec![make_test_graph()],
            objectives: vec![],
        }
    }

    fn make_case(case_id: &str, raw_output: &str, step_trace: Vec<(&str, &str)>) -> CaseResult {
        CaseResult {
            case_id: Some(case_id.into()),
            passed: false,
            checker_results: vec![],
            output_excerpt: Some(parse_test_failure(raw_output)),
            raw_output: Some(raw_output.into()),
            model_response: None,
            step_trace: step_trace
                .into_iter()
                .map(|(step, value)| (step.to_string(), value.to_string()))
                .collect(),
        }
    }

    fn make_candidate(
        id: usize,
        parent_id: Option<usize>,
        score: f64,
        mutations: Vec<Mutation>,
        case_results: Vec<CaseResult>,
        total_cases: usize,
        cases_passed: usize,
        passed_case_ids: Vec<&str>,
        graph: &GraphIR,
        ir: &ScaffoldIR,
    ) -> Candidate {
        Candidate {
            id,
            parent_id,
            graph: graph.clone(),
            overrides: HashMap::new(),
            mutations,
            score: Some(score),
            metric_scores: HashMap::new(),
            descriptor: GraphDescriptor::from_graph(graph, ir),
            children_count: 0,
            meta_reasoning: None,
            case_results,
            total_cases,
            cases_passed,
            passed_case_ids: passed_case_ids
                .into_iter()
                .map(|case| case.to_string())
                .collect(),
        }
    }

    fn make_objective() -> ObjectiveIR {
        ObjectiveIR {
            name: "aider_polyglot".into(),
            graph: "solve".into(),
            dataset: DatasetSpecIR::Inline { cases: vec![] },
            checkers: vec![],
            judges: vec![],
            metrics: vec![],
            score: ExprIR::LitFloat { value: 0.0 },
            repeats: None,
            split: None,
            select: None,
            tunables: vec![],
            topology: None,
            subs: vec![],
        }
    }

    #[test]
    fn test_failure_clusters_group_distinct_contracts() {
        let ir = make_test_ir();
        let graph = make_test_graph();
        let candidate = make_candidate(
            0,
            None,
            0.2,
            vec![],
            vec![
                make_case(
                    "python/bottle-song",
                    "AssertionError: Lists differ: ['Ten'] != ['ten']",
                    vec![],
                ),
                make_case(
                    "python/go-counting",
                    "ImportError: cannot import name 'WHITE' from 'go_counting'",
                    vec![],
                ),
                make_case(
                    "python/book-store",
                    "AssertionError: 146.4 != 14560",
                    vec![],
                ),
            ],
            3,
            0,
            vec![],
            &graph,
            &ir,
        );

        let clusters = summarize_failure_clusters(&candidate);
        let as_map: HashMap<&str, usize> = clusters
            .iter()
            .map(|cluster| (cluster.kind.label(), cluster.count))
            .collect();

        assert_eq!(as_map.get("exact-output-contract"), Some(&1));
        assert_eq!(as_map.get("api-shape-contract"), Some(&1));
        assert_eq!(as_map.get("algorithm-search-logic"), Some(&1));
    }

    #[test]
    fn test_repair_behavior_detects_noop_and_changed_attempts() {
        let ir = make_test_ir();
        let graph = make_test_graph();
        let candidate = make_candidate(
            0,
            None,
            0.2,
            vec![],
            vec![
                make_case(
                    "python/connect",
                    "{\"passed\": false, \"test_output\": \"AssertionError: '' != 'O'\"}",
                    vec![
                        ("attempt1", "code v1"),
                        (
                            "result1",
                            "{\"passed\": false, \"test_output\": \"AssertionError: '' != 'O'\"}",
                        ),
                        ("attempt2", "code v1"),
                        (
                            "result2",
                            "{\"passed\": false, \"test_output\": \"AssertionError: '' != 'O'\"}",
                        ),
                    ],
                ),
                make_case(
                    "python/book-store",
                    "{\"passed\": false, \"test_output\": \"AssertionError: 146.4 != 14560\"}",
                    vec![
                        ("attempt1", "greedy version"),
                        (
                            "result1",
                            "{\"passed\": false, \"test_output\": \"AssertionError: IndexError\"}",
                        ),
                        ("attempt2", "patched version"),
                        (
                            "result2",
                            "{\"passed\": false, \"test_output\": \"AssertionError: 146.4 != 14560\"}",
                        ),
                    ],
                ),
            ],
            2,
            0,
            vec![],
            &graph,
            &ir,
        );

        let summary = summarize_repair_behavior(&candidate, &ir);
        assert_eq!(summary.no_op, 1);
        assert_eq!(summary.changed_unresolved, 1);
        assert_eq!(summary.no_repair_signal, 0);

        // Per-case details
        assert_eq!(summary.per_case.len(), 2);
        assert_eq!(summary.per_case[0].case_id, "python/connect");
        assert_eq!(summary.per_case[0].outcome, RepairOutcome::NoOp);
        assert_eq!(
            summary.per_case[0].repair_node.as_deref(),
            Some("fix_code")
        );
        assert_eq!(summary.per_case[1].case_id, "python/book-store");
        assert_eq!(summary.per_case[1].outcome, RepairOutcome::ShiftedError);
    }

    #[test]
    fn test_repair_behavior_same_error_classification() {
        // When repair produces different code but the same error signature → SameError
        let ir = make_test_ir();
        let graph = make_test_graph();
        let candidate = make_candidate(
            0,
            None,
            0.2,
            vec![],
            vec![make_case(
                "python/matrix",
                "{\"passed\": false, \"test_output\": \"AssertionError: 1 != 2\"}",
                vec![
                    ("attempt1", "code version A"),
                    (
                        "result1",
                        "{\"passed\": false, \"test_output\": \"AssertionError: 1 != 2\"}",
                    ),
                    ("attempt2", "code version B with changes"),
                    (
                        "result2",
                        "{\"passed\": false, \"test_output\": \"AssertionError: 1 != 2\"}",
                    ),
                ],
            )],
            1,
            0,
            vec![],
            &graph,
            &ir,
        );

        let summary = summarize_repair_behavior(&candidate, &ir);
        assert_eq!(summary.no_op, 1); // SameError also counts as no_op
        assert_eq!(summary.per_case.len(), 1);
        assert_eq!(summary.per_case[0].outcome, RepairOutcome::SameError);
    }

    #[test]
    fn test_repair_behavior_single_attempt() {
        // When only one prompt step exists → SingleAttempt
        let ir = make_test_ir();
        let graph = make_test_graph();
        let candidate = make_candidate(
            0,
            None,
            0.2,
            vec![],
            vec![make_case(
                "python/hello",
                "{\"passed\": false, \"test_output\": \"AssertionError\"}",
                vec![
                    ("attempt1", "some code"),
                    (
                        "result1",
                        "{\"passed\": false, \"test_output\": \"AssertionError\"}",
                    ),
                ],
            )],
            1,
            0,
            vec![],
            &graph,
            &ir,
        );

        let summary = summarize_repair_behavior(&candidate, &ir);
        assert_eq!(summary.no_repair_signal, 1);
        assert_eq!(summary.per_case.len(), 1);
        assert_eq!(summary.per_case[0].outcome, RepairOutcome::SingleAttempt);
    }

    #[test]
    fn test_mutation_group_trend_flags_recent_saturation() {
        let ir = make_test_ir();
        let graph = make_test_graph();
        let seed = make_candidate(0, None, 0.2, vec![], vec![], 0, 0, vec![], &graph, &ir);
        let child1 = make_candidate(
            1,
            Some(0),
            0.3,
            vec![Mutation::RewriteSystem {
                node: "solve_code".into(),
                new_system: "v1".into(),
            }],
            vec![],
            0,
            0,
            vec![],
            &graph,
            &ir,
        );
        let child2 = make_candidate(
            2,
            Some(1),
            0.3,
            vec![Mutation::RewriteSystem {
                node: "solve_code".into(),
                new_system: "v2".into(),
            }],
            vec![],
            0,
            0,
            vec![],
            &graph,
            &ir,
        );
        let child3 = make_candidate(
            3,
            Some(2),
            0.28,
            vec![Mutation::RewriteSystem {
                node: "solve_code".into(),
                new_system: "v3".into(),
            }],
            vec![],
            0,
            0,
            vec![],
            &graph,
            &ir,
        );

        let mut archive = Archive::new();
        archive.add(seed);
        archive.add(child1);
        archive.add(child2);
        archive.add(child3);
        let trend = mutation_group_trend(
            &[
                &archive.candidates[1],
                &archive.candidates[2],
                &archive.candidates[3],
            ],
            &archive,
        );

        assert!(trend.ever_improved);
        assert!(trend.saturated);
        assert!(trend.verdict.contains("saturated"));
        assert_eq!(trend.recent_deltas.len(), 3);
        assert!((trend.recent_deltas[0] - 0.1).abs() < 1e-9);
        assert!(trend.recent_deltas[1].abs() < 1e-9);
        assert!((trend.recent_deltas[2] + 0.02).abs() < 1e-9);
    }

    #[test]
    fn test_build_context_includes_failure_decomposition_hints() {
        let ir = make_test_ir();
        let graph = make_test_graph();
        let objective = make_objective();

        let seed = make_candidate(
            0,
            None,
            0.2,
            vec![],
            vec![],
            4,
            1,
            vec!["python/beer-song"],
            &graph,
            &ir,
        );
        let parent = make_candidate(
            1,
            Some(0),
            0.25,
            vec![Mutation::RewriteSystem {
                node: "solve_code".into(),
                new_system: "be more exact".into(),
            }],
            vec![
                make_case(
                    "python/connect",
                    "{\"passed\": false, \"test_output\": \"AssertionError: '' != 'O'\"}",
                    vec![
                        ("attempt1", "code v1"),
                        (
                            "result1",
                            "{\"passed\": false, \"test_output\": \"AssertionError: '' != 'O'\"}",
                        ),
                        ("attempt2", "code v1"),
                        (
                            "result2",
                            "{\"passed\": false, \"test_output\": \"AssertionError: '' != 'O'\"}",
                        ),
                    ],
                ),
                make_case(
                    "python/go-counting",
                    "ImportError: cannot import name 'WHITE' from 'go_counting'",
                    vec![],
                ),
                make_case(
                    "python/book-store",
                    "AssertionError: 146.4 != 14560",
                    vec![],
                ),
            ],
            4,
            1,
            vec!["python/beer-song"],
            &graph,
            &ir,
        );
        let flat_child = make_candidate(
            2,
            Some(1),
            0.25,
            vec![Mutation::RewriteSystem {
                node: "solve_code".into(),
                new_system: "be more exact v2".into(),
            }],
            vec![],
            4,
            1,
            vec!["python/beer-song"],
            &graph,
            &ir,
        );

        let mut archive = Archive::new();
        archive.add(seed);
        archive.add(parent.clone());
        archive.add(flat_child);

        let ctx = build_context(
            &archive.candidates[1],
            &archive,
            &ir,
            &objective,
            &["add_prompt_step".into(), "set_config".into()],
        );

        assert!(ctx.contains("## Failure Decomposition Hints"));
        assert!(ctx.contains("Structural pressure:"));
        assert!(ctx.contains("Recent deltas:"));
        assert!(ctx.contains("exact-output-contract") || ctx.contains("api-shape-contract"));
    }

    // ── edit_graph tests ──

    #[test]
    fn test_parse_and_validate_graph_edit_valid() {
        let ir = make_test_ir();
        let parent_graph = make_test_graph();
        let objective = make_objective();

        let graph_source = r#"
            graph solve {
                in: string
                out: string
                step attempt1 = solve_code(input)
                step result1 = eval_exercise(attempt1)
                step attempt2 = fix_code(attempt1)
                step result2 = eval_exercise(attempt2)
                emit result2
            }
        "#;

        let (new_graph, new_nodes) =
            parse_and_validate_graph_edit(graph_source, &[], &parent_graph, &ir, &objective)
                .unwrap();
        assert_eq!(new_graph.name, "solve");
        assert!(new_nodes.is_empty()); // No new nodes — all existing
    }

    #[test]
    fn test_parse_and_validate_graph_edit_with_new_node() {
        let ir = make_test_ir();
        let parent_graph = make_test_graph();
        let objective = make_objective();

        let graph_source = r#"
            graph solve {
                in: string
                out: string
                step plan = planner(input)
                step attempt1 = solve_code(plan)
                step result1 = eval_exercise(attempt1)
                emit result1
            }
        "#;
        let new_node_source = r#"
            node planner: prompt {
                in: string
                out: string
                template: "Plan: {{ input }}"
            }
        "#;

        let (new_graph, new_nodes) = parse_and_validate_graph_edit(
            graph_source,
            &[new_node_source.to_string()],
            &parent_graph,
            &ir,
            &objective,
        )
        .unwrap();
        assert_eq!(new_graph.name, "solve");
        assert_eq!(new_nodes.len(), 1);
        assert_eq!(new_nodes[0].name, "planner");
    }

    #[test]
    fn test_parse_and_validate_graph_edit_reject_type_mismatch() {
        let ir = make_test_ir();
        let parent_graph = make_test_graph(); // input: string, output: string
        let objective = make_objective();

        let graph_source = r#"
            graph solve {
                in: int
                out: string
                step attempt1 = solve_code(input)
                emit attempt1
            }
        "#;

        let result =
            parse_and_validate_graph_edit(graph_source, &[], &parent_graph, &ir, &objective);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("input type mismatch"), "got: {}", err);
    }

    #[test]
    fn test_parse_and_validate_graph_edit_reject_missing_preserved() {
        let ir = make_test_ir();
        let parent_graph = make_test_graph();
        let mut objective = make_objective();
        objective.topology = Some(TopologyIR {
            mutations: vec!["insert_step".into()],
            max_nodes: None,
            max_depth: None,
            preserve: vec!["attempt1".into()],
            target_score: None,
        });

        // Graph without the preserved step "attempt1"
        let graph_source = r#"
            graph solve {
                in: string
                out: string
                step x = solve_code(input)
                emit x
            }
        "#;

        let result =
            parse_and_validate_graph_edit(graph_source, &[], &parent_graph, &ir, &objective);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("preserved step 'attempt1' missing"), "got: {}", err);
    }

    #[test]
    fn test_parse_and_validate_graph_edit_reject_unknown_node() {
        let ir = make_test_ir();
        let parent_graph = make_test_graph();
        let objective = make_objective();

        let graph_source = r#"
            graph solve {
                in: string
                out: string
                step attempt1 = nonexistent_node(input)
                emit attempt1
            }
        "#;

        let result =
            parse_and_validate_graph_edit(graph_source, &[], &parent_graph, &ir, &objective);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("unknown node 'nonexistent_node'"), "got: {}", err);
    }

    #[test]
    fn test_parse_and_validate_graph_edit_reject_wrong_graph_name() {
        let ir = make_test_ir();
        let parent_graph = make_test_graph(); // name: "solve"
        let objective = make_objective();

        let graph_source = r#"
            graph different_name {
                in: string
                out: string
                step attempt1 = solve_code(input)
                emit attempt1
            }
        "#;

        let result =
            parse_and_validate_graph_edit(graph_source, &[], &parent_graph, &ir, &objective);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("graph 'solve' not found"), "got: {}", err);
    }

    #[test]
    fn test_parse_proposal_edit_graph() {
        let ir = make_test_ir();
        let parent_graph = make_test_graph();
        let objective = make_objective();

        let response = r#"{
            "kind": "edit_graph",
            "graph": "graph solve {\n    in: string\n    out: string\n    step attempt1 = solve_code(input)\n    step result1 = eval_exercise(attempt1)\n    emit result1\n}",
            "description": "remove retry step",
            "reasoning": "simplify the graph"
        }"#;

        let result = parse_proposal(
            response,
            &ir,
            &parent_graph,
            &objective,
            &["insert_step".into()], // edit_graph allowed since structural mutations present
        );
        match result {
            Ok(proposal) => {
                assert_eq!(proposal.mutation.short_label(), "edit: remove retry step");
                if let Mutation::EditGraph { new_graph, description, .. } = &proposal.mutation {
                    assert_eq!(new_graph.name, "solve");
                    assert_eq!(description, "remove retry step");
                } else {
                    panic!("expected EditGraph");
                }
            }
            Err(e) => panic!("unexpected error: {}", e),
        }
    }

    #[test]
    fn test_parse_proposal_edit_graph_not_allowed_without_structural() {
        let ir = make_test_ir();
        let parent_graph = make_test_graph();
        let objective = make_objective();

        let response = r#"{
            "kind": "edit_graph",
            "graph": "graph solve {\n    in: string\n    out: string\n    step attempt1 = solve_code(input)\n    emit attempt1\n}",
            "description": "simplify",
            "reasoning": "test"
        }"#;

        // No structural mutations in allowed list — only content
        let result = parse_proposal(
            response,
            &ir,
            &parent_graph,
            &objective,
            &["set_config".into()],
        );
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("not in allowed mutations"), "got: {}", err);
    }

    #[test]
    fn test_build_context_shows_edit_graph_for_meta_agent() {
        let ir = make_test_ir();
        let graph = make_test_graph();
        let objective = make_objective();
        let seed = make_candidate(0, None, 0.5, vec![], vec![], 4, 2, vec!["a", "b"], &graph, &ir);

        let mut archive = Archive::new();
        archive.add(seed.clone());

        let ctx = build_context(
            &archive.candidates[0],
            &archive,
            &ir,
            &objective,
            &["insert_step".into(), "remove_step".into()],
        );

        // Should show edit_graph instead of individual structural mutations
        assert!(ctx.contains("edit_graph"), "context should contain edit_graph");
        assert!(!ctx.contains("Allowed mutations: insert_step"), "should not list individual structural mutations");
    }
}
