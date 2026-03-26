//! LLM-guided meta-agent for proposing targeted mutations (Algorithm 2, DGM-H).
//!
//! Instead of random dice rolls, the meta-agent receives rich context about the
//! parent candidate, archive history, and available nodes, then proposes a
//! structured mutation via an LLM call.

use scaffold_ir::ir::*;

use crate::error::{Error, Result};
use crate::llm::{self, LlmConfig};
use crate::mutations::Mutation;
use crate::node_runner::load_template;
use crate::optimizer::{Archive, Candidate};

/// A mutation proposal from the meta-agent, with reasoning.
#[derive(Debug, Clone)]
pub struct MutationProposal {
    pub mutation: Mutation,
    pub reasoning: String,
}

/// The meta-agent: an LLM-based mutation proposer.
pub struct MetaAgent {
    model: String,
}

impl MetaAgent {
    pub fn new(model: &str) -> Self {
        Self {
            model: model.to_string(),
        }
    }

    /// Propose a mutation given the parent candidate, archive, IR, and objective.
    pub async fn propose_mutation(
        &self,
        parent: &Candidate,
        archive: &Archive,
        ir: &ScaffoldIR,
        objective: &ObjectiveIR,
        allowed_mutations: &[String],
    ) -> Result<MutationProposal> {
        let context = build_context(parent, archive, ir, objective, allowed_mutations);
        let llm_config = LlmConfig::new()
            .with_model(&self.model)
            .with_temperature(0.7)
            .with_system_prompt(SYSTEM_PROMPT);

        let response = llm::query_with_config(&context, &llm_config).await?;

        parse_proposal(&response, ir, &parent.graph, objective, allowed_mutations)
    }
}

const SYSTEM_PROMPT: &str = r#"You are a meta-optimizer for computational graphs. You analyze an optimization archive and propose targeted mutations to improve a graph's score on a dataset evaluation.

You can propose exactly ONE mutation as a JSON object with these fields:
- "kind": one of the mutation types listed below
- "reasoning": a brief explanation of why this mutation should help
- Plus kind-specific fields

STRUCTURAL mutations:
- {"kind":"insert_verify","after_step":"<step>","verify_node":"<node>","max_retries":<n>,"reasoning":"..."} (verify_node MUST be a node of kind "verify", NOT prompt/tool/agent)
- {"kind":"wrap_retry","step":"<step>","verify_node":"<node>","max_retries":<n>,"reasoning":"..."} (verify_node MUST be a node of kind "verify", NOT prompt/tool/agent)
- {"kind":"insert_step","after_step":"<step>","new_step_name":"<name>","node":"<node>","reasoning":"..."}
- {"kind":"remove_step","step":"<step>","reasoning":"..."}
- {"kind":"replace_component","step":"<step>","new_node":"<node>","reasoning":"..."}
- {"kind":"set_config","node":"<node>","field":"<field>","value":<json_value>,"reasoning":"..."}

PROMPT mutations:
- {"kind":"rewrite_prompt","node":"<node>","new_instructions":"<instructions text ONLY>","reasoning":"..."} — provide ONLY the natural-language instructions (rules, guidance, constraints). The data bindings ({{ variables }}, {% for %} loops) and output format block ({% raw %}...{% endraw %}) from the original template are automatically preserved. Do NOT include any Jinja2 tags in new_instructions.
- {"kind":"rewrite_system","node":"<node>","new_system":"<full system prompt text>","reasoning":"..."}

TOOL mutations:
- {"kind":"rewrite_shell","node":"<tool_node>","new_shell":"<shell command template>","reasoning":"..."}

CRITICAL RULES:
1. ALWAYS check the "Mutation History" section first. NEVER propose a mutation that previously caused a catastrophic regression.
2. If a rewrite_prompt or rewrite_system for a specific node previously scored 0.0 or near-zero, DO NOT rewrite that node's prompt again.
3. Preserve what works — most cases may already pass, so avoid changes that risk breaking them.
4. If you see a "Stagnation Warning", you MUST diversify: do NOT repeat the same mutation type that has already been tried without improving the best score. Try a different node, a different mutation kind, or a more creative approach.
5. Node type constraints — rewrite_prompt and rewrite_system apply ONLY to prompt/agent nodes. rewrite_shell applies ONLY to tool nodes. Check the "Available Nodes" section for each node's type and valid mutations.
6. For rewrite_prompt, provide ONLY the instructional text in "new_instructions". Do NOT include {{ variables }}, {% for %} loops, or {% raw %} blocks — these are automatically preserved from the original template. Focus only on improving the natural-language rules and guidance.

Consider:
- The overall pass rate and which cases pass vs fail
- Which candidates improved and why (their mutations)
- Which candidates regressed and why — learn from failures
- Whether the graph needs structural changes or prompt refinement
- What specific failure patterns suggest (missing verification, wrong output format, insufficient context)
- Make incremental changes to prompts rather than full rewrites when possible

Return ONLY a single JSON object. No markdown, no explanation outside the JSON."#;

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
    // Filter out verify-requiring mutations (insert_verify, wrap_retry) if no Verify nodes exist.
    let has_verify_nodes = ir.nodes.iter().any(|n| n.kind == NodeKindIR::Verify);
    let has_tool_nodes = ir.nodes.iter().any(|n| n.kind == NodeKindIR::Tool);
    let has_prompt_nodes = ir.nodes.iter().any(|n| matches!(n.kind, NodeKindIR::Prompt | NodeKindIR::Agent));
    let mut all_available: Vec<String> = allowed_mutations
        .iter()
        .filter(|m| {
            if !has_verify_nodes && (*m == "insert_verify" || *m == "wrap_retry") {
                return false;
            }
            true
        })
        .cloned()
        .collect();
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
    if !has_verify_nodes {
        ctx.push_str("Note: No verify nodes exist — insert_verify and wrap_retry are NOT available.\n");
    }
    if !has_tool_nodes {
        ctx.push_str("Note: No tool nodes exist — rewrite_shell is NOT available.\n");
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
    ctx.push_str("## Parent Graph (current best candidate to mutate)\n");
    ctx.push_str(&format!(
        "Score: {:.4}\n",
        parent.score.unwrap_or(0.0)
    ));
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

    // 3. Archive summary (top 10 candidates)
    ctx.push_str("## Archive (top candidates by score)\n");
    let ranked = archive.ranked();
    for (i, c) in ranked.iter().take(10).enumerate() {
        let mutations_str = if c.mutations.is_empty() {
            "seed".to_string()
        } else {
            c.mutations
                .iter()
                .map(|m| m.short_label())
                .collect::<Vec<_>>()
                .join(", ")
        };
        ctx.push_str(&format!(
            "{}. id={} score={:.4} parent={:?} mutations=[{}] children={}\n",
            i + 1,
            c.id,
            c.score.unwrap_or(0.0),
            c.parent_id,
            mutations_str,
            c.children_count,
        ));
    }
    ctx.push('\n');

    // 3.5 Mutation history — flag catastrophic regressions so the meta-agent avoids repeating them
    {
        let mut catastrophic: Vec<String> = Vec::new();
        let mut regressions: Vec<String> = Vec::new();

        for c in archive.candidates.iter().filter(|c| c.score.is_some() && !c.mutations.is_empty()) {
            let score = c.score.unwrap();
            let parent_score = c.parent_id
                .and_then(|pid| archive.candidates.iter().find(|p| p.id == pid))
                .and_then(|p| p.score);

            let label = c.mutations.last().map(|m| m.short_label()).unwrap_or_default();

            // Catastrophic: score dropped to near-zero or dropped by >50% from parent
            let is_catastrophic = match parent_score {
                Some(ps) if ps > 0.1 => score < ps * 0.5,
                _ => score <= 0.01,
            };

            if is_catastrophic {
                catastrophic.push(format!(
                    "- CATASTROPHIC: {} → score={:.4} (parent score={:.4}) [candidate {}]",
                    label, score, parent_score.unwrap_or(0.0), c.id,
                ));
            } else if let Some(ps) = parent_score {
                if score < ps * 0.8 {
                    regressions.push(format!(
                        "- REGRESSED: {} → score={:.4} (parent score={:.4}) [candidate {}]",
                        label, score, ps, c.id,
                    ));
                }
            }
        }

        if !catastrophic.is_empty() || !regressions.is_empty() {
            ctx.push_str("## Mutation History — AVOID REPEATING FAILURES\n");
            if !catastrophic.is_empty() {
                ctx.push_str("**The following mutations caused catastrophic score drops. DO NOT propose similar mutations:**\n");
                for line in &catastrophic {
                    ctx.push_str(line);
                    ctx.push('\n');
                }
            }
            if !regressions.is_empty() {
                ctx.push_str("Regressions (score dropped significantly):\n");
                for line in &regressions {
                    ctx.push_str(line);
                    ctx.push('\n');
                }
            }
            ctx.push('\n');
        }
    }

    // 3.6 Stagnation detection — track consecutive non-improving generations
    {
        let ranked = archive.ranked();
        let best_score = ranked.first().map(|c| c.score.unwrap_or(0.0)).unwrap_or(0.0);
        let best_id = ranked.first().map(|c| c.id).unwrap_or(0);

        // Count evaluated candidates added after the best one
        let stagnant_candidates: Vec<&Candidate> = archive.candidates.iter()
            .filter(|c| c.id > best_id && c.score.is_some() && !c.mutations.is_empty())
            .collect();
        let stagnation_count = stagnant_candidates.len();

        if stagnation_count >= 2 {
            // Track mutation type frequencies among non-improving candidates
            let mut mutation_type_counts: Vec<(String, usize)> = Vec::new();
            for c in &stagnant_candidates {
                let label = c.mutations.last().map(|m| m.short_label()).unwrap_or_default();
                if let Some(entry) = mutation_type_counts.iter_mut().find(|(l, _)| *l == label) {
                    entry.1 += 1;
                } else {
                    mutation_type_counts.push((label, 1));
                }
            }
            mutation_type_counts.sort_by(|a, b| b.1.cmp(&a.1));

            ctx.push_str("## Stagnation Warning\n");
            ctx.push_str(&format!(
                "Best score ({:.4}) has NOT improved for {} consecutive candidates.\n",
                best_score, stagnation_count
            ));
            if !mutation_type_counts.is_empty() {
                ctx.push_str("Mutation types tried since the best (NONE improved):\n");
                for (label, count) in &mutation_type_counts {
                    ctx.push_str(&format!("- {} (tried {} time{})\n", label, count, if *count > 1 { "s" } else { "" }));
                }
            }
            ctx.push_str("You MUST try a fundamentally different mutation type or target a different node.\n");
            ctx.push_str("Do NOT repeat any of the mutation types listed above.\n\n");
        }
    }

    // 4. Parent's evaluation summary + failed cases
    if parent.total_cases > 0 {
        let failed = parent.total_cases - parent.cases_passed;
        ctx.push_str("## Parent Evaluation Summary\n");
        ctx.push_str(&format!(
            "Score: {:.4} | {}/{} cases PASSED, {}/{} FAILED\n",
            parent.score.unwrap_or(0.0),
            parent.cases_passed, parent.total_cases,
            failed, parent.total_cases,
        ));
        // Per-metric breakdown
        if !parent.metric_scores.is_empty() {
            ctx.push_str("Per-metric scores: ");
            let metrics: Vec<String> = parent.metric_scores.iter()
                .map(|(k, v)| format!("{}={:.4}", k, v))
                .collect();
            ctx.push_str(&metrics.join(", "));
            ctx.push('\n');
        }
        ctx.push('\n');

        if !parent.case_results.is_empty() {
            ctx.push_str(&format!("Failed cases ({}):\n", parent.case_results.len()));
            for (i, case) in parent.case_results.iter().take(20).enumerate() {
                let id = case.case_id.as_deref().unwrap_or("?");
                let checkers: Vec<String> = case.checker_results.iter()
                    .map(|(name, ok)| format!("{}={}", name, if *ok { "PASS" } else { "FAIL" }))
                    .collect();
                ctx.push_str(&format!(
                    "  {}. case={} [{}]\n",
                    i + 1, id, checkers.join(", ")
                ));
            }
            if parent.case_results.len() > 20 {
                ctx.push_str(&format!("  ... and {} more\n", parent.case_results.len() - 20));
            }
        }
        ctx.push('\n');
    }

    // 5. Available nodes with template excerpts
    ctx.push_str("## Available Nodes\n");
    for node in &ir.nodes {
        let kind_str = match node.kind {
            NodeKindIR::Prompt => "prompt",
            NodeKindIR::Tool => "tool",
            NodeKindIR::Agent => "agent",
            NodeKindIR::Verify => "verify",
        };

        // Show applicable mutations per node type
        let applicable = match node.kind {
            NodeKindIR::Prompt | NodeKindIR::Agent => "rewrite_prompt, rewrite_system, set_config",
            NodeKindIR::Tool => "rewrite_shell, set_config",
            NodeKindIR::Verify => "set_config",
        };

        // Show available template variables (input field names)
        let input_fields = resolve_input_fields(ir, node);
        let vars_str = if input_fields.is_empty() {
            String::new()
        } else {
            format!(" [vars: {}]", input_fields.join(", "))
        };

        let output_type_str = scaffold_ir::pretty::format_type(&node.output);
        // Also resolve named types to show actual structure
        let resolved_output = resolve_type_for_display(ir, &node.output);
        let output_display = if resolved_output != output_type_str {
            format!("{} = {}", output_type_str, resolved_output)
        } else {
            output_type_str
        };
        ctx.push_str(&format!("- {} ({}) [mutations: {}]{}\n  output type: {}", node.name, kind_str, applicable, vars_str, output_display));

        // Include template: show instructions section only (what meta-agent can change)
        let override_template_key = format!("{}.template", node.name);
        let full_template = if let Some(override_val) = parent.overrides.get(&override_template_key) {
            override_val.as_str().map(|s| s.to_string())
        } else if let Some(ref sof) = node.config.template {
            load_template(sof).ok()
        } else {
            None
        };
        if let Some(content) = full_template {
            let data_section = extract_data_section(&content);
            let instructions = if !data_section.is_empty() {
                content[..content.len() - data_section.len()].trim().to_string()
            } else {
                content.clone()
            };
            let is_overridden = parent.overrides.contains_key(&override_template_key);
            let label = if is_overridden { "instructions (OVERRIDDEN, editable)" } else { "instructions (editable)" };
            let excerpt: String = instructions.chars().take(500).collect();
            let truncated = if instructions.len() > 500 { "..." } else { "" };
            ctx.push_str(&format!(
                "\n  {}: \"{}{}\"\n",
                label,
                excerpt.replace('\n', "\\n"),
                truncated
            ));
            if !data_section.is_empty() {
                ctx.push_str("  [data bindings + format block auto-preserved]\n");
            }
        }

        // Include system prompt excerpt — use override if present, else original
        let override_system_key = format!("{}.system", node.name);
        if let Some(override_val) = parent.overrides.get(&override_system_key) {
            if let Some(content) = override_val.as_str() {
                let excerpt: String = content.chars().take(300).collect();
                let truncated = if content.len() > 300 { "..." } else { "" };
                ctx.push_str(&format!(
                    "  system (OVERRIDDEN): \"{}{}\"\n",
                    excerpt.replace('\n', "\\n"),
                    truncated
                ));
            }
        } else if let Some(ref sof) = node.config.system {
            if let Ok(content) = load_template(sof) {
                let excerpt: String = content.chars().take(300).collect();
                let truncated = if content.len() > 300 { "..." } else { "" };
                ctx.push_str(&format!(
                    "  system: \"{}{}\"\n",
                    excerpt.replace('\n', "\\n"),
                    truncated
                ));
            }
        }

        // Include shell command for tool nodes — use override if present
        let override_shell_key = format!("{}.shell", node.name);
        if let Some(override_val) = parent.overrides.get(&override_shell_key) {
            if let Some(content) = override_val.as_str() {
                let excerpt: String = content.chars().take(500).collect();
                let truncated = if content.len() > 500 { "..." } else { "" };
                ctx.push_str(&format!(
                    "  shell (OVERRIDDEN): \"{}{}\"\n",
                    excerpt.replace('\n', "\\n"),
                    truncated
                ));
            }
        } else if let Some(ref shell) = node.config.shell {
            let excerpt: String = shell.chars().take(500).collect();
            let truncated = if shell.len() > 500 { "..." } else { "" };
            ctx.push_str(&format!(
                "  shell: \"{}{}\"\n",
                excerpt.replace('\n', "\\n"),
                truncated
            ));
        }

        if let Some(ref model) = node.config.model {
            ctx.push_str(&format!("  model: {}\n", model));
        }

        ctx.push('\n');
    }

    // 5. Graph steps for reference
    ctx.push_str("## Steps in Parent Graph\n");
    let step_names = crate::mutations::collect_step_names(&parent.graph.body);
    for name in &step_names {
        // Find which node this step uses
        let node_name = find_step_node(&parent.graph.body, name);
        ctx.push_str(&format!(
            "- step '{}' → node '{}'\n",
            name,
            node_name.unwrap_or_else(|| "?".to_string())
        ));
    }

    ctx
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

/// Parse the LLM response into a MutationProposal.
fn parse_proposal(
    response: &str,
    ir: &ScaffoldIR,
    parent_graph: &GraphIR,
    objective: &ObjectiveIR,
    allowed_mutations: &[String],
) -> Result<MutationProposal> {
    let json: serde_json::Value = llm::parse_json_with_repairs(response).map_err(|e| {
        Error::Runtime(format!("meta-agent returned invalid JSON: {}", e))
    })?;

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
    let content_mutations = ["rewrite_prompt", "rewrite_system", "rewrite_shell"];
    if !content_mutations.contains(&kind) && !allowed_mutations.contains(&kind.to_string()) {
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
            Mutation::ReplaceComponent { step, new_node }
        }
        "set_config" => {
            let raw_node = require_str(&json, "node")?;
            let raw_field = json.get("field").and_then(|v| v.as_str()).map(|s| s.to_string());
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
                (raw_node[..dot_pos].to_string(), raw_node[dot_pos + 1..].to_string())
            } else {
                return Err(Error::Runtime(
                    "meta-agent response missing 'field' for set_config".into()
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
        "rewrite_prompt" => {
            let node = require_str(&json, "node")?;
            // Accept either "new_instructions" (preferred) or "new_template" (legacy)
            let new_instructions = json.get("new_instructions").and_then(|v| v.as_str())
                .or_else(|| json.get("new_template").and_then(|v| v.as_str()))
                .map(|s| s.to_string())
                .ok_or_else(|| Error::Runtime(
                    "meta-agent response missing 'new_instructions' field for rewrite_prompt".into()
                ))?;
            validate_node_exists(ir, &node, None)?;

            // Load original template and split into instructions vs data+format
            let original = load_original_template(ir, parent_graph, &node);
            let new_template = match original {
                Some(ref orig) => {
                    let data_section = extract_data_section(orig);
                    if data_section.is_empty() {
                        // No data section found — use as-is
                        new_instructions
                    } else {
                        // Strip any Jinja2 tags the meta-agent may have included
                        let clean_instructions = strip_jinja_tags(&new_instructions);
                        format!("{}\n\n{}", clean_instructions.trim(), data_section)
                    }
                }
                None => new_instructions,
            };

            validate_template_syntax(&new_template, "new_template")?;
            validate_template_variables(&new_template, ir, &node, "new_template")?;
            Mutation::RewritePrompt { node, new_template }
        }
        "rewrite_system" => {
            let node = require_str(&json, "node")?;
            let new_system = require_str(&json, "new_system")?;
            validate_node_exists(ir, &node, None)?;
            validate_template_syntax(&new_system, "new_system")?;
            validate_template_variables(&new_system, ir, &node, "new_system")?;
            Mutation::RewriteSystem { node, new_system }
        }
        "rewrite_shell" => {
            let node = require_str(&json, "node")?;
            let new_shell = require_str(&json, "new_shell")?;
            validate_node_exists(ir, &node, Some(NodeKindIR::Tool))?;
            Mutation::RewriteShell { node, new_shell }
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
    })
}

fn require_str(json: &serde_json::Value, field: &str) -> Result<String> {
    json.get(field)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| Error::Runtime(format!("meta-agent response missing '{}' field", field)))
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

/// Load the original template for a node, checking overrides first then IR config.
fn load_original_template(ir: &ScaffoldIR, _parent_graph: &GraphIR, node_name: &str) -> Option<String> {
    // Try to load from the IR node config
    let node = ir.nodes.iter().find(|n| n.name == node_name)?;
    let sof = node.config.template.as_ref()?;
    load_template(sof).ok()
}

/// Extract the "data section" of a template — everything from the first line containing
/// a Jinja2 tag ({{ or {%) to the end. This includes variable bindings, for loops,
/// and the {% raw %}...{% endraw %} format specification.
fn extract_data_section(template: &str) -> String {
    let lines: Vec<&str> = template.lines().collect();
    // Find the first line with a Jinja2 tag
    for (i, line) in lines.iter().enumerate() {
        if line.contains("{{") || line.contains("{%") {
            return lines[i..].join("\n");
        }
    }
    String::new()
}

/// Strip Jinja2 tags from text that should be instructions-only.
/// Removes lines containing {{ }}, {% %}, and {% raw %} blocks.
fn strip_jinja_tags(text: &str) -> String {
    let mut result = Vec::new();
    let mut in_raw = false;
    for line in text.lines() {
        if line.contains("{% raw %}") {
            in_raw = true;
            continue;
        }
        if line.contains("{% endraw %}") {
            in_raw = false;
            continue;
        }
        if in_raw {
            continue;
        }
        if line.contains("{{") || line.contains("{%") {
            continue;
        }
        result.push(line);
    }
    result.join("\n")
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

/// Recursively resolve named types and format for display.
fn resolve_type_for_display(ir: &ScaffoldIR, ty: &scaffold_ir::ir::TypeIR) -> String {
    match ty {
        scaffold_ir::ir::TypeIR::Named { name } => {
            if let Some(td) = ir.types.iter().find(|t| t.name == *name) {
                resolve_type_for_display(ir, &td.ty)
            } else {
                name.clone()
            }
        }
        scaffold_ir::ir::TypeIR::Struct { fields } => {
            let fs: Vec<String> = fields.iter()
                .map(|f| format!("{}: {}", f.name, resolve_type_for_display(ir, &f.ty)))
                .collect();
            format!("{{ {} }}", fs.join(", "))
        }
        scaffold_ir::ir::TypeIR::List { element } => {
            format!("list<{}>", resolve_type_for_display(ir, element))
        }
        other => scaffold_ir::pretty::format_type(other),
    }
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
fn validate_template_variables(template: &str, ir: &ScaffoldIR, node_name: &str, field_name: &str) -> Result<()> {
    let node = match ir.nodes.iter().find(|n| n.name == node_name) {
        Some(n) => n,
        None => return Ok(()), // node validation will catch this separately
    };
    let known_fields = resolve_input_fields(ir, node);
    if known_fields.is_empty() {
        return Ok(()); // can't validate if we don't know the fields
    }

    // Try rendering with a context that has all known fields set to empty strings.
    // If it still fails with "undefined value", the template references unknown variables.
    let env = minijinja::Environment::new();
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

fn validate_node_exists(ir: &ScaffoldIR, node_name: &str, expected_kind: Option<NodeKindIR>) -> Result<()> {
    let node = ir.nodes.iter().find(|n| n.name == node_name);
    match node {
        None => Err(Error::Runtime(format!(
            "meta-agent referenced non-existent node '{}'",
            node_name
        ))),
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
    }
}
