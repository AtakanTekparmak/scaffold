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
- {"kind":"insert_verify","after_step":"<step>","verify_node":"<node>","max_retries":<n>,"reasoning":"..."}
- {"kind":"wrap_retry","step":"<step>","verify_node":"<node>","max_retries":<n>,"reasoning":"..."}
- {"kind":"insert_step","after_step":"<step>","new_step_name":"<name>","node":"<node>","reasoning":"..."}
- {"kind":"remove_step","step":"<step>","reasoning":"..."}
- {"kind":"replace_component","step":"<step>","new_node":"<node>","reasoning":"..."}
- {"kind":"set_config","node":"<node>","field":"<field>","value":<json_value>,"reasoning":"..."}

PROMPT mutations:
- {"kind":"rewrite_prompt","node":"<node>","new_template":"<full template text>","reasoning":"..."}
- {"kind":"rewrite_system","node":"<node>","new_system":"<full system prompt text>","reasoning":"..."}

TOOL mutations:
- {"kind":"rewrite_shell","node":"<tool_node>","new_shell":"<shell command template>","reasoning":"..."}

Consider:
- Which candidates improved and why (their mutations)
- Which candidates regressed and why
- Whether the graph needs structural changes or prompt refinement
- What error patterns suggest (missing verification, wrong output format, insufficient context)

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
    ctx.push_str(&format!(
        "Checkers: {}\n",
        objective
            .checkers
            .iter()
            .map(|c| c.name.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    ));
    ctx.push_str(&format!(
        "Metrics: {}\n",
        objective
            .metrics
            .iter()
            .map(|m| format!("{} (from {})", m.name, m.checker))
            .collect::<Vec<_>>()
            .join(", ")
    ));
    // Show structural mutations from topology + content mutations (always available)
    let mut all_available: Vec<String> = allowed_mutations.to_vec();
    for content in &["rewrite_prompt", "rewrite_system", "rewrite_shell"] {
        if !all_available.iter().any(|m| m == content) {
            all_available.push(content.to_string());
        }
    }
    ctx.push_str(&format!(
        "Allowed mutations: {}\n",
        all_available.join(", ")
    ));

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

    // 4. Parent's failed cases (so the meta-agent can learn from failure patterns)
    if !parent.case_results.is_empty() {
        ctx.push_str("## Parent Failed Cases\n");
        ctx.push_str(&format!(
            "Score: {:.4} ({} failed cases stored for analysis)\n\n",
            parent.score.unwrap_or(0.0),
            parent.case_results.len(),
        ));
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

        ctx.push_str(&format!("- {} ({})", node.name, kind_str));

        // Include template excerpt if available
        if let Some(ref sof) = node.config.template {
            if let Ok(content) = load_template(sof) {
                let excerpt: String = content.chars().take(500).collect();
                let truncated = if content.len() > 500 { "..." } else { "" };
                ctx.push_str(&format!(
                    "\n  template: \"{}{}\"\n",
                    excerpt.replace('\n', "\\n"),
                    truncated
                ));
            }
        }

        // Include system prompt excerpt if available
        if let Some(ref sof) = node.config.system {
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

        // Include shell command for tool nodes
        if let Some(ref shell) = node.config.shell {
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
            let node = require_str(&json, "node")?;
            let field = require_str(&json, "field")?;
            let value = json
                .get("value")
                .cloned()
                .unwrap_or(serde_json::Value::Null);
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
            let new_template = require_str(&json, "new_template")?;
            validate_node_exists(ir, &node, None)?;
            Mutation::RewritePrompt { node, new_template }
        }
        "rewrite_system" => {
            let node = require_str(&json, "node")?;
            let new_system = require_str(&json, "new_system")?;
            validate_node_exists(ir, &node, None)?;
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
