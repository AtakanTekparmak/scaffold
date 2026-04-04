//! Motif library: diagnosis-driven decomposition patterns.
//!
//! A motif is a named graph transformation triggered by specific failure patterns.
//! Instead of the meta-agent rewriting the graph as free-form DSL, it selects a
//! motif and configures it. Each motif maps to a concrete graph transformation that
//! inserts well-defined nodes with structured prompts.

use std::collections::HashMap;

use scaffold_ir::ir::*;
use serde::{Deserialize, Serialize};

/// A decomposition motif: a named graph transformation pattern.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Motif {
    /// Invalid format / invalid label → normalize + verify.
    NormalizeVerify,
    /// Many close label confusions → shortlist + select.
    ShortlistSelect,
    /// Clear domain clusters → router + expert ensemble.
    RouterExpert,
    /// Missing external facts → retrieve + decide.
    RetrieveDecide,
    /// High output variance → vote or critique + repair.
    VoteCritiqueRepair,
    /// Tool-in-the-loop → generate, validate with tool, refine based on feedback.
    GenerateValidateRefine,
}

impl std::fmt::Display for Motif {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Motif::NormalizeVerify => write!(f, "normalize_verify"),
            Motif::ShortlistSelect => write!(f, "shortlist_select"),
            Motif::RouterExpert => write!(f, "router_expert"),
            Motif::RetrieveDecide => write!(f, "retrieve_decide"),
            Motif::VoteCritiqueRepair => write!(f, "vote_critique_repair"),
            Motif::GenerateValidateRefine => write!(f, "generate_validate_refine"),
        }
    }
}

/// Result of applying a motif: new graph body, synthetic nodes, and optional local checkers.
pub struct MotifApplication {
    /// New graph body statements with the motif's nodes inserted.
    pub new_body: Vec<GraphStmtIR>,
    /// Synthetic nodes created by the motif (stored as overrides).
    pub synthetic_nodes: Vec<SyntheticNode>,
    /// Local checkers attached to new nodes.
    pub local_checkers: Vec<LocalChecker>,
}

/// A synthetic node created by a motif, stored as overrides.
pub struct SyntheticNode {
    pub name: String,
    pub kind: NodeKindIR,
    pub template: String,
    pub system: Option<String>,
    /// Shell command for tool nodes (stored as `<name>.shell` override).
    pub shell: Option<String>,
    /// ToolSpec JSON for tool nodes (stored as `<name>._tool_spec` override).
    /// Takes precedence over `shell` at runtime.
    pub tool_spec_json: Option<serde_json::Value>,
}

/// A local checker attached to a node's output.
pub struct LocalChecker {
    pub node: String,
    pub name: String,
    pub expr: String,
}

// ── Motif Templates ──

const SHORTLIST_TEMPLATE: &str = r#"Analyze the input and produce {{k}} candidate answers. Return a JSON list of candidates.
Return JSON: {"candidates": ["..."]}

INPUT: {{input}}"#;

const CHOOSE_TEMPLATE: &str = r#"Select the single best answer from CANDIDATES.
Return JSON: {"answer": "..."}.
If uncertain, still choose the best from CANDIDATES only.

INPUT: {{input}}
CANDIDATES: {{candidates}}"#;

const NORMALIZE_TEMPLATE: &str = r#"Normalize RAW to match the expected output format.
Fix spelling, case, singular/plural, and alias variants.
Output ONLY the corrected value — no explanation, no markdown, no quotes, no preamble.

RAW: {{raw}}"#;

// Reserved for future use when NormalizeVerify gains a verify step.
#[allow(dead_code)]
const VERIFY_FORMAT_TEMPLATE: &str = r#"Check if OUTPUT matches the expected format and constraints.
Return JSON: {"pass": true} if it matches, {"pass": false, "reason": "..."} if not.

OUTPUT: {{output}}"#;

const ROUTER_TEMPLATE: &str = r#"Classify the input into one category: {{domains}}.
Return JSON: {"domain": "..."}.

INPUT: {{input}}"#;

const RETRIEVE_TEMPLATE: &str = r#"Extract key facts, entities, and context needed to answer the question.
Return JSON: {"facts": ["...", "..."], "context": "..."}.

INPUT: {{input}}"#;

const DECIDE_TEMPLATE: &str = r#"Given the retrieved context below, produce the final answer.
Output ONLY the answer — no explanation, no markdown, no quotes, no preamble.

RETRIEVED: {{retrieved}}
INPUT: {{input}}"#;

const CRITIQUE_TEMPLATE: &str = r#"Evaluate the PROPOSED answer. Is it correct, or should it change?
Return ONLY a JSON object: {"verdict":"keep","better_label":"","confidence":1.0} if correct, or {"verdict":"change","better_label":"<corrected answer>","confidence":0.0-1.0} if wrong.
No explanation outside the JSON.

PROPOSED: {{proposed}}
INPUT: {{input}}"#;

/// Python script for repair gate: reads critique from stdin, proposed as argv[1].
/// Handles two critique formats:
///   1. JSON: {"verdict":"change","better_label":"..."}
///   2. Text: "CORRECT" or "WRONG - [corrected label]"
/// Falls back to proposed on any parse failure.
/// Uses ToolSpec (argv + stdin) to avoid shell quoting issues with LLM output.
const REPAIR_GATE_SCRIPT: &str = r#"import sys, json
proposed = sys.argv[1]
critique = sys.stdin.read().strip()
# Try JSON format first (anywhere in the text)
for line in critique.split('\n'):
    line = line.strip()
    if not line:
        continue
    try:
        c = json.loads(line)
        if c.get("verdict") == "change" and c.get("better_label", "").strip():
            print(c["better_label"].strip())
        else:
            print(proposed)
        sys.exit(0)
    except Exception:
        pass
# Try text format: scan lines in reverse to find the verdict line.
# LLMs often produce analysis before the verdict.
for line in reversed(critique.split('\n')):
    line = line.strip()
    if not line:
        continue
    up = line.upper()
    if up.startswith("CORRECT"):
        print(proposed)
        sys.exit(0)
    if up.startswith("WRONG"):
        for sep in [" - ", ": ", " – ", "- "]:
            if sep in line:
                label = line.split(sep, 1)[1].strip()
                if label:
                    print(label)
                    sys.exit(0)
        print(proposed)
        sys.exit(0)
# No verdict found — keep proposed unchanged
print(proposed)"#;

const VALIDATE_TOOL_TEMPLATE: &str = "echo '{{proposed}}'";

const REFINE_TEMPLATE: &str = r#"The previous answer was validated. FEEDBACK from the validator:
{{feedback}}

If the feedback indicates issues, fix them. If it says PASS, return the original unchanged.
Output ONLY the corrected answer — no explanation, no markdown, no quotes, no preamble.

PREVIOUS: {{proposed}}
INPUT: {{input}}"#;

/// System prompt for terminal motif nodes (normalize, decide, repair) that produce the final output.
/// Forces the model to output only the answer with no extra text.
const TERMINAL_SYSTEM: &str = "You are a precise assistant. Output ONLY the requested value. \
No explanation, no markdown formatting, no quotes around the answer, no preamble. \
Just the raw answer text.";

/// Resolve a template from config overrides, falling back to the default.
///
/// The meta-agent can provide task-specific templates via `config.templates.<key>`.
fn resolve_template(config: &HashMap<String, serde_json::Value>, key: &str, default: &str) -> String {
    config
        .get("templates")
        .and_then(|t| t.get(key))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| default.to_string())
}

/// Apply a motif to a graph, targeting a specific step.
///
/// Returns the new graph body with motif nodes inserted, synthetic nodes to create,
/// and local checkers for the new nodes.
pub fn apply_motif(
    motif: &Motif,
    graph: &GraphIR,
    target_step: &str,
    config: &HashMap<String, serde_json::Value>,
    ir: &ScaffoldIR,
) -> Result<MotifApplication, String> {
    // Find the target step and its position
    let step_idx = find_step_index(&graph.body, target_step)
        .ok_or_else(|| format!("target step '{}' not found in graph", target_step))?;

    let step = match &graph.body[step_idx] {
        GraphStmtIR::Step(s) => s.clone(),
        _ => return Err(format!("'{}' is not a step statement", target_step)),
    };

    match motif {
        Motif::ShortlistSelect => apply_shortlist_select(graph, &step, step_idx, config),
        Motif::NormalizeVerify => apply_normalize_verify(graph, &step, step_idx, config, ir),
        Motif::RouterExpert => apply_router_expert(graph, &step, step_idx, config),
        Motif::RetrieveDecide => apply_retrieve_decide(graph, &step, step_idx, config),
        Motif::VoteCritiqueRepair => apply_vote_critique_repair(graph, &step, step_idx, config),
        Motif::GenerateValidateRefine => apply_generate_validate_refine(graph, &step, step_idx, config),
    }
}

/// Find the index of a step with the given name in a flat body.
fn find_step_index(body: &[GraphStmtIR], name: &str) -> Option<usize> {
    body.iter().position(|stmt| match stmt {
        GraphStmtIR::Step(s) => s.name == name,
        _ => false,
    })
}

/// Build a step IR that passes a single positional argument.
fn make_step(name: &str, node: &str, args: Vec<StepArgIR>) -> GraphStmtIR {
    GraphStmtIR::Step(StepIR {
        name: name.to_string(),
        node: node.to_string(),
        args,
    })
}

/// Build a positional arg referencing an identifier.
#[allow(dead_code)]
fn pos_ident(name: &str) -> StepArgIR {
    StepArgIR::Positional {
        value: ExprIR::Ident {
            name: name.to_string(),
        },
    }
}

/// Build a named arg referencing an identifier.
fn named_ident(name: &str, ident: &str) -> StepArgIR {
    StepArgIR::Named {
        name: name.to_string(),
        value: ExprIR::Ident {
            name: ident.to_string(),
        },
    }
}

/// Build a named arg referencing a field access (e.g., input.labels).
#[allow(dead_code)]
fn named_field(name: &str, base: &str, field: &str) -> StepArgIR {
    StepArgIR::Named {
        name: name.to_string(),
        value: ExprIR::FieldAccess {
            base: Box::new(ExprIR::Ident {
                name: base.to_string(),
            }),
            field: field.to_string(),
        },
    }
}

// ── ShortlistSelect ──
// Before: step result = classify(input)
// After:
//   step shortlist = shortlist_labels(task_text: input.task_text, labels: input.labels)
//   step raw = choose_label(task_text: input.task_text, candidates: shortlist)
//   step result = normalize_label(raw: raw, labels: input.labels)

fn apply_shortlist_select(
    graph: &GraphIR,
    target_step: &StepIR,
    step_idx: usize,
    config: &HashMap<String, serde_json::Value>,
) -> Result<MotifApplication, String> {
    let k = config
        .get("shortlist_k")
        .and_then(|v| v.as_u64())
        .unwrap_or(3);

    // Generate unique names
    let shortlist_node = format!("_motif.{}.shortlist", target_step.name);
    let choose_node = format!("_motif.{}.choose", target_step.name);
    let normalize_node = format!("_motif.{}.normalize", target_step.name);

    let shortlist_step = make_step(
        &format!("{}_shortlist", target_step.name),
        &shortlist_node,
        std::iter::once(named_ident("input", "input"))
            .chain(target_step.args.iter().cloned())
            .collect(),
    );

    let choose_step = make_step(
        &format!("{}_choose", target_step.name),
        &choose_node,
        vec![
            named_ident("input", "input"),
            named_ident(
                "candidates",
                &format!("{}_shortlist", target_step.name),
            ),
        ]
        .into_iter()
        .chain(target_step.args.iter().cloned())
        .collect(),
    );

    let normalize_step = make_step(
        &target_step.name,
        &normalize_node,
        vec![
            named_ident("input", "input"),
            named_ident("raw", &format!("{}_choose", target_step.name)),
        ]
        .into_iter()
        .chain(
            target_step
                .args
                .iter()
                .filter(|a| matches!(a, StepArgIR::Named { name, .. } if name != "task_text"))
                .cloned(),
        )
        .collect(),
    );

    // Build new body: replace target step with the 3-step sequence
    let mut new_body = graph.body.clone();
    new_body.splice(step_idx..=step_idx, vec![shortlist_step, choose_step, normalize_step]);

    let shortlist_tmpl = resolve_template(config, "shortlist", SHORTLIST_TEMPLATE)
        .replace("{{k}}", &k.to_string());
    let choose_tmpl = resolve_template(config, "choose", CHOOSE_TEMPLATE);
    let normalize_tmpl = resolve_template(config, "normalize", NORMALIZE_TEMPLATE);

    Ok(MotifApplication {
        new_body,
        synthetic_nodes: vec![
            SyntheticNode {
                name: shortlist_node.clone(),
                kind: NodeKindIR::Prompt,
                template: shortlist_tmpl,
                system: None,
                shell: None,
                tool_spec_json: None,
            },
            SyntheticNode {
                name: choose_node.clone(),
                kind: NodeKindIR::Prompt,
                template: choose_tmpl,
                system: None,
                shell: None,
                tool_spec_json: None,
            },
            SyntheticNode {
                name: normalize_node.clone(),
                kind: NodeKindIR::Prompt,
                template: normalize_tmpl,
                system: Some(TERMINAL_SYSTEM.to_string()),
                shell: None,
                tool_spec_json: None,
            },
        ],
        local_checkers: vec![],
    })
}

// ── NormalizeVerify ──
// Before: step result = classify(input)
// After:
//   step raw = classify(input)
//   step result = normalize_label(raw: raw, labels: input.labels)

fn apply_normalize_verify(
    graph: &GraphIR,
    target_step: &StepIR,
    step_idx: usize,
    config: &HashMap<String, serde_json::Value>,
    _ir: &ScaffoldIR,
) -> Result<MotifApplication, String> {
    let normalize_node = format!("_motif.{}.normalize", target_step.name);

    // Rename original step to raw
    let raw_step_name = format!("{}_raw", target_step.name);
    let raw_step = GraphStmtIR::Step(StepIR {
        name: raw_step_name.clone(),
        node: target_step.node.clone(),
        args: target_step.args.clone(),
    });

    let normalize_step = make_step(
        &target_step.name,
        &normalize_node,
        vec![named_ident("input", "input"), named_ident("raw", &raw_step_name)]
            .into_iter()
            .chain(
                target_step
                    .args
                    .iter()
                    .filter(|a| matches!(a, StepArgIR::Named { name, .. } if name != "task_text"))
                    .cloned(),
            )
            .collect(),
    );

    let mut new_body = graph.body.clone();
    new_body.splice(step_idx..=step_idx, vec![raw_step, normalize_step]);

    let normalize_tmpl = resolve_template(config, "normalize", NORMALIZE_TEMPLATE);

    Ok(MotifApplication {
        new_body,
        synthetic_nodes: vec![SyntheticNode {
            name: normalize_node,
            kind: NodeKindIR::Prompt,
            template: normalize_tmpl,
            system: Some(TERMINAL_SYSTEM.to_string()),
            shell: None,
            tool_spec_json: None,
        }],
        local_checkers: vec![],
    })
}

// ── RouterExpert ──
// Before: step result = classify(input)
// After:
//   step route = router(task_text: input.task_text)
//   step result = classify(input)  [unchanged, but router output is available for context]

fn apply_router_expert(
    graph: &GraphIR,
    target_step: &StepIR,
    step_idx: usize,
    config: &HashMap<String, serde_json::Value>,
) -> Result<MotifApplication, String> {
    let router_node = format!("_motif.{}.router", target_step.name);

    let router_step = make_step(
        &format!("{}_route", target_step.name),
        &router_node,
        std::iter::once(named_ident("input", "input"))
            .chain(target_step.args.iter().cloned())
            .collect(),
    );

    // Original step gets the router output as an extra named arg
    let mut augmented_args = target_step.args.clone();
    augmented_args.push(named_ident("domain", &format!("{}_route", target_step.name)));

    let result_step = GraphStmtIR::Step(StepIR {
        name: target_step.name.clone(),
        node: target_step.node.clone(),
        args: augmented_args,
    });

    let mut new_body = graph.body.clone();
    new_body.splice(step_idx..=step_idx, vec![router_step, result_step]);

    let router_tmpl = resolve_template(config, "router", ROUTER_TEMPLATE);

    Ok(MotifApplication {
        new_body,
        synthetic_nodes: vec![SyntheticNode {
            name: router_node,
            kind: NodeKindIR::Prompt,
            template: router_tmpl,
            system: None,
            shell: None,
            tool_spec_json: None,
        }],
        local_checkers: vec![],
    })
}

// ── RetrieveDecide ──
// Before: step result = classify(input)
// After:
//   step retrieve = retriever(task_text: input.task_text)
//   step result = decider(facts: retrieve.facts, context: retrieve.context, ...)

fn apply_retrieve_decide(
    graph: &GraphIR,
    target_step: &StepIR,
    step_idx: usize,
    config: &HashMap<String, serde_json::Value>,
) -> Result<MotifApplication, String> {
    let retrieve_node = format!("_motif.{}.retrieve", target_step.name);
    let decide_node = format!("_motif.{}.decide", target_step.name);

    let retrieve_step = make_step(
        &format!("{}_retrieve", target_step.name),
        &retrieve_node,
        std::iter::once(named_ident("input", "input"))
            .chain(target_step.args.iter().cloned())
            .collect(),
    );

    let retrieve_step_name = format!("{}_retrieve", target_step.name);
    let decide_step = make_step(
        &target_step.name,
        &decide_node,
        vec![
            named_ident("input", "input"),
            named_ident("retrieved", &retrieve_step_name),
        ]
        .into_iter()
        .chain(target_step.args.iter().cloned())
        .collect(),
    );

    let mut new_body = graph.body.clone();
    new_body.splice(step_idx..=step_idx, vec![retrieve_step, decide_step]);

    let retrieve_tmpl = resolve_template(config, "retrieve", RETRIEVE_TEMPLATE);
    let decide_tmpl = resolve_template(config, "decide", DECIDE_TEMPLATE);

    Ok(MotifApplication {
        new_body,
        synthetic_nodes: vec![
            SyntheticNode {
                name: retrieve_node,
                kind: NodeKindIR::Prompt,
                template: retrieve_tmpl,
                system: None,
                shell: None,
                tool_spec_json: None,
            },
            SyntheticNode {
                name: decide_node,
                kind: NodeKindIR::Prompt,
                template: decide_tmpl,
                system: Some(TERMINAL_SYSTEM.to_string()),
                shell: None,
                tool_spec_json: None,
            },
        ],
        local_checkers: vec![],
    })
}

// ── VoteCritiqueRepair ──
// Before: step result = classify(input)
// After:
//   step proposed = classify(input)
//   step critique = critic(proposed: proposed, ...)
//   step result = repairer(proposed: proposed, critique: critique, ...)

fn apply_vote_critique_repair(
    graph: &GraphIR,
    target_step: &StepIR,
    step_idx: usize,
    config: &HashMap<String, serde_json::Value>,
) -> Result<MotifApplication, String> {
    let critique_node = format!("_motif.{}.critique", target_step.name);
    let repair_node = format!("_motif.{}.repair", target_step.name);

    let proposed_step_name = format!("{}_proposed", target_step.name);
    let proposed_step = GraphStmtIR::Step(StepIR {
        name: proposed_step_name.clone(),
        node: target_step.node.clone(),
        args: target_step.args.clone(),
    });

    let critique_step_name = format!("{}_critique", target_step.name);
    let critique_step = make_step(
        &critique_step_name,
        &critique_node,
        vec![
            named_ident("input", "input"),
            named_ident("proposed", &proposed_step_name),
        ]
            .into_iter()
            .chain(target_step.args.iter().cloned())
            .collect(),
    );

    let repair_step = make_step(
        &target_step.name,
        &repair_node,
        vec![
            named_ident("input", "input"),
            named_ident("proposed", &proposed_step_name),
            named_ident("critique", &critique_step_name),
        ]
        .into_iter()
        .chain(target_step.args.iter().cloned())
        .collect(),
    );

    let mut new_body = graph.body.clone();
    new_body.splice(
        step_idx..=step_idx,
        vec![proposed_step, critique_step, repair_step],
    );

    let critique_tmpl = resolve_template(config, "critique", CRITIQUE_TEMPLATE);

    // Build ToolSpec for repair gate: passes proposed as argv arg, critique via stdin.
    // This avoids shell quoting issues when LLM output contains special characters.
    let repair_tool_spec = serde_json::json!({
        "argv": ["python3", "-c", REPAIR_GATE_SCRIPT, "{{proposed}}"],
        "stdin_template": "{{critique}}",
        "workdir": ".",
        "timeout": 10,
        "net": false,
        "mounts": []
    });

    Ok(MotifApplication {
        new_body,
        synthetic_nodes: vec![
            SyntheticNode {
                name: critique_node,
                kind: NodeKindIR::Prompt,
                template: critique_tmpl,
                system: None,
                shell: None,
                tool_spec_json: None,
            },
            SyntheticNode {
                name: repair_node,
                kind: NodeKindIR::Tool,
                template: String::new(),
                system: None,
                shell: None,
                tool_spec_json: Some(repair_tool_spec),
            },
        ],
        local_checkers: vec![],
    })
}

// ── GenerateValidateRefine ──
// Before: step result = solve(input)
// After:
//   step proposed = solve(input)                          // original node, unchanged
//   step validated = _motif.result.validate(proposed, input, ...)  // tool node
//   step result = _motif.result.refine(proposed, feedback=validated, input, ...)  // prompt node

fn apply_generate_validate_refine(
    graph: &GraphIR,
    target_step: &StepIR,
    step_idx: usize,
    config: &HashMap<String, serde_json::Value>,
) -> Result<MotifApplication, String> {
    let validate_shell = config
        .get("validate_shell")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| VALIDATE_TOOL_TEMPLATE.to_string());

    let validate_timeout = config
        .get("validate_timeout")
        .and_then(|v| v.as_u64())
        .unwrap_or(30);

    let validate_node = format!("_motif.{}.validate", target_step.name);
    let refine_node = format!("_motif.{}.refine", target_step.name);

    // proposed: rename original step
    let proposed_step_name = format!("{}_proposed", target_step.name);
    let proposed_step = GraphStmtIR::Step(StepIR {
        name: proposed_step_name.clone(),
        node: target_step.node.clone(),
        args: target_step.args.clone(),
    });

    // validate: tool node receiving proposed + original args + input
    let validate_step_name = format!("{}_validated", target_step.name);
    let validate_step = make_step(
        &validate_step_name,
        &validate_node,
        vec![
            named_ident("input", "input"),
            named_ident("proposed", &proposed_step_name),
        ]
        .into_iter()
        .chain(target_step.args.iter().cloned())
        .collect(),
    );

    // refine: prompt node receiving proposed + feedback + original args + input
    let refine_step = make_step(
        &target_step.name,
        &refine_node,
        vec![
            named_ident("input", "input"),
            named_ident("proposed", &proposed_step_name),
            named_ident("feedback", &validate_step_name),
        ]
        .into_iter()
        .chain(target_step.args.iter().cloned())
        .collect(),
    );

    let mut new_body = graph.body.clone();
    new_body.splice(
        step_idx..=step_idx,
        vec![proposed_step, validate_step, refine_step],
    );

    // Validate node template is just a placeholder — the shell command is what matters
    let validate_tmpl = resolve_template(config, "validate", &validate_shell);
    let refine_tmpl = resolve_template(config, "refine", REFINE_TEMPLATE);

    Ok(MotifApplication {
        new_body,
        synthetic_nodes: vec![
            SyntheticNode {
                name: validate_node,
                kind: NodeKindIR::Tool,
                template: validate_tmpl,
                system: None,
                shell: Some(format!("timeout {} {}", validate_timeout, validate_shell)),
                tool_spec_json: None,
            },
            SyntheticNode {
                name: refine_node,
                kind: NodeKindIR::Prompt,
                template: refine_tmpl,
                system: Some(TERMINAL_SYSTEM.to_string()),
                shell: None,
                tool_spec_json: None,
            },
        ],
        local_checkers: vec![],
    })
}

/// Describe what each motif does, for meta-agent context.
pub fn motif_description(motif: &Motif) -> &'static str {
    match motif {
        Motif::NormalizeVerify => {
            "Append a normalizer node after the target step. Fixes spelling, case, and alias \
             variants. Best when failures are due to invalid format or label not in allowed set."
        }
        Motif::ShortlistSelect => {
            "Replace one step with shortlist→choose→normalize pipeline. Narrows label space \
             before final selection. Best when errors are confusion among nearby labels."
        }
        Motif::RouterExpert => {
            "Insert a router before the target step that classifies input domain. The domain \
             output is passed as extra context. Best when clear domain clusters exist."
        }
        Motif::RetrieveDecide => {
            "Insert a retrieval step before decision. Extracts key facts first, then decides. \
             Best when model lacks external knowledge needed for the task."
        }
        Motif::VoteCritiqueRepair => {
            "Run the target step, critique its output, then repair. Creates a closed-loop \
             error correction cycle. Best when output variance is high."
        }
        Motif::GenerateValidateRefine => {
            "Run the target step, validate with a tool (shell command), then refine based on \
             tool feedback. The validate step is a tool node — deterministic, no hallucination. \
             Best when correctness can be checked programmatically (tests, format checks, APIs)."
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_graph() -> GraphIR {
        GraphIR {
            name: "test".to_string(),
            input: TypeIR::Struct {
                fields: vec![
                    FieldIR {
                        name: "task_text".to_string(),
                        ty: TypeIR::String,
                    },
                    FieldIR {
                        name: "labels".to_string(),
                        ty: TypeIR::String,
                    },
                ],
            },
            output: TypeIR::String,
            body: vec![GraphStmtIR::Step(StepIR {
                name: "result".to_string(),
                node: "classify".to_string(),
                args: vec![pos_ident("input")],
            })],
        }
    }

    fn make_test_ir() -> ScaffoldIR {
        ScaffoldIR {
            nodes: vec![NodeIR {
                name: "classify".to_string(),
                kind: NodeKindIR::Prompt,
                input: TypeIR::String,
                output: TypeIR::String,
                config: NodeConfigIR::default(),
            }],
            ..Default::default()
        }
    }

    #[test]
    fn test_shortlist_select_produces_3_steps() {
        let graph = make_test_graph();
        let ir = make_test_ir();
        let config = HashMap::new();
        let result = apply_motif(&Motif::ShortlistSelect, &graph, "result", &config, &ir).unwrap();
        // Original had 1 step, now should have 3
        assert_eq!(result.new_body.len(), 3);
        assert_eq!(result.synthetic_nodes.len(), 3);
    }

    #[test]
    fn test_normalize_verify_produces_2_steps() {
        let graph = make_test_graph();
        let ir = make_test_ir();
        let config = HashMap::new();
        let result =
            apply_motif(&Motif::NormalizeVerify, &graph, "result", &config, &ir).unwrap();
        assert_eq!(result.new_body.len(), 2);
        assert_eq!(result.synthetic_nodes.len(), 1);
    }

    #[test]
    fn test_vote_critique_repair_produces_3_steps() {
        let graph = make_test_graph();
        let ir = make_test_ir();
        let config = HashMap::new();
        let result =
            apply_motif(&Motif::VoteCritiqueRepair, &graph, "result", &config, &ir).unwrap();
        assert_eq!(result.new_body.len(), 3);
        assert_eq!(result.synthetic_nodes.len(), 2);
    }

    #[test]
    fn test_router_expert_produces_2_steps() {
        let graph = make_test_graph();
        let ir = make_test_ir();
        let config = HashMap::new();
        let result = apply_motif(&Motif::RouterExpert, &graph, "result", &config, &ir).unwrap();
        assert_eq!(result.new_body.len(), 2);
        assert_eq!(result.synthetic_nodes.len(), 1);
    }

    #[test]
    fn test_retrieve_decide_produces_2_steps() {
        let graph = make_test_graph();
        let ir = make_test_ir();
        let config = HashMap::new();
        let result =
            apply_motif(&Motif::RetrieveDecide, &graph, "result", &config, &ir).unwrap();
        assert_eq!(result.new_body.len(), 2);
        assert_eq!(result.synthetic_nodes.len(), 2);
    }

    #[test]
    fn test_generate_validate_refine_produces_3_steps() {
        let graph = make_test_graph();
        let ir = make_test_ir();
        let mut config = HashMap::new();
        config.insert("validate_shell".to_string(), serde_json::json!("python3 -c 'print(\"PASS\")'"));
        let result =
            apply_motif(&Motif::GenerateValidateRefine, &graph, "result", &config, &ir).unwrap();
        assert_eq!(result.new_body.len(), 3);
        assert_eq!(result.synthetic_nodes.len(), 2);
        // First synthetic node should be a tool node with shell
        assert_eq!(result.synthetic_nodes[0].kind, NodeKindIR::Tool);
        assert!(result.synthetic_nodes[0].shell.is_some());
        // Second should be a prompt node (refine)
        assert_eq!(result.synthetic_nodes[1].kind, NodeKindIR::Prompt);
        assert!(result.synthetic_nodes[1].shell.is_none());
        // Last step should still be named "result"
        if let GraphStmtIR::Step(s) = &result.new_body[2] {
            assert_eq!(s.name, "result");
        } else {
            panic!("expected step");
        }
    }

    #[test]
    fn test_missing_step_returns_error() {
        let graph = make_test_graph();
        let ir = make_test_ir();
        let config = HashMap::new();
        let result = apply_motif(&Motif::ShortlistSelect, &graph, "nonexistent", &config, &ir);
        assert!(result.is_err());
    }

    #[test]
    fn test_shortlist_preserves_step_name() {
        let graph = make_test_graph();
        let ir = make_test_ir();
        let config = HashMap::new();
        let result = apply_motif(&Motif::ShortlistSelect, &graph, "result", &config, &ir).unwrap();
        // The last step should still be named "result" to preserve downstream references
        if let GraphStmtIR::Step(s) = &result.new_body[2] {
            assert_eq!(s.name, "result");
        } else {
            panic!("expected step");
        }
    }

    #[test]
    fn test_motif_serde_roundtrip() {
        let motif = Motif::ShortlistSelect;
        let json = serde_json::to_string(&motif).unwrap();
        assert_eq!(json, "\"shortlist_select\"");
        let parsed: Motif = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, motif);

        let motif2 = Motif::GenerateValidateRefine;
        let json2 = serde_json::to_string(&motif2).unwrap();
        assert_eq!(json2, "\"generate_validate_refine\"");
        let parsed2: Motif = serde_json::from_str(&json2).unwrap();
        assert_eq!(parsed2, motif2);
    }
}
