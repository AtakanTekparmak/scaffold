//! LLM-guided meta-agent for proposing targeted mutations (Algorithm 2, DGM-H).
//!
//! Instead of random dice rolls, the meta-agent receives rich context about the
//! parent candidate, archive history, and available nodes, then proposes a
//! structured mutation via an LLM call.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::io::Write;
use std::path::PathBuf;

use scaffold_ir::ir::*;

use crate::error::{Error, Result};
use crate::llm::{self, LlmConfig};
use crate::mutations::Mutation;

use crate::optimizer::{Archive, ArchiveEntry, CandidateDelta, EvalResults};

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
        parent: &CandidateDelta,
        parent_eval: &EvalResults,
        archive: &Archive,
        ir: &ScaffoldIR,
        objective: &ObjectiveIR,
        allowed_mutations: &[String],
        epoch_start_id: Option<usize>,
        meta_full_traces: bool,
        online: bool,
    ) -> Result<MutationProposal> {
        const MAX_RETRIES: usize = 10;

        let mut base_context = build_context(parent, parent_eval, archive, ir, objective, allowed_mutations, epoch_start_id, meta_full_traces);
        if online {
            base_context.push_str(r#"
## Network Access — ENABLED

Tool nodes CAN access the internet. This is a major advantage — USE IT. The filesystem sandbox restricts access to the working directory, but network calls (curl, wget, python requests/urllib) are fully available.

**You should strongly consider creating tool nodes that leverage the internet**, especially when LLM-only approaches are plateauing. Concrete strategies:

1. **Reference fetching**: fetch Wikipedia summaries, documentation, or domain knowledge to augment LLM context
   - `node fetch_info: tool { in: { query: string } out: string shell: "curl -sL 'https://en.wikipedia.org/api/rest_v1/page/summary/{{query}}' | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get(\"extract\",\"\"))'" }`

2. **External classification APIs**: call specialized APIs for domain-specific tasks
   - `node web_search: tool { in: { text: string } out: string shell: "python3 -c 'import urllib.request,urllib.parse,json; q=urllib.parse.quote(sys.argv[1][:200]); ...' '{{text}}'" }`

3. **Library-powered tools**: use python libraries (scikit-learn, nltk, etc.) for feature extraction, text analysis, or rule-based pre-classification
   - `node detect_domain: tool { in: { text: string } out: string shell: "python3 -c 'import sys; t=sys.argv[1]; print(\"legal\" if any(c > chr(0x4e00) for c in t) else \"medical\" if any(w in t.lower() for w in [\"symptom\",\"pain\",\"fever\"]) else \"chemistry\")' '{{text}}'" }`

4. **Output validation with fuzzy matching**: deterministically match LLM output to the closest valid label
   - `node match_label: tool { in: { prediction: string, labels: string } out: string shell: "python3 -c 'import sys,difflib; pred=sys.argv[1].strip(); labels=[l.strip(\"- \") for l in sys.argv[2].split(chr(10)) if l.strip(\"- \")]; m=difflib.get_close_matches(pred,labels,n=1,cutoff=0.3); print(m[0] if m else pred)' '{{prediction}}' '{{labels}}'" }`

Combine these with LLM nodes: use a tool to pre-process or fetch context, feed the result to an LLM for reasoning, then use another tool to validate/normalize the output.
"#);
        }
        let system_chars = SYSTEM_PROMPT.len();
        let llm_config = LlmConfig::new()
            .with_model(&self.model)
            .with_temperature(0.7)
            .with_system_prompt(SYSTEM_PROMPT);

        // Log base context on first call (system prompt is constant, log it once)
        self.log_section("SYSTEM PROMPT", SYSTEM_PROMPT);
        self.log_section("BASE CONTEXT", &base_context);

        let saturated_labels = compute_saturated_families(archive, epoch_start_id.unwrap_or(0));

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

                    // Hard mask: reject saturated mutation families
                    if saturated_labels.iter().any(|sat| label == *sat) {
                        let msg = format!(
                            "SATURATED: '{}' is off-limits (no improvement in recent attempts). Choose a fundamentally different mutation.",
                            label,
                        );
                        self.log_section("REJECTED (saturated)", &msg);
                        errors.push(msg);
                        continue;
                    }

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

/// Structural mutation kinds (used for permission checks).
const STRUCTURAL_MUTATIONS: &[&str] = &[
    "insert_verify",
    "wrap_retry",
    "insert_step",
    "remove_step",
    "replace_component",
    "fan_out",
    "replace_with_subgraph",
    "add_prompt_step",
    "propose_decomposition",
];

const SYSTEM_PROMPT: &str = r#"You are a meta-optimizer for computational graphs. Your goal: propose a single mutation that maximizes score improvement on a dataset evaluation while minimizing regression on already-passing cases.

You receive rich context — pass/fail matrix, mutation history, candidate lineage, failure evidence, and node definitions. Use all of it.

Propose exactly ONE mutation as a JSON object:
- "kind": a mutation type from "Mutation Reference"
- "reasoning": why this mutation should help
- Plus kind-specific fields

RULES:

1. Diagnose before acting. Study the failure evidence — checker output, step traces, repair behavior tags — to understand WHY cases fail before choosing WHAT to change.

2. Never repeat what doesn't work. If a mutation type+target is marked "saturated" or showed zero/negative improvement across multiple attempts, it is off-limits. Choose a fundamentally different approach.

3. Prefer structural diversity. Avoid repeating the same graph topology when 2+ candidates already share the same structure. When the "Content Plateau" indicator says PLATEAUED, structural change is strongly recommended. Different means different control flow — loops instead of nested ifs, parallel blocks instead of sequential chains, verify gates, new tool nodes, planning steps.

4. Explore before exploiting. If the last 3+ mutations targeted the same node or used the same strategy, switch to a different lever: a different node, a different mutation kind, or a different structural pattern. Diminishing returns are real.

5. Preserve what works. A hard regression gate protects cases that any previous best candidate passed — a mutation CANNOT produce a new best if it fails any protected case, even with a higher overall score. Target always-failing cases; never risk flipping protected P→F cases.

6. Content mutations (rewrite_prompt, rewrite_system, rewrite_shell, rewrite_tool_spec, set_config) change a single node's behavior. Use when the graph structure is sound but a node's output is wrong — format, constraints, instructions.

7. Structural mutations change the graph's topology. Use propose_decomposition for decomposition patterns, or individual structural mutations (insert_step, wrap_retry, insert_verify, etc.) for targeted changes.

8. **Use propose_decomposition for adding structure.** When the "Suggested decompositions" section recommends a motif, use propose_decomposition. Motifs produce validated, tested graph transformations. Available motifs:
   - **normalize_verify**: append a normalizer node that fixes spelling/case/alias. Use when failures are invalid format or label not in allowed set. normalize_verify is ONLY for format errors (wrong case, truncated, misspelled). It CANNOT fix semantic confusion where the model picks a valid but wrong label.
   - **shortlist_select**: replace one step with shortlist→choose→normalize. Narrows label space before selection. Use when errors are confusion among nearby labels.
   - **router_expert**: insert a domain router before the step. Use when clear domain clusters exist in the data.
   - **retrieve_decide**: insert a fact extraction step before decision. Use when the model lacks external knowledge.
   - **vote_critique_repair**: run the step, critique output, then repair. Creates a closed-loop error correction cycle. Use when output variance is high or the model picks wrong-but-related answers.
   - **generate_validate_refine**: run the step, validate with a tool (shell command), then refine based on tool feedback. The validate step is a tool node — deterministic, no hallucination. Use when correctness can be checked programmatically (run tests, check format, call a validator API). Requires `config.validate_shell` (shell command for tool validation). Optional: `config.validate_timeout` (seconds, default 30).
   Motif templates are task-generic by default (use {{input}}). To specialize them for your task, provide `config.templates` with task-specific prompt text. Template keys per motif:
   - **shortlist_select**: `shortlist`, `choose`, `normalize`
   - **normalize_verify**: `normalize`
   - **router_expert**: `router`
   - **retrieve_decide**: `retrieve`, `decide`
   - **vote_critique_repair**: `critique` (the repair gate is a deterministic tool node — do NOT provide a template for `repair_gate`). The critique template should produce either JSON `{"verdict":"keep"|"change","better_label":"...","confidence":0.0-1.0}` or text `CORRECT` / `WRONG - [corrected label]`. Both formats are accepted by the repair gate.
   - **generate_validate_refine**: `validate`, `refine`
   **Template variables available in motif nodes:**
   - Motif-specific: {{proposed}}, {{critique}}, {{candidates}}, {{raw}}, {{retrieved}}, {{feedback}} (per role)
   - Graph input: {{input}} (full graph input struct — use {{input.field_name}} for specific fields)
   - All original step arguments are forwarded directly: e.g. if the target step receives `task_text` and `label_guide`, use {{task_text}} and {{label_guide}} in templates.
   Check "Steps in Parent Graph" to see which arguments the target step receives.
   After applying a motif, follow up with rewrite_prompt on the generated `_motif.*` nodes to further refine their templates.

9. Tool nodes are first-class citizens — use them proactively, not as a last resort. Tool nodes run shell commands and produce deterministic, reproducible results. Unlike LLM nodes, they never hallucinate. Use them for:
   - **Pre-processing**: extract structure, normalize data, detect domain/language before LLM classification
   - **Post-processing**: validate LLM output against allowed values, fuzzy-match labels, fix formatting
   - **Computation**: count items, measure similarity, apply rules that an LLM would get wrong
   - **Orchestration**: route inputs to different strategies based on deterministic analysis
   Tool nodes use shell commands with template variables ({{var}}). Use `python3 -c '...'` for complex logic, shell pipelines for composition, and heredocs for multi-line stdin.
   Example: `node normalize: tool { in: { label: string, allowed: string } out: string shell: "python3 -c 'import sys,difflib; label=sys.argv[1]; allowed=sys.argv[2].split(chr(10)); m=difflib.get_close_matches(label.strip(),allowed,n=1,cutoff=0.5); print(m[0] if m else label)' '{{label}}' '{{allowed}}'" }`
   If the "Network Access" section is present in context, tool nodes can also access the internet — this is an extremely powerful capability. Check context for details.

10. Node type constraints: rewrite_prompt and rewrite_system apply ONLY to prompt/agent/verify nodes. rewrite_shell and rewrite_tool_spec apply ONLY to tool nodes. Prefer rewrite_tool_spec over rewrite_shell for new tool commands — it avoids shell quoting issues and enables structured sandboxing. Check "Available Nodes" for types.

11. For rewrite_prompt, provide the COMPLETE template including all {{ variable }} references from the original. You may change prose, formatting, and structure freely.

12. Start lightweight. Before complex structural changes (propose_decomposition with multi-node motifs, fan_out, replace_with_subgraph), try add_local_checker, normalize_verify, or attach_example_policy. These are cheap, composable, and often sufficient.

13. Write generalizable prompts. The failure evidence shows training-set errors — use them to understand the KIND of mistake, not to hard-code fixes for specific cases. Do NOT embed specific label pairs (e.g. "distinguish X from Y"), case IDs, or error examples from the failure evidence into prompt rewrites. Instead, write instructions that teach the model the general principle (e.g. "read the full text before classifying" rather than "if you see symptoms X, output disease Y").

14. Prefer generalizable example strategies. When using attach_example_policy, prefer `domain_conditioned` (selects examples matching the input's domain) or `nearest` (selects by input similarity) over `confusion_cover` (which overfits to training-set confusion pairs). Use confusion_cover only when domain_conditioned has already been tried.

15. Consider simplifying before adding complexity. If the base classifier is strong (within 5% of best score without critique/repair), the critique loop may be adding noise rather than value. Use remove_step to ablate the critique pipeline and test whether the simpler graph performs equally well. Complexity must earn its keep.

Return ONLY a single JSON object. No markdown, no explanation outside the JSON."#;

/// Per-mutation-kind documentation. Keys are the kind strings used in proposals.
fn mutation_doc(kind: &str) -> Option<&'static str> {
    match kind {
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
        "rewrite_tool_spec" => Some(
            r#"{"kind":"rewrite_tool_spec","node":"<tool_node>","spec":{"argv":["cmd","arg1","{{var}}"],"stdin_template":"optional stdin with {{var}}","workdir":".","timeout":30,"net":false,"mounts":[]},"reasoning":"..."} — structured tool specification. Each argv element is a template with {{var}} references. stdin_template is optional. net=true allows network access. Preferred over rewrite_shell for new tool commands."#,
        ),
        "propose_decomposition" => Some(
            r#"{"kind":"propose_decomposition","target_step":"<step>","motif":"<motif_name>","reason":"<diagnosis>","config":{"shortlist_k":3,"templates":{"<key>":"<template text>"},"validate_shell":"<shell cmd>","validate_timeout":30},"reasoning":"..."} — decompose a step using a motif from the library. Available motifs: normalize_verify, shortlist_select, router_expert, retrieve_decide, vote_critique_repair, generate_validate_refine. Config keys: shortlist_k (for shortlist_select), validate_shell (for generate_validate_refine, required), validate_timeout (for generate_validate_refine, default 30s), templates (optional task-specific templates). Template keys per motif: shortlist_select uses "shortlist","choose","normalize"; normalize_verify uses "normalize"; router_expert uses "router"; retrieve_decide uses "retrieve","decide"; vote_critique_repair uses "critique" (repair gate accepts both JSON {"verdict":"keep"|"change","better_label":"..."} and text CORRECT / WRONG - [label]); generate_validate_refine uses "validate","refine". Template variables: (1) motif-specific: {{proposed}}, {{critique}}, {{candidates}}, {{raw}}, {{retrieved}}, {{feedback}}; (2) graph input: {{input}} (full input object, use {{input.field_name}} for fields); (3) all original step arguments are forwarded as named variables (e.g. if step receives task_text and label_guide, use {{task_text}}, {{label_guide}})."#,
        ),
        "attach_example_policy" => Some(
            r#"{"kind":"attach_example_policy","node":"<node>","policy":{"strategy":"<strategy>","k":<n>},"reasoning":"..."} — attach few-shot examples to a node. Strategies: confusion_cover, nearest_plus_hard_negative, synthetic_aliases, random, domain_conditioned. domain_conditioned prefers same-domain examples when the input has a "domain" or "category" field. Examples are selected from the example bank at execution time."#,
        ),
        "add_local_checker" => Some(
            r#"{"kind":"add_local_checker","node":"<node>","checker_name":"<name>","expr":"<expression>","reasoning":"..."} — add a validation expression evaluated after the node runs. Uses the same expression syntax as objective checkers."#,
        ),
        _ => None,
    }
}

/// Estimate the total context size (in chars) that would be sent to the meta-agent.
pub fn estimate_context_chars(
    parent: &CandidateDelta,
    parent_eval: &EvalResults,
    archive: &Archive,
    ir: &ScaffoldIR,
    objective: &ObjectiveIR,
    allowed_mutations: &[String],
    epoch_start_id: Option<usize>,
    meta_full_traces: bool,
) -> usize {
    // Building the full string is cheap (no I/O), so just measure it directly.
    let ctx = build_context(parent, parent_eval, archive, ir, objective, allowed_mutations, epoch_start_id, meta_full_traces);
    ctx.len() + SYSTEM_PROMPT.len()
}

/// Build the context string sent to the meta-agent LLM.
///
/// When `epoch_start_id` is set, only candidates with `id >= epoch_start_id` are shown
/// in the archive, pass/fail matrix, and mutation effects sections. This keeps context
/// bounded across long runs while the TUI still shows full lineage.
fn build_context(
    parent: &CandidateDelta,
    parent_eval: &EvalResults,
    archive: &Archive,
    ir: &ScaffoldIR,
    objective: &ObjectiveIR,
    allowed_mutations: &[String],
    epoch_start_id: Option<usize>,
    meta_full_traces: bool,
) -> String {
    let mut ctx = String::with_capacity(4096);
    let epoch_start = epoch_start_id.unwrap_or(0);

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
    let has_synthetic_tool = parent.overrides.iter().any(|(k, v)| {
        k.starts_with("_node.") && v.get("kind").and_then(|k| k.as_str()) == Some("tool")
    });
    let has_tool_nodes = has_synthetic_tool || ir.nodes.iter().any(|n| n.kind == NodeKindIR::Tool);
    let has_prompt_nodes = ir
        .nodes
        .iter()
        .any(|n| matches!(n.kind, NodeKindIR::Prompt | NodeKindIR::Agent));
    let has_any_structural = allowed_mutations
        .iter()
        .any(|m| STRUCTURAL_MUTATIONS.contains(&m.as_str()));
    let mut all_available: Vec<String> = allowed_mutations.to_vec();
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
        if !all_available.iter().any(|m| m == "rewrite_tool_spec") {
            all_available.push("rewrite_tool_spec".to_string());
        }
    }
    // Motif decomposition: available when structural mutations are allowed
    if has_any_structural {
        if !all_available.iter().any(|m| m == "propose_decomposition") {
            all_available.push("propose_decomposition".to_string());
        }
    }
    // Content-level mutations always available
    for content in &["attach_example_policy", "add_local_checker"] {
        if !all_available.iter().any(|m| m == content) {
            all_available.push(content.to_string());
        }
    }
    ctx.push_str(&format!(
        "Allowed mutations: {}\n",
        all_available.join(", ")
    ));
    if !has_tool_nodes {
        ctx.push_str("Note: No tool nodes exist — rewrite_shell/rewrite_tool_spec are NOT available.\n");
    }

    // Mutation reference: only show docs for mutations actually available
    ctx.push_str("\n## Mutation Reference\n");
    for kind in &all_available {
        if let Some(doc) = mutation_doc(kind) {
            ctx.push_str(&format!("- {}\n", doc));
        }
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
    ctx.push_str(&format!("Score: {:.4}\n", parent_eval.score().unwrap_or(0.0)));
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

    let full_lineage = candidate_lineage(archive, parent);
    // Truncate lineage at epoch boundary — pre-epoch ancestors collapsed to one line
    let lineage: Vec<&ArchiveEntry> = if epoch_start_id.is_some() {
        full_lineage.iter().copied().filter(|e| e.delta.id >= epoch_start).collect()
    } else {
        full_lineage.clone()
    };
    if !lineage.is_empty() {
        ctx.push_str("## Selected Candidate Lineage\n");
        ctx.push_str("Exact causal path from seed to the selected candidate. Use this to reason about what changed locally.\n\n");
        // If lineage was truncated, show a one-line summary of the epoch baseline
        if epoch_start_id.is_some() && full_lineage.len() > lineage.len() {
            if let Some(first) = lineage.first() {
                let ancestor_count = full_lineage.len() - lineage.len();
                ctx.push_str(&format!(
                    "- (epoch baseline: {} prior ancestors, starting from {} score={:.4})\n",
                    ancestor_count,
                    candidate_label(&first.delta, &first.eval),
                    first.eval.score().unwrap_or(0.0),
                ));
            }
        }
        for entry in &lineage {
            ctx.push_str(&format!(
                "- {} score={:.4} parent={}\n",
                candidate_label(&entry.delta, &entry.eval),
                entry.eval.score().unwrap_or(0.0),
                entry.delta.parent_id
                    .and_then(|pid| archive.entries.iter().find(|p| p.delta.id == pid))
                    .map(|p| candidate_label(&p.delta, &p.eval))
                    .unwrap_or_else(|| "none".to_string())
            ));
            if is_seed_candidate(&entry.delta) {
                ctx.push_str("  exact mutation: seed\n");
            } else {
                for mutation in &entry.delta.mutations {
                    ctx.push_str(&format!(
                        "  exact mutation: {}\n",
                        format_mutation_for_meta(mutation)
                    ));
                }
            }
            let active = format_overrides_inline(&entry.delta.overrides);
            ctx.push_str(&format!("  active overrides: {}\n", active));
            if let Some(parent_entry) = entry.delta
                .parent_id
                .and_then(|pid| archive.entries.iter().find(|p| p.delta.id == pid))
            {
                let delta = candidate_case_delta(&parent_entry.eval, &entry.eval);
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
            candidate_label(&best.delta, &best.eval),
            best.eval.score().unwrap_or(0.0)
        ));
        push_candidate_overrides(&mut ctx, "Best Candidate Active Overrides", &best.delta);
        ctx.push_str("Graph:\n```\n");
        ctx.push_str(&scaffold_ir::pretty::pretty_print_graph(&best.delta.graph));
        ctx.push_str("```\n");
        push_candidate_node_state(&mut ctx, "Nodes in Best Candidate", &best.delta, ir);
        push_step_details(&mut ctx, "Steps in Best Candidate", &best.delta.graph);
        ctx.push('\n');
    }

    // 3. Archive summary (top 10 candidates)
    // When epoch filtering is active, only show candidates from the current epoch.
    let epoch_label = if epoch_start_id.is_some() {
        format!(" (epoch from #{})", epoch_start)
    } else {
        String::new()
    };
    ctx.push_str(&format!("## Archive (top candidates by score{})\n", epoch_label));
    let ranked: Vec<&ArchiveEntry> = archive.ranked()
        .into_iter()
        .filter(|e| e.delta.id >= epoch_start)
        .collect();
    for (i, e) in ranked.iter().take(10).enumerate() {
        let mutations_str = if is_seed_candidate(&e.delta) {
            "seed".to_string()
        } else if e.delta.mutations.is_empty() {
            "none".to_string()
        } else {
            e.delta.mutations
                .iter()
                .map(|m| m.short_label())
                .collect::<Vec<_>>()
                .join(", ")
        };
        ctx.push_str(&format!(
            "{}. {} score={:.4} parent={} mutations=[{}] children={}\n",
            i + 1,
            candidate_label(&e.delta, &e.eval),
            e.eval.score().unwrap_or(0.0),
            e.delta.parent_id
                .and_then(|pid| archive.entries.iter().find(|p| p.delta.id == pid))
                .map(|p| candidate_label(&p.delta, &p.eval))
                .unwrap_or_else(|| "none".to_string()),
            mutations_str,
            e.delta.children_count,
        ));
    }
    ctx.push('\n');

    // 3.4 Pass/Fail Matrix — cases (rows) × candidates (columns)
    // Uses TRAIN results only to avoid leaking val labels.
    {
        let mut all_case_ids: BTreeSet<String> = BTreeSet::new();
        for e in &archive.entries {
            if e.eval.score().is_none() || e.delta.id < epoch_start {
                continue;
            }
            for id in e.eval.train_passed_case_ids() {
                all_case_ids.insert(id.clone());
            }
            for cr in e.eval.train_case_results() {
                if let Some(ref id) = cr.case_id {
                    all_case_ids.insert(id.clone());
                }
            }
        }
        let all_case_ids: Vec<String> = all_case_ids.into_iter().collect();

        let mut scored: Vec<&ArchiveEntry> = archive
            .entries
            .iter()
            .filter(|e| e.eval.score().is_some() && e.delta.id >= epoch_start)
            .collect();
        scored.sort_by_key(|e| e.delta.id);

        if !all_case_ids.is_empty() && scored.len() > 1 {
            // Limit to 12 columns to prevent excessive width
            let show: Vec<&ArchiveEntry> = scored.into_iter().take(12).collect();

            ctx.push_str("## Pass/Fail Matrix (train)\n");
            let max_id_len = all_case_ids.iter().map(|id| id.len()).max().unwrap_or(10);
            let pad = max_id_len + 2;

            // Header
            ctx.push_str(&format!("{:pad$}", "Case", pad = pad));
            for e in &show {
                let label = candidate_label(&e.delta, &e.eval);
                ctx.push_str(&format!("{:>6}", label));
            }
            ctx.push('\n');

            // Rows
            for case_id in &all_case_ids {
                ctx.push_str(&format!("{:pad$}", case_id, pad = pad));
                for e in &show {
                    let passed = e.eval.train_passed_case_ids().iter().any(|id| id == case_id);
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
        let evaluated: Vec<&ArchiveEntry> = archive
            .entries
            .iter()
            .filter(|e| e.eval.score().is_some() && !e.delta.mutations.is_empty() && e.delta.id >= epoch_start)
            .collect();

        if !evaluated.is_empty() {
            // Group entries by mutation label (e.g. "rewrite_prompt(solve_code)")
            let mut groups: Vec<(String, Vec<&ArchiveEntry>)> = Vec::new();
            for e in &evaluated {
                let label = e.delta
                    .mutations
                    .last()
                    .map(|m| m.short_label())
                    .unwrap_or_default();
                if let Some(entry) = groups.iter_mut().find(|(l, _)| *l == label) {
                    entry.1.push(e);
                } else {
                    groups.push((label, vec![e]));
                }
            }

            ctx.push_str("## Mutation Effects (grouped by type)\n");
            ctx.push_str("Score impact of each mutation type. Compare with the Pass/Fail Matrix above for case-level detail.\n\n");

            for (label, entries) in &groups {
                let trend = mutation_group_trend(entries, archive);

                ctx.push_str(&format!(
                    "### {} — {} attempt{}, best={:.4}, {}\n",
                    label,
                    entries.len(),
                    if entries.len() > 1 { "s" } else { "" },
                    trend.best_score,
                    trend.verdict,
                ));
                ctx.push_str(&format!(
                    "  Recent deltas: [{}] | ever improved: {} | saturated: {}\n",
                    format_delta_list(&trend.recent_deltas),
                    if trend.ever_improved { "yes" } else { "no" },
                    if trend.saturated { "yes" } else { "no" },
                ));

                let mut non_best_count = 0usize;
                for e in entries {
                    let score = e.eval.score().unwrap_or(0.0);
                    let parent_entry = e.delta
                        .parent_id
                        .and_then(|pid| archive.entries.iter().find(|p| p.delta.id == pid));
                    let parent_score = parent_entry.and_then(|p| p.eval.score()).unwrap_or(0.0);
                    let parent_label = match parent_entry {
                        Some(p) => candidate_label(&p.delta, &p.eval),
                        None => "?".to_string(),
                    };
                    let delta = score - parent_score;

                    // Was this parent the best candidate when the mutation was applied?
                    let was_best = {
                        let best_at_time = archive.entries.iter()
                            .filter(|x| x.delta.id < e.delta.id && x.eval.score().is_some())
                            .max_by(|a, b| a.eval.score().unwrap().partial_cmp(&b.eval.score().unwrap()).unwrap_or(std::cmp::Ordering::Equal));
                        best_at_time.map(|b| b.delta.id) == e.delta.parent_id
                    };
                    let parent_ctx = if was_best { "" } else { " (non-best parent)" };
                    if !was_best { non_best_count += 1; }

                    // Flag likely runtime/template errors (0 passed out of N = total crash)
                    let crash_note = if score == 0.0 && e.eval.total_cases() > 0 && e.eval.cases_passed() == 0 {
                        " ⚠ LIKELY RUNTIME ERROR — mutation idea may be valid, template was broken"
                    } else {
                        ""
                    };
                    ctx.push_str(&format!(
                        "  {} ({}@{:.4} → {:.4}, Δ{}) passed {}/{}{}{}\n",
                        candidate_label(&e.delta, &e.eval),
                        parent_label,
                        parent_score,
                        score,
                        format_delta(delta),
                        e.eval.cases_passed(),
                        e.eval.total_cases(),
                        parent_ctx,
                        crash_note,
                    ));

                    // Show meta-agent reasoning if available (brief)
                    if let Some(ref reasoning) = e.delta.meta_reasoning {
                        let short: String = reasoning.chars().take(200).collect();
                        ctx.push_str(&format!("    Reasoning: \"{}\"\n", short));
                    }
                }
                if non_best_count > 0 && entries.len() > 1 {
                    ctx.push_str(&format!(
                        "  Note: {}/{} attempts were from non-best parents. Recovery deltas don't indicate general effectiveness.\n",
                        non_best_count, entries.len(),
                    ));
                }
                ctx.push('\n');
            }

            // Stagnation warning (inline)
            let epoch_ranked: Vec<&ArchiveEntry> = archive.ranked()
                .into_iter()
                .filter(|e| e.delta.id >= epoch_start)
                .collect();
            let best_score = epoch_ranked
                .first()
                .map(|e| e.eval.score().unwrap_or(0.0))
                .unwrap_or(0.0);
            let stagnation_count = stagnation_count_after_best(archive, epoch_start);
            if stagnation_count >= 2 {
                ctx.push_str(&format!(
                    "⚠ STAGNATION: Best score ({:.4}) unchanged for {} candidates. Try a fundamentally different approach.\n\n",
                    best_score, stagnation_count
                ));
            }
        }
    }

    // Content plateau indicator
    {
        let (plateaued, content_count) = content_plateau_indicator(archive, epoch_start);
        if plateaued {
            ctx.push_str("## Content Plateau: PLATEAUED\n");
            ctx.push_str(&format!(
                "The last 3 content-only mutations (out of {} total) all showed zero or negative improvement.\n",
                content_count,
            ));
            ctx.push_str("Content mutations (rewrite_prompt/system/shell, set_config) are exhausted for now.\n");
            ctx.push_str("**Strongly consider structural changes**: propose_decomposition, insert_step, add_prompt_step, or tool nodes.\n\n");
        }
    }

    let failure_clusters = summarize_failure_clusters(parent_eval);
    let repair_behavior = summarize_repair_behavior(parent, parent_eval, ir);
    let structural_pressure =
        structural_pressure_summary(archive, &failure_clusters, &repair_behavior, epoch_start);

    if !failure_clusters.is_empty() || parent_eval.total_cases() > 0 {
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

        // Collect prior decomposition attempts per motif for annotation.
        let prior_decompositions: Vec<(String, String, f64, bool)> = archive.entries.iter()
            .filter_map(|entry| {
                let score = entry.eval.score().unwrap_or(0.0);
                entry.delta.mutations.iter().find_map(|m| {
                    if let crate::mutations::Mutation::ProposeDecomposition { motif, target_step, config, .. } = m {
                        let had_custom_templates = config.contains_key("templates");
                        Some((motif.to_string(), target_step.clone(), score, had_custom_templates))
                    } else {
                        None
                    }
                })
            })
            .collect();

        // Classify error types for smarter motif suggestions
        let (semantic_confusion, format_errors) = classify_error_types(parent_eval.train_case_results());

        // Quick Wins: lightweight fixes that should be tried before complex structural changes
        ctx.push_str("## Quick Wins (try these first)\n");
        ctx.push_str("Before complex structural changes, consider these lightweight fixes:\n");
        ctx.push_str("- **add_local_checker**: Add a validation expression after a node to catch bad outputs early.\n");
        if format_errors > 0 {
            ctx.push_str("- **normalize_verify** (via propose_decomposition): Append a normalizer to fix spelling/case/format.\n");
        } else if semantic_confusion > 0 {
            ctx.push_str("- **rewrite_prompt**: Refine the classifier's instructions to reduce label confusion (the main error type).\n");
        }
        ctx.push_str("- **attach_example_policy**: Inject few-shot examples from the example bank to guide the model.\n");
        ctx.push_str("These are cheap, composable, and often sufficient before investing in multi-node motifs.\n\n");

        // Motif suggestions based on failure patterns — prominent top-level section
        let motif_suggestions = suggest_motifs(&failure_clusters, semantic_confusion, format_errors);
        if !motif_suggestions.is_empty() {
            ctx.push_str("## RECOMMENDED: Use propose_decomposition\n");
            ctx.push_str("The following motifs match the observed failure patterns. Use propose_decomposition:\n");
            for (motif, reason) in &motif_suggestions {
                ctx.push_str(&format!(
                    "- **{}**: {} — {}\n",
                    motif,
                    reason,
                    crate::motifs::motif_description(motif)
                ));
                // Annotate with prior attempts if any
                let prior: Vec<_> = prior_decompositions.iter()
                    .filter(|(m, _, _, _)| m == &motif.to_string())
                    .collect();
                if !prior.is_empty() {
                    for (_, step, score, had_templates) in &prior {
                        let tmpl_note = if *had_templates { "with custom templates" } else { "with default templates" };
                        ctx.push_str(&format!(
                            "  ⚠ Previously tried on step '{}' {} → score={:.4}. Use DIFFERENT templates/config if retrying.\n",
                            step, tmpl_note, score
                        ));
                    }
                }
                ctx.push_str(&format!(
                    "  → `{{\"kind\":\"propose_decomposition\",\"target_step\":\"<step>\",\"motif\":\"{}\",\"reason\":\"...\",\"config\":{{}},\"reasoning\":\"...\"}}`\n",
                    motif
                ));
            }
            ctx.push('\n');
        }

        // Annotate when semantic confusion is dominant
        if semantic_confusion > format_errors && semantic_confusion > 0 {
            ctx.push_str(&format!(
                "Note: {}/{} failures are valid-but-wrong labels (semantic confusion). normalize_verify CANNOT fix these — it only fixes format/spelling issues. Focus on rewrite_prompt, shortlist_select, router_expert, or attach_example_policy instead.\n\n",
                semantic_confusion, semantic_confusion + format_errors,
            ));
        }
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

    // Step-Transition Attribution (for multi-step pipelines)
    {
        let transitions = compute_step_transitions(parent_eval, parent);
        let has_content = !transitions.correct_to_wrong.is_empty()
            || !transitions.wrong_to_correct.is_empty();
        if has_content {
            ctx.push_str("## Step Attribution\n");
            ctx.push_str("Per-step correctness analysis for multi-step pipelines.\n\n");

            if !transitions.correct_to_wrong.is_empty() {
                ctx.push_str(&format!(
                    "**CORRECT→WRONG** ({} transition{}): An intermediate step had the right answer, but a later step corrupted it.\n",
                    transitions.correct_to_wrong.len(),
                    if transitions.correct_to_wrong.len() > 1 { "s" } else { "" },
                ));
                for t in transitions.correct_to_wrong.iter().take(5) {
                    ctx.push_str(&format!(
                        "  {} — '{}' was correct, '{}' corrupted the output\n",
                        t.case_id, t.from_step, t.to_step,
                    ));
                }
                if transitions.correct_to_wrong.len() > 5 {
                    ctx.push_str(&format!(
                        "  ... and {} more\n",
                        transitions.correct_to_wrong.len() - 5,
                    ));
                }
                ctx.push('\n');
            }

            if !transitions.wrong_to_correct.is_empty() {
                ctx.push_str(&format!(
                    "**WRONG→CORRECT** ({} transition{}): A repair/normalize step successfully fixed the output.\n",
                    transitions.wrong_to_correct.len(),
                    if transitions.wrong_to_correct.len() > 1 { "s" } else { "" },
                ));
                for t in transitions.wrong_to_correct.iter().take(5) {
                    ctx.push_str(&format!(
                        "  {} — '{}' was wrong, '{}' fixed it\n",
                        t.case_id, t.from_step, t.to_step,
                    ));
                }
                if transitions.wrong_to_correct.len() > 5 {
                    ctx.push_str(&format!(
                        "  ... and {} more\n",
                        transitions.wrong_to_correct.len() - 5,
                    ));
                }
                ctx.push('\n');
            }

            ctx.push_str(&format!(
                "Pipeline-wide: all_correct={}, all_wrong={}\n\n",
                transitions.all_correct, transitions.all_wrong,
            ));
        }
    }

    // 4. Parent's evaluation summary + failed cases
    // Val is blind (aggregate only); train has case-level detail.
    if parent_eval.total_cases() > 0 || parent_eval.train_total() > 0 {
        ctx.push_str("## Parent Evaluation Summary\n");
        // Blind val line: aggregate score only
        // score() returns val score when available, else train score
        let score_label = if parent_eval.val.is_some() { "Val Score" } else { "Train Score" };
        ctx.push_str(&format!(
            "{}: {:.4} ({}/{} passed)\n",
            score_label,
            parent_eval.score().unwrap_or(0.0),
            parent_eval.cases_passed(),
            parent_eval.total_cases(),
        ));
        // Train detail header
        if parent_eval.train_total() > 0 {
            let train_failed = parent_eval.train_total() - parent_eval.train_passed();
            ctx.push_str(&format!(
                "Train Batch: {}/{} PASSED, {}/{} FAILED\n",
                parent_eval.train_passed(),
                parent_eval.train_total(),
                train_failed,
                parent_eval.train_total(),
            ));
        }
        // Per-metric breakdown
        if !parent_eval.metric_scores().is_empty() {
            ctx.push_str("Per-metric scores: ");
            let metrics: Vec<String> = parent_eval
                .metric_scores()
                .iter()
                .map(|(k, v)| format!("{}={:.4}", k, v))
                .collect();
            ctx.push_str(&metrics.join(", "));
            ctx.push('\n');
        }
        ctx.push('\n');

        // Regression gate status
        if !archive.regression_set.is_empty() {
            let regressions = archive.count_regressions(parent_eval.train_passed_case_ids());
            let reg_size = archive.regression_set.len();
            if regressions > 0 {
                ctx.push_str(&format!(
                    "Regression gate: parent FAILS {}/{} protected cases — cannot become best even with higher score.\n",
                    regressions, reg_size,
                ));
                let passed_set: std::collections::HashSet<&str> = parent_eval.train_passed_case_ids().iter().map(|s| s.as_str()).collect();
                let regressed: Vec<&String> = archive.regression_set.iter()
                    .filter(|id| !passed_set.contains(id.as_str()))
                    .take(10)
                    .collect();
                ctx.push_str(&format!("  Must fix: [{}]\n", regressed.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")));
            } else {
                ctx.push_str(&format!(
                    "Regression gate: parent passes all {}/{} protected cases\n",
                    reg_size, reg_size,
                ));
            }
            ctx.push('\n');
        }

        // Per-domain breakdown: detect if optimizer is trading domains
        {
            let mut domain_stats: BTreeMap<String, (usize, usize)> = BTreeMap::new(); // (passed, total)
            for case in parent_eval.train_case_results() {
                let domain = case.case_id.as_deref()
                    .map(|id| id.split('_').next().unwrap_or("unknown"))
                    .unwrap_or("unknown")
                    .to_string();
                let entry = domain_stats.entry(domain).or_insert((0, 0));
                entry.1 += 1;
                if case.passed { entry.0 += 1; }
            }
            if domain_stats.len() >= 2 {
                ctx.push_str("Per-domain accuracy:\n");
                let mut worst_pct = 100.0_f64;
                let mut best_pct = 0.0_f64;
                for (domain, (passed, total)) in &domain_stats {
                    let pct = if *total > 0 { *passed as f64 / *total as f64 * 100.0 } else { 0.0 };
                    ctx.push_str(&format!("  {}: {}/{} ({:.0}%)\n", domain, passed, total, pct));
                    worst_pct = worst_pct.min(pct);
                    best_pct = best_pct.max(pct);
                }
                if best_pct - worst_pct > 20.0 {
                    ctx.push_str(&format!(
                        "DOMAIN IMBALANCE: best={:.0}%, worst={:.0}%. The optimizer may be trading domains against each other. Consider router_expert to handle domains separately, or rewrite_prompt with domain-agnostic instructions.\n",
                        best_pct, worst_pct,
                    ));
                }
                ctx.push('\n');
            }
        }

        // Generalization signal: val checkpoint history
        if !archive.val_checkpoints.is_empty() {
            ctx.push_str("## Generalization Signal (Val Checkpoints)\n");
            ctx.push_str("Val is evaluated on held-out data whenever a new best candidate is found.\n");
            for vc in &archive.val_checkpoints {
                let gap = vc.train_score - vc.val_score;
                ctx.push_str(&format!(
                    "  Candidate #{}: train={:.4}, val={:.4}, gap={:+.4}\n",
                    vc.candidate_id, vc.train_score, vc.val_score, gap,
                ));
            }
            // Warning if latest gap is large
            if let Some(latest) = archive.val_checkpoints.last() {
                let gap = latest.train_score - latest.val_score;
                if gap > 0.10 {
                    ctx.push_str(&format!(
                        "\nOVERFITTING WARNING: Train-val gap is {:.0}%. Recent improvements are NOT generalizing to held-out data.\n\
                         Focus on GENERALIZABLE changes: domain_conditioned examples, structural motifs (normalize_verify, shortlist_select, router_expert), and general instructions rather than training-specific rules.\n",
                        gap * 100.0,
                    ));
                }
            }
            ctx.push('\n');
        }

        // Output pattern analysis: group failures by actual output to surface biases.
        // Only shown when recurring patterns exist (≥2 cases with same output).
        {
            let failure_source_for_patterns = parent_eval.train_case_results();
            let pattern_groups = summarize_output_patterns(failure_source_for_patterns);
            if !pattern_groups.is_empty() {
                ctx.push_str("## Output Pattern Analysis\n");
                ctx.push_str("Recurring patterns in failed case outputs. Use these to diagnose systematic biases.\n\n");
                for group in &pattern_groups {
                    let expected_summary = count_expected_labels(&group.expected_excerpts);
                    if group.is_short_garbage {
                        ctx.push_str(&format!(
                            "**SHORT/GARBAGE**: output={:?} → {} case(s) [{}]\n  Expected: {}\n\n",
                            group.output_repr,
                            group.count,
                            group.case_ids.join(", "),
                            expected_summary,
                        ));
                    } else {
                        ctx.push_str(&format!(
                            "**OUTPUT BIAS**: output={:?} → {} case(s) [{}]\n  Expected: {}\n\n",
                            group.output_repr,
                            group.count,
                            group.case_ids.join(", "),
                            expected_summary,
                        ));
                    }
                }
            }
        }

        // Use train_case_results only — val case detail is intentionally blinded.
        let failure_source = parent_eval.train_case_results();
        let failure_source_label = "train batch";

        if !failure_source.is_empty() {
            // Blind summary: score + pass count only, no case IDs or expected values
            let detail_score_label = if parent_eval.val.is_some() { "Val score" } else { "Train score" };
            ctx.push_str(&format!(
                "{}: {:.4} ({}/{} passed)\n",
                detail_score_label,
                parent_eval.score().unwrap_or(0.0),
                parent_eval.cases_passed(),
                parent_eval.total_cases(),
            ));

            // Detailed failures from the failure source (train batch or val)
            let failed_count = failure_source.iter().filter(|c| !c.passed).count();
            let detailed_limit = if meta_full_traces { failed_count } else { 5 };

            let failed_cases: Vec<_> = failure_source.iter().filter(|c| !c.passed).collect();
            ctx.push_str(&format!(
                "\nFailed cases from {} ({}):\n",
                failure_source_label,
                failed_cases.len()
            ));
            for (i, case) in failed_cases.iter().enumerate() {
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
                            failed_cases.len() - detailed_limit,
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

    // 4b. Execution traces for changed cases (when meta_full_traces is enabled).
    // Shows full step traces for cases that flipped between the parent's parent and the parent,
    // giving the meta-agent concrete evidence of what mutations actually changed.
    if meta_full_traces {
        push_changed_case_traces(&mut ctx, archive, parent, parent_eval, ir);
    }

    // 5. Available nodes with DSL source + override annotations
    ctx.push_str("## Available Nodes\n");
    for node in &ir.nodes {
        // Show the node definition as .scaffold DSL source
        ctx.push_str("```\n");
        ctx.push_str(&scaffold_ir::pretty::pretty_print_node(node));
        ctx.push_str("```\n");

        // Show override annotations OR base template/system content
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
        } else if matches!(node.kind, NodeKindIR::Prompt | NodeKindIR::Agent) {
            // Show base template content so the meta-agent knows what it's rewriting
            if let Some(ref sof) = node.config.template {
                if let Ok(content) = crate::node_runner::load_template(sof) {
                    let excerpt: String = content.chars().take(2000).collect();
                    let truncated = if content.chars().count() > 2000 { "..." } else { "" };
                    ctx.push_str(&format!(
                        "  Base template: \"{}{}\"\n",
                        excerpt.replace('\n', "\\n"),
                        truncated
                    ));
                }
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
        } else if matches!(node.kind, NodeKindIR::Prompt | NodeKindIR::Agent) {
            // Show base system prompt content
            if let Some(ref sof) = node.config.system {
                if let Ok(content) = crate::node_runner::load_template(sof) {
                    let excerpt: String = content.chars().take(500).collect();
                    let truncated = if content.chars().count() > 500 { "..." } else { "" };
                    ctx.push_str(&format!(
                        "  Base system: \"{}{}\"\n",
                        excerpt.replace('\n', "\\n"),
                        truncated
                    ));
                }
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
    for (key, val) in &parent.overrides {
        if let Some(name) = key.strip_prefix("_node.") {
            let template_key = format!("{}.template", name);
            let system_key = format!("{}.system", name);
            let model_key = format!("{}.model", name);
            let shell_key = format!("{}.shell", name);
            let kind_str = val.get("kind").and_then(|v| v.as_str()).unwrap_or("prompt");
            let mutations_str = if kind_str == "tool" {
                "rewrite_shell"
            } else {
                "rewrite_prompt, rewrite_system"
            };
            ctx.push_str(&format!("- {} ({}, SYNTHETIC) [mutations: {}] [vars: {}]\n  output type: string", name, kind_str, mutations_str, synthetic_vars));
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
            if let Some(shell) = parent.overrides.get(&shell_key).and_then(|v| v.as_str()) {
                let excerpt: String = shell.chars().take(500).collect();
                let truncated = if shell.len() > 500 { "..." } else { "" };
                ctx.push_str(&format!(
                    "  shell (OVERRIDDEN): \"{}{}\"\n",
                    excerpt.replace('\n', "\\n"),
                    truncated
                ));
            }
            ctx.push('\n');
        }
    }

    // 5. Graph steps for reference
    push_step_details(&mut ctx, "Steps in Parent Graph", &parent.graph);

    ctx
}

// ── Output Pattern Analysis ──

/// A group of failed cases that produced the same (or nearly same) output.
#[derive(Debug)]
struct OutputPatternGroup {
    /// Representative output (truncated for display).
    output_repr: String,
    /// Number of failed cases with this output.
    count: usize,
    /// Case IDs in this group.
    case_ids: Vec<String>,
    /// Expected values for cases in this group (for showing what they should have been).
    expected_excerpts: Vec<String>,
    /// True if the output is very short (≤3 chars) — likely garbage/truncated.
    is_short_garbage: bool,
}

/// Truncate a string to `max` chars, appending "..." if truncated.
fn truncate_str(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max).collect();
        format!("{}...", truncated)
    }
}

/// Summarize expected labels: count occurrences and format as "label(N), label(N), ...".
fn count_expected_labels(expected: &[String]) -> String {
    if expected.is_empty() {
        return "unknown".to_string();
    }
    let mut counts: Vec<(String, usize)> = Vec::new();
    for e in expected {
        if let Some(entry) = counts.iter_mut().find(|(label, _)| label == e) {
            entry.1 += 1;
        } else {
            counts.push((e.clone(), 1));
        }
    }
    counts.sort_by(|a, b| b.1.cmp(&a.1));
    counts
        .iter()
        .map(|(label, count)| {
            if *count > 1 {
                format!("{}({})", label, count)
            } else {
                label.clone()
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// Group failed cases by their actual output to surface recurring patterns.
///
/// Returns groups with ≥2 cases (single-occurrence outputs are not interesting).
/// For code-generation benchmarks, outputs are unique per case → returns empty vec.
fn summarize_output_patterns(cases: &[crate::optimizer::CaseResult]) -> Vec<OutputPatternGroup> {
    let failed: Vec<&crate::optimizer::CaseResult> =
        cases.iter().filter(|c| !c.passed).collect();
    if failed.is_empty() {
        return vec![];
    }

    // Extract the "actual output" for each failed case.
    // For classification: this is the model_response or output_excerpt (the predicted label).
    // For code-gen: this is typically unique per case.
    let mut groups: Vec<OutputPatternGroup> = Vec::new();
    for case in &failed {
        // Use model_response first (raw LLM output), fall back to output_excerpt
        let raw_output = case
            .model_response
            .as_deref()
            .or(case.output_excerpt.as_deref())
            .unwrap_or("");
        // Normalize: trim whitespace, lowercase for grouping
        let normalized = raw_output.trim().to_lowercase();
        // Use first 200 chars for grouping key (avoids grouping by long identical prefixes)
        let group_key: String = normalized.chars().take(200).collect();

        let case_id = case.case_id.as_deref().unwrap_or("?").to_string();
        let expected = case
            .expected_excerpt
            .as_deref()
            .unwrap_or("?")
            .to_string();

        if let Some(entry) = groups.iter_mut().find(|g| {
            let g_key: String = g.output_repr.trim().to_lowercase().chars().take(200).collect();
            g_key == group_key
        }) {
            entry.count += 1;
            entry.case_ids.push(case_id);
            entry.expected_excerpts.push(expected);
        } else {
            let repr = truncate_str(raw_output.trim(), 80);
            let is_short = raw_output.trim().chars().count() <= 3;
            groups.push(OutputPatternGroup {
                output_repr: repr,
                count: 1,
                case_ids: vec![case_id],
                expected_excerpts: vec![expected],
                is_short_garbage: is_short,
            });
        }
    }

    // Only keep groups with ≥2 cases (recurring patterns)
    groups.retain(|g| g.count >= 2);
    // Sort by count descending
    groups.sort_by(|a, b| b.count.cmp(&a.count));
    groups
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

/// Returns (semantic_confusion_count, format_error_count) from failed cases.
/// Semantic confusion = output matches an expected value from another case.
/// Format error = output doesn't match any known expected value.
fn classify_error_types(cases: &[crate::optimizer::CaseResult]) -> (usize, usize) {
    let expected_set: HashSet<String> = cases.iter()
        .filter_map(|c| c.expected_excerpt.as_ref())
        .map(|e| e.trim().to_lowercase())
        .collect();
    let mut semantic = 0;
    let mut format_err = 0;
    for case in cases.iter().filter(|c| !c.passed) {
        if let Some(ref output) = case.output_excerpt {
            if expected_set.contains(&output.trim().to_lowercase()) {
                semantic += 1;
            } else {
                format_err += 1;
            }
        } else {
            format_err += 1;
        }
    }
    (semantic, format_err)
}

/// Suggest relevant motifs based on failure cluster analysis.
fn suggest_motifs(clusters: &[FailureClusterSummary], semantic_confusion: usize, format_errors: usize) -> Vec<(crate::motifs::Motif, String)> {
    let mut suggestions = vec![];
    // Only suggest NormalizeVerify when format errors exist (truncated, wrong case, not in allowed set).
    // Skip when failures are predominantly semantic confusion (valid but wrong label).
    if format_errors > 0 {
        suggestions.push((
            crate::motifs::Motif::NormalizeVerify,
            format!("{} format error(s) (truncated/case/whitespace) — normalize_verify can fix these", format_errors),
        ));
    }
    if semantic_confusion > 0 && format_errors == 0 {
        // Annotate that normalize won't help
        suggestions.push((
            crate::motifs::Motif::ShortlistSelect,
            format!("{} semantic confusion(s) (valid but wrong label) — shortlist_select narrows the candidate space", semantic_confusion),
        ));
    }
    for cluster in clusters {
        match cluster.kind {
            FailureClusterKind::ExactOutputContract => {
                // Already covered by the blanket NormalizeVerify above, but add specific context
                suggestions.push((
                    crate::motifs::Motif::NormalizeVerify,
                    format!("{} case(s) with exact-output failures (format/case/whitespace)", cluster.count),
                ));
            }
            FailureClusterKind::ApiShapeContract => {
                suggestions.push((
                    crate::motifs::Motif::ShortlistSelect,
                    format!("{} case(s) with label/shape confusion", cluster.count),
                ));
            }
            FailureClusterKind::AlgorithmSearchLogic => {
                suggestions.push((
                    crate::motifs::Motif::VoteCritiqueRepair,
                    format!("{} case(s) with wrong algorithm/logic — critique+repair may catch errors", cluster.count),
                ));
                suggestions.push((
                    crate::motifs::Motif::GenerateValidateRefine,
                    format!("{} case(s) with wrong algorithm/logic — tool-based validation can verify correctness deterministically", cluster.count),
                ));
                suggestions.push((
                    crate::motifs::Motif::ShortlistSelect,
                    format!("{} case(s) with wrong algorithm/logic — narrowing candidates before selecting may reduce confusion", cluster.count),
                ));
                suggestions.push((
                    crate::motifs::Motif::RouterExpert,
                    format!("{} case(s) with wrong algorithm/logic — routing by domain may help if inputs span multiple categories", cluster.count),
                ));
            }
            FailureClusterKind::RuntimeException => {
                suggestions.push((
                    crate::motifs::Motif::VoteCritiqueRepair,
                    format!("{} case(s) with runtime exceptions — critique may catch invalid code", cluster.count),
                ));
                suggestions.push((
                    crate::motifs::Motif::GenerateValidateRefine,
                    format!("{} case(s) with runtime exceptions — tool-based validation can catch runtime errors deterministically", cluster.count),
                ));
            }
            _ => {} // Other clusters don't have a strong motif mapping
        }
    }
    suggestions.dedup_by(|a, b| std::mem::discriminant(&a.0) == std::mem::discriminant(&b.0));
    suggestions
}

// ── Step-Transition Attribution ──

/// Per-step correctness in a multi-step pipeline.
#[derive(Debug, Clone)]
struct StepCorrectness {
    step_name: String,
    /// Whether this step's output matched expected (approximate).
    correct: bool,
}

/// A transition between two adjacent pipeline steps where correctness changed.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct StepTransition {
    case_id: String,
    from_step: String,
    to_step: String,
    /// true = correct→wrong (corruption), false = wrong→correct (repair)
    is_corruption: bool,
}

/// Summary of step-level attribution across all cases.
#[derive(Debug, Default)]
struct TransitionSummary {
    /// Cases where an intermediate step was correct but a later step corrupted the answer.
    correct_to_wrong: Vec<StepTransition>,
    /// Cases where an earlier step was wrong but a later step fixed it.
    wrong_to_correct: Vec<StepTransition>,
    /// Cases where all steps were correct throughout.
    all_correct: usize,
    /// Cases where all steps were wrong throughout.
    all_wrong: usize,
}

/// Compute step-transition attribution for multi-step pipelines.
///
/// For each case with step_trace.len() >= 2, checks each intermediate step output
/// against the expected value using simple trim + case-insensitive comparison.
/// This is appropriate for classification tasks where outputs are short labels.
fn compute_step_transitions(
    eval: &EvalResults,
    _candidate: &CandidateDelta,
) -> TransitionSummary {
    let mut summary = TransitionSummary::default();

    for case in eval.train_case_results() {
        if case.step_trace.len() < 2 {
            continue;
        }
        let expected = match case.expected_excerpt.as_deref() {
            Some(e) => e.trim(),
            None => continue,
        };
        if expected.is_empty() {
            continue;
        }

        let case_id = case.case_id.as_deref().unwrap_or("?").to_string();

        let correctness: Vec<StepCorrectness> = case
            .step_trace
            .iter()
            .map(|(step_name, output)| {
                let correct = output.trim().eq_ignore_ascii_case(expected);
                StepCorrectness {
                    step_name: step_name.clone(),
                    correct,
                }
            })
            .collect();

        // Check if all correct or all wrong
        let all_correct = correctness.iter().all(|s| s.correct);
        let all_wrong = correctness.iter().all(|s| !s.correct);
        if all_correct {
            summary.all_correct += 1;
            continue;
        }
        if all_wrong {
            summary.all_wrong += 1;
            continue;
        }

        // Find transitions
        for window in correctness.windows(2) {
            let from = &window[0];
            let to = &window[1];
            if from.correct && !to.correct {
                summary.correct_to_wrong.push(StepTransition {
                    case_id: case_id.clone(),
                    from_step: from.step_name.clone(),
                    to_step: to.step_name.clone(),
                    is_corruption: true,
                });
            } else if !from.correct && to.correct {
                summary.wrong_to_correct.push(StepTransition {
                    case_id: case_id.clone(),
                    from_step: from.step_name.clone(),
                    to_step: to.step_name.clone(),
                    is_corruption: false,
                });
            }
        }
    }

    summary
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

fn is_seed_candidate(candidate: &CandidateDelta) -> bool {
    candidate.parent_id.is_none()
        && candidate.mutations.is_empty()
        && candidate.overrides.is_empty()
}

fn candidate_label(candidate: &CandidateDelta, _eval: &EvalResults) -> String {
    if is_seed_candidate(candidate) {
        "seed".to_string()
    } else {
        format!("#{}", candidate.id)
    }
}

fn candidate_lineage<'a>(archive: &'a Archive, candidate: &'a CandidateDelta) -> Vec<&'a ArchiveEntry> {
    let mut lineage: Vec<&'a ArchiveEntry> = Vec::new();
    // Find the entry for the candidate itself
    let mut current_entry = archive.entries.iter().find(|e| e.delta.id == candidate.id);
    while let Some(entry) = current_entry {
        lineage.push(entry);
        current_entry = entry.delta
            .parent_id
            .and_then(|pid| archive.entries.iter().find(|p| p.delta.id == pid));
    }
    lineage.reverse();
    lineage
}

fn candidate_case_delta(parent_eval: &EvalResults, child_eval: &EvalResults) -> CaseDelta {
    let parent_pass: BTreeSet<&str> = parent_eval
        .train_passed_case_ids()
        .iter()
        .map(|id| id.as_str())
        .collect();
    let child_pass: BTreeSet<&str> = child_eval.train_passed_case_ids().iter().map(|id| id.as_str()).collect();

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

fn summarize_failure_clusters(eval: &EvalResults) -> Vec<FailureClusterSummary> {
    let mut grouped: BTreeMap<FailureClusterKind, FailureClusterSummary> = BTreeMap::new();

    // Use train results (val is blinded)
    for case in eval.train_case_results() {
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

fn summarize_repair_behavior(candidate: &CandidateDelta, eval: &EvalResults, ir: &ScaffoldIR) -> RepairBehaviorSummary {
    let mut summary = RepairBehaviorSummary::default();

    // Use train results (val is blinded)
    for case in eval.train_case_results() {
        let case_id = case
            .case_id
            .as_deref()
            .unwrap_or("?")
            .to_string();

        // Collect prompt/agent outputs with their step→node mapping
        let mut prompt_steps: Vec<(String, String, &str)> = Vec::new(); // (step, node, value)
        let mut non_agent_outputs: Vec<&str> = Vec::new();
        let mut all_steps: Vec<(String, String)> = Vec::new(); // (step, node) in execution order

        for (step_name, value) in &case.step_trace {
            let node_name = find_step_node(&candidate.graph.body, step_name)
                .unwrap_or_else(|| "?".to_string());
            let kind = node_kind_label(ir, &node_name, &candidate.overrides);
            all_steps.push((step_name.clone(), node_name.clone()));
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

        // The repair node is the last step UNLESS it's a repeated evaluator
        // (same node appeared earlier — e.g. eval_exercise runs after each attempt).
        // In that case, the second-to-last step is the actual repair node.
        let repair_node = if all_steps.len() >= 2 {
            let last_node = &all_steps[all_steps.len() - 1].1;
            let is_repeated = all_steps[..all_steps.len() - 1]
                .iter()
                .any(|(_, n)| n == last_node);
            if is_repeated {
                Some(all_steps[all_steps.len() - 2].1.clone())
            } else {
                Some(last_node.clone())
            }
        } else {
            all_steps.last().map(|(_, node)| node.clone())
        };

        // Compare first prompt output (proposed answer) with final pipeline output.
        // This catches VoteCritiqueRepair where critique outputs JSON but repair tool
        // may keep the proposed answer unchanged.
        let first_prompt_output = prompt_steps.first().map(|(_, _, v)| *v);
        let final_output = case.step_trace.last().map(|(_, v)| v.as_str());
        let pipeline_noop = match (first_prompt_output, final_output) {
            (Some(a), Some(b)) => a.trim() == b.trim(),
            _ => false,
        };

        // Also check original prompt_same for backward compatibility
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

        let outcome = if pipeline_noop || prompt_same {
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

fn mutation_group_trend(group: &[&ArchiveEntry], archive: &Archive) -> MutationTrendSummary {
    let mut ordered: Vec<&ArchiveEntry> = group.to_vec();
    ordered.sort_by_key(|entry| entry.delta.id);

    let deltas: Vec<f64> = ordered
        .iter()
        .map(|entry| {
            let parent_score = entry.delta
                .parent_id
                .and_then(|pid| archive.entries.iter().find(|parent| parent.delta.id == pid))
                .and_then(|parent| parent.eval.score())
                .unwrap_or(0.0);
            entry.eval.score().unwrap_or(0.0) - parent_score
        })
        .collect();

    let best_score = ordered
        .iter()
        .filter_map(|entry| entry.eval.score())
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

/// Detect whether content-only mutations have plateaued.
///
/// Returns `(plateaued, count)` where `plateaued` is true if the last 3 consecutive
/// content mutations all had delta <= 0 (no improvement).
fn content_plateau_indicator(archive: &Archive, epoch_start: usize) -> (bool, usize) {
    let content_kinds = [
        "rewrite_prompt", "rewrite_system", "rewrite_shell", "rewrite_tool_spec", "set_config",
    ];
    let content_entries: Vec<&ArchiveEntry> = archive
        .entries
        .iter()
        .filter(|e| {
            e.eval.score().is_some()
                && e.delta.id >= epoch_start
                && e.delta.mutations.last().map(|m| {
                    let label = m.short_label();
                    content_kinds.iter().any(|k| label.starts_with(k))
                }).unwrap_or(false)
        })
        .collect();

    let count = content_entries.len();
    if count < 3 {
        return (false, count);
    }

    // Check last 3 content mutations for non-positive deltas
    let last_3: Vec<f64> = content_entries[content_entries.len() - 3..]
        .iter()
        .map(|entry| {
            let parent_score = entry.delta
                .parent_id
                .and_then(|pid| archive.entries.iter().find(|p| p.delta.id == pid))
                .and_then(|p| p.eval.score())
                .unwrap_or(0.0);
            entry.eval.score().unwrap_or(0.0) - parent_score
        })
        .collect();

    let plateaued = last_3.iter().all(|d| *d <= 0.0);
    (plateaued, count)
}

/// Return mutation short_labels whose trend is saturated within the current epoch.
///
/// Used by `propose_mutation` to hard-block the meta-agent from repeating exhausted
/// mutation families.
fn compute_saturated_families(archive: &Archive, epoch_start: usize) -> Vec<String> {
    let evaluated: Vec<&ArchiveEntry> = archive
        .entries
        .iter()
        .filter(|e| e.eval.score().is_some() && !e.delta.mutations.is_empty() && e.delta.id >= epoch_start)
        .collect();
    let mut groups: Vec<(String, Vec<&ArchiveEntry>)> = Vec::new();
    for entry in &evaluated {
        let label = entry.delta
            .mutations
            .last()
            .map(|m| m.short_label())
            .unwrap_or_default();
        if let Some(grp) = groups.iter_mut().find(|(l, _)| *l == label) {
            grp.1.push(entry);
        } else {
            groups.push((label, vec![entry]));
        }
    }
    groups
        .into_iter()
        .filter(|(_, g)| mutation_group_trend(g.as_slice(), archive).saturated)
        .map(|(label, _)| label)
        .collect()
}

fn stagnation_count_after_best(archive: &Archive, epoch_start: usize) -> usize {
    let best_id = archive.ranked()
        .into_iter()
        .filter(|e| e.delta.id >= epoch_start)
        .map(|e| e.delta.id)
        .next()
        .unwrap_or(0);
    archive
        .entries
        .iter()
        .filter(|e| e.eval.score().is_some() && e.delta.id > best_id && e.delta.id >= epoch_start)
        .count()
}

fn structural_pressure_summary(
    archive: &Archive,
    clusters: &[FailureClusterSummary],
    repair: &RepairBehaviorSummary,
    epoch_start: usize,
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

    let evaluated: Vec<&ArchiveEntry> = archive
        .entries
        .iter()
        .filter(|e| e.eval.score().is_some() && !e.delta.mutations.is_empty() && e.delta.id >= epoch_start)
        .collect();
    let mut saturated_families = 0usize;
    let mut groups: Vec<(String, Vec<&ArchiveEntry>)> = Vec::new();
    for entry in evaluated {
        let label = entry.delta
            .mutations
            .last()
            .map(|mutation| mutation.short_label())
            .unwrap_or_default();
        if let Some(grp) = groups
            .iter_mut()
            .find(|(group_label, _)| *group_label == label)
        {
            grp.1.push(entry);
        } else {
            groups.push((label, vec![entry]));
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

    let stagnation = stagnation_count_after_best(archive, epoch_start);
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

fn push_candidate_overrides(ctx: &mut String, title: &str, candidate: &CandidateDelta) {
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
    candidate: &CandidateDelta,
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

fn active_node_config_summary(candidate: &CandidateDelta, node: &NodeIR) -> Option<String> {
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

fn synthetic_node_config_summary(candidate: &CandidateDelta, node_name: &str) -> String {
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
    lineage: &[&ArchiveEntry],
    ir: &ScaffoldIR,
) {
    if lineage.len() <= 1 {
        return;
    }

    ctx.push_str("## Lineage Mutation Evidence\n");
    ctx.push_str("Raw failure evidence for cases each lineage mutation fixed or broke. For fixed cases, evidence is taken from the parent failure that the child resolved.\n\n");

    for window in lineage.windows(2) {
        let parent_entry = window[0];
        let child_entry = window[1];
        let delta = candidate_case_delta(&parent_entry.eval, &child_entry.eval);
        ctx.push_str(&format!(
            "### {} from {}\n",
            candidate_label(&child_entry.delta, &child_entry.eval),
            candidate_label(&parent_entry.delta, &parent_entry.eval)
        ));
        for mutation in &child_entry.delta.mutations {
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
            if let Some(case) = find_case_result(&parent_entry.eval, case_id) {
                ctx.push_str(&format!(
                    "Pre-fix failure evidence from {}:\n",
                    candidate_label(&parent_entry.delta, &parent_entry.eval)
                ));
                ctx.push_str(&format!(
                    "Child outcome: F -> P (resolved in {})\n",
                    candidate_label(&child_entry.delta, &child_entry.eval)
                ));
                push_case_evidence(ctx, &parent_entry.delta, case, ir, 2500, None);
            }
        }

        for case_id in delta.broken.iter().take(3) {
            if let Some(case) = find_case_result(&child_entry.eval, case_id) {
                ctx.push_str(&format!(
                    "Regression evidence from {}:\n",
                    candidate_label(&child_entry.delta, &child_entry.eval)
                ));
                ctx.push_str(&format!(
                    "Child outcome: P -> F (regressed in {})\n",
                    candidate_label(&child_entry.delta, &child_entry.eval)
                ));
                push_case_evidence(ctx, &child_entry.delta, case, ir, 2500, None);
            }
        }

        if let Some(best) = archive.best() {
            if best.delta.id == child_entry.delta.id {
                ctx.push_str("This mutation is currently on the best-known path.\n");
            }
        }
        ctx.push('\n');
    }
}

/// Show full execution traces for cases that flipped (passed↔failed) between
/// the parent's parent and the parent. This gives the meta-agent concrete evidence
/// of what the most recent mutation actually changed.
fn push_changed_case_traces(
    ctx: &mut String,
    archive: &Archive,
    parent: &CandidateDelta,
    parent_eval: &EvalResults,
    ir: &ScaffoldIR,
) {
    // Find the grandparent (parent's parent)
    let grandparent = parent
        .parent_id
        .and_then(|pid| archive.entries.iter().find(|e| e.delta.id == pid));

    let grandparent = match grandparent {
        Some(gp) => gp,
        None => return, // seed candidate, no prior generation to compare
    };

    let delta = candidate_case_delta(&grandparent.eval, parent_eval);
    if delta.fixed.is_empty() && delta.broken.is_empty() {
        return; // no flips, nothing to show
    }

    ctx.push_str("## Execution Traces for Changed Cases\n");
    ctx.push_str("Full step traces for cases that flipped between the grandparent and parent.\n\n");

    // Fixed cases: show the parent's trace (the successful execution)
    for case_id in &delta.fixed {
        if let Some(case) = find_case_result(parent_eval, case_id) {
            ctx.push_str(&format!("### {} (FIXED: F → P)\n", case_id));
            push_case_evidence(ctx, parent, case, ir, 4000, None);
        }
    }

    // Broken cases: show the parent's trace (the failing execution)
    for case_id in &delta.broken {
        if let Some(case) = find_case_result(parent_eval, case_id) {
            ctx.push_str(&format!("### {} (BROKEN: P → F)\n", case_id));
            push_case_evidence(ctx, parent, case, ir, 4000, None);
        }
    }
    ctx.push('\n');
}

fn find_case_result<'a>(
    eval: &'a EvalResults,
    case_id: &str,
) -> Option<&'a crate::optimizer::CaseResult> {
    // Search train results (val case_results are blinded)
    eval
        .train_case_results()
        .iter()
        .find(|case| case.case_id.as_deref() == Some(case_id))
}

fn push_case_evidence(
    ctx: &mut String,
    candidate: &CandidateDelta,
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
    if let Some(val) = overrides.get(&format!("_node.{}", node_name)) {
        return match val.get("kind").and_then(|v| v.as_str()) {
            Some("tool") => "tool",
            Some("agent") => "agent",
            Some("verify") => "verify",
            _ => "prompt",
        };
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
    // propose_decomposition is allowed when any structural mutation is in the allowed list.
    let content_mutations = [
        "rewrite_prompt", "rewrite_system", "rewrite_shell", "rewrite_tool_spec",
        "attach_example_policy", "add_local_checker",
    ];
    let has_any_structural = allowed_mutations
        .iter()
        .any(|m| STRUCTURAL_MUTATIONS.contains(&m.as_str()));
    if !content_mutations.contains(&kind)
        && !(kind == "propose_decomposition" && has_any_structural)
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
        "rewrite_tool_spec" => {
            let node = require_str(&json, "node")?;
            validate_node_exists(ir, &node, Some(NodeKindIR::Tool))?;
            let spec_value = json.get("spec").ok_or_else(|| {
                Error::Runtime("meta-agent response missing 'spec' field for rewrite_tool_spec".into())
            })?;
            let spec: crate::mutations::ToolSpec = serde_json::from_value(spec_value.clone())
                .map_err(|e| Error::Runtime(format!("invalid tool spec: {}", e)))?;
            if spec.argv.is_empty() {
                return Err(Error::Runtime("rewrite_tool_spec: argv must not be empty".into()));
            }
            Mutation::RewriteToolSpec { node, spec }
        }
        "propose_decomposition" => {
            let target_step = require_str(&json, "target_step")?;
            let motif_str = require_str(&json, "motif")?;
            let reason = json
                .get("reason")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let motif: crate::motifs::Motif = serde_json::from_value(serde_json::json!(motif_str))
                .map_err(|e| Error::Runtime(format!(
                    "invalid motif '{}': {}. Valid motifs: normalize_verify, shortlist_select, router_expert, retrieve_decide, vote_critique_repair",
                    motif_str, e
                )))?;
            validate_step_exists(parent_graph, &target_step)?;
            let config: std::collections::HashMap<String, serde_json::Value> = json
                .get("config")
                .and_then(|v| serde_json::from_value(v.clone()).ok())
                .unwrap_or_default();
            // Validate the motif can be applied
            crate::motifs::apply_motif(&motif, parent_graph, &target_step, &config, ir)
                .map_err(|e| Error::Runtime(format!("propose_decomposition: {}", e)))?;
            Mutation::ProposeDecomposition {
                target_step,
                motif,
                reason,
                config,
            }
        }
        "attach_example_policy" => {
            let node = require_str(&json, "node")?;
            validate_node_exists(ir, &node, None)?;
            let policy_value = json.get("policy").ok_or_else(|| {
                Error::Runtime("meta-agent response missing 'policy' field for attach_example_policy".into())
            })?;
            let policy: crate::example_bank::ExamplePolicy =
                serde_json::from_value(policy_value.clone())
                    .map_err(|e| Error::Runtime(format!("invalid example policy: {}", e)))?;
            if policy.k == 0 {
                return Err(Error::Runtime("attach_example_policy: k must be > 0".into()));
            }
            Mutation::AttachExamplePolicy { node, policy }
        }
        "add_local_checker" => {
            let node = require_str(&json, "node")?;
            let checker_name = require_str(&json, "checker_name")?;
            let expr = require_str(&json, "expr")?;
            validate_node_exists(ir, &node, None)?;
            Mutation::AddLocalChecker {
                node,
                checker_name,
                expr,
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
            // Synthetic node (from motif or AddPromptStep): allowed vars = input + graph
            // input fields + named arguments from the step that invokes this node.
            if let Some(graph) = parent_graph {
                let mut vars = vec!["input".to_string()];
                vars.extend(crate::mutations::resolve_graph_input_fields(graph, ir));
                // Find steps that invoke this node and add their named arg names.
                // This covers motif-wired variables like `proposed`, `critique`,
                // `feedback`, `candidates`, `raw`, `retrieved`, etc.
                fn collect_step_args(body: &[GraphStmtIR], node_name: &str, vars: &mut Vec<String>) {
                    for stmt in body {
                        match stmt {
                            GraphStmtIR::Step(s) if s.node == node_name => {
                                for arg in &s.args {
                                    if let StepArgIR::Named { name, .. } = arg {
                                        if !vars.contains(name) {
                                            vars.push(name.clone());
                                        }
                                    }
                                }
                            }
                            GraphStmtIR::Loop(l) => {
                                collect_step_args(&l.body, node_name, vars);
                            }
                            GraphStmtIR::If(i) => {
                                collect_step_args(&i.then_body, node_name, vars);
                                collect_step_args(&i.else_body, node_name, vars);
                            }
                            _ => {}
                        }
                    }
                }
                collect_step_args(&graph.body, node_name, &mut vars);
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
    use crate::optimizer::{BlindEval, CaseResult, TrainEval};

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
            expected_excerpt: None,
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
    ) -> (CandidateDelta, EvalResults) {
        let passed_ids: Vec<String> = passed_case_ids
            .into_iter()
            .map(|case| case.to_string())
            .collect();
        let delta = CandidateDelta {
            id,
            parent_id,
            graph: graph.clone(),
            overrides: HashMap::new(),
            mutations,
            descriptor: GraphDescriptor::from_graph(graph, ir),
            children_count: 0,
            meta_reasoning: None,
        };
        let eval = EvalResults {
            val: Some(BlindEval {
                score,
                metric_scores: HashMap::new(),
                total: total_cases,
                passed: cases_passed,
            }),
            train: Some(TrainEval {
                score,
                metric_scores: HashMap::new(),
                cases: case_results,
                total: total_cases,
                passed: cases_passed,
                passed_case_ids: passed_ids,
            }),
        };
        (delta, eval)
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
        let (_delta, eval) = make_candidate(
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

        let clusters = summarize_failure_clusters(&eval);
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
        let (delta, eval) = make_candidate(
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

        let summary = summarize_repair_behavior(&delta, &eval, &ir);
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
        let (delta, eval) = make_candidate(
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

        let summary = summarize_repair_behavior(&delta, &eval, &ir);
        assert_eq!(summary.no_op, 1); // SameError also counts as no_op
        assert_eq!(summary.per_case.len(), 1);
        assert_eq!(summary.per_case[0].outcome, RepairOutcome::SameError);
    }

    #[test]
    fn test_repair_behavior_single_attempt() {
        // When only one prompt step exists → SingleAttempt
        let ir = make_test_ir();
        let graph = make_test_graph();
        let (delta, eval) = make_candidate(
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

        let summary = summarize_repair_behavior(&delta, &eval, &ir);
        assert_eq!(summary.no_repair_signal, 1);
        assert_eq!(summary.per_case.len(), 1);
        assert_eq!(summary.per_case[0].outcome, RepairOutcome::SingleAttempt);
    }

    #[test]
    fn test_mutation_group_trend_flags_recent_saturation() {
        let ir = make_test_ir();
        let graph = make_test_graph();
        let (seed_d, seed_e) = make_candidate(0, None, 0.2, vec![], vec![], 0, 0, vec![], &graph, &ir);
        let (child1_d, child1_e) = make_candidate(
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
        let (child2_d, child2_e) = make_candidate(
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
        let (child3_d, child3_e) = make_candidate(
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
        archive.add(seed_d, seed_e);
        archive.add(child1_d, child1_e);
        archive.add(child2_d, child2_e);
        archive.add(child3_d, child3_e);
        let trend = mutation_group_trend(
            &[
                &archive.entries[1],
                &archive.entries[2],
                &archive.entries[3],
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

        let (seed_d, seed_e) = make_candidate(
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
        let (parent_d, parent_e) = make_candidate(
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
        let (flat_d, flat_e) = make_candidate(
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
        archive.add(seed_d, seed_e);
        archive.add(parent_d, parent_e);
        archive.add(flat_d, flat_e);

        let ctx = build_context(
            &archive.entries[1].delta,
            &archive.entries[1].eval,
            &archive,
            &ir,
            &objective,
            &["add_prompt_step".into(), "set_config".into()],
            None,
            false,
        );

        assert!(ctx.contains("## Failure Decomposition Hints"));
        assert!(ctx.contains("Structural pressure:"));
        assert!(ctx.contains("Recent deltas:"));
        assert!(ctx.contains("exact-output-contract") || ctx.contains("api-shape-contract"));
    }

}
