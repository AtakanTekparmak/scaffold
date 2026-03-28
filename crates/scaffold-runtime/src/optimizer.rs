//! Optimization loop for scaffold v2.
//!
//! Supports two backends:
//! - **Grid**: enumerate all tunable combinations (for small search spaces)
//! - **Evolutionary**: topology-aware evolutionary search with archive + novelty
//!
//! The optimizer works on three search layers (in priority order):
//! 1. Topology — graph structure mutations
//! 2. Components — node selection
//! 3. Text — prompt content

use std::collections::HashMap;
use std::path::PathBuf;

use scaffold_ir::ir::*;
use scaffold_ir::pretty::pretty_print;

use crate::error::{Error, Result};
use crate::executor::{GraphExecutor, TunableOverrides};
use crate::mutations::{
    apply_mutation, check_constraints, collect_step_names, GraphDescriptor, Mutation,
    MutationResult,
};
use crate::scope::Scope;
use crate::value::Value;

// ── Event types for live TUI ──

/// Events emitted by the optimizer for live visualization.
#[derive(Debug, Clone)]
pub enum OptEvent {
    SubObjectiveStarted {
        sub_name: String,
        graph_name: String,
    },
    SubObjectiveCompleted {
        sub_name: String,
        best_score: Option<f64>,
        total_candidates: usize,
    },
    ParentPhaseStarted {
        objective_name: String,
    },
    PhaseChanged {
        phase: OptPhase,
    },
    CandidateEvaluated {
        candidate_id: usize,
        parent_id: Option<usize>,
        score: f64,
        metric_scores: HashMap<String, f64>,
        best_so_far: f64,
        mutations: Vec<String>,
        generation: usize,
        max_generations: usize,
    },
    EvaluationStarted {
        candidate_id: usize,
        total_cases: usize,
    },
    CaseCompleted {
        candidate_id: usize,
        case_index: usize,
        total_cases: usize,
        passed: bool,
        case_id: Option<String>,
    },
    MutationSkipped {
        reason: String,
    },
    MetaProposal {
        reasoning: String,
        mutation_label: String,
        context_tokens: usize,
    },
    EarlyStopped {
        score: f64,
        target: f64,
    },
    Completed {
        objective_name: String,
        best_score: Option<f64>,
        total_candidates: usize,
    },
    Log {
        message: String,
    },
    MetaAgentActive {
        model: String,
    },
    MetaAgentThinking,
    HarnessInfo {
        models: Vec<String>,
    },
    SummarizingFailures {
        done: usize,
        total: usize,
    },
}

/// Optimizer phase.
#[derive(Debug, Clone, Copy)]
pub enum OptPhase {
    Seeding,
    TunableSweep,
    Evolutionary,
}

/// Optimization options.
#[derive(Debug, Clone)]
pub struct OptimizationOptions {
    /// Maximum number of successful evolutionary generations.
    /// Tunable parameter sweeps run as an uncapped seeding phase before this.
    pub max_candidates: usize,
    /// Optimization backend to use.
    pub backend: OptimizationBackend,
    /// Directory to write reports.
    pub report_dir: Option<PathBuf>,
    /// Write the best candidate as a standalone `.scaffold` program to this file.
    pub write_best: Option<PathBuf>,
    /// Optional event sender for live TUI visualization.
    pub event_tx: Option<tokio::sync::mpsc::UnboundedSender<OptEvent>>,
    /// Number of dataset cases to evaluate concurrently per candidate.
    /// Defaults to 1 (sequential).
    pub concurrency: usize,
    /// LLM model for meta-agent guided mutations.
    /// None = random mutations (backward compatible).
    pub meta_model: Option<String>,
    /// Path to a debug log file for meta-agent context/responses.
    /// When set, every meta-agent LLM call appends the full system prompt,
    /// context, raw response, and parse result to this file.
    pub meta_log: Option<PathBuf>,
}

impl Default for OptimizationOptions {
    fn default() -> Self {
        Self {
            max_candidates: 20,
            backend: OptimizationBackend::Evolutionary,
            report_dir: None,
            write_best: None,
            event_tx: None,
            concurrency: 1,
            meta_model: None,
            meta_log: None,
        }
    }
}

/// Emit an event to the TUI if a sender is configured.
fn emit(options: &OptimizationOptions, event: OptEvent) {
    if let Some(ref tx) = options.event_tx {
        let _ = tx.send(event);
    }
}

/// Reset the meta-agent debug log once at the start of a top-level optimize run.
///
/// The meta-agent itself appends within a run; this ensures separate runs don't
/// accumulate into one ever-growing file.
fn reset_meta_log(options: &OptimizationOptions) -> Result<()> {
    let Some(path) = options.meta_log.as_ref() else {
        return Ok(());
    };

    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|e| {
                Error::Runtime(format!(
                    "failed to create meta log directory '{}': {}",
                    parent.display(),
                    e
                ))
            })?;
        }
    }

    std::fs::File::create(path).map_err(|e| {
        Error::Runtime(format!(
            "failed to reset meta log '{}': {}",
            path.display(),
            e
        ))
    })?;

    Ok(())
}

/// Backend selection.
#[derive(Debug, Clone, Copy)]
pub enum OptimizationBackend {
    /// Enumerate all tunable combinations.
    Grid,
    /// Evolutionary search with topology mutations.
    Evolutionary,
}

/// Per-case evaluation result (for meta-agent feedback).
#[derive(Debug, Clone)]
pub struct CaseResult {
    pub case_id: Option<String>,
    pub passed: bool,
    /// Per-checker results for this case.
    pub checker_results: Vec<(String, bool)>,
    /// Truncated graph output for failed cases (so the meta-agent can see *why* it failed).
    pub output_excerpt: Option<String>,
    /// Full graph output / checker output for failed cases.
    pub raw_output: Option<String>,
    /// The model's actual response (e.g. generated code) — lets the meta-agent see
    /// how the prompt was interpreted, not just that it failed.
    pub model_response: Option<String>,
    /// Per-step outputs captured during graph execution (step_name → output).
    pub step_trace: Vec<(String, String)>,
}

/// A candidate in the search archive.
#[derive(Debug, Clone)]
pub struct Candidate {
    /// Unique candidate ID.
    pub id: usize,
    /// Parent candidate ID (0 = seed).
    pub parent_id: Option<usize>,
    /// The graph IR for this candidate.
    pub graph: GraphIR,
    /// Tunable overrides.
    pub overrides: TunableOverrides,
    /// Mutations applied from parent.
    pub mutations: Vec<Mutation>,
    /// Evaluation score (None if not yet evaluated).
    pub score: Option<f64>,
    /// Per-metric scores.
    pub metric_scores: HashMap<String, f64>,
    /// Graph structural descriptor.
    pub descriptor: GraphDescriptor,
    /// Number of times this candidate has been selected as a parent.
    pub children_count: usize,
    /// Meta-agent reasoning (if mutation was proposed by the meta-agent).
    pub meta_reasoning: Option<String>,
    /// Per-case evaluation results (for meta-agent learning).
    pub case_results: Vec<CaseResult>,
    /// Total dataset cases evaluated.
    pub total_cases: usize,
    /// Number of cases that passed the primary checker.
    pub cases_passed: usize,
    /// IDs of cases that passed (for computing diffs between candidates).
    pub passed_case_ids: Vec<String>,
}

/// The optimization archive.
pub struct Archive {
    pub candidates: Vec<Candidate>,
    next_id: usize,
}

impl Archive {
    pub fn new() -> Self {
        Self {
            candidates: Vec::new(),
            next_id: 0,
        }
    }

    pub fn add(&mut self, candidate: Candidate) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        self.candidates.push(Candidate { id, ..candidate });
        id
    }

    /// Get the best candidate by score.
    pub fn best(&self) -> Option<&Candidate> {
        self.candidates
            .iter()
            .filter(|c| c.score.is_some())
            .max_by(|a, b| {
                a.score
                    .unwrap()
                    .partial_cmp(&b.score.unwrap())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Increment the children count for a parent candidate.
    pub fn increment_children(&mut self, parent_id: usize) {
        if let Some(c) = self.candidates.iter_mut().find(|c| c.id == parent_id) {
            c.children_count += 1;
        }
    }

    /// Get evaluated candidates sorted by score descending.
    pub fn ranked(&self) -> Vec<&Candidate> {
        let mut evaluated: Vec<&Candidate> = self
            .candidates
            .iter()
            .filter(|c| c.score.is_some())
            .collect();
        evaluated.sort_by(|a, b| {
            b.score
                .unwrap()
                .partial_cmp(&a.score.unwrap())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        evaluated
    }
}

/// Optimization report.
#[derive(Debug, Clone, serde::Serialize)]
pub struct OptimizationReport {
    pub objective_name: String,
    pub total_candidates: usize,
    pub best_score: Option<f64>,
    pub best_candidate_id: Option<usize>,
    pub candidate_scores: Vec<CandidateScore>,
    /// Best candidate's overrides (rewritten prompts, config changes).
    /// Persisted so the winning changes are inspectable after the run.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub best_overrides: Option<HashMap<String, serde_json::Value>>,
    /// Best candidate's per-metric scores.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub best_metric_scores: Option<HashMap<String, f64>>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct CandidateScore {
    pub id: usize,
    pub parent_id: Option<usize>,
    pub score: Option<f64>,
    pub mutations: Vec<String>,
    pub node_count: u64,
    pub verify_count: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub meta_reasoning: Option<String>,
}

/// Result of optimizing a single objective (report + best graph + overrides).
pub struct OptimizationResult {
    pub report: OptimizationReport,
    pub best_graph: Option<GraphIR>,
    pub best_overrides: Option<HashMap<String, serde_json::Value>>,
}

/// Hierarchical optimization report.
#[derive(Debug, Clone, serde::Serialize)]
pub struct HierarchicalReport {
    pub objective_name: String,
    pub sub_reports: Vec<SubOptimizationReport>,
    pub parent_report: OptimizationReport,
}

/// Report for a single sub-objective optimization pass.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SubOptimizationReport {
    pub sub_name: String,
    pub graph_name: String,
    pub report: OptimizationReport,
}

/// Run the optimization loop for an objective.
pub async fn optimize(
    ir: &ScaffoldIR,
    objective_name: &str,
    options: &OptimizationOptions,
) -> Result<OptimizationReport> {
    let objective = ir
        .objectives
        .iter()
        .find(|o| o.name == objective_name)
        .ok_or_else(|| Error::Runtime(format!("objective '{}' not found", objective_name)))?;

    let graph = ir
        .graphs
        .iter()
        .find(|g| g.name == objective.graph)
        .ok_or_else(|| {
            Error::Runtime(format!(
                "graph '{}' referenced by objective not found",
                objective.graph
            ))
        })?;

    // Load dataset
    let dataset = load_dataset_from_spec(&objective.dataset)?;

    // Notify TUI of harness models and meta-agent early so the UI
    // reflects them from the very first frame (including during seeding).
    {
        let mut models: Vec<String> = ir
            .nodes
            .iter()
            .filter_map(|n| n.config.model.clone())
            .collect();
        models.sort();
        models.dedup();
        emit(options, OptEvent::HarnessInfo { models });
    }
    if let Some(ref model) = options.meta_model {
        emit(
            options,
            OptEvent::MetaAgentActive {
                model: model.clone(),
            },
        );
    }

    match options.backend {
        OptimizationBackend::Grid => run_grid_search(ir, objective, graph, &dataset, options).await,
        OptimizationBackend::Evolutionary => {
            run_evolutionary(ir, objective, graph, &dataset, options).await
        }
    }
}

/// Hierarchical optimization: optimize sub-objectives first, freeze, then optimize parent.
///
/// If the objective has no subs, delegates to flat `optimize()`.
pub async fn optimize_hierarchical(
    ir: &ScaffoldIR,
    objective_name: &str,
    options: &OptimizationOptions,
) -> Result<HierarchicalReport> {
    reset_meta_log(options)?;

    let objective = ir
        .objectives
        .iter()
        .find(|o| o.name == objective_name)
        .ok_or_else(|| Error::Runtime(format!("objective '{}' not found", objective_name)))?;

    if objective.subs.is_empty() {
        // No subs — run flat optimization and wrap in hierarchical report
        let report = optimize(ir, objective_name, options).await?;
        return Ok(HierarchicalReport {
            objective_name: objective_name.to_string(),
            sub_reports: vec![],
            parent_report: report,
        });
    }

    // Compute dependency order for subs
    let ordered_subs = compute_sub_dependency_order(&objective.subs, ir)?;

    let mut working_ir = ir.clone();
    let mut sub_reports = Vec::new();
    let mut frozen_step_names: Vec<String> = Vec::new();

    // Optimize each sub in dependency order
    for sub_idx in ordered_subs {
        let sub = &objective.subs[sub_idx];
        emit(
            options,
            OptEvent::SubObjectiveStarted {
                sub_name: sub.name.clone(),
                graph_name: sub.graph.clone(),
            },
        );
        if options.event_tx.is_none() {
            eprintln!(
                "[optimizer] optimizing sub '{}' (graph: '{}')",
                sub.name, sub.graph
            );
        }

        // Convert sub to a temporary objective for the flat optimizer
        let temp_objective = sub_to_objective_ir(sub);
        let temp_name = temp_objective.name.clone();

        // Insert the temporary objective into working IR
        working_ir.objectives.push(temp_objective);

        let result = optimize_and_extract(&working_ir, &temp_name, options).await?;

        // Remove the temporary objective
        working_ir.objectives.retain(|o| o.name != temp_name);

        emit(
            options,
            OptEvent::SubObjectiveCompleted {
                sub_name: sub.name.clone(),
                best_score: result.report.best_score,
                total_candidates: result.report.total_candidates,
            },
        );

        sub_reports.push(SubOptimizationReport {
            sub_name: sub.name.clone(),
            graph_name: sub.graph.clone(),
            report: result.report,
        });

        // Freeze best graph into the working IR
        if let Some(best_graph) = result.best_graph {
            for g in &mut working_ir.graphs {
                if g.name == sub.graph {
                    *g = best_graph.clone();
                    break;
                }
            }
            // Collect step names that call this sub's graph for auto-preserve
            collect_steps_calling_graph(
                &working_ir,
                &objective.graph,
                &sub.graph,
                &mut frozen_step_names,
            );
        }

        // Bake sub-objective's best overrides into the IR node configs.
        // This ensures the parent phase starts with the sub's optimized
        // prompts/configs as the new defaults, rather than losing them.
        if let Some(overrides) = result.best_overrides {
            bake_overrides_into_ir(&mut working_ir, &overrides);
        }
    }

    // Auto-add frozen subgraph step names to parent's topology.preserve
    // We do this by modifying the objective in working_ir
    // But since optimize() takes objective by name from IR, we need to update it in place
    {
        if let Some(obj) = working_ir
            .objectives
            .iter_mut()
            .find(|o| o.name == objective_name)
        {
            let topo = obj.topology.get_or_insert_with(|| TopologyIR {
                mutations: vec![],
                max_nodes: None,
                max_depth: None,
                preserve: vec![],
                target_score: None,
            });
            for step_name in &frozen_step_names {
                if !topo.preserve.contains(step_name) {
                    topo.preserve.push(step_name.clone());
                }
            }
        }
    }

    emit(
        options,
        OptEvent::ParentPhaseStarted {
            objective_name: objective_name.to_string(),
        },
    );
    if options.event_tx.is_none() {
        eprintln!(
            "[optimizer] optimizing parent objective '{}' with frozen children",
            objective_name
        );
    }

    let parent_report = optimize(&working_ir, objective_name, options).await?;

    Ok(HierarchicalReport {
        objective_name: objective_name.to_string(),
        sub_reports,
        parent_report,
    })
}

/// Compute topological order for sub-objectives based on graph call dependencies.
///
/// Returns indices into the subs Vec in dependency order (leaves first).
fn compute_sub_dependency_order(subs: &[SubObjectiveIR], ir: &ScaffoldIR) -> Result<Vec<usize>> {
    let graph_names: std::collections::HashSet<String> =
        ir.graphs.iter().map(|g| g.name.clone()).collect();

    // Map sub graph name → sub index
    let sub_graph_to_idx: HashMap<String, usize> = subs
        .iter()
        .enumerate()
        .map(|(i, s)| (s.graph.clone(), i))
        .collect();

    // Build adjacency: sub_idx → set of sub_idx it depends on
    let mut deps: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut in_degree: HashMap<usize, usize> = HashMap::new();

    for (i, sub) in subs.iter().enumerate() {
        deps.entry(i).or_default();
        in_degree.entry(i).or_insert(0);

        // Find which graphs this sub's graph transitively calls
        if let Some(graph) = ir.graphs.iter().find(|g| g.name == sub.graph) {
            let mut called = std::collections::HashSet::new();
            crate::mutations::collect_graph_refs_from_body(&graph.body, &graph_names, &mut called);

            for called_graph in &called {
                if let Some(&dep_idx) = sub_graph_to_idx.get(called_graph) {
                    if dep_idx != i {
                        deps.entry(i).or_default().push(dep_idx);
                        *in_degree.entry(i).or_insert(0) += 0; // ensure entry exists
                                                               // dep_idx is depended on by i, so i has an incoming edge
                    }
                }
            }
        }
    }

    // Kahn's algorithm
    // We want leaves first: nodes with no dependencies come first
    // Recompute in_degree properly: in_degree[j] = number of subs that j depends on
    let mut real_in_degree: HashMap<usize, usize> = (0..subs.len()).map(|i| (i, 0)).collect();
    for (i, dep_list) in &deps {
        // sub i depends on each dep in dep_list
        // But we want to process leaves first, so in_degree = number of things I depend on
        real_in_degree.insert(*i, dep_list.len());
    }

    let mut queue: std::collections::VecDeque<usize> = real_in_degree
        .iter()
        .filter(|(_, &deg)| deg == 0)
        .map(|(&idx, _)| idx)
        .collect();

    let mut order = Vec::new();
    while let Some(idx) = queue.pop_front() {
        order.push(idx);
        // For each sub that depends on idx, decrement its in-degree
        for (other_idx, dep_list) in &deps {
            if dep_list.contains(&idx) {
                let deg = real_in_degree.get_mut(other_idx).unwrap();
                *deg -= 1;
                if *deg == 0 {
                    queue.push_back(*other_idx);
                }
            }
        }
    }

    if order.len() != subs.len() {
        return Err(Error::Runtime(
            "cycle detected among sub-objective graph dependencies".to_string(),
        ));
    }

    Ok(order)
}

/// Convert a SubObjectiveIR to a full ObjectiveIR for the flat optimizer.
fn sub_to_objective_ir(sub: &SubObjectiveIR) -> ObjectiveIR {
    ObjectiveIR {
        name: format!("__sub_{}", sub.name),
        graph: sub.graph.clone(),
        dataset: sub.dataset.clone(),
        checkers: sub.checkers.clone(),
        judges: sub.judges.clone(),
        metrics: sub.metrics.clone(),
        score: sub.score.clone(),
        repeats: sub.repeats,
        split: sub.split.clone(),
        select: sub.select.clone(),
        tunables: sub.tunables.clone(),
        topology: sub.topology.clone(),
        subs: vec![],
    }
}

/// Run optimize() and extract both the report and the best graph.
async fn optimize_and_extract(
    ir: &ScaffoldIR,
    objective_name: &str,
    options: &OptimizationOptions,
) -> Result<OptimizationResult> {
    let objective = ir
        .objectives
        .iter()
        .find(|o| o.name == objective_name)
        .ok_or_else(|| Error::Runtime(format!("objective '{}' not found", objective_name)))?;

    let graph = ir
        .graphs
        .iter()
        .find(|g| g.name == objective.graph)
        .ok_or_else(|| {
            Error::Runtime(format!(
                "graph '{}' referenced by objective not found",
                objective.graph
            ))
        })?;

    let dataset = load_dataset_from_spec(&objective.dataset)?;

    let mut archive = Archive::new();

    match options.backend {
        OptimizationBackend::Grid => {
            run_grid_search_into(ir, objective, graph, &dataset, options, &mut archive).await?;
        }
        OptimizationBackend::Evolutionary => {
            run_evolutionary_into(ir, objective, graph, &dataset, options, &mut archive).await?;
        }
    }

    let best_graph = archive.best().map(|c| c.graph.clone());
    let best_overrides = archive
        .best()
        .map(|c| c.overrides.clone())
        .filter(|o| !o.is_empty());
    let report = build_report(ir, objective, &archive, options)?;

    Ok(OptimizationResult {
        report,
        best_graph,
        best_overrides,
    })
}

/// Collect step names in parent_graph that call the given sub_graph.
fn collect_steps_calling_graph(
    ir: &ScaffoldIR,
    parent_graph_name: &str,
    sub_graph_name: &str,
    step_names: &mut Vec<String>,
) {
    if let Some(graph) = ir.graphs.iter().find(|g| g.name == parent_graph_name) {
        collect_steps_calling_graph_in_body(&graph.body, sub_graph_name, step_names);
    }
}

fn collect_steps_calling_graph_in_body(
    stmts: &[GraphStmtIR],
    sub_graph_name: &str,
    step_names: &mut Vec<String>,
) {
    for stmt in stmts {
        match stmt {
            GraphStmtIR::Step(s) => {
                if s.node == sub_graph_name && !step_names.contains(&s.name) {
                    step_names.push(s.name.clone());
                }
            }
            GraphStmtIR::Loop(l) => {
                collect_steps_calling_graph_in_body(&l.body, sub_graph_name, step_names);
            }
            GraphStmtIR::If(i) => {
                collect_steps_calling_graph_in_body(&i.then_body, sub_graph_name, step_names);
                collect_steps_calling_graph_in_body(&i.else_body, sub_graph_name, step_names);
            }
            GraphStmtIR::Parallel(p) => {
                collect_steps_calling_graph_in_body(&p.body, sub_graph_name, step_names);
            }
            _ => {}
        }
    }
}

/// Bake runtime overrides into the IR node configs so they become the new defaults.
///
/// This also materializes synthetic nodes created by `AddPromptStep` so the baked IR
/// remains executable without needing runtime-only `_node.*` overrides.
///
/// Override keys follow the pattern `"node_name.field"` where field is one of:
/// template, system, shell, temperature, model, max_tokens, max_turns, timeout.
fn bake_overrides_into_ir(ir: &mut ScaffoldIR, overrides: &HashMap<String, serde_json::Value>) {
    use scaffold_ir::ir::{NodeConfigIR, NodeIR, NodeKindIR, StringOrFileIR, TypeIR};

    for (key, value) in overrides {
        if let Some(node_name) = key.strip_prefix("_node.") {
            if ir.nodes.iter().any(|n| n.name == node_name) {
                continue;
            }

            let kind = value
                .as_object()
                .and_then(|obj| obj.get("kind"))
                .and_then(|kind| kind.as_str())
                .map(|kind| match kind {
                    "tool" => NodeKindIR::Tool,
                    "agent" => NodeKindIR::Agent,
                    "verify" => NodeKindIR::Verify,
                    _ => NodeKindIR::Prompt,
                })
                .unwrap_or(NodeKindIR::Prompt);

            ir.nodes.push(NodeIR {
                name: node_name.to_string(),
                kind,
                input: TypeIR::String,
                output: TypeIR::String,
                config: NodeConfigIR::default(),
            });
        }
    }

    for (key, value) in overrides {
        let parts: Vec<&str> = key.splitn(2, '.').collect();
        if parts.len() != 2 {
            continue;
        }
        let (node_name, field) = (parts[0], parts[1]);

        if let Some(node) = ir.nodes.iter_mut().find(|n| n.name == node_name) {
            match field {
                "template" => {
                    if let Some(s) = value.as_str() {
                        node.config.template = Some(StringOrFileIR::Literal {
                            value: s.to_string(),
                        });
                    }
                }
                "system" => {
                    if let Some(s) = value.as_str() {
                        node.config.system = Some(StringOrFileIR::Literal {
                            value: s.to_string(),
                        });
                    }
                }
                "shell" => {
                    if let Some(s) = value.as_str() {
                        node.config.shell = Some(s.to_string());
                    }
                }
                "temperature" => {
                    node.config.temperature = value.as_f64();
                }
                "model" => {
                    if let Some(s) = value.as_str() {
                        node.config.model = Some(s.to_string());
                    }
                }
                "max_tokens" => {
                    node.config.max_tokens = value.as_u64();
                }
                "max_turns" => {
                    node.config.max_turns = value.as_u64();
                }
                "timeout" => {
                    node.config.timeout = value.as_u64();
                }
                _ => {} // Unknown field, skip
            }
        }
    }
}

fn materialize_best_candidate_ir(
    ir: &ScaffoldIR,
    objective: &ObjectiveIR,
    best: &Candidate,
) -> ScaffoldIR {
    let mut best_ir = ir.clone();

    if let Some(graph) = best_ir
        .graphs
        .iter_mut()
        .find(|graph| graph.name == objective.graph)
    {
        *graph = best.graph.clone();
    } else {
        best_ir.graphs.push(best.graph.clone());
    }

    bake_overrides_into_ir(&mut best_ir, &best.overrides);
    best_ir
}

/// A single dataset case for evaluation.
#[derive(Debug, Clone)]
pub struct DatasetCase {
    pub input: Value,
    pub expected: Value,
    pub id: Option<String>,
}

/// Load dataset from objective spec.
pub fn load_dataset_from_spec(spec: &DatasetSpecIR) -> Result<Vec<DatasetCase>> {
    match spec {
        DatasetSpecIR::Inline { cases } => {
            let mut result = Vec::new();
            for case in cases {
                result.push(DatasetCase {
                    input: expr_to_value(&case.input),
                    expected: expr_to_value(&case.expected),
                    id: case.id.clone(),
                });
            }
            Ok(result)
        }
        DatasetSpecIR::File { path } => {
            let content = std::fs::read_to_string(path).map_err(|e| {
                Error::Runtime(format!("failed to read dataset file '{}': {}", path, e))
            })?;
            // Parse JSONL format
            let mut cases = Vec::new();
            for (i, line) in content.lines().enumerate() {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                let json: serde_json::Value = serde_json::from_str(line).map_err(|e| {
                    Error::Runtime(format!("failed to parse line {} of dataset: {}", i + 1, e))
                })?;
                let input = json
                    .get("input")
                    .cloned()
                    .map(Value::from)
                    .unwrap_or(Value::Null);
                let expected = json
                    .get("expected")
                    .cloned()
                    .map(Value::from)
                    .unwrap_or(Value::Null);
                let id = json
                    .get("id")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                cases.push(DatasetCase {
                    input,
                    expected,
                    id,
                });
            }
            Ok(cases)
        }
    }
}

/// Convert an ExprIR literal to a Value (for inline dataset cases).
fn expr_to_value(expr: &ExprIR) -> Value {
    match expr {
        ExprIR::LitInt { value } => Value::Int(*value),
        ExprIR::LitFloat { value } => Value::Float(*value),
        ExprIR::LitString { value } => Value::String(value.clone()),
        ExprIR::LitBool { value } => Value::Bool(*value),
        ExprIR::LitNull => Value::Null,
        ExprIR::List { elements } => Value::List(elements.iter().map(expr_to_value).collect()),
        ExprIR::Record { fields } => {
            let map: HashMap<String, Value> = fields
                .iter()
                .map(|f| (f.key.clone(), expr_to_value(&f.value)))
                .collect();
            Value::Map(map)
        }
        _ => Value::Null,
    }
}

/// Evaluate a candidate on a dataset, returning the aggregate score.
///
/// `display_id` is the ID shown in TUI events (since `candidate.id` is always
/// 0 before `archive.add()` assigns the real one).
async fn evaluate_candidate(
    ir: &ScaffoldIR,
    objective: &ObjectiveIR,
    candidate: &Candidate,
    dataset: &[DatasetCase],
    options: &OptimizationOptions,
    display_id: usize,
) -> Result<(
    f64,
    HashMap<String, f64>,
    Vec<CaseResult>,
    usize,
    usize,
    Vec<String>,
)> {
    use futures::stream::{self, StreamExt};

    // Build a modified IR with the candidate's graph
    let mut modified_ir = ir.clone();
    for g in &mut modified_ir.graphs {
        if g.name == objective.graph {
            *g = candidate.graph.clone();
        }
    }

    let executor = GraphExecutor::new(modified_ir).with_overrides(candidate.overrides.clone());

    let case_count = dataset.len();
    if case_count == 0 {
        return Ok((0.0, HashMap::new(), vec![], 0, 0, vec![]));
    }

    let concurrency = options.concurrency.max(1);

    emit(
        options,
        OptEvent::EvaluationStarted {
            candidate_id: display_id,
            total_cases: case_count,
        },
    );

    // Find the primary checker name (used to determine per-case pass/fail in TUI).
    // Resolve: score expr → metric name → checker name.
    let primary_checker = match &objective.score {
        ExprIR::Ident { name } => objective
            .metrics
            .iter()
            .find(|m| m.name == *name)
            .map(|m| m.checker.clone()),
        _ => None,
    };

    let mut checker_totals: HashMap<String, f64> = HashMap::new();
    let mut case_results: Vec<CaseResult> = Vec::with_capacity(case_count);
    let mut completed = 0usize;
    let mut cases_passed = 0usize;
    let mut passed_case_ids: Vec<String> = Vec::new();

    // Run cases concurrently with controlled parallelism.
    // buffer_unordered polls up to `concurrency` futures at once on the
    // same task (no Send required), yielding results as they complete.
    let graph_name = &objective.graph;
    let executor_ref = &executor;
    // Identify prompt/agent node names so we can extract model responses from step traces.
    let prompt_node_names: Vec<&str> = ir
        .nodes
        .iter()
        .filter(|n| matches!(n.kind, NodeKindIR::Prompt | NodeKindIR::Agent))
        .map(|n| n.name.as_str())
        .collect();
    // Map step → node for the graph, so we know which steps are prompt calls.
    let step_node_map: HashMap<String, String> = {
        let graph_ir = ir
            .graphs
            .iter()
            .find(|g| g.name == objective.graph)
            .unwrap();
        let step_names = crate::mutations::collect_step_names(&graph_ir.body);
        step_names
            .into_iter()
            .filter_map(|s| {
                crate::meta_agent::find_step_node_pub(&graph_ir.body, &s).map(|n| (s, n))
            })
            .collect()
    };
    let futs = dataset.iter().enumerate().map(|(idx, case)| {
        let input = case.input.clone();
        async move {
            // Use traced execution to capture intermediate step outputs
            let mut result = executor_ref
                .execute_graph_traced(graph_name, input.clone())
                .await;
            for retry in 0..2 {
                if let Err(ref e) = result {
                    let msg = e.to_string().to_lowercase();
                    let is_transient = msg.contains("http")
                        || msg.contains("429")
                        || msg.contains("500")
                        || msg.contains("502")
                        || msg.contains("503")
                        || msg.contains("504")
                        || msg.contains("rate")
                        || msg.contains("timed out")
                        || msg.contains("connection");
                    if is_transient {
                        tokio::time::sleep(std::time::Duration::from_secs(2u64.pow(retry + 1)))
                            .await;
                        result = executor_ref
                            .execute_graph_traced(graph_name, input.clone())
                            .await;
                        continue;
                    }
                }
                break;
            }
            (idx, result)
        }
    });

    let mut stream = stream::iter(futs).buffer_unordered(concurrency);

    while let Some((idx, result)) = stream.next().await {
        let case = &dataset[idx];
        completed += 1;

        let (passed, checker_bits, output_excerpt, raw_output, model_response, step_trace) =
            match result {
                Ok((output, step_trace)) => {
                    let mut primary_pass = true;
                    let mut all_pass = !objective.checkers.is_empty();
                    let mut bits = Vec::with_capacity(objective.checkers.len());
                    for checker in &objective.checkers {
                        let val =
                            eval_checker_expr(&executor, &checker.expr, &output, &case.expected);
                        *checker_totals.entry(checker.name.clone()).or_default() += val;
                        let ok = val >= 1.0;
                        bits.push((checker.name.clone(), ok));
                        if !ok {
                            all_pass = false;
                        }
                        if primary_checker.as_deref() == Some(&checker.name) {
                            primary_pass = ok;
                        }
                    }
                    let p = if primary_checker.is_some() {
                        primary_pass
                    } else {
                        all_pass
                    };
                    // For failed cases, extract test output and model response from step trace
                    let (excerpt, raw, model_resp) = if !p {
                        let out_str = output.to_string();
                        let truncated: String = if out_str.len() > 300 {
                            let s: String = out_str.chars().take(300).collect();
                            format!("{}...", s)
                        } else {
                            out_str.clone()
                        };
                        emit(
                            options,
                            OptEvent::Log {
                                message: format!(
                                    "Case {} FAILED — output: {}",
                                    case.id.as_deref().unwrap_or("?"),
                                    truncated,
                                ),
                            },
                        );
                        // Extract test_output from JSON if possible
                        let useful =
                            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&out_str) {
                                json.get("test_output")
                                    .and_then(|v| v.as_str())
                                    .map(|s| s.to_string())
                                    .unwrap_or(out_str.clone())
                            } else {
                                out_str.clone()
                            };
                        let max_len = 1500;
                        let excerpt = if useful.len() > max_len {
                            Some(format!("{}...", &useful[..max_len]))
                        } else {
                            Some(useful.clone())
                        };
                        // Extract model response from step trace: find the LAST prompt node output
                        let model_resp = step_trace
                            .iter()
                            .rev()
                            .find(|(step_name, _)| {
                                step_node_map
                                    .get(step_name)
                                    .map(|node| prompt_node_names.contains(&node.as_str()))
                                    .unwrap_or(false)
                            })
                            .map(|(step_name, value)| {
                                let max = 500;
                                let truncated: String = value.chars().take(max).collect();
                                if truncated.len() < value.len() {
                                    format!("[{}] {}...", step_name, truncated)
                                } else {
                                    format!("[{}] {}", step_name, value)
                                }
                            });
                        (excerpt, Some(useful), model_resp)
                    } else {
                        (None, None, None)
                    };
                    let trace = if !p { step_trace } else { Vec::new() };
                    (p, bits, excerpt, raw, model_resp, trace)
                }
                Err(e) => {
                    // Log graph execution errors so template / rendering failures
                    // are visible instead of silently scoring 0.
                    let err_str = format!("execution error: {}", e);
                    emit(
                        options,
                        OptEvent::Log {
                            message: format!(
                                "Case {} {}",
                                case.id.as_deref().unwrap_or("?"),
                                err_str,
                            ),
                        },
                    );
                    let bits: Vec<_> = objective
                        .checkers
                        .iter()
                        .map(|c| (c.name.clone(), false))
                        .collect();
                    (
                        false,
                        bits,
                        Some(err_str.clone()),
                        Some(err_str),
                        None,
                        Vec::new(),
                    )
                }
            };

        if passed {
            cases_passed += 1;
            if let Some(ref id) = case.id {
                passed_case_ids.push(id.clone());
            }
        }

        // Only store failed cases — these are what the meta-agent learns from.
        if !passed {
            case_results.push(CaseResult {
                case_id: case.id.clone(),
                passed,
                checker_results: checker_bits,
                output_excerpt,
                raw_output,
                model_response,
                step_trace,
            });
        }

        emit(
            options,
            OptEvent::CaseCompleted {
                candidate_id: display_id,
                case_index: completed,
                total_cases: case_count,
                passed,
                case_id: case.id.clone(),
            },
        );
    }

    // Compute metric averages (each metric references a checker)
    let mut metric_avgs: HashMap<String, f64> = HashMap::new();
    for metric in &objective.metrics {
        let checker_total = checker_totals.get(&metric.checker).copied().unwrap_or(0.0);
        metric_avgs.insert(metric.name.clone(), checker_total / case_count as f64);
    }

    // Evaluate the score expression with metric averages in scope
    let score_scope = Scope::with_bindings(
        metric_avgs
            .iter()
            .map(|(k, v)| (k.as_str(), Value::Float(*v)))
            .collect(),
    );
    let score = match executor.eval_expr(&objective.score, &score_scope) {
        Ok(Value::Float(f)) => f,
        Ok(Value::Int(i)) => i as f64,
        Ok(Value::Bool(true)) => 1.0,
        Ok(Value::Bool(false)) => 0.0,
        _ => 0.0,
    };

    Ok((
        score,
        metric_avgs,
        case_results,
        case_count,
        cases_passed,
        passed_case_ids,
    ))
}

/// Evaluate a checker expression with `output` and `expected` in scope.
/// Returns 1.0 for true, 0.0 for false.
pub fn eval_checker_expr(
    executor: &GraphExecutor,
    expr: &ExprIR,
    output: &Value,
    expected: &Value,
) -> f64 {
    let scope = Scope::with_bindings(vec![
        ("output", output.clone()),
        ("expected", expected.clone()),
    ]);
    match executor.eval_expr(expr, &scope) {
        Ok(Value::Bool(true)) => 1.0,
        Ok(Value::Bool(false)) => 0.0,
        Ok(Value::Float(f)) => f,
        Ok(Value::Int(i)) => i as f64,
        _ => 0.0,
    }
}

// ── Grid Search ──

async fn run_grid_search(
    ir: &ScaffoldIR,
    objective: &ObjectiveIR,
    graph: &GraphIR,
    dataset: &[DatasetCase],
    options: &OptimizationOptions,
) -> Result<OptimizationReport> {
    let mut archive = Archive::new();
    run_grid_search_into(ir, objective, graph, dataset, options, &mut archive).await?;
    build_report(ir, objective, &archive, options)
}

async fn run_grid_search_into(
    ir: &ScaffoldIR,
    objective: &ObjectiveIR,
    graph: &GraphIR,
    dataset: &[DatasetCase],
    options: &OptimizationOptions,
    archive: &mut Archive,
) -> Result<()> {
    // Generate all tunable combinations
    let combinations = generate_tunable_combinations(&objective.tunables);

    for (i, overrides) in combinations.iter().enumerate() {
        if i >= options.max_candidates {
            break;
        }

        let candidate = Candidate {
            id: 0,
            parent_id: None,
            graph: graph.clone(),
            overrides: overrides.clone(),
            mutations: vec![],
            score: None,
            metric_scores: HashMap::new(),
            descriptor: GraphDescriptor::from_graph(graph, ir),
            children_count: 0,
            meta_reasoning: None,
            case_results: vec![],
            total_cases: 0,
            cases_passed: 0,
            passed_case_ids: vec![],
        };

        let cand_id = archive.candidates.len();
        let (score, metrics, cases, total, passed, passed_ids) =
            evaluate_candidate(ir, objective, &candidate, dataset, options, cand_id).await?;
        let best_before = archive.best().and_then(|c| c.score).unwrap_or(0.0);

        emit(
            options,
            OptEvent::CandidateEvaluated {
                candidate_id: cand_id,
                parent_id: None,
                score,
                metric_scores: metrics.clone(),
                best_so_far: best_before.max(score),
                mutations: vec![],
                generation: i + 1,
                max_generations: combinations.len().min(options.max_candidates),
            },
        );
        if options.event_tx.is_none() {
            eprintln!(
                "[optimizer] candidate {}/{}: score={:.4}",
                i + 1,
                combinations.len().min(options.max_candidates),
                score
            );
        }

        archive.add(Candidate {
            score: Some(score),
            metric_scores: metrics,
            case_results: cases,
            total_cases: total,
            cases_passed: passed,
            passed_case_ids: passed_ids,
            ..candidate
        });
    }

    Ok(())
}

/// Generate all combinations of tunable values.
fn generate_tunable_combinations(tunables: &[TunableIR]) -> Vec<TunableOverrides> {
    if tunables.is_empty() {
        return vec![HashMap::new()];
    }

    let mut all_combos = vec![HashMap::new()];

    for tunable in tunables {
        let path_key = tunable.path.join(".");
        let values: Vec<serde_json::Value> = tunable
            .domain
            .iter()
            .map(|e| expr_to_json_value(e))
            .collect();

        let mut new_combos = Vec::new();
        for combo in &all_combos {
            for val in &values {
                let mut new_combo = combo.clone();
                new_combo.insert(path_key.clone(), val.clone());
                new_combos.push(new_combo);
            }
        }
        all_combos = new_combos;
    }

    all_combos
}

fn expr_to_json_value(expr: &ExprIR) -> serde_json::Value {
    match expr {
        ExprIR::LitInt { value } => serde_json::json!(value),
        ExprIR::LitFloat { value } => serde_json::json!(value),
        ExprIR::LitString { value } => serde_json::json!(value),
        ExprIR::LitBool { value } => serde_json::json!(value),
        ExprIR::LitNull => serde_json::Value::Null,
        _ => serde_json::Value::Null,
    }
}

// ── Evolutionary Search ──

async fn run_evolutionary(
    ir: &ScaffoldIR,
    objective: &ObjectiveIR,
    graph: &GraphIR,
    dataset: &[DatasetCase],
    options: &OptimizationOptions,
) -> Result<OptimizationReport> {
    let mut archive = Archive::new();
    run_evolutionary_into(ir, objective, graph, dataset, options, &mut archive).await?;
    build_report(ir, objective, &archive, options)
}

async fn run_evolutionary_into(
    ir: &ScaffoldIR,
    objective: &ObjectiveIR,
    graph: &GraphIR,
    dataset: &[DatasetCase],
    options: &OptimizationOptions,
    archive: &mut Archive,
) -> Result<()> {
    let mut rng = SimpleRng::new(42);

    let topology = objective.topology.as_ref();
    let target_score = topology.and_then(|t| t.target_score);

    // ── Phase 1: Seed candidates (uncapped) ──
    // The original graph is always the first seed.
    let seed = Candidate {
        id: 0,
        parent_id: None,
        graph: graph.clone(),
        overrides: HashMap::new(),
        mutations: vec![],
        score: None,
        metric_scores: HashMap::new(),
        descriptor: GraphDescriptor::from_graph(graph, ir),
        children_count: 0,
        meta_reasoning: None,
        case_results: vec![],
        total_cases: 0,
        cases_passed: 0,
        passed_case_ids: vec![],
    };

    let next_id = archive.candidates.len();
    let (seed_score, seed_metrics, seed_cases, seed_total, seed_passed, seed_passed_ids) =
        evaluate_candidate(ir, objective, &seed, dataset, options, next_id).await?;
    let seed_id = archive.add(Candidate {
        score: Some(seed_score),
        metric_scores: seed_metrics,
        case_results: seed_cases,
        total_cases: seed_total,
        cases_passed: seed_passed,
        passed_case_ids: seed_passed_ids,
        ..seed
    });

    emit(
        options,
        OptEvent::PhaseChanged {
            phase: OptPhase::Seeding,
        },
    );
    emit(
        options,
        OptEvent::CandidateEvaluated {
            candidate_id: seed_id,
            parent_id: None,
            score: seed_score,
            metric_scores: archive
                .candidates
                .last()
                .map(|c| c.metric_scores.clone())
                .unwrap_or_default(),
            best_so_far: seed_score,
            mutations: vec![],
            generation: 0,
            max_generations: options.max_candidates,
        },
    );
    if options.event_tx.is_none() {
        eprintln!("[optimizer] seed candidate: score={:.4}", seed_score);
    }

    // Check early stop after seed
    if let Some(ts) = target_score {
        if seed_score >= ts {
            emit(
                options,
                OptEvent::EarlyStopped {
                    score: seed_score,
                    target: ts,
                },
            );
            if options.event_tx.is_none() {
                eprintln!(
                    "[optimizer] target score {:.4} reached by seed, stopping early",
                    ts
                );
            }
            return Ok(());
        }
    }

    // Seed with all tunable combinations (no cap — this is exhaustive parameter sweep).
    // Skip when meta-agent is active — it will explore configs intelligently.
    let tunable_combos = if options.meta_model.is_some() {
        vec![]
    } else {
        generate_tunable_combinations(&objective.tunables)
    };
    for overrides in &tunable_combos {
        let candidate = Candidate {
            id: 0,
            parent_id: Some(seed_id),
            graph: graph.clone(),
            overrides: overrides.clone(),
            mutations: vec![],
            score: None,
            metric_scores: HashMap::new(),
            descriptor: GraphDescriptor::from_graph(graph, ir),
            children_count: 0,
            meta_reasoning: None,
            case_results: vec![],
            total_cases: 0,
            cases_passed: 0,
            passed_case_ids: vec![],
        };
        let tunable_id = archive.candidates.len();
        let (score, metrics, cases, total, passed, passed_ids) =
            evaluate_candidate(ir, objective, &candidate, dataset, options, tunable_id).await?;
        let best_before = archive.best().and_then(|c| c.score).unwrap_or(0.0);

        archive.add(Candidate {
            score: Some(score),
            metric_scores: metrics.clone(),
            case_results: cases,
            total_cases: total,
            cases_passed: passed,
            passed_case_ids: passed_ids,
            ..candidate
        });

        emit(
            options,
            OptEvent::PhaseChanged {
                phase: OptPhase::TunableSweep,
            },
        );
        emit(
            options,
            OptEvent::CandidateEvaluated {
                candidate_id: tunable_id,
                parent_id: Some(seed_id),
                score,
                metric_scores: metrics,
                best_so_far: best_before.max(score),
                mutations: vec![],
                generation: tunable_id,
                max_generations: tunable_combos.len() + 1,
            },
        );
        if options.event_tx.is_none() {
            eprintln!(
                "[optimizer] tunable seed {}: score={:.4}",
                tunable_id, score
            );
        }

        // Check early stop after each tunable seed
        if let Some(ts) = target_score {
            if score >= ts {
                emit(options, OptEvent::EarlyStopped { score, target: ts });
                if options.event_tx.is_none() {
                    eprintln!("[optimizer] target score {:.4} reached during tunable sweep, stopping early", ts);
                }
                return Ok(());
            }
        }
    }

    // ── Phase 2: Evolutionary mutations ──
    // max_candidates = number of successful evolutionary generations (not total archive size).
    // Skipped/failed mutations do NOT count against the budget.
    let mut allowed_mutations = topology.map(|t| t.mutations.clone()).unwrap_or_default();

    // Auto-include add_prompt_step when prompt/agent nodes exist so the optimizer
    // can actually grow the graph with synthetic prompt nodes.
    if ir
        .nodes
        .iter()
        .any(|n| matches!(n.kind, NodeKindIR::Prompt | NodeKindIR::Agent))
        && !allowed_mutations.contains(&"add_prompt_step".to_string())
    {
        allowed_mutations.push("add_prompt_step".to_string());
    }

    // insert_step only works when there is at least one node that can accept a
    // single positional input value directly. Hide it otherwise.
    if allowed_mutations.contains(&"insert_step".to_string())
        && !has_single_input_compatible_node(ir)
    {
        allowed_mutations.retain(|m| m != "insert_step");
    }

    // Auto-include set_config when tunables are declared so the meta-agent
    // (and random mutations) can propose config changes during evolution.
    if !objective.tunables.is_empty() && !allowed_mutations.contains(&"set_config".to_string()) {
        allowed_mutations.push("set_config".to_string());
    }

    if allowed_mutations.is_empty() {
        emit(
            options,
            OptEvent::Log {
                message: "no mutations configured, skipping evolutionary phase".into(),
            },
        );
        if options.event_tx.is_none() {
            eprintln!("[optimizer] no mutations configured, skipping evolutionary phase");
        }
        return Ok(());
    }

    let max_generations = options.max_candidates;
    let mut successful_generations = 0;
    // Cap total attempts to avoid infinite loops when all mutations are skipped
    let max_attempts = max_generations * 5;
    let mut total_attempts = 0;

    while successful_generations < max_generations && total_attempts < max_attempts {
        total_attempts += 1;

        // Select parent — sigmoid weighting favors high-scoring candidates.
        // Novelty bonus (down-weight overused parents) only for random mode;
        // meta-agent provides its own diversity through LLM reasoning.
        let use_novelty = options.meta_model.is_none();
        let parent = select_parent(archive, &mut rng, use_novelty);
        if parent.is_none() {
            break;
        }
        let parent = parent.unwrap();
        let parent_id = parent.id;
        let parent_graph = parent.graph.clone();
        let parent_overrides = parent.overrides.clone();

        // Generate mutation: LLM-guided or random
        let (mutation, meta_reasoning) = if let Some(ref meta_model) = options.meta_model {
            emit(options, OptEvent::MetaAgentThinking);
            // Clone parent for the meta-agent (borrow released)
            let parent_clone = parent.clone();

            let meta =
                crate::meta_agent::MetaAgent::new(meta_model).with_log(options.meta_log.clone());
            match meta
                .propose_mutation(&parent_clone, archive, ir, objective, &allowed_mutations)
                .await
            {
                Ok(proposal) => {
                    // Estimate tokens: ~4 chars per token (system + context)
                    let total_chars = proposal.context_chars + proposal.system_chars;
                    let est_tokens = total_chars / 4;
                    emit(
                        options,
                        OptEvent::MetaProposal {
                            reasoning: proposal.reasoning.clone(),
                            mutation_label: proposal.mutation.short_label(),
                            context_tokens: est_tokens,
                        },
                    );
                    emit(
                        options,
                        OptEvent::Log {
                            message: format!(
                                "Meta-agent: {} ({}) [latest ctx: ~{}k tokens]",
                                proposal.reasoning,
                                proposal.mutation.short_label(),
                                est_tokens / 1000,
                            ),
                        },
                    );
                    // Log the full content of rewrite mutations for debugging
                    match &proposal.mutation {
                        Mutation::RewritePrompt { node, new_template } => {
                            emit(
                                options,
                                OptEvent::Log {
                                    message: format!(
                                        "Rewrite template for '{}' ({} chars):\n{}",
                                        node,
                                        new_template.len(),
                                        new_template
                                    ),
                                },
                            );
                        }
                        Mutation::RewriteSystem { node, new_system } => {
                            emit(
                                options,
                                OptEvent::Log {
                                    message: format!(
                                        "Rewrite system for '{}' ({} chars):\n{}",
                                        node,
                                        new_system.len(),
                                        new_system
                                    ),
                                },
                            );
                        }
                        Mutation::RewriteShell { node, new_shell } => {
                            emit(
                                options,
                                OptEvent::Log {
                                    message: format!(
                                        "Rewrite shell for '{}' ({} chars):\n{}",
                                        node,
                                        new_shell.len(),
                                        new_shell
                                    ),
                                },
                            );
                        }
                        Mutation::AddPromptStep {
                            after_step,
                            new_step_name,
                            template,
                            ..
                        } => {
                            emit(
                                options,
                                OptEvent::Log {
                                    message: format!(
                                        "Add prompt step '{}' after '{}' ({} chars):\n{}",
                                        new_step_name,
                                        after_step,
                                        template.len(),
                                        template
                                    ),
                                },
                            );
                        }
                        Mutation::EditGraph {
                            new_graph,
                            new_nodes,
                            description,
                        } => {
                            let graph_src =
                                scaffold_ir::pretty::pretty_print_graph(new_graph);
                            emit(
                                options,
                                OptEvent::Log {
                                    message: format!(
                                        "Edit graph '{}' ({} new nodes):\n{}",
                                        description,
                                        new_nodes.len(),
                                        graph_src
                                    ),
                                },
                            );
                        }
                        _ => {}
                    }
                    (Some(proposal.mutation), Some(proposal.reasoning))
                }
                Err(e) => {
                    emit(
                        options,
                        OptEvent::Log {
                            message: format!("Meta-agent exhausted retries: {}", e),
                        },
                    );
                    continue; // skip this generation attempt
                }
            }
        } else {
            (
                generate_random_mutation(
                    &parent_graph,
                    ir,
                    &allowed_mutations,
                    &objective.tunables,
                    &mut rng,
                ),
                None,
            )
        };

        let mutation = match mutation {
            Some(m) => m,
            None => continue, // No applicable mutation found, try again
        };

        match apply_mutation(&parent_graph, &mutation, ir) {
            MutationResult::Ok(new_graph) => {
                // Check topology constraints
                if let Some(topo) = topology {
                    let violations = check_constraints(&new_graph, topo);
                    if !violations.is_empty() {
                        continue; // Constraint violation, doesn't count
                    }
                }

                // For override-based mutations, store in overrides (not graph)
                let mut overrides = parent_overrides;
                match &mutation {
                    Mutation::RewritePrompt { node, new_template } => {
                        overrides.insert(
                            format!("{}.template", node),
                            serde_json::json!(new_template),
                        );
                    }
                    Mutation::RewriteSystem { node, new_system } => {
                        overrides.insert(format!("{}.system", node), serde_json::json!(new_system));
                    }
                    Mutation::RewriteShell { node, new_shell } => {
                        overrides.insert(format!("{}.shell", node), serde_json::json!(new_shell));
                    }
                    Mutation::SetConfig { node, field, value } => {
                        overrides.insert(format!("{}.{}", node, field), value.clone());
                    }
                    Mutation::AddPromptStep {
                        new_step_name,
                        template,
                        system,
                        model,
                        ..
                    } => {
                        overrides.insert(
                            format!("_node.{}", new_step_name),
                            serde_json::json!({"kind": "prompt"}),
                        );
                        overrides.insert(
                            format!("{}.template", new_step_name),
                            serde_json::json!(template),
                        );
                        if let Some(sys) = system {
                            overrides.insert(
                                format!("{}.system", new_step_name),
                                serde_json::json!(sys),
                            );
                        }
                        if let Some(m) = model {
                            overrides
                                .insert(format!("{}.model", new_step_name), serde_json::json!(m));
                        }
                    }
                    Mutation::EditGraph { new_nodes, .. } => {
                        for node in new_nodes {
                            if ir.nodes.iter().any(|n| n.name == node.name) {
                                // Modified existing node: store field-level overrides
                                if let Some(ref sof) = node.config.template {
                                    let val = match sof {
                                        scaffold_ir::ir::StringOrFileIR::Literal { value } => {
                                            value.clone()
                                        }
                                        scaffold_ir::ir::StringOrFileIR::File { path } => {
                                            path.clone()
                                        }
                                    };
                                    overrides.insert(
                                        format!("{}.template", node.name),
                                        serde_json::json!(val),
                                    );
                                }
                                if let Some(ref sof) = node.config.system {
                                    let val = match sof {
                                        scaffold_ir::ir::StringOrFileIR::Literal { value } => {
                                            value.clone()
                                        }
                                        scaffold_ir::ir::StringOrFileIR::File { path } => {
                                            path.clone()
                                        }
                                    };
                                    overrides.insert(
                                        format!("{}.system", node.name),
                                        serde_json::json!(val),
                                    );
                                }
                                if let Some(ref shell) = node.config.shell {
                                    overrides.insert(
                                        format!("{}.shell", node.name),
                                        serde_json::json!(shell),
                                    );
                                }
                                if let Some(ref model) = node.config.model {
                                    overrides.insert(
                                        format!("{}.model", node.name),
                                        serde_json::json!(model),
                                    );
                                }
                                if let Some(temp) = node.config.temperature {
                                    overrides.insert(
                                        format!("{}.temperature", node.name),
                                        serde_json::json!(temp),
                                    );
                                }
                            } else {
                                // New node: synthetic node pattern (same as AddPromptStep)
                                let kind_str = match node.kind {
                                    scaffold_ir::ir::NodeKindIR::Tool => "tool",
                                    scaffold_ir::ir::NodeKindIR::Agent => "agent",
                                    scaffold_ir::ir::NodeKindIR::Verify => "verify",
                                    _ => "prompt",
                                };
                                overrides.insert(
                                    format!("_node.{}", node.name),
                                    serde_json::json!({"kind": kind_str}),
                                );
                                if let Some(ref sof) = node.config.template {
                                    let val = match sof {
                                        scaffold_ir::ir::StringOrFileIR::Literal { value } => {
                                            value.clone()
                                        }
                                        scaffold_ir::ir::StringOrFileIR::File { path } => {
                                            path.clone()
                                        }
                                    };
                                    overrides.insert(
                                        format!("{}.template", node.name),
                                        serde_json::json!(val),
                                    );
                                }
                                if let Some(ref sof) = node.config.system {
                                    let val = match sof {
                                        scaffold_ir::ir::StringOrFileIR::Literal { value } => {
                                            value.clone()
                                        }
                                        scaffold_ir::ir::StringOrFileIR::File { path } => {
                                            path.clone()
                                        }
                                    };
                                    overrides.insert(
                                        format!("{}.system", node.name),
                                        serde_json::json!(val),
                                    );
                                }
                                if let Some(ref shell) = node.config.shell {
                                    overrides.insert(
                                        format!("{}.shell", node.name),
                                        serde_json::json!(shell),
                                    );
                                }
                                if let Some(ref model) = node.config.model {
                                    overrides.insert(
                                        format!("{}.model", node.name),
                                        serde_json::json!(model),
                                    );
                                }
                            }
                        }
                    }
                    _ => {}
                }

                let candidate = Candidate {
                    id: 0,
                    parent_id: Some(parent_id),
                    graph: new_graph.clone(),
                    overrides,
                    mutations: vec![mutation],
                    score: None,
                    metric_scores: HashMap::new(),
                    descriptor: GraphDescriptor::from_graph(&new_graph, ir),
                    children_count: 0,
                    meta_reasoning,
                    case_results: vec![],
                    total_cases: 0,
                    cases_passed: 0,
                    passed_case_ids: vec![],
                };

                let cand_id = archive.candidates.len();
                let (score, metrics, cases, total, passed, passed_ids) =
                    evaluate_candidate(ir, objective, &candidate, dataset, options, cand_id)
                        .await?;

                successful_generations += 1;

                if successful_generations == 1 {
                    emit(
                        options,
                        OptEvent::PhaseChanged {
                            phase: OptPhase::Evolutionary,
                        },
                    );
                }
                let best_before = archive.best().and_then(|c| c.score).unwrap_or(0.0);
                let mutation_labels: Vec<String> = candidate
                    .mutations
                    .iter()
                    .map(|m| m.short_label())
                    .collect();

                emit(
                    options,
                    OptEvent::CandidateEvaluated {
                        candidate_id: cand_id,
                        parent_id: Some(parent_id),
                        score,
                        metric_scores: metrics.clone(),
                        best_so_far: best_before.max(score),
                        mutations: mutation_labels,
                        generation: successful_generations,
                        max_generations,
                    },
                );
                if options.event_tx.is_none() {
                    eprintln!(
                        "[optimizer] gen={}/{} candidate {}: score={:.4}",
                        successful_generations, max_generations, cand_id, score
                    );
                }

                archive.add(Candidate {
                    score: Some(score),
                    metric_scores: metrics,
                    case_results: cases,
                    total_cases: total,
                    cases_passed: passed,
                    passed_case_ids: passed_ids,
                    ..candidate
                });

                // Track parent usage for novelty weighting
                archive.increment_children(parent_id);

                // Check early stop
                if let Some(ts) = target_score {
                    if score >= ts {
                        emit(options, OptEvent::EarlyStopped { score, target: ts });
                        if options.event_tx.is_none() {
                            eprintln!(
                                "[optimizer] target score {:.4} reached at gen {}, stopping early",
                                ts, successful_generations
                            );
                        }
                        return Ok(());
                    }
                }
            }
            MutationResult::Skipped(_) => {
                // Doesn't count against budget, try again
            }
        }
    }

    if total_attempts >= max_attempts {
        emit(
            options,
            OptEvent::Log {
                message: format!(
                    "exhausted {} attempts with only {} successful generations",
                    max_attempts, successful_generations
                ),
            },
        );
        if options.event_tx.is_none() {
            eprintln!(
                "[optimizer] exhausted {} attempts with only {} successful generations",
                max_attempts, successful_generations
            );
        }
    }

    Ok(())
}

/// Select a parent using sigmoid-weighted selection (Algorithm 2, DGM-H).
///
/// High-scoring candidates are preferred via a sigmoid transform around the
/// dynamic midpoint (average of top-3 scores). When `use_novelty` is true,
/// candidates that have already been selected many times are down-weighted
/// by a novelty bonus `1/(1+children)`. When the meta-agent is active,
/// novelty is disabled since the LLM provides its own diversity.
fn select_parent<'a>(
    archive: &'a Archive,
    rng: &mut SimpleRng,
    use_novelty: bool,
) -> Option<&'a Candidate> {
    let evaluated: Vec<&Candidate> = archive
        .candidates
        .iter()
        .filter(|c| c.score.is_some())
        .collect();

    if evaluated.is_empty() {
        return None;
    }

    // Dynamic midpoint: average of top-3 scores
    let mut scores: Vec<f64> = evaluated.iter().map(|c| c.score.unwrap()).collect();
    scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let m = 3.min(scores.len());
    let alpha_mid: f64 = scores[..m].iter().sum::<f64>() / m as f64;

    // Sigmoid transform, with optional novelty bonus
    let lambda = 10.0;
    let weights: Vec<f64> = evaluated
        .iter()
        .map(|c| {
            let si = 1.0 / (1.0 + (-lambda * (c.score.unwrap() - alpha_mid)).exp());
            if use_novelty {
                let hi = 1.0 / (1.0 + c.children_count as f64);
                si * hi
            } else {
                si
            }
        })
        .collect();

    // Categorical sampling
    let total: f64 = weights.iter().sum();
    if total <= 0.0 {
        return Some(evaluated[rng.next_usize() % evaluated.len()]);
    }
    let r = (rng.next_u64() as f64 / u64::MAX as f64) * total;
    let mut cumulative = 0.0;
    for (i, w) in weights.iter().enumerate() {
        cumulative += w;
        if r <= cumulative {
            return Some(evaluated[i]);
        }
    }
    Some(evaluated.last().unwrap())
}

/// Generate a random mutation for the given graph.
fn generate_random_mutation(
    graph: &GraphIR,
    ir: &ScaffoldIR,
    allowed: &[String],
    tunables: &[TunableIR],
    rng: &mut SimpleRng,
) -> Option<Mutation> {
    if allowed.is_empty() {
        return None;
    }

    let step_names = collect_step_names(&graph.body);
    if step_names.is_empty() {
        return None;
    }

    let mutation_type = &allowed[rng.next_usize() % allowed.len()];
    let target_step = &step_names[rng.next_usize() % step_names.len()];

    match mutation_type.as_str() {
        "insert_verify" => {
            let verify_nodes: Vec<&NodeIR> = ir
                .nodes
                .iter()
                .filter(|n| n.kind == NodeKindIR::Verify)
                .collect();
            if verify_nodes.is_empty() {
                return None;
            }
            let verify = &verify_nodes[rng.next_usize() % verify_nodes.len()];
            Some(Mutation::InsertVerify {
                after_step: target_step.clone(),
                verify_node: verify.name.clone(),
                max_retries: (rng.next_usize() % 3 + 1) as u32,
            })
        }
        "wrap_retry" => {
            let verify_nodes: Vec<&NodeIR> = ir
                .nodes
                .iter()
                .filter(|n| n.kind == NodeKindIR::Verify)
                .collect();
            if verify_nodes.is_empty() {
                return None;
            }
            let verify = &verify_nodes[rng.next_usize() % verify_nodes.len()];
            Some(Mutation::WrapRetry {
                step: target_step.clone(),
                verify_node: verify.name.clone(),
                max_retries: (rng.next_usize() % 3 + 1) as u32,
            })
        }
        "insert_step" => {
            let compatible_nodes: Vec<&NodeIR> = ir
                .nodes
                .iter()
                .filter(|n| has_single_input_compatible_type(ir, &n.input))
                .collect();
            if compatible_nodes.is_empty() {
                return None;
            }
            let node = &compatible_nodes[rng.next_usize() % compatible_nodes.len()];
            let new_name = format!("inserted_{}", rng.next_usize() % 1000);
            Some(Mutation::InsertStep {
                after_step: target_step.clone(),
                new_step_name: new_name,
                node: node.name.clone(),
            })
        }
        "add_prompt_step" => {
            let prompt_steps =
                collect_steps_for_node_kinds(graph, ir, &[NodeKindIR::Prompt, NodeKindIR::Agent]);
            if prompt_steps.is_empty() {
                return None;
            }
            let after_step = &prompt_steps[rng.next_usize() % prompt_steps.len()];
            let new_name = format!("review_{}", rng.next_usize() % 1000);
            Some(Mutation::AddPromptStep {
                after_step: after_step.clone(),
                new_step_name: new_name,
                template: "Improve the following output while preserving its intended format exactly. Return only the improved result.\n\n{{ input }}".to_string(),
                system: None,
                model: None,
            })
        }
        "remove_step" => Some(Mutation::RemoveStep {
            step: target_step.clone(),
        }),
        "replace_component" => {
            let nodes: Vec<&NodeIR> = ir
                .nodes
                .iter()
                .filter(|n| n.kind == NodeKindIR::Prompt)
                .collect();
            if nodes.is_empty() {
                return None;
            }
            let node = &nodes[rng.next_usize() % nodes.len()];
            Some(Mutation::ReplaceComponent {
                step: target_step.clone(),
                new_node: node.name.clone(),
            })
        }
        "set_config" => {
            // Pick a random tunable and sample a value from its domain.
            // If no tunables are declared, skip — we never use hardcoded values.
            if tunables.is_empty() {
                return None;
            }
            let tunable = &tunables[rng.next_usize() % tunables.len()];
            if tunable.domain.is_empty() || tunable.path.len() < 2 {
                return None;
            }
            let value =
                expr_to_json_value(&tunable.domain[rng.next_usize() % tunable.domain.len()]);
            let node = tunable.path[0].clone();
            let field = tunable.path[1..].join(".");
            Some(Mutation::SetConfig { node, field, value })
        }
        _ => None,
    }
}

fn has_single_input_compatible_node(ir: &ScaffoldIR) -> bool {
    ir.nodes
        .iter()
        .any(|node| has_single_input_compatible_type(ir, &node.input))
}

fn has_single_input_compatible_type(ir: &ScaffoldIR, ty: &TypeIR) -> bool {
    match ty {
        TypeIR::Struct { .. } => false,
        TypeIR::Named { name } => ir
            .types
            .iter()
            .find(|t| t.name == *name)
            .map(|t| has_single_input_compatible_type(ir, &t.ty))
            .unwrap_or(true),
        _ => true,
    }
}

fn collect_steps_for_node_kinds(
    graph: &GraphIR,
    ir: &ScaffoldIR,
    kinds: &[NodeKindIR],
) -> Vec<String> {
    let mut steps = Vec::new();
    collect_steps_for_node_kinds_in_stmts(&graph.body, ir, kinds, &mut steps);
    steps
}

fn collect_steps_for_node_kinds_in_stmts(
    stmts: &[GraphStmtIR],
    ir: &ScaffoldIR,
    kinds: &[NodeKindIR],
    out: &mut Vec<String>,
) {
    for stmt in stmts {
        match stmt {
            GraphStmtIR::Step(step) => {
                if ir
                    .nodes
                    .iter()
                    .find(|n| n.name == step.node)
                    .map(|n| kinds.contains(&n.kind))
                    .unwrap_or(false)
                {
                    out.push(step.name.clone());
                }
            }
            GraphStmtIR::Loop(loop_stmt) => {
                collect_steps_for_node_kinds_in_stmts(&loop_stmt.body, ir, kinds, out)
            }
            GraphStmtIR::If(if_stmt) => {
                collect_steps_for_node_kinds_in_stmts(&if_stmt.then_body, ir, kinds, out);
                collect_steps_for_node_kinds_in_stmts(&if_stmt.else_body, ir, kinds, out);
            }
            GraphStmtIR::Parallel(par_stmt) => {
                collect_steps_for_node_kinds_in_stmts(&par_stmt.body, ir, kinds, out)
            }
            _ => {}
        }
    }
}

/// Build the optimization report.
fn build_report(
    ir: &ScaffoldIR,
    objective: &ObjectiveIR,
    archive: &Archive,
    options: &OptimizationOptions,
) -> Result<OptimizationReport> {
    let best = archive.best();

    let report = OptimizationReport {
        objective_name: objective.name.clone(),
        total_candidates: archive.candidates.len(),
        best_score: best.map(|c| c.score.unwrap()),
        best_candidate_id: best.map(|c| c.id),
        candidate_scores: archive
            .candidates
            .iter()
            .map(|c| CandidateScore {
                id: c.id,
                parent_id: c.parent_id,
                score: c.score,
                mutations: c.mutations.iter().map(|m| m.short_label()).collect(),
                node_count: c.descriptor.node_count,
                verify_count: c.descriptor.verify_count,
                meta_reasoning: c.meta_reasoning.clone(),
            })
            .collect(),
        best_overrides: best.map(|c| c.overrides.clone()).filter(|o| !o.is_empty()),
        best_metric_scores: best
            .map(|c| c.metric_scores.clone())
            .filter(|m| !m.is_empty()),
    };

    // Write report to directory if configured
    if let Some(ref dir) = options.report_dir {
        std::fs::create_dir_all(dir)
            .map_err(|e| Error::Runtime(format!("failed to create report dir: {}", e)))?;
        let report_json = serde_json::to_string_pretty(&report).unwrap_or_default();
        std::fs::write(dir.join("report.json"), report_json)
            .map_err(|e| Error::Runtime(format!("failed to write report: {}", e)))?;

        // Write best candidate details (full mutations + overrides with rewritten prompts)
        if let Some(best) = best {
            let mutations_json: Vec<serde_json::Value> = best
                .mutations
                .iter()
                .filter_map(|m| serde_json::to_value(m).ok())
                .collect();
            let best_detail = serde_json::json!({
                "id": best.id,
                "parent_id": best.parent_id,
                "score": best.score,
                "metric_scores": best.metric_scores,
                "mutations": mutations_json,
                "overrides": best.overrides,
                "meta_reasoning": best.meta_reasoning,
                "cases_passed": best.cases_passed,
                "total_cases": best.total_cases,
            });
            let best_json = serde_json::to_string_pretty(&best_detail).unwrap_or_default();
            std::fs::write(dir.join("best_candidate.json"), best_json)
                .map_err(|e| Error::Runtime(format!("failed to write best candidate: {}", e)))?;
        }
    }

    // Write best candidate source if configured
    if let Some(ref path) = options.write_best {
        if let Some(best) = best {
            if let Some(parent) = path.parent() {
                if !parent.as_os_str().is_empty() {
                    std::fs::create_dir_all(parent).map_err(|e| {
                        Error::Runtime(format!(
                            "failed to create best candidate directory '{}': {}",
                            parent.display(),
                            e
                        ))
                    })?;
                }
            }

            let best_ir = materialize_best_candidate_ir(ir, objective, best);
            let source = pretty_print(&best_ir);
            std::fs::write(path, source)
                .map_err(|e| Error::Runtime(format!("failed to write best candidate: {}", e)))?;
        }
    }

    emit(
        options,
        OptEvent::Completed {
            objective_name: objective.name.clone(),
            best_score: report.best_score,
            total_candidates: report.total_candidates,
        },
    );
    if options.event_tx.is_none() {
        eprintln!(
            "[optimizer] done. {} candidates evaluated. best score: {:.4}",
            report.total_candidates,
            report.best_score.unwrap_or(0.0)
        );
    }

    Ok(report)
}

/// Simple deterministic RNG (xorshift64).
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_usize(&mut self) -> usize {
        self.next_u64() as usize
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    #[test]
    fn test_generate_tunable_combinations() {
        let tunables = vec![
            TunableIR {
                path: vec!["solver".into(), "model".into()],
                domain: vec![
                    ExprIR::LitString {
                        value: "gpt-4o".into(),
                    },
                    ExprIR::LitString {
                        value: "gpt-4o-mini".into(),
                    },
                ],
            },
            TunableIR {
                path: vec!["solver".into(), "temperature".into()],
                domain: vec![
                    ExprIR::LitFloat { value: 0.0 },
                    ExprIR::LitFloat { value: 0.7 },
                ],
            },
        ];

        let combos = generate_tunable_combinations(&tunables);
        assert_eq!(combos.len(), 4); // 2 models × 2 temperatures
    }

    #[test]
    fn test_empty_tunables() {
        let combos = generate_tunable_combinations(&[]);
        assert_eq!(combos.len(), 1);
        assert!(combos[0].is_empty());
    }

    #[test]
    fn test_expr_to_value() {
        assert_eq!(expr_to_value(&ExprIR::LitInt { value: 42 }), Value::Int(42));
        assert_eq!(
            expr_to_value(&ExprIR::LitString {
                value: "hello".into()
            }),
            Value::String("hello".into())
        );
    }

    #[test]
    fn test_simple_rng() {
        let mut rng = SimpleRng::new(42);
        let v1 = rng.next_u64();
        let v2 = rng.next_u64();
        assert_ne!(v1, v2);
        // Deterministic: same seed produces same sequence
        let mut rng2 = SimpleRng::new(42);
        assert_eq!(v1, rng2.next_u64());
        assert_eq!(v2, rng2.next_u64());
    }

    #[test]
    fn test_archive_best() {
        let mut archive = Archive::new();
        let desc = GraphDescriptor {
            topology_hash: 0,
            node_count: 1,
            verify_count: 0,
            max_depth: 0,
        };

        archive.add(Candidate {
            id: 0,
            parent_id: None,
            graph: GraphIR {
                name: "test".to_string(),
                input: TypeIR::String,
                output: TypeIR::String,
                body: vec![],
            },
            overrides: HashMap::new(),
            mutations: vec![],
            score: Some(0.5),
            metric_scores: HashMap::new(),
            descriptor: desc.clone(),
            children_count: 0,
            meta_reasoning: None,
            case_results: vec![],
            total_cases: 0,
            cases_passed: 0,
            passed_case_ids: vec![],
        });

        archive.add(Candidate {
            id: 0,
            parent_id: None,
            graph: GraphIR {
                name: "test".to_string(),
                input: TypeIR::String,
                output: TypeIR::String,
                body: vec![],
            },
            overrides: HashMap::new(),
            mutations: vec![],
            score: Some(0.8),
            metric_scores: HashMap::new(),
            descriptor: desc,
            children_count: 0,
            meta_reasoning: None,
            case_results: vec![],
            total_cases: 0,
            cases_passed: 0,
            passed_case_ids: vec![],
        });

        let best = archive.best().unwrap();
        assert_eq!(best.score, Some(0.8));
    }

    #[test]
    fn test_reset_meta_log_truncates_existing_file() {
        let unique = format!(
            "scaffold-meta-log-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let dir = std::env::temp_dir().join(unique);
        let path = dir.join("meta_debug.log");

        fs::create_dir_all(&dir).unwrap();
        fs::write(&path, "old log contents").unwrap();

        let options = OptimizationOptions {
            meta_log: Some(path.clone()),
            ..OptimizationOptions::default()
        };

        reset_meta_log(&options).unwrap();

        let contents = fs::read_to_string(&path).unwrap();
        assert!(contents.is_empty());

        let _ = fs::remove_file(&path);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_bake_overrides_materializes_synthetic_prompt_node() {
        let mut ir = ScaffoldIR::default();
        let mut overrides = HashMap::new();
        overrides.insert("_node.repair".into(), serde_json::json!({"kind": "prompt"}));
        overrides.insert(
            "repair.template".into(),
            serde_json::json!("Fix this carefully."),
        );
        overrides.insert("repair.system".into(), serde_json::json!("Be exact."));
        overrides.insert("repair.model".into(), serde_json::json!("gpt-4o"));
        overrides.insert("repair.max_turns".into(), serde_json::json!(2));
        overrides.insert("repair.timeout".into(), serde_json::json!(30));

        bake_overrides_into_ir(&mut ir, &overrides);

        let node = ir
            .nodes
            .iter()
            .find(|node| node.name == "repair")
            .expect("synthetic prompt node should be materialized");

        assert_eq!(node.kind, NodeKindIR::Prompt);
        assert!(matches!(node.input, TypeIR::String));
        assert!(matches!(node.output, TypeIR::String));
        assert!(matches!(
            node.config.template,
            Some(StringOrFileIR::Literal { ref value }) if value == "Fix this carefully."
        ));
        assert!(matches!(
            node.config.system,
            Some(StringOrFileIR::Literal { ref value }) if value == "Be exact."
        ));
        assert_eq!(node.config.model.as_deref(), Some("gpt-4o"));
        assert_eq!(node.config.max_turns, Some(2));
        assert_eq!(node.config.timeout, Some(30));
    }

    #[test]
    fn test_build_report_writes_best_candidate_scaffold_source() {
        let unique = format!(
            "scaffold-best-write-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let dir = std::env::temp_dir().join(unique);
        let path = dir.join("nested").join("best_candidate.scaffold");

        let objective = ObjectiveIR {
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
        };

        let ir = ScaffoldIR {
            version: "2.0.0".into(),
            types: vec![],
            nodes: vec![NodeIR {
                name: "solve_code".into(),
                kind: NodeKindIR::Prompt,
                input: TypeIR::String,
                output: TypeIR::String,
                config: NodeConfigIR {
                    template: Some(StringOrFileIR::Literal {
                        value: "base".into(),
                    }),
                    ..NodeConfigIR::default()
                },
            }],
            graphs: vec![GraphIR {
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
                    GraphStmtIR::Emit(EmitIR::Direct {
                        value: ExprIR::Ident {
                            name: "attempt1".into(),
                        },
                    }),
                ],
            }],
            objectives: vec![objective.clone()],
        };

        let best_graph = GraphIR {
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
                    name: "repair".into(),
                    node: "repair".into(),
                    args: vec![StepArgIR::Positional {
                        value: ExprIR::Ident {
                            name: "attempt1".into(),
                        },
                    }],
                }),
                GraphStmtIR::Emit(EmitIR::Direct {
                    value: ExprIR::Ident {
                        name: "repair".into(),
                    },
                }),
            ],
        };

        let mut overrides = HashMap::new();
        overrides.insert("_node.repair".into(), serde_json::json!({"kind": "prompt"}));
        overrides.insert(
            "repair.template".into(),
            serde_json::json!("Improve {{ input }}"),
        );
        overrides.insert("repair.system".into(), serde_json::json!("Be surgical."));

        let mut archive = Archive::new();
        archive.add(Candidate {
            id: 0,
            parent_id: None,
            graph: best_graph.clone(),
            overrides,
            mutations: vec![],
            score: Some(0.9),
            metric_scores: HashMap::new(),
            descriptor: GraphDescriptor::from_graph(&best_graph, &ir),
            children_count: 0,
            meta_reasoning: None,
            case_results: vec![],
            total_cases: 0,
            cases_passed: 0,
            passed_case_ids: vec![],
        });

        let options = OptimizationOptions {
            write_best: Some(path.clone()),
            ..OptimizationOptions::default()
        };

        build_report(&ir, &objective, &archive, &options).unwrap();

        let written = fs::read_to_string(&path).unwrap();
        assert!(written.contains("node solve_code: prompt"));
        assert!(written.contains("node repair: prompt"));
        assert!(written.contains("graph solve {"));
        assert!(written.contains("step repair = repair(attempt1)"));
        assert!(written.contains("objective aider_polyglot {"));

        let _ = fs::remove_file(&path);
        let _ = fs::remove_dir_all(&dir);
    }

    fn make_graph(name: &str, steps: Vec<(&str, &str)>) -> GraphIR {
        GraphIR {
            name: name.to_string(),
            input: TypeIR::String,
            output: TypeIR::String,
            body: steps
                .into_iter()
                .map(|(sname, node)| {
                    GraphStmtIR::Step(StepIR {
                        name: sname.to_string(),
                        node: node.to_string(),
                        args: vec![],
                    })
                })
                .collect(),
        }
    }

    fn make_sub(name: &str, graph: &str) -> SubObjectiveIR {
        SubObjectiveIR {
            name: name.to_string(),
            graph: graph.to_string(),
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
        }
    }

    #[test]
    fn test_sub_dependency_order_independent() {
        // Two independent subs — both should appear, order doesn't matter
        let ir = ScaffoldIR {
            graphs: vec![
                make_graph("outer", vec![("s1", "graph_a"), ("s2", "graph_b")]),
                make_graph("graph_a", vec![("s", "some_node")]),
                make_graph("graph_b", vec![("s", "some_node")]),
            ],
            ..Default::default()
        };
        let subs = vec![make_sub("sub_a", "graph_a"), make_sub("sub_b", "graph_b")];
        let order = compute_sub_dependency_order(&subs, &ir).unwrap();
        assert_eq!(order.len(), 2);
        // Both should appear
        assert!(order.contains(&0));
        assert!(order.contains(&1));
    }

    #[test]
    fn test_sub_dependency_order_chain() {
        // graph_a calls graph_b, so sub for graph_b should come first
        let ir = ScaffoldIR {
            graphs: vec![
                make_graph("outer", vec![("s1", "graph_a")]),
                make_graph("graph_a", vec![("s", "graph_b")]),
                make_graph("graph_b", vec![("s", "some_node")]),
            ],
            ..Default::default()
        };
        let subs = vec![make_sub("sub_a", "graph_a"), make_sub("sub_b", "graph_b")];
        let order = compute_sub_dependency_order(&subs, &ir).unwrap();
        assert_eq!(order.len(), 2);
        // sub_b (index 1) must come before sub_a (index 0) since graph_a depends on graph_b
        let pos_a = order.iter().position(|&x| x == 0).unwrap();
        let pos_b = order.iter().position(|&x| x == 1).unwrap();
        assert!(pos_b < pos_a, "sub_b should be optimized before sub_a");
    }

    #[test]
    fn test_sub_dependency_order_cycle() {
        // graph_a calls graph_b, graph_b calls graph_a — cycle!
        let ir = ScaffoldIR {
            graphs: vec![
                make_graph("graph_a", vec![("s", "graph_b")]),
                make_graph("graph_b", vec![("s", "graph_a")]),
            ],
            ..Default::default()
        };
        let subs = vec![make_sub("sub_a", "graph_a"), make_sub("sub_b", "graph_b")];
        let result = compute_sub_dependency_order(&subs, &ir);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("cycle"));
    }
}
