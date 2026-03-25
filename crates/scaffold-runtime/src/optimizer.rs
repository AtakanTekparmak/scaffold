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

use crate::error::{Error, Result};
use crate::executor::{GraphExecutor, TunableOverrides};
use crate::mutations::{
    apply_mutation, check_constraints, collect_step_names, GraphDescriptor, Mutation,
    MutationResult,
};
use crate::value::Value;

/// Optimization options.
#[derive(Debug, Clone)]
pub struct OptimizationOptions {
    /// Maximum number of candidates to evaluate.
    pub max_candidates: usize,
    /// Optimization backend to use.
    pub backend: OptimizationBackend,
    /// Directory to write reports.
    pub report_dir: Option<PathBuf>,
    /// Write the best candidate IR to this file.
    pub write_best: Option<PathBuf>,
}

impl Default for OptimizationOptions {
    fn default() -> Self {
        Self {
            max_candidates: 20,
            backend: OptimizationBackend::Evolutionary,
            report_dir: None,
            write_best: None,
        }
    }
}

/// Backend selection.
#[derive(Debug, Clone, Copy)]
pub enum OptimizationBackend {
    /// Enumerate all tunable combinations.
    Grid,
    /// Evolutionary search with topology mutations.
    Evolutionary,
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

    /// Get evaluated candidates sorted by score descending.
    pub fn ranked(&self) -> Vec<&Candidate> {
        let mut evaluated: Vec<&Candidate> =
            self.candidates.iter().filter(|c| c.score.is_some()).collect();
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
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct CandidateScore {
    pub id: usize,
    pub parent_id: Option<usize>,
    pub score: Option<f64>,
    pub mutations: Vec<String>,
    pub node_count: u64,
    pub verify_count: u64,
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
        .ok_or_else(|| {
            Error::Runtime(format!("objective '{}' not found", objective_name))
        })?;

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

    match options.backend {
        OptimizationBackend::Grid => {
            run_grid_search(ir, objective, graph, &dataset, options).await
        }
        OptimizationBackend::Evolutionary => {
            run_evolutionary(ir, objective, graph, &dataset, options).await
        }
    }
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
        ExprIR::List { elements } => {
            Value::List(elements.iter().map(expr_to_value).collect())
        }
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
async fn evaluate_candidate(
    ir: &ScaffoldIR,
    objective: &ObjectiveIR,
    candidate: &Candidate,
    dataset: &[DatasetCase],
) -> Result<(f64, HashMap<String, f64>)> {
    // Build a modified IR with the candidate's graph
    let mut modified_ir = ir.clone();
    for g in &mut modified_ir.graphs {
        if g.name == objective.graph {
            *g = candidate.graph.clone();
        }
    }

    let executor = GraphExecutor::new(modified_ir)
        .with_overrides(candidate.overrides.clone());

    let mut total_score = 0.0;
    let mut case_count = 0;
    let mut metric_totals: HashMap<String, f64> = HashMap::new();

    for case in dataset {
        let result = executor
            .execute_graph(&objective.graph, case.input.clone())
            .await;

        match result {
            Ok(output) => {
                // Evaluate checkers against the output
                for metric in &objective.metrics {
                    let checker = objective
                        .checkers
                        .iter()
                        .find(|c| c.name == metric.checker);
                    let metric_score = if let Some(_checker) = checker {
                        // Evaluate checker expression with output and expected
                        evaluate_checker(&output, &case.expected)
                    } else {
                        0.0
                    };
                    *metric_totals.entry(metric.name.clone()).or_default() += metric_score;
                }
                total_score += 1.0; // Base score for successful execution
            }
            Err(_) => {
                // Failed execution = 0 score
            }
        }
        case_count += 1;
    }

    if case_count == 0 {
        return Ok((0.0, HashMap::new()));
    }

    let avg_score = total_score / case_count as f64;
    let avg_metrics: HashMap<String, f64> = metric_totals
        .into_iter()
        .map(|(k, v)| (k, v / case_count as f64))
        .collect();

    Ok((avg_score, avg_metrics))
}

/// Simple checker: compares output to expected.
fn evaluate_checker(output: &Value, expected: &Value) -> f64 {
    // Exact match
    if output == expected {
        return 1.0;
    }
    // String containment
    if let (Value::String(out), Value::String(exp)) = (output, expected) {
        if out.contains(exp.as_str()) || exp.contains(out.as_str()) {
            return 0.5;
        }
    }
    0.0
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
        };

        let (score, metrics) = evaluate_candidate(ir, objective, &candidate, dataset).await?;

        archive.add(Candidate {
            score: Some(score),
            metric_scores: metrics,
            ..candidate
        });

        eprintln!(
            "[optimizer] candidate {}/{}: score={:.4}",
            i + 1,
            combinations.len().min(options.max_candidates),
            score
        );
    }

    build_report(objective, &archive, options)
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
    let mut rng = SimpleRng::new(42);

    let topology = objective.topology.as_ref();

    // Seed candidate (the original graph)
    let seed = Candidate {
        id: 0,
        parent_id: None,
        graph: graph.clone(),
        overrides: HashMap::new(),
        mutations: vec![],
        score: None,
        metric_scores: HashMap::new(),
        descriptor: GraphDescriptor::from_graph(graph, ir),
    };

    let (seed_score, seed_metrics) =
        evaluate_candidate(ir, objective, &seed, dataset).await?;
    let seed_id = archive.add(Candidate {
        score: Some(seed_score),
        metric_scores: seed_metrics,
        ..seed
    });

    eprintln!(
        "[optimizer] seed candidate: score={:.4}",
        seed_score
    );

    // Also seed with tunable combinations if any
    let tunable_combos = generate_tunable_combinations(&objective.tunables);
    for overrides in tunable_combos.iter().take(3) {
        if archive.candidates.len() >= options.max_candidates {
            break;
        }
        let candidate = Candidate {
            id: 0,
            parent_id: Some(seed_id),
            graph: graph.clone(),
            overrides: overrides.clone(),
            mutations: vec![],
            score: None,
            metric_scores: HashMap::new(),
            descriptor: GraphDescriptor::from_graph(graph, ir),
        };
        let (score, metrics) = evaluate_candidate(ir, objective, &candidate, dataset).await?;
        archive.add(Candidate {
            score: Some(score),
            metric_scores: metrics,
            ..candidate
        });
    }

    // Main evolutionary loop
    let allowed_mutations = topology
        .map(|t| t.mutations.clone())
        .unwrap_or_default();

    for gen in 0..options.max_candidates {
        if archive.candidates.len() >= options.max_candidates {
            break;
        }

        // Select parent (tournament selection)
        let parent = select_parent(&archive, &mut rng);
        if parent.is_none() {
            break;
        }
        let parent = parent.unwrap();

        // Generate mutations
        let mutation = generate_random_mutation(
            &parent.graph,
            ir,
            &allowed_mutations,
            &mut rng,
        );

        if let Some(mutation) = mutation {
            match apply_mutation(&parent.graph, &mutation, ir) {
                MutationResult::Ok(new_graph) => {
                    // Check constraints
                    if let Some(topo) = topology {
                        let violations = check_constraints(&new_graph, topo);
                        if !violations.is_empty() {
                            continue;
                        }
                    }

                    let candidate = Candidate {
                        id: 0,
                        parent_id: Some(parent.id),
                        graph: new_graph.clone(),
                        overrides: parent.overrides.clone(),
                        mutations: vec![mutation],
                        score: None,
                        metric_scores: HashMap::new(),
                        descriptor: GraphDescriptor::from_graph(&new_graph, ir),
                    };

                    let (score, metrics) =
                        evaluate_candidate(ir, objective, &candidate, dataset).await?;

                    eprintln!(
                        "[optimizer] gen={} candidate {}: score={:.4}",
                        gen,
                        archive.candidates.len(),
                        score
                    );

                    archive.add(Candidate {
                        score: Some(score),
                        metric_scores: metrics,
                        ..candidate
                    });
                }
                MutationResult::Skipped(_) => {
                    // Try again next generation
                }
            }
        }
    }

    build_report(objective, &archive, options)
}

/// Select a parent using tournament selection.
fn select_parent<'a>(archive: &'a Archive, rng: &mut SimpleRng) -> Option<&'a Candidate> {
    let evaluated: Vec<&Candidate> = archive
        .candidates
        .iter()
        .filter(|c| c.score.is_some())
        .collect();

    if evaluated.is_empty() {
        return None;
    }

    // Tournament of 3
    let mut best: Option<&Candidate> = None;
    for _ in 0..3.min(evaluated.len()) {
        let idx = rng.next_usize() % evaluated.len();
        let candidate = evaluated[idx];
        if best.is_none() || candidate.score > best.unwrap().score {
            best = Some(candidate);
        }
    }

    best
}

/// Generate a random mutation for the given graph.
fn generate_random_mutation(
    graph: &GraphIR,
    ir: &ScaffoldIR,
    allowed: &[String],
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
            let prompt_nodes: Vec<&NodeIR> = ir
                .nodes
                .iter()
                .filter(|n| n.kind == NodeKindIR::Prompt)
                .collect();
            if prompt_nodes.is_empty() {
                return None;
            }
            let node = &prompt_nodes[rng.next_usize() % prompt_nodes.len()];
            let new_name = format!("inserted_{}", rng.next_usize() % 1000);
            Some(Mutation::InsertStep {
                after_step: target_step.clone(),
                new_step_name: new_name,
                node: node.name.clone(),
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
            let models = ["gpt-4o", "gpt-4o-mini", "claude-3-sonnet", "claude-3-haiku"];
            let model = models[rng.next_usize() % models.len()];
            // Find the node for this step
            let step_node = graph.body.iter().find_map(|stmt| {
                if let GraphStmtIR::Step(s) = stmt {
                    if s.name == *target_step {
                        return Some(s.node.clone());
                    }
                }
                None
            });
            if let Some(node_name) = step_node {
                Some(Mutation::SetConfig {
                    node: node_name,
                    field: "model".to_string(),
                    value: serde_json::json!(model),
                })
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Build the optimization report.
fn build_report(
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
                mutations: c
                    .mutations
                    .iter()
                    .map(|m| format!("{:?}", m))
                    .collect(),
                node_count: c.descriptor.node_count,
                verify_count: c.descriptor.verify_count,
            })
            .collect(),
    };

    // Write report to directory if configured
    if let Some(ref dir) = options.report_dir {
        std::fs::create_dir_all(dir).map_err(|e| {
            Error::Runtime(format!("failed to create report dir: {}", e))
        })?;
        let report_json =
            serde_json::to_string_pretty(&report).unwrap_or_default();
        std::fs::write(dir.join("report.json"), report_json).map_err(|e| {
            Error::Runtime(format!("failed to write report: {}", e))
        })?;
    }

    // Write best candidate IR if configured
    if let Some(ref path) = options.write_best {
        if let Some(best) = best {
            let mut best_ir = ScaffoldIR::default();
            // Include the best graph
            best_ir.graphs.push(best.graph.clone());
            let json = serde_json::to_string_pretty(&best_ir).unwrap_or_default();
            std::fs::write(path, json).map_err(|e| {
                Error::Runtime(format!("failed to write best candidate: {}", e))
            })?;
        }
    }

    eprintln!(
        "[optimizer] done. {} candidates evaluated. best score: {:.4}",
        report.total_candidates,
        report.best_score.unwrap_or(0.0)
    );

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
        });

        let best = archive.best().unwrap();
        assert_eq!(best.score, Some(0.8));
    }
}
