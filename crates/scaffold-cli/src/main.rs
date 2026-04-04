//! Scaffold v2 CLI
//!
//! Commands:
//! - check: Parse, type check, and verify a scaffold file
//! - compile: Lower to IR and output JSON
//! - run: Execute a named graph with JSON input
//! - evaluate: Evaluate one candidate for an objective
//! - optimize: Full optimization with topology mutations

use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use ariadne::{Color, Label, Report, ReportKind, Source};
use clap::{Parser, Subcommand, ValueEnum};

use scaffold_ir::{lower, pretty_print, to_json, to_json_compact, ScaffoldIR};
use scaffold_runtime::trace::{init_tracer, TraceFormat, TraceLevel, TraceOutput, TracerConfig};
use scaffold_runtime::{GraphExecutor, OptimizationBackend, OptimizationOptions, Value};
use scaffold_syntax::parse;
use scaffold_types::check;
use scaffold_verify::{verify, Severity};

mod tui;

#[derive(Parser)]
#[command(name = "scaffold")]
#[command(author, version, about = "Scaffold v2 DSL compiler and runtime", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Clone, Debug, ValueEnum)]
enum OptimizeBackendArg {
    Grid,
    Evolutionary,
}

#[derive(Subcommand)]
enum Commands {
    /// Parse, type check, and verify a scaffold file
    Check {
        /// Input file
        #[arg(value_name = "FILE")]
        file: PathBuf,

        /// Show verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Compile to IR and output JSON
    Compile {
        /// Input file
        #[arg(value_name = "FILE")]
        file: PathBuf,

        /// Output file (default: stdout)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Compact JSON output
        #[arg(long)]
        compact: bool,
    },

    /// Execute a named graph with JSON input
    Run {
        /// Input file
        #[arg(value_name = "FILE")]
        file: PathBuf,

        /// Graph name to execute
        #[arg(long)]
        graph: String,

        /// Input JSON (or @file for file input)
        #[arg(long)]
        input: String,

        /// Enable live tracing
        #[arg(long)]
        live: bool,
    },

    /// Evaluate one candidate for an objective
    Evaluate {
        /// Input file
        #[arg(value_name = "FILE")]
        file: PathBuf,

        /// Objective name
        #[arg(long)]
        objective: String,

        /// Tunable overrides as JSON (node.field=value pairs)
        #[arg(long)]
        assignments: Option<String>,

        /// Override the objective's dataset with a different file
        #[arg(long)]
        dataset: Option<PathBuf>,

        /// Enable live tracing
        #[arg(long)]
        live: bool,
    },

    /// Run optimization for an objective
    Optimize {
        /// Input file
        #[arg(value_name = "FILE")]
        file: PathBuf,

        /// Objective name
        #[arg(long)]
        objective: String,

        /// Maximum candidates to evaluate
        #[arg(long, default_value = "20")]
        max_candidates: usize,

        /// Optimization backend
        #[arg(long, default_value = "evolutionary")]
        backend: OptimizeBackendArg,

        /// Directory to write optimization reports
        #[arg(long)]
        report_dir: Option<PathBuf>,

        /// Write the best candidate `.scaffold` to this file
        #[arg(long)]
        write_best: Option<PathBuf>,

        /// Enable live tracing
        #[arg(long)]
        live: bool,

        /// Number of dataset cases to evaluate concurrently per candidate
        #[arg(long, default_value = "1")]
        concurrency: usize,

        /// LLM model for meta-agent guided mutations (e.g. gpt-4o). Omit for random mutations.
        #[arg(long)]
        meta_model: Option<String>,

        /// Path to a debug log file for meta-agent context/responses.
        #[arg(long)]
        meta_log: Option<PathBuf>,

        /// Restart meta-agent context every N generations to prevent context bloat.
        #[arg(long)]
        meta_restart: Option<usize>,

        /// Show full execution traces for all failed cases in meta-agent context.
        /// Also enables changed-case trace analysis between generations.
        #[arg(long)]
        meta_full_traces: bool,

        /// Number of training cases to sample per generation for meta-agent context.
        /// Only effective when the objective declares a split.
        #[arg(long)]
        batch_size: Option<usize>,

        /// Pool train+val cases and sample a fresh random subset for each candidate
        /// evaluation. Prevents overfitting to a small fixed val set.
        #[arg(long)]
        rotating_val: bool,

        /// Allow meta-agent proposed tool nodes to access the network.
        /// Filesystem sandbox remains active (CWD only).
        #[arg(long)]
        online: bool,
    },

    /// Pretty-print IR back to scaffold source
    Print {
        /// Input file (scaffold source or IR JSON)
        #[arg(value_name = "FILE")]
        file: PathBuf,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    match cli.command {
        Commands::Check { file, verbose } => cmd_check(&file, verbose),
        Commands::Compile {
            file,
            output,
            compact,
        } => cmd_compile(&file, output.as_deref(), compact),
        Commands::Run {
            file,
            graph,
            input,
            live,
        } => cmd_run(&file, &graph, &input, live),
        Commands::Evaluate {
            file,
            objective,
            assignments,
            dataset,
            live,
        } => cmd_evaluate(&file, &objective, assignments.as_deref(), dataset, live),
        Commands::Optimize {
            file,
            objective,
            max_candidates,
            backend,
            report_dir,
            write_best,
            live,
            concurrency,
            meta_model,
            meta_log,
            meta_restart,
            meta_full_traces,
            batch_size,
            rotating_val,
            online,
        } => cmd_optimize(
            &file,
            &objective,
            max_candidates,
            backend,
            report_dir,
            write_best,
            live,
            concurrency,
            meta_model,
            meta_log,
            meta_restart,
            meta_full_traces,
            batch_size,
            rotating_val,
            online,
        ),
        Commands::Print { file } => cmd_print(&file),
    }
}

// ── Check ──

fn cmd_check(file: &PathBuf, verbose: bool) -> ExitCode {
    let source = match fs::read_to_string(file) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot read {}: {}", file.display(), e);
            return ExitCode::FAILURE;
        }
    };

    let filename = file.to_string_lossy().to_string();

    // Parse
    let program = match parse(&source) {
        Ok(p) => p,
        Err(e) => {
            print_parse_error(&filename, &source, &e);
            return ExitCode::FAILURE;
        }
    };

    if verbose {
        eprintln!("parsed {} declarations", program.declarations.len());
    }

    // Type check
    let (_type_env, type_errors) = check(&program);
    let mut has_errors = false;
    for err in &type_errors {
        Report::build(ReportKind::Error, &filename, err.span.start)
            .with_label(
                Label::new((&filename, err.span.start..err.span.end))
                    .with_message(&err.message)
                    .with_color(Color::Red),
            )
            .finish()
            .eprint((&filename, Source::from(&source)))
            .ok();
        has_errors = true;
    }

    // Lower to IR for verification
    let ir = match lower(&program) {
        Ok(ir) => ir,
        Err(e) => {
            for le in &e {
                eprintln!("error: lowering: {}", le);
            }
            return ExitCode::FAILURE;
        }
    };

    // Verify
    let findings = verify(&ir);
    for finding in &findings {
        match finding.severity {
            Severity::Error => {
                eprintln!("[error] {}", finding);
                has_errors = true;
            }
            Severity::Warning => {
                if verbose {
                    eprintln!("[warning] {}", finding);
                }
            }
        }
    }

    if has_errors {
        eprintln!("check failed");
        ExitCode::FAILURE
    } else {
        eprintln!("OK");
        ExitCode::SUCCESS
    }
}

// ── Compile ──

fn cmd_compile(file: &PathBuf, output: Option<&std::path::Path>, compact: bool) -> ExitCode {
    let source = match fs::read_to_string(file) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot read {}: {}", file.display(), e);
            return ExitCode::FAILURE;
        }
    };

    let program = match parse(&source) {
        Ok(p) => p,
        Err(e) => {
            let filename = file.to_string_lossy().to_string();
            print_parse_error(&filename, &source, &e);
            return ExitCode::FAILURE;
        }
    };

    let ir = match lower(&program) {
        Ok(ir) => ir,
        Err(e) => {
            for le in &e {
                eprintln!("error: lowering: {}", le);
            }
            return ExitCode::FAILURE;
        }
    };

    let json = if compact {
        to_json_compact(&ir).unwrap()
    } else {
        to_json(&ir).unwrap()
    };

    match output {
        Some(path) => {
            if let Err(e) = fs::write(path, &json) {
                eprintln!("error: cannot write {}: {}", path.display(), e);
                return ExitCode::FAILURE;
            }
            eprintln!("wrote {}", path.display());
        }
        None => println!("{}", json),
    }

    ExitCode::SUCCESS
}

// ── Run ──

fn cmd_run(file: &PathBuf, graph_name: &str, input_arg: &str, live: bool) -> ExitCode {
    if live {
        setup_tracer();
    }

    let ir = match load_ir(file) {
        Ok(ir) => ir,
        Err(msg) => {
            eprintln!("{}", msg);
            return ExitCode::FAILURE;
        }
    };

    // Parse input
    let input = match parse_input(input_arg) {
        Ok(v) => v,
        Err(msg) => {
            eprintln!("error: {}", msg);
            return ExitCode::FAILURE;
        }
    };

    // Set up prompt manager from the file's directory
    let prompt_mgr = setup_prompt_manager(file);

    let executor = GraphExecutor::new(ir).with_prompt_manager(prompt_mgr);

    let rt = tokio::runtime::Runtime::new().unwrap();
    match rt.block_on(executor.execute_graph(graph_name, input)) {
        Ok(result) => {
            let json: serde_json::Value = result.into();
            println!("{}", serde_json::to_string_pretty(&json).unwrap());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {}", e);
            ExitCode::FAILURE
        }
    }
}

// ── Evaluate ──

fn cmd_evaluate(
    file: &PathBuf,
    objective_name: &str,
    assignments: Option<&str>,
    dataset_override: Option<PathBuf>,
    live: bool,
) -> ExitCode {
    if live {
        setup_tracer();
    }

    let ir = match load_ir(file) {
        Ok(ir) => ir,
        Err(msg) => {
            eprintln!("{}", msg);
            return ExitCode::FAILURE;
        }
    };

    // Parse assignments
    let overrides = match assignments {
        Some(json_str) => {
            let parsed: serde_json::Value = match serde_json::from_str(json_str) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("error: invalid assignments JSON: {}", e);
                    return ExitCode::FAILURE;
                }
            };
            match parsed.as_object() {
                Some(obj) => obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
                None => {
                    eprintln!("error: assignments must be a JSON object");
                    return ExitCode::FAILURE;
                }
            }
        }
        None => std::collections::HashMap::new(),
    };

    let objective = match ir.objectives.iter().find(|o| o.name == objective_name) {
        Some(o) => o,
        None => {
            eprintln!("error: objective '{}' not found", objective_name);
            return ExitCode::FAILURE;
        }
    };

    let _graph = match ir.graphs.iter().find(|g| g.name == objective.graph) {
        Some(g) => g,
        None => {
            eprintln!(
                "error: graph '{}' referenced by objective not found",
                objective.graph
            );
            return ExitCode::FAILURE;
        }
    };

    let prompt_mgr = setup_prompt_manager(file);
    let executor = GraphExecutor::new(ir.clone())
        .with_prompt_manager(prompt_mgr)
        .with_overrides(overrides);

    // Load dataset (with optional override)
    let dataset_spec = match dataset_override {
        Some(ref path) => scaffold_ir::DatasetSpecIR::File {
            path: path.display().to_string(),
        },
        None => objective.dataset.clone(),
    };
    let dataset = match scaffold_runtime::optimizer::load_dataset_from_spec(&dataset_spec) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("error: failed to load dataset: {}", e);
            return ExitCode::FAILURE;
        }
    };

    eprintln!(
        "evaluating objective '{}' on graph '{}' ({} cases)",
        objective_name,
        objective.graph,
        dataset.len()
    );

    let rt = tokio::runtime::Runtime::new().unwrap();
    let case_count = dataset.len();
    let mut checker_totals: std::collections::HashMap<String, f64> =
        std::collections::HashMap::new();
    let mut errors = 0;

    for case in &dataset {
        match rt.block_on(executor.execute_graph(&objective.graph, case.input.clone())) {
            Ok(output) => {
                // Evaluate each checker expression
                let mut case_checks = Vec::new();
                for checker in &objective.checkers {
                    let val = scaffold_runtime::eval_checker_expr(
                        &executor,
                        &checker.expr,
                        &output,
                        &case.expected,
                    );
                    *checker_totals.entry(checker.name.clone()).or_default() += val;
                    case_checks.push(format!(
                        "{}={}",
                        checker.name,
                        if val >= 1.0 { "PASS" } else { "FAIL" }
                    ));
                }
                let json: serde_json::Value = output.into();
                eprintln!(
                    "  case {:?}: {} [{}]",
                    case.id,
                    serde_json::to_string(&json).unwrap_or_default(),
                    case_checks.join(", ")
                );
            }
            Err(e) => {
                eprintln!("  case {:?}: error: {}", case.id, e);
                errors += 1;
            }
        }
    }

    if case_count > 0 {
        eprintln!("---");
        // Print per-checker averages
        for checker in &objective.checkers {
            let total = checker_totals.get(&checker.name).copied().unwrap_or(0.0);
            eprintln!("  {}: {:.4}", checker.name, total / case_count as f64);
        }
        // Print per-metric averages
        let mut metric_avgs: std::collections::HashMap<String, f64> =
            std::collections::HashMap::new();
        for metric in &objective.metrics {
            let checker_total = checker_totals.get(&metric.checker).copied().unwrap_or(0.0);
            let avg = checker_total / case_count as f64;
            metric_avgs.insert(metric.name.clone(), avg);
        }
        // Evaluate score expression
        let score_scope = scaffold_runtime::Scope::with_bindings(
            metric_avgs
                .iter()
                .map(|(k, v)| (k.as_str(), scaffold_runtime::Value::Float(*v)))
                .collect(),
        );
        let score = match executor.eval_expr(&objective.score, &score_scope) {
            Ok(scaffold_runtime::Value::Float(f)) => f,
            Ok(scaffold_runtime::Value::Int(i)) => i as f64,
            _ => 0.0,
        };
        eprintln!(
            "score: {:.4} ({} errors / {} cases)",
            score, errors, case_count
        );
    }

    ExitCode::SUCCESS
}

// ── Optimize ──

fn default_best_candidate_path(
    file: &PathBuf,
    objective_name: &str,
    report_dir: Option<&PathBuf>,
    write_best: Option<PathBuf>,
) -> PathBuf {
    if let Some(path) = write_best {
        return path;
    }

    if let Some(dir) = report_dir {
        return dir.join("best_candidate.scaffold");
    }

    let stem = file
        .file_stem()
        .and_then(|stem| stem.to_str())
        .filter(|stem| !stem.is_empty())
        .unwrap_or("best_candidate");
    let safe_objective: String = objective_name
        .chars()
        .map(|ch| match ch {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' => ch,
            _ => '_',
        })
        .collect();
    file.with_file_name(format!("{stem}.{safe_objective}.best.scaffold"))
}

fn cmd_optimize(
    file: &PathBuf,
    objective_name: &str,
    max_candidates: usize,
    backend: OptimizeBackendArg,
    report_dir: Option<PathBuf>,
    write_best: Option<PathBuf>,
    live: bool,
    concurrency: usize,
    meta_model: Option<String>,
    meta_log: Option<PathBuf>,
    meta_restart: Option<usize>,
    meta_full_traces: bool,
    batch_size: Option<usize>,
    rotating_val: bool,
    online: bool,
) -> ExitCode {
    let ir = match load_ir(file) {
        Ok(ir) => ir,
        Err(msg) => {
            eprintln!("{}", msg);
            return ExitCode::FAILURE;
        }
    };

    let backend = match backend {
        OptimizeBackendArg::Grid => OptimizationBackend::Grid,
        OptimizeBackendArg::Evolutionary => OptimizationBackend::Evolutionary,
    };
    let write_best = Some(default_best_candidate_path(
        file,
        objective_name,
        report_dir.as_ref(),
        write_best,
    ));

    if live {
        // TUI mode: spawn optimizer in a thread, run TUI on main thread
        let (event_tx, event_rx) = tokio::sync::mpsc::unbounded_channel();
        let (result_tx, result_rx) = std::sync::mpsc::channel();

        let options = OptimizationOptions {
            max_candidates,
            backend,
            report_dir,
            write_best,
            event_tx: Some(event_tx),
            concurrency,
            meta_model: meta_model.clone(),
            meta_log: meta_log.clone(),
            meta_context_restart: meta_restart,
            meta_full_traces,
            batch_size,
            rotating_val,
            online,
        };

        let obj_name = objective_name.to_string();
        std::thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().unwrap();
            let result = rt.block_on(scaffold_runtime::optimize_hierarchical(
                &ir, &obj_name, &options,
            ));
            let _ = result_tx.send(result.map_err(|e| e.to_string()));
        });

        match tui::run_tui(event_rx, result_rx) {
            Ok(report) => {
                let json = serde_json::to_string_pretty(&report).unwrap_or_default();
                println!("{}", json);
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("error: {}", e);
                ExitCode::FAILURE
            }
        }
    } else {
        // Non-live mode: run optimizer directly
        let options = OptimizationOptions {
            max_candidates,
            backend,
            report_dir,
            write_best,
            event_tx: None,
            concurrency,
            meta_model,
            meta_log,
            meta_context_restart: meta_restart,
            meta_full_traces,
            batch_size,
            rotating_val,
            online,
        };

        let rt = tokio::runtime::Runtime::new().unwrap();
        match rt.block_on(scaffold_runtime::optimize_hierarchical(
            &ir,
            objective_name,
            &options,
        )) {
            Ok(report) => {
                let json = serde_json::to_string_pretty(&report).unwrap_or_default();
                println!("{}", json);
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("error: {}", e);
                ExitCode::FAILURE
            }
        }
    }
}

// ── Print ──

fn cmd_print(file: &PathBuf) -> ExitCode {
    let source = match fs::read_to_string(file) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot read {}: {}", file.display(), e);
            return ExitCode::FAILURE;
        }
    };

    // Try parsing as IR JSON first
    if let Ok(ir) = serde_json::from_str::<ScaffoldIR>(&source) {
        println!("{}", pretty_print(&ir));
        return ExitCode::SUCCESS;
    }

    // Parse as scaffold source
    let program = match parse(&source) {
        Ok(p) => p,
        Err(e) => {
            let filename = file.to_string_lossy().to_string();
            print_parse_error(&filename, &source, &e);
            return ExitCode::FAILURE;
        }
    };

    let ir = match lower(&program) {
        Ok(ir) => ir,
        Err(e) => {
            for le in &e {
                eprintln!("error: lowering: {}", le);
            }
            return ExitCode::FAILURE;
        }
    };

    println!("{}", pretty_print(&ir));
    ExitCode::SUCCESS
}

// ── Helpers ──

fn load_ir(file: &PathBuf) -> Result<ScaffoldIR, String> {
    let source = fs::read_to_string(file)
        .map_err(|e| format!("error: cannot read {}: {}", file.display(), e))?;

    // Try as IR JSON first
    if let Ok(ir) = serde_json::from_str::<ScaffoldIR>(&source) {
        return Ok(ir);
    }

    // Parse as scaffold source
    let program = parse(&source).map_err(|e| format!("parse error: {}", e))?;
    lower(&program).map_err(|errs| {
        errs.iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join("; ")
    })
}

fn parse_input(input_arg: &str) -> Result<Value, String> {
    if let Some(file_path) = input_arg.strip_prefix('@') {
        let content = fs::read_to_string(file_path)
            .map_err(|e| format!("cannot read {}: {}", file_path, e))?;
        let json: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| format!("invalid JSON in {}: {}", file_path, e))?;
        Ok(Value::from(json))
    } else {
        let json: serde_json::Value =
            serde_json::from_str(input_arg).map_err(|e| format!("invalid JSON input: {}", e))?;
        Ok(Value::from(json))
    }
}

fn setup_prompt_manager(file: &PathBuf) -> scaffold_runtime::PromptManager {
    // Look for prompts in the file's parent directory
    if let Some(parent) = file.parent() {
        let prompts_dir = parent.join("prompts");
        if prompts_dir.exists() {
            if let Ok(pm) = scaffold_runtime::PromptManager::with_template_dir(&prompts_dir) {
                return pm;
            }
        }
        // Also try examples/prompts
        let examples_prompts = parent.join("examples").join("prompts");
        if examples_prompts.exists() {
            if let Ok(pm) = scaffold_runtime::PromptManager::with_template_dir(&examples_prompts) {
                return pm;
            }
        }
    }
    scaffold_runtime::PromptManager::new()
}

fn setup_tracer() {
    init_tracer(TracerConfig {
        enabled: true,
        min_level: TraceLevel::Info,
        format: TraceFormat::Pretty,
        output: TraceOutput::Stderr,
        include_bodies: false,
    });
}

fn print_parse_error(filename: &str, source: &str, error: &scaffold_syntax::ParseError) {
    Report::build(ReportKind::Error, filename, error.span.start)
        .with_label(
            Label::new((filename, error.span.start..error.span.end))
                .with_message(&error.message)
                .with_color(Color::Red),
        )
        .finish()
        .eprint((filename, Source::from(source)))
        .ok();
}

#[cfg(test)]
mod tests {
    use super::default_best_candidate_path;
    use std::path::PathBuf;

    #[test]
    fn test_default_best_candidate_path_prefers_report_dir() {
        let file = PathBuf::from("/tmp/solve.scaffold");
        let report_dir = PathBuf::from("/tmp/reports");

        let path = default_best_candidate_path(&file, "aider_polyglot", Some(&report_dir), None);

        assert_eq!(path, report_dir.join("best_candidate.scaffold"));
    }

    #[test]
    fn test_default_best_candidate_path_derives_from_input_file() {
        let file = PathBuf::from("/tmp/solve.scaffold");

        let path = default_best_candidate_path(&file, "aider/polyglot", None, None);

        assert_eq!(
            path,
            PathBuf::from("/tmp/solve.aider_polyglot.best.scaffold")
        );
    }
}
