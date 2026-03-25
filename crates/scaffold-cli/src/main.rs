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
use scaffold_runtime::trace::{init_tracer, TracerConfig, TraceFormat, TraceLevel, TraceOutput};
use scaffold_runtime::{
    GraphExecutor, OptimizationBackend, OptimizationOptions, Value,
};
use scaffold_syntax::parse;
use scaffold_types::check;
use scaffold_verify::{verify, Severity};

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

        /// Write best candidate IR to this file
        #[arg(long)]
        write_best: Option<PathBuf>,

        /// Enable live tracing
        #[arg(long)]
        live: bool,
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
            live,
        } => cmd_evaluate(&file, &objective, assignments.as_deref(), live),
        Commands::Optimize {
            file,
            objective,
            max_candidates,
            backend,
            report_dir,
            write_best,
            live,
        } => cmd_optimize(
            &file,
            &objective,
            max_candidates,
            backend,
            report_dir,
            write_best,
            live,
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
                Some(obj) => obj
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect(),
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

    // Load dataset
    let dataset = match scaffold_runtime::optimizer::load_dataset_from_spec(&objective.dataset) {
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
    let mut total = 0.0;
    let mut count = 0;

    for case in &dataset {
        match rt.block_on(executor.execute_graph(&objective.graph, case.input.clone())) {
            Ok(result) => {
                total += 1.0;
                let json: serde_json::Value = result.into();
                eprintln!("  case {:?}: {}", case.id, serde_json::to_string(&json).unwrap_or_default());
            }
            Err(e) => {
                eprintln!("  case {:?}: error: {}", case.id, e);
            }
        }
        count += 1;
    }

    if count > 0 {
        eprintln!("score: {:.4}", total / count as f64);
    }

    ExitCode::SUCCESS
}

// ── Optimize ──

fn cmd_optimize(
    file: &PathBuf,
    objective_name: &str,
    max_candidates: usize,
    backend: OptimizeBackendArg,
    report_dir: Option<PathBuf>,
    write_best: Option<PathBuf>,
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

    let options = OptimizationOptions {
        max_candidates,
        backend: match backend {
            OptimizeBackendArg::Grid => OptimizationBackend::Grid,
            OptimizeBackendArg::Evolutionary => OptimizationBackend::Evolutionary,
        },
        report_dir,
        write_best,
    };

    let rt = tokio::runtime::Runtime::new().unwrap();
    match rt.block_on(scaffold_runtime::optimize(&ir, objective_name, &options)) {
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
    let source = fs::read_to_string(file).map_err(|e| format!("error: cannot read {}: {}", file.display(), e))?;

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
        let content =
            fs::read_to_string(file_path).map_err(|e| format!("cannot read {}: {}", file_path, e))?;
        let json: serde_json::Value =
            serde_json::from_str(&content).map_err(|e| format!("invalid JSON in {}: {}", file_path, e))?;
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
