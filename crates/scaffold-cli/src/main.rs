//! Scaffold DSL CLI
//!
//! Commands:
//! - check: Parse and type check a scaffold file
//! - compile: Compile to IR and output JSON
//! - parse: Parse a scaffold file and list declarations
//! - run: Execute a task directly from IR with an optional harness
//! - optimize: Search the declared harness space for an objective

use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use ariadne::{Color, Label, Report, ReportKind, Source};
use clap::{Parser, Subcommand, ValueEnum};

use scaffold_ir::{to_json, Lowerer, ScaffoldIR};
use scaffold_runtime::{
    evaluate_objective_candidate as evaluate_ir_objective_candidate,
    execute_task as execute_ir_task, optimize_objective_with_artifacts as optimize_ir_objective,
    CandidateOptimizationReport, ObjectiveOptimizationArtifacts, OptimizationBackendKind,
    OptimizationOptions,
};
use scaffold_syntax::parse;
use scaffold_types::check;
use scaffold_verify::{verify, Severity};

#[derive(Parser)]
#[command(name = "scaffold")]
#[command(author, version, about = "Scaffold DSL compiler", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Clone, Debug, ValueEnum)]
enum OptimizeBackend {
    Interpreter,
    Dspy,
}

#[derive(Subcommand)]
enum Commands {
    /// Parse and type check a scaffold file
    Check {
        /// Input file
        #[arg(value_name = "FILE")]
        file: PathBuf,

        /// Show verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Compile a scaffold file to IR
    Compile {
        /// Input file
        #[arg(value_name = "FILE")]
        file: PathBuf,

        /// Output file (defaults to stdout)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Compact JSON output
        #[arg(short, long)]
        compact: bool,
    },

    /// Parse only (for debugging)
    Parse {
        /// Input file
        #[arg(value_name = "FILE")]
        file: PathBuf,
    },

    /// Execute a task from a scaffold file
    Run {
        /// Input file
        #[arg(value_name = "FILE")]
        file: PathBuf,

        /// Task to execute
        #[arg(short, long)]
        task: Option<String>,

        /// Harness to apply when running a task
        #[arg(long)]
        harness: Option<String>,

        /// Input JSON (or @filename for file input)
        #[arg(short, long, default_value = "{}")]
        input: String,

        /// Show verbose output
        #[arg(short, long)]
        verbose: bool,

        /// Enable verification checks
        #[arg(long)]
        verify: bool,

        /// Config file to load into the runtime
        #[arg(long)]
        config: Option<PathBuf>,
    },

    /// Optimize an objective by searching its declared harness space
    Optimize {
        /// Input file
        #[arg(value_name = "FILE")]
        file: PathBuf,

        /// Objective to optimize
        #[arg(short, long)]
        objective: Option<String>,

        /// Show verbose output
        #[arg(short, long)]
        verbose: bool,

        /// Enable verification checks
        #[arg(long)]
        verify: bool,

        /// Config file to load into the runtime
        #[arg(long)]
        config: Option<PathBuf>,

        /// Maximum number of candidates to evaluate
        #[arg(long, default_value_t = 256)]
        max_candidates: usize,

        /// Candidate proposal backend
        #[arg(long, value_enum, default_value_t = OptimizeBackend::Interpreter)]
        backend: OptimizeBackend,

        /// Override the backend command for external backends like DSPy
        #[arg(long)]
        backend_command: Option<String>,

        /// Directory where the optimization report and candidate summaries are written
        #[arg(long)]
        report_dir: Option<PathBuf>,
    },

    /// Evaluate an objective with the harness defaults or explicit assignments
    Evaluate {
        /// Input file
        #[arg(value_name = "FILE")]
        file: PathBuf,

        /// Objective to evaluate
        #[arg(short, long)]
        objective: Option<String>,

        /// Candidate assignment JSON object to apply on top of the harness defaults
        #[arg(long, default_value = "{}")]
        assignments: String,

        /// Restrict evaluation to a single dataset case id
        #[arg(long)]
        case_id: Option<String>,

        /// Enable verification checks
        #[arg(long)]
        verify: bool,

        /// Config file to load into the runtime
        #[arg(long)]
        config: Option<PathBuf>,
    },

    #[command(hide = true, name = "internal-evaluate-candidate")]
    InternalEvaluateCandidate {
        #[arg(value_name = "FILE")]
        file: PathBuf,

        #[arg(long)]
        objective: String,

        #[arg(long)]
        assignments: String,

        #[arg(long)]
        case_id: Option<String>,

        #[arg(long)]
        verify: bool,

        #[arg(long)]
        config: Option<PathBuf>,
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
        Commands::Parse { file } => cmd_parse(&file),
        Commands::Run {
            file,
            task,
            harness,
            input,
            verbose,
            verify,
            config,
        } => cmd_run(
            &file,
            task.as_deref(),
            harness.as_deref(),
            &input,
            verbose,
            verify,
            config.as_deref(),
        ),
        Commands::Optimize {
            file,
            objective,
            verbose,
            verify,
            config,
            max_candidates,
            backend,
            backend_command,
            report_dir,
        } => cmd_optimize(
            &file,
            objective.as_deref(),
            verbose,
            verify,
            config.as_deref(),
            max_candidates,
            backend,
            backend_command.as_deref(),
            report_dir.as_deref(),
        ),
        Commands::Evaluate {
            file,
            objective,
            assignments,
            case_id,
            verify,
            config,
        } => cmd_evaluate(
            &file,
            objective.as_deref(),
            &assignments,
            case_id.as_deref(),
            verify,
            config.as_deref(),
        ),
        Commands::InternalEvaluateCandidate {
            file,
            objective,
            assignments,
            case_id,
            verify,
            config,
        } => cmd_internal_evaluate_candidate(
            &file,
            &objective,
            &assignments,
            case_id.as_deref(),
            verify,
            config.as_deref(),
        ),
    }
}

fn cmd_check(file: &PathBuf, verbose: bool) -> ExitCode {
    let source = match fs::read_to_string(file) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error reading file: {}", e);
            return ExitCode::FAILURE;
        }
    };
    let file_name = file.display().to_string();

    let program = match parse(&source) {
        Ok(program) => program,
        Err(error) => {
            emit_parse_error(
                &file_name,
                &source,
                &error.message,
                error.span.start,
                error.span.end,
            );
            return ExitCode::FAILURE;
        }
    };

    if verbose {
        println!("Parsed {} declarations", program.declarations.len());
    }

    let type_env = match check(&program) {
        Ok(env) => env,
        Err(errors) => {
            emit_type_errors(&file_name, &source, &errors);
            return ExitCode::FAILURE;
        }
    };

    if verbose {
        println!("Type checking passed");
        println!("  {} types defined", type_env.types.len());
    }

    let verify_result = verify(&program, &type_env);
    let mut has_errors = false;
    for error in &verify_result.errors {
        if error.severity == Severity::Error {
            has_errors = true;
        }
        Report::build(
            match error.severity {
                Severity::Error => ReportKind::Error,
                Severity::Warning => ReportKind::Warning,
            },
            &file_name,
            error.span.start,
        )
        .with_message("Verification")
        .with_label(
            Label::new((&file_name, error.span.start..error.span.end))
                .with_message(&error.message)
                .with_color(match error.severity {
                    Severity::Error => Color::Red,
                    Severity::Warning => Color::Yellow,
                }),
        )
        .finish()
        .eprint((&file_name, Source::from(&source)))
        .unwrap();
    }

    if has_errors {
        return ExitCode::FAILURE;
    }

    if verbose {
        println!("Verification passed");
    }

    println!("{}: OK", file.display());
    ExitCode::SUCCESS
}

fn cmd_compile(file: &PathBuf, output: Option<&Path>, compact: bool) -> ExitCode {
    let ir = match parse_typecheck_lower(file, true) {
        Ok(ir) => ir,
        Err(code) => return code,
    };

    let json = if compact {
        scaffold_ir::to_json_compact(&ir)
    } else {
        to_json(&ir)
    };

    let json = match json {
        Ok(json) => json,
        Err(error) => {
            eprintln!("Error serializing IR: {}", error);
            return ExitCode::FAILURE;
        }
    };

    match output {
        Some(path) => {
            if let Err(error) = fs::write(path, &json) {
                eprintln!("Error writing output file: {}", error);
                return ExitCode::FAILURE;
            }
            eprintln!("Compiled to {}", path.display());
        }
        None => println!("{}", json),
    }

    ExitCode::SUCCESS
}

fn cmd_parse(file: &PathBuf) -> ExitCode {
    let source = match fs::read_to_string(file) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error reading file: {}", e);
            return ExitCode::FAILURE;
        }
    };
    let file_name = file.display().to_string();

    match parse(&source) {
        Ok(program) => {
            println!("Parsed successfully!");
            println!("Declarations: {}", program.declarations.len());
            for decl in &program.declarations {
                match decl {
                    scaffold_syntax::Declaration::Type(t) => println!("  type {}", t.name.node),
                    scaffold_syntax::Declaration::Artifact(a) => {
                        println!("  artifact {}", a.name.node)
                    }
                    scaffold_syntax::Declaration::ExternCrate(e) => {
                        println!("  extern crate {} = \"{}\"", e.name.node, e.version)
                    }
                    scaffold_syntax::Declaration::Foreign(f) => {
                        println!("  foreign {} {}", f.language.node, f.name.node)
                    }
                    scaffold_syntax::Declaration::Tool(t) => println!("  tool {}", t.name.node),
                    scaffold_syntax::Declaration::Prompt(p) => {
                        println!("  prompt {}", p.name.node)
                    }
                    scaffold_syntax::Declaration::Agent(a) => println!("  agent {}", a.name.node),
                    scaffold_syntax::Declaration::Pipeline(p) => {
                        println!("  pipeline {}", p.name.node)
                    }
                    scaffold_syntax::Declaration::Task(t) => println!("  task {}", t.name.node),
                    scaffold_syntax::Declaration::Harness(h) => {
                        println!("  harness {}", h.name.node)
                    }
                    scaffold_syntax::Declaration::Objective(o) => {
                        println!("  objective {}", o.name.node)
                    }
                }
            }
            ExitCode::SUCCESS
        }
        Err(error) => {
            emit_parse_error(
                &file_name,
                &source,
                &error.message,
                error.span.start,
                error.span.end,
            );
            ExitCode::FAILURE
        }
    }
}

fn cmd_run(
    file: &PathBuf,
    task: Option<&str>,
    harness: Option<&str>,
    input_str: &str,
    verbose: bool,
    verify_enabled: bool,
    config: Option<&Path>,
) -> ExitCode {
    let input_json = match parse_run_input(input_str) {
        Ok(value) => value,
        Err(code) => return code,
    };

    let ir = match parse_typecheck_lower(file, verify_enabled) {
        Ok(ir) => ir,
        Err(code) => return code,
    };

    let task_name = if let Some(name) = task {
        name.to_string()
    } else if ir.tasks.len() == 1 {
        ir.tasks[0].name.clone()
    } else if ir.tasks.is_empty() {
        eprintln!("No tasks found in {}", file.display());
        return ExitCode::FAILURE;
    } else {
        let task_names = ir
            .tasks
            .iter()
            .map(|task| task.name.clone())
            .collect::<Vec<_>>();
        eprintln!("Multiple tasks available. Please specify one:");
        eprintln!("  Tasks: {:?}", task_names);
        eprintln!(
            "\nUsage: scaffold run {} --task <TASK> [--harness <HARNESS>]",
            file.display()
        );
        return ExitCode::FAILURE;
    };

    if !ir.tasks.iter().any(|candidate| candidate.name == task_name) {
        eprintln!("task '{}' not found in {}", task_name, file.display());
        return ExitCode::FAILURE;
    }

    if let Some(config_path) = config {
        if let Err(error) = scaffold_runtime::config::init_from_path(config_path) {
            if error != "Config already initialized" {
                eprintln!(
                    "Failed to initialize runtime config from {}: {}",
                    config_path.display(),
                    error
                );
                return ExitCode::FAILURE;
            }
        }
        // Propagate explicit config selection to subprocess-based optimizer backends.
        std::env::set_var("SCAFFOLD_CONFIG_PATH", config_path);
    }

    let input_value = match serde_json::from_str::<serde_json::Value>(&input_json) {
        Ok(value) => scaffold_runtime::Value::from(value),
        Err(error) => {
            eprintln!("Error parsing input JSON: {}", error);
            return ExitCode::FAILURE;
        }
    };

    if verbose {
        if let Some(harness_name) = harness {
            eprintln!("Running task {} with harness {}", task_name, harness_name);
        } else {
            eprintln!("Running task {}", task_name);
        }
    }

    let base_dir = file.parent().unwrap_or_else(|| Path::new("."));
    match execute_ir_task(&ir, &task_name, harness, input_value, base_dir) {
        Ok(output) => match serde_json::to_string_pretty(&serde_json::Value::from(output)) {
            Ok(json) => {
                println!("{}", json);
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("Failed to serialize task output: {}", error);
                ExitCode::FAILURE
            }
        },
        Err(error) => {
            eprintln!("Task execution failed: {}", error);
            ExitCode::FAILURE
        }
    }
}

fn cmd_optimize(
    file: &PathBuf,
    objective: Option<&str>,
    verbose: bool,
    verify_enabled: bool,
    config: Option<&Path>,
    max_candidates: usize,
    backend: OptimizeBackend,
    backend_command: Option<&str>,
    report_dir: Option<&Path>,
) -> ExitCode {
    let ir = match parse_typecheck_lower(file, verify_enabled) {
        Ok(ir) => ir,
        Err(code) => return code,
    };

    let objective_name = if let Some(name) = objective {
        name.to_string()
    } else if ir.objectives.len() == 1 {
        ir.objectives[0].name.clone()
    } else if ir.objectives.is_empty() {
        eprintln!("No objectives found in {}", file.display());
        return ExitCode::FAILURE;
    } else {
        let objective_names = ir
            .objectives
            .iter()
            .map(|objective| objective.name.clone())
            .collect::<Vec<_>>();
        eprintln!("Multiple objectives available. Please specify one:");
        eprintln!("  Objectives: {:?}", objective_names);
        eprintln!(
            "\nUsage: scaffold optimize {} --objective <OBJECTIVE>",
            file.display()
        );
        return ExitCode::FAILURE;
    };

    if !ir
        .objectives
        .iter()
        .any(|candidate| candidate.name == objective_name)
    {
        eprintln!(
            "objective '{}' not found in {}",
            objective_name,
            file.display()
        );
        return ExitCode::FAILURE;
    }

    if let Some(config_path) = config {
        if let Err(error) = scaffold_runtime::config::init_from_path(config_path) {
            if error != "Config already initialized" {
                eprintln!(
                    "Failed to initialize runtime config from {}: {}",
                    config_path.display(),
                    error
                );
                return ExitCode::FAILURE;
            }
        }
        std::env::set_var("SCAFFOLD_CONFIG_PATH", config_path);
    }

    if verbose {
        eprintln!(
            "Optimizing objective {} with {:?} backend and up to {} candidate(s)",
            objective_name, backend, max_candidates
        );
    }

    let base_dir = file.parent().unwrap_or_else(|| Path::new("."));
    let options = OptimizationOptions {
        backend: match backend {
            OptimizeBackend::Interpreter => OptimizationBackendKind::Interpreter,
            OptimizeBackend::Dspy => OptimizationBackendKind::Dspy,
        },
        max_candidates,
        backend_command: backend_command.map(str::to_string),
        source_file: Some(fs::canonicalize(file).unwrap_or_else(|_| file.clone())),
    };
    match optimize_ir_objective(&ir, &objective_name, base_dir, options) {
        Ok(artifacts) => {
            if let Some(report_dir) = report_dir {
                match write_optimization_report_dir(report_dir, &artifacts) {
                    Ok(path) => eprintln!("Wrote optimization report to {}", path.display()),
                    Err(error) => {
                        eprintln!("Failed to write optimization report dir: {}", error);
                        return ExitCode::FAILURE;
                    }
                }
            }
            match serde_json::to_string_pretty(&artifacts.report) {
                Ok(json) => {
                    println!("{}", json);
                    ExitCode::SUCCESS
                }
                Err(error) => {
                    eprintln!("Failed to serialize optimization report: {}", error);
                    ExitCode::FAILURE
                }
            }
        }
        Err(error) => {
            eprintln!("Optimization failed: {}", error);
            ExitCode::FAILURE
        }
    }
}

fn write_optimization_report_dir(
    report_dir: &Path,
    artifacts: &ObjectiveOptimizationArtifacts,
) -> Result<PathBuf, String> {
    fs::create_dir_all(report_dir).map_err(|error| {
        format!(
            "failed to create report directory {}: {}",
            report_dir.display(),
            error
        )
    })?;

    let report_path = report_dir.join("report.json");
    let best_candidate_path = report_dir.join("best.candidate.json");
    let best_assignments_path = report_dir.join("best.assignments.json");
    let candidates_path = report_dir.join("candidates.jsonl");

    let report_json = serde_json::to_string_pretty(&artifacts.report)
        .map_err(|error| format!("failed to serialize report.json: {}", error))?;
    fs::write(&report_path, report_json)
        .map_err(|error| format!("failed to write {}: {}", report_path.display(), error))?;

    let best_candidate_json = serde_json::to_string_pretty(&artifacts.report.best)
        .map_err(|error| format!("failed to serialize best.candidate.json: {}", error))?;
    fs::write(&best_candidate_path, best_candidate_json).map_err(|error| {
        format!(
            "failed to write {}: {}",
            best_candidate_path.display(),
            error
        )
    })?;

    let best_assignments_json = serde_json::to_string_pretty(&artifacts.report.best.assignments)
        .map_err(|error| format!("failed to serialize best.assignments.json: {}", error))?;
    fs::write(&best_assignments_path, best_assignments_json).map_err(|error| {
        format!(
            "failed to write {}: {}",
            best_assignments_path.display(),
            error
        )
    })?;

    let mut candidates_jsonl = String::new();
    for (index, candidate) in artifacts.candidates.iter().enumerate() {
        let line = candidate_report_line(index, candidate)
            .map_err(|error| format!("failed to serialize candidates.jsonl: {}", error))?;
        candidates_jsonl.push_str(&line);
        candidates_jsonl.push('\n');
    }
    fs::write(&candidates_path, candidates_jsonl)
        .map_err(|error| format!("failed to write {}: {}", candidates_path.display(), error))?;

    Ok(fs::canonicalize(report_dir).unwrap_or_else(|_| report_dir.to_path_buf()))
}

fn candidate_report_line(
    index: usize,
    candidate: &CandidateOptimizationReport,
) -> serde_json::Result<String> {
    let mut payload = serde_json::to_value(candidate)?;
    let object = payload
        .as_object_mut()
        .expect("candidate reports always serialize to objects");
    object.insert("index".to_string(), serde_json::json!(index));
    serde_json::to_string(&payload)
}

fn cmd_evaluate(
    file: &PathBuf,
    objective: Option<&str>,
    assignments: &str,
    case_id: Option<&str>,
    verify_enabled: bool,
    config: Option<&Path>,
) -> ExitCode {
    let ir = match parse_typecheck_lower(file, verify_enabled) {
        Ok(ir) => ir,
        Err(code) => return code,
    };

    let objective_name = if let Some(name) = objective {
        name.to_string()
    } else if ir.objectives.len() == 1 {
        ir.objectives[0].name.clone()
    } else if ir.objectives.is_empty() {
        eprintln!("No objectives found in {}", file.display());
        return ExitCode::FAILURE;
    } else {
        let objective_names = ir
            .objectives
            .iter()
            .map(|objective| objective.name.clone())
            .collect::<Vec<_>>();
        eprintln!("Multiple objectives available. Please specify one:");
        eprintln!("  Objectives: {:?}", objective_names);
        eprintln!(
            "\nUsage: scaffold evaluate {} --objective <OBJECTIVE>",
            file.display()
        );
        return ExitCode::FAILURE;
    };

    if !ir
        .objectives
        .iter()
        .any(|candidate| candidate.name == objective_name)
    {
        eprintln!(
            "objective '{}' not found in {}",
            objective_name,
            file.display()
        );
        return ExitCode::FAILURE;
    }

    if let Some(config_path) = config {
        if let Err(error) = scaffold_runtime::config::init_from_path(config_path) {
            if error != "Config already initialized" {
                eprintln!(
                    "Failed to initialize runtime config from {}: {}",
                    config_path.display(),
                    error
                );
                return ExitCode::FAILURE;
            }
        }
        std::env::set_var("SCAFFOLD_CONFIG_PATH", config_path);
    }

    cmd_evaluate_candidate(file, &ir, &objective_name, assignments, case_id)
}

fn cmd_internal_evaluate_candidate(
    file: &PathBuf,
    objective: &str,
    assignments: &str,
    case_id: Option<&str>,
    verify_enabled: bool,
    config: Option<&Path>,
) -> ExitCode {
    let ir = match parse_typecheck_lower(file, verify_enabled) {
        Ok(ir) => ir,
        Err(code) => return code,
    };

    let inherited_config = std::env::var_os("SCAFFOLD_CONFIG_PATH").map(PathBuf::from);
    if let Some(config_path) = config.map(PathBuf::from).or(inherited_config) {
        if let Err(error) = scaffold_runtime::config::init_from_path(&config_path) {
            if error != "Config already initialized" {
                eprintln!(
                    "Failed to initialize runtime config from {}: {}",
                    config_path.display(),
                    error
                );
                return ExitCode::FAILURE;
            }
        }
    }

    cmd_evaluate_candidate(file, &ir, objective, assignments, case_id)
}

fn cmd_evaluate_candidate(
    file: &PathBuf,
    ir: &ScaffoldIR,
    objective: &str,
    assignments: &str,
    case_id: Option<&str>,
) -> ExitCode {
    let assignments_json = match serde_json::from_str::<serde_json::Value>(assignments) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("Failed to parse assignments JSON: {}", error);
            return ExitCode::FAILURE;
        }
    };
    let assignments_object = match assignments_json {
        serde_json::Value::Object(map) => map,
        _ => {
            eprintln!("Assignments JSON must be an object");
            return ExitCode::FAILURE;
        }
    };

    let assignments_map = assignments_object
        .into_iter()
        .map(|(key, value)| (key, scaffold_runtime::Value::from(value)))
        .collect();

    let base_dir = file.parent().unwrap_or_else(|| Path::new("."));
    match evaluate_ir_objective_candidate(&ir, objective, &assignments_map, base_dir, case_id) {
        Ok(report) => match serde_json::to_string_pretty(&report) {
            Ok(json) => {
                println!("{}", json);
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("Failed to serialize candidate evaluation report: {}", error);
                ExitCode::FAILURE
            }
        },
        Err(error) => {
            eprintln!("Candidate evaluation failed: {}", error);
            ExitCode::FAILURE
        }
    }
}

fn parse_run_input(input_str: &str) -> Result<String, ExitCode> {
    let json_input = if let Some(input_path) = input_str.strip_prefix('@') {
        match fs::read_to_string(input_path) {
            Ok(content) => content,
            Err(error) => {
                eprintln!("Error reading input file {}: {}", input_path, error);
                return Err(ExitCode::FAILURE);
            }
        }
    } else {
        input_str.to_string()
    };

    if let Err(error) = serde_json::from_str::<serde_json::Value>(&json_input) {
        eprintln!("Error parsing input JSON: {}", error);
        return Err(ExitCode::FAILURE);
    }

    Ok(json_input)
}

fn parse_typecheck_lower(file: &Path, verify_enabled: bool) -> Result<ScaffoldIR, ExitCode> {
    let source = match fs::read_to_string(file) {
        Ok(s) => s,
        Err(error) => {
            eprintln!("Error reading file {}: {}", file.display(), error);
            return Err(ExitCode::FAILURE);
        }
    };
    let file_name = file.display().to_string();

    let program = match parse(&source) {
        Ok(program) => program,
        Err(error) => {
            emit_parse_error(
                &file_name,
                &source,
                &error.message,
                error.span.start,
                error.span.end,
            );
            return Err(ExitCode::FAILURE);
        }
    };

    let type_env = match check(&program) {
        Ok(env) => env,
        Err(errors) => {
            emit_type_errors(&file_name, &source, &errors);
            return Err(ExitCode::FAILURE);
        }
    };

    if verify_enabled {
        let verify_result = verify(&program, &type_env);
        let mut has_errors = false;
        for error in &verify_result.errors {
            if error.severity == Severity::Error {
                has_errors = true;
            }
            Report::build(
                match error.severity {
                    Severity::Error => ReportKind::Error,
                    Severity::Warning => ReportKind::Warning,
                },
                &file_name,
                error.span.start,
            )
            .with_message("Verification")
            .with_label(
                Label::new((&file_name, error.span.start..error.span.end))
                    .with_message(&error.message)
                    .with_color(match error.severity {
                        Severity::Error => Color::Red,
                        Severity::Warning => Color::Yellow,
                    }),
            )
            .finish()
            .eprint((&file_name, Source::from(&source)))
            .unwrap();
        }
        if has_errors {
            return Err(ExitCode::FAILURE);
        }
    }

    let lowerer = Lowerer::new().with_source_file(file_name.clone());
    match lowerer.lower(&program, &type_env) {
        Ok(ir) => Ok(ir),
        Err(error) => {
            Report::build(ReportKind::Error, &file_name, error.span.start)
                .with_message("Lowering error")
                .with_label(
                    Label::new((&file_name, error.span.start..error.span.end))
                        .with_message(&error.message)
                        .with_color(Color::Red),
                )
                .finish()
                .eprint((&file_name, Source::from(&source)))
                .unwrap();
            Err(ExitCode::FAILURE)
        }
    }
}

fn emit_parse_error(file_name: &str, source: &str, message: &str, start: usize, end: usize) {
    Report::build(ReportKind::Error, file_name, start)
        .with_message("Parse error")
        .with_label(
            Label::new((file_name, start..end))
                .with_message(message)
                .with_color(Color::Red),
        )
        .finish()
        .eprint((file_name, Source::from(source)))
        .unwrap();
}

fn emit_type_errors(file_name: &str, source: &str, errors: &[scaffold_types::TypeError]) {
    for error in errors {
        Report::build(ReportKind::Error, file_name, error.span.start)
            .with_message("Type error")
            .with_label(
                Label::new((file_name, error.span.start..error.span.end))
                    .with_message(&error.message)
                    .with_color(Color::Red),
            )
            .finish()
            .eprint((file_name, Source::from(source)))
            .unwrap();
    }
}
