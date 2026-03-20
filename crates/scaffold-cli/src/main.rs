//! Scaffold DSL CLI
//!
//! Commands:
//! - check: Parse and type check a scaffold file
//! - compile: Compile to IR and output JSON
//! - parse: Parse a scaffold file and list declarations
//! - run: Execute a task directly from IR with an optional harness
//! - optimize: Search the declared harness space for an objective

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use ariadne::{Color, Label, Report, ReportKind, Source};
use clap::{Parser, Subcommand, ValueEnum};

use scaffold_ir::{
    pretty_print, to_json, AgentIR, BindingIR, BindingPathIR, ExprIR, HarnessIR, Lowerer, PromptIR,
    ScaffoldIR, StageKindIR, StringOrFileIR, TaskIR, TaskNodeIR,
};
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

        /// Write a self-contained scaffold file with the best harness materialized
        #[arg(long)]
        write_best: Option<PathBuf>,
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
            write_best,
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
            write_best.as_deref(),
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
    write_best: Option<&Path>,
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
            if let Some(write_best) = write_best {
                match write_materialized_best_scaffold(
                    write_best,
                    &ir,
                    &objective_name,
                    &artifacts.report.best,
                    base_dir,
                ) {
                    Ok(path) => eprintln!("Wrote materialized scaffold to {}", path.display()),
                    Err(error) => {
                        eprintln!("Failed to write materialized scaffold: {}", error);
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

fn write_materialized_best_scaffold(
    output_path: &Path,
    ir: &ScaffoldIR,
    objective_name: &str,
    best: &CandidateOptimizationReport,
    base_dir: &Path,
) -> Result<PathBuf, String> {
    let objective = ir
        .objectives
        .iter()
        .find(|objective| objective.name == objective_name)
        .ok_or_else(|| format!("objective '{}' not found in IR", objective_name))?;
    let original_task = ir
        .tasks
        .iter()
        .find(|task| task.name == objective.task)
        .ok_or_else(|| format!("task '{}' not found in IR", objective.task))?;
    let original_harness = ir
        .harnesses
        .iter()
        .find(|harness| harness.name == objective.harness)
        .ok_or_else(|| format!("harness '{}' not found in IR", objective.harness))?;

    let mut generated = ScaffoldIR::new();
    generated.types = ir.types.clone();
    generated.extern_crates = ir.extern_crates.clone();
    generated.foreign_modules = ir.foreign_modules.clone();
    generated.tools = ir.tools.clone();
    generated.prompts = ir.prompts.clone();
    generated.agents = ir.agents.clone();

    let mut task = original_task.clone();
    let mut harness = original_harness.clone();
    harness.name = unique_harness_name(ir, &format!("{}_optimized", objective.harness));
    harness.tunables.clear();
    apply_candidate_assignments(&mut harness, &best.assignments)?;
    materialize_task_text_surfaces(&mut task, &mut harness, &mut generated, ir, base_dir)?;
    generated.tasks = vec![task];
    generated.harnesses = vec![harness];

    let rendered = pretty_print(&generated);
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            format!(
                "failed to create parent directory {}: {}",
                parent.display(),
                error
            )
        })?;
    }
    fs::write(output_path, rendered)
        .map_err(|error| format!("failed to write {}: {}", output_path.display(), error))?;
    Ok(fs::canonicalize(output_path).unwrap_or_else(|_| output_path.to_path_buf()))
}

fn unique_harness_name(ir: &ScaffoldIR, base: &str) -> String {
    let used = ir
        .harnesses
        .iter()
        .map(|harness| harness.name.as_str())
        .collect::<HashSet<_>>();
    if !used.contains(base) {
        return base.to_string();
    }
    let mut index = 2usize;
    loop {
        let candidate = format!("{}_{}", base, index);
        if !used.contains(candidate.as_str()) {
            return candidate;
        }
        index += 1;
    }
}

fn apply_candidate_assignments(
    harness: &mut HarnessIR,
    assignments: &std::collections::HashMap<String, scaffold_runtime::Value>,
) -> Result<(), String> {
    for (path, value) in assignments {
        let segments = path.split('.').map(str::to_string).collect::<Vec<_>>();
        if segments.is_empty() {
            continue;
        }
        let expr = value_to_expr(value);
        if segments.len() == 1 {
            upsert_binding(&mut harness.defaults, segments, expr);
        } else {
            let target = segments[0].clone();
            let key = segments[1..].to_vec();
            let block = harness
                .bindings
                .iter_mut()
                .find(|binding| binding.target == target);
            match block {
                Some(block) => upsert_binding(&mut block.bindings, key, expr),
                None => harness.bindings.push(scaffold_ir::TargetBindingIR {
                    target,
                    bindings: vec![BindingIR {
                        key: BindingPathIR { segments: key },
                        value: expr,
                    }],
                }),
            }
        }
    }
    Ok(())
}

fn upsert_binding(bindings: &mut Vec<BindingIR>, segments: Vec<String>, value: ExprIR) {
    if let Some(existing) = bindings
        .iter_mut()
        .find(|binding| binding.key.segments == segments)
    {
        existing.value = value;
    } else {
        bindings.push(BindingIR {
            key: BindingPathIR { segments },
            value,
        });
    }
}

fn value_to_expr(value: &scaffold_runtime::Value) -> ExprIR {
    match value {
        scaffold_runtime::Value::Int(value) => ExprIR::Literal {
            value: scaffold_ir::LiteralIR::Int { value: *value },
        },
        scaffold_runtime::Value::Float(value) => ExprIR::Literal {
            value: scaffold_ir::LiteralIR::Float { value: *value },
        },
        scaffold_runtime::Value::String(value) => ExprIR::Literal {
            value: scaffold_ir::LiteralIR::String {
                value: value.clone(),
            },
        },
        scaffold_runtime::Value::Bytes(values) => ExprIR::List {
            elements: values
                .iter()
                .map(|value| ExprIR::Literal {
                    value: scaffold_ir::LiteralIR::Int {
                        value: i64::from(*value),
                    },
                })
                .collect(),
        },
        scaffold_runtime::Value::Bool(value) => ExprIR::Literal {
            value: scaffold_ir::LiteralIR::Bool { value: *value },
        },
        scaffold_runtime::Value::Null => ExprIR::Literal {
            value: scaffold_ir::LiteralIR::Null,
        },
        scaffold_runtime::Value::List(values) => ExprIR::List {
            elements: values.iter().map(value_to_expr).collect(),
        },
        scaffold_runtime::Value::Map(values)
        | scaffold_runtime::Value::Struct { fields: values, .. } => ExprIR::Record {
            fields: values
                .iter()
                .map(|(key, value)| scaffold_ir::ExprFieldIR {
                    key: key.clone(),
                    value: value_to_expr(value),
                })
                .collect(),
        },
        scaffold_runtime::Value::Result(result) => match &**result {
            scaffold_runtime::ResultValue::Ok(value)
            | scaffold_runtime::ResultValue::Err(value) => value_to_expr(value),
        },
    }
}

fn materialize_task_text_surfaces(
    task: &mut TaskIR,
    harness: &mut HarnessIR,
    generated: &mut ScaffoldIR,
    source_ir: &ScaffoldIR,
    base_dir: &Path,
) -> Result<(), String> {
    materialize_task_nodes(
        &mut task.body,
        harness,
        generated,
        source_ir,
        base_dir,
        &task.name,
    )
}

fn materialize_task_nodes(
    nodes: &mut [TaskNodeIR],
    harness: &mut HarnessIR,
    generated: &mut ScaffoldIR,
    source_ir: &ScaffoldIR,
    base_dir: &Path,
    task_name: &str,
) -> Result<(), String> {
    for node in nodes {
        match node {
            TaskNodeIR::Stage(stage) => {
                let resolved_component = harness_string_field(harness, &stage.name, "component")?
                    .unwrap_or_else(|| stage.component.clone());
                match stage.stage_kind {
                    StageKindIR::Prompt => {
                        let prompt = source_ir
                            .prompts
                            .iter()
                            .find(|prompt| prompt.name == resolved_component)
                            .ok_or_else(|| {
                                format!(
                                    "prompt '{}' not found for stage '{}'",
                                    resolved_component, stage.name
                                )
                            })?;
                        let cloned_name = unique_component_name(
                            generated.prompts.iter().map(|prompt| prompt.name.as_str()),
                            &format!("{}_{}_optimized", prompt.name, stage.name),
                        );
                        let template =
                            resolved_prompt_template(prompt, harness, &stage.name, base_dir)?;
                        let system =
                            resolved_prompt_system(prompt, harness, &stage.name, base_dir)?;
                        generated.prompts.push(PromptIR {
                            name: cloned_name.clone(),
                            input: prompt.input.clone(),
                            output: prompt.output.clone(),
                            template: StringOrFileIR::Literal { value: template },
                            system: system.map(|value| StringOrFileIR::Literal { value }),
                        });
                        stage.component = cloned_name;
                        strip_stage_bindings(
                            harness,
                            &stage.name,
                            &["component", "variant", "system_prompt"],
                        );
                    }
                    StageKindIR::Agent => {
                        let agent = source_ir
                            .agents
                            .iter()
                            .find(|agent| agent.name == resolved_component)
                            .ok_or_else(|| {
                                format!(
                                    "agent '{}' not found for stage '{}'",
                                    resolved_component, stage.name
                                )
                            })?;
                        let cloned_name = unique_component_name(
                            generated.agents.iter().map(|agent| agent.name.as_str()),
                            &format!("{}_{}_optimized", agent.name, stage.name),
                        );
                        let system = resolved_agent_system(agent, harness, &stage.name, base_dir)?;
                        let mut cloned = agent.clone();
                        cloned.name = cloned_name.clone();
                        cloned.system = StringOrFileIR::Literal { value: system };
                        generated.agents.push(cloned);
                        stage.component = cloned_name;
                        strip_stage_bindings(
                            harness,
                            &stage.name,
                            &["component", "variant", "system_prompt"],
                        );
                    }
                    StageKindIR::Tool => {
                        stage.component = resolved_component;
                        strip_stage_bindings(harness, &stage.name, &["component"]);
                    }
                }
            }
            TaskNodeIR::Loop(loop_ir) => {
                materialize_task_nodes(
                    &mut loop_ir.body,
                    harness,
                    generated,
                    source_ir,
                    base_dir,
                    task_name,
                )?;
            }
            TaskNodeIR::Branch(branch) => {
                materialize_task_nodes(
                    &mut branch.then_body,
                    harness,
                    generated,
                    source_ir,
                    base_dir,
                    task_name,
                )?;
                materialize_task_nodes(
                    &mut branch.else_body,
                    harness,
                    generated,
                    source_ir,
                    base_dir,
                    task_name,
                )?;
            }
        }
    }
    Ok(())
}

fn unique_component_name<'a>(existing: impl Iterator<Item = &'a str>, base: &str) -> String {
    let used = existing.collect::<HashSet<_>>();
    if !used.contains(base) {
        return base.to_string();
    }
    let mut index = 2usize;
    loop {
        let candidate = format!("{}_{}", base, index);
        if !used.contains(candidate.as_str()) {
            return candidate;
        }
        index += 1;
    }
}

fn resolved_prompt_template(
    prompt: &PromptIR,
    harness: &HarnessIR,
    target: &str,
    base_dir: &Path,
) -> Result<String, String> {
    if let Some(variant) = harness_string_field(harness, target, "variant")? {
        return resolve_text_surface(base_dir, &variant, true);
    }
    resolve_string_or_file(base_dir, &prompt.template)
}

fn resolved_prompt_system(
    prompt: &PromptIR,
    harness: &HarnessIR,
    target: &str,
    base_dir: &Path,
) -> Result<Option<String>, String> {
    if let Some(system_prompt) = harness_string_field(harness, target, "system_prompt")? {
        return resolve_text_surface(base_dir, &system_prompt, false).map(Some);
    }
    prompt
        .system
        .as_ref()
        .map(|value| resolve_string_or_file(base_dir, value))
        .transpose()
}

fn resolved_agent_system(
    agent: &AgentIR,
    harness: &HarnessIR,
    target: &str,
    base_dir: &Path,
) -> Result<String, String> {
    if let Some(system_prompt) = harness_string_field(harness, target, "system_prompt")? {
        return resolve_text_surface(base_dir, &system_prompt, false);
    }
    if let Some(variant) = harness_string_field(harness, target, "variant")? {
        return resolve_text_surface(base_dir, &variant, true);
    }
    resolve_string_or_file(base_dir, &agent.system)
}

fn harness_string_field(
    harness: &HarnessIR,
    target: &str,
    field: &str,
) -> Result<Option<String>, String> {
    if let Some(value) = harness
        .bindings
        .iter()
        .find(|binding| binding.target == target)
        .and_then(|binding| {
            binding
                .bindings
                .iter()
                .find(|item| item.key.segments.join(".") == field)
        })
        .map(|binding| &binding.value)
    {
        return expr_to_string(value).map(Some);
    }

    harness
        .defaults
        .iter()
        .find(|binding| binding.key.segments.join(".") == field)
        .map(|binding| expr_to_string(&binding.value))
        .transpose()
}

fn strip_stage_bindings(harness: &mut HarnessIR, target: &str, fields: &[&str]) {
    let fields = fields.iter().copied().collect::<HashSet<_>>();
    if let Some(index) = harness
        .bindings
        .iter()
        .position(|binding| binding.target == target)
    {
        harness.bindings[index]
            .bindings
            .retain(|item| !fields.contains(item.key.segments.join(".").as_str()));
        if harness.bindings[index].bindings.is_empty() {
            harness.bindings.remove(index);
        }
    }
}

fn expr_to_string(expr: &ExprIR) -> Result<String, String> {
    match expr {
        ExprIR::Literal {
            value: scaffold_ir::LiteralIR::String { value },
        } => Ok(value.clone()),
        ExprIR::Call { function, args } if function == "variant" && args.len() == 2 => {
            let group = expr_to_string(&args[0])?;
            let name = expr_to_string(&args[1])?;
            Ok(format!("{}::{}", group, name))
        }
        ExprIR::Ident { name } => Ok(name.clone()),
        other => Err(format!(
            "cannot materialize non-string binding expression: {:?}",
            other
        )),
    }
}

fn resolve_string_or_file(base_dir: &Path, value: &StringOrFileIR) -> Result<String, String> {
    match value {
        StringOrFileIR::Literal { value } => Ok(value.clone()),
        StringOrFileIR::File { path } => {
            let resolved = base_dir.join(path);
            fs::read_to_string(&resolved)
                .map_err(|error| format!("failed to read {}: {}", resolved.display(), error))
        }
    }
}

fn resolve_text_surface(
    base_dir: &Path,
    value: &str,
    require_variant: bool,
) -> Result<String, String> {
    if let Some((group, name)) = value.split_once("::") {
        if let Some(path) = find_variant_file(base_dir, group, name) {
            return fs::read_to_string(&path)
                .map_err(|error| format!("failed to read {}: {}", path.display(), error));
        }
        if require_variant {
            return Err(format!(
                "variant '{}' could not be resolved under {}",
                value,
                base_dir.display()
            ));
        }
    }
    Ok(value.to_string())
}

fn find_variant_file(base_dir: &Path, group: &str, name: &str) -> Option<PathBuf> {
    let candidates = [
        base_dir
            .join("variants")
            .join(group)
            .join(format!("{}.md", name)),
        base_dir
            .join("variants")
            .join(group)
            .join(format!("{}.txt", name)),
        base_dir
            .join("variants")
            .join(group)
            .join(format!("{}.prompt", name)),
        base_dir
            .join("prompts")
            .join(group)
            .join(format!("{}.md", name)),
        base_dir
            .join("prompts")
            .join(group)
            .join(format!("{}.txt", name)),
        base_dir.join("prompts").join(format!("{}.md", name)),
        base_dir.join("prompts").join(format!("{}.txt", name)),
    ];
    candidates.into_iter().find(|path| path.is_file())
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
