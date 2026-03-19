//! Scaffold DSL CLI
//!
//! Commands:
//! - check: Parse and type check a scaffold file
//! - compile: Compile to IR and output JSON
//! - parse: Parse a scaffold file and list declarations
//! - run: Execute a task directly from IR with an optional harness

use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use ariadne::{Color, Label, Report, ReportKind, Source};
use clap::{Parser, Subcommand};

use scaffold_ir::{to_json, Lowerer, ScaffoldIR};
use scaffold_runtime::execute_task as execute_ir_task;
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
