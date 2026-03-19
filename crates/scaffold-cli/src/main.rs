//! Scaffold DSL CLI
//!
//! Commands:
//! - check: Parse and type check a scaffold file
//! - compile: Compile to IR and output JSON
//! - codegen: Generate Rust code from scaffold file
//! - run: Build and execute a scaffold file via generated binary

use std::fs;
use std::hash::{Hash, Hasher};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{ExitCode, Stdio};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use ariadne::{Color, Label, Report, ReportKind, Source};
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};

use scaffold_codegen::CodeGenerator;
use scaffold_ir::{to_json, Lowerer, ScaffoldIR};
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

    /// Generate Rust code from scaffold file
    Codegen {
        /// Input file
        #[arg(value_name = "FILE")]
        file: PathBuf,

        /// Output directory
        #[arg(short, long)]
        output: PathBuf,

        /// Format generated code with rustfmt (enabled by default)
        #[arg(long, default_value = "true", action = clap::ArgAction::Set)]
        format: bool,

        /// Enforce strict validation (no unresolved identifiers/calls)
        #[arg(long)]
        strict: bool,
    },

    /// Build a native binary from a scaffold file (codegen + cargo build)
    Build {
        /// Input file
        #[arg(value_name = "FILE")]
        file: PathBuf,

        /// Output directory for the generated crate (defaults to ./generated)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Binary name override (defaults to first pipeline/tool name)
        #[arg(long)]
        bin_name: Option<String>,

        /// Format generated code with rustfmt (enabled by default)
        #[arg(long, default_value = "true", action = clap::ArgAction::Set)]
        format: bool,

        /// Enforce strict validation (no unresolved identifiers/calls)
        #[arg(long)]
        strict: bool,

        /// Path to scaffold-runtime crate (auto-detected if not specified)
        #[arg(long)]
        runtime_path: Option<String>,

        /// Embed a scaffold config TOML into the generated binary
        #[arg(long)]
        embed_config: Option<PathBuf>,

        /// Build in release mode
        #[arg(long)]
        release: bool,
    },

    /// Build and run a scaffold file
    Run {
        /// Input file
        #[arg(value_name = "FILE")]
        file: PathBuf,

        /// Task to execute
        #[arg(short, long)]
        task: Option<String>,

        /// Tool to execute (alternative to task)
        #[arg(long)]
        tool: Option<String>,

        /// Prompt to execute
        #[arg(long)]
        prompt: Option<String>,

        /// Agent to execute
        #[arg(long)]
        agent: Option<String>,

        /// Pipeline to execute
        #[arg(long)]
        pipeline: Option<String>,

        /// Input JSON (or @filename for file input)
        #[arg(short, long, default_value = "{}")]
        input: String,

        /// Show verbose output
        #[arg(short, long)]
        verbose: bool,

        /// Enable verification checks
        #[arg(long)]
        verify: bool,

        /// Config file to pass to the compiled binary
        #[arg(long)]
        config: Option<PathBuf>,

        /// Output directory for generated run crate
        #[arg(long)]
        build_dir: Option<PathBuf>,

        /// Save agent conversations (including nested agents in pipelines) to JSONL
        #[arg(long)]
        save_convos: bool,
    },

    /// Evaluate a harness scaffold against a fixed objective and persist rollouts/traces
    Optimize {
        /// Harness scaffold file that can be tuned/mutated
        #[arg(value_name = "HARNESS")]
        harness: PathBuf,

        /// Objective file (JSON) that stays fixed
        #[arg(long, value_name = "FILE")]
        objective: PathBuf,

        /// Config file to pass to the compiled binary
        #[arg(long)]
        config: Option<PathBuf>,

        /// Enable verification checks before optimization run
        #[arg(long)]
        verify: bool,

        /// Output directory for generated optimize crate
        #[arg(long)]
        build_dir: Option<PathBuf>,

        /// Output directory for rollout/trace artifacts
        #[arg(long)]
        output_dir: Option<PathBuf>,

        /// Show verbose output
        #[arg(short, long)]
        verbose: bool,

        /// Save agent conversations (including nested agents in pipelines) to JSONL
        #[arg(long)]
        save_convos: bool,
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
        Commands::Codegen {
            file,
            output,
            format,
            strict,
        } => cmd_codegen(&file, &output, format, strict),
        Commands::Build {
            file,
            output,
            bin_name,
            format,
            strict,
            runtime_path,
            embed_config,
            release,
        } => cmd_build(
            &file,
            output.as_deref(),
            bin_name.as_deref(),
            format,
            strict,
            runtime_path.as_deref(),
            embed_config.as_deref(),
            release,
        ),
        Commands::Run {
            file,
            task,
            tool,
            prompt,
            agent,
            pipeline,
            input,
            verbose,
            verify,
            config,
            build_dir,
            save_convos,
        } => cmd_run(
            &file,
            task.as_deref(),
            tool.as_deref(),
            prompt.as_deref(),
            agent.as_deref(),
            pipeline.as_deref(),
            &input,
            verbose,
            verify,
            config.as_deref(),
            build_dir.as_deref(),
            save_convos,
        ),
        Commands::Optimize {
            harness,
            objective,
            config,
            verify,
            build_dir,
            output_dir,
            verbose,
            save_convos,
        } => cmd_optimize(
            &harness,
            &objective,
            config.as_deref(),
            verify,
            build_dir.as_deref(),
            output_dir.as_deref(),
            verbose,
            save_convos,
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

    // Parse
    let program = match parse(&source) {
        Ok(p) => p,
        Err(e) => {
            Report::build(ReportKind::Error, &file_name, e.span.start)
                .with_message("Parse error")
                .with_label(
                    Label::new((&file_name, e.span.start..e.span.end))
                        .with_message(&e.message)
                        .with_color(Color::Red),
                )
                .finish()
                .eprint((&file_name, Source::from(&source)))
                .unwrap();
            return ExitCode::FAILURE;
        }
    };

    if verbose {
        println!("Parsed {} declarations", program.declarations.len());
    }

    // Type check
    let type_env = match check(&program) {
        Ok(env) => env,
        Err(errors) => {
            for error in &errors {
                Report::build(ReportKind::Error, &file_name, error.span.start)
                    .with_message("Type error")
                    .with_label(
                        Label::new((&file_name, error.span.start..error.span.end))
                            .with_message(&error.message)
                            .with_color(Color::Red),
                    )
                    .finish()
                    .eprint((&file_name, Source::from(&source)))
                    .unwrap();
            }
            return ExitCode::FAILURE;
        }
    };

    if verbose {
        println!("Type checking passed");
        println!("  {} types defined", type_env.types.len());
    }

    // Verify
    let verify_result = verify(&program, &type_env);

    let mut has_errors = false;
    for error in &verify_result.errors {
        let kind = match error.severity {
            Severity::Error => {
                has_errors = true;
                ReportKind::Error
            }
            Severity::Warning => ReportKind::Warning,
        };

        Report::build(kind, &file_name, error.span.start)
            .with_message("Verification")
            .with_label(
                Label::new((&file_name, error.span.start..error.span.end))
                    .with_message(&error.message)
                    .with_color(if has_errors {
                        Color::Red
                    } else {
                        Color::Yellow
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
        if !verify_result.deadlock.is_empty() {
            println!(
                "  Deadlock analysis: {} tasks checked",
                verify_result.deadlock.len()
            );
        }
        if !verify_result.bounds.is_empty() {
            println!(
                "  Bounds analysis: {} subgoals checked",
                verify_result.bounds.len()
            );
        }
    }

    println!("{}: OK", file.display());
    ExitCode::SUCCESS
}

fn cmd_compile(file: &PathBuf, output: Option<&Path>, compact: bool) -> ExitCode {
    let source = match fs::read_to_string(file) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error reading file: {}", e);
            return ExitCode::FAILURE;
        }
    };

    let file_name = file.display().to_string();

    // Parse
    let program = match parse(&source) {
        Ok(p) => p,
        Err(e) => {
            Report::build(ReportKind::Error, &file_name, e.span.start)
                .with_message("Parse error")
                .with_label(
                    Label::new((&file_name, e.span.start..e.span.end))
                        .with_message(&e.message)
                        .with_color(Color::Red),
                )
                .finish()
                .eprint((&file_name, Source::from(&source)))
                .unwrap();
            return ExitCode::FAILURE;
        }
    };

    // Type check
    let type_env = match check(&program) {
        Ok(env) => env,
        Err(errors) => {
            for error in &errors {
                Report::build(ReportKind::Error, &file_name, error.span.start)
                    .with_message("Type error")
                    .with_label(
                        Label::new((&file_name, error.span.start..error.span.end))
                            .with_message(&error.message)
                            .with_color(Color::Red),
                    )
                    .finish()
                    .eprint((&file_name, Source::from(&source)))
                    .unwrap();
            }
            return ExitCode::FAILURE;
        }
    };

    // Verify
    let verify_result = verify(&program, &type_env);

    if verify_result.has_errors() {
        for error in &verify_result.errors {
            if error.severity == Severity::Error {
                Report::build(ReportKind::Error, &file_name, error.span.start)
                    .with_message("Verification error")
                    .with_label(
                        Label::new((&file_name, error.span.start..error.span.end))
                            .with_message(&error.message)
                            .with_color(Color::Red),
                    )
                    .finish()
                    .eprint((&file_name, Source::from(&source)))
                    .unwrap();
            }
        }
        return ExitCode::FAILURE;
    }

    // Lower to IR
    let lowerer = Lowerer::new().with_source_file(file_name.clone());
    let ir = match lowerer.lower(&program, &type_env) {
        Ok(ir) => ir,
        Err(e) => {
            Report::build(ReportKind::Error, &file_name, e.span.start)
                .with_message("Lowering error")
                .with_label(
                    Label::new((&file_name, e.span.start..e.span.end))
                        .with_message(&e.message)
                        .with_color(Color::Red),
                )
                .finish()
                .eprint((&file_name, Source::from(&source)))
                .unwrap();
            return ExitCode::FAILURE;
        }
    };

    // Serialize to JSON
    let json = if compact {
        scaffold_ir::to_json_compact(&ir)
    } else {
        to_json(&ir)
    };

    let json = match json {
        Ok(j) => j,
        Err(e) => {
            eprintln!("Error serializing IR: {}", e);
            return ExitCode::FAILURE;
        }
    };

    // Write output
    match output {
        Some(path) => {
            if let Err(e) = fs::write(path, &json) {
                eprintln!("Error writing output file: {}", e);
                return ExitCode::FAILURE;
            }
            eprintln!("Compiled to {}", path.display());
        }
        None => {
            println!("{}", json);
        }
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
                    scaffold_syntax::Declaration::Type(t) => {
                        println!("  type {}", t.name.node);
                    }
                    scaffold_syntax::Declaration::Artifact(a) => {
                        println!("  artifact {}", a.name.node);
                    }
                    scaffold_syntax::Declaration::ExternCrate(e) => {
                        println!("  extern crate {} = \"{}\"", e.name.node, e.version);
                    }
                    scaffold_syntax::Declaration::Foreign(f) => {
                        println!("  foreign {} {}", f.language.node, f.name.node);
                    }
                    scaffold_syntax::Declaration::Tool(t) => {
                        println!("  tool {}", t.name.node);
                    }
                    scaffold_syntax::Declaration::Prompt(p) => {
                        println!("  prompt {}", p.name.node);
                    }
                    scaffold_syntax::Declaration::Agent(a) => {
                        println!("  agent {}", a.name.node);
                    }
                    scaffold_syntax::Declaration::Pipeline(p) => {
                        println!("  pipeline {}", p.name.node);
                    }
                    scaffold_syntax::Declaration::Task(t) => {
                        println!("  task {}", t.name.node);
                    }
                    scaffold_syntax::Declaration::Harness(h) => {
                        println!("  harness {}", h.name.node);
                    }
                    scaffold_syntax::Declaration::Objective(o) => {
                        println!("  objective {}", o.name.node);
                    }
                }
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            Report::build(ReportKind::Error, &file_name, e.span.start)
                .with_message("Parse error")
                .with_label(
                    Label::new((&file_name, e.span.start..e.span.end))
                        .with_message(&e.message)
                        .with_color(Color::Red),
                )
                .finish()
                .eprint((&file_name, Source::from(&source)))
                .unwrap();
            ExitCode::FAILURE
        }
    }
}

/// Try to auto-detect scaffold-runtime path from the CLI executable location
fn detect_runtime_path(hint: Option<&Path>) -> Option<String> {
    // If env var is already set, use that
    if std::env::var("SCAFFOLD_RUNTIME_PATH").is_ok()
        || std::env::var("SCAFFOLD_RUNTIME_VERSION").is_ok()
    {
        return None;
    }

    let mut roots: Vec<PathBuf> = Vec::new();

    if let Some(hint_path) = hint {
        let base = if hint_path.is_file() {
            hint_path.parent().unwrap_or(hint_path).to_path_buf()
        } else {
            hint_path.to_path_buf()
        };
        roots.push(base);
    }

    if let Ok(cwd) = std::env::current_dir() {
        roots.push(cwd);
    }

    for root in roots {
        let mut path = root;
        for _ in 0..5 {
            let runtime_path = path.join("crates/scaffold-runtime");
            if runtime_path.join("Cargo.toml").exists() {
                let abs = runtime_path.canonicalize().unwrap_or_else(|_| runtime_path);
                return Some(abs.to_string_lossy().to_string());
            }
            path = match path.parent() {
                Some(p) => p.to_path_buf(),
                None => break,
            };
        }
    }

    // Try to find scaffold-runtime relative to the current executable
    if let Ok(exe_path) = std::env::current_exe() {
        // Go up from target/debug or target/release to find crates/scaffold-runtime
        let mut path = exe_path.clone();
        for _ in 0..5 {
            path = match path.parent() {
                Some(p) => p.to_path_buf(),
                None => break,
            };
            let runtime_path = path.join("crates/scaffold-runtime");
            if runtime_path.join("Cargo.toml").exists() {
                let abs = runtime_path.canonicalize().unwrap_or_else(|_| runtime_path);
                return Some(abs.to_string_lossy().to_string());
            }
        }
    }

    None
}

fn cmd_codegen(file: &PathBuf, output: &PathBuf, format: bool, strict: bool) -> ExitCode {
    // Auto-detect scaffold-runtime path
    if let Some(runtime_path) = detect_runtime_path(Some(file.as_path())) {
        std::env::set_var("SCAFFOLD_RUNTIME_PATH", runtime_path);
    }
    let source = match fs::read_to_string(file) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error reading file: {}", e);
            return ExitCode::FAILURE;
        }
    };

    let file_name = file.display().to_string();

    // Parse
    let program = match parse(&source) {
        Ok(p) => p,
        Err(e) => {
            Report::build(ReportKind::Error, &file_name, e.span.start)
                .with_message("Parse error")
                .with_label(
                    Label::new((&file_name, e.span.start..e.span.end))
                        .with_message(&e.message)
                        .with_color(Color::Red),
                )
                .finish()
                .eprint((&file_name, Source::from(&source)))
                .unwrap();
            return ExitCode::FAILURE;
        }
    };

    // Type check
    let type_env = match check(&program) {
        Ok(env) => env,
        Err(errors) => {
            for error in &errors {
                Report::build(ReportKind::Error, &file_name, error.span.start)
                    .with_message("Type error")
                    .with_label(
                        Label::new((&file_name, error.span.start..error.span.end))
                            .with_message(&error.message)
                            .with_color(Color::Red),
                    )
                    .finish()
                    .eprint((&file_name, Source::from(&source)))
                    .unwrap();
            }
            return ExitCode::FAILURE;
        }
    };

    // Verify
    let verify_result = verify(&program, &type_env);

    if verify_result.has_errors() {
        for error in &verify_result.errors {
            if error.severity == Severity::Error {
                Report::build(ReportKind::Error, &file_name, error.span.start)
                    .with_message("Verification error")
                    .with_label(
                        Label::new((&file_name, error.span.start..error.span.end))
                            .with_message(&error.message)
                            .with_color(Color::Red),
                    )
                    .finish()
                    .eprint((&file_name, Source::from(&source)))
                    .unwrap();
            }
        }
        return ExitCode::FAILURE;
    }

    // Lower to IR
    let lowerer = Lowerer::new().with_source_file(file_name.clone());
    let ir = match lowerer.lower(&program, &type_env) {
        Ok(ir) => ir,
        Err(e) => {
            Report::build(ReportKind::Error, &file_name, e.span.start)
                .with_message("Lowering error")
                .with_label(
                    Label::new((&file_name, e.span.start..e.span.end))
                        .with_message(&e.message)
                        .with_color(Color::Red),
                )
                .finish()
                .eprint((&file_name, Source::from(&source)))
                .unwrap();
            return ExitCode::FAILURE;
        }
    };

    // Generate code
    let generator = CodeGenerator::new()
        .with_formatting(format)
        .with_strict(strict);
    let generated = match generator.generate(&ir) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Code generation error: {}", e);
            return ExitCode::FAILURE;
        }
    };

    // Write output
    if let Err(e) = generated.write_to_dir(output) {
        eprintln!("Error writing output: {}", e);
        return ExitCode::FAILURE;
    }

    eprintln!(
        "Generated {} files to {}",
        generated.files.len(),
        output.display()
    );
    for path in generated.files.keys() {
        eprintln!("  {}", path);
    }

    ExitCode::SUCCESS
}

fn cmd_build(
    file: &PathBuf,
    output: Option<&Path>,
    bin_name: Option<&str>,
    format: bool,
    strict: bool,
    runtime_path: Option<&str>,
    embed_config: Option<&Path>,
    release: bool,
) -> ExitCode {
    let source = match fs::read_to_string(file) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error reading file: {}", e);
            return ExitCode::FAILURE;
        }
    };

    let file_name = file.display().to_string();

    // Parse
    let program = match parse(&source) {
        Ok(p) => p,
        Err(e) => {
            Report::build(ReportKind::Error, &file_name, e.span.start)
                .with_message("Parse error")
                .with_label(
                    Label::new((&file_name, e.span.start..e.span.end))
                        .with_message(&e.message)
                        .with_color(Color::Red),
                )
                .finish()
                .eprint((&file_name, Source::from(&source)))
                .unwrap();
            return ExitCode::FAILURE;
        }
    };

    // Type check
    let type_env = match check(&program) {
        Ok(env) => env,
        Err(errors) => {
            for error in &errors {
                Report::build(ReportKind::Error, &file_name, error.span.start)
                    .with_message("Type error")
                    .with_label(
                        Label::new((&file_name, error.span.start..error.span.end))
                            .with_message(&error.message)
                            .with_color(Color::Red),
                    )
                    .finish()
                    .eprint((&file_name, Source::from(&source)))
                    .unwrap();
            }
            return ExitCode::FAILURE;
        }
    };

    // Lower to IR
    let lowerer = Lowerer::new().with_source_file(file_name.clone());
    let ir = match lowerer.lower(&program, &type_env) {
        Ok(ir) => ir,
        Err(e) => {
            Report::build(ReportKind::Error, &file_name, e.span.start)
                .with_message("Lowering error")
                .with_label(
                    Label::new((&file_name, e.span.start..e.span.end))
                        .with_message(&e.message)
                        .with_color(Color::Red),
                )
                .finish()
                .eprint((&file_name, Source::from(&source)))
                .unwrap();
            return ExitCode::FAILURE;
        }
    };

    // Prepare output dir
    let out_dir = output
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("generated"));
    if let Err(e) = fs::create_dir_all(&out_dir) {
        eprintln!("Error creating output dir {}: {}", out_dir.display(), e);
        return ExitCode::FAILURE;
    }

    // Configure runtime path env for codegen
    if let Some(path) = runtime_path {
        std::env::set_var("SCAFFOLD_RUNTIME_PATH", path);
    } else if let Some(detected) = detect_runtime_path(Some(file.as_path())) {
        std::env::set_var("SCAFFOLD_RUNTIME_PATH", detected);
    }

    // Optional embedded config for the generated binary
    let embedded_config = if let Some(path) = embed_config {
        match fs::read_to_string(path) {
            Ok(contents) => Some(contents),
            Err(e) => {
                eprintln!("Error reading embed config {}: {}", path.display(), e);
                return ExitCode::FAILURE;
            }
        }
    } else {
        None
    };

    // Generate code
    let generator = CodeGenerator::new()
        .with_formatting(format)
        .with_strict(strict)
        .with_embedded_config(embedded_config);
    let generated = match generator.generate(&ir) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Code generation error: {}", e);
            return ExitCode::FAILURE;
        }
    };

    if let Err(e) = generated.write_to_dir(&out_dir) {
        eprintln!("Error writing output: {}", e);
        return ExitCode::FAILURE;
    }

    // Optionally rename binary in Cargo.toml by modifying package/bin name
    if let Some(name) = bin_name {
        let manifest_path = out_dir.join("Cargo.toml");
        if let Ok(mut cargo_toml) = fs::read_to_string(&manifest_path) {
            // Replace package name and bin name heuristically
            // If no pipeline/tool present, default name is scaffold_generated
            if cargo_toml.contains("name = \"scaffold_generated\"") {
                cargo_toml = cargo_toml.replace(
                    "name = \"scaffold_generated\"",
                    &format!("name = \"{}\"", name),
                );
            }
            if cargo_toml.contains("[[bin]]\nname = \"scaffold_generated\"") {
                cargo_toml = cargo_toml.replace(
                    "[[bin]]\nname = \"scaffold_generated\"",
                    &format!("[[bin]]\nname = \"{}\"", name),
                );
            }
            if let Err(e) = fs::write(&manifest_path, cargo_toml) {
                eprintln!("Warning: failed to set bin name: {}", e);
            }
        }
    }

    // Run cargo build in the generated dir
    let mut cmd = std::process::Command::new("cargo");
    cmd.arg("build");
    if release {
        cmd.arg("--release");
    }
    cmd.current_dir(&out_dir);
    match cmd.status() {
        Ok(status) if status.success() => {
            // Print path to binary
            let pkg_name = bin_name.map(|s| s.to_string()).unwrap_or_else(|| {
                ir.pipelines
                    .first()
                    .map(|p| p.name.clone())
                    .or_else(|| ir.tools.first().map(|t| t.name.clone()))
                    .unwrap_or_else(|| "scaffold_generated".to_string())
            });
            let bin_dir = if release { "release" } else { "debug" };
            let bin_path = out_dir.join("target").join(bin_dir).join(&pkg_name);
            println!("Built binary: {}", bin_path.display());
            ExitCode::SUCCESS
        }
        Ok(status) => {
            eprintln!("cargo build failed with status: {}", status);
            ExitCode::FAILURE
        }
        Err(e) => {
            eprintln!("Failed to run cargo build: {}", e);
            ExitCode::FAILURE
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ObjectiveTarget {
    kind: String,
    name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ObjectiveCase {
    #[serde(default)]
    id: Option<String>,
    input: serde_json::Value,
    #[serde(default)]
    expected: Option<serde_json::Value>,
    #[serde(default)]
    expected_contains: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ObjectiveSpec {
    target: ObjectiveTarget,
    #[serde(default)]
    dataset: Vec<ObjectiveCase>,
    #[serde(default)]
    dataset_file: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
struct ObjectiveSnapshot {
    source_path: String,
    loaded_at_ms: u64,
    target: ObjectiveTarget,
    dataset: Vec<ObjectiveCase>,
}

#[derive(Debug, Serialize)]
struct RolloutRecord {
    run_id: String,
    case_id: String,
    target_kind: String,
    target_name: String,
    input: serde_json::Value,
    expected: Option<serde_json::Value>,
    expected_contains: Option<String>,
    output: Option<serde_json::Value>,
    output_text: String,
    stderr: String,
    exit_code: Option<i32>,
    duration_ms: u64,
    passed: Option<bool>,
    score: Option<f64>,
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct RolloutTrace {
    case_id: String,
    started_at_ms: u64,
    duration_ms: u64,
    command: Vec<String>,
    exit_code: Option<i32>,
    stdout: String,
    stderr: String,
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct OptimizeSummary {
    run_id: String,
    harness: String,
    objective: String,
    target_kind: String,
    target_name: String,
    total_cases: usize,
    scored_cases: usize,
    passed_cases: usize,
    failed_cases: usize,
    errored_cases: usize,
    average_score: Option<f64>,
    artifacts_dir: String,
    rollouts_file: String,
    traces_dir: String,
    agent_convos_file: Option<String>,
    binary_path: String,
    build_dir: String,
}

fn parse_run_input(input_str: &str) -> Result<String, ExitCode> {
    let json_input = if input_str.starts_with('@') {
        let input_path = &input_str[1..];
        match fs::read_to_string(input_path) {
            Ok(content) => content,
            Err(e) => {
                eprintln!("Error reading input file {}: {}", input_path, e);
                return Err(ExitCode::FAILURE);
            }
        }
    } else {
        input_str.to_string()
    };

    if let Err(e) = serde_json::from_str::<serde_json::Value>(&json_input) {
        eprintln!("Error parsing input JSON: {}", e);
        return Err(ExitCode::FAILURE);
    }

    Ok(json_input)
}

fn current_timestamp_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn sanitize_id(raw: &str) -> String {
    raw.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect::<String>()
}

fn parse_dataset_file(path: &Path) -> Result<Vec<ObjectiveCase>, String> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    let content = fs::read_to_string(path)
        .map_err(|e| format!("failed to read dataset file {}: {}", path.display(), e))?;

    if ext == "jsonl" {
        let mut rows = Vec::new();
        for (idx, line_result) in BufReader::new(content.as_bytes()).lines().enumerate() {
            let line =
                line_result.map_err(|e| format!("failed reading jsonl line {}: {}", idx + 1, e))?;
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            let mut row: ObjectiveCase = serde_json::from_str(trimmed).map_err(|e| {
                format!(
                    "invalid jsonl line {} in {}: {}",
                    idx + 1,
                    path.display(),
                    e
                )
            })?;
            if row.id.is_none() {
                row.id = Some(format!("case-{:04}", rows.len() + 1));
            }
            rows.push(row);
        }
        return Ok(rows);
    }

    let value: serde_json::Value = serde_json::from_str(&content)
        .map_err(|e| format!("invalid json in dataset file {}: {}", path.display(), e))?;

    if let Some(arr) = value.as_array() {
        let mut rows = Vec::with_capacity(arr.len());
        for (idx, item) in arr.iter().enumerate() {
            let mut row: ObjectiveCase = serde_json::from_value(item.clone()).map_err(|e| {
                format!(
                    "invalid dataset row {} in {}: {}",
                    idx + 1,
                    path.display(),
                    e
                )
            })?;
            if row.id.is_none() {
                row.id = Some(format!("case-{:04}", idx + 1));
            }
            rows.push(row);
        }
        return Ok(rows);
    }

    if let Some(arr) = value.get("dataset").and_then(|v| v.as_array()) {
        let mut rows = Vec::with_capacity(arr.len());
        for (idx, item) in arr.iter().enumerate() {
            let mut row: ObjectiveCase = serde_json::from_value(item.clone()).map_err(|e| {
                format!(
                    "invalid dataset row {} in {}: {}",
                    idx + 1,
                    path.display(),
                    e
                )
            })?;
            if row.id.is_none() {
                row.id = Some(format!("case-{:04}", idx + 1));
            }
            rows.push(row);
        }
        return Ok(rows);
    }

    Err(format!(
        "dataset file {} must be JSON array, object with 'dataset', or JSONL",
        path.display()
    ))
}

fn load_objective(objective_path: &Path) -> Result<ObjectiveSpec, String> {
    let content = fs::read_to_string(objective_path).map_err(|e| {
        format!(
            "failed to read objective {}: {}",
            objective_path.display(),
            e
        )
    })?;
    let mut spec: ObjectiveSpec = serde_json::from_str(&content)
        .map_err(|e| format!("invalid objective JSON {}: {}", objective_path.display(), e))?;

    if let Some(dataset_file) = spec.dataset_file.clone() {
        let resolved = if dataset_file.is_absolute() {
            dataset_file
        } else {
            objective_path
                .parent()
                .unwrap_or(Path::new("."))
                .join(dataset_file)
        };
        spec.dataset = parse_dataset_file(&resolved)?;
    }

    if spec.dataset.is_empty() {
        return Err("objective dataset is empty; provide 'dataset' or 'dataset_file'".to_string());
    }

    let kind = spec.target.kind.trim().to_ascii_lowercase();
    if !matches!(kind.as_str(), "tool" | "prompt" | "agent" | "pipeline") {
        return Err(format!(
            "objective target.kind must be one of tool|prompt|agent|pipeline (got '{}')",
            spec.target.kind
        ));
    }
    spec.target.kind = kind;

    for (idx, case) in spec.dataset.iter_mut().enumerate() {
        if case.id.is_none() {
            case.id = Some(format!("case-{:04}", idx + 1));
        }
    }

    Ok(spec)
}

fn parse_typecheck_lower(file: &Path, verify_enabled: bool) -> Result<ScaffoldIR, ExitCode> {
    let source = match fs::read_to_string(file) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error reading file {}: {}", file.display(), e);
            return Err(ExitCode::FAILURE);
        }
    };
    let file_name = file.display().to_string();

    let program = match parse(&source) {
        Ok(p) => p,
        Err(e) => {
            Report::build(ReportKind::Error, &file_name, e.span.start)
                .with_message("Parse error")
                .with_label(
                    Label::new((&file_name, e.span.start..e.span.end))
                        .with_message(&e.message)
                        .with_color(Color::Red),
                )
                .finish()
                .eprint((&file_name, Source::from(&source)))
                .unwrap();
            return Err(ExitCode::FAILURE);
        }
    };

    let type_env = match check(&program) {
        Ok(env) => env,
        Err(errors) => {
            for error in &errors {
                Report::build(ReportKind::Error, &file_name, error.span.start)
                    .with_message("Type error")
                    .with_label(
                        Label::new((&file_name, error.span.start..error.span.end))
                            .with_message(&error.message)
                            .with_color(Color::Red),
                    )
                    .finish()
                    .eprint((&file_name, Source::from(&source)))
                    .unwrap();
            }
            return Err(ExitCode::FAILURE);
        }
    };

    if verify_enabled {
        let verify_result = verify(&program, &type_env);
        let mut has_errors = false;
        for error in &verify_result.errors {
            let kind = match error.severity {
                Severity::Error => {
                    has_errors = true;
                    ReportKind::Error
                }
                Severity::Warning => ReportKind::Warning,
            };
            Report::build(kind, &file_name, error.span.start)
                .with_message("Verification")
                .with_label(
                    Label::new((&file_name, error.span.start..error.span.end))
                        .with_message(&error.message)
                        .with_color(if has_errors {
                            Color::Red
                        } else {
                            Color::Yellow
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
    let ir = match lowerer.lower(&program, &type_env) {
        Ok(ir) => ir,
        Err(e) => {
            Report::build(ReportKind::Error, &file_name, e.span.start)
                .with_message("Lowering error")
                .with_label(
                    Label::new((&file_name, e.span.start..e.span.end))
                        .with_message(&e.message)
                        .with_color(Color::Red),
                )
                .finish()
                .eprint((&file_name, Source::from(&source)))
                .unwrap();
            return Err(ExitCode::FAILURE);
        }
    };

    Ok(ir)
}

fn default_optimize_build_dir(file: &Path) -> PathBuf {
    let stem = file
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("scaffold")
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect::<String>();

    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    file.to_string_lossy().hash(&mut hasher);
    let hash = hasher.finish();

    PathBuf::from("target")
        .join("scaffold-opt")
        .join("build")
        .join(format!("{}-{:x}", stem, hash))
}

fn default_optimize_artifacts_dir() -> PathBuf {
    PathBuf::from("target")
        .join("scaffold-opt")
        .join("runs")
        .join(format!("run-{}", current_timestamp_ms()))
}

fn binary_name_from_ir(ir: &ScaffoldIR) -> String {
    ir.pipelines
        .first()
        .map(|p| p.name.clone())
        .or_else(|| ir.tools.first().map(|t| t.name.clone()))
        .unwrap_or_else(|| "scaffold_generated".to_string())
}

fn ir_has_target(ir: &ScaffoldIR, kind: &str, name: &str) -> bool {
    match kind {
        "tool" => ir.tools.iter().any(|t| t.name == name),
        "prompt" => ir.prompts.iter().any(|p| p.name == name),
        "agent" => ir.agents.iter().any(|a| a.name == name),
        "pipeline" => ir.pipelines.iter().any(|p| p.name == name),
        _ => false,
    }
}

fn evaluate_case(
    case: &ObjectiveCase,
    output: Option<&serde_json::Value>,
    output_text: &str,
) -> (Option<bool>, Option<f64>, Option<String>) {
    let mut checks = 0usize;
    let mut passed_checks = 0usize;
    let mut failures: Vec<String> = Vec::new();

    if let Some(expected) = &case.expected {
        checks += 1;
        match output {
            Some(actual) if actual == expected => passed_checks += 1,
            Some(actual) => failures.push(format!("expected JSON {}, got {}", expected, actual)),
            None => failures
                .push("expected JSON output but command did not emit valid JSON".to_string()),
        }
    }

    if let Some(substr) = &case.expected_contains {
        checks += 1;
        if output_text.contains(substr) {
            passed_checks += 1;
        } else {
            failures.push(format!("expected output to contain '{}'", substr));
        }
    }

    if checks == 0 {
        return (None, None, None);
    }

    let passed = passed_checks == checks;
    let score = passed_checks as f64 / checks as f64;
    let note = if passed {
        None
    } else {
        Some(failures.join("; "))
    };
    (Some(passed), Some(score), note)
}

fn default_run_build_dir(file: &Path) -> PathBuf {
    let stem = file
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("scaffold")
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect::<String>();

    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    file.to_string_lossy().hash(&mut hasher);
    let hash = hasher.finish();

    PathBuf::from("target")
        .join("scaffold-run")
        .join(format!("{}-{:x}", stem, hash))
}

fn cmd_run(
    file: &PathBuf,
    _task: Option<&str>, // Deprecated - tasks removed
    tool: Option<&str>,
    prompt: Option<&str>,
    agent: Option<&str>,
    pipeline: Option<&str>,
    input_str: &str,
    verbose: bool,
    verify_enabled: bool,
    config: Option<&Path>,
    build_dir: Option<&Path>,
    save_convos: bool,
) -> ExitCode {
    let input_json = match parse_run_input(input_str) {
        Ok(v) => v,
        Err(code) => return code,
    };

    let source = match fs::read_to_string(file) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error reading file: {}", e);
            return ExitCode::FAILURE;
        }
    };
    let file_name = file.display().to_string();

    let program = match parse(&source) {
        Ok(p) => p,
        Err(e) => {
            Report::build(ReportKind::Error, &file_name, e.span.start)
                .with_message("Parse error")
                .with_label(
                    Label::new((&file_name, e.span.start..e.span.end))
                        .with_message(&e.message)
                        .with_color(Color::Red),
                )
                .finish()
                .eprint((&file_name, Source::from(&source)))
                .unwrap();
            return ExitCode::FAILURE;
        }
    };

    let mut tools: Vec<String> = Vec::new();
    let mut prompts: Vec<String> = Vec::new();
    let mut agents: Vec<String> = Vec::new();
    let mut pipelines: Vec<String> = Vec::new();

    for decl in &program.declarations {
        match decl {
            scaffold_syntax::Declaration::Tool(t) => tools.push(t.name.node.clone()),
            scaffold_syntax::Declaration::Prompt(p) => prompts.push(p.name.node.clone()),
            scaffold_syntax::Declaration::Agent(a) => agents.push(a.name.node.clone()),
            scaffold_syntax::Declaration::Pipeline(p) => pipelines.push(p.name.node.clone()),
            _ => {}
        }
    }

    let mut chosen: Vec<(&str, String)> = Vec::new();
    if let Some(name) = tool {
        chosen.push(("tool", name.to_string()));
    }
    if let Some(name) = prompt {
        chosen.push(("prompt", name.to_string()));
    }
    if let Some(name) = agent {
        chosen.push(("agent", name.to_string()));
    }
    if let Some(name) = pipeline {
        chosen.push(("pipeline", name.to_string()));
    }

    if chosen.len() > 1 {
        eprintln!("Specify only one of --tool, --prompt, --agent, or --pipeline.");
        return ExitCode::FAILURE;
    }

    let (target_kind, target_name) = if let Some((k, n)) = chosen.into_iter().next() {
        (k.to_string(), n)
    } else {
        let total = tools.len() + prompts.len() + agents.len() + pipelines.len();
        if total == 0 {
            eprintln!(
                "No tools, prompts, agents, or pipelines found in {}",
                file.display()
            );
            return ExitCode::FAILURE;
        }

        if tools.len() == 1 && prompts.is_empty() && agents.is_empty() && pipelines.is_empty() {
            ("tool".to_string(), tools[0].clone())
        } else if prompts.len() == 1
            && tools.is_empty()
            && agents.is_empty()
            && pipelines.is_empty()
        {
            ("prompt".to_string(), prompts[0].clone())
        } else if agents.len() == 1
            && tools.is_empty()
            && prompts.is_empty()
            && pipelines.is_empty()
        {
            ("agent".to_string(), agents[0].clone())
        } else if pipelines.len() == 1
            && tools.is_empty()
            && prompts.is_empty()
            && agents.is_empty()
        {
            ("pipeline".to_string(), pipelines[0].clone())
        } else {
            eprintln!("Multiple items available. Please specify one:");
            if !tools.is_empty() {
                eprintln!("  Tools: {:?}", tools);
            }
            if !prompts.is_empty() {
                eprintln!("  Prompts: {:?}", prompts);
            }
            if !agents.is_empty() {
                eprintln!("  Agents: {:?}", agents);
            }
            if !pipelines.is_empty() {
                eprintln!("  Pipelines: {:?}", pipelines);
            }
            eprintln!("\nUsage: scaffold run {} --tool <TOOL> | --prompt <PROMPT> | --agent <AGENT> | --pipeline <PIPELINE>", file.display());
            return ExitCode::FAILURE;
        }
    };

    let target_exists = match target_kind.as_str() {
        "tool" => tools.iter().any(|n| n == &target_name),
        "prompt" => prompts.iter().any(|n| n == &target_name),
        "agent" => agents.iter().any(|n| n == &target_name),
        "pipeline" => pipelines.iter().any(|n| n == &target_name),
        _ => false,
    };
    if !target_exists {
        eprintln!(
            "{} '{}' not found in {}",
            target_kind,
            target_name,
            file.display()
        );
        return ExitCode::FAILURE;
    }

    let type_env = match check(&program) {
        Ok(env) => env,
        Err(errors) => {
            for error in &errors {
                Report::build(ReportKind::Error, &file_name, error.span.start)
                    .with_message("Type error")
                    .with_label(
                        Label::new((&file_name, error.span.start..error.span.end))
                            .with_message(&error.message)
                            .with_color(Color::Red),
                    )
                    .finish()
                    .eprint((&file_name, Source::from(&source)))
                    .unwrap();
            }
            return ExitCode::FAILURE;
        }
    };

    if verify_enabled {
        let verify_result = verify(&program, &type_env);
        let mut has_errors = false;
        for error in &verify_result.errors {
            let kind = match error.severity {
                Severity::Error => {
                    has_errors = true;
                    ReportKind::Error
                }
                Severity::Warning => ReportKind::Warning,
            };
            Report::build(kind, &file_name, error.span.start)
                .with_message("Verification")
                .with_label(
                    Label::new((&file_name, error.span.start..error.span.end))
                        .with_message(&error.message)
                        .with_color(if has_errors {
                            Color::Red
                        } else {
                            Color::Yellow
                        }),
                )
                .finish()
                .eprint((&file_name, Source::from(&source)))
                .unwrap();
        }
        if has_errors {
            return ExitCode::FAILURE;
        }
    }

    let lowerer = Lowerer::new().with_source_file(file_name.clone());
    let ir = match lowerer.lower(&program, &type_env) {
        Ok(ir) => ir,
        Err(e) => {
            Report::build(ReportKind::Error, &file_name, e.span.start)
                .with_message("Lowering error")
                .with_label(
                    Label::new((&file_name, e.span.start..e.span.end))
                        .with_message(&e.message)
                        .with_color(Color::Red),
                )
                .finish()
                .eprint((&file_name, Source::from(&source)))
                .unwrap();
            return ExitCode::FAILURE;
        }
    };

    let out_dir = build_dir
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| default_run_build_dir(file));
    if let Err(e) = fs::create_dir_all(&out_dir) {
        eprintln!("Error creating output dir {}: {}", out_dir.display(), e);
        return ExitCode::FAILURE;
    }
    let convos_path = if save_convos {
        Some(out_dir.join("agent_convos.jsonl"))
    } else {
        None
    };
    if let Some(path) = &convos_path {
        if let Some(parent) = path.parent() {
            if let Err(e) = fs::create_dir_all(parent) {
                eprintln!(
                    "Error creating conversations dir {}: {}",
                    parent.display(),
                    e
                );
                return ExitCode::FAILURE;
            }
        }
        // Start each run with a fresh conversation log.
        let _ = fs::remove_file(path);
    }

    if let Some(detected) = detect_runtime_path(Some(file.as_path())) {
        std::env::set_var("SCAFFOLD_RUNTIME_PATH", detected);
    }

    if verbose {
        eprintln!("Generating crate in {}", out_dir.display());
    }
    let generator = CodeGenerator::new()
        .with_formatting(true)
        .with_strict(false);
    let generated = match generator.generate(&ir) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Code generation error: {}", e);
            return ExitCode::FAILURE;
        }
    };
    if let Err(e) = generated.write_to_dir(&out_dir) {
        eprintln!("Error writing output: {}", e);
        return ExitCode::FAILURE;
    }

    let pkg_name = ir
        .pipelines
        .first()
        .map(|p| p.name.clone())
        .or_else(|| ir.tools.first().map(|t| t.name.clone()))
        .unwrap_or_else(|| "scaffold_generated".to_string());

    if verbose {
        eprintln!("Building generated binary...");
    }
    let mut build_cmd = std::process::Command::new("cargo");
    build_cmd.arg("build").current_dir(&out_dir);
    match build_cmd.status() {
        Ok(status) if status.success() => {}
        Ok(status) => {
            eprintln!("cargo build failed with status: {}", status);
            return ExitCode::FAILURE;
        }
        Err(e) => {
            eprintln!("Failed to run cargo build: {}", e);
            return ExitCode::FAILURE;
        }
    }

    let binary_path = out_dir.join("target").join("debug").join(pkg_name);
    let run_id = format!("run-{}", current_timestamp_ms());
    if verbose {
        eprintln!("Running {} {}", target_kind, target_name);
        if let Some(path) = &convos_path {
            eprintln!("Saving agent conversations to {}", path.display());
        }
    }
    let mut run_cmd = std::process::Command::new(&binary_path);
    if let Some(config_path) = config {
        run_cmd.arg("--config").arg(config_path);
    }
    if verbose {
        run_cmd.env("SCAFFOLD_LOG", "1");
    }
    if let Some(path) = &convos_path {
        run_cmd.env("SCAFFOLD_AGENT_CONVOS_PATH", path);
        run_cmd.env("SCAFFOLD_RUN_ID", &run_id);
        run_cmd.env("SCAFFOLD_CASE_ID", "single-run");
        run_cmd.env("SCAFFOLD_TARGET_KIND", &target_kind);
        run_cmd.env("SCAFFOLD_TARGET_NAME", &target_name);
    }
    run_cmd
        .arg(&target_kind)
        .arg(&target_name)
        .arg("--input")
        .arg(input_json);

    match run_cmd.status() {
        Ok(status) if status.success() => ExitCode::SUCCESS,
        Ok(status) => {
            eprintln!("Execution failed with status: {}", status);
            ExitCode::FAILURE
        }
        Err(e) => {
            eprintln!("Failed to run compiled binary: {}", e);
            ExitCode::FAILURE
        }
    }
}

fn cmd_optimize(
    harness: &PathBuf,
    objective_path: &PathBuf,
    config: Option<&Path>,
    verify_enabled: bool,
    build_dir: Option<&Path>,
    output_dir: Option<&Path>,
    verbose: bool,
    save_convos: bool,
) -> ExitCode {
    let objective = match load_objective(objective_path) {
        Ok(spec) => spec,
        Err(msg) => {
            eprintln!("Objective error: {}", msg);
            return ExitCode::FAILURE;
        }
    };

    let artifacts_dir = output_dir
        .map(|p| p.to_path_buf())
        .unwrap_or_else(default_optimize_artifacts_dir);
    if let Err(e) = fs::create_dir_all(&artifacts_dir) {
        eprintln!(
            "Failed to create optimize artifacts dir {}: {}",
            artifacts_dir.display(),
            e
        );
        return ExitCode::FAILURE;
    }
    let traces_dir = artifacts_dir.join("traces");
    if let Err(e) = fs::create_dir_all(&traces_dir) {
        eprintln!(
            "Failed to create optimize traces dir {}: {}",
            traces_dir.display(),
            e
        );
        return ExitCode::FAILURE;
    }
    let convos_path = if save_convos {
        Some(artifacts_dir.join("agent_convos.jsonl"))
    } else {
        None
    };
    if let Some(path) = &convos_path {
        if let Some(parent) = path.parent() {
            if let Err(e) = fs::create_dir_all(parent) {
                eprintln!(
                    "Failed to create conversations dir {}: {}",
                    parent.display(),
                    e
                );
                return ExitCode::FAILURE;
            }
        }
        // Start each optimize run with a fresh conversation log.
        let _ = fs::remove_file(path);
    }

    let run_id = artifacts_dir
        .file_name()
        .and_then(|s| s.to_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("run-{}", current_timestamp_ms()));
    if verbose {
        if let Some(path) = &convos_path {
            eprintln!("Saving agent conversations to {}", path.display());
        }
    }

    // Snapshot inputs so objective/data are immutable for this run.
    let _ = fs::copy(harness, artifacts_dir.join("harness.snapshot.scaffold"));
    let _ = fs::copy(
        objective_path,
        artifacts_dir.join("objective.snapshot.json"),
    );
    let objective_snapshot = ObjectiveSnapshot {
        source_path: objective_path.display().to_string(),
        loaded_at_ms: current_timestamp_ms(),
        target: objective.target.clone(),
        dataset: objective.dataset.clone(),
    };
    if let Ok(snapshot_json) = serde_json::to_string_pretty(&objective_snapshot) {
        let _ = fs::write(artifacts_dir.join("objective.lock.json"), snapshot_json);
    }

    let ir = match parse_typecheck_lower(harness, verify_enabled) {
        Ok(ir) => ir,
        Err(code) => return code,
    };

    if !ir_has_target(&ir, &objective.target.kind, &objective.target.name) {
        eprintln!(
            "Objective target {} '{}' not found in {}",
            objective.target.kind,
            objective.target.name,
            harness.display()
        );
        return ExitCode::FAILURE;
    }

    let out_dir = build_dir
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| default_optimize_build_dir(harness));
    if let Err(e) = fs::create_dir_all(&out_dir) {
        eprintln!(
            "Failed to create optimize build dir {}: {}",
            out_dir.display(),
            e
        );
        return ExitCode::FAILURE;
    }

    if let Some(detected) = detect_runtime_path(Some(harness.as_path())) {
        std::env::set_var("SCAFFOLD_RUNTIME_PATH", detected);
    }

    if verbose {
        eprintln!("Generating optimize crate in {}", out_dir.display());
    }
    let generator = CodeGenerator::new()
        .with_formatting(true)
        .with_strict(false);
    let generated = match generator.generate(&ir) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Code generation error: {}", e);
            return ExitCode::FAILURE;
        }
    };
    if let Err(e) = generated.write_to_dir(&out_dir) {
        eprintln!("Error writing generated optimize crate: {}", e);
        return ExitCode::FAILURE;
    }

    if verbose {
        eprintln!("Building optimize crate...");
    }
    let mut build_cmd = std::process::Command::new("cargo");
    build_cmd.arg("build").current_dir(&out_dir);
    match build_cmd.status() {
        Ok(status) if status.success() => {}
        Ok(status) => {
            eprintln!("cargo build failed with status: {}", status);
            return ExitCode::FAILURE;
        }
        Err(e) => {
            eprintln!("Failed to run cargo build: {}", e);
            return ExitCode::FAILURE;
        }
    }

    let binary_path = out_dir
        .join("target")
        .join("debug")
        .join(binary_name_from_ir(&ir));
    if !binary_path.exists() {
        eprintln!(
            "Built binary not found at expected path {}",
            binary_path.display()
        );
        return ExitCode::FAILURE;
    }

    let rollouts_path = artifacts_dir.join("rollouts.jsonl");
    let mut rollouts_file = match fs::File::create(&rollouts_path) {
        Ok(file) => file,
        Err(e) => {
            eprintln!(
                "Failed to create rollouts file {}: {}",
                rollouts_path.display(),
                e
            );
            return ExitCode::FAILURE;
        }
    };

    let mut scored_cases = 0usize;
    let mut passed_cases = 0usize;
    let mut errored_cases = 0usize;
    let mut score_sum = 0.0f64;

    for (idx, case) in objective.dataset.iter().enumerate() {
        let case_id = case
            .id
            .clone()
            .unwrap_or_else(|| format!("case-{:04}", idx + 1));
        let input_json = match serde_json::to_string(&case.input) {
            Ok(s) => s,
            Err(e) => {
                let record = RolloutRecord {
                    run_id: run_id.clone(),
                    case_id: case_id.clone(),
                    target_kind: objective.target.kind.clone(),
                    target_name: objective.target.name.clone(),
                    input: case.input.clone(),
                    expected: case.expected.clone(),
                    expected_contains: case.expected_contains.clone(),
                    output: None,
                    output_text: String::new(),
                    stderr: String::new(),
                    exit_code: None,
                    duration_ms: 0,
                    passed: None,
                    score: None,
                    error: Some(format!("failed to serialize case input: {}", e)),
                };
                let _ = serde_json::to_writer(&mut rollouts_file, &record);
                let _ = writeln!(&mut rollouts_file);
                errored_cases += 1;
                continue;
            }
        };

        let mut command_preview = vec![binary_path.display().to_string()];
        if let Some(config_path) = config {
            command_preview.push("--config".to_string());
            command_preview.push(config_path.display().to_string());
        }
        command_preview.push(objective.target.kind.clone());
        command_preview.push(objective.target.name.clone());
        command_preview.push("--input".to_string());
        command_preview.push(input_json.clone());

        let started_at_ms = current_timestamp_ms();
        let started = Instant::now();

        let mut run_cmd = std::process::Command::new(&binary_path);
        if let Some(config_path) = config {
            run_cmd.arg("--config").arg(config_path);
        }
        if verbose {
            run_cmd.env("SCAFFOLD_LOG", "1");
        }
        if let Some(path) = &convos_path {
            run_cmd.env("SCAFFOLD_AGENT_CONVOS_PATH", path);
            run_cmd.env("SCAFFOLD_RUN_ID", &run_id);
            run_cmd.env("SCAFFOLD_CASE_ID", &case_id);
            run_cmd.env("SCAFFOLD_TARGET_KIND", &objective.target.kind);
            run_cmd.env("SCAFFOLD_TARGET_NAME", &objective.target.name);
        }
        run_cmd
            .arg(&objective.target.kind)
            .arg(&objective.target.name)
            .arg("--input")
            .arg(&input_json)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let output = run_cmd.output();
        let elapsed_ms = started.elapsed().as_millis() as u64;

        let (output_text, stderr_text, exit_code, parsed_output, execution_error) = match output {
            Ok(output) => {
                let stdout_text = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr_text = String::from_utf8_lossy(&output.stderr).to_string();
                let parsed_output =
                    serde_json::from_str::<serde_json::Value>(stdout_text.trim()).ok();
                let mut error = None;
                if !output.status.success() {
                    error = Some(format!("command exited with status {}", output.status));
                }
                (
                    stdout_text,
                    stderr_text,
                    output.status.code(),
                    parsed_output,
                    error,
                )
            }
            Err(e) => (
                String::new(),
                String::new(),
                None,
                None,
                Some(format!("failed to execute command: {}", e)),
            ),
        };

        let (passed, score, check_error) =
            evaluate_case(case, parsed_output.as_ref(), &output_text);
        let merged_error = match (execution_error.clone(), check_error) {
            (Some(a), Some(b)) => Some(format!("{}; {}", a, b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        };

        if let Some(s) = score {
            scored_cases += 1;
            score_sum += s;
        }
        if let Some(true) = passed {
            passed_cases += 1;
        }
        if execution_error.is_some() {
            errored_cases += 1;
        }

        let record = RolloutRecord {
            run_id: run_id.clone(),
            case_id: case_id.clone(),
            target_kind: objective.target.kind.clone(),
            target_name: objective.target.name.clone(),
            input: case.input.clone(),
            expected: case.expected.clone(),
            expected_contains: case.expected_contains.clone(),
            output: parsed_output.clone(),
            output_text: output_text.clone(),
            stderr: stderr_text.clone(),
            exit_code,
            duration_ms: elapsed_ms,
            passed,
            score,
            error: merged_error.clone(),
        };
        if serde_json::to_writer(&mut rollouts_file, &record).is_err()
            || writeln!(&mut rollouts_file).is_err()
        {
            eprintln!("Failed writing rollout record for case {}", case_id);
            return ExitCode::FAILURE;
        }

        let trace = RolloutTrace {
            case_id: case_id.clone(),
            started_at_ms,
            duration_ms: elapsed_ms,
            command: command_preview,
            exit_code,
            stdout: output_text,
            stderr: stderr_text,
            error: merged_error,
        };
        if let Ok(trace_json) = serde_json::to_string_pretty(&trace) {
            let trace_name = format!("{}.json", sanitize_id(&case_id));
            let _ = fs::write(traces_dir.join(trace_name), trace_json);
        }
    }

    let _ = rollouts_file.flush();
    let total_cases = objective.dataset.len();
    let failed_cases = scored_cases.saturating_sub(passed_cases);
    let average_score = if scored_cases > 0 {
        Some(score_sum / scored_cases as f64)
    } else {
        None
    };

    let summary = OptimizeSummary {
        run_id: run_id.clone(),
        harness: harness.display().to_string(),
        objective: objective_path.display().to_string(),
        target_kind: objective.target.kind.clone(),
        target_name: objective.target.name.clone(),
        total_cases,
        scored_cases,
        passed_cases,
        failed_cases,
        errored_cases,
        average_score,
        artifacts_dir: artifacts_dir.display().to_string(),
        rollouts_file: rollouts_path.display().to_string(),
        traces_dir: traces_dir.display().to_string(),
        agent_convos_file: convos_path.as_ref().map(|p| p.display().to_string()),
        binary_path: binary_path.display().to_string(),
        build_dir: out_dir.display().to_string(),
    };

    let summary_path = artifacts_dir.join("summary.json");
    match serde_json::to_string_pretty(&summary) {
        Ok(json) => {
            if let Err(e) = fs::write(&summary_path, json) {
                eprintln!("Failed to write summary {}: {}", summary_path.display(), e);
                return ExitCode::FAILURE;
            }
        }
        Err(e) => {
            eprintln!("Failed to serialize optimize summary: {}", e);
            return ExitCode::FAILURE;
        }
    }

    match serde_json::to_string_pretty(&summary) {
        Ok(json) => println!("{}", json),
        Err(_) => println!(
            "Optimization run complete. Summary written to {}",
            summary_path.display()
        ),
    }

    if errored_cases == total_cases {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}
