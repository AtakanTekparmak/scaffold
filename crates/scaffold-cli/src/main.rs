//! Scaffold DSL CLI
//!
//! Commands:
//! - check: Parse and type check a scaffold file
//! - compile: Compile to IR and output JSON
//! - codegen: Generate Rust code from scaffold file
//! - run: Build and execute a scaffold file via generated binary

use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use ariadne::{Color, Label, Report, ReportKind, Source};
use clap::{Parser, Subcommand};

use scaffold_codegen::CodeGenerator;
use scaffold_ir::{to_json, Lowerer};
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
                let abs = runtime_path
                    .canonicalize()
                    .unwrap_or_else(|_| runtime_path);
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
                let abs = runtime_path
                    .canonicalize()
                    .unwrap_or_else(|_| runtime_path);
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
        eprintln!("{} '{}' not found in {}", target_kind, target_name, file.display());
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
    if verbose {
        eprintln!("Running {} {}", target_kind, target_name);
    }
    let mut run_cmd = std::process::Command::new(&binary_path);
    if let Some(config_path) = config {
        run_cmd.arg("--config").arg(config_path);
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
