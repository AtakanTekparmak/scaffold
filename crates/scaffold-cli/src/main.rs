//! Scaffold DSL CLI
//!
//! Commands:
//! - check: Parse and type check a scaffold file
//! - compile: Compile to IR and output JSON
//! - codegen: Generate Rust code from scaffold file
//! - run: Execute a scaffold file directly (interpreter)

use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use ariadne::{Color, Label, Report, ReportKind, Source};
use clap::{Parser, Subcommand};

use scaffold_codegen::CodeGenerator;
use scaffold_interpreter::Interpreter;
use scaffold_ir::{to_json, Lowerer};
use scaffold_runtime::Value;
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

        /// Format generated code with rustfmt
        #[arg(long)]
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

        /// Format generated code with rustfmt
        #[arg(long)]
        format: bool,

        /// Enforce strict validation (no unresolved identifiers/calls)
        #[arg(long)]
        strict: bool,

        /// Path to scaffold-runtime crate (if not using SCAFFOLD_RUNTIME_VERSION)
        #[arg(long)]
        runtime_path: Option<String>,

        /// Build in release mode
        #[arg(long)]
        release: bool,
    },

    /// Run a scaffold file directly (interpreter mode)
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
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    match cli.command {
        Commands::Check { file, verbose } => cmd_check(&file, verbose),
        Commands::Compile { file, output, compact } => cmd_compile(&file, output.as_deref(), compact),
        Commands::Parse { file } => cmd_parse(&file),
        Commands::Codegen { file, output, format, strict } => cmd_codegen(&file, &output, format, strict),
        Commands::Build { file, output, bin_name, format, strict, runtime_path, release } => {
            cmd_build(&file, output.as_deref(), bin_name.as_deref(), format, strict, runtime_path.as_deref(), release)
        }
        Commands::Run { file, task, tool, prompt, agent, pipeline, input, verbose, verify } => {
            // Run the async runtime
            let rt = tokio::runtime::Runtime::new().expect("Failed to create runtime");
            rt.block_on(cmd_run(
                &file,
                task.as_deref(),
                tool.as_deref(),
                prompt.as_deref(),
                agent.as_deref(),
                pipeline.as_deref(),
                &input,
                verbose,
                verify,
            ))
        }
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
                    .with_color(if has_errors { Color::Red } else { Color::Yellow }),
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
            println!("  Deadlock analysis: {} tasks checked", verify_result.deadlock.len());
        }
        if !verify_result.bounds.is_empty() {
            println!("  Bounds analysis: {} subgoals checked", verify_result.bounds.len());
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

fn cmd_codegen(file: &PathBuf, output: &PathBuf, format: bool, strict: bool) -> ExitCode {
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
    let generator = CodeGenerator::new().with_formatting(format).with_strict(strict);
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

    eprintln!("Generated {} files to {}", generated.files.len(), output.display());
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

    // Configure runtime path env for codegen if provided
    if let Some(path) = runtime_path {
        std::env::set_var("SCAFFOLD_RUNTIME_PATH", path);
    }

    // Generate code
    let generator = CodeGenerator::new().with_formatting(format).with_strict(strict);
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
                cargo_toml = cargo_toml.replace("name = \"scaffold_generated\"", &format!("name = \"{}\"", name));
            }
            if cargo_toml.contains("[[bin]]\nname = \"scaffold_generated\"") {
                cargo_toml = cargo_toml.replace("[[bin]]\nname = \"scaffold_generated\"", &format!("[[bin]]\nname = \"{}\"", name));
            }
            if let Err(e) = fs::write(&manifest_path, cargo_toml) {
                eprintln!("Warning: failed to set bin name: {}", e);
            }
        }
    }

    // Run cargo build in the generated dir
    let mut cmd = std::process::Command::new("cargo");
    cmd.arg("build");
    if release { cmd.arg("--release"); }
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

async fn cmd_run(
    file: &PathBuf,
    _task: Option<&str>,  // Deprecated - tasks removed
    tool: Option<&str>,
    prompt: Option<&str>,
    agent: Option<&str>,
    pipeline: Option<&str>,
    input_str: &str,
    verbose: bool,
    _verify_enabled: bool,
) -> ExitCode {
    // Parse input JSON
    let input: Value = if input_str.starts_with('@') {
        // Read from file
        let input_path = &input_str[1..];
        match fs::read_to_string(input_path) {
            Ok(content) => match serde_json::from_str(&content) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("Error parsing input JSON from {}: {}", input_path, e);
                    return ExitCode::FAILURE;
                }
            },
            Err(e) => {
                eprintln!("Error reading input file {}: {}", input_path, e);
                return ExitCode::FAILURE;
            }
        }
    } else {
        match serde_json::from_str(input_str) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("Error parsing input JSON: {}", e);
                return ExitCode::FAILURE;
            }
        }
    };

    if verbose {
        eprintln!("Loading: {}", file.display());
    }

    // Load the interpreter
    let mut interpreter = match Interpreter::load(file) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("Error loading scaffold file: {}", e);
            return ExitCode::FAILURE;
        }
    };

    if verbose {
        eprintln!("Available tools: {:?}", interpreter.tool_names());
        eprintln!("Available prompts: {:?}", interpreter.prompt_names());
        eprintln!("Available agents: {:?}", interpreter.agent_names());
        eprintln!("Available pipelines: {:?}", interpreter.pipeline_names());
    }

    // Execute based on what was requested
    let result = if let Some(tool_name) = tool {
        if verbose {
            eprintln!("Running tool: {}", tool_name);
        }
        interpreter.run_tool(tool_name, input).await
    } else if let Some(prompt_name) = prompt {
        if verbose {
            eprintln!("Running prompt: {}", prompt_name);
        }
        interpreter.run_prompt(prompt_name, input).await
    } else if let Some(agent_name) = agent {
        if verbose {
            eprintln!("Running agent: {}", agent_name);
        }
        interpreter.run_agent(agent_name, input).await
    } else if let Some(pipeline_name) = pipeline {
        if verbose {
            eprintln!("Running pipeline: {}", pipeline_name);
        }
        interpreter.run_pipeline(pipeline_name, input).await
    } else {
        // Default: try to run first tool/agent/etc or list available
        let tools: Vec<String> = interpreter.tool_names().iter().map(|s| s.to_string()).collect();
        let prompts: Vec<String> = interpreter.prompt_names().iter().map(|s| s.to_string()).collect();
        let agents: Vec<String> = interpreter.agent_names().iter().map(|s| s.to_string()).collect();
        let pipelines: Vec<String> = interpreter.pipeline_names().iter().map(|s| s.to_string()).collect();

        let total = tools.len() + prompts.len() + agents.len() + pipelines.len();

        if total == 0 {
            eprintln!("No tools, prompts, agents, or pipelines found in {}", file.display());
            return ExitCode::FAILURE;
        }

        if tools.len() == 1 && prompts.is_empty() && agents.is_empty() && pipelines.is_empty() {
            let tool_name = &tools[0];
            if verbose {
                eprintln!("Running default tool: {}", tool_name);
            }
            interpreter.run_tool(tool_name, input).await
        } else if prompts.len() == 1 && tools.is_empty() && agents.is_empty() && pipelines.is_empty() {
            let prompt_name = &prompts[0];
            if verbose {
                eprintln!("Running default prompt: {}", prompt_name);
            }
            interpreter.run_prompt(prompt_name, input).await
        } else if agents.len() == 1 && tools.is_empty() && prompts.is_empty() && pipelines.is_empty() {
            let agent_name = &agents[0];
            if verbose {
                eprintln!("Running default agent: {}", agent_name);
            }
            interpreter.run_agent(agent_name, input).await
        } else if pipelines.len() == 1 && tools.is_empty() && prompts.is_empty() && agents.is_empty() {
            let pipeline_name = &pipelines[0];
            if verbose {
                eprintln!("Running default pipeline: {}", pipeline_name);
            }
            interpreter.run_pipeline(pipeline_name, input).await
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

    match result {
        Ok(output) => {
            // Output the result as JSON
            let json = serde_json::to_string_pretty(&output).unwrap_or_else(|_| format!("{:?}", output));
            println!("{}", json);
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("Execution error: {}", e);
            ExitCode::FAILURE
        }
    }
}
