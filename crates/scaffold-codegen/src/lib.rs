//! Scaffold Code Generator
//!
//! This crate generates Rust code from scaffold IR.
//!
//! # Usage
//!
//! ```ignore
//! use scaffold_codegen::CodeGenerator;
//! use scaffold_ir::ScaffoldIR;
//!
//! let ir: ScaffoldIR = // ... load IR
//! let generator = CodeGenerator::new();
//! let output = generator.generate(&ir)?;
//!
//! // Write generated files to disk
//! output.write_to_dir("/path/to/output")?;
//! ```

pub mod agents;
pub mod expr;
pub mod foreign;
pub mod pipelines;
pub mod prompts;
pub mod tools;
pub mod types;
pub mod util;

use scaffold_ir::{ExprIR, ScaffoldIR, TypeDefIR};
use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::Path;

pub use agents::{gen_agent_module, gen_agents_mod};
pub use expr::gen_expr;
pub use foreign::{
    gen_extern_crate_deps, gen_foreign_impl_module, gen_foreign_module, gen_foreign_types_module,
};
pub use pipelines::{gen_pipeline_module, gen_pipelines_mod};
pub use prompts::{gen_prompt_module, gen_prompts_mod};
pub use tools::{gen_tool_module, gen_tools_mod};
pub use types::{gen_struct_def, gen_type, gen_types_module};
pub use util::{to_pascal_case, to_snake_case};

/// Generated code output
#[derive(Debug, Default)]
pub struct GeneratedOutput {
    /// Map of relative path to file contents
    pub files: HashMap<String, String>,
}

impl GeneratedOutput {
    /// Create new empty output
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a file to the output
    pub fn add_file(&mut self, path: impl Into<String>, content: impl Into<String>) {
        self.files.insert(path.into(), content.into());
    }

    /// Write all files to a directory
    pub fn write_to_dir(&self, dir: impl AsRef<Path>) -> io::Result<()> {
        let dir = dir.as_ref();

        for (rel_path, content) in &self.files {
            let full_path = dir.join(rel_path);

            // Create parent directories
            if let Some(parent) = full_path.parent() {
                fs::create_dir_all(parent)?;
            }

            fs::write(&full_path, content)?;
        }

        Ok(())
    }

    /// Get file content by path
    pub fn get(&self, path: &str) -> Option<&str> {
        self.files.get(path).map(|s| s.as_str())
    }
}

/// Code generator for scaffold IR
#[derive(Debug, Default)]
pub struct CodeGenerator {
    /// Whether to format generated code
    format_code: bool,
    /// Whether to enforce strict validation (no unresolved idents/calls)
    strict_mode: bool,
}

impl CodeGenerator {
    /// Create a new code generator
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable code formatting (requires rustfmt)
    pub fn with_formatting(mut self, enabled: bool) -> Self {
        self.format_code = enabled;
        self
    }

    /// Enable strict validation for code generation
    pub fn with_strict(mut self, enabled: bool) -> Self {
        self.strict_mode = enabled;
        self
    }

    /// Generate Rust code from scaffold IR
    pub fn generate(&self, ir: &ScaffoldIR) -> Result<GeneratedOutput, CodeGenError> {
        let mut output = GeneratedOutput::new();

        // Strict validation: reject unknown identifiers and function calls in expressions
        if self.strict_mode {
            validate_strict(ir)?;
        }

        // Build type definition map (reserved for future use)
        let _type_map: HashMap<String, TypeDefIR> = ir
            .types
            .iter()
            .map(|t| (t.name.clone(), t.clone()))
            .collect();

        // Generate Cargo.toml
        output.add_file("Cargo.toml", self.gen_cargo_toml(ir));

        // Generate src/lib.rs
        output.add_file("src/lib.rs", self.gen_lib_rs(ir));

        // Generate src/main.rs (CLI entrypoint)
        output.add_file("src/main.rs", self.gen_main_rs(ir));

        // Generate src/types.rs
        let types_code = gen_types_module(&ir.types);
        output.add_file("src/types.rs", self.format_tokens(types_code));

        // Generate foreign modules
        if !ir.foreign_modules.is_empty() {
            // Generate foreign_types.rs (wrapper types for foreign type aliases)
            let foreign_types_code = foreign::gen_foreign_types_module(&ir.foreign_modules);
            output.add_file(
                "src/foreign_types.rs",
                self.format_tokens(foreign_types_code),
            );

            // Generate foreign_impl.rs (implementation stubs for foreign functions)
            let foreign_impl_code = foreign::gen_foreign_impl_module(&ir.foreign_modules);
            output.add_file("src/foreign_impl.rs", self.format_tokens(foreign_impl_code));

            // Generate foreign/mod.rs
            let foreign_names: Vec<_> =
                ir.foreign_modules.iter().map(|m| m.name.as_str()).collect();
            let foreign_mod = foreign::gen_foreign_mod(&foreign_names);
            output.add_file("src/foreign/mod.rs", self.format_tokens(foreign_mod));

            // Generate individual foreign module files
            for module in &ir.foreign_modules {
                let module_code = foreign::gen_foreign_module(module);
                let filename = format!("src/foreign/{}.rs", to_snake_case(&module.name));
                output.add_file(&filename, self.format_tokens(module_code));
            }
        }

        // Build maps of tools and prompts by name for context-aware generation (used by pipelines)
        let mut tools_index: HashMap<String, scaffold_ir::ToolIR> = HashMap::new();
        for t in &ir.tools {
            tools_index.insert(t.name.clone(), t.clone());
        }

        let mut prompts_index: HashMap<String, scaffold_ir::PromptIR> = HashMap::new();
        for p in &ir.prompts {
            prompts_index.insert(p.name.clone(), p.clone());
        }

        let mut agents_index: HashMap<String, scaffold_ir::AgentIR> = HashMap::new();
        for a in &ir.agents {
            agents_index.insert(a.name.clone(), a.clone());
        }

        let mut types_index: HashMap<String, scaffold_ir::TypeDefIR> = HashMap::new();
        for t in &ir.types {
            types_index.insert(t.name.clone(), t.clone());
        }

        // Generate tools
        if !ir.tools.is_empty() {
            // Generate tools/mod.rs
            let tool_names: Vec<_> = ir.tools.iter().map(|t| t.name.as_str()).collect();
            let tools_mod = tools::gen_tools_mod(&tool_names);
            output.add_file("src/tools/mod.rs", self.format_tokens(tools_mod));

            // Generate individual tool files with context
            for tool in &ir.tools {
                let tool_code = tools::gen_tool_module(tool, &tools_index);
                let filename = format!("src/tools/{}.rs", to_snake_case(&tool.name));
                output.add_file(&filename, self.format_tokens(tool_code));
            }
        }

        // Generate prompts
        if !ir.prompts.is_empty() {
            // Generate prompts/mod.rs
            let prompt_names: Vec<_> = ir.prompts.iter().map(|p| p.name.as_str()).collect();
            let prompts_mod = prompts::gen_prompts_mod(&prompt_names);
            output.add_file("src/prompts/mod.rs", self.format_tokens(prompts_mod));

            // Generate individual prompt files
            for prompt in &ir.prompts {
                let prompt_code = prompts::gen_prompt_module(prompt);
                let filename = format!("src/prompts/{}.rs", to_snake_case(&prompt.name));
                output.add_file(&filename, self.format_tokens(prompt_code));
            }
        }

        // Generate agents
        if !ir.agents.is_empty() {
            // Generate agents/mod.rs
            let agent_names: Vec<_> = ir.agents.iter().map(|a| a.name.as_str()).collect();
            let agents_mod = agents::gen_agents_mod(&agent_names);
            output.add_file("src/agents/mod.rs", self.format_tokens(agents_mod));

            // Generate individual agent files
            for agent in &ir.agents {
                let agent_code = agents::gen_agent_module(agent);
                let filename = format!("src/agents/{}.rs", to_snake_case(&agent.name));
                output.add_file(&filename, self.format_tokens(agent_code));
            }
        }

        // Generate pipelines
        if !ir.pipelines.is_empty() {
            // Generate pipelines/mod.rs
            let pipeline_names: Vec<_> = ir.pipelines.iter().map(|p| p.name.as_str()).collect();
            let pipelines_mod = pipelines::gen_pipelines_mod(&pipeline_names);
            output.add_file("src/pipelines/mod.rs", self.format_tokens(pipelines_mod));

            // Generate individual pipeline files
            for pipeline in &ir.pipelines {
                let pipeline_code = pipelines::gen_pipeline_module(
                    pipeline,
                    &tools_index,
                    &prompts_index,
                    &agents_index,
                    &types_index,
                );
                let filename = format!("src/pipelines/{}.rs", to_snake_case(&pipeline.name));
                output.add_file(&filename, self.format_tokens(pipeline_code));
            }
        }

        // Always generate helpers module (may be empty if no helper functions needed)
        let helpers_code = self.gen_helpers_module(ir);
        output.add_file("src/helpers.rs", helpers_code);

        Ok(output)
    }

    /// Collect all function calls from the IR with their argument counts
    fn collect_function_calls(&self, _ir: &ScaffoldIR) -> HashMap<String, usize> {
        // Function calls are now collected from agents/pipelines if needed
        HashMap::new()
    }

    /// Recursively collect function calls from an expression
    fn collect_calls_from_expr(&self, expr: &ExprIR, calls: &mut HashMap<String, usize>) {
        match expr {
            ExprIR::Call { function, args } => {
                // Skip built-in functions
                let builtins = [
                    "len",
                    "is_empty",
                    "contains",
                    "is_some",
                    "is_none",
                    "unwrap",
                    "unwrap_or",
                    "abs",
                    "min",
                    "max",
                    "not",
                ];
                if !builtins.contains(&function.as_str()) {
                    // Track maximum arg count seen for this function
                    let entry = calls.entry(function.clone()).or_insert(0);
                    *entry = (*entry).max(args.len());
                }
                for arg in args {
                    self.collect_calls_from_expr(arg, calls);
                }
            }
            ExprIR::Binary { left, right, .. } => {
                self.collect_calls_from_expr(left, calls);
                self.collect_calls_from_expr(right, calls);
            }
            ExprIR::FieldAccess { base, .. } => {
                self.collect_calls_from_expr(base, calls);
            }
            ExprIR::ForeignCall { args, .. } => {
                // Foreign calls - just recurse into arguments
                for arg in args {
                    self.collect_calls_from_expr(arg, calls);
                }
            }
            ExprIR::Literal { .. } | ExprIR::Ident { .. } => {}
        }
    }

    /// Generate helpers module with stub functions
    fn gen_helpers_module(&self, ir: &ScaffoldIR) -> String {
        let calls = self.collect_function_calls(ir);

        let mut code = String::new();
        code.push_str("//! Helper functions\n");
        code.push_str("//!\n");
        code.push_str("//! Implement any custom helper functions here.\n\n");
        code.push_str("#![allow(unused_imports, unused_variables)]\n\n");
        code.push_str("use crate::types::*;\n\n");

        if calls.is_empty() {
            code.push_str("// No helper function stubs needed\n");
            return code;
        }

        code.push_str("// The following stub functions were auto-generated.\n");
        code.push_str("// Replace them with actual implementations.\n\n");

        for (func, arg_count) in calls {
            let snake = to_snake_case(&func);
            code.push_str("/// TODO: Implement this function\n");
            code.push_str(&format!(
                "/// This is a stub generated because '{}' was called in the scaffold file.\n",
                func
            ));

            // Generate function signature based on argument count
            // Use references (&impl ...) to avoid move errors
            let params: Vec<String> = (0..arg_count)
                .map(|i| format!("arg{}: &impl std::fmt::Debug", i))
                .collect();
            let param_str = params.join(", ");

            code.push_str(&format!("pub fn {}({}) -> i64 {{\n", snake, param_str));
            code.push_str("    // TODO: Implement actual logic\n");

            // Generate print statement with all args
            let arg_refs: Vec<String> = (0..arg_count).map(|i| format!("arg{}", i)).collect();
            if arg_count == 0 {
                code.push_str(&format!(
                    "    eprintln!(\"WARNING: stub function '{}' called\");\n",
                    snake
                ));
            } else if arg_count == 1 {
                code.push_str(&format!(
                    "    eprintln!(\"WARNING: stub function '{}' called with {{:?}}\", {});\n",
                    snake, arg_refs[0]
                ));
            } else {
                let format_placeholders: Vec<&str> = (0..arg_count).map(|_| "{:?}").collect();
                code.push_str(&format!(
                    "    eprintln!(\"WARNING: stub function '{}' called with ({})\", {});\n",
                    snake,
                    format_placeholders.join(", "),
                    arg_refs.join(", ")
                ));
            }
            code.push_str("    0\n");
            code.push_str("}\n\n");
        }

        code
    }

    /// Generate Cargo.toml for the output crate
    fn gen_cargo_toml(&self, ir: &ScaffoldIR) -> String {
        // Extract a crate name from first pipeline or tool
        let crate_name = ir
            .pipelines
            .first()
            .map(|p| to_snake_case(&p.name))
            .or_else(|| ir.tools.first().map(|t| to_snake_case(&t.name)))
            .unwrap_or_else(|| "scaffold_generated".to_string());

        // Generate extern crate dependencies
        let extern_deps = foreign::gen_extern_crate_deps(&ir.extern_crates);

        // Resolve scaffold-runtime dependency from environment or default to local path
        // Options (by precedence):
        // 1) SCAFFOLD_RUNTIME_VERSION="x.y" -> use crates.io version
        // 2) SCAFFOLD_RUNTIME_PATH="/path/to/scaffold-runtime" -> use path dependency
        // 3) default to "../scaffold-runtime"
        let runtime_dep = if let Ok(ver) = std::env::var("SCAFFOLD_RUNTIME_VERSION") {
            format!("scaffold-runtime = \"{}\"", ver)
        } else if let Ok(path) = std::env::var("SCAFFOLD_RUNTIME_PATH") {
            format!("scaffold-runtime = {{ path = \"{}\" }}", path)
        } else {
            "scaffold-runtime = { path = \"../scaffold-runtime\" }".to_string()
        };

        format!(
            r#"[package]
name = "{crate_name}"
version = "0.1.0"
edition = "2021"

[workspace]

[[bin]]
name = "{crate_name}"
path = "src/main.rs"

[lib]
name = "{crate_name}_lib"
path = "src/lib.rs"

[dependencies]
{runtime_dep}
serde = {{ version = "1.0", features = ["derive"] }}
serde_json = "1.0"
schemars = "1.2.0"
clap = {{ version = "4.0", features = ["derive"] }}
tokio = {{ version = "1.0", features = ["full"] }}
{extern_deps}"#
        )
    }

    /// Generate src/main.rs with CLI entrypoint
    fn gen_main_rs(&self, ir: &ScaffoldIR) -> String {
        let crate_name = ir
            .pipelines
            .first()
            .map(|p| to_snake_case(&p.name))
            .or_else(|| ir.tools.first().map(|t| to_snake_case(&t.name)))
            .unwrap_or_else(|| "scaffold_generated".to_string());

        let lib_name = format!("{}_lib", crate_name);

        // Collect available components
        let pipeline_names: Vec<_> = ir.pipelines.iter().map(|p| p.name.as_str()).collect();
        let tool_names: Vec<_> = ir.tools.iter().map(|t| t.name.as_str()).collect();
        let agent_names: Vec<_> = ir.agents.iter().map(|a| a.name.as_str()).collect();
        let prompt_names: Vec<_> = ir.prompts.iter().map(|p| p.name.as_str()).collect();

        // Generate match arms for pipelines
        let pipeline_arms: String = pipeline_names
            .iter()
            .map(|name| {
                let snake = to_snake_case(name);
                format!(
                    r#"            "{name}" => {{
                let input: {lib_name}::pipelines::{snake}::Input = serde_json::from_str(&input_json)?;
                let result = {lib_name}::pipelines::{snake}::run(input).await?;
                println!("{{}}", serde_json::to_string_pretty(&result)?);
            }}"#
                )
            })
            .collect::<Vec<_>>()
            .join("\n");

        // Generate match arms for tools
        let tool_arms: String = tool_names
            .iter()
            .map(|name| {
                let snake = to_snake_case(name);
                format!(
                    r#"            "{name}" => {{
                let input: {lib_name}::tools::{snake}::Input = serde_json::from_str(&input_json)?;
                let result = {lib_name}::tools::{snake}::run(input).await?;
                println!("{{}}", serde_json::to_string_pretty(&result)?);
            }}"#
                )
            })
            .collect::<Vec<_>>()
            .join("\n");

        // Generate match arms for agents
        let agent_arms: String = agent_names
            .iter()
            .map(|name| {
                let snake = to_snake_case(name);
                format!(
                    r#"            "{name}" => {{
                let input: {lib_name}::agents::{snake}::Input = serde_json::from_str(&input_json)?;
                let result = {lib_name}::agents::{snake}::run(input).await?;
                println!("{{}}", serde_json::to_string_pretty(&result)?);
            }}"#
                )
            })
            .collect::<Vec<_>>()
            .join("\n");

        // Generate match arms for prompts
        let prompt_arms: String = prompt_names
            .iter()
            .map(|name| {
                let snake = to_snake_case(name);
                format!(
                    r#"            "{name}" => {{
                let input: {lib_name}::prompts::{snake}::Input = serde_json::from_str(&input_json)?;
                let result = {lib_name}::prompts::{snake}::run(input).await?;
                println!("{{}}", serde_json::to_string_pretty(&result)?);
            }}"#
                )
            })
            .collect::<Vec<_>>()
            .join("\n");

        // Generate list output
        let list_pipelines = pipeline_names
            .iter()
            .map(|n| format!("  - {}", n))
            .collect::<Vec<_>>()
            .join("\n");
        let list_tools = tool_names
            .iter()
            .map(|n| format!("  - {}", n))
            .collect::<Vec<_>>()
            .join("\n");
        let list_agents = agent_names
            .iter()
            .map(|n| format!("  - {}", n))
            .collect::<Vec<_>>()
            .join("\n");
        let list_prompts = prompt_names
            .iter()
            .map(|n| format!("  - {}", n))
            .collect::<Vec<_>>()
            .join("\n");

        format!(
            r#"//! Generated CLI entrypoint
//!
//! Run with: cargo run -- --help

use clap::{{Parser, Subcommand}};

#[derive(Parser)]
#[command(name = "{crate_name}")]
#[command(about = "Generated scaffold CLI", long_about = None)]
struct Cli {{
    #[command(subcommand)]
    command: Commands,
}}

#[derive(Subcommand)]
enum Commands {{
    /// Run a pipeline
    Pipeline {{
        /// Name of the pipeline to run
        name: String,
        /// JSON input
        #[arg(short, long)]
        input: String,
    }},
    /// Run a tool
    Tool {{
        /// Name of the tool to run
        name: String,
        /// JSON input
        #[arg(short, long)]
        input: String,
    }},
    /// Run an agent
    Agent {{
        /// Name of the agent to run
        name: String,
        /// JSON input
        #[arg(short, long)]
        input: String,
    }},
    /// Run a prompt
    Prompt {{
        /// Name of the prompt to run
        name: String,
        /// JSON input
        #[arg(short, long)]
        input: String,
    }},
    /// List available components
    List,
}}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {{
    let cli = Cli::parse();

    match cli.command {{
        Commands::Pipeline {{ name, input: input_json }} => {{
            match name.as_str() {{
{pipeline_arms}
                _ => eprintln!("Unknown pipeline: {{}}. Use 'list' to see available pipelines.", name),
            }}
        }}
        Commands::Tool {{ name, input: input_json }} => {{
            match name.as_str() {{
{tool_arms}
                _ => eprintln!("Unknown tool: {{}}. Use 'list' to see available tools.", name),
            }}
        }}
        Commands::Agent {{ name, input: input_json }} => {{
            match name.as_str() {{
{agent_arms}
                _ => eprintln!("Unknown agent: {{}}. Use 'list' to see available agents.", name),
            }}
        }}
        Commands::Prompt {{ name, input: input_json }} => {{
            match name.as_str() {{
{prompt_arms}
                _ => eprintln!("Unknown prompt: {{}}. Use 'list' to see available prompts.", name),
            }}
        }}
        Commands::List => {{
            println!("Available components:\n");
            println!("Pipelines:");
            println!("{list_pipelines}");
            println!("\nTools:");
            println!("{list_tools}");
            println!("\nAgents:");
            println!("{list_agents}");
            println!("\nPrompts:");
            println!("{list_prompts}");
        }}
    }}

    Ok(())
}}
"#
        )
    }

    /// Generate src/lib.rs
    fn gen_lib_rs(&self, ir: &ScaffoldIR) -> String {
        let mut modules = vec!["pub mod types;", "pub mod helpers;"];

        if !ir.foreign_modules.is_empty() {
            modules.push("pub mod foreign;");
            modules.push("pub mod foreign_types;");
            modules.push("pub mod foreign_impl;");
        }

        if !ir.tools.is_empty() {
            modules.push("pub mod tools;");
        }

        if !ir.prompts.is_empty() {
            modules.push("pub mod prompts;");
        }

        if !ir.agents.is_empty() {
            modules.push("pub mod agents;");
        }

        if !ir.pipelines.is_empty() {
            modules.push("pub mod pipelines;");
        }

        let mut code = String::new();
        code.push_str("//! Generated from scaffold IR\n\n");

        for module in &modules {
            code.push_str(module);
            code.push('\n');
        }

        code.push_str("\n// Re-exports for convenience\n");
        code.push_str("pub use scaffold_runtime::prelude::*;\n");
        code.push_str("pub use types::*;\n");
        code.push_str("pub use helpers::*;\n");

        if !ir.foreign_modules.is_empty() {
            code.push_str("pub use foreign::*;\n");
        }

        if !ir.tools.is_empty() {
            code.push_str("pub use tools::*;\n");
        }

        if !ir.prompts.is_empty() {
            code.push_str("pub use prompts::*;\n");
        }

        if !ir.agents.is_empty() {
            code.push_str("pub use agents::*;\n");
        }

        if !ir.pipelines.is_empty() {
            code.push_str("pub use pipelines::*;\n");
        }

        code
    }

    /// Format token stream to string
    fn format_tokens(&self, tokens: proc_macro2::TokenStream) -> String {
        let code = tokens.to_string();

        if self.format_code {
            // Try to format with rustfmt
            self.try_format(&code).unwrap_or(code)
        } else {
            code
        }
    }

    /// Try to format code with rustfmt
    fn try_format(&self, code: &str) -> Option<String> {
        use std::io::Write;
        use std::process::{Command, Stdio};

        let mut child = Command::new("rustfmt")
            .args(["--edition", "2021"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .ok()?;

        {
            let stdin = child.stdin.as_mut()?;
            stdin.write_all(code.as_bytes()).ok()?;
        }

        let output = child.wait_with_output().ok()?;
        if output.status.success() {
            String::from_utf8(output.stdout).ok()
        } else {
            None
        }
    }
}

/// Code generation error
#[derive(Debug)]
pub enum CodeGenError {
    /// IO error
    Io(io::Error),
    /// Invalid IR
    InvalidIR(String),
}

impl std::fmt::Display for CodeGenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CodeGenError::Io(e) => write!(f, "IO error: {}", e),
            CodeGenError::InvalidIR(msg) => write!(f, "Invalid IR: {}", msg),
        }
    }
}

impl std::error::Error for CodeGenError {}

impl From<io::Error> for CodeGenError {
    fn from(e: io::Error) -> Self {
        CodeGenError::Io(e)
    }
}

// ================= Strict validation helpers =================
#[allow(dead_code)]
fn validate_strict(_ir: &ScaffoldIR) -> Result<(), CodeGenError> {
    // Strict validation was for tasks, which have been removed
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use scaffold_ir::{ScaffoldIR, TypeDefIR, TypeIR};

    fn create_simple_ir() -> ScaffoldIR {
        ScaffoldIR {
            version: "0.1.0".to_string(),
            types: vec![TypeDefIR {
                name: "Position".to_string(),
                definition: TypeIR::Struct {
                    fields: {
                        let mut fields = HashMap::new();
                        fields.insert("x".to_string(), TypeIR::Int);
                        fields.insert("y".to_string(), TypeIR::Int);
                        fields
                    },
                },
            }],
            extern_crates: vec![],
            foreign_modules: vec![],
            tools: vec![],
            prompts: vec![],
            agents: vec![],
            pipelines: vec![],
        }
    }

    #[test]
    fn test_generate_types() {
        let ir = create_simple_ir();
        let generator = CodeGenerator::new();
        let output = generator.generate(&ir).unwrap();

        assert!(output.files.contains_key("src/types.rs"));
        let types_code = output.get("src/types.rs").unwrap();
        assert!(types_code.contains("Position"));
    }

    #[test]
    fn test_generate_cargo_toml() {
        let ir = create_simple_ir();
        let generator = CodeGenerator::new();
        let output = generator.generate(&ir).unwrap();

        assert!(output.files.contains_key("Cargo.toml"));
        let cargo = output.get("Cargo.toml").unwrap();
        assert!(cargo.contains("scaffold-runtime"));
    }
}
