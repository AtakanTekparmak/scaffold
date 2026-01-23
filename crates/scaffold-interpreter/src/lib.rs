//! Scaffold Interpreter
//!
//! Executes .scaffold files directly without compilation.
//!
//! # Architecture
//!
//! ```text
//! .scaffold file
//!       ↓
//!   [Parser] → AST
//!       ↓
//!   [Type Checker] → Validated AST
//!       ↓
//!   [IR Lowering] → IR
//!       ↓
//!   [Interpreter] → Execution
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use scaffold_interpreter::Interpreter;
//!
//! let mut interp = Interpreter::load("agent.scaffold")?;
//! let result = interp.run_task("analyze", input).await?;
//! ```

pub mod error;
pub mod executor;
pub mod foreign;
pub mod loader;

use scaffold_ir::ScaffoldIR;
use scaffold_runtime::{PromptManager, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub use error::{InterpreterError, Result};
pub use executor::ToolExecutor;
pub use foreign::{ForeignRegistry, ForeignRegistryBuilder};
pub use loader::Loader;

/// Main interpreter for scaffold files
pub struct Interpreter {
    /// Loaded IR
    ir: ScaffoldIR,
    /// Source file path
    source_path: PathBuf,
    /// Prompt manager for templates
    prompts: PromptManager,
    /// Global state
    state: HashMap<String, Value>,
    /// Tool executor
    tool_executor: ToolExecutor,
}

impl Interpreter {
    /// Load a scaffold file
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let loader = Loader::new();
        let ir = loader.load(path)?;

        // Load prompts from prompts/ directory if it exists
        let prompts_dir = path.parent().map(|p| p.join("prompts"));
        let prompts = if let Some(dir) = prompts_dir {
            PromptManager::with_template_dir(dir)
                .unwrap_or_else(|_| PromptManager::new())
        } else {
            PromptManager::new()
        };

        // Set up tool executor with base path for file() references
        let base_path = path.parent().map(|p| p.to_path_buf());
        let mut tool_executor = if let Some(ref bp) = base_path {
            ToolExecutor::new().with_base_path(bp.clone())
        } else {
            ToolExecutor::new()
        };
        tool_executor.register_tools(&ir.tools);
        tool_executor.register_prompts(&ir.prompts);
        tool_executor.register_agents(&ir.agents);
        tool_executor.register_pipelines(&ir.pipelines);

        Ok(Self {
            ir,
            source_path: path.to_path_buf(),
            prompts,
            state: HashMap::new(),
            tool_executor,
        })
    }

    /// Reload the scaffold file (for hot-reload)
    pub fn reload(&mut self) -> Result<()> {
        let loader = Loader::new();
        self.ir = loader.load(&self.source_path)?;
        self.tool_executor.register_tools(&self.ir.tools);
        self.tool_executor.register_prompts(&self.ir.prompts);
        self.tool_executor.register_agents(&self.ir.agents);
        self.tool_executor.register_pipelines(&self.ir.pipelines);
        self.prompts.reload().map_err(|e| InterpreterError::Runtime(e.to_string()))?;
        Ok(())
    }

    /// Get available tool names
    pub fn tool_names(&self) -> Vec<&str> {
        self.ir.tools.iter().map(|t| t.name.as_str()).collect()
    }

    /// Get available prompt names
    pub fn prompt_names(&self) -> Vec<&str> {
        self.ir.prompts.iter().map(|p| p.name.as_str()).collect()
    }

    /// Get available agent names
    pub fn agent_names(&self) -> Vec<&str> {
        self.ir.agents.iter().map(|a| a.name.as_str()).collect()
    }

    /// Get available pipeline names
    pub fn pipeline_names(&self) -> Vec<&str> {
        self.ir.pipelines.iter().map(|p| p.name.as_str()).collect()
    }

    /// Run a tool with the given input
    pub async fn run_tool(&mut self, tool_name: &str, input: Value) -> Result<Value> {
        let tool = self.ir.tools.iter()
            .find(|t| t.name == tool_name)
            .ok_or_else(|| InterpreterError::ToolNotFound(tool_name.to_string()))?
            .clone();

        self.tool_executor.execute(&tool, input, &self.prompts).await
    }

    /// Run a prompt with the given input
    pub async fn run_prompt(&mut self, prompt_name: &str, input: Value) -> Result<Value> {
        let prompt = self.ir.prompts.iter()
            .find(|p| p.name == prompt_name)
            .ok_or_else(|| InterpreterError::Runtime(format!("Prompt '{}' not found", prompt_name)))?
            .clone();

        self.tool_executor.execute_prompt(&prompt, input, &self.prompts).await
    }

    /// Run an agent with the given input
    pub async fn run_agent(&mut self, agent_name: &str, input: Value) -> Result<Value> {
        let agent = self.ir.agents.iter()
            .find(|a| a.name == agent_name)
            .ok_or_else(|| InterpreterError::Runtime(format!("Agent '{}' not found", agent_name)))?
            .clone();

        self.tool_executor.execute_agent(&agent, input, &self.prompts).await
    }

    /// Run a pipeline with the given input
    pub async fn run_pipeline(&mut self, pipeline_name: &str, input: Value) -> Result<Value> {
        let pipeline = self.ir.pipelines.iter()
            .find(|p| p.name == pipeline_name)
            .ok_or_else(|| InterpreterError::Runtime(format!("Pipeline '{}' not found", pipeline_name)))?
            .clone();

        self.tool_executor.execute_pipeline(&pipeline, input, &self.prompts).await
    }

    /// Get the loaded IR (for inspection)
    pub fn ir(&self) -> &ScaffoldIR {
        &self.ir
    }

    /// Get/set global state
    pub fn state(&self) -> &HashMap<String, Value> {
        &self.state
    }

    pub fn state_mut(&mut self) -> &mut HashMap<String, Value> {
        &mut self.state
    }

    /// Get prompt manager for template management
    pub fn prompts(&self) -> &PromptManager {
        &self.prompts
    }

    /// Register a foreign function
    ///
    /// # Example
    /// ```ignore
    /// interp.register_foreign("math", "double", |args| {
    ///     let n = args.first().and_then(|v| v.as_int()).unwrap_or(0);
    ///     Ok(Value::Int(n * 2))
    /// });
    /// ```
    pub fn register_foreign<F>(&mut self, module: &str, function: &str, f: F)
    where
        F: Fn(Vec<Value>) -> Result<Value> + Send + Sync + 'static,
    {
        self.tool_executor.foreign_registry_mut().register(module, function, f);
    }

    /// Initialize standard library foreign functions
    pub fn with_stdlib(mut self) -> Self {
        let registry = ForeignRegistryBuilder::new()
            .with_stdlib()
            .build();
        self.tool_executor = self.tool_executor.with_foreign_registry(registry);
        self
    }

    /// Get access to the foreign registry
    pub fn foreign_registry(&self) -> &ForeignRegistry {
        self.tool_executor.foreign_registry()
    }
}

/// Configuration for the interpreter
#[derive(Debug, Clone, Default)]
pub struct InterpreterConfig {
    /// Directory for prompt templates
    pub prompts_dir: Option<PathBuf>,
    /// Enable verbose logging
    pub verbose: bool,
    /// Timeout for LLM calls (ms)
    pub llm_timeout_ms: u64,
    /// Timeout for shell commands (ms)
    pub shell_timeout_ms: u64,
}

impl InterpreterConfig {
    pub fn new() -> Self {
        Self {
            prompts_dir: None,
            verbose: false,
            llm_timeout_ms: 30000,
            shell_timeout_ms: 60000,
        }
    }
}
