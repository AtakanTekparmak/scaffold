# Scaffold Interpreter Architecture

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        scaffold-cli                              │
│  scaffold run file.scaffold    # interpreted (development)      │
│  scaffold watch file.scaffold  # hot-reload mode                │
│  scaffold codegen file.scaffold # compiled (production)         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    scaffold-interpreter                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │   Loader    │  │  Executor   │  │    Prompt Manager       │ │
│  │ .scaffold   │  │  IR → Run   │  │  *.jinja templates      │ │
│  │  + prompts/ │  │             │  │  hot-reload support     │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      scaffold-runtime                            │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────────────┐ │
│  │  Value   │ │   Tool   │ │   LLM    │ │  Template Engine   │ │
│  │  System  │ │  Runner  │ │  Client  │ │    (minijinja)     │ │
│  └──────────┘ └──────────┘ └──────────┘ └────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
scaffold-lang/
├── crates/
│   ├── scaffold-syntax/      # Parser (exists)
│   ├── scaffold-types/       # Type checker (exists)
│   ├── scaffold-ir/          # IR definitions (exists)
│   ├── scaffold-verify/      # Verification (exists)
│   ├── scaffold-runtime/     # Runtime library (exists, extend)
│   ├── scaffold-interpreter/ # NEW: Interpreter
│   ├── scaffold-codegen/     # Rust codegen (exists, for production)
│   └── scaffold-cli/         # CLI (exists, extend)
```

## Core Components

### 1. Runtime Value System (scaffold-runtime)

Dynamic values that can hold any scaffold type:

```rust
pub enum Value {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    Bytes(Vec<u8>),
    List(Vec<Value>),
    Map(HashMap<String, Value>),
    Struct { type_name: String, fields: HashMap<String, Value> },
    Option(Option<Box<Value>>),
    Result(Result<Box<Value>, Box<Value>>),
    // Function/closure for tool references
    Tool(ToolRef),
}
```

### 2. Interpreter (scaffold-interpreter)

```rust
pub struct Interpreter {
    /// Loaded IR
    ir: ScaffoldIR,
    /// Runtime state
    state: ExecutionState,
    /// Prompt templates
    prompts: PromptManager,
    /// Tool implementations
    tools: ToolRegistry,
    /// LLM client
    llm: Box<dyn LlmBackend>,
}

impl Interpreter {
    /// Load a scaffold file
    pub fn load(path: &Path) -> Result<Self>;

    /// Execute a task
    pub async fn run_task(&mut self, task: &str, input: Value) -> Result<Value>;

    /// Execute a tool
    pub async fn run_tool(&mut self, tool: &str, input: Value) -> Result<Value>;

    /// Hot-reload prompts
    pub fn reload_prompts(&mut self) -> Result<()>;
}
```

### 3. Prompt Manager

```rust
pub struct PromptManager {
    /// Template engine
    engine: minijinja::Environment<'static>,
    /// Loaded templates (name -> source)
    templates: HashMap<String, String>,
    /// Watch for changes
    watcher: Option<FileWatcher>,
}

impl PromptManager {
    /// Render a prompt with context
    pub fn render(&self, name: &str, ctx: &Value) -> Result<String>;

    /// Load prompts from directory
    pub fn load_dir(&mut self, path: &Path) -> Result<()>;

    /// Reload changed templates
    pub fn reload(&mut self) -> Result<()>;
}
```

### 4. Tool Execution

```rust
pub enum ToolImpl {
    /// Shell command with template
    Shell { command_template: String },
    /// LLM call with prompt template
    Llm { prompt_template: String },
    /// Foreign function (Rust)
    Foreign { module: String, function: String },
    /// Another tool
    ToolCall { tool: String },
    /// Sequence of operations
    Sequence { steps: Vec<ToolStep> },
    /// Conditional
    Conditional { condition: ExprIR, then: Box<ToolImpl>, else_: Option<Box<ToolImpl>> },
}
```

## Execution Flow

```
1. Load .scaffold file
   └─> Parse → Type Check → Lower to IR

2. Load prompts/*.jinja (if exists)
   └─> Register with template engine

3. Run task
   └─> Create execution context
   └─> Execute subgoals in order
       └─> Check preconditions
       └─> Select action (from LLM or policy)
       └─> Execute tool
           └─> Render templates with context
           └─> Run shell/LLM/foreign call
       └─> Check postconditions
       └─> Update state
   └─> Return output

4. Hot-reload (watch mode)
   └─> Detect file changes
   └─> Reload prompts (instant)
   └─> Reload .scaffold (re-parse, re-check)
```

## Prompt Template Convention

```
project/
├── agent.scaffold
└── prompts/
    ├── analyze_code.jinja
    ├── plan_task.jinja
    └── summarize.jinja
```

In scaffold file:
```scaffold
tool analyze_code {
    input: { code: string, question: string }
    output: string

    impl: llm("analyze_code")  // References prompts/analyze_code.jinja
}
```

In `prompts/analyze_code.jinja`:
```jinja
Analyze this code:
```{{ language }}
{{ code }}
```

Question: {{ question }}

{% if code | length > 1000 %}
Note: This is a large file. Focus on the main logic first.
{% endif %}

Provide a detailed analysis.
```

## CLI Commands

```bash
# Run a scaffold file (interpreted)
scaffold run agent.scaffold --task analyze_elf --input '{"path": "binary"}'

# Watch mode with hot-reload
scaffold watch agent.scaffold

# Type check only
scaffold check agent.scaffold

# Compile to Rust (for production)
scaffold codegen agent.scaffold -o ./generated

# REPL mode
scaffold repl agent.scaffold
```

## Implementation Order

1. **Phase 1: Basic Interpreter**
   - [ ] Extend Value enum in scaffold-runtime
   - [ ] Create scaffold-interpreter crate
   - [ ] Implement tool execution (shell, llm)
   - [ ] Implement basic task execution

2. **Phase 2: Prompt System**
   - [ ] Add minijinja to runtime
   - [ ] Implement PromptManager
   - [ ] Support external prompt files
   - [ ] Variable interpolation in shell/llm

3. **Phase 3: CLI Integration**
   - [ ] Add `scaffold run` command
   - [ ] Add `scaffold watch` command
   - [ ] Add `scaffold repl` command

4. **Phase 4: Hot-Reload**
   - [ ] File watching for prompts
   - [ ] File watching for .scaffold
   - [ ] Graceful reload without restart

5. **Phase 5: Production Features**
   - [ ] Async execution
   - [ ] Parallel tool execution
   - [ ] Caching
   - [ ] Telemetry/tracing