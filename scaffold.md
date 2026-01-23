# Scaffold Language Design Document

> **Version**: 2.0-draft
> **Last Updated**: 2025-01-21
> **Status**: Active Development

## Table of Contents

1. [Overview](#overview)
2. [Language Scope](#language-scope)
3. [Tool System](#tool-system)
4. [State & Effects](#state--effects)
5. [Action Selection & Harness](#action-selection--harness)
6. [Verification](#verification)
7. [IR & Codegen](#ir--codegen)
8. [Runtime Requirements](#runtime-requirements)
9. [Optimization & RL](#optimization--rl)
10. [Safety & Sandbox](#safety--sandbox)
11. [Developer Experience](#developer-experience)
12. [Roadmap](#roadmap)

---

## Overview

Scaffold is a domain-specific language for defining hierarchical task decomposition with:
- **Declarative task structure** - Define what, not how
- **Grounded actions** - Bind abstract actions to concrete implementations
- **Static verification** - Prove properties at compile time
- **Code generation** - Emit executable Rust (or Python) code
- **Optimization support** - Tune hyperparameters and prompts

### Current Implementation Status

| Component | Status | Location |
|-----------|--------|----------|
| Lexer/Parser | ✅ Complete | `crates/scaffold-syntax` |
| Type Checker | ✅ Complete | `crates/scaffold-types` |
| Verifier | ✅ Complete | `crates/scaffold-verify` |
| IR | ✅ Complete | `crates/scaffold-ir` |
| Runtime | ✅ Complete | `crates/scaffold-runtime` |
| Codegen | ✅ Complete | `crates/scaffold-codegen` |
| CLI | ✅ Complete | `crates/scaffold-cli` |

---

## Language Scope

### Core Constructs

The following constructs are required to generate working Rust code from DSL:

```scaffold
// 1. Type declarations
type Position = { x: int, y: int }
type Agent = { id: string, location: Position }

// 2. Function declarations (NEW - v2)
fn distance(a: Position, b: Position) -> float {
    sqrt((a.x - b.x)^2 + (a.y - b.y)^2)
}

// 3. Tool declarations (NEW - v2)
tool move_north() -> { success: bool, new_pos: Position }
tool pathfind(start: Position, goal: Position) -> { path: list<Position> }

// 4. Task declarations with explicit state
task navigate_to_target {
    input: { agent: Agent, target: Position }
    output: { success: bool, path: list<Position> }

    // NEW: Explicit state declaration
    state: {
        path: list<Position>?
        current_step: int
        path_complete: bool
    }

    decompose { ... }
    subgoal plan_path { ... }
    verify { ... }
    on_fail { ... }
}

// 5. Grounding declarations
grounding navigation_env {
    action move_north: tool("env.move(0, 1)")
    action astar: sandbox(python { ... })
}
```

### Identifier Resolution

**Configuration**: `--strict` mode (configurable)

| Mode | Unknown Identifier | Behavior |
|------|-------------------|----------|
| Default | Warning | Infer as `any`, track in IR for runtime binding |
| Strict | Error | Must be declared or in scope |

Unknown identifiers are tracked in IR:
```rust
pub struct ExprIR {
    // ...existing variants...
    UnresolvedIdent { name: String, inferred_type: TypeIR },
}
```

### Optional Subgoals

Syntax: `subgoal foo?` or `optional: true`

**Skip Conditions** (configurable per-subgoal):
- Precondition fails
- Timeout expires
- Action fails (non-critical)
- Custom expression evaluates to true

**Never skip when**:
- On critical path to exit node
- Explicitly marked `required: true`

**IR Representation**:
```rust
pub struct SubgoalIR {
    pub name: String,
    pub optional: bool,
    pub skip_conditions: Vec<SkipCondition>,
    // ...
}

pub enum SkipCondition {
    PreconditionFailed,
    Timeout,
    ActionFailed { action: String },
    Custom { expr: ExprIR },
}
```

### Pure Functions

First-class support for reusable pure computations:

```scaffold
// Declaration
fn path_length(path: list<Position>) -> int {
    len(path)
}

fn crash_quality(crashes: list<Crash>) -> float {
    sum(map(crashes, |c| c.severity)) / len(crashes)
}

// Usage in expressions
subgoal plan_path {
    reward: -path_length(state.path)
}
```

**Constraints**:
- No side effects
- No state mutations
- No tool/LLM calls
- Compiles directly to Rust functions

---

## Tool System

### Tool Signatures (v2)

Tools must declare typed signatures:

```scaffold
// Simple tool
tool check_syntax(code: string) -> { valid: bool, errors: list<string> }

// Tool with optional params
tool run_code(
    code: string,
    timeout: int = 5000,
    capture_stderr: bool = true
) -> {
    stdout: string,
    stderr: string,
    exit_code: int
}

// Tool that may fail
tool fetch_url(url: string) -> Result<{ body: string, status: int }>
```

### Error Modeling

Every tool returns a result-like structure:

```rust
// Runtime representation
pub struct ToolResult {
    pub ok: bool,
    pub value: Option<Value>,
    pub error: Option<ToolError>,
}

pub struct ToolError {
    pub code: String,
    pub message: String,
    pub details: Option<Value>,
}
```

DSL syntax for error handling:
```scaffold
subgoal fetch_data {
    let result = fetch_url(input.url)

    // Option 1: Propagate with ?
    let body = result?

    // Option 2: Explicit handling
    match result {
        Ok(data) => state.content = data.body
        Err(e) => state.error = e.message
    }
}
```

### Sandbox Configuration

Full sandbox control:

```scaffold
sandbox python {
    // Resource limits
    timeout: 5s
    memory: 256mb
    cpu_cores: 2
    gpu: false

    // Security
    network: false
    filesystem: readonly("/data")
    env: { PYTHONPATH: "/lib" }

    // Runtime
    runtime: "3.11"
    working_dir: "/workspace"
}

sandbox docker {
    image: "python:3.11-slim"
    timeout: 30s
    memory: 1gb
    network: false
    volumes: [
        "/data:/data:ro",
        "/output:/output:rw"
    ]
}
```

### Implementation Binding

Tools bind to runtime adapters only (v1):

```scaffold
grounding my_env {
    // Bind to shell command
    action check_syntax: tool("python -m py_compile {file}")

    // Bind to sandbox
    action run_tests: sandbox(python { timeout: 30s })

    // Bind to LLM
    action suggest_fix: llm("Given this error: {error}, suggest a fix")

    // Bind to HTTP endpoint (future)
    action fetch_data: http("GET", "https://api.example.com/data")
}
```

---

## State & Effects

### Explicit State Declaration

Tasks declare mutable state:

```scaffold
task exploit_finder {
    input: { binary: string, timeout: int }
    output: { exploits: list<Exploit>, success: bool }

    state: {
        // Mutable state variables
        crashes: list<Crash>
        coverage: CoverageMap
        current_input: string?
        iteration: int
        best_crash: Crash?
    }

    // State is accessible in all subgoals
    subgoal fuzz {
        done: len(state.crashes) > 10
        post: state.iteration > 0
    }
}
```

**Generated Rust**:
```rust
#[derive(Debug, Clone, Default)]
pub struct ExploitFinderState {
    pub crashes: Vec<Crash>,
    pub coverage: CoverageMap,
    pub current_input: Option<String>,
    pub iteration: i64,
    pub best_crash: Option<Crash>,
}
```

### Variable Binding

Explicit `let` syntax for binding results:

```scaffold
subgoal plan_path {
    pre: input.agent.location != input.target

    // Bind tool result to local variable
    let result = astar(input.agent.location, input.target)

    // Update state
    state.path = result.path
    state.path_length = len(result.path)

    done: state.path != null
    post: len(state.path) > 0
}
```

### Effects Clause

Model side effects for verification:

```scaffold
subgoal execute_moves {
    pre: state.path != null

    options: [move_north, move_south, move_east, move_west]

    // Declare effects on state
    effects: [
        state.current_step += 1,
        state.agent_pos = next_waypoint(state.path, state.current_step),
        state.path_complete = (state.current_step >= len(state.path))
    ]

    done: state.path_complete
    post: state.agent_pos == input.target
}
```

**Benefits**:
- Verifier can reason about state transitions
- Enables deadlock/livelock detection
- Supports rollback semantics

---

## Action Selection & Harness

### Policy Configuration

Default and per-task policies:

```scaffold
// Task-level policy
task code_generator {
    policy: llm("gpt-4o-mini")  // LLM selects actions
    // or: random              // Random selection
    // or: first               // Always first option (deterministic)
    // or: ucb(alpha=0.1)      // Upper confidence bound
    // or: bandit(epsilon=0.1) // Epsilon-greedy
}

// Subgoal-level override
subgoal explore {
    policy: random  // Override for exploration
    options: [fuzz, mutate, generate]
}
```

### Harness Declaration

First-class harness configuration:

```scaffold
harness code_gen {
    // Model configuration
    model: "gpt-4o"
    temperature: 0.7
    max_tokens: 4096

    // Tunable parameter space
    params: {
        temperature: [0.0, 0.2, 0.5, 0.8, 1.0]
        top_p: [0.9, 0.95, 1.0]
        system_prompt: [
            "You are a helpful coding assistant.",
            "You are an expert programmer. Be concise.",
        ]
    }

    // Tool configuration
    tools: [run_code, check_syntax, run_tests]

    // Prompts
    system_prompt: "You are an expert programmer."
    user_template: "Write code to: {task_description}"
}

harness fast_harness {
    model: "gpt-4o-mini"
    temperature: 0.0
    max_tokens: 1024
}
```

### Harness Binding

Task-level default with per-subgoal overrides:

```scaffold
task exploit_finder {
    harness: security_harness  // Default for all subgoals

    subgoal triage {
        harness: fast_harness  // Override: use cheaper model
    }

    subgoal exploit_dev {
        harness: code_gen  // Override: use code-focused harness
    }
}
```

### Metrics Configuration

Define optimization objectives:

```scaffold
task exploit_finder {
    metrics: {
        // Primary objective (required)
        primary: success_rate

        // Secondary objectives (optional)
        secondary: [
            latency,
            token_cost,
            crash_quality,  // Custom metric
        ]

        // Weights for multi-objective
        weights: {
            success_rate: 1.0,
            latency: -0.1,
            token_cost: -0.05,
        }
    }
}

// Custom metric function
fn crash_quality(state: TaskState) -> float {
    if len(state.crashes) == 0 { return 0.0 }

    let severity_sum = sum(map(state.crashes, |c| c.severity))
    let unique_paths = len(unique(map(state.crashes, |c| c.path)))

    (severity_sum / len(state.crashes)) * log(unique_paths + 1)
}
```

---

## Verification

### Check Categories

| Check | Default | Description |
|-------|---------|-------------|
| `grounded` | **Mandatory** | All actions have implementations |
| `no_deadlock` | **Mandatory** | No cycles without progress |
| `reachable(exit)` | **Mandatory** | Exit nodes reachable from entry |
| `bounded(subgoal, n)` | Opt-in | Subgoal terminates within n iterations |
| `terminates` | Opt-in | Task always eventually completes |
| `type_safe` | **Mandatory** | All expressions type-check |

### Verification Syntax

```scaffold
task navigate_to_target {
    // ... task body ...

    verify {
        // Mandatory (always checked)
        grounded(options)
        no_deadlock(decompose)
        reachable(verify_arrival.done)

        // Opt-in bounds
        bounded(execute_moves, 1000)
        bounded(plan_path, 100)

        // Opt-in termination
        terminates(timeout: 60s)

        // Custom invariants
        invariant: state.current_step >= 0
        invariant: state.current_step <= len(state.path)
    }
}
```

### Solver Scope

**Phase 1** (Current):
- Propositional logic
- Graph analysis (reachability, cycles)

**Phase 2** (v2):
- QF_LIA (Quantifier-Free Linear Integer Arithmetic)
- Z3 via `z3-sys` crate
- Bounds checking
- Simple invariants

**Phase 3** (Future):
- Arrays and sequences
- Uninterpreted functions
- Non-linear arithmetic (limited)

### Boundedness Proof Strategies

1. **User-declared bounds**:
   ```scaffold
   verify { bounded(fuzz_loop, 10000) }
   ```

2. **Timeout-derived bounds**:
   ```
   bound = timeout_ms / min_action_time_ms
   ```

3. **Ranking functions** (future):
   ```scaffold
   verify {
       bounded(search, ranking: len(unexplored))
   }
   ```

### Verification Artifacts

Extended IR for verification results:

```rust
pub struct VerifyResultIR {
    pub check: String,
    pub args: Vec<String>,
    pub passed: bool,
    pub message: Option<String>,

    // NEW: Diagnostic artifacts
    pub counterexample: Option<CounterexampleIR>,
    pub unsat_core: Option<Vec<String>>,
    pub proof_hint: Option<String>,
}

pub struct CounterexampleIR {
    pub trace: Vec<TraceStep>,
    pub final_state: HashMap<String, Value>,
    pub violated_property: String,
}
```

---

## IR & Codegen

### IR Structure (v2)

Extended IR for new features:

```rust
pub struct ScaffoldIR {
    pub version: String,
    pub tasks: Vec<TaskIR>,
    pub types: Vec<TypeDefIR>,
    pub groundings: Vec<GroundingIR>,

    // NEW
    pub functions: Vec<FunctionIR>,
    pub tools: Vec<ToolDeclIR>,
    pub harnesses: Vec<HarnessIR>,
}

pub struct TaskIR {
    pub name: String,
    pub input_type: TypeRefIR,
    pub output_type: TypeRefIR,
    pub state_type: Option<TypeRefIR>,     // NEW
    pub decomposition: DecompGraphIR,
    pub subgoals: Vec<SubgoalIR>,
    pub verification: Vec<VerifyResultIR>,
    pub failure_strategy: FailStrategyIR,
    pub harness: Option<String>,            // NEW
    pub policy: Option<PolicyIR>,           // NEW
    pub metrics: Vec<MetricIR>,             // NEW
    pub source_span: SourceSpanIR,
}

pub struct SubgoalIR {
    pub name: String,
    pub precondition: Option<ExprIR>,
    pub postcondition: Option<ExprIR>,
    pub termination: Option<ExprIR>,
    pub actions: Vec<String>,
    pub bindings: Vec<BindingIR>,           // NEW
    pub effects: Vec<EffectIR>,             // NEW
    pub reward: Option<ExprIR>,
    pub timeout_ms: Option<u64>,
    pub optional: bool,
    pub skip_conditions: Vec<SkipCondition>, // NEW
    pub harness_override: Option<String>,    // NEW
    pub policy_override: Option<PolicyIR>,   // NEW
}

pub struct FunctionIR {
    pub name: String,
    pub params: Vec<(String, TypeIR)>,
    pub return_type: TypeIR,
    pub body: ExprIR,
}

pub struct ToolDeclIR {
    pub name: String,
    pub params: Vec<ToolParamIR>,
    pub return_type: TypeIR,
    pub may_fail: bool,
}

pub struct HarnessIR {
    pub name: String,
    pub model: String,
    pub params: HashMap<String, ParamSpaceIR>,
    pub tools: Vec<String>,
    pub system_prompt: Option<String>,
}
```

### Expression Compilation

Expressions compile directly to Rust (no interpreter):

```scaffold
// DSL
fn distance(a: Position, b: Position) -> float {
    sqrt((a.x - b.x)^2 + (a.y - b.y)^2)
}
```

```rust
// Generated Rust
pub fn distance(a: &Position, b: &Position) -> f64 {
    ((a.x - b.x).pow(2) + (a.y - b.y).pow(2) as f64).sqrt()
}
```

### Backend Targets

| Target | Status | Priority |
|--------|--------|----------|
| Rust | ✅ v1 | Primary |
| Python | 🔄 v2 | Secondary |
| TypeScript | 📋 v3 | Future |

IR is designed to be backend-agnostic:
- No Rust-specific constructs in IR
- Type system maps to multiple targets
- Codegen is pluggable via `CodeGenerator` trait

---

## Runtime Requirements

### Core Traits

```rust
//! scaffold-runtime/src/traits.rs

/// Core task execution
pub trait Task {
    type Input: Clone;
    type Output: Clone;
    type State: Default + Clone;

    fn execute(&mut self, ctx: &mut TaskContext, input: Self::Input) -> Result<Self::Output>;
    fn failure_strategy(&self) -> FailureStrategy;
    fn name(&self) -> &str;
}

/// Action dispatch
pub trait ActionDispatcher {
    fn dispatch(&self, action: &str, ctx: &mut TaskContext) -> Result<ActionResult>;
    fn available_actions(&self) -> &[&str];
}

/// Tool execution
pub trait ToolRunner {
    fn run(&self, name: &str, args: &Value) -> Result<ToolResult>;
    fn signature(&self, name: &str) -> Option<&ToolSignature>;
    fn available_tools(&self) -> &[&str];
}

/// LLM integration
pub trait LlmClient: Send + Sync {
    fn complete(
        &self,
        prompt: &str,
        config: &LlmConfig,
    ) -> impl Future<Output = Result<LlmResponse>>;

    fn complete_with_tools(
        &self,
        prompt: &str,
        tools: &[ToolSignature],
        config: &LlmConfig,
    ) -> impl Future<Output = Result<ToolCallResponse>>;
}

/// Sandbox execution
pub trait SandboxRunner {
    fn run(
        &self,
        config: &SandboxConfig,
        code: &str,
    ) -> Result<SandboxResult>;

    fn run_with_files(
        &self,
        config: &SandboxConfig,
        code: &str,
        files: &[(String, Vec<u8>)],
    ) -> Result<SandboxResult>;
}

/// Observability
pub trait Tracer {
    fn span(&self, name: &str) -> SpanGuard;
    fn event(&self, event: TraceEvent);
    fn set_attribute(&self, key: &str, value: impl Into<Value>);
}

/// Secret management
pub trait SecretStore {
    fn get(&self, key: &str) -> Option<SecretString>;
    fn get_env(&self, name: &str) -> Option<SecretString>;
}
```

### Scheduling Semantics

| Mode | Description | Use Case |
|------|-------------|----------|
| Serial | Subgoals execute sequentially | Default, simplest |
| Async | Subgoals can await I/O | Network/LLM calls |
| Parallel | Independent subgoals run concurrently | Performance |

```scaffold
task parallel_search {
    mode: parallel  // Enable parallel execution

    decompose {
        // These run in parallel
        search_a, search_b, search_c -> merge_results
    }
}
```

### Retry and Recovery

```scaffold
task robust_task {
    subgoal flaky_operation {
        retry: {
            max_attempts: 3
            backoff: exponential(base: 1s, max: 30s)
            retry_on: [Timeout, ActionFailed]
        }
    }

    on_fail {
        // Strategy options:
        retry(3)              // Retry current subgoal
        rollback(checkpoint)  // Restore to checkpoint
        replan               // Request new decomposition
        abort                // Fail the task
    }
}
```

### Observability

```rust
/// Structured execution trace
pub struct ExecutionTrace {
    pub task_name: String,
    pub trace_id: String,
    pub parent_trace_id: Option<String>,

    pub subgoals: Vec<SubgoalTrace>,
    pub actions: Vec<ActionTrace>,
    pub llm_calls: Vec<LlmCallTrace>,

    pub metrics: ExecutionMetrics,
    pub outcome: TaskOutcome,
}

pub struct ExecutionMetrics {
    pub wall_time_ms: u64,
    pub cpu_time_ms: u64,
    pub total_tokens: u64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub estimated_cost_usd: f64,
    pub action_count: u64,
    pub retry_count: u64,
}

pub struct SubgoalTrace {
    pub name: String,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub outcome: SubgoalOutcome,
    pub state_before: Value,
    pub state_after: Value,
}
```

### Resource Budgets

```scaffold
task expensive_task {
    budget: {
        max_tokens: 100000       // Total LLM tokens
        max_cost_usd: 5.00      // Estimated cost
        max_time: 10m           // Wall-clock time
        max_actions: 1000       // Action count
        max_retries: 10         // Total retries
    }
}
```

Runtime enforcement:
```rust
impl TaskContext {
    pub fn check_budget(&self) -> Result<()> {
        if self.metrics.total_tokens > self.budget.max_tokens {
            return Err(Error::BudgetExceeded("tokens"));
        }
        // ... other checks
        Ok(())
    }
}
```

---

## Optimization & RL

### Optimization Syntax

```scaffold
optimize task code_generator {
    // Search space (from harness params)
    harness: code_gen

    // Objective
    objective: maximize(success_rate)

    // Constraints
    constraints: {
        latency < 5s
        token_cost < 0.10
    }

    // Budget
    budget: {
        max_trials: 100
        max_time: 2h
        max_tokens: 1000000
    }

    // Early stopping
    early_stop: {
        patience: 10           // Trials without improvement
        min_delta: 0.01        // Minimum improvement threshold
    }

    // Algorithm
    algorithm: tpe            // or: random, grid, cma_es, ucb
}
```

### Supported Algorithms

| Algorithm | Description | Best For |
|-----------|-------------|----------|
| `grid` | Exhaustive grid search | Small discrete spaces |
| `random` | Random sampling | Baseline, high-dim spaces |
| `tpe` | Tree-Parzen Estimator | General purpose, default |
| `cma_es` | Covariance Matrix Adaptation | Continuous params |
| `ucb` | Upper Confidence Bound | Online optimization |

### Budget Types

```rust
pub struct OptimizationBudget {
    pub max_trials: Option<u64>,
    pub max_time: Option<Duration>,
    pub max_tokens: Option<u64>,
    pub max_cost_usd: Option<f64>,
}

pub struct EarlyStopConfig {
    pub patience: u64,
    pub min_delta: f64,
    pub min_trials: u64,
}
```

### Trace Logging for Offline RL

```rust
/// RL-compatible trace format
pub struct RLTrace {
    pub episode_id: String,
    pub task_name: String,
    pub timestamp: DateTime<Utc>,

    // MDP components
    pub states: Vec<StateSnapshot>,
    pub actions: Vec<ActionTaken>,
    pub rewards: Vec<f64>,
    pub terminal: bool,

    // Metadata
    pub config: HashMap<String, Value>,
    pub metrics: ExecutionMetrics,

    // Privacy controls
    pub redaction_level: RedactionLevel,
}

pub enum RedactionLevel {
    None,                    // Full data
    RedactSecrets,          // Remove API keys, passwords
    RedactPrompts,          // Remove prompt contents
    RedactAll,              // Only structure, no content
}
```

Storage formats:
- JSON Lines (`.jsonl`) - Human readable, streaming
- Parquet - Columnar, efficient for analysis
- SQLite - Queryable, single file

### Optimization Artifacts

```rust
pub struct OptimizationResult {
    pub best_config: HashMap<String, Value>,
    pub best_score: f64,
    pub all_trials: Vec<TrialResult>,

    // Exportable artifacts
    pub artifacts: OptimizationArtifacts,
}

pub struct OptimizationArtifacts {
    pub best_prompts: Option<Vec<String>>,
    pub few_shot_examples: Option<Vec<Example>>,
    pub param_importance: HashMap<String, f64>,
    pub convergence_curve: Vec<(u64, f64)>,
}
```

---

## Safety & Sandbox

### Threat Model

| Component | Trust Level | Mitigations |
|-----------|-------------|-------------|
| DSL source | Trusted | Author-controlled |
| Tool commands | **Untrusted** | Escape, validate, sandbox |
| LLM outputs | **Untrusted** | Parse, validate, sandbox |
| Generated code | **Untrusted** | Sandbox execution |
| User prompts | Semi-trusted | Template injection prevention |

### Secure Defaults

```rust
pub const SANDBOX_DEFAULTS: SandboxConfig = SandboxConfig {
    // Network: OFF by default
    network: NetworkAccess::None,

    // Filesystem: None by default
    filesystem: FilesystemAccess::None,

    // Resources: Conservative limits
    timeout_ms: 30_000,
    memory_bytes: 256 * 1024 * 1024,  // 256 MB
    cpu_cores: 1,

    // Environment: Isolated
    env_inherit: false,
    env_vars: HashMap::new(),

    // Execution: Restricted
    allow_exec: false,
    allowlist_commands: vec![],
};
```

Explicit opt-in for dangerous operations:
```scaffold
sandbox python {
    network: allow(["api.openai.com", "pypi.org"])  // Explicit allowlist
    filesystem: readwrite("/workspace")              // Explicit path
    env_inherit: true                                // Explicit flag
}
```

### Reproducibility

```scaffold
task deterministic_task {
    reproducibility: {
        seed: 42                    // Fixed seed
        llm_seed: true             // Pass seed to LLM API
        record_nondeterminism: true // Log any non-deterministic events
    }
}
```

Runtime support:
```rust
pub struct ReproducibilityConfig {
    pub seed: u64,
    pub llm_seed: bool,
    pub sandbox_seed: bool,
    pub record_timestamps: bool,
}

impl TaskContext {
    pub fn rng(&self) -> impl Rng {
        StdRng::seed_from_u64(self.config.seed)
    }
}
```

### Secret Management

```scaffold
task api_task {
    secrets: {
        openai_key: env("OPENAI_API_KEY")
        db_password: vault("prod/database/password")
        api_token: file("/run/secrets/api_token")
    }
}
```

Runtime trait:
```rust
pub trait SecretStore {
    fn get(&self, key: &str) -> Option<SecretString>;
    fn get_env(&self, name: &str) -> Option<SecretString>;
    fn get_vault(&self, path: &str) -> Result<SecretString>;
    fn get_file(&self, path: &str) -> Result<SecretString>;
}

/// Secret string that redacts in logs
pub struct SecretString(String);

impl Debug for SecretString {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "[REDACTED]")
    }
}
```

---

## Developer Experience

### CLI Commands

```bash
# Validation
scaffold check FILE              # Parse, type-check, verify
scaffold check FILE --strict     # Strict mode (errors on unknown idents)
scaffold check FILE --warn-unknown # Explicit warning mode

# Compilation
scaffold compile FILE            # Compile to IR (JSON)
scaffold compile FILE -o OUT     # Output to file
scaffold compile FILE --compact  # Compact JSON

# Code Generation
scaffold codegen FILE -o DIR     # Generate Rust code
scaffold codegen FILE -o DIR --format  # With rustfmt
scaffold codegen FILE -o DIR --target python  # Future: Python

# Execution (NEW)
scaffold run FILE                # Execute task
scaffold run FILE --input data.json
scaffold run FILE --trace        # Enable tracing
scaffold run FILE --dry-run      # Validate without executing

# Optimization (NEW)
scaffold optimize FILE           # Run optimization
scaffold optimize FILE --trials 100
scaffold optimize FILE --algorithm tpe
scaffold optimize FILE --output results/

# Formatting (NEW)
scaffold fmt FILE                # Format in place
scaffold fmt FILE --check        # Check formatting

# Documentation (NEW)
scaffold doc FILE                # Generate markdown docs
scaffold doc FILE -o docs/
```

### LSP Server

**Features**:
- Diagnostics (errors, warnings)
- Completion (keywords, types, actions)
- Hover (type info, docs)
- Go-to-definition
- Find references
- Rename symbol
- Code actions (quick fixes)

**Configuration**:
```json
{
  "scaffold.strictMode": false,
  "scaffold.warnUnknown": true,
  "scaffold.formatOnSave": true,
  "scaffold.verifyOnSave": true
}
```

### Testing Strategy

1. **Unit Tests**: Per-crate functionality
   ```rust
   #[test]
   fn test_parse_task() { ... }
   ```

2. **Golden IR Tests**: Serialization stability
   ```
   tests/golden/
   ├── navigate.scaffold
   ├── navigate.ir.json     # Expected IR
   └── navigate.rs          # Expected generated code
   ```

3. **E2E Execution Tests**: With mock tools
   ```rust
   #[test]
   fn test_navigate_execution() {
       let mock_env = MockNavigationEnv::new();
       let task = NavigateToTargetTask::new(mock_env);
       let result = task.execute(&mut ctx, input);
       assert!(result.is_ok());
   }
   ```

4. **Property Tests**: Fuzzing with proptest
   ```rust
   proptest! {
       #[test]
       fn parse_roundtrip(source in scaffold_source()) {
           let ast = parse(&source)?;
           let reparsed = parse(&format(&ast))?;
           assert_eq!(ast, reparsed);
       }
   }
   ```

5. **Dataset-backed Eval**: For optimization
   ```
   eval/
   ├── code_gen/
   │   ├── dataset.jsonl    # Test cases
   │   ├── baseline.json    # Baseline results
   │   └── eval.py          # Evaluation script
   ```

---

## Roadmap

### Phase 1: Core Codegen (✅ Complete)

- [x] Lexer and parser
- [x] Type system
- [x] Verification (basic)
- [x] IR definition
- [x] Runtime traits
- [x] Rust code generation
- [x] CLI (check, compile, codegen)

### Phase 2: Language Extensions (v1.1)

**Duration**: 2-3 weeks

- [ ] Explicit `state` blocks in tasks
- [ ] `let` bindings in subgoals
- [ ] Typed tool declarations
- [ ] Pure function declarations (`fn`)
- [ ] Effects clause
- [ ] Optional subgoal skip conditions

**Deliverables**:
- Extended grammar and parser
- Updated IR structures
- Enhanced code generation
- Updated documentation

### Phase 3: Runtime Implementation (v1.2)

**Duration**: 2-3 weeks

- [ ] `ToolRunner` trait + implementations
- [ ] `LlmClient` trait + OpenAI/Anthropic
- [ ] `SandboxRunner` trait + Docker/gVisor
- [ ] `Tracer` trait + tracing integration
- [ ] `scaffold run` command
- [ ] Basic observability

**Deliverables**:
- Runtime crate with trait implementations
- Integration with external services
- Execution traces and metrics

### Phase 4: Optimization & Harness (v2.0)

**Duration**: 4-6 weeks

- [ ] Harness declarations
- [ ] Metrics and objectives
- [ ] Optimization algorithms (grid, random, TPE)
- [ ] Budget enforcement
- [ ] `scaffold optimize` command
- [ ] Artifact export

**Deliverables**:
- Optimization framework
- Harness configuration
- Results visualization

### Phase 5: Advanced Verification (v2.1)

**Duration**: 4-6 weeks

- [ ] Z3 integration via `z3-sys`
- [ ] QF_LIA constraint solving
- [ ] Invariant checking
- [ ] Counterexample generation
- [ ] Bounded model checking

**Deliverables**:
- SMT-backed verification
- Better error messages
- Proof artifacts

### Phase 6: Ecosystem (v3.0)

**Duration**: Ongoing

- [ ] LSP server
- [ ] VS Code extension
- [ ] Python codegen backend
- [ ] Package registry
- [ ] Standard library
- [ ] Community groundings

---

## Appendix A: Grammar (EBNF)

```ebnf
program     = { declaration } ;
declaration = type_decl | fn_decl | tool_decl | task_decl | grounding_decl ;

type_decl   = "type" IDENT "=" type_expr ;
type_expr   = primitive | struct_type | list_type | map_type | option_type | IDENT ;
primitive   = "bool" | "int" | "float" | "string" | "any" ;
struct_type = "{" [ field { "," field } ] "}" ;
field       = IDENT ":" type_expr ;
list_type   = "list" "<" type_expr ">" ;
map_type    = "map" "<" type_expr "," type_expr ">" ;
option_type = type_expr "?" ;

fn_decl     = "fn" IDENT "(" [ param { "," param } ] ")" "->" type_expr block ;
param       = IDENT ":" type_expr [ "=" expr ] ;
block       = "{" expr "}" ;

tool_decl   = "tool" IDENT "(" [ param { "," param } ] ")" "->" type_expr ;

task_decl   = "task" IDENT "{" task_body "}" ;
task_body   = { task_item } ;
task_item   = input_decl | output_decl | state_decl | decompose_decl
            | subgoal_decl | verify_decl | on_fail_decl
            | harness_decl | policy_decl | metrics_decl | budget_decl ;

input_decl  = "input" ":" type_expr ;
output_decl = "output" ":" type_expr ;
state_decl  = "state" ":" struct_type ;

decompose_decl = "decompose" "{" decompose_chain "}" ;
decompose_chain = IDENT { "->" IDENT } { "," decompose_chain } ;

subgoal_decl = "subgoal" IDENT [ "?" ] "{" subgoal_body "}" ;
subgoal_body = { subgoal_item } ;
subgoal_item = pre_clause | post_clause | done_clause | options_clause
             | reward_clause | timeout_clause | let_binding | effects_clause
             | retry_clause | harness_clause | policy_clause ;

pre_clause     = "pre" ":" expr ;
post_clause    = "post" ":" expr ;
done_clause    = "done" ":" expr ;
options_clause = "options" ":" "[" IDENT { "," IDENT } "]" ;
reward_clause  = "reward" ":" expr ;
timeout_clause = "timeout" ":" duration ;
let_binding    = "let" IDENT "=" expr ;
effects_clause = "effects" ":" "[" effect { "," effect } "]" ;
effect         = IDENT "=" expr | IDENT "+=" expr | IDENT "-=" expr ;

verify_decl  = "verify" "{" { verify_stmt } "}" ;
verify_stmt  = IDENT "(" [ expr { "," expr } ] ")" ;

on_fail_decl = "on_fail" "{" fail_strategy "}" ;
fail_strategy = "retry" "(" INT ")" | "rollback" "(" IDENT ")" | "replan" | "abort" ;

grounding_decl = "grounding" IDENT "{" { action_binding } "}" ;
action_binding = "action" IDENT ":" grounding_target ;
grounding_target = tool_target | llm_target | sandbox_target ;
tool_target    = "tool" "(" STRING ")" ;
llm_target     = "llm" "(" STRING ")" ;
sandbox_target = "sandbox" "(" sandbox_config ")" ;
sandbox_config = IDENT "{" { config_item } "}" ;
config_item    = IDENT ":" expr ;

expr = literal | IDENT | field_access | binary_op | unary_op | call | paren ;
literal = INT | FLOAT | STRING | BOOL | "null" ;
field_access = expr "." IDENT ;
binary_op = expr OP expr ;
unary_op = OP expr ;
call = IDENT "(" [ expr { "," expr } ] ")" ;
paren = "(" expr ")" ;

duration = INT TIME_UNIT ;
TIME_UNIT = "ms" | "s" | "m" | "h" ;
```

---

## Appendix B: Type Mapping

| Scaffold Type | Rust Type | Python Type | Notes |
|---------------|-----------|-------------|-------|
| `bool` | `bool` | `bool` | |
| `int` | `i64` | `int` | |
| `float` | `f64` | `float` | |
| `string` | `String` | `str` | |
| `any` | `Value` | `Any` | Dynamic type |
| `list<T>` | `Vec<T>` | `List[T]` | |
| `map<K,V>` | `HashMap<K,V>` | `Dict[K,V]` | |
| `T?` | `Option<T>` | `Optional[T]` | Nullable |
| `{ fields }` | `struct` | `@dataclass` | Named struct generated |
| `Result<T>` | `Result<T, Error>` | `Result[T, E]` | Error handling |

---

## Appendix C: Built-in Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `len(x)` | `list<T> -> int` | List length |
| `is_empty(x)` | `list<T> -> bool` | Check if empty |
| `contains(list, item)` | `(list<T>, T) -> bool` | Contains check |
| `map(list, fn)` | `(list<T>, T->U) -> list<U>` | Transform list |
| `filter(list, fn)` | `(list<T>, T->bool) -> list<T>` | Filter list |
| `sum(list)` | `list<int\|float> -> int\|float` | Sum elements |
| `min(a, b)` | `(T, T) -> T` | Minimum |
| `max(a, b)` | `(T, T) -> T` | Maximum |
| `abs(x)` | `int\|float -> int\|float` | Absolute value |
| `sqrt(x)` | `float -> float` | Square root |
| `log(x)` | `float -> float` | Natural log |
| `is_some(x)` | `T? -> bool` | Check if Some |
| `is_none(x)` | `T? -> bool` | Check if None |
| `unwrap(x)` | `T? -> T` | Unwrap or panic |
| `unwrap_or(x, default)` | `(T?, T) -> T` | Unwrap with default |
