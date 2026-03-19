//! Intermediate Representation (IR) definitions for the Scaffold DSL
//!
//! The IR is designed to be:
//! - Serializable: JSON/binary format for storage and transmission
//! - Executable: Sufficient info for a runtime to execute
//! - Debuggable: Preserve source locations for error messages

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Current IR version
pub const IR_VERSION: &str = "0.1.0";

/// Top-level scaffold IR
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScaffoldIR {
    /// IR version for compatibility checking
    pub version: String,
    /// Type definitions
    #[serde(default)]
    pub types: Vec<TypeDefIR>,
    /// External crate declarations
    #[serde(default)]
    pub extern_crates: Vec<ExternCrateIR>,
    /// Foreign module declarations
    #[serde(default)]
    pub foreign_modules: Vec<ForeignModuleIR>,
    /// Tool definitions (deterministic code)
    #[serde(default)]
    pub tools: Vec<ToolIR>,
    /// Prompt definitions (single LLM calls)
    #[serde(default)]
    pub prompts: Vec<PromptIR>,
    /// Agent definitions (multi-turn LLM with tools)
    #[serde(default)]
    pub agents: Vec<AgentIR>,
    /// Pipeline definitions (fixed sequences)
    #[serde(default)]
    pub pipelines: Vec<PipelineIR>,
    /// Task definitions (typed orchestration units)
    #[serde(default)]
    pub tasks: Vec<TaskIR>,
    /// Harness definitions (typed execution overlays)
    #[serde(default)]
    pub harnesses: Vec<HarnessIR>,
    /// Objective definitions (evaluation and optimization contracts)
    #[serde(default)]
    pub objectives: Vec<ObjectiveIR>,
}

impl ScaffoldIR {
    pub fn new() -> Self {
        Self {
            version: IR_VERSION.to_string(),
            types: Vec::new(),
            extern_crates: Vec::new(),
            foreign_modules: Vec::new(),
            tools: Vec::new(),
            prompts: Vec::new(),
            agents: Vec::new(),
            pipelines: Vec::new(),
            tasks: Vec::new(),
            harnesses: Vec::new(),
            objectives: Vec::new(),
        }
    }
}

impl Default for ScaffoldIR {
    fn default() -> Self {
        Self::new()
    }
}

/// Type definition IR
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeDefIR {
    /// Whether this is a general type or a task-flow artifact
    #[serde(default)]
    pub kind: TypeDefKindIR,
    /// Type name
    pub name: String,
    /// Type structure
    pub definition: TypeIR,
}

/// Distinguishes plain types from artifacts in IR
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum TypeDefKindIR {
    #[default]
    Type,
    Artifact,
}

/// Type IR
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum TypeIR {
    Bool,
    Int,
    Float,
    String,
    Bytes,
    Any,
    List {
        element: Box<TypeIR>,
    },
    Map {
        key: Box<TypeIR>,
        value: Box<TypeIR>,
    },
    Option {
        inner: Box<TypeIR>,
    },
    Result {
        ok: Box<TypeIR>,
        err: Box<TypeIR>,
    },
    Struct {
        fields: HashMap<String, TypeIR>,
    },
    Named {
        name: String,
    },
}

/// Type reference IR (for input/output types)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TypeRefIR {
    Named { ref_name: String },
    Inline(TypeIR),
}

/// Expression IR
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum ExprIR {
    Literal {
        value: LiteralIR,
    },
    Ident {
        name: String,
    },
    FieldAccess {
        base: Box<ExprIR>,
        field: String,
    },
    Binary {
        left: Box<ExprIR>,
        op: String,
        right: Box<ExprIR>,
    },
    Call {
        function: String,
        args: Vec<ExprIR>,
    },
    ForeignCall {
        module: String,
        function: String,
        args: Vec<ExprIR>,
    },
    List {
        elements: Vec<ExprIR>,
    },
    Record {
        fields: Vec<ExprFieldIR>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExprFieldIR {
    pub key: String,
    pub value: ExprIR,
}

/// Literal value IR
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum LiteralIR {
    Int { value: i64 },
    Float { value: f64 },
    String { value: String },
    Bool { value: bool },
    Null,
}

// ============================================
// Foreign and Tool IR
// ============================================

/// External crate declaration IR
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternCrateIR {
    /// Crate name
    pub name: String,
    /// Version requirement
    pub version: String,
    /// Optional features
    pub features: Vec<String>,
}

/// Foreign module declaration IR
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForeignModuleIR {
    /// Language (e.g., "rust")
    pub language: String,
    /// Module name
    pub name: String,
    /// Type aliases
    pub type_aliases: Vec<ForeignTypeAliasIR>,
    /// Function declarations
    pub functions: Vec<ForeignFnIR>,
}

/// Foreign type alias IR
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForeignTypeAliasIR {
    /// Scaffold type name
    pub name: String,
    /// External type path
    pub external_type: String,
}

/// Foreign function declaration IR
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForeignFnIR {
    /// Function name
    pub name: String,
    /// Parameters
    pub params: Vec<ForeignParamIR>,
    /// Return type
    pub return_type: TypeIR,
}

/// Foreign function parameter IR
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForeignParamIR {
    /// Parameter name
    pub name: String,
    /// Parameter type
    pub ty: TypeIR,
}

/// Tool definition IR
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolIR {
    /// Tool name
    pub name: String,
    /// Input type
    pub input: TypeIR,
    /// Output type
    pub output: TypeIR,
    /// Tool implementation
    pub implementation: Option<ToolImplIR>,
    /// Tool specification
    pub spec: Option<ToolSpecIR>,
    /// Implementation variants
    pub variants: Vec<ToolVariantIR>,
}

/// Tool implementation IR
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum ToolImplIR {
    /// Simple expression
    Expr { expr: ToolExprIR },
    /// Sequence of operations
    Sequence { statements: Vec<ToolStatementIR> },
    /// Parallel operations
    Parallel { statements: Vec<ToolStatementIR> },
}

/// Tool expression IR
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum ToolExprIR {
    /// Variable reference
    Ident { name: String },
    /// Field access
    FieldAccess {
        base: Box<ToolExprIR>,
        field: String,
    },
    /// Foreign function call
    ForeignCall {
        module: String,
        function: String,
        args: Vec<ToolExprIR>,
    },
    /// Local tool call
    ToolCall { tool: String, args: Vec<ToolExprIR> },
    /// Shell command
    Shell { command: String },
    /// Pipe expression
    Pipe {
        left: Box<ToolExprIR>,
        right: Box<ToolExprIR>,
    },
    /// Conditional
    If {
        condition: ExprIR,
        then_branch: Box<ToolImplIR>,
        else_branch: Option<Box<ToolImplIR>>,
    },
    /// Match expression
    Match {
        scrutinee: Box<ToolExprIR>,
        arms: Vec<MatchArmIR>,
    },
    /// For loop
    For {
        variable: String,
        iterable: Box<ToolExprIR>,
        body: Box<ToolImplIR>,
    },
    /// While loop
    While {
        condition: ExprIR,
        body: Box<ToolImplIR>,
    },
    /// Infinite loop
    Loop { body: Box<ToolImplIR> },
    /// Break out of loop
    Break,
    /// Continue to next iteration
    Continue,
    /// Literal value
    Literal { value: LiteralIR },
    /// Map/JSON literal
    MapLiteral { entries: Vec<MapEntryIR> },
    /// General expression (arithmetic, comparisons, etc.)
    Expr { expr: Box<ExprIR> },
}

/// Tool statement IR (for sequence/parallel blocks)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolStatementIR {
    /// Optional binding name
    pub binding: Option<String>,
    /// The expression
    pub expr: ToolExprIR,
}

/// Map/json literal entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MapEntryIR {
    pub key: String,
    pub value: ToolExprIR,
}

/// Match arm IR
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchArmIR {
    /// Pattern (as expression)
    pub pattern: ExprIR,
    /// Body
    pub body: ToolImplIR,
}

/// Tool specification IR
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSpecIR {
    /// Preconditions
    pub preconditions: Vec<ExprIR>,
    /// Postconditions
    pub postconditions: Vec<ExprIR>,
    /// Whether the tool is pure
    pub pure: bool,
}

/// Tool variant IR
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolVariantIR {
    /// Variant name
    pub name: String,
    /// Implementation
    pub implementation: ToolImplIR,
}

/// Source span IR for debugging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceSpanIR {
    /// Start byte offset
    pub start: usize,
    /// End byte offset
    pub end: usize,
    /// Source file (if known)
    pub file: Option<String>,
}

impl SourceSpanIR {
    pub fn new(start: usize, end: usize) -> Self {
        Self {
            start,
            end,
            file: None,
        }
    }

    pub fn with_file(mut self, file: String) -> Self {
        self.file = Some(file);
        self
    }
}

// ============================================
// Prompt IR (Single LLM Call)
// ============================================

/// Prompt definition IR - single LLM call with typed I/O
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptIR {
    /// Prompt name
    pub name: String,
    /// Input type
    pub input: TypeIR,
    /// Output type
    pub output: TypeIR,
    /// Template string (with {var} interpolation)
    pub template: StringOrFileIR,
    /// Optional system prompt
    pub system: Option<StringOrFileIR>,
}

/// String literal or file reference
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum StringOrFileIR {
    /// Inline string
    Literal { value: String },
    /// File reference
    File { path: String },
}

// ============================================
// Agent IR (Multi-turn LLM)
// ============================================

/// Error handling strategy for agents
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum ErrorStrategyIR {
    /// Fail immediately on error (default)
    Abort,
    /// Retry up to N times before failing
    Retry { count: u64 },
}

impl Default for ErrorStrategyIR {
    fn default() -> Self {
        ErrorStrategyIR::Abort
    }
}

/// Agent definition IR - multi-turn LLM with tool access
/// In the RL optimization context, agents act as subgoals with process rewards
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentIR {
    /// Agent name
    pub name: String,
    /// Input type
    pub input: TypeIR,
    /// Output type
    pub output: TypeIR,
    /// Available tools (list of tool names)
    pub tools: Vec<String>,
    /// System prompt (defines agent behavior)
    pub system: StringOrFileIR,
    /// Model to use (e.g., "gpt-4o", "claude-sonnet-4-20250514")
    pub model: Option<String>,
    /// Maximum turns before termination
    pub max_turns: Option<u64>,
    /// Process reward expression (for RL optimization)
    pub reward: Option<ExprIR>,
    /// Explicit termination condition (beyond max_turns)
    pub done: Option<ExprIR>,
    /// Error handling strategy
    #[serde(default)]
    pub on_error: ErrorStrategyIR,
    /// Execution timeout in seconds
    pub timeout: Option<u64>,
}

// ============================================
// Pipeline IR (Fixed Sequence)
// ============================================

/// Pipeline definition IR - fixed sequence of prompts/tools
/// In the RL optimization context, pipelines represent task decomposition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineIR {
    /// Pipeline name
    pub name: String,
    /// Input type
    pub input: TypeIR,
    /// Output type
    pub output: TypeIR,
    /// Sequence of steps
    pub steps: Vec<PipelineStepIR>,
    /// Total task reward expression (for RL optimization)
    pub reward: Option<ExprIR>,
}

/// Step in a pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStepIR {
    /// Optional binding name
    pub binding: Option<String>,
    /// The call
    pub call: PipelineCallIR,
}

/// Call in a pipeline step
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum PipelineCallIR {
    /// Call a prompt
    Prompt { name: String, args: Vec<ToolExprIR> },
    /// Call a tool
    Tool { name: String, args: Vec<ToolExprIR> },
    /// Call an agent
    Agent { name: String, args: Vec<ToolExprIR> },
    /// Evaluate an expression
    Expr { expr: ToolExprIR },
    /// Parallel branches
    Parallel { branches: Vec<Vec<PipelineStepIR>> },
    /// Conditional branch
    If {
        condition: ExprIR,
        then_steps: Vec<PipelineStepIR>,
        else_steps: Vec<PipelineStepIR>,
    },
    /// Match branch
    Match {
        scrutinee: ExprIR,
        arms: Vec<PipelineMatchArmIR>,
    },
}

/// Match arm in pipeline IR
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineMatchArmIR {
    pub pattern: ExprIR,
    pub steps: Vec<PipelineStepIR>,
}

// ============================================
// Task / Harness / Objective IR
// ============================================

/// Declared task artifact slot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactSlotIR {
    pub name: String,
    pub ty: TypeIR,
}

/// Task definition IR
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskIR {
    pub name: String,
    pub input: TypeIR,
    pub output: TypeIR,
    pub artifacts: Vec<ArtifactSlotIR>,
    pub body: Vec<TaskNodeIR>,
    pub emit: Vec<EmitFieldIR>,
}

/// Task body node IR
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum TaskNodeIR {
    Stage(StageIR),
    Loop(LoopIR),
    Branch(BranchIR),
}

/// Stage kind in a task
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StageKindIR {
    Tool,
    Prompt,
    Agent,
}

/// Task stage IR
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageIR {
    pub name: String,
    pub stage_kind: StageKindIR,
    pub component: String,
    pub input: ExprIR,
    pub output: String,
    pub when: Option<ExprIR>,
}

/// Explicit loop IR
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoopIR {
    pub name: String,
    pub max_iters: ExprIR,
    pub carry: Vec<String>,
    pub while_condition: Option<ExprIR>,
    pub until: Option<ExprIR>,
    pub body: Vec<TaskNodeIR>,
}

/// Structured branch IR
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchIR {
    pub condition: ExprIR,
    pub then_body: Vec<TaskNodeIR>,
    pub else_body: Vec<TaskNodeIR>,
}

/// Final task output mapping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmitFieldIR {
    pub name: String,
    pub value: ExprIR,
}

/// Harness definition IR
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarnessIR {
    pub name: String,
    pub task: String,
    pub defaults: Vec<BindingIR>,
    pub bindings: Vec<TargetBindingIR>,
    pub tunables: Vec<TunableIR>,
}

/// Binding block for a named task target
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetBindingIR {
    pub target: String,
    pub bindings: Vec<BindingIR>,
}

/// Bound configurable value
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BindingIR {
    pub key: BindingPathIR,
    pub value: ExprIR,
}

/// Dot-separated configurable field path
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BindingPathIR {
    pub segments: Vec<String>,
}

/// Tunable field declaration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TunableIR {
    pub path: BindingPathIR,
    pub operator: TuneOperatorIR,
    pub domain: FiniteDomainIR,
}

/// Tune operator IR
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TuneOperatorIR {
    In,
    SubsetOf,
}

/// Finite search domain IR
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum FiniteDomainIR {
    List { values: Vec<ExprIR> },
    Variants { name: String },
}

/// Objective definition IR
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectiveIR {
    pub name: String,
    pub task: String,
    pub harness: String,
    pub dataset: DatasetSpecIR,
    pub repeats: Option<u64>,
    pub constraints: Vec<MetricIR>,
    pub checkers: Vec<MetricIR>,
    pub judges: Vec<MetricIR>,
    pub metrics: Vec<MetricIR>,
    pub score: ExprIR,
    pub split: Option<SplitIR>,
    pub select: Option<SelectIR>,
}

/// Dataset specification IR
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum DatasetSpecIR {
    File { path: String },
    Inline { cases: Vec<InlineDatasetCaseIR> },
}

/// Inline dataset case IR
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InlineDatasetCaseIR {
    pub input: ExprIR,
    pub expected: Option<ExprIR>,
    pub id: Option<String>,
}

/// Metric IR
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricIR {
    pub name: String,
    pub expr: ExprIR,
}

/// Split weights IR
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitIR {
    pub train: f64,
    pub val: f64,
    pub test: f64,
}

/// Objective selection IR
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectIR {
    pub primary: ExprIR,
    pub tie_breakers: Vec<ExprIR>,
}
