//! Scaffold v2 Intermediate Representation
//!
//! Mirrors AST but spanless, serde-friendly, with tagged unions.

use serde::{Deserialize, Serialize};

/// Top-level IR container
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScaffoldIR {
    pub version: String,
    #[serde(default)]
    pub types: Vec<TypeDefIR>,
    pub nodes: Vec<NodeIR>,
    pub graphs: Vec<GraphIR>,
    #[serde(default)]
    pub objectives: Vec<ObjectiveIR>,
}

impl Default for ScaffoldIR {
    fn default() -> Self {
        Self {
            version: "2.0.0".to_string(),
            types: Vec::new(),
            nodes: Vec::new(),
            graphs: Vec::new(),
            objectives: Vec::new(),
        }
    }
}

// ── Types ──

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeDefIR {
    pub name: String,
    pub ty: TypeIR,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TypeIR {
    Bool,
    Int,
    Float,
    String,
    Bytes,
    Any,
    Named { name: String },
    List { element: Box<TypeIR> },
    Map { key: Box<TypeIR>, value: Box<TypeIR> },
    Option { inner: Box<TypeIR> },
    Struct { fields: Vec<FieldIR> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldIR {
    pub name: String,
    pub ty: TypeIR,
}

// ── Nodes ──

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeIR {
    pub name: String,
    pub kind: NodeKindIR,
    pub input: TypeIR,
    pub output: TypeIR,
    pub config: NodeConfigIR,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeKindIR {
    Prompt,
    Tool,
    Agent,
    Verify,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NodeConfigIR {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub template: Option<StringOrFileIR>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<StringOrFileIR>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_turns: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub on_error: Option<ErrorStrategyIR>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shell: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub json: Option<Vec<JsonFieldIR>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StringOrFileIR {
    Literal { value: String },
    File { path: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ErrorStrategyIR {
    Abort,
    Retry { max: u64 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonFieldIR {
    pub key: String,
    pub value: ExprIR,
}

// ── Graphs ──

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphIR {
    pub name: String,
    pub input: TypeIR,
    pub output: TypeIR,
    pub body: Vec<GraphStmtIR>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GraphStmtIR {
    Step(StepIR),
    Loop(LoopIR),
    If(IfIR),
    Choose(ChooseIR),
    Parallel(ParallelIR),
    Emit(EmitIR),
    Carry(CarryIR),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepIR {
    pub name: String,
    pub node: String,
    pub args: Vec<StepArgIR>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StepArgIR {
    Positional { value: ExprIR },
    Named { name: String, value: ExprIR },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoopIR {
    pub max: ExprIR,
    pub while_cond: ExprIR,
    pub body: Vec<GraphStmtIR>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IfIR {
    pub cond: ExprIR,
    pub then_body: Vec<GraphStmtIR>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub else_body: Vec<GraphStmtIR>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChooseIR {
    pub alternatives: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelIR {
    pub var: String,
    pub collection: ExprIR,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reduce: Option<String>,
    pub body: Vec<GraphStmtIR>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "emit_kind", rename_all = "snake_case")]
pub enum EmitIR {
    Direct { value: ExprIR },
    Record { fields: Vec<EmitFieldIR> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmitFieldIR {
    pub name: String,
    pub value: ExprIR,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CarryIR {
    pub name: String,
    pub value: ExprIR,
}

// ── Objectives ──

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectiveIR {
    pub name: String,
    pub graph: String,
    pub dataset: DatasetSpecIR,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub checkers: Vec<CheckerIR>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub judges: Vec<JudgeIR>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub metrics: Vec<MetricIR>,
    pub score: ExprIR,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repeats: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub split: Option<SplitIR>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub select: Option<SelectIR>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tunables: Vec<TunableIR>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub topology: Option<TopologyIR>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckerIR {
    pub name: String,
    pub expr: ExprIR,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JudgeIR {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub template: Option<StringOrFileIR>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rubric: Option<StringOrFileIR>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricIR {
    pub name: String,
    pub checker: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DatasetSpecIR {
    File { path: String },
    Inline { cases: Vec<InlineCaseIR> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InlineCaseIR {
    pub input: ExprIR,
    pub expected: ExprIR,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitIR {
    pub train: f64,
    pub val: f64,
    pub test: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectIR {
    pub primary: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tie_breakers: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TunableIR {
    pub path: Vec<String>,
    pub domain: Vec<ExprIR>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyIR {
    #[serde(default)]
    pub mutations: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_nodes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_depth: Option<u64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub preserve: Vec<String>,
}

// ── Expressions ──

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ExprIR {
    LitInt { value: i64 },
    LitFloat { value: f64 },
    LitString { value: String },
    LitBool { value: bool },
    LitNull,
    Ident { name: String },
    FieldAccess { base: Box<ExprIR>, field: String },
    Index { base: Box<ExprIR>, index: Box<ExprIR> },
    UnaryNot { operand: Box<ExprIR> },
    UnaryNeg { operand: Box<ExprIR> },
    Binary { left: Box<ExprIR>, op: String, right: Box<ExprIR> },
    Call { name: String, args: Vec<ExprIR> },
    List { elements: Vec<ExprIR> },
    Record { fields: Vec<ExprFieldIR> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExprFieldIR {
    pub key: String,
    pub value: ExprIR,
}
