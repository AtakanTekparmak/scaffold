//! Abstract Syntax Tree for Scaffold v2
//!
//! 4 declaration kinds: type, node, graph, objective.

/// Byte offset range in source
pub type Span = std::ops::Range<usize>;

/// A value annotated with its source span
#[derive(Debug, Clone)]
pub struct Spanned<T> {
    pub node: T,
    pub span: Span,
}

impl<T> Spanned<T> {
    pub fn new(node: T, span: Span) -> Self {
        Self { node, span }
    }
}

/// An identifier with source span
#[derive(Debug, Clone)]
pub struct Ident {
    pub name: String,
    pub span: Span,
}

impl Ident {
    pub fn new(name: impl Into<String>, span: Span) -> Self {
        Self {
            name: name.into(),
            span,
        }
    }
}

// ── Program ──────────────────────────────────────────────

pub struct Program {
    pub declarations: Vec<Declaration>,
}

pub enum Declaration {
    Type(TypeDecl),
    Node(NodeDecl),
    Graph(GraphDecl),
    Objective(ObjectiveDecl),
}

// ── Types ────────────────────────────────────────────────

pub struct TypeDecl {
    pub name: Ident,
    pub ty: Spanned<TypeExpr>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum TypeExpr {
    Primitive(PrimitiveType),
    Named(String),
    List(Box<Spanned<TypeExpr>>),
    Map(Box<Spanned<TypeExpr>>, Box<Spanned<TypeExpr>>),
    Option(Box<Spanned<TypeExpr>>),
    Struct(Vec<FieldDecl>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimitiveType {
    Bool,
    Int,
    Float,
    String,
    Bytes,
    Any,
}

#[derive(Debug, Clone)]
pub struct FieldDecl {
    pub name: Ident,
    pub ty: Spanned<TypeExpr>,
}

// ── Nodes ────────────────────────────────────────────────

pub struct NodeDecl {
    pub name: Ident,
    pub kind: Spanned<NodeKind>,
    pub input: Spanned<TypeExpr>,
    pub output: Spanned<TypeExpr>,
    pub config: NodeConfig,
    pub span: Span,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeKind {
    Prompt,
    Tool,
    Agent,
    Verify,
}

#[derive(Debug, Clone, Default)]
pub struct NodeConfig {
    pub template: Option<StringOrFile>,
    pub system: Option<StringOrFile>,
    pub model: Option<String>,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u64>,
    pub tools: Vec<Ident>,
    pub max_turns: Option<u64>,
    pub timeout: Option<u64>,
    pub on_error: Option<ErrorStrategy>,
    pub shell: Option<String>,
    pub json: Option<Vec<JsonField>>,
}

#[derive(Debug, Clone)]
pub enum StringOrFile {
    Literal(String),
    File(String),
}

#[derive(Debug, Clone, Copy)]
pub enum ErrorStrategy {
    Abort,
    Retry(u64),
}

#[derive(Debug, Clone)]
pub struct JsonField {
    pub key: String,
    pub value: Spanned<Expr>,
    pub span: Span,
}

// ── Graphs ───────────────────────────────────────────────

pub struct GraphDecl {
    pub name: Ident,
    pub input: Spanned<TypeExpr>,
    pub output: Spanned<TypeExpr>,
    pub body: Vec<GraphStmt>,
    pub span: Span,
}

pub enum GraphStmt {
    Step(StepStmt),
    Loop(LoopStmt),
    If(IfStmt),
    Choose(ChooseStmt),
    Parallel(ParallelStmt),
    Emit(EmitStmt),
    Carry(CarryStmt),
}

pub struct StepStmt {
    pub name: Ident,
    pub node: Ident,
    pub args: Vec<StepArg>,
    pub span: Span,
}

pub enum StepArg {
    Positional(Spanned<Expr>),
    Named {
        name: Ident,
        value: Spanned<Expr>,
    },
}

pub struct LoopStmt {
    pub max: Spanned<Expr>,
    pub while_cond: Spanned<Expr>,
    pub body: Vec<GraphStmt>,
    pub span: Span,
}

pub struct IfStmt {
    pub cond: Spanned<Expr>,
    pub then_body: Vec<GraphStmt>,
    pub else_body: Vec<GraphStmt>,
    pub span: Span,
}

pub struct ChooseStmt {
    pub alternatives: Vec<Ident>,
    pub span: Span,
}

pub struct ParallelStmt {
    pub var: Ident,
    pub collection: Spanned<Expr>,
    pub reduce: Option<Ident>,
    pub body: Vec<GraphStmt>,
    pub span: Span,
}

pub enum EmitStmt {
    Direct {
        value: Spanned<Expr>,
        span: Span,
    },
    Record {
        fields: Vec<EmitField>,
        span: Span,
    },
}

pub struct EmitField {
    pub name: Ident,
    pub value: Spanned<Expr>,
}

pub struct CarryStmt {
    pub name: Ident,
    pub value: Spanned<Expr>,
    pub span: Span,
}

// ── Objectives ───────────────────────────────────────────

pub struct ObjectiveDecl {
    pub name: Ident,
    pub graph: Ident,
    pub dataset: DatasetSpec,
    pub checkers: Vec<CheckerDecl>,
    pub judges: Vec<JudgeDecl>,
    pub metrics: Vec<MetricDecl>,
    pub score: Spanned<Expr>,
    pub repeats: Option<u64>,
    pub split: Option<SplitDecl>,
    pub select: Option<SelectDecl>,
    pub tunables: Vec<TuneStmt>,
    pub topology: Option<TopologyDecl>,
    pub span: Span,
}

pub struct CheckerDecl {
    pub name: Ident,
    pub expr: Spanned<Expr>,
    pub span: Span,
}

pub struct JudgeDecl {
    pub name: Ident,
    pub model: Option<String>,
    pub template: Option<StringOrFile>,
    pub rubric: Option<StringOrFile>,
    pub span: Span,
}

pub struct MetricDecl {
    pub name: Ident,
    pub checker: Ident,
    pub span: Span,
}

pub enum DatasetSpec {
    File(String),
    Inline { cases: Vec<InlineCase> },
}

pub struct InlineCase {
    pub input: Spanned<Expr>,
    pub expected: Spanned<Expr>,
    pub id: Option<String>,
    pub span: Span,
}

pub struct SplitDecl {
    pub train: f64,
    pub val: f64,
    pub test: f64,
    pub span: Span,
}

pub struct SelectDecl {
    pub primary: Ident,
    pub tie_breakers: Vec<Ident>,
    pub span: Span,
}

pub struct TuneStmt {
    pub path: Vec<Ident>,
    pub domain: Vec<Spanned<Expr>>,
    pub span: Span,
}

pub struct TopologyDecl {
    pub mutations: Vec<String>,
    pub max_nodes: Option<u64>,
    pub max_depth: Option<u64>,
    pub preserve: Vec<String>,
    pub span: Span,
}

// ── Expressions ──────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum Expr {
    Literal(Literal),
    Ident(String),
    FieldAccess(Box<Spanned<Expr>>, Ident),
    Index(Box<Spanned<Expr>>, Box<Spanned<Expr>>),
    UnaryNot(Box<Spanned<Expr>>),
    UnaryNeg(Box<Spanned<Expr>>),
    Binary(Box<Spanned<Expr>>, BinOp, Box<Spanned<Expr>>),
    Call(String, Vec<Spanned<Expr>>),
    List(Vec<Spanned<Expr>>),
    Record(Vec<ExprField>),
    Paren(Box<Spanned<Expr>>),
}

#[derive(Debug, Clone)]
pub enum Literal {
    Int(i64),
    Float(f64),
    String(String),
    Bool(bool),
    Null,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
    And,
    Or,
    Add,
    Sub,
    Mul,
    Div,
}

impl std::fmt::Display for BinOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            BinOp::Eq => "==",
            BinOp::Ne => "!=",
            BinOp::Lt => "<",
            BinOp::Gt => ">",
            BinOp::Le => "<=",
            BinOp::Ge => ">=",
            BinOp::And => "&&",
            BinOp::Or => "||",
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
        };
        write!(f, "{}", s)
    }
}

#[derive(Debug, Clone)]
pub struct ExprField {
    pub key: Ident,
    pub value: Spanned<Expr>,
}
