//! AST node definitions for the Scaffold DSL

use std::fmt;

/// Source span for error reporting
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    pub fn merge(self, other: Span) -> Span {
        Span {
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
    }
}

/// A node with source location information
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

/// Identifier with span
pub type Ident = Spanned<String>;

/// Top-level program
#[derive(Debug, Clone)]
pub struct Program {
    pub declarations: Vec<Declaration>,
}

/// Top-level declarations
#[derive(Debug, Clone)]
pub enum Declaration {
    Type(TypeDecl),
    /// extern crate name = "version"
    ExternCrate(ExternCrateDecl),
    /// foreign rust name { ... }
    Foreign(ForeignDecl),
    /// tool name { ... } - deterministic code only
    Tool(ToolDecl),
    /// prompt name { ... } - single LLM call with typed I/O
    Prompt(PromptDecl),
    /// agent name { ... } - multi-turn LLM with tools
    Agent(AgentDecl),
    /// pipeline name { ... } - fixed sequence of prompts/tools
    Pipeline(PipelineDecl),
}

/// Duration value (kept for potential future use)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Duration {
    pub value: u64,
    pub unit: DurationUnit,
}

impl Duration {
    pub fn to_millis(&self) -> u64 {
        match self.unit {
            DurationUnit::Milliseconds => self.value,
            DurationUnit::Seconds => self.value * 1000,
            DurationUnit::Minutes => self.value * 60 * 1000,
            DurationUnit::Hours => self.value * 60 * 60 * 1000,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DurationUnit {
    Milliseconds,
    Seconds,
    Minutes,
    Hours,
}

/// Size value (for memory)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Size {
    pub value: u64,
    pub unit: SizeUnit,
}

impl Size {
    pub fn to_bytes(&self) -> u64 {
        match self.unit {
            SizeUnit::Kilobytes => self.value * 1024,
            SizeUnit::Megabytes => self.value * 1024 * 1024,
            SizeUnit::Gigabytes => self.value * 1024 * 1024 * 1024,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SizeUnit {
    Kilobytes,
    Megabytes,
    Gigabytes,
}

/// Type declaration
#[derive(Debug, Clone)]
pub struct TypeDecl {
    pub name: Ident,
    pub ty: Spanned<TypeExpr>,
    pub span: Span,
}

/// Type expression
#[derive(Debug, Clone)]
pub enum TypeExpr {
    /// Primitive types: bool, int, float, string, any, bytes
    Primitive(PrimitiveType),
    /// Named type reference
    Named(String),
    /// list<T>
    List(Box<Spanned<TypeExpr>>),
    /// map<K, V>
    Map(Box<Spanned<TypeExpr>>, Box<Spanned<TypeExpr>>),
    /// option<T>
    Option(Box<Spanned<TypeExpr>>),
    /// result<T, E>
    Result(Box<Spanned<TypeExpr>>, Box<Spanned<TypeExpr>>),
    /// Struct type: { field: T, ... }
    Struct(Vec<FieldDecl>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimitiveType {
    Bool,
    Int,
    Float,
    String,
    Any,
    Bytes,
}

impl fmt::Display for PrimitiveType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PrimitiveType::Bool => write!(f, "bool"),
            PrimitiveType::Int => write!(f, "int"),
            PrimitiveType::Float => write!(f, "float"),
            PrimitiveType::String => write!(f, "string"),
            PrimitiveType::Any => write!(f, "any"),
            PrimitiveType::Bytes => write!(f, "bytes"),
        }
    }
}

/// Field declaration in a struct
#[derive(Debug, Clone)]
pub struct FieldDecl {
    pub name: Ident,
    pub ty: Spanned<TypeExpr>,
}

/// Expression
#[derive(Debug, Clone)]
pub enum Expr {
    /// Literal value
    Literal(Literal),
    /// Identifier
    Ident(String),
    /// Field access: expr.field
    FieldAccess(Box<Spanned<Expr>>, Ident),
    /// Binary expression: expr op expr
    Binary(Box<Spanned<Expr>>, BinOp, Box<Spanned<Expr>>),
    /// Function call: func(args...)
    Call(String, Vec<Spanned<Expr>>),
    /// Foreign function call: module::func(args...)
    ForeignCall {
        module: String,
        function: String,
        args: Vec<Spanned<Expr>>,
    },
    /// Parenthesized expression
    Paren(Box<Spanned<Expr>>),
}

/// Literal values
#[derive(Debug, Clone)]
pub enum Literal {
    Int(i64),
    Float(f64),
    String(String),
    Bool(bool),
    Null,
}

/// Binary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    // Comparison
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
    // Logical
    And,
    Or,
    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,
}

impl fmt::Display for BinOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BinOp::Eq => write!(f, "=="),
            BinOp::Ne => write!(f, "!="),
            BinOp::Lt => write!(f, "<"),
            BinOp::Gt => write!(f, ">"),
            BinOp::Le => write!(f, "<="),
            BinOp::Ge => write!(f, ">="),
            BinOp::And => write!(f, "&&"),
            BinOp::Or => write!(f, "||"),
            BinOp::Add => write!(f, "+"),
            BinOp::Sub => write!(f, "-"),
            BinOp::Mul => write!(f, "*"),
            BinOp::Div => write!(f, "/"),
        }
    }
}

impl BinOp {
    /// Returns the precedence of this operator (higher = binds tighter)
    pub fn precedence(&self) -> u8 {
        match self {
            BinOp::Or => 1,
            BinOp::And => 2,
            BinOp::Eq | BinOp::Ne => 3,
            BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge => 4,
            BinOp::Add | BinOp::Sub => 5,
            BinOp::Mul | BinOp::Div => 6,
        }
    }
}

// ============================================
// Foreign Declarations
// ============================================

/// External crate declaration: extern crate goblin = "0.7"
#[derive(Debug, Clone)]
pub struct ExternCrateDecl {
    pub name: Ident,
    pub version: String,
    /// Optional features: { features = ["derive"] }
    pub features: Vec<String>,
    pub span: Span,
}

/// Foreign module declaration: foreign rust parsing { ... }
#[derive(Debug, Clone)]
pub struct ForeignDecl {
    /// Language (currently only "rust")
    pub language: Ident,
    /// Module name
    pub name: Ident,
    /// Type aliases: type ElfBinary = goblin::elf::Elf
    pub type_aliases: Vec<ForeignTypeAlias>,
    /// Function declarations
    pub functions: Vec<ForeignFn>,
    pub span: Span,
}

/// Type alias in foreign block: type ElfBinary = goblin::elf::Elf
#[derive(Debug, Clone)]
pub struct ForeignTypeAlias {
    /// Scaffold type name
    pub name: Ident,
    /// External Rust type path
    pub external_type: String,
    pub span: Span,
}

/// Foreign function declaration: fn parse_elf(data: bytes) -> result<ElfBinary, string>
#[derive(Debug, Clone)]
pub struct ForeignFn {
    pub name: Ident,
    pub params: Vec<ForeignParam>,
    pub return_type: Spanned<TypeExpr>,
    pub span: Span,
}

/// Foreign function parameter
#[derive(Debug, Clone)]
pub struct ForeignParam {
    pub name: Ident,
    pub ty: Spanned<TypeExpr>,
}

// ============================================
// Tool Definitions
// ============================================

/// Tool declaration
#[derive(Debug, Clone)]
pub struct ToolDecl {
    pub name: Ident,
    pub input: Spanned<TypeExpr>,
    pub output: Spanned<TypeExpr>,
    /// Tool implementation
    pub implementation: Option<ToolImpl>,
    /// Tool specification (pre/post/pure)
    pub spec: Option<ToolSpec>,
    /// Implementation variants
    pub variants: Vec<ToolVariant>,
    pub span: Span,
}

/// Tool implementation
#[derive(Debug, Clone)]
pub enum ToolImpl {
    /// Simple expression or foreign call
    Expr(Spanned<ToolExpr>),
    /// Sequence of operations: sequence { ... }
    Sequence(Vec<ToolStatement>),
    /// Parallel operations: parallel { ... }
    Parallel(Vec<ToolStatement>),
}

/// Tool expression (used in impl)
#[derive(Debug, Clone)]
pub enum ToolExpr {
    /// Variable reference
    Ident(String),
    /// Field access: expr.field
    FieldAccess(Box<Spanned<ToolExpr>>, Ident),
    /// Foreign function call: module::func(args)
    ForeignCall {
        module: String,
        function: String,
        args: Vec<Spanned<ToolExpr>>,
    },
    /// Local tool call: tool_name(args)
    ToolCall {
        tool: String,
        args: Vec<Spanned<ToolExpr>>,
    },
    /// Shell command: shell("command")
    Shell(String),
    /// Pipe expression: expr |> func
    Pipe(Box<Spanned<ToolExpr>>, Box<Spanned<ToolExpr>>),
    /// Conditional: if cond { then } else { else }
    If {
        condition: Box<Spanned<Expr>>,
        then_branch: Box<ToolImpl>,
        else_branch: Option<Box<ToolImpl>>,
    },
    /// Match expression
    Match {
        scrutinee: Box<Spanned<ToolExpr>>,
        arms: Vec<MatchArm>,
    },
    /// For loop: for item in collection { ... }
    For {
        variable: Ident,
        iterable: Box<Spanned<ToolExpr>>,
        body: Box<ToolImpl>,
    },
    /// While loop: while condition { ... }
    While {
        condition: Box<Spanned<Expr>>,
        body: Box<ToolImpl>,
    },
    /// Infinite loop: loop { ... }
    Loop { body: Box<ToolImpl> },
    /// Break out of loop
    Break,
    /// Continue to next iteration
    Continue,
    /// Literal value
    Literal(Literal),
    /// Wrapped general expression (for arithmetic, comparisons, etc.)
    Expr(Box<Spanned<Expr>>),
}

/// Statement in a tool sequence/parallel block
#[derive(Debug, Clone)]
pub struct ToolStatement {
    /// Optional binding: let name = expr
    pub binding: Option<Ident>,
    /// The expression
    pub expr: Spanned<ToolExpr>,
    pub span: Span,
}

/// Match arm in tool expression
#[derive(Debug, Clone)]
pub struct MatchArm {
    pub pattern: Spanned<Expr>,
    pub body: ToolImpl,
    pub span: Span,
}

/// Tool specification
#[derive(Debug, Clone)]
pub struct ToolSpec {
    /// Preconditions
    pub preconditions: Vec<Spanned<Expr>>,
    /// Postconditions
    pub postconditions: Vec<Spanned<Expr>>,
    /// Whether the tool is pure (no side effects)
    pub pure: bool,
    pub span: Span,
}

/// Tool variant (alternative implementation)
#[derive(Debug, Clone)]
pub struct ToolVariant {
    pub name: Ident,
    pub implementation: ToolImpl,
    pub span: Span,
}

// ============================================
// Prompt Definitions (Single LLM Call)
// ============================================

/// Prompt declaration - single LLM call with typed I/O
#[derive(Debug, Clone)]
pub struct PromptDecl {
    pub name: Ident,
    pub input: Spanned<TypeExpr>,
    pub output: Spanned<TypeExpr>,
    /// Template string (with {var} interpolation)
    pub template: StringOrFile,
    /// Optional system prompt
    pub system: Option<StringOrFile>,
    pub span: Span,
}

/// String literal or file reference
#[derive(Debug, Clone)]
pub enum StringOrFile {
    /// Inline string: "content"
    Literal(String),
    /// File reference: file("path/to/file.md")
    File(String),
}

// ============================================
// Agent Definitions (Multi-turn LLM)
// ============================================

/// Error handling strategy for agents
#[derive(Debug, Clone)]
pub enum ErrorStrategy {
    /// Fail immediately on error (default)
    Abort,
    /// Retry up to N times before failing
    Retry(u64),
}

impl Default for ErrorStrategy {
    fn default() -> Self {
        ErrorStrategy::Abort
    }
}

/// Agent declaration - multi-turn LLM with tool access
#[derive(Debug, Clone)]
pub struct AgentDecl {
    pub name: Ident,
    pub input: Spanned<TypeExpr>,
    pub output: Spanned<TypeExpr>,
    /// Available tools (list of tool names)
    pub tools: Vec<Ident>,
    /// System prompt (defines agent behavior)
    pub system: StringOrFile,
    /// Model to use for this agent (e.g., "gpt-4o", "claude-sonnet-4-20250514")
    pub model: Option<String>,
    /// Maximum turns before termination
    pub max_turns: Option<u64>,
    /// Process reward expression (for RL optimization)
    pub reward: Option<Spanned<Expr>>,
    /// Explicit termination condition (beyond max_turns)
    pub done: Option<Spanned<Expr>>,
    /// Error handling strategy
    pub on_error: ErrorStrategy,
    /// Execution timeout in seconds
    pub timeout: Option<u64>,
    pub span: Span,
}

// ============================================
// Pipeline Definitions (Fixed Sequence)
// ============================================

/// Pipeline declaration - fixed sequence of prompts/tools
#[derive(Debug, Clone)]
pub struct PipelineDecl {
    pub name: Ident,
    pub input: Spanned<TypeExpr>,
    pub output: Spanned<TypeExpr>,
    /// Sequence of steps
    pub steps: Vec<PipelineStep>,
    /// Total task reward expression (for RL optimization)
    pub reward: Option<Spanned<Expr>>,
    pub span: Span,
}

/// Step in a pipeline
#[derive(Debug, Clone)]
pub struct PipelineStep {
    /// Optional binding: let name = ...
    pub binding: Option<Ident>,
    /// The call (prompt/tool) or expression
    pub call: PipelineCall,
    pub span: Span,
}

/// Call in a pipeline step
#[derive(Debug, Clone)]
pub enum PipelineCall {
    /// Call a prompt: prompt_name(args)
    Prompt {
        name: String,
        args: Vec<Spanned<ToolExpr>>,
    },
    /// Call a tool: tool_name(args)
    Tool {
        name: String,
        args: Vec<Spanned<ToolExpr>>,
    },
    /// Call an agent: agent_name(args)
    Agent {
        name: String,
        args: Vec<Spanned<ToolExpr>>,
    },
    /// Evaluate an expression: e.g., field access or literal/identifier
    Expr(Spanned<ToolExpr>),
    /// Parallel branches: parallel { { ... } { ... } }
    Parallel {
        branches: Vec<Vec<PipelineStep>>,
    },
    /// Conditional branch: if cond { ... } else { ... }
    If {
        condition: Spanned<Expr>,
        then_steps: Vec<PipelineStep>,
        else_steps: Vec<PipelineStep>,
    },
    /// Match branch: match expr { pattern => { ... } ... }
    Match {
        scrutinee: Spanned<Expr>,
        arms: Vec<PipelineMatchArm>,
    },
}

/// Match arm in a pipeline match block
#[derive(Debug, Clone)]
pub struct PipelineMatchArm {
    pub pattern: Spanned<Expr>,
    pub steps: Vec<PipelineStep>,
    pub span: Span,
}
