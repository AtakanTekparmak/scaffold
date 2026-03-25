//! Lexer for Scaffold v2 using logos

use logos::Logos;

use crate::ast::Span;

/// Scaffold v2 token set
#[derive(Logos, Debug, Clone, PartialEq)]
#[logos(skip r"[ \t\r\n\f]+")]
#[logos(skip r"//[^\n]*")]
#[logos(skip r"/\*([^*]|\*[^/])*\*/")]
pub enum Token {
    // ── Keywords: declarations ──
    #[token("type")]
    Type,
    #[token("node")]
    Node,
    #[token("graph")]
    Graph,
    #[token("objective")]
    Objective,

    // ── Keywords: node kinds ──
    #[token("prompt")]
    Prompt,
    #[token("tool")]
    Tool,
    #[token("agent")]
    Agent,
    #[token("verify")]
    Verify,

    // ── Keywords: type primitives ──
    #[token("bool")]
    Bool,
    #[token("int")]
    KwInt,
    #[token("float")]
    KwFloat,
    #[token("string")]
    KwString,
    #[token("bytes")]
    Bytes,
    #[token("any")]
    Any,
    #[token("list")]
    List,
    #[token("map")]
    Map,
    #[token("option")]
    Option,

    // ── Keywords: node fields ──
    #[token("in")]
    In,
    #[token("out")]
    Out,
    #[token("template")]
    Template,
    #[token("system")]
    System,
    #[token("model")]
    Model,
    #[token("temperature")]
    Temperature,
    #[token("max_tokens")]
    MaxTokens,
    #[token("tools")]
    Tools,
    #[token("max_turns")]
    MaxTurns,
    #[token("timeout")]
    Timeout,
    #[token("on_error")]
    OnError,
    #[token("shell")]
    Shell,
    #[token("json")]
    Json,

    // ── Keywords: error strategies ──
    #[token("abort")]
    Abort,
    #[token("retry")]
    Retry,

    // ── Keywords: graph statements ──
    #[token("step")]
    Step,
    #[token("loop")]
    Loop,
    #[token("if")]
    If,
    #[token("else")]
    Else,
    #[token("choose")]
    Choose,
    #[token("parallel")]
    Parallel,
    #[token("emit")]
    Emit,
    #[token("carry")]
    Carry,
    #[token("max")]
    Max,
    #[token("while")]
    While,
    #[token("reduce")]
    Reduce,
    #[token("file")]
    File,

    // ── Keywords: objective ──
    #[token("dataset")]
    Dataset,
    #[token("cases")]
    Cases,
    #[token("checker")]
    Checker,
    #[token("judge")]
    Judge,
    #[token("metric")]
    Metric,
    #[token("score")]
    Score,
    #[token("repeats")]
    Repeats,
    #[token("split")]
    Split,
    #[token("train")]
    Train,
    #[token("val")]
    Val,
    #[token("test")]
    Test,
    #[token("select")]
    Select,
    #[token("primary")]
    Primary,
    #[token("tie_breakers")]
    TieBreakers,
    #[token("tune")]
    Tune,
    #[token("topology")]
    Topology,
    #[token("mutations")]
    Mutations,
    #[token("max_nodes")]
    MaxNodes,
    #[token("max_depth")]
    MaxDepth,
    #[token("preserve")]
    Preserve,
    #[token("input")]
    Input,
    #[token("expected")]
    Expected,
    #[token("id")]
    Id,
    #[token("rubric")]
    Rubric,

    // ── Literals ──
    #[token("true")]
    True,
    #[token("false")]
    False,
    #[token("null")]
    Null,

    #[regex(r"[0-9]+\.[0-9]+([eE][+-]?[0-9]+)?", |lex| lex.slice().parse::<f64>().ok())]
    FloatLit(f64),

    #[regex(r"[0-9]+", |lex| lex.slice().parse::<i64>().ok(), priority = 3)]
    IntLit(i64),

    #[regex(r#""([^"\\]|\\.)*""#, parse_string)]
    StringLit(String),

    #[regex(r"[a-zA-Z_][a-zA-Z0-9_]*", |lex| lex.slice().to_string(), priority = 1)]
    Ident(String),

    // ── Punctuation ──
    #[token("{")]
    LBrace,
    #[token("}")]
    RBrace,
    #[token("(")]
    LParen,
    #[token(")")]
    RParen,
    #[token("[")]
    LBracket,
    #[token("]")]
    RBracket,
    #[token(":")]
    Colon,
    #[token(",")]
    Comma,
    #[token("=")]
    Eq,
    #[token(".")]
    Dot,

    // ── Operators ──
    #[token("==")]
    EqEq,
    #[token("!=")]
    BangEq,
    #[token("<")]
    Lt,
    #[token(">")]
    Gt,
    #[token("<=")]
    Le,
    #[token(">=")]
    Ge,
    #[token("&&")]
    AmpAmp,
    #[token("||")]
    PipePipe,
    #[token("+")]
    Plus,
    #[token("-")]
    Minus,
    #[token("*")]
    Star,
    #[token("/")]
    Slash,
    #[token("!")]
    Bang,
}

fn parse_string(lex: &mut logos::Lexer<Token>) -> Option<String> {
    let slice = lex.slice();
    // Strip surrounding quotes
    let inner = &slice[1..slice.len() - 1];
    let mut result = String::new();
    let mut chars = inner.chars();
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            match chars.next() {
                Some('n') => result.push('\n'),
                Some('t') => result.push('\t'),
                Some('r') => result.push('\r'),
                Some('\\') => result.push('\\'),
                Some('"') => result.push('"'),
                Some(other) => {
                    result.push('\\');
                    result.push(other);
                }
                None => result.push('\\'),
            }
        } else {
            result.push(ch);
        }
    }
    Some(result)
}

impl std::fmt::Display for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Token::Type => write!(f, "type"),
            Token::Node => write!(f, "node"),
            Token::Graph => write!(f, "graph"),
            Token::Objective => write!(f, "objective"),
            Token::Prompt => write!(f, "prompt"),
            Token::Tool => write!(f, "tool"),
            Token::Agent => write!(f, "agent"),
            Token::Verify => write!(f, "verify"),
            Token::Bool => write!(f, "bool"),
            Token::KwInt => write!(f, "int"),
            Token::KwFloat => write!(f, "float"),
            Token::KwString => write!(f, "string"),
            Token::Bytes => write!(f, "bytes"),
            Token::Any => write!(f, "any"),
            Token::List => write!(f, "list"),
            Token::Map => write!(f, "map"),
            Token::Option => write!(f, "option"),
            Token::In => write!(f, "in"),
            Token::Out => write!(f, "out"),
            Token::Template => write!(f, "template"),
            Token::System => write!(f, "system"),
            Token::Model => write!(f, "model"),
            Token::Temperature => write!(f, "temperature"),
            Token::MaxTokens => write!(f, "max_tokens"),
            Token::Tools => write!(f, "tools"),
            Token::MaxTurns => write!(f, "max_turns"),
            Token::Timeout => write!(f, "timeout"),
            Token::OnError => write!(f, "on_error"),
            Token::Shell => write!(f, "shell"),
            Token::Json => write!(f, "json"),
            Token::Abort => write!(f, "abort"),
            Token::Retry => write!(f, "retry"),
            Token::Step => write!(f, "step"),
            Token::Loop => write!(f, "loop"),
            Token::If => write!(f, "if"),
            Token::Else => write!(f, "else"),
            Token::Choose => write!(f, "choose"),
            Token::Parallel => write!(f, "parallel"),
            Token::Emit => write!(f, "emit"),
            Token::Carry => write!(f, "carry"),
            Token::Max => write!(f, "max"),
            Token::While => write!(f, "while"),
            Token::Reduce => write!(f, "reduce"),
            Token::File => write!(f, "file"),
            Token::Dataset => write!(f, "dataset"),
            Token::Cases => write!(f, "cases"),
            Token::Checker => write!(f, "checker"),
            Token::Judge => write!(f, "judge"),
            Token::Metric => write!(f, "metric"),
            Token::Score => write!(f, "score"),
            Token::Repeats => write!(f, "repeats"),
            Token::Split => write!(f, "split"),
            Token::Train => write!(f, "train"),
            Token::Val => write!(f, "val"),
            Token::Test => write!(f, "test"),
            Token::Select => write!(f, "select"),
            Token::Primary => write!(f, "primary"),
            Token::TieBreakers => write!(f, "tie_breakers"),
            Token::Tune => write!(f, "tune"),
            Token::Topology => write!(f, "topology"),
            Token::Mutations => write!(f, "mutations"),
            Token::MaxNodes => write!(f, "max_nodes"),
            Token::MaxDepth => write!(f, "max_depth"),
            Token::Preserve => write!(f, "preserve"),
            Token::Input => write!(f, "input"),
            Token::Expected => write!(f, "expected"),
            Token::Id => write!(f, "id"),
            Token::Rubric => write!(f, "rubric"),
            Token::True => write!(f, "true"),
            Token::False => write!(f, "false"),
            Token::Null => write!(f, "null"),
            Token::FloatLit(v) => write!(f, "{}", v),
            Token::IntLit(v) => write!(f, "{}", v),
            Token::StringLit(s) => write!(f, "\"{}\"", s),
            Token::Ident(s) => write!(f, "{}", s),
            Token::LBrace => write!(f, "{{"),
            Token::RBrace => write!(f, "}}"),
            Token::LParen => write!(f, "("),
            Token::RParen => write!(f, ")"),
            Token::LBracket => write!(f, "["),
            Token::RBracket => write!(f, "]"),
            Token::Colon => write!(f, ":"),
            Token::Comma => write!(f, ","),
            Token::Eq => write!(f, "="),
            Token::Dot => write!(f, "."),
            Token::EqEq => write!(f, "=="),
            Token::BangEq => write!(f, "!="),
            Token::Lt => write!(f, "<"),
            Token::Gt => write!(f, ">"),
            Token::Le => write!(f, "<="),
            Token::Ge => write!(f, ">="),
            Token::AmpAmp => write!(f, "&&"),
            Token::PipePipe => write!(f, "||"),
            Token::Plus => write!(f, "+"),
            Token::Minus => write!(f, "-"),
            Token::Star => write!(f, "*"),
            Token::Slash => write!(f, "/"),
            Token::Bang => write!(f, "!"),
        }
    }
}

/// A token with its source span
pub type SpannedToken = (Token, Span);

/// Lexer wrapper that produces SpannedToken sequences
pub struct Lexer<'source> {
    inner: logos::Lexer<'source, Token>,
    peeked: Option<Option<SpannedToken>>,
}

impl<'source> Lexer<'source> {
    pub fn new(source: &'source str) -> Self {
        Self {
            inner: Token::lexer(source),
            peeked: None,
        }
    }

    /// Peek at the next token without consuming it
    pub fn peek(&mut self) -> Option<&SpannedToken> {
        if self.peeked.is_none() {
            self.peeked = Some(self.next_inner());
        }
        self.peeked.as_ref().and_then(|o| o.as_ref())
    }

    /// Consume and return the next token
    pub fn next_token(&mut self) -> Option<SpannedToken> {
        if let Some(peeked) = self.peeked.take() {
            peeked
        } else {
            self.next_inner()
        }
    }

    fn next_inner(&mut self) -> Option<SpannedToken> {
        loop {
            match self.inner.next() {
                Some(Ok(token)) => {
                    let span = self.inner.span();
                    return Some((token, span));
                }
                Some(Err(_)) => {
                    // Skip unrecognized tokens
                    continue;
                }
                None => return None,
            }
        }
    }
}
