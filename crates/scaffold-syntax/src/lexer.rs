//! Lexer for the Scaffold DSL using logos

use logos::Logos;

use crate::ast::Span;

/// Token types for the Scaffold language
#[derive(Logos, Debug, Clone, PartialEq)]
#[logos(skip r"[ \t\n\r]+")]
#[logos(skip r"//[^\n]*")]
#[logos(skip r"/\*([^*]|\*[^/])*\*/")]
pub enum Token {
    // Keywords
    #[token("task")]
    Task,
    #[token("artifact")]
    Artifact,
    #[token("artifacts")]
    Artifacts,
    #[token("type")]
    Type,
    #[token("input")]
    Input,
    #[token("output")]
    Output,
    #[token("state")]
    State,
    #[token("decompose")]
    Decompose,
    #[token("subgoal")]
    Subgoal,
    #[token("pre")]
    Pre,
    #[token("post")]
    Post,
    #[token("options")]
    Options,
    #[token("done")]
    Done,
    #[token("reward")]
    Reward,
    #[token("timeout")]
    Timeout,
    #[token("verify")]
    Verify,
    #[token("on_fail")]
    OnFail,
    #[token("on_error")]
    OnError,
    #[token("tool")]
    Tool,
    #[token("stage")]
    Stage,
    #[token("using")]
    Using,
    #[token("when")]
    When,
    #[token("emit")]
    Emit,
    #[token("harness")]
    Harness,
    #[token("defaults")]
    Defaults,
    #[token("bind")]
    Bind,
    #[token("tune")]
    Tune,
    #[token("objective")]
    Objective,
    #[token("dataset")]
    Dataset,
    #[token("constraint")]
    Constraint,
    #[token("checker")]
    Checker,
    #[token("judge")]
    Judge,
    #[token("metric")]
    Metric,
    #[token("score")]
    Score,
    #[token("split")]
    Split,
    #[token("select")]
    Select,
    #[token("repeats")]
    Repeats,
    #[token("train")]
    Train,
    #[token("val")]
    Val,
    #[token("test")]
    Test,
    #[token("primary")]
    Primary,
    #[token("tie_breakers")]
    TieBreakers,
    #[token("carry")]
    Carry,
    #[token("until")]
    Until,
    #[token("subset_of")]
    SubsetOf,
    #[token("max_iters")]
    MaxIters,

    // New semantic constructs
    #[token("prompt")]
    Prompt,
    #[token("agent")]
    Agent,
    #[token("pipeline")]
    Pipeline,
    #[token("system")]
    System,
    #[token("template")]
    Template,
    #[token("tools")]
    Tools,
    #[token("max_turns")]
    MaxTurns,
    #[token("model")]
    Model,
    #[token("file")]
    File,

    // Foreign and tool keywords
    #[token("extern")]
    Extern,
    #[token("crate")]
    Crate,
    #[token("foreign")]
    Foreign,
    #[token("fn")]
    Fn,
    #[token("impl")]
    Impl,
    #[token("spec")]
    Spec,
    #[token("variants")]
    Variants,
    #[token("pure")]
    Pure,
    #[token("sequence")]
    Sequence,
    #[token("parallel")]
    Parallel,
    #[token("let")]
    Let,
    #[token("for")]
    For,
    #[token("in")]
    In,
    #[token("while")]
    While,
    #[token("loop")]
    Loop,
    #[token("break")]
    Break,
    #[token("continue")]
    Continue,
    #[token("if")]
    If,
    #[token("else")]
    Else,
    #[token("match")]
    Match,

    // Additional types for foreign
    #[token("result")]
    Result_,
    #[token("bytes")]
    Bytes,

    // Verification functions
    #[token("reachable")]
    Reachable,
    #[token("no_deadlock")]
    NoDeadlock,
    #[token("bounded")]
    Bounded,
    #[token("grounded")]
    Grounded,
    #[token("terminates")]
    Terminates,

    // Failure strategies
    #[token("retry")]
    Retry,
    #[token("rollback")]
    Rollback,
    #[token("abort")]
    Abort,
    #[token("replan")]
    Replan,

    // Type keywords
    #[token("bool")]
    Bool,
    #[token("int")]
    Int,
    #[token("float")]
    Float,
    #[token("string")]
    String_,
    #[token("any")]
    Any,
    #[token("list")]
    List,
    #[token("map")]
    Map,
    #[token("option")]
    Option_,
    #[token("json")]
    Json,

    // Literals
    #[token("true")]
    True,
    #[token("false")]
    False,
    #[token("null")]
    Null,

    // Punctuation
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
    #[token("<")]
    Lt,
    #[token(">")]
    Gt,
    #[token(":")]
    Colon,
    #[token(",")]
    Comma,
    #[token(".")]
    Dot,
    #[token("=")]
    Eq,
    #[token("?")]
    Question,

    // Operators
    #[token("->")]
    Arrow,
    #[token("==")]
    EqEq,
    #[token("!=")]
    Ne,
    #[token("<=")]
    Le,
    #[token(">=")]
    Ge,
    #[token("&&")]
    AndAnd,
    #[token("||")]
    OrOr,
    #[token("|>")]
    Pipe,
    #[token("+")]
    Plus,
    #[token("-")]
    Minus,
    #[token("*")]
    Star,
    #[token("/")]
    Slash,

    // Duration suffixes are handled in parser
    // Size suffixes are handled in parser

    // Identifier
    #[regex(r"[a-zA-Z_][a-zA-Z0-9_]*", |lex| lex.slice().to_string())]
    Ident(String),

    // Integer literal
    #[regex(r"[0-9]+", |lex| lex.slice().parse::<i64>().ok())]
    IntLit(i64),

    // Float literal
    #[regex(r"[0-9]+\.[0-9]+", |lex| lex.slice().parse::<f64>().ok())]
    FloatLit(f64),

    // String literal
    #[regex(r#""([^"\\]|\\.)*""#, |lex| {
        let s = lex.slice();
        // Remove quotes and unescape
        Some(unescape_string(&s[1..s.len()-1]))
    })]
    StringLit(String),
}

fn unescape_string(s: &str) -> String {
    let mut result = String::new();
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\\' {
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
            result.push(c);
        }
    }
    result
}

impl std::fmt::Display for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Token::Task => write!(f, "task"),
            Token::Artifact => write!(f, "artifact"),
            Token::Artifacts => write!(f, "artifacts"),
            Token::Type => write!(f, "type"),
            Token::Input => write!(f, "input"),
            Token::Output => write!(f, "output"),
            Token::State => write!(f, "state"),
            Token::Decompose => write!(f, "decompose"),
            Token::Subgoal => write!(f, "subgoal"),
            Token::Pre => write!(f, "pre"),
            Token::Post => write!(f, "post"),
            Token::Options => write!(f, "options"),
            Token::Done => write!(f, "done"),
            Token::Reward => write!(f, "reward"),
            Token::Timeout => write!(f, "timeout"),
            Token::Verify => write!(f, "verify"),
            Token::OnFail => write!(f, "on_fail"),
            Token::OnError => write!(f, "on_error"),
            Token::Tool => write!(f, "tool"),
            Token::Stage => write!(f, "stage"),
            Token::Using => write!(f, "using"),
            Token::When => write!(f, "when"),
            Token::Emit => write!(f, "emit"),
            Token::Harness => write!(f, "harness"),
            Token::Defaults => write!(f, "defaults"),
            Token::Bind => write!(f, "bind"),
            Token::Tune => write!(f, "tune"),
            Token::Objective => write!(f, "objective"),
            Token::Dataset => write!(f, "dataset"),
            Token::Constraint => write!(f, "constraint"),
            Token::Checker => write!(f, "checker"),
            Token::Judge => write!(f, "judge"),
            Token::Metric => write!(f, "metric"),
            Token::Score => write!(f, "score"),
            Token::Split => write!(f, "split"),
            Token::Select => write!(f, "select"),
            Token::Repeats => write!(f, "repeats"),
            Token::Train => write!(f, "train"),
            Token::Val => write!(f, "val"),
            Token::Test => write!(f, "test"),
            Token::Primary => write!(f, "primary"),
            Token::TieBreakers => write!(f, "tie_breakers"),
            Token::Carry => write!(f, "carry"),
            Token::Until => write!(f, "until"),
            Token::SubsetOf => write!(f, "subset_of"),
            Token::MaxIters => write!(f, "max_iters"),
            Token::Prompt => write!(f, "prompt"),
            Token::Agent => write!(f, "agent"),
            Token::Pipeline => write!(f, "pipeline"),
            Token::System => write!(f, "system"),
            Token::Template => write!(f, "template"),
            Token::Tools => write!(f, "tools"),
            Token::MaxTurns => write!(f, "max_turns"),
            Token::Model => write!(f, "model"),
            Token::File => write!(f, "file"),
            Token::Extern => write!(f, "extern"),
            Token::Crate => write!(f, "crate"),
            Token::Foreign => write!(f, "foreign"),
            Token::Fn => write!(f, "fn"),
            Token::Impl => write!(f, "impl"),
            Token::Spec => write!(f, "spec"),
            Token::Variants => write!(f, "variants"),
            Token::Pure => write!(f, "pure"),
            Token::Sequence => write!(f, "sequence"),
            Token::Parallel => write!(f, "parallel"),
            Token::Let => write!(f, "let"),
            Token::For => write!(f, "for"),
            Token::In => write!(f, "in"),
            Token::While => write!(f, "while"),
            Token::Loop => write!(f, "loop"),
            Token::Break => write!(f, "break"),
            Token::Continue => write!(f, "continue"),
            Token::If => write!(f, "if"),
            Token::Else => write!(f, "else"),
            Token::Match => write!(f, "match"),
            Token::Result_ => write!(f, "result"),
            Token::Bytes => write!(f, "bytes"),
            Token::Reachable => write!(f, "reachable"),
            Token::NoDeadlock => write!(f, "no_deadlock"),
            Token::Bounded => write!(f, "bounded"),
            Token::Grounded => write!(f, "grounded"),
            Token::Terminates => write!(f, "terminates"),
            Token::Retry => write!(f, "retry"),
            Token::Rollback => write!(f, "rollback"),
            Token::Abort => write!(f, "abort"),
            Token::Replan => write!(f, "replan"),
            Token::Bool => write!(f, "bool"),
            Token::Int => write!(f, "int"),
            Token::Float => write!(f, "float"),
            Token::String_ => write!(f, "string"),
            Token::Any => write!(f, "any"),
            Token::List => write!(f, "list"),
            Token::Map => write!(f, "map"),
            Token::Option_ => write!(f, "option"),
            Token::Json => write!(f, "json"),
            Token::True => write!(f, "true"),
            Token::False => write!(f, "false"),
            Token::Null => write!(f, "null"),
            Token::LBrace => write!(f, "{{"),
            Token::RBrace => write!(f, "}}"),
            Token::LParen => write!(f, "("),
            Token::RParen => write!(f, ")"),
            Token::LBracket => write!(f, "["),
            Token::RBracket => write!(f, "]"),
            Token::Lt => write!(f, "<"),
            Token::Gt => write!(f, ">"),
            Token::Colon => write!(f, ":"),
            Token::Comma => write!(f, ","),
            Token::Dot => write!(f, "."),
            Token::Eq => write!(f, "="),
            Token::Question => write!(f, "?"),
            Token::Arrow => write!(f, "->"),
            Token::EqEq => write!(f, "=="),
            Token::Ne => write!(f, "!="),
            Token::Le => write!(f, "<="),
            Token::Ge => write!(f, ">="),
            Token::AndAnd => write!(f, "&&"),
            Token::OrOr => write!(f, "||"),
            Token::Pipe => write!(f, "|>"),
            Token::Plus => write!(f, "+"),
            Token::Minus => write!(f, "-"),
            Token::Star => write!(f, "*"),
            Token::Slash => write!(f, "/"),
            Token::Ident(s) => write!(f, "{}", s),
            Token::IntLit(n) => write!(f, "{}", n),
            Token::FloatLit(n) => write!(f, "{}", n),
            Token::StringLit(s) => write!(f, "\"{}\"", s),
        }
    }
}

/// A token with its source span
#[derive(Debug, Clone)]
pub struct SpannedToken {
    pub token: Token,
    pub span: Span,
}

/// Lexer that produces tokens with spans
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

    pub fn next_token(&mut self) -> Option<SpannedToken> {
        if let Some(peeked) = self.peeked.take() {
            return peeked;
        }
        self.advance()
    }

    pub fn peek(&mut self) -> Option<&SpannedToken> {
        if self.peeked.is_none() {
            self.peeked = Some(self.advance());
        }
        self.peeked.as_ref().and_then(|opt| opt.as_ref())
    }

    fn advance(&mut self) -> Option<SpannedToken> {
        loop {
            match self.inner.next() {
                Some(Ok(token)) => {
                    let span = self.inner.span();
                    return Some(SpannedToken {
                        token,
                        span: Span::new(span.start, span.end),
                    });
                }
                Some(Err(_)) => {
                    // Skip invalid tokens for now, parser will handle errors
                    continue;
                }
                None => return None,
            }
        }
    }

    pub fn source(&self) -> &'source str {
        self.inner.source()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_tokens() {
        let source = "task foo { input: int }";
        let mut lexer = Lexer::new(source);

        assert!(matches!(lexer.next_token().unwrap().token, Token::Task));
        assert!(matches!(lexer.next_token().unwrap().token, Token::Ident(s) if s == "foo"));
        assert!(matches!(lexer.next_token().unwrap().token, Token::LBrace));
        assert!(matches!(lexer.next_token().unwrap().token, Token::Input));
        assert!(matches!(lexer.next_token().unwrap().token, Token::Colon));
        assert!(matches!(lexer.next_token().unwrap().token, Token::Int));
        assert!(matches!(lexer.next_token().unwrap().token, Token::RBrace));
        assert!(lexer.next_token().is_none());
    }

    #[test]
    fn test_string_literal() {
        let source = r#""hello world""#;
        let mut lexer = Lexer::new(source);

        assert!(matches!(
            lexer.next_token().unwrap().token,
            Token::StringLit(s) if s == "hello world"
        ));
    }

    #[test]
    fn test_operators() {
        let source = "== != <= >= && || -> + - * /";
        let mut lexer = Lexer::new(source);

        assert!(matches!(lexer.next_token().unwrap().token, Token::EqEq));
        assert!(matches!(lexer.next_token().unwrap().token, Token::Ne));
        assert!(matches!(lexer.next_token().unwrap().token, Token::Le));
        assert!(matches!(lexer.next_token().unwrap().token, Token::Ge));
        assert!(matches!(lexer.next_token().unwrap().token, Token::AndAnd));
        assert!(matches!(lexer.next_token().unwrap().token, Token::OrOr));
        assert!(matches!(lexer.next_token().unwrap().token, Token::Arrow));
        assert!(matches!(lexer.next_token().unwrap().token, Token::Plus));
        assert!(matches!(lexer.next_token().unwrap().token, Token::Minus));
        assert!(matches!(lexer.next_token().unwrap().token, Token::Star));
        assert!(matches!(lexer.next_token().unwrap().token, Token::Slash));
    }
}
