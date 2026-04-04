//! Recursive descent parser for Scaffold v2

use crate::ast::*;
use crate::lexer::{Lexer, SpannedToken, Token};

/// A parse error with location and message
#[derive(Debug, Clone)]
pub struct ParseError {
    pub message: String,
    pub span: Span,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "parse error at {:?}: {}", self.span, self.message)
    }
}

impl std::error::Error for ParseError {}

pub type ParseResult<T> = Result<T, ParseError>;

/// Convenience function to parse a complete program
pub fn parse(source: &str) -> ParseResult<Program> {
    let mut parser = Parser::new(source);
    parser.parse_program()
}

/// Parsed objective fields (shared between objective and sub-objective).
struct ObjectiveFields {
    graph: Option<Ident>,
    dataset: Option<DatasetSpec>,
    checkers: Vec<CheckerDecl>,
    judges: Vec<JudgeDecl>,
    metrics: Vec<MetricDecl>,
    score: Option<Spanned<Expr>>,
    repeats: Option<u64>,
    split: Option<SplitDecl>,
    select: Option<SelectDecl>,
    tunables: Vec<TuneStmt>,
    topology: Option<TopologyDecl>,
    subs: Vec<SubObjectiveDecl>,
}

pub struct Parser<'src> {
    lexer: Lexer<'src>,
    source: &'src str,
    last_span: Span,
}

impl<'src> Parser<'src> {
    pub fn new(source: &'src str) -> Self {
        Self {
            lexer: Lexer::new(source),
            source,
            last_span: 0..0,
        }
    }

    // ── Helpers ──

    fn peek(&mut self) -> Option<&Token> {
        self.lexer.peek().map(|(tok, _)| tok)
    }

    fn peek_span(&mut self) -> Span {
        self.lexer
            .peek()
            .map(|(_, s)| s.clone())
            .unwrap_or_else(|| {
                let end = self.source.len();
                end..end
            })
    }

    fn advance(&mut self) -> Option<SpannedToken> {
        let tok = self.lexer.next_token();
        if let Some((_, ref span)) = tok {
            self.last_span = span.clone();
        }
        tok
    }

    fn expect(&mut self, expected: &Token) -> ParseResult<Span> {
        match self.advance() {
            Some((tok, span)) if &tok == expected => Ok(span),
            Some((tok, span)) => Err(ParseError {
                message: format!("expected '{}', found '{}'", expected, tok),
                span,
            }),
            None => Err(ParseError {
                message: format!("expected '{}', found end of input", expected),
                span: self.last_span.clone(),
            }),
        }
    }

    fn expect_ident(&mut self) -> ParseResult<Ident> {
        match self.advance() {
            Some((Token::Ident(name), span)) => Ok(Ident::new(name, span)),
            // Also accept keywords-as-identifiers in specific contexts
            Some((tok, span)) if is_keyword_ident(&tok) => {
                Ok(Ident::new(tok.to_string(), span))
            }
            Some((tok, span)) => Err(ParseError {
                message: format!("expected identifier, found '{}'", tok),
                span,
            }),
            None => Err(ParseError {
                message: "expected identifier, found end of input".into(),
                span: self.last_span.clone(),
            }),
        }
    }

    /// Parse a possibly-dotted identifier like `_motif.result.shortlist`.
    /// Consumes `Ident(.Ident)*` and joins with dots.
    /// Parse a possibly-dotted identifier like `_motif.result.shortlist`.
    /// Consumes `Ident(.Ident)*` where each segment can be an identifier or keyword-as-ident.
    fn expect_dotted_ident(&mut self) -> ParseResult<Ident> {
        let first = self.expect_ident()?;
        let start = first.span.start;
        let mut name = first.name;
        while self.peek() == Some(&Token::Dot) {
            // Check if the dot is followed by an ident or keyword-as-ident
            let is_ident_after = self.peek_is_ident_after_dot();
            if !is_ident_after {
                break;
            }
            self.advance(); // consume Dot
            let next = self.expect_ident()?;
            name.push('.');
            name.push_str(&next.name);
        }
        let end = self.last_span.end;
        Ok(Ident::new(name, start..end))
    }

    /// Peek past the current Dot token to check if what follows is an identifier
    /// (regular ident or keyword-as-ident). Used by expect_dotted_ident to avoid
    /// consuming dots that are part of field access syntax.
    fn peek_is_ident_after_dot(&mut self) -> bool {
        // We know current peek is Dot. We need to look 2 tokens ahead.
        // Since our lexer only supports single-token peek, we check if the
        // token after dot would be accepted by expect_ident (Ident or keyword-as-ident).
        // For now, always consume — if the dot is followed by something that
        // expect_ident can parse, it works. If not, we get a parse error which is
        // acceptable since dotted names in node/step contexts should always be valid.
        true
    }

    fn expect_string(&mut self) -> ParseResult<(String, Span)> {
        match self.advance() {
            Some((Token::StringLit(s), span)) => Ok((s, span)),
            Some((tok, span)) => Err(ParseError {
                message: format!("expected string literal, found '{}'", tok),
                span,
            }),
            None => Err(ParseError {
                message: "expected string literal, found end of input".into(),
                span: self.last_span.clone(),
            }),
        }
    }

    fn expect_int(&mut self) -> ParseResult<(i64, Span)> {
        match self.advance() {
            Some((Token::IntLit(n), span)) => Ok((n, span)),
            Some((tok, span)) => Err(ParseError {
                message: format!("expected integer literal, found '{}'", tok),
                span,
            }),
            None => Err(ParseError {
                message: "expected integer literal, found end of input".into(),
                span: self.last_span.clone(),
            }),
        }
    }

    fn expect_float_or_int(&mut self) -> ParseResult<(f64, Span)> {
        match self.advance() {
            Some((Token::FloatLit(f), span)) => Ok((f, span)),
            Some((Token::IntLit(i), span)) => Ok((i as f64, span)),
            Some((tok, span)) => Err(ParseError {
                message: format!("expected number, found '{}'", tok),
                span,
            }),
            None => Err(ParseError {
                message: "expected number, found end of input".into(),
                span: self.last_span.clone(),
            }),
        }
    }

    fn eat(&mut self, expected: &Token) -> bool {
        if self.peek() == Some(expected) {
            self.advance();
            true
        } else {
            false
        }
    }

    fn at(&mut self, expected: &Token) -> bool {
        self.peek() == Some(expected)
    }

    // ── Program ──

    pub fn parse_program(&mut self) -> ParseResult<Program> {
        let mut declarations = Vec::new();
        while self.peek().is_some() {
            declarations.push(self.parse_declaration()?);
        }
        Ok(Program { declarations })
    }

    fn parse_declaration(&mut self) -> ParseResult<Declaration> {
        match self.peek() {
            Some(Token::Type) => Ok(Declaration::Type(self.parse_type_decl()?)),
            Some(Token::Node) => Ok(Declaration::Node(self.parse_node_decl()?)),
            Some(Token::Graph) => Ok(Declaration::Graph(self.parse_graph_decl()?)),
            Some(Token::Objective) => Ok(Declaration::Objective(self.parse_objective_decl()?)),
            Some(_) => {
                let (tok, span) = self.advance().unwrap();
                Err(ParseError {
                    message: format!(
                        "expected 'type', 'node', 'graph', or 'objective', found '{}'",
                        tok
                    ),
                    span,
                })
            }
            None => Err(ParseError {
                message: "unexpected end of input".into(),
                span: self.last_span.clone(),
            }),
        }
    }

    // ── Type declarations ──

    fn parse_type_decl(&mut self) -> ParseResult<TypeDecl> {
        let start = self.expect(&Token::Type)?;
        let name = self.expect_ident()?;
        self.expect(&Token::Eq)?;
        let ty = self.parse_type_expr()?;
        Ok(TypeDecl {
            span: start.start..ty.span.end,
            name,
            ty,
        })
    }

    fn parse_type_expr(&mut self) -> ParseResult<Spanned<TypeExpr>> {
        let start = self.peek_span();
        match self.peek() {
            Some(Token::Bool) => {
                let span = self.advance().unwrap().1;
                Ok(Spanned::new(TypeExpr::Primitive(PrimitiveType::Bool), span))
            }
            Some(Token::KwInt) => {
                let span = self.advance().unwrap().1;
                Ok(Spanned::new(TypeExpr::Primitive(PrimitiveType::Int), span))
            }
            Some(Token::KwFloat) => {
                let span = self.advance().unwrap().1;
                Ok(Spanned::new(
                    TypeExpr::Primitive(PrimitiveType::Float),
                    span,
                ))
            }
            Some(Token::KwString) => {
                let span = self.advance().unwrap().1;
                Ok(Spanned::new(
                    TypeExpr::Primitive(PrimitiveType::String),
                    span,
                ))
            }
            Some(Token::Bytes) => {
                let span = self.advance().unwrap().1;
                Ok(Spanned::new(
                    TypeExpr::Primitive(PrimitiveType::Bytes),
                    span,
                ))
            }
            Some(Token::Any) => {
                let span = self.advance().unwrap().1;
                Ok(Spanned::new(TypeExpr::Primitive(PrimitiveType::Any), span))
            }
            Some(Token::List) => {
                self.advance();
                self.expect(&Token::Lt)?;
                let inner = self.parse_type_expr()?;
                let end = self.expect(&Token::Gt)?;
                Ok(Spanned::new(
                    TypeExpr::List(Box::new(inner)),
                    start.start..end.end,
                ))
            }
            Some(Token::Map) => {
                self.advance();
                self.expect(&Token::Lt)?;
                let key = self.parse_type_expr()?;
                self.expect(&Token::Comma)?;
                let val = self.parse_type_expr()?;
                let end = self.expect(&Token::Gt)?;
                Ok(Spanned::new(
                    TypeExpr::Map(Box::new(key), Box::new(val)),
                    start.start..end.end,
                ))
            }
            Some(Token::Option) => {
                self.advance();
                self.expect(&Token::Lt)?;
                let inner = self.parse_type_expr()?;
                let end = self.expect(&Token::Gt)?;
                Ok(Spanned::new(
                    TypeExpr::Option(Box::new(inner)),
                    start.start..end.end,
                ))
            }
            Some(Token::LBrace) => {
                self.advance();
                let mut fields = Vec::new();
                while !self.at(&Token::RBrace) {
                    let fname = self.expect_ident()?;
                    self.expect(&Token::Colon)?;
                    let fty = self.parse_type_expr()?;
                    fields.push(FieldDecl { name: fname, ty: fty });
                    if !self.eat(&Token::Comma) {
                        break;
                    }
                }
                let end = self.expect(&Token::RBrace)?;
                Ok(Spanned::new(
                    TypeExpr::Struct(fields),
                    start.start..end.end,
                ))
            }
            Some(Token::Ident(_)) => {
                let ident = self.expect_ident()?;
                let span = ident.span.clone();
                Ok(Spanned::new(TypeExpr::Named(ident.name), span))
            }
            _ => {
                let span = self.peek_span();
                Err(ParseError {
                    message: "expected type expression".into(),
                    span,
                })
            }
        }
    }

    // ── Node declarations ──

    fn parse_node_decl(&mut self) -> ParseResult<NodeDecl> {
        let start = self.expect(&Token::Node)?;
        let name = self.expect_dotted_ident()?;
        self.expect(&Token::Colon)?;
        let kind = self.parse_node_kind()?;
        self.expect(&Token::LBrace)?;

        let mut input = None;
        let mut output = None;
        let mut config = NodeConfig::default();

        while !self.at(&Token::RBrace) {
            match self.peek() {
                Some(Token::In) => {
                    self.advance();
                    self.expect(&Token::Colon)?;
                    input = Some(self.parse_type_expr()?);
                }
                Some(Token::Out) => {
                    self.advance();
                    self.expect(&Token::Colon)?;
                    output = Some(self.parse_type_expr()?);
                }
                Some(Token::Template) => {
                    self.advance();
                    self.expect(&Token::Colon)?;
                    config.template = Some(self.parse_string_or_file()?);
                }
                Some(Token::System) => {
                    self.advance();
                    self.expect(&Token::Colon)?;
                    config.system = Some(self.parse_string_or_file()?);
                }
                Some(Token::Model) => {
                    self.advance();
                    self.expect(&Token::Colon)?;
                    let (s, _) = self.expect_string()?;
                    config.model = Some(s);
                }
                Some(Token::Temperature) => {
                    self.advance();
                    self.expect(&Token::Colon)?;
                    let (v, _) = self.expect_float_or_int()?;
                    config.temperature = Some(v);
                }
                Some(Token::MaxTokens) => {
                    self.advance();
                    self.expect(&Token::Colon)?;
                    let (v, _) = self.expect_int()?;
                    config.max_tokens = Some(v as u64);
                }
                Some(Token::Tools) => {
                    self.advance();
                    self.expect(&Token::Colon)?;
                    self.expect(&Token::LBracket)?;
                    let mut tools = Vec::new();
                    while !self.at(&Token::RBracket) {
                        tools.push(self.expect_ident()?);
                        if !self.eat(&Token::Comma) {
                            break;
                        }
                    }
                    self.expect(&Token::RBracket)?;
                    config.tools = tools;
                }
                Some(Token::MaxTurns) => {
                    self.advance();
                    self.expect(&Token::Colon)?;
                    let (v, _) = self.expect_int()?;
                    config.max_turns = Some(v as u64);
                }
                Some(Token::Timeout) => {
                    self.advance();
                    self.expect(&Token::Colon)?;
                    let (v, _) = self.expect_int()?;
                    config.timeout = Some(v as u64);
                }
                Some(Token::OnError) => {
                    self.advance();
                    self.expect(&Token::Colon)?;
                    config.on_error = Some(self.parse_error_strategy()?);
                }
                Some(Token::Shell) => {
                    self.advance();
                    self.expect(&Token::Colon)?;
                    let (s, _) = self.expect_string()?;
                    config.shell = Some(s);
                }
                Some(Token::Json) => {
                    self.advance();
                    self.expect(&Token::Colon)?;
                    self.expect(&Token::LBrace)?;
                    let mut fields = Vec::new();
                    while !self.at(&Token::RBrace) {
                        let fstart = self.peek_span();
                        let (key, _) = self.expect_string()?;
                        self.expect(&Token::Colon)?;
                        let value = self.parse_expr()?;
                        let fend = value.span.end;
                        fields.push(JsonField {
                            key,
                            value,
                            span: fstart.start..fend,
                        });
                        if !self.eat(&Token::Comma) {
                            break;
                        }
                    }
                    self.expect(&Token::RBrace)?;
                    config.json = Some(fields);
                }
                _ => {
                    let span = self.peek_span();
                    return Err(ParseError {
                        message: "expected node field (in, out, template, model, ...)".into(),
                        span,
                    });
                }
            }
        }

        let end = self.expect(&Token::RBrace)?;

        let input = input.ok_or_else(|| ParseError {
            message: format!("node '{}' missing 'in' type", name.name),
            span: name.span.clone(),
        })?;
        let output = output.ok_or_else(|| ParseError {
            message: format!("node '{}' missing 'out' type", name.name),
            span: name.span.clone(),
        })?;

        Ok(NodeDecl {
            span: start.start..end.end,
            name,
            kind,
            input,
            output,
            config,
        })
    }

    fn parse_node_kind(&mut self) -> ParseResult<Spanned<NodeKind>> {
        match self.advance() {
            Some((Token::Prompt, span)) => Ok(Spanned::new(NodeKind::Prompt, span)),
            Some((Token::Tool, span)) => Ok(Spanned::new(NodeKind::Tool, span)),
            Some((Token::Agent, span)) => Ok(Spanned::new(NodeKind::Agent, span)),
            Some((Token::Verify, span)) => Ok(Spanned::new(NodeKind::Verify, span)),
            Some((tok, span)) => Err(ParseError {
                message: format!(
                    "expected node kind (prompt, tool, agent, verify), found '{}'",
                    tok
                ),
                span,
            }),
            None => Err(ParseError {
                message: "expected node kind, found end of input".into(),
                span: self.last_span.clone(),
            }),
        }
    }

    fn parse_string_or_file(&mut self) -> ParseResult<StringOrFile> {
        if self.eat(&Token::File) {
            self.expect(&Token::LParen)?;
            let (s, _) = self.expect_string()?;
            self.expect(&Token::RParen)?;
            Ok(StringOrFile::File(s))
        } else {
            let (s, _) = self.expect_string()?;
            Ok(StringOrFile::Literal(s))
        }
    }

    fn parse_error_strategy(&mut self) -> ParseResult<ErrorStrategy> {
        match self.peek() {
            Some(Token::Abort) => {
                self.advance();
                Ok(ErrorStrategy::Abort)
            }
            Some(Token::Retry) => {
                self.advance();
                self.expect(&Token::LParen)?;
                let (n, _) = self.expect_int()?;
                self.expect(&Token::RParen)?;
                Ok(ErrorStrategy::Retry(n as u64))
            }
            _ => {
                let span = self.peek_span();
                Err(ParseError {
                    message: "expected 'abort' or 'retry(N)'".into(),
                    span,
                })
            }
        }
    }

    // ── Graph declarations ──

    fn parse_graph_decl(&mut self) -> ParseResult<GraphDecl> {
        let start = self.expect(&Token::Graph)?;
        let name = self.expect_ident()?;
        self.expect(&Token::LBrace)?;

        // in: type
        self.expect(&Token::In)?;
        self.expect(&Token::Colon)?;
        let input = self.parse_type_expr()?;

        // out: type
        self.expect(&Token::Out)?;
        self.expect(&Token::Colon)?;
        let output = self.parse_type_expr()?;

        // body
        let mut body = Vec::new();
        while !self.at(&Token::RBrace) {
            body.push(self.parse_graph_stmt()?);
        }
        let end = self.expect(&Token::RBrace)?;

        Ok(GraphDecl {
            span: start.start..end.end,
            name,
            input,
            output,
            body,
        })
    }

    fn parse_graph_stmt(&mut self) -> ParseResult<GraphStmt> {
        match self.peek() {
            Some(Token::Step) => Ok(GraphStmt::Step(self.parse_step_stmt()?)),
            Some(Token::Loop) => Ok(GraphStmt::Loop(self.parse_loop_stmt()?)),
            Some(Token::If) => Ok(GraphStmt::If(self.parse_if_stmt()?)),
            Some(Token::Choose) => Ok(GraphStmt::Choose(self.parse_choose_stmt()?)),
            Some(Token::Parallel) => Ok(GraphStmt::Parallel(self.parse_parallel_stmt()?)),
            Some(Token::Emit) => Ok(GraphStmt::Emit(self.parse_emit_stmt()?)),
            Some(Token::Carry) => Ok(GraphStmt::Carry(self.parse_carry_stmt()?)),
            _ => {
                let span = self.peek_span();
                Err(ParseError {
                    message: "expected graph statement (step, loop, if, choose, parallel, emit, carry)".into(),
                    span,
                })
            }
        }
    }

    fn parse_step_stmt(&mut self) -> ParseResult<StepStmt> {
        let start = self.expect(&Token::Step)?;
        let name = self.expect_ident()?;
        self.expect(&Token::Eq)?;
        let node = self.expect_dotted_ident()?;
        self.expect(&Token::LParen)?;

        let mut args = Vec::new();
        while !self.at(&Token::RParen) {
            args.push(self.parse_step_arg()?);
            if !self.eat(&Token::Comma) {
                break;
            }
        }
        let end = self.expect(&Token::RParen)?;

        Ok(StepStmt {
            span: start.start..end.end,
            name,
            node,
            args,
        })
    }

    fn parse_step_arg(&mut self) -> ParseResult<StepArg> {
        // Try named: IDENT ":" expr
        // We need lookahead to distinguish named from positional
        // Also accept keyword-as-ident (e.g. `input:`) for named args
        if self.peek_is_ident_like() {
            // Check if next-next is ':'
            // Use a simple approach: parse expr, and if the expr is an ident followed by :, it's named
            let start = self.peek_span();
            let expr = self.parse_expr()?;

            // If we just parsed an Ident and the next token is ':', treat as named arg
            if let Expr::Ident(ref name) = expr.node {
                if self.eat(&Token::Colon) {
                    let ident = Ident::new(name.clone(), start);
                    let value = self.parse_expr()?;
                    return Ok(StepArg::Named { name: ident, value });
                }
            }

            Ok(StepArg::Positional(expr))
        } else {
            let expr = self.parse_expr()?;
            Ok(StepArg::Positional(expr))
        }
    }

    fn parse_loop_stmt(&mut self) -> ParseResult<LoopStmt> {
        let start = self.expect(&Token::Loop)?;
        self.expect(&Token::LParen)?;

        // max: expr
        self.expect(&Token::Max)?;
        self.expect(&Token::Colon)?;
        let max = self.parse_expr()?;

        self.expect(&Token::Comma)?;

        // while: expr
        self.expect(&Token::While)?;
        self.expect(&Token::Colon)?;
        let while_cond = self.parse_expr()?;

        self.expect(&Token::RParen)?;
        self.expect(&Token::LBrace)?;

        let mut body = Vec::new();
        while !self.at(&Token::RBrace) {
            body.push(self.parse_graph_stmt()?);
        }
        let end = self.expect(&Token::RBrace)?;

        Ok(LoopStmt {
            span: start.start..end.end,
            max,
            while_cond,
            body,
        })
    }

    fn parse_if_stmt(&mut self) -> ParseResult<IfStmt> {
        let start = self.expect(&Token::If)?;
        let cond = self.parse_expr()?;
        self.expect(&Token::LBrace)?;

        let mut then_body = Vec::new();
        while !self.at(&Token::RBrace) {
            then_body.push(self.parse_graph_stmt()?);
        }
        self.expect(&Token::RBrace)?;

        let mut else_body = Vec::new();
        if self.eat(&Token::Else) {
            self.expect(&Token::LBrace)?;
            while !self.at(&Token::RBrace) {
                else_body.push(self.parse_graph_stmt()?);
            }
            self.expect(&Token::RBrace)?;
        }

        let end_pos = self.last_span.end;

        Ok(IfStmt {
            span: start.start..end_pos,
            cond,
            then_body,
            else_body,
        })
    }

    fn parse_choose_stmt(&mut self) -> ParseResult<ChooseStmt> {
        let start = self.expect(&Token::Choose)?;
        self.expect(&Token::LBracket)?;
        let mut alternatives = Vec::new();
        while !self.at(&Token::RBracket) {
            alternatives.push(self.expect_ident()?);
            if !self.eat(&Token::Comma) {
                break;
            }
        }
        let end = self.expect(&Token::RBracket)?;

        Ok(ChooseStmt {
            span: start.start..end.end,
            alternatives,
        })
    }

    fn parse_parallel_stmt(&mut self) -> ParseResult<ParallelStmt> {
        let start = self.expect(&Token::Parallel)?;
        self.expect(&Token::LParen)?;
        let var = self.expect_ident()?;
        self.expect(&Token::In)?;
        let collection = self.parse_expr()?;

        let reduce = if self.eat(&Token::Comma) {
            self.expect(&Token::Reduce)?;
            self.expect(&Token::Colon)?;
            Some(self.expect_ident()?)
        } else {
            None
        };

        self.expect(&Token::RParen)?;
        self.expect(&Token::LBrace)?;

        let mut body = Vec::new();
        while !self.at(&Token::RBrace) {
            body.push(self.parse_graph_stmt()?);
        }
        let end = self.expect(&Token::RBrace)?;

        Ok(ParallelStmt {
            span: start.start..end.end,
            var,
            collection,
            reduce,
            body,
        })
    }

    fn parse_emit_stmt(&mut self) -> ParseResult<EmitStmt> {
        let start = self.expect(&Token::Emit)?;

        if self.at(&Token::LBrace) {
            self.advance();
            let mut fields = Vec::new();
            while !self.at(&Token::RBrace) {
                let name = self.expect_ident()?;
                self.expect(&Token::Colon)?;
                let value = self.parse_expr()?;
                fields.push(EmitField { name, value });
                if !self.eat(&Token::Comma) {
                    break;
                }
            }
            let end = self.expect(&Token::RBrace)?;
            Ok(EmitStmt::Record {
                fields,
                span: start.start..end.end,
            })
        } else {
            let value = self.parse_expr()?;
            let end = value.span.end;
            Ok(EmitStmt::Direct {
                value,
                span: start.start..end,
            })
        }
    }

    fn parse_carry_stmt(&mut self) -> ParseResult<CarryStmt> {
        let start = self.expect(&Token::Carry)?;
        let name = self.expect_ident()?;
        self.expect(&Token::Eq)?;
        let value = self.parse_expr()?;
        let end = value.span.end;
        Ok(CarryStmt {
            span: start.start..end,
            name,
            value,
        })
    }

    // ── Objective declarations ──

    /// Parse the body fields of an objective or sub-objective.
    /// When `allow_subs` is false, `sub` blocks are rejected (for sub-objectives).
    fn parse_objective_fields(&mut self, allow_subs: bool) -> ParseResult<ObjectiveFields> {
        let mut fields = ObjectiveFields {
            graph: None,
            dataset: None,
            checkers: Vec::new(),
            judges: Vec::new(),
            metrics: Vec::new(),
            score: None,
            repeats: None,
            split: None,
            select: None,
            tunables: Vec::new(),
            topology: None,
            subs: Vec::new(),
        };

        while !self.at(&Token::RBrace) {
            match self.peek() {
                Some(Token::Graph) => {
                    self.advance();
                    self.expect(&Token::Colon)?;
                    fields.graph = Some(self.expect_ident()?);
                }
                Some(Token::Dataset) => {
                    self.advance();
                    self.expect(&Token::Colon)?;
                    fields.dataset = Some(self.parse_dataset_spec()?);
                }
                Some(Token::Checker) => {
                    self.advance();
                    let cname = self.expect_ident()?;
                    self.expect(&Token::LBrace)?;
                    let expr = self.parse_expr()?;
                    let cend = self.expect(&Token::RBrace)?;
                    fields.checkers.push(CheckerDecl {
                        span: cname.span.start..cend.end,
                        name: cname,
                        expr,
                    });
                }
                Some(Token::Judge) => {
                    self.advance();
                    let jname = self.expect_ident()?;
                    self.expect(&Token::LBrace)?;

                    let mut jmodel = None;
                    let mut jtemplate = None;
                    let mut jrubric = None;

                    while !self.at(&Token::RBrace) {
                        match self.peek() {
                            Some(Token::Model) => {
                                self.advance();
                                self.expect(&Token::Colon)?;
                                let (s, _) = self.expect_string()?;
                                jmodel = Some(s);
                            }
                            Some(Token::Template) => {
                                self.advance();
                                self.expect(&Token::Colon)?;
                                jtemplate = Some(self.parse_string_or_file()?);
                            }
                            Some(Token::Rubric) => {
                                self.advance();
                                self.expect(&Token::Colon)?;
                                jrubric = Some(self.parse_string_or_file()?);
                            }
                            _ => {
                                let span = self.peek_span();
                                return Err(ParseError {
                                    message: "expected judge field (model, template, rubric)".into(),
                                    span,
                                });
                            }
                        }
                    }
                    let jend = self.expect(&Token::RBrace)?;
                    fields.judges.push(JudgeDecl {
                        span: jname.span.start..jend.end,
                        name: jname,
                        model: jmodel,
                        template: jtemplate,
                        rubric: jrubric,
                    });
                }
                Some(Token::Metric) => {
                    self.advance();
                    let mname = self.expect_ident()?;
                    self.expect(&Token::LBrace)?;
                    self.expect(&Token::Checker)?;
                    self.expect(&Token::Colon)?;
                    let checker = self.expect_ident()?;
                    let mend = self.expect(&Token::RBrace)?;
                    fields.metrics.push(MetricDecl {
                        span: mname.span.start..mend.end,
                        name: mname,
                        checker,
                    });
                }
                Some(Token::Score) => {
                    self.advance();
                    self.expect(&Token::Colon)?;
                    fields.score = Some(self.parse_expr()?);
                }
                Some(Token::Repeats) => {
                    self.advance();
                    self.expect(&Token::Colon)?;
                    let (v, _) = self.expect_int()?;
                    fields.repeats = Some(v as u64);
                }
                Some(Token::Split) => {
                    self.advance();
                    self.expect(&Token::LBrace)?;
                    let sstart = self.last_span.clone();

                    self.expect(&Token::Train)?;
                    self.expect(&Token::Colon)?;
                    let (train, _) = self.expect_float_or_int()?;
                    self.expect(&Token::Comma)?;

                    self.expect(&Token::Val)?;
                    self.expect(&Token::Colon)?;
                    let (val, _) = self.expect_float_or_int()?;
                    self.expect(&Token::Comma)?;

                    self.expect(&Token::Test)?;
                    self.expect(&Token::Colon)?;
                    let (test, _) = self.expect_float_or_int()?;

                    // optional trailing comma
                    self.eat(&Token::Comma);

                    let send = self.expect(&Token::RBrace)?;
                    fields.split = Some(SplitDecl {
                        train,
                        val,
                        test,
                        span: sstart.start..send.end,
                    });
                }
                Some(Token::Select) => {
                    self.advance();
                    self.expect(&Token::LBrace)?;
                    let sstart = self.last_span.clone();

                    self.expect(&Token::Primary)?;
                    self.expect(&Token::Colon)?;
                    let primary = self.expect_ident()?;

                    let mut tie_breakers = Vec::new();
                    if self.eat(&Token::Comma) {
                        if self.eat(&Token::TieBreakers) {
                            self.expect(&Token::Colon)?;
                            self.expect(&Token::LBracket)?;
                            while !self.at(&Token::RBracket) {
                                tie_breakers.push(self.expect_ident()?);
                                if !self.eat(&Token::Comma) {
                                    break;
                                }
                            }
                            self.expect(&Token::RBracket)?;
                        }
                    }

                    let send = self.expect(&Token::RBrace)?;
                    fields.select = Some(SelectDecl {
                        primary,
                        tie_breakers,
                        span: sstart.start..send.end,
                    });
                }
                Some(Token::Tune) => {
                    self.advance();
                    self.expect(&Token::LBrace)?;

                    while !self.at(&Token::RBrace) {
                        let tstart = self.peek_span();
                        // parse path: IDENT ("." IDENT)*
                        let mut path = vec![self.expect_ident()?];
                        while self.eat(&Token::Dot) {
                            path.push(self.expect_ident()?);
                        }

                        self.expect(&Token::In)?;
                        self.expect(&Token::LBracket)?;
                        let mut domain = Vec::new();
                        while !self.at(&Token::RBracket) {
                            domain.push(self.parse_expr()?);
                            if !self.eat(&Token::Comma) {
                                break;
                            }
                        }
                        let tend = self.expect(&Token::RBracket)?;

                        fields.tunables.push(TuneStmt {
                            path,
                            domain,
                            span: tstart.start..tend.end,
                        });
                    }
                    self.expect(&Token::RBrace)?;
                }
                Some(Token::Topology) => {
                    self.advance();
                    self.expect(&Token::LBrace)?;
                    let tstart = self.last_span.clone();

                    let mut mutations = Vec::new();
                    let mut max_nodes = None;
                    let mut max_depth = None;
                    let mut preserve = Vec::new();
                    let mut target_score = None;

                    while !self.at(&Token::RBrace) {
                        match self.peek() {
                            Some(Token::Mutations) => {
                                self.advance();
                                self.expect(&Token::Colon)?;
                                self.expect(&Token::LBracket)?;
                                while !self.at(&Token::RBracket) {
                                    let id = self.expect_ident()?;
                                    mutations.push(id.name);
                                    if !self.eat(&Token::Comma) {
                                        break;
                                    }
                                }
                                self.expect(&Token::RBracket)?;
                            }
                            Some(Token::MaxNodes) => {
                                self.advance();
                                self.expect(&Token::Colon)?;
                                let (v, _) = self.expect_int()?;
                                max_nodes = Some(v as u64);
                            }
                            Some(Token::MaxDepth) => {
                                self.advance();
                                self.expect(&Token::Colon)?;
                                let (v, _) = self.expect_int()?;
                                max_depth = Some(v as u64);
                            }
                            Some(Token::Preserve) => {
                                self.advance();
                                self.expect(&Token::Colon)?;
                                self.expect(&Token::LBracket)?;
                                while !self.at(&Token::RBracket) {
                                    let id = self.expect_ident()?;
                                    preserve.push(id.name);
                                    if !self.eat(&Token::Comma) {
                                        break;
                                    }
                                }
                                self.expect(&Token::RBracket)?;
                            }
                            Some(Token::TargetScore) => {
                                self.advance();
                                self.expect(&Token::Colon)?;
                                let (v, _) = self.expect_float_or_int()?;
                                target_score = Some(v);
                            }
                            _ => {
                                let span = self.peek_span();
                                return Err(ParseError {
                                    message: "expected topology field (mutations, max_nodes, max_depth, preserve, target_score)".into(),
                                    span,
                                });
                            }
                        }
                    }

                    let tend = self.expect(&Token::RBrace)?;
                    fields.topology = Some(TopologyDecl {
                        mutations,
                        max_nodes,
                        max_depth,
                        preserve,
                        target_score,
                        span: tstart.start..tend.end,
                    });
                }
                Some(Token::Sub) if allow_subs => {
                    fields.subs.push(self.parse_sub_objective_decl()?);
                }
                Some(Token::Sub) => {
                    let span = self.peek_span();
                    return Err(ParseError {
                        message: "nested sub blocks are not allowed".into(),
                        span,
                    });
                }
                _ => {
                    let span = self.peek_span();
                    return Err(ParseError {
                        message: "expected objective field".into(),
                        span,
                    });
                }
            }
        }

        Ok(fields)
    }

    fn parse_objective_decl(&mut self) -> ParseResult<ObjectiveDecl> {
        let start = self.expect(&Token::Objective)?;
        let name = self.expect_ident()?;
        self.expect(&Token::LBrace)?;

        let fields = self.parse_objective_fields(true)?;
        let end = self.expect(&Token::RBrace)?;

        let graph = fields.graph.ok_or_else(|| ParseError {
            message: format!("objective '{}' missing 'graph' field", name.name),
            span: name.span.clone(),
        })?;
        let dataset = fields.dataset.ok_or_else(|| ParseError {
            message: format!("objective '{}' missing 'dataset' field", name.name),
            span: name.span.clone(),
        })?;
        let score = fields.score.ok_or_else(|| ParseError {
            message: format!("objective '{}' missing 'score' field", name.name),
            span: name.span.clone(),
        })?;

        Ok(ObjectiveDecl {
            span: start.start..end.end,
            name,
            graph,
            dataset,
            checkers: fields.checkers,
            judges: fields.judges,
            metrics: fields.metrics,
            score,
            repeats: fields.repeats,
            split: fields.split,
            select: fields.select,
            tunables: fields.tunables,
            topology: fields.topology,
            subs: fields.subs,
        })
    }

    fn parse_sub_objective_decl(&mut self) -> ParseResult<SubObjectiveDecl> {
        let start = self.expect(&Token::Sub)?;
        let name = self.expect_ident()?;
        self.expect(&Token::LBrace)?;

        let fields = self.parse_objective_fields(false)?;
        let end = self.expect(&Token::RBrace)?;

        let graph = fields.graph.ok_or_else(|| ParseError {
            message: format!("sub '{}' missing 'graph' field", name.name),
            span: name.span.clone(),
        })?;
        let dataset = fields.dataset.ok_or_else(|| ParseError {
            message: format!("sub '{}' missing 'dataset' field", name.name),
            span: name.span.clone(),
        })?;
        let score = fields.score.ok_or_else(|| ParseError {
            message: format!("sub '{}' missing 'score' field", name.name),
            span: name.span.clone(),
        })?;

        Ok(SubObjectiveDecl {
            span: start.start..end.end,
            name,
            graph,
            dataset,
            checkers: fields.checkers,
            judges: fields.judges,
            metrics: fields.metrics,
            score,
            repeats: fields.repeats,
            split: fields.split,
            select: fields.select,
            tunables: fields.tunables,
            topology: fields.topology,
        })
    }

    fn parse_dataset_spec(&mut self) -> ParseResult<DatasetSpec> {
        if self.eat(&Token::File) {
            self.expect(&Token::LParen)?;
            let (s, _) = self.expect_string()?;
            self.expect(&Token::RParen)?;
            Ok(DatasetSpec::File(s))
        } else if self.eat(&Token::Cases) {
            self.expect(&Token::LBracket)?;
            let mut cases = Vec::new();
            while !self.at(&Token::RBracket) {
                cases.push(self.parse_inline_case()?);
                self.eat(&Token::Comma);
            }
            self.expect(&Token::RBracket)?;
            Ok(DatasetSpec::Inline { cases })
        } else {
            let span = self.peek_span();
            Err(ParseError {
                message: "expected 'file(...)' or 'cases [...]'".into(),
                span,
            })
        }
    }

    fn parse_inline_case(&mut self) -> ParseResult<InlineCase> {
        let start = self.expect(&Token::LBrace)?;

        self.expect(&Token::Input)?;
        self.expect(&Token::Colon)?;
        let input = self.parse_expr()?;
        self.expect(&Token::Comma)?;

        self.expect(&Token::Expected)?;
        self.expect(&Token::Colon)?;
        let expected = self.parse_expr()?;

        let mut id = None;
        if self.eat(&Token::Comma) {
            if self.eat(&Token::Id) {
                self.expect(&Token::Colon)?;
                let (s, _) = self.expect_string()?;
                id = Some(s);
            }
            // eat trailing comma
            self.eat(&Token::Comma);
        }

        let end = self.expect(&Token::RBrace)?;

        Ok(InlineCase {
            input,
            expected,
            id,
            span: start.start..end.end,
        })
    }

    // ── Expression parsing (precedence climbing) ──

    pub fn parse_expr(&mut self) -> ParseResult<Spanned<Expr>> {
        self.parse_or_expr()
    }

    fn parse_or_expr(&mut self) -> ParseResult<Spanned<Expr>> {
        let mut left = self.parse_and_expr()?;
        while self.at(&Token::PipePipe) {
            self.advance();
            let right = self.parse_and_expr()?;
            let span = left.span.start..right.span.end;
            left = Spanned::new(
                Expr::Binary(Box::new(left), BinOp::Or, Box::new(right)),
                span,
            );
        }
        Ok(left)
    }

    fn parse_and_expr(&mut self) -> ParseResult<Spanned<Expr>> {
        let mut left = self.parse_eq_expr()?;
        while self.at(&Token::AmpAmp) {
            self.advance();
            let right = self.parse_eq_expr()?;
            let span = left.span.start..right.span.end;
            left = Spanned::new(
                Expr::Binary(Box::new(left), BinOp::And, Box::new(right)),
                span,
            );
        }
        Ok(left)
    }

    fn parse_eq_expr(&mut self) -> ParseResult<Spanned<Expr>> {
        let left = self.parse_cmp_expr()?;
        match self.peek() {
            Some(Token::EqEq) => {
                self.advance();
                let right = self.parse_cmp_expr()?;
                let span = left.span.start..right.span.end;
                Ok(Spanned::new(
                    Expr::Binary(Box::new(left), BinOp::Eq, Box::new(right)),
                    span,
                ))
            }
            Some(Token::BangEq) => {
                self.advance();
                let right = self.parse_cmp_expr()?;
                let span = left.span.start..right.span.end;
                Ok(Spanned::new(
                    Expr::Binary(Box::new(left), BinOp::Ne, Box::new(right)),
                    span,
                ))
            }
            _ => Ok(left),
        }
    }

    fn parse_cmp_expr(&mut self) -> ParseResult<Spanned<Expr>> {
        let left = self.parse_add_expr()?;
        let op = match self.peek() {
            Some(Token::Lt) => BinOp::Lt,
            Some(Token::Gt) => BinOp::Gt,
            Some(Token::Le) => BinOp::Le,
            Some(Token::Ge) => BinOp::Ge,
            _ => return Ok(left),
        };
        self.advance();
        let right = self.parse_add_expr()?;
        let span = left.span.start..right.span.end;
        Ok(Spanned::new(
            Expr::Binary(Box::new(left), op, Box::new(right)),
            span,
        ))
    }

    fn parse_add_expr(&mut self) -> ParseResult<Spanned<Expr>> {
        let mut left = self.parse_mul_expr()?;
        loop {
            let op = match self.peek() {
                Some(Token::Plus) => BinOp::Add,
                Some(Token::Minus) => BinOp::Sub,
                _ => break,
            };
            self.advance();
            let right = self.parse_mul_expr()?;
            let span = left.span.start..right.span.end;
            left = Spanned::new(
                Expr::Binary(Box::new(left), op, Box::new(right)),
                span,
            );
        }
        Ok(left)
    }

    fn parse_mul_expr(&mut self) -> ParseResult<Spanned<Expr>> {
        let mut left = self.parse_unary()?;
        loop {
            let op = match self.peek() {
                Some(Token::Star) => BinOp::Mul,
                Some(Token::Slash) => BinOp::Div,
                _ => break,
            };
            self.advance();
            let right = self.parse_unary()?;
            let span = left.span.start..right.span.end;
            left = Spanned::new(
                Expr::Binary(Box::new(left), op, Box::new(right)),
                span,
            );
        }
        Ok(left)
    }

    fn parse_unary(&mut self) -> ParseResult<Spanned<Expr>> {
        match self.peek() {
            Some(Token::Bang) => {
                let start = self.advance().unwrap().1;
                let operand = self.parse_postfix()?;
                let span = start.start..operand.span.end;
                Ok(Spanned::new(Expr::UnaryNot(Box::new(operand)), span))
            }
            Some(Token::Minus) => {
                let start = self.advance().unwrap().1;
                let operand = self.parse_postfix()?;
                let span = start.start..operand.span.end;
                Ok(Spanned::new(Expr::UnaryNeg(Box::new(operand)), span))
            }
            _ => self.parse_postfix(),
        }
    }

    fn parse_postfix(&mut self) -> ParseResult<Spanned<Expr>> {
        let mut expr = self.parse_primary()?;
        loop {
            match self.peek() {
                Some(Token::Dot) => {
                    self.advance();
                    let field = self.expect_ident()?;
                    let span = expr.span.start..field.span.end;
                    expr = Spanned::new(Expr::FieldAccess(Box::new(expr), field), span);
                }
                Some(Token::LBracket) => {
                    self.advance();
                    let index = self.parse_expr()?;
                    let end = self.expect(&Token::RBracket)?;
                    let span = expr.span.start..end.end;
                    expr = Spanned::new(
                        Expr::Index(Box::new(expr), Box::new(index)),
                        span,
                    );
                }
                _ => break,
            }
        }
        Ok(expr)
    }

    fn peek_is_ident_like(&mut self) -> bool {
        match self.peek() {
            Some(Token::Ident(_)) => true,
            Some(tok) => is_keyword_ident(tok),
            None => false,
        }
    }

    fn parse_primary(&mut self) -> ParseResult<Spanned<Expr>> {
        if self.peek_is_ident_like() {
            let ident = self.expect_ident()?;
            if self.at(&Token::LParen) {
                self.advance();
                let mut args = Vec::new();
                while !self.at(&Token::RParen) {
                    args.push(self.parse_expr()?);
                    if !self.eat(&Token::Comma) {
                        break;
                    }
                }
                let end = self.expect(&Token::RParen)?;
                let span = ident.span.start..end.end;
                return Ok(Spanned::new(Expr::Call(ident.name, args), span));
            }
            let span = ident.span.clone();
            return Ok(Spanned::new(Expr::Ident(ident.name), span));
        }

        match self.peek() {
            Some(Token::IntLit(_)) => {
                let (tok, span) = self.advance().unwrap();
                if let Token::IntLit(v) = tok {
                    Ok(Spanned::new(Expr::Literal(Literal::Int(v)), span))
                } else {
                    unreachable!()
                }
            }
            Some(Token::FloatLit(_)) => {
                let (tok, span) = self.advance().unwrap();
                if let Token::FloatLit(v) = tok {
                    Ok(Spanned::new(Expr::Literal(Literal::Float(v)), span))
                } else {
                    unreachable!()
                }
            }
            Some(Token::StringLit(_)) => {
                let (tok, span) = self.advance().unwrap();
                if let Token::StringLit(s) = tok {
                    Ok(Spanned::new(Expr::Literal(Literal::String(s)), span))
                } else {
                    unreachable!()
                }
            }
            Some(Token::True) => {
                let span = self.advance().unwrap().1;
                Ok(Spanned::new(Expr::Literal(Literal::Bool(true)), span))
            }
            Some(Token::False) => {
                let span = self.advance().unwrap().1;
                Ok(Spanned::new(Expr::Literal(Literal::Bool(false)), span))
            }
            Some(Token::Null) => {
                let span = self.advance().unwrap().1;
                Ok(Spanned::new(Expr::Literal(Literal::Null), span))
            }
            Some(Token::LBracket) => {
                let start = self.advance().unwrap().1;
                let mut elements = Vec::new();
                while !self.at(&Token::RBracket) {
                    elements.push(self.parse_expr()?);
                    if !self.eat(&Token::Comma) {
                        break;
                    }
                }
                let end = self.expect(&Token::RBracket)?;
                Ok(Spanned::new(
                    Expr::List(elements),
                    start.start..end.end,
                ))
            }
            Some(Token::LBrace) => {
                let start = self.advance().unwrap().1;
                let mut fields = Vec::new();
                while !self.at(&Token::RBrace) {
                    let key = self.expect_ident()?;
                    self.expect(&Token::Colon)?;
                    let value = self.parse_expr()?;
                    fields.push(ExprField { key, value });
                    if !self.eat(&Token::Comma) {
                        break;
                    }
                }
                let end = self.expect(&Token::RBrace)?;
                Ok(Spanned::new(
                    Expr::Record(fields),
                    start.start..end.end,
                ))
            }
            Some(Token::LParen) => {
                let start = self.advance().unwrap().1;
                let inner = self.parse_expr()?;
                let end = self.expect(&Token::RParen)?;
                Ok(Spanned::new(
                    Expr::Paren(Box::new(inner)),
                    start.start..end.end,
                ))
            }
            _ => {
                let span = self.peek_span();
                Err(ParseError {
                    message: "expected expression".into(),
                    span,
                })
            }
        }
    }
}

/// Check if a token is a keyword that can also be used as an identifier.
/// Many v2 keywords appear in identifier positions (step names, field access, etc).
fn is_keyword_ident(tok: &Token) -> bool {
    matches!(
        tok,
        Token::Input
            | Token::Expected
            | Token::Id
            | Token::Score
            | Token::Max
            | Token::Train
            | Token::Val
            | Token::Test
            | Token::Primary
            | Token::Rubric
            | Token::Reduce
            | Token::Checker
            | Token::Judge
            | Token::Metric
            | Token::Prompt
            | Token::Tool
            | Token::Agent
            | Token::Verify
            | Token::Template
            | Token::System
            | Token::Model
            | Token::Shell
            | Token::Json
            | Token::Abort
            | Token::Retry
            | Token::Topology
            | Token::Mutations
            | Token::Preserve
            | Token::TargetScore
            | Token::Sub
            | Token::Dataset
            | Token::Cases
            | Token::Tune
            | Token::Select
            | Token::Repeats
            | Token::Split
            | Token::Emit
            | Token::Carry
            | Token::Step
            | Token::Choose
            | Token::Parallel
            | Token::File
            | Token::Graph
            | Token::Node
            | Token::Objective
            | Token::Type
            | Token::Temperature
            | Token::MaxTokens
            | Token::MaxTurns
            | Token::Timeout
            | Token::OnError
            | Token::Tools
            | Token::In
            | Token::Out
            | Token::Loop
            | Token::If
            | Token::Else
            | Token::While
            | Token::MaxNodes
            | Token::MaxDepth
            | Token::TieBreakers
            | Token::Bool
            | Token::KwInt
            | Token::KwFloat
            | Token::KwString
            | Token::Bytes
            | Token::Any
            | Token::List
            | Token::Map
            | Token::Option
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_type_decl() {
        let src = r#"type Point = { x: float, y: float }"#;
        let program = parse(src).unwrap();
        assert_eq!(program.declarations.len(), 1);
        assert!(matches!(program.declarations[0], Declaration::Type(_)));
    }

    #[test]
    fn parse_node_decl() {
        let src = r#"
            node solver: prompt {
                in: { question: string }
                out: { answer: string }
                model: "gpt-4o"
                template: "Solve: {{ question }}"
                temperature: 0.7
            }
        "#;
        let program = parse(src).unwrap();
        assert_eq!(program.declarations.len(), 1);
        assert!(matches!(program.declarations[0], Declaration::Node(_)));
    }

    #[test]
    fn parse_graph_decl() {
        let src = r#"
            graph solve_pipeline {
                in: string
                out: string
                step s1 = solver(input)
                emit s1.answer
            }
        "#;
        let program = parse(src).unwrap();
        assert_eq!(program.declarations.len(), 1);
        assert!(matches!(program.declarations[0], Declaration::Graph(_)));
    }

    #[test]
    fn parse_loop_stmt() {
        let src = r#"
            graph retry_pipeline {
                in: string
                out: string
                loop (max: 3, while: !result.pass) {
                    step result = checker(input)
                }
                emit result.answer
            }
        "#;
        let program = parse(src).unwrap();
        assert_eq!(program.declarations.len(), 1);
    }

    #[test]
    fn parse_objective_decl() {
        let src = r#"
            objective eval {
                graph: solve_pipeline
                dataset: cases [
                    { input: "2+2", expected: "4" }
                    { input: "3+3", expected: "6" }
                ]
                checker exact { output == expected }
                metric accuracy { checker: exact }
                score: accuracy
            }
        "#;
        let program = parse(src).unwrap();
        assert_eq!(program.declarations.len(), 1);
        assert!(matches!(
            program.declarations[0],
            Declaration::Objective(_)
        ));
    }

    #[test]
    fn parse_expressions() {
        let src = r#"
            graph expr_test {
                in: int
                out: bool
                emit (1 + 2) * 3 == 9 && true
            }
        "#;
        let program = parse(src).unwrap();
        assert_eq!(program.declarations.len(), 1);
    }

    #[test]
    fn parse_tool_node() {
        let src = r#"
            node run_cmd: tool {
                in: { command: string }
                out: string
                shell: "bash -c '{command}'"
            }
        "#;
        let program = parse(src).unwrap();
        assert_eq!(program.declarations.len(), 1);
    }

    #[test]
    fn parse_parallel_stmt() {
        let src = r#"
            graph par_test {
                in: list<string>
                out: list<string>
                parallel (item in input, reduce: concat) {
                    step r = solver(item)
                    emit r.answer
                }
            }
        "#;
        let program = parse(src).unwrap();
        assert_eq!(program.declarations.len(), 1);
    }

    #[test]
    fn parse_if_stmt() {
        let src = r#"
            graph cond_test {
                in: { x: int }
                out: string
                if input.x > 0 {
                    emit "positive"
                } else {
                    emit "non-positive"
                }
            }
        "#;
        let program = parse(src).unwrap();
        assert_eq!(program.declarations.len(), 1);
    }

    #[test]
    fn parse_full_program() {
        let src = r#"
            type Question = { text: string, difficulty: int }

            node solver: prompt {
                in: Question
                out: { answer: string, confidence: float }
                model: "gpt-4o"
                template: file("prompts/solver.j2")
                temperature: 0.5
            }

            node checker: verify {
                in: { answer: string, expected: string }
                out: { pass: bool, reason: string }
                model: "gpt-4o-mini"
                template: "Is '{{ answer }}' equivalent to '{{ expected }}'? Reply JSON {pass: bool, reason: string}"
            }

            graph solve_and_verify {
                in: Question
                out: { answer: string, pass: bool }
                step s = solver(input)
                step v = checker(s.answer, input.text)
                loop (max: 3, while: !v.pass) {
                    step s = solver(input)
                    step v = checker(s.answer, input.text)
                }
                emit { answer: s.answer, pass: v.pass }
            }

            objective accuracy_eval {
                graph: solve_and_verify
                dataset: file("data/questions.jsonl")
                checker exact { output.answer == expected.answer }
                metric acc { checker: exact }
                score: acc
                repeats: 3
                split { train: 0.7, val: 0.15, test: 0.15 }
                select { primary: acc }
                tune {
                    solver.model in ["gpt-4o", "gpt-4o-mini"]
                    solver.temperature in [0.3, 0.5, 0.7, 1.0]
                }
                topology {
                    mutations: [insert_verify, wrap_retry, fan_out]
                    max_nodes: 10
                    max_depth: 5
                    preserve: [solver]
                }
            }
        "#;
        let program = parse(src).unwrap();
        assert_eq!(program.declarations.len(), 5);
    }

    #[test]
    fn parse_objective_with_single_sub() {
        let src = r#"
            objective eval {
                graph: outer
                dataset: file("data/full.jsonl")
                checker exact { output == expected }
                metric accuracy { checker: exact }
                score: accuracy

                sub inner_opt {
                    graph: inner
                    dataset: file("data/inner.jsonl")
                    checker sub_exact { output == expected }
                    metric sub_acc { checker: sub_exact }
                    score: sub_acc
                }
            }
        "#;
        let program = parse(src).unwrap();
        assert_eq!(program.declarations.len(), 1);
        if let Declaration::Objective(ref obj) = program.declarations[0] {
            assert_eq!(obj.subs.len(), 1);
            assert_eq!(obj.subs[0].name.name, "inner_opt");
            assert_eq!(obj.subs[0].graph.name, "inner");
        } else {
            panic!("expected objective");
        }
    }

    #[test]
    fn parse_objective_with_multiple_subs() {
        let src = r#"
            objective eval {
                graph: outer
                dataset: cases [{ input: "x", expected: "y" }]
                checker exact { output == expected }
                metric acc { checker: exact }
                score: acc

                sub sub_a {
                    graph: graph_a
                    dataset: file("a.jsonl")
                    checker c { output == expected }
                    metric m { checker: c }
                    score: m
                }

                sub sub_b {
                    graph: graph_b
                    dataset: file("b.jsonl")
                    checker c { output == expected }
                    metric m { checker: c }
                    score: m
                    tune {
                        solver.model in ["gpt-4o", "gpt-4o-mini"]
                    }
                    topology {
                        mutations: [insert_verify, wrap_retry]
                        max_nodes: 8
                    }
                }
            }
        "#;
        let program = parse(src).unwrap();
        if let Declaration::Objective(ref obj) = program.declarations[0] {
            assert_eq!(obj.subs.len(), 2);
            assert_eq!(obj.subs[0].name.name, "sub_a");
            assert_eq!(obj.subs[1].name.name, "sub_b");
            assert_eq!(obj.subs[1].tunables.len(), 1);
            assert!(obj.subs[1].topology.is_some());
        } else {
            panic!("expected objective");
        }
    }

    #[test]
    fn parse_nested_sub_rejected() {
        let src = r#"
            objective eval {
                graph: outer
                dataset: file("data.jsonl")
                checker exact { output == expected }
                metric acc { checker: exact }
                score: acc

                sub inner {
                    graph: inner
                    dataset: file("inner.jsonl")
                    checker c { output == expected }
                    metric m { checker: c }
                    score: m

                    sub nested {
                        graph: deep
                        dataset: file("deep.jsonl")
                        checker c { output == expected }
                        metric m { checker: c }
                        score: m
                    }
                }
            }
        "#;
        let result = parse(src);
        let err = result.err().expect("should fail to parse nested sub");
        assert!(err.message.contains("nested sub"), "error was: {}", err.message);
    }
}
