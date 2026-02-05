//! Recursive descent parser for the Scaffold DSL

use crate::ast::*;
use crate::lexer::{Lexer, SpannedToken, Token};

/// Parse error
#[derive(Debug, Clone)]
pub struct ParseError {
    pub message: String,
    pub span: Span,
}

impl ParseError {
    pub fn new(message: impl Into<String>, span: Span) -> Self {
        Self {
            message: message.into(),
            span,
        }
    }
}

pub type ParseResult<T> = Result<T, ParseError>;

/// Parser for the Scaffold DSL
pub struct Parser<'source> {
    lexer: Lexer<'source>,
    source: &'source str,
}

impl<'source> Parser<'source> {
    pub fn new(source: &'source str) -> Self {
        Self {
            lexer: Lexer::new(source),
            source,
        }
    }

    /// Parse a complete program
    pub fn parse_program(&mut self) -> ParseResult<Program> {
        let mut declarations = Vec::new();
        while self.lexer.peek().is_some() {
            declarations.push(self.parse_declaration()?);
        }
        Ok(Program { declarations })
    }

    fn parse_declaration(&mut self) -> ParseResult<Declaration> {
        let token = self.peek_token()?;
        match &token.token {
            Token::Type => Ok(Declaration::Type(self.parse_type_decl()?)),
            Token::Extern => Ok(Declaration::ExternCrate(self.parse_extern_crate_decl()?)),
            Token::Foreign => Ok(Declaration::Foreign(self.parse_foreign_decl()?)),
            Token::Tool => Ok(Declaration::Tool(self.parse_tool_decl()?)),
            Token::Prompt => Ok(Declaration::Prompt(self.parse_prompt_decl()?)),
            Token::Agent => Ok(Declaration::Agent(self.parse_agent_decl()?)),
            Token::Pipeline => Ok(Declaration::Pipeline(self.parse_pipeline_decl()?)),
            _ => Err(ParseError::new(
                format!("expected declaration (type, extern, foreign, tool, prompt, agent, pipeline), found '{}'", token.token),
                token.span,
            )),
        }
    }

    // =========== Type Parsing ===========

    fn parse_type_decl(&mut self) -> ParseResult<TypeDecl> {
        let start = self.expect(Token::Type)?.span;
        let name = self.parse_ident()?;
        self.expect(Token::Eq)?;
        let ty = self.parse_type_expr()?;

        Ok(TypeDecl {
            name,
            span: start.merge(ty.span),
            ty,
        })
    }

    fn parse_type_expr(&mut self) -> ParseResult<Spanned<TypeExpr>> {
        let token = self.peek_token()?;
        match &token.token {
            Token::Bool => {
                self.advance();
                Ok(Spanned::new(
                    TypeExpr::Primitive(PrimitiveType::Bool),
                    token.span,
                ))
            }
            Token::Int => {
                self.advance();
                Ok(Spanned::new(
                    TypeExpr::Primitive(PrimitiveType::Int),
                    token.span,
                ))
            }
            Token::Float => {
                self.advance();
                Ok(Spanned::new(
                    TypeExpr::Primitive(PrimitiveType::Float),
                    token.span,
                ))
            }
            Token::String_ => {
                self.advance();
                Ok(Spanned::new(
                    TypeExpr::Primitive(PrimitiveType::String),
                    token.span,
                ))
            }
            Token::Any => {
                self.advance();
                Ok(Spanned::new(
                    TypeExpr::Primitive(PrimitiveType::Any),
                    token.span,
                ))
            }
            Token::Bytes => {
                self.advance();
                Ok(Spanned::new(
                    TypeExpr::Primitive(PrimitiveType::Bytes),
                    token.span,
                ))
            }
            Token::Result_ => {
                let start = token.span;
                self.advance();
                self.expect(Token::Lt)?;
                let ok_type = self.parse_type_expr()?;
                self.expect(Token::Comma)?;
                let err_type = self.parse_type_expr()?;
                let end = self.expect(Token::Gt)?.span;
                Ok(Spanned::new(
                    TypeExpr::Result(Box::new(ok_type), Box::new(err_type)),
                    start.merge(end),
                ))
            }
            Token::List => {
                let start = token.span;
                self.advance();
                self.expect(Token::Lt)?;
                let inner = self.parse_type_expr()?;
                let end = self.expect(Token::Gt)?.span;
                Ok(Spanned::new(
                    TypeExpr::List(Box::new(inner)),
                    start.merge(end),
                ))
            }
            Token::Map => {
                let start = token.span;
                self.advance();
                self.expect(Token::Lt)?;
                let key = self.parse_type_expr()?;
                self.expect(Token::Comma)?;
                let value = self.parse_type_expr()?;
                let end = self.expect(Token::Gt)?.span;
                Ok(Spanned::new(
                    TypeExpr::Map(Box::new(key), Box::new(value)),
                    start.merge(end),
                ))
            }
            Token::Option_ => {
                let start = token.span;
                self.advance();
                self.expect(Token::Lt)?;
                let inner = self.parse_type_expr()?;
                let end = self.expect(Token::Gt)?.span;
                Ok(Spanned::new(
                    TypeExpr::Option(Box::new(inner)),
                    start.merge(end),
                ))
            }
            Token::LBrace => {
                let start = token.span;
                self.advance();
                let mut fields = vec![self.parse_field_decl()?];
                while self.check(&Token::Comma) {
                    self.advance();
                    // Allow trailing comma
                    if self.check(&Token::RBrace) {
                        break;
                    }
                    fields.push(self.parse_field_decl()?);
                }
                let end = self.expect(Token::RBrace)?.span;
                Ok(Spanned::new(TypeExpr::Struct(fields), start.merge(end)))
            }
            Token::Ident(name) => {
                let name = name.clone();
                self.advance();
                Ok(Spanned::new(TypeExpr::Named(name), token.span))
            }
            _ => Err(ParseError::new(
                format!("expected type expression, found '{}'", token.token),
                token.span,
            )),
        }
    }

    fn parse_field_decl(&mut self) -> ParseResult<FieldDecl> {
        let name = self.parse_ident()?;
        self.expect(Token::Colon)?;
        let ty = self.parse_type_expr()?;
        Ok(FieldDecl { name, ty })
    }

    // =========== Foreign and Tool Parsing ===========

    /// Parse: extern crate goblin = "0.7"
    fn parse_extern_crate_decl(&mut self) -> ParseResult<ExternCrateDecl> {
        let start = self.expect(Token::Extern)?.span;
        self.expect(Token::Crate)?;
        let name = self.parse_ident()?;
        self.expect(Token::Eq)?;
        let version = self.parse_string()?;

        // Optional features block: { features = ["derive"] }
        let mut features = Vec::new();
        if self.check(&Token::LBrace) {
            self.advance();
            // Parse features = [...]
            if self.check_ident("features") {
                self.advance();
                self.expect(Token::Eq)?;
                self.expect(Token::LBracket)?;
                if !self.check(&Token::RBracket) {
                    features.push(self.parse_string()?);
                    while self.check(&Token::Comma) {
                        self.advance();
                        if self.check(&Token::RBracket) {
                            break;
                        }
                        features.push(self.parse_string()?);
                    }
                }
                self.expect(Token::RBracket)?;
            }
            self.expect(Token::RBrace)?;
        }

        Ok(ExternCrateDecl {
            name: name.clone(),
            version,
            features,
            span: start.merge(name.span),
        })
    }

    /// Parse: foreign rust module_name { ... }
    fn parse_foreign_decl(&mut self) -> ParseResult<ForeignDecl> {
        let start = self.expect(Token::Foreign)?.span;
        let language = self.parse_ident()?;
        let name = self.parse_ident()?;
        self.expect(Token::LBrace)?;

        let mut type_aliases = Vec::new();
        let mut functions = Vec::new();

        while !self.check(&Token::RBrace) {
            let token = self.peek_token()?;
            match &token.token {
                Token::Type => {
                    type_aliases.push(self.parse_foreign_type_alias()?);
                }
                Token::Fn => {
                    functions.push(self.parse_foreign_fn()?);
                }
                _ => {
                    return Err(ParseError::new(
                        format!(
                            "expected 'type' or 'fn' in foreign block, found '{}'",
                            token.token
                        ),
                        token.span,
                    ));
                }
            }
        }

        let end = self.expect(Token::RBrace)?.span;

        Ok(ForeignDecl {
            language,
            name,
            type_aliases,
            functions,
            span: start.merge(end),
        })
    }

    /// Parse: type ScaffoldName = external::rust::Type
    fn parse_foreign_type_alias(&mut self) -> ParseResult<ForeignTypeAlias> {
        let start = self.expect(Token::Type)?.span;
        let name = self.parse_ident()?;
        self.expect(Token::Eq)?;

        // Parse the external type path (e.g., goblin::elf::Elf)
        let mut external_type = String::new();
        let first = self.parse_ident()?;
        external_type.push_str(&first.node);

        while self.check(&Token::Colon) {
            self.advance();
            self.expect(Token::Colon)?; // expect ::
            external_type.push_str("::");
            let part = self.parse_ident()?;
            external_type.push_str(&part.node);
        }

        Ok(ForeignTypeAlias {
            name: name.clone(),
            external_type,
            span: start.merge(name.span),
        })
    }

    /// Parse: fn func_name(param: type, ...) -> return_type
    fn parse_foreign_fn(&mut self) -> ParseResult<ForeignFn> {
        let start = self.expect(Token::Fn)?.span;
        let name = self.parse_ident()?;
        self.expect(Token::LParen)?;

        let mut params = Vec::new();
        if !self.check(&Token::RParen) {
            params.push(self.parse_foreign_param()?);
            while self.check(&Token::Comma) {
                self.advance();
                if self.check(&Token::RParen) {
                    break;
                }
                params.push(self.parse_foreign_param()?);
            }
        }
        self.expect(Token::RParen)?;

        self.expect(Token::Arrow)?;
        let return_type = self.parse_type_expr()?;

        Ok(ForeignFn {
            name: name.clone(),
            params,
            return_type: return_type.clone(),
            span: start.merge(return_type.span),
        })
    }

    fn parse_foreign_param(&mut self) -> ParseResult<ForeignParam> {
        let name = self.parse_ident()?;
        self.expect(Token::Colon)?;
        let ty = self.parse_type_expr()?;
        Ok(ForeignParam { name, ty })
    }

    /// Parse: tool name { input: ..., output: ..., impl: ..., spec: ... }
    fn parse_tool_decl(&mut self) -> ParseResult<ToolDecl> {
        let start = self.expect(Token::Tool)?.span;
        let name = self.parse_ident()?;
        self.expect(Token::LBrace)?;

        let mut input = None;
        let mut output = None;
        let mut implementation = None;
        let mut spec = None;
        let mut variants = Vec::new();

        while !self.check(&Token::RBrace) {
            let token = self.peek_token()?;
            match &token.token {
                Token::Input => {
                    self.advance();
                    self.expect(Token::Colon)?;
                    input = Some(self.parse_type_expr()?);
                }
                Token::Output => {
                    self.advance();
                    self.expect(Token::Colon)?;
                    output = Some(self.parse_type_expr()?);
                }
                Token::Impl => {
                    self.advance();
                    self.expect(Token::Colon)?;
                    implementation = Some(self.parse_tool_impl()?);
                }
                Token::Spec => {
                    spec = Some(self.parse_tool_spec()?);
                }
                Token::Variants => {
                    self.advance();
                    self.expect(Token::LBrace)?;
                    while !self.check(&Token::RBrace) {
                        variants.push(self.parse_tool_variant()?);
                    }
                    self.expect(Token::RBrace)?;
                }
                _ => {
                    return Err(ParseError::new(
                        format!(
                            "expected tool item (input, output, impl, spec, variants), found '{}'",
                            token.token
                        ),
                        token.span,
                    ));
                }
            }
        }

        let end = self.expect(Token::RBrace)?.span;

        // Default types if not specified
        let input =
            input.unwrap_or_else(|| Spanned::new(TypeExpr::Primitive(PrimitiveType::Any), start));
        let output =
            output.unwrap_or_else(|| Spanned::new(TypeExpr::Primitive(PrimitiveType::Any), start));

        Ok(ToolDecl {
            name,
            input,
            output,
            implementation,
            spec,
            variants,
            span: start.merge(end),
        })
    }

    fn parse_tool_impl(&mut self) -> ParseResult<ToolImpl> {
        let token = self.peek_token()?;
        match &token.token {
            Token::Sequence => {
                self.advance();
                self.expect(Token::LBrace)?;
                let mut statements = Vec::new();
                while !self.check(&Token::RBrace) {
                    statements.push(self.parse_tool_statement()?);
                }
                self.expect(Token::RBrace)?;
                Ok(ToolImpl::Sequence(statements))
            }
            Token::Parallel => {
                self.advance();
                self.expect(Token::LBrace)?;
                let mut statements = Vec::new();
                while !self.check(&Token::RBrace) {
                    statements.push(self.parse_tool_statement()?);
                }
                self.expect(Token::RBrace)?;
                Ok(ToolImpl::Parallel(statements))
            }
            _ => {
                let expr = self.parse_tool_expr()?;
                Ok(ToolImpl::Expr(expr))
            }
        }
    }

    /// Parse statements inside a block body (for/while/loop/if bodies)
    /// This allows let bindings and multiple statements without requiring explicit `sequence { }`
    fn parse_tool_block_body(&mut self) -> ParseResult<ToolImpl> {
        let mut statements = Vec::new();
        while !self.check(&Token::RBrace) {
            statements.push(self.parse_tool_statement()?);
        }
        // If there's only one statement without a binding, return as Expr for simplicity
        if statements.len() == 1 && statements[0].binding.is_none() {
            Ok(ToolImpl::Expr(statements.into_iter().next().unwrap().expr))
        } else {
            Ok(ToolImpl::Sequence(statements))
        }
    }

    fn parse_tool_statement(&mut self) -> ParseResult<ToolStatement> {
        let start_span = self.peek_token()?.span;

        // Check for let binding: let name = expr
        let binding = if self.check(&Token::Let) {
            self.advance();
            let name = self.parse_ident()?;
            self.expect(Token::Eq)?;
            Some(name)
        } else {
            None
        };

        // For let bindings, parse as general expression (supports arithmetic, etc.)
        // For standalone statements, parse as tool expression (tool calls, FFI, etc.)
        let expr = if binding.is_some() {
            let general_expr = self.parse_expr()?;
            let span = general_expr.span;
            Spanned::new(ToolExpr::Expr(Box::new(general_expr)), span)
        } else {
            self.parse_tool_expr()?
        };
        let span = start_span.merge(expr.span);

        Ok(ToolStatement {
            binding,
            expr,
            span,
        })
    }

    fn parse_tool_expr(&mut self) -> ParseResult<Spanned<ToolExpr>> {
        let mut left = self.parse_tool_primary_expr()?;

        // Handle pipe operator: expr |> func
        while self.check(&Token::Pipe) {
            self.advance();
            let right = self.parse_tool_primary_expr()?;
            let span = left.span.merge(right.span);
            left = Spanned::new(ToolExpr::Pipe(Box::new(left), Box::new(right)), span);
        }

        Ok(left)
    }

    fn parse_tool_primary_expr(&mut self) -> ParseResult<Spanned<ToolExpr>> {
        let token = self.peek_token()?;

        match &token.token {
            // Shell command: shell("command")
            Token::Ident(name) if name == "shell" => {
                let start = self.advance().span;
                self.expect(Token::LParen)?;
                let cmd = self.parse_string()?;
                let end = self.expect(Token::RParen)?.span;
                Ok(Spanned::new(ToolExpr::Shell(cmd), start.merge(end)))
            }
            // If expression
            Token::If => {
                let start = self.advance().span;
                let condition = self.parse_expr()?;
                self.expect(Token::LBrace)?;
                let then_impl = self.parse_tool_block_body()?;
                self.expect(Token::RBrace)?;

                let else_branch = if self.check(&Token::Else) {
                    self.advance();
                    self.expect(Token::LBrace)?;
                    let else_impl = self.parse_tool_block_body()?;
                    self.expect(Token::RBrace)?;
                    Some(Box::new(else_impl))
                } else {
                    None
                };

                let end_span = self.peek_token().map(|t| t.span).unwrap_or(start);
                Ok(Spanned::new(
                    ToolExpr::If {
                        condition: Box::new(condition),
                        then_branch: Box::new(then_impl),
                        else_branch,
                    },
                    start.merge(end_span),
                ))
            }
            // Match expression
            Token::Match => {
                let start = self.advance().span;
                let scrutinee = self.parse_tool_expr()?;
                self.expect(Token::LBrace)?;

                let mut arms = Vec::new();
                while !self.check(&Token::RBrace) {
                    let pattern = self.parse_expr()?;
                    self.expect(Token::Arrow)?;
                    let body = self.parse_tool_impl()?;
                    let arm_span = pattern
                        .span
                        .merge(self.peek_token().map(|t| t.span).unwrap_or(pattern.span));
                    arms.push(MatchArm {
                        pattern,
                        body,
                        span: arm_span,
                    });
                    // Optional comma between arms
                    if self.check(&Token::Comma) {
                        self.advance();
                    }
                }

                let end = self.expect(Token::RBrace)?.span;
                Ok(Spanned::new(
                    ToolExpr::Match {
                        scrutinee: Box::new(scrutinee),
                        arms,
                    },
                    start.merge(end),
                ))
            }
            // For loop: for item in collection { ... }
            Token::For => {
                let start = self.advance().span;
                let variable = self.parse_ident()?;
                self.expect(Token::In)?;
                let iterable = self.parse_tool_expr()?;
                self.expect(Token::LBrace)?;
                let body = self.parse_tool_block_body()?;
                let end = self.expect(Token::RBrace)?.span;
                Ok(Spanned::new(
                    ToolExpr::For {
                        variable,
                        iterable: Box::new(iterable),
                        body: Box::new(body),
                    },
                    start.merge(end),
                ))
            }
            // While loop: while condition { ... }
            Token::While => {
                let start = self.advance().span;
                let condition = self.parse_expr()?;
                self.expect(Token::LBrace)?;
                let body = self.parse_tool_block_body()?;
                let end = self.expect(Token::RBrace)?.span;
                Ok(Spanned::new(
                    ToolExpr::While {
                        condition: Box::new(condition),
                        body: Box::new(body),
                    },
                    start.merge(end),
                ))
            }
            // Infinite loop: loop { ... }
            Token::Loop => {
                let start = self.advance().span;
                self.expect(Token::LBrace)?;
                let body = self.parse_tool_block_body()?;
                let end = self.expect(Token::RBrace)?.span;
                Ok(Spanned::new(
                    ToolExpr::Loop {
                        body: Box::new(body),
                    },
                    start.merge(end),
                ))
            }
            // Break statement
            Token::Break => {
                let span = self.advance().span;
                Ok(Spanned::new(ToolExpr::Break, span))
            }
            // Continue statement
            Token::Continue => {
                let span = self.advance().span;
                Ok(Spanned::new(ToolExpr::Continue, span))
            }
            // Literal values
            Token::IntLit(n) => {
                let n = *n;
                let span = self.advance().span;
                Ok(Spanned::new(ToolExpr::Literal(Literal::Int(n)), span))
            }
            Token::StringLit(s) => {
                let s = s.clone();
                let span = self.advance().span;
                Ok(Spanned::new(ToolExpr::Literal(Literal::String(s)), span))
            }
            Token::True => {
                let span = self.advance().span;
                Ok(Spanned::new(ToolExpr::Literal(Literal::Bool(true)), span))
            }
            Token::False => {
                let span = self.advance().span;
                Ok(Spanned::new(ToolExpr::Literal(Literal::Bool(false)), span))
            }
            // Keywords that can be used as identifiers in tool expressions
            Token::Input | Token::Output | Token::State | Token::Result_ => {
                let name = match &token.token {
                    Token::Input => "input".to_string(),
                    Token::Output => "output".to_string(),
                    Token::State => "state".to_string(),
                    Token::Result_ => "result".to_string(),
                    _ => unreachable!(),
                };
                let start = self.advance().span;
                self.parse_tool_ident_continuation(name, start)
            }
            // Identifier, foreign call, or tool call
            Token::Ident(name) => {
                let name = name.clone();
                let start = self.advance().span;
                self.parse_tool_ident_continuation(name, start)
            }
            _ => Err(ParseError::new(
                format!("expected tool expression, found '{}'", token.token),
                token.span,
            )),
        }
    }

    fn parse_tool_ident_continuation(
        &mut self,
        name: String,
        start: Span,
    ) -> ParseResult<Spanned<ToolExpr>> {
        // Check for :: (foreign module call)
        if self.check(&Token::Colon) {
            self.advance();
            self.expect(Token::Colon)?;
            let func_name = self.parse_ident()?;
            self.expect(Token::LParen)?;

            let mut args = Vec::new();
            if !self.check(&Token::RParen) {
                args.push(self.parse_tool_expr()?);
                while self.check(&Token::Comma) {
                    self.advance();
                    if self.check(&Token::RParen) {
                        break;
                    }
                    args.push(self.parse_tool_expr()?);
                }
            }
            let end = self.expect(Token::RParen)?.span;

            Ok(Spanned::new(
                ToolExpr::ForeignCall {
                    module: name,
                    function: func_name.node,
                    args,
                },
                start.merge(end),
            ))
        }
        // Check for ( (tool call or function)
        else if self.check(&Token::LParen) {
            self.advance();
            let mut args = Vec::new();
            if !self.check(&Token::RParen) {
                args.push(self.parse_tool_expr()?);
                while self.check(&Token::Comma) {
                    self.advance();
                    if self.check(&Token::RParen) {
                        break;
                    }
                    args.push(self.parse_tool_expr()?);
                }
            }
            let end = self.expect(Token::RParen)?.span;

            Ok(Spanned::new(
                ToolExpr::ToolCall { tool: name, args },
                start.merge(end),
            ))
        }
        // Check for . (field access)
        else if self.check(&Token::Dot) {
            let mut expr = Spanned::new(ToolExpr::Ident(name), start);
            while self.check(&Token::Dot) {
                self.advance();
                let field = self.parse_ident()?;
                let span = expr.span.merge(field.span);
                expr = Spanned::new(ToolExpr::FieldAccess(Box::new(expr), field), span);
            }
            Ok(expr)
        }
        // Just an identifier
        else {
            Ok(Spanned::new(ToolExpr::Ident(name), start))
        }
    }

    fn parse_tool_spec(&mut self) -> ParseResult<ToolSpec> {
        let start = self.expect(Token::Spec)?.span;
        self.expect(Token::LBrace)?;

        let mut preconditions = Vec::new();
        let mut postconditions = Vec::new();
        let mut pure = false;

        while !self.check(&Token::RBrace) {
            let token = self.peek_token()?;
            match &token.token {
                Token::Pre => {
                    self.advance();
                    self.expect(Token::Colon)?;
                    preconditions.push(self.parse_expr()?);
                }
                Token::Post => {
                    self.advance();
                    self.expect(Token::Colon)?;
                    postconditions.push(self.parse_expr()?);
                }
                Token::Pure => {
                    self.advance();
                    // `pure` can be standalone or followed by `: true/false`
                    if self.check(&Token::Colon) {
                        self.advance();
                        pure = self.parse_bool()?;
                    } else {
                        pure = true;
                    }
                }
                _ => {
                    return Err(ParseError::new(
                        format!(
                            "expected spec item (pre, post, pure), found '{}'",
                            token.token
                        ),
                        token.span,
                    ));
                }
            }
        }

        let end = self.expect(Token::RBrace)?.span;

        Ok(ToolSpec {
            preconditions,
            postconditions,
            pure,
            span: start.merge(end),
        })
    }

    fn parse_tool_variant(&mut self) -> ParseResult<ToolVariant> {
        let name = self.parse_ident()?;
        let start = name.span;

        // Variant can be just a name or name { impl }
        let implementation = if self.check(&Token::LBrace) {
            self.advance();
            let impl_ = self.parse_tool_impl()?;
            self.expect(Token::RBrace)?;
            impl_
        } else if self.check(&Token::Colon) {
            self.advance();
            self.parse_tool_impl()?
        } else {
            // Just reference the variant by name
            ToolImpl::Expr(Spanned::new(ToolExpr::Ident(name.node.clone()), name.span))
        };

        Ok(ToolVariant {
            name,
            implementation,
            span: start,
        })
    }

    /// Helper to check if next token is a specific identifier
    fn check_ident(&mut self, expected: &str) -> bool {
        self.lexer
            .peek()
            .map(|t| matches!(&t.token, Token::Ident(s) if s == expected))
            .unwrap_or(false)
    }

    // =========== Prompt Parsing ===========

    /// Parse: prompt name { input: ..., output: ..., template: ..., system: ... }
    fn parse_prompt_decl(&mut self) -> ParseResult<PromptDecl> {
        let start = self.expect(Token::Prompt)?.span;
        let name = self.parse_ident()?;
        self.expect(Token::LBrace)?;

        let mut input = None;
        let mut output = None;
        let mut template = None;
        let mut system = None;

        while !self.check(&Token::RBrace) {
            let token = self.peek_token()?;
            match &token.token {
                Token::Input => {
                    self.advance();
                    self.expect(Token::Colon)?;
                    input = Some(self.parse_type_expr()?);
                }
                Token::Output => {
                    self.advance();
                    self.expect(Token::Colon)?;
                    output = Some(self.parse_type_expr()?);
                }
                Token::Template => {
                    self.advance();
                    self.expect(Token::Colon)?;
                    template = Some(self.parse_string_or_file()?);
                }
                Token::System => {
                    self.advance();
                    self.expect(Token::Colon)?;
                    system = Some(self.parse_string_or_file()?);
                }
                _ => {
                    return Err(ParseError::new(
                        format!(
                            "expected prompt item (input, output, template, system), found '{}'",
                            token.token
                        ),
                        token.span,
                    ));
                }
            }
        }

        let end = self.expect(Token::RBrace)?.span;

        // Template is required
        let template =
            template.ok_or_else(|| ParseError::new("prompt requires a template", start))?;

        // Default types if not specified
        let input =
            input.unwrap_or_else(|| Spanned::new(TypeExpr::Primitive(PrimitiveType::Any), start));
        let output =
            output.unwrap_or_else(|| Spanned::new(TypeExpr::Primitive(PrimitiveType::Any), start));

        Ok(PromptDecl {
            name,
            input,
            output,
            template,
            system,
            span: start.merge(end),
        })
    }

    // =========== Agent Parsing ===========

    /// Parse: agent name { input: ..., output: ..., tools: [...], system: ..., model: ..., max_turns: ..., reward: ..., done: ... }
    fn parse_agent_decl(&mut self) -> ParseResult<AgentDecl> {
        let start = self.expect(Token::Agent)?.span;
        let name = self.parse_ident()?;
        self.expect(Token::LBrace)?;

        let mut input = None;
        let mut output = None;
        let mut tools = Vec::new();
        let mut system = None;
        let mut model = None;
        let mut max_turns = None;
        let mut reward = None;
        let mut done = None;
        let mut on_error = ErrorStrategy::default();
        let mut timeout = None;

        while !self.check(&Token::RBrace) {
            let token = self.peek_token()?;
            match &token.token {
                Token::Input => {
                    self.advance();
                    self.expect(Token::Colon)?;
                    input = Some(self.parse_type_expr()?);
                }
                Token::Output => {
                    self.advance();
                    self.expect(Token::Colon)?;
                    output = Some(self.parse_type_expr()?);
                }
                Token::Tools => {
                    self.advance();
                    self.expect(Token::Colon)?;
                    self.expect(Token::LBracket)?;
                    if !self.check(&Token::RBracket) {
                        tools.push(self.parse_ident()?);
                        while self.check(&Token::Comma) {
                            self.advance();
                            if self.check(&Token::RBracket) {
                                break;
                            }
                            tools.push(self.parse_ident()?);
                        }
                    }
                    self.expect(Token::RBracket)?;
                }
                Token::System => {
                    self.advance();
                    self.expect(Token::Colon)?;
                    system = Some(self.parse_string_or_file()?);
                }
                Token::Model => {
                    self.advance();
                    self.expect(Token::Colon)?;
                    model = Some(self.parse_string()?);
                }
                Token::MaxTurns => {
                    self.advance();
                    self.expect(Token::Colon)?;
                    max_turns = Some(self.parse_int()? as u64);
                }
                Token::Reward => {
                    self.advance();
                    self.expect(Token::Colon)?;
                    reward = Some(self.parse_expr()?);
                }
                Token::Done => {
                    self.advance();
                    self.expect(Token::Colon)?;
                    done = Some(self.parse_expr()?);
                }
                Token::OnError => {
                    self.advance();
                    self.expect(Token::Colon)?;
                    on_error = self.parse_error_strategy()?;
                }
                Token::Timeout => {
                    self.advance();
                    self.expect(Token::Colon)?;
                    timeout = Some(self.parse_int()? as u64);
                }
                _ => {
                    return Err(ParseError::new(
                        format!("expected agent item (input, output, tools, system, model, max_turns, reward, done, on_error, timeout), found '{}'", token.token),
                        token.span,
                    ));
                }
            }
        }

        let end = self.expect(Token::RBrace)?.span;

        // System is required for agents
        let system =
            system.ok_or_else(|| ParseError::new("agent requires a system prompt", start))?;

        // Default types if not specified
        let input =
            input.unwrap_or_else(|| Spanned::new(TypeExpr::Primitive(PrimitiveType::Any), start));
        let output =
            output.unwrap_or_else(|| Spanned::new(TypeExpr::Primitive(PrimitiveType::Any), start));

        Ok(AgentDecl {
            name,
            input,
            output,
            tools,
            system,
            model,
            max_turns,
            reward,
            done,
            on_error,
            timeout,
            span: start.merge(end),
        })
    }

    /// Parse error strategy: `abort` or `retry(N)`
    fn parse_error_strategy(&mut self) -> ParseResult<ErrorStrategy> {
        let token = self.peek_token()?;
        match &token.token {
            Token::Abort => {
                self.advance();
                Ok(ErrorStrategy::Abort)
            }
            Token::Retry => {
                self.advance();
                self.expect(Token::LParen)?;
                let count = self.parse_int()? as u64;
                self.expect(Token::RParen)?;
                Ok(ErrorStrategy::Retry(count))
            }
            _ => Err(ParseError::new(
                format!(
                    "expected error strategy (abort, retry(N)), found '{}'",
                    token.token
                ),
                token.span,
            )),
        }
    }

    // =========== Pipeline Parsing ===========

    /// Parse: pipeline name { input: ..., output: ..., steps { ... }, reward: ... }
    fn parse_pipeline_decl(&mut self) -> ParseResult<PipelineDecl> {
        let start = self.expect(Token::Pipeline)?.span;
        let name = self.parse_ident()?;
        self.expect(Token::LBrace)?;

        let mut input = None;
        let mut output = None;
        let mut steps = Vec::new();
        let mut reward = None;

        while !self.check(&Token::RBrace) {
            let token = self.peek_token()?;
            match &token.token {
                Token::Input => {
                    self.advance();
                    self.expect(Token::Colon)?;
                    input = Some(self.parse_type_expr()?);
                }
                Token::Output => {
                    self.advance();
                    self.expect(Token::Colon)?;
                    output = Some(self.parse_type_expr()?);
                }
                Token::Ident(s) if s == "steps" => {
                    self.advance();
                    self.expect(Token::LBrace)?;
                    while !self.check(&Token::RBrace) {
                        steps.push(self.parse_pipeline_step()?);
                    }
                    self.expect(Token::RBrace)?;
                }
                Token::Reward => {
                    self.advance();
                    self.expect(Token::Colon)?;
                    reward = Some(self.parse_expr()?);
                }
                _ => {
                    return Err(ParseError::new(
                        format!(
                            "expected pipeline item (input, output, steps, reward), found '{}'",
                            token.token
                        ),
                        token.span,
                    ));
                }
            }
        }

        let end = self.expect(Token::RBrace)?.span;

        // Default types if not specified
        let input =
            input.unwrap_or_else(|| Spanned::new(TypeExpr::Primitive(PrimitiveType::Any), start));
        let output =
            output.unwrap_or_else(|| Spanned::new(TypeExpr::Primitive(PrimitiveType::Any), start));

        Ok(PipelineDecl {
            name,
            input,
            output,
            steps,
            reward,
            span: start.merge(end),
        })
    }

    fn parse_pipeline_step(&mut self) -> ParseResult<PipelineStep> {
        let start_span = self.peek_token()?.span;

        // Check for let binding: let name = expr
        let binding = if self.check(&Token::Let) {
            self.advance();
            let name = self.parse_ident()?;
            self.expect(Token::Eq)?;
            Some(name)
        } else {
            None
        };

        // Control-flow steps
        if self.check(&Token::Parallel) {
            let (call, end_span) = self.parse_pipeline_parallel()?;
            if binding.is_some() {
                return Err(ParseError::new(
                    "parallel steps cannot be bound with let".to_string(),
                    start_span,
                ));
            }
            return Ok(PipelineStep {
                binding,
                call,
                span: start_span.merge(end_span),
            });
        }
        if self.check(&Token::If) {
            let (call, end_span) = self.parse_pipeline_if()?;
            return Ok(PipelineStep {
                binding,
                call,
                span: start_span.merge(end_span),
            });
        }
        if self.check(&Token::Match) {
            let (call, end_span) = self.parse_pipeline_match()?;
            return Ok(PipelineStep {
                binding,
                call,
                span: start_span.merge(end_span),
            });
        }

        // Parse a general tool expression, then downcast to call if applicable
        let expr = self.parse_tool_expr()?;
        let end = expr.span;
        let call = match &expr.node {
            ToolExpr::ToolCall { tool, args } => PipelineCall::Tool {
                name: tool.clone(),
                args: args.clone(),
            },
            ToolExpr::ForeignCall {
                module: _,
                function: _,
                args: _,
            } => {
                // Treat foreign calls as tool calls under a synthetic name if needed; for now, wrap as expr
                PipelineCall::Expr(expr)
            }
            _ => PipelineCall::Expr(expr),
        };

        Ok(PipelineStep {
            binding,
            call,
            span: start_span.merge(end),
        })
    }

    fn parse_pipeline_parallel(&mut self) -> ParseResult<(PipelineCall, Span)> {
        let _start = self.expect(Token::Parallel)?.span;
        self.expect(Token::LBrace)?;

        let mut branches = Vec::new();
        while !self.check(&Token::RBrace) {
            self.expect(Token::LBrace)?;
            let mut steps = Vec::new();
            while !self.check(&Token::RBrace) {
                steps.push(self.parse_pipeline_step()?);
            }
            let _end_branch = self.expect(Token::RBrace)?.span;
            branches.push(steps);
            // Optional comma between branches
            if self.check(&Token::Comma) {
                self.advance();
            }
            // If there's no comma, continue until RBrace
        }

        let end = self.expect(Token::RBrace)?.span;
        Ok((PipelineCall::Parallel { branches }, end))
    }

    fn parse_pipeline_if(&mut self) -> ParseResult<(PipelineCall, Span)> {
        let _start = self.expect(Token::If)?.span;
        let condition = self.parse_expr()?;
        self.expect(Token::LBrace)?;
        let mut then_steps = Vec::new();
        while !self.check(&Token::RBrace) {
            then_steps.push(self.parse_pipeline_step()?);
        }
        let mut end_span = self.expect(Token::RBrace)?.span;

        let mut else_steps = Vec::new();
        if self.check(&Token::Else) {
            self.advance();
            self.expect(Token::LBrace)?;
            while !self.check(&Token::RBrace) {
                else_steps.push(self.parse_pipeline_step()?);
            }
            end_span = self.expect(Token::RBrace)?.span;
        }

        Ok((
            PipelineCall::If {
                condition,
                then_steps,
                else_steps,
            },
            end_span,
        ))
    }

    fn parse_pipeline_match(&mut self) -> ParseResult<(PipelineCall, Span)> {
        let _start = self.expect(Token::Match)?.span;
        let scrutinee = self.parse_expr()?;
        self.expect(Token::LBrace)?;

        let mut arms = Vec::new();
        while !self.check(&Token::RBrace) {
            let pattern = self.parse_expr()?;
            self.expect(Token::Arrow)?;
            self.expect(Token::LBrace)?;
            let mut steps = Vec::new();
            while !self.check(&Token::RBrace) {
                steps.push(self.parse_pipeline_step()?);
            }
            let arm_end = self.expect(Token::RBrace)?.span;
            let arm_span = pattern.span.merge(arm_end);
            arms.push(PipelineMatchArm {
                pattern,
                steps,
                span: arm_span,
            });
            if self.check(&Token::Comma) {
                self.advance();
            }
        }

        let end = self.expect(Token::RBrace)?.span;
        Ok((PipelineCall::Match { scrutinee, arms }, end))
    }

    /// Parse a string literal or file("path") reference
    fn parse_string_or_file(&mut self) -> ParseResult<StringOrFile> {
        let token = self.peek_token()?;
        match &token.token {
            Token::StringLit(s) => {
                let s = s.clone();
                self.advance();
                Ok(StringOrFile::Literal(s))
            }
            Token::File => {
                self.advance();
                self.expect(Token::LParen)?;
                let path = self.parse_string()?;
                self.expect(Token::RParen)?;
                Ok(StringOrFile::File(path))
            }
            _ => Err(ParseError::new(
                format!("expected string or file(\"path\"), found '{}'", token.token),
                token.span,
            )),
        }
    }

    // =========== Expression Parsing ===========

    fn parse_expr(&mut self) -> ParseResult<Spanned<Expr>> {
        self.parse_binary_expr(0)
    }

    fn parse_binary_expr(&mut self, min_prec: u8) -> ParseResult<Spanned<Expr>> {
        let mut left = self.parse_unary_expr()?;

        while let Some(op) = self.peek_binop() {
            let prec = op.precedence();
            if prec < min_prec {
                break;
            }
            self.advance(); // consume operator
            let right = self.parse_binary_expr(prec + 1)?;
            let span = left.span.merge(right.span);
            left = Spanned::new(Expr::Binary(Box::new(left), op, Box::new(right)), span);
        }

        Ok(left)
    }

    fn peek_binop(&mut self) -> Option<BinOp> {
        let token = self.lexer.peek()?;
        match &token.token {
            Token::EqEq => Some(BinOp::Eq),
            Token::Ne => Some(BinOp::Ne),
            Token::Lt => Some(BinOp::Lt),
            Token::Gt => Some(BinOp::Gt),
            Token::Le => Some(BinOp::Le),
            Token::Ge => Some(BinOp::Ge),
            Token::AndAnd => Some(BinOp::And),
            Token::OrOr => Some(BinOp::Or),
            Token::Plus => Some(BinOp::Add),
            Token::Minus => Some(BinOp::Sub),
            Token::Star => Some(BinOp::Mul),
            Token::Slash => Some(BinOp::Div),
            _ => None,
        }
    }

    fn parse_unary_expr(&mut self) -> ParseResult<Spanned<Expr>> {
        // Handle unary minus
        if self.check(&Token::Minus) {
            let start = self.advance().span;
            let inner = self.parse_unary_expr()?;
            let span = start.merge(inner.span);
            // Represent -x as (0 - x)
            return Ok(Spanned::new(
                Expr::Binary(
                    Box::new(Spanned::new(Expr::Literal(Literal::Int(0)), start)),
                    BinOp::Sub,
                    Box::new(inner),
                ),
                span,
            ));
        }
        self.parse_postfix_expr()
    }

    fn parse_postfix_expr(&mut self) -> ParseResult<Spanned<Expr>> {
        let mut expr = self.parse_primary_expr()?;

        loop {
            if self.check(&Token::Dot) {
                self.advance();
                let field = self.parse_field_name()?;
                let span = expr.span.merge(field.span);
                expr = Spanned::new(Expr::FieldAccess(Box::new(expr), field), span);
            } else if self.check(&Token::Colon) {
                // Foreign call: module::func(args) - only valid for identifiers
                if let Expr::Ident(module) = &expr.node {
                    let module = module.clone();
                    let start = expr.span;
                    self.advance(); // first :
                    self.expect(Token::Colon)?; // second :
                    let func_name = self.parse_ident()?;
                    self.expect(Token::LParen)?;
                    let mut args = Vec::new();
                    if !self.check(&Token::RParen) {
                        args.push(self.parse_expr()?);
                        while self.check(&Token::Comma) {
                            self.advance();
                            if self.check(&Token::RParen) {
                                break;
                            }
                            args.push(self.parse_expr()?);
                        }
                    }
                    let end = self.expect(Token::RParen)?.span;
                    expr = Spanned::new(
                        Expr::ForeignCall {
                            module,
                            function: func_name.node,
                            args,
                        },
                        start.merge(end),
                    );
                } else {
                    break;
                }
            } else if self.check(&Token::LParen) {
                // Function call - only valid for identifiers
                if let Expr::Ident(name) = &expr.node {
                    let name = name.clone();
                    self.advance();
                    let mut args = Vec::new();
                    if !self.check(&Token::RParen) {
                        args.push(self.parse_expr()?);
                        while self.check(&Token::Comma) {
                            self.advance();
                            args.push(self.parse_expr()?);
                        }
                    }
                    let end = self.expect(Token::RParen)?.span;
                    let span = expr.span.merge(end);
                    expr = Spanned::new(Expr::Call(name, args), span);
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        Ok(expr)
    }

    fn parse_primary_expr(&mut self) -> ParseResult<Spanned<Expr>> {
        let token = self.next_token()?;
        match &token.token {
            Token::IntLit(n) => Ok(Spanned::new(Expr::Literal(Literal::Int(*n)), token.span)),
            Token::FloatLit(n) => Ok(Spanned::new(Expr::Literal(Literal::Float(*n)), token.span)),
            Token::StringLit(s) => Ok(Spanned::new(
                Expr::Literal(Literal::String(s.clone())),
                token.span,
            )),
            Token::True => Ok(Spanned::new(Expr::Literal(Literal::Bool(true)), token.span)),
            Token::False => Ok(Spanned::new(
                Expr::Literal(Literal::Bool(false)),
                token.span,
            )),
            Token::Null => Ok(Spanned::new(Expr::Literal(Literal::Null), token.span)),
            Token::Ident(name) => Ok(Spanned::new(Expr::Ident(name.clone()), token.span)),
            Token::LParen => {
                let inner = self.parse_expr()?;
                let end = self.expect(Token::RParen)?.span;
                let span = token.span.merge(end);
                Ok(Spanned::new(Expr::Paren(Box::new(inner)), span))
            }
            // Handle keywords that might be used as identifiers in expressions
            Token::Decompose => Ok(Spanned::new(
                Expr::Ident("decompose".to_string()),
                token.span,
            )),
            Token::Options => Ok(Spanned::new(Expr::Ident("options".to_string()), token.span)),
            Token::Input => Ok(Spanned::new(Expr::Ident("input".to_string()), token.span)),
            Token::Output => Ok(Spanned::new(Expr::Ident("output".to_string()), token.span)),
            Token::State => Ok(Spanned::new(Expr::Ident("state".to_string()), token.span)),
            Token::Result_ => Ok(Spanned::new(Expr::Ident("result".to_string()), token.span)),
            Token::Pre => Ok(Spanned::new(Expr::Ident("pre".to_string()), token.span)),
            Token::Post => Ok(Spanned::new(Expr::Ident("post".to_string()), token.span)),
            Token::Done => Ok(Spanned::new(Expr::Ident("done".to_string()), token.span)),
            _ => Err(ParseError::new(
                format!("expected expression, found '{}'", token.token),
                token.span,
            )),
        }
    }

    // =========== Helper Methods ===========

    fn parse_ident(&mut self) -> ParseResult<Ident> {
        let token = self.next_token()?;
        match token.token {
            Token::Ident(name) => Ok(Spanned::new(name, token.span)),
            _ => Err(ParseError::new(
                format!("expected identifier, found '{}'", token.token),
                token.span,
            )),
        }
    }

    /// Parse an identifier that can also be a keyword (for field access)
    fn parse_field_name(&mut self) -> ParseResult<Ident> {
        let token = self.next_token()?;
        let name = match &token.token {
            Token::Ident(name) => name.clone(),
            // Allow keywords as field names
            Token::Done => "done".to_string(),
            Token::Pre => "pre".to_string(),
            Token::Post => "post".to_string(),
            Token::Input => "input".to_string(),
            Token::Output => "output".to_string(),
            Token::State => "state".to_string(),
            Token::Options => "options".to_string(),
            Token::Reward => "reward".to_string(),
            Token::Timeout => "timeout".to_string(),
            Token::Type => "type".to_string(),
            Token::Task => "task".to_string(),
            Token::Subgoal => "subgoal".to_string(),
            Token::Decompose => "decompose".to_string(),
            Token::Verify => "verify".to_string(),
            _ => {
                return Err(ParseError::new(
                    format!("expected field name, found '{}'", token.token),
                    token.span,
                ))
            }
        };
        Ok(Spanned::new(name, token.span))
    }

    fn parse_string(&mut self) -> ParseResult<String> {
        let token = self.next_token()?;
        match token.token {
            Token::StringLit(s) => Ok(s),
            _ => Err(ParseError::new(
                format!("expected string literal, found '{}'", token.token),
                token.span,
            )),
        }
    }

    fn parse_int(&mut self) -> ParseResult<i64> {
        let token = self.next_token()?;
        match token.token {
            Token::IntLit(n) => Ok(n),
            _ => Err(ParseError::new(
                format!("expected integer literal, found '{}'", token.token),
                token.span,
            )),
        }
    }

    fn parse_bool(&mut self) -> ParseResult<bool> {
        let token = self.next_token()?;
        match token.token {
            Token::True => Ok(true),
            Token::False => Ok(false),
            _ => Err(ParseError::new(
                format!("expected boolean, found '{}'", token.token),
                token.span,
            )),
        }
    }

    fn expect(&mut self, expected: Token) -> ParseResult<SpannedToken> {
        let token = self.next_token()?;
        if std::mem::discriminant(&token.token) == std::mem::discriminant(&expected) {
            Ok(token)
        } else {
            Err(ParseError::new(
                format!("expected '{}', found '{}'", expected, token.token),
                token.span,
            ))
        }
    }

    fn check(&mut self, expected: &Token) -> bool {
        self.lexer
            .peek()
            .map(|t| std::mem::discriminant(&t.token) == std::mem::discriminant(expected))
            .unwrap_or(false)
    }

    fn advance(&mut self) -> SpannedToken {
        self.lexer.next_token().unwrap()
    }

    fn next_token(&mut self) -> ParseResult<SpannedToken> {
        self.lexer.next_token().ok_or_else(|| {
            ParseError::new(
                "unexpected end of input",
                Span::new(self.source.len(), self.source.len()),
            )
        })
    }

    fn peek_token(&mut self) -> ParseResult<SpannedToken> {
        self.lexer.peek().cloned().ok_or_else(|| {
            ParseError::new(
                "unexpected end of input",
                Span::new(self.source.len(), self.source.len()),
            )
        })
    }
}

/// Parse a source string into a program AST
pub fn parse(source: &str) -> ParseResult<Program> {
    let mut parser = Parser::new(source);
    parser.parse_program()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_type_decl() {
        let source = "type Position = { x: int, y: int }";
        let program = parse(source).unwrap();
        assert_eq!(program.declarations.len(), 1);
        match &program.declarations[0] {
            Declaration::Type(t) => {
                assert_eq!(t.name.node, "Position");
            }
            _ => panic!("expected type declaration"),
        }
    }
}
