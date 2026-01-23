//! Scaffold DSL syntax: lexer, parser, and AST definitions

pub mod ast;
pub mod lexer;
pub mod parser;

pub use ast::*;
pub use lexer::{Lexer, Token};
pub use parser::{parse, ParseError, ParseResult, Parser};
