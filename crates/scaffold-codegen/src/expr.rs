//! Expression IR to Rust expression generation

use crate::util::{escape_string, to_ident};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use scaffold_ir::{ExprIR, LiteralIR};

/// Generate Rust expression from ExprIR
pub fn gen_expr(expr: &ExprIR) -> TokenStream {
    match expr {
        ExprIR::Literal { value } => gen_literal(value),
        ExprIR::Ident { name } => gen_ident(name),
        ExprIR::FieldAccess { base, field } => gen_field_access(base, field),
        ExprIR::Binary { left, op, right } => gen_binary(left, op, right),
        ExprIR::Call { function, args } => gen_call(function, args),
    }
}

/// Generate literal value
fn gen_literal(lit: &LiteralIR) -> TokenStream {
    match lit {
        LiteralIR::Int { value } => {
            let v = *value;
            quote! { #v }
        }
        LiteralIR::Float { value } => {
            let v = *value;
            quote! { #v }
        }
        LiteralIR::String { value } => {
            let escaped = escape_string(value);
            quote! { #escaped.to_string() }
        }
        LiteralIR::Bool { value } => {
            let v = *value;
            quote! { #v }
        }
        LiteralIR::Null => quote! { None },
    }
}

/// Generate identifier reference
fn gen_ident(name: &str) -> TokenStream {
    // Handle special identifiers
    match name {
        "input" => quote! { input },
        "output" => quote! { output },
        "state" => quote! { self.state },
        "null" | "None" => quote! { None },
        "true" => quote! { true },
        "false" => quote! { false },
        _ => {
            // Check for dotted paths like "input.x"
            if name.contains('.') {
                let parts: Vec<_> = name.split('.').collect();
                let base = format_ident!("{}", to_ident(parts[0]));
                let mut result = quote! { #base };
                for part in &parts[1..] {
                    let field = format_ident!("{}", to_ident(part));
                    result = quote! { #result.#field };
                }
                result
            } else {
                let ident = format_ident!("{}", to_ident(name));
                quote! { #ident }
            }
        }
    }
}

/// Generate condition expression with input field prefixing
/// Top-level identifiers (except special ones) are prefixed with "input."
pub fn gen_condition_with_input(expr: &ExprIR) -> TokenStream {
    gen_expr_with_input_prefix(expr, true, &[])
}

/// Generate condition expression with selective input field prefixing
/// Only identifiers in input_fields are prefixed with "input."
pub fn gen_condition_with_input_fields(expr: &ExprIR, input_fields: &[String]) -> TokenStream {
    gen_expr_with_field_prefixing(expr, input_fields, &[])
}

/// Generate condition expression with both input and state field prefixing
/// Input fields are prefixed with "input.", state fields with "self.state."
pub fn gen_condition_with_fields(expr: &ExprIR, input_fields: &[String], state_fields: &[String]) -> TokenStream {
    gen_expr_with_field_prefixing(expr, input_fields, state_fields)
}

/// Check if an expression is a null literal
fn is_null_literal(expr: &ExprIR) -> bool {
    match expr {
        ExprIR::Literal { value: LiteralIR::Null } => true,
        ExprIR::Ident { name } if name == "null" || name == "None" => true,
        _ => false,
    }
}

fn gen_expr_with_field_prefixing(expr: &ExprIR, input_fields: &[String], state_fields: &[String]) -> TokenStream {
    match expr {
        ExprIR::Literal { value } => gen_literal(value),
        ExprIR::Ident { name } => {
            // Special identifiers that shouldn't be prefixed
            match name.as_str() {
                "input" | "output" | "state" | "null" | "None" | "true" | "false" => {
                    gen_ident(name)
                }
                _ => {
                    // Check if it's an input field
                    if input_fields.contains(&name.to_string()) {
                        let ident = format_ident!("{}", to_ident(name));
                        quote! { input.#ident }
                    // Check if it's a state field
                    } else if state_fields.contains(&name.to_string()) {
                        let ident = format_ident!("{}", to_ident(name));
                        quote! { self.state.#ident }
                    } else {
                        gen_ident(name)
                    }
                }
            }
        }
        ExprIR::FieldAccess { base, field } => {
            let base_expr = gen_expr_with_field_prefixing(base, input_fields, state_fields);
            let field_ident = format_ident!("{}", to_ident(field));
            quote! { #base_expr.#field_ident }
        }
        ExprIR::Binary { left, op, right } => {
            // Special handling for null comparisons - use .is_some()/.is_none()
            let right_is_null = is_null_literal(right);
            let left_is_null = is_null_literal(left);

            if right_is_null && (op == "==" || op == "!=") {
                let left_expr = gen_expr_with_field_prefixing(left, input_fields, state_fields);
                if op == "==" {
                    return quote! { #left_expr.is_none() };
                } else {
                    return quote! { #left_expr.is_some() };
                }
            }
            if left_is_null && (op == "==" || op == "!=") {
                let right_expr = gen_expr_with_field_prefixing(right, input_fields, state_fields);
                if op == "==" {
                    return quote! { #right_expr.is_none() };
                } else {
                    return quote! { #right_expr.is_some() };
                }
            }

            let left_expr = gen_expr_with_field_prefixing(left, input_fields, state_fields);
            let right_expr = gen_expr_with_field_prefixing(right, input_fields, state_fields);
            match op.as_str() {
                "==" => quote! { (#left_expr == #right_expr) },
                "!=" => quote! { (#left_expr != #right_expr) },
                "<" => quote! { (#left_expr < #right_expr) },
                "<=" => quote! { (#left_expr <= #right_expr) },
                ">" => quote! { (#left_expr > #right_expr) },
                ">=" => quote! { (#left_expr >= #right_expr) },
                "&&" | "and" => quote! { (#left_expr && #right_expr) },
                "||" | "or" => quote! { (#left_expr || #right_expr) },
                "+" => quote! { (#left_expr + #right_expr) },
                "-" => quote! { (#left_expr - #right_expr) },
                "*" => quote! { (#left_expr * #right_expr) },
                "/" => quote! { (#left_expr / #right_expr) },
                "%" => quote! { (#left_expr % #right_expr) },
                other => {
                    let method = format_ident!("{}", to_ident(other));
                    quote! { #left_expr.#method(#right_expr) }
                }
            }
        }
        ExprIR::Call { function, args } => {
            let arg_exprs: Vec<_> = args
                .iter()
                .map(|a| gen_expr_with_field_prefixing(a, input_fields, state_fields))
                .collect();
            let func_name = format_ident!("{}", to_ident(function));

            // Check if this is a built-in function (don't pass by reference)
            let builtins = [
                "len", "is_empty", "contains", "is_some", "is_none",
                "unwrap", "unwrap_or", "abs", "min", "max", "not"
            ];

            if builtins.contains(&function.as_str()) {
                if arg_exprs.is_empty() {
                    quote! { #func_name() }
                } else {
                    quote! { #func_name(#(#arg_exprs),*) }
                }
            } else {
                // User-defined functions take references to avoid move errors
                if arg_exprs.is_empty() {
                    quote! { #func_name() }
                } else {
                    quote! { #func_name(#(&#arg_exprs),*) }
                }
            }
        }
    }
}

fn gen_expr_with_input_prefix(expr: &ExprIR, prefix_all: bool, input_fields: &[String]) -> TokenStream {
    match expr {
        ExprIR::Literal { value } => gen_literal(value),
        ExprIR::Ident { name } => {
            // Special identifiers that shouldn't be prefixed
            match name.as_str() {
                "input" | "output" | "state" | "null" | "None" | "true" | "false" => {
                    gen_ident(name)
                }
                _ => {
                    // Prefix with input. if:
                    // - prefix_all is true, OR
                    // - the identifier is in input_fields
                    let should_prefix = prefix_all || input_fields.contains(&name.to_string());
                    if should_prefix {
                        let ident = format_ident!("{}", to_ident(name));
                        quote! { input.#ident }
                    } else {
                        gen_ident(name)
                    }
                }
            }
        }
        ExprIR::FieldAccess { base, field } => {
            let base_expr = gen_expr_with_input_prefix(base, prefix_all, input_fields);
            let field_ident = format_ident!("{}", to_ident(field));
            quote! { #base_expr.#field_ident }
        }
        ExprIR::Binary { left, op, right } => {
            let left_expr = gen_expr_with_input_prefix(left, prefix_all, input_fields);
            let right_expr = gen_expr_with_input_prefix(right, prefix_all, input_fields);
            match op.as_str() {
                "==" => quote! { (#left_expr == #right_expr) },
                "!=" => quote! { (#left_expr != #right_expr) },
                "<" => quote! { (#left_expr < #right_expr) },
                "<=" => quote! { (#left_expr <= #right_expr) },
                ">" => quote! { (#left_expr > #right_expr) },
                ">=" => quote! { (#left_expr >= #right_expr) },
                "&&" | "and" => quote! { (#left_expr && #right_expr) },
                "||" | "or" => quote! { (#left_expr || #right_expr) },
                "+" => quote! { (#left_expr + #right_expr) },
                "-" => quote! { (#left_expr - #right_expr) },
                "*" => quote! { (#left_expr * #right_expr) },
                "/" => quote! { (#left_expr / #right_expr) },
                "%" => quote! { (#left_expr % #right_expr) },
                other => {
                    let method = format_ident!("{}", to_ident(other));
                    quote! { #left_expr.#method(#right_expr) }
                }
            }
        }
        ExprIR::Call { function, args } => {
            let arg_exprs: Vec<_> = args
                .iter()
                .map(|a| gen_expr_with_input_prefix(a, prefix_all, input_fields))
                .collect();
            let func_name = format_ident!("{}", to_ident(function));
            if arg_exprs.is_empty() {
                quote! { #func_name() }
            } else {
                quote! { #func_name(#(#arg_exprs),*) }
            }
        }
    }
}

/// Generate field access expression
fn gen_field_access(base: &ExprIR, field: &str) -> TokenStream {
    let base_expr = gen_expr(base);
    let field_ident = format_ident!("{}", to_ident(field));
    quote! { #base_expr.#field_ident }
}

/// Generate binary operation
fn gen_binary(left: &ExprIR, op: &str, right: &ExprIR) -> TokenStream {
    let left_expr = gen_expr(left);
    let right_expr = gen_expr(right);

    match op {
        // Comparison operators
        "==" => quote! { (#left_expr == #right_expr) },
        "!=" => quote! { (#left_expr != #right_expr) },
        "<" => quote! { (#left_expr < #right_expr) },
        "<=" => quote! { (#left_expr <= #right_expr) },
        ">" => quote! { (#left_expr > #right_expr) },
        ">=" => quote! { (#left_expr >= #right_expr) },

        // Logical operators
        "&&" | "and" => quote! { (#left_expr && #right_expr) },
        "||" | "or" => quote! { (#left_expr || #right_expr) },

        // Arithmetic operators
        "+" => quote! { (#left_expr + #right_expr) },
        "-" => quote! { (#left_expr - #right_expr) },
        "*" => quote! { (#left_expr * #right_expr) },
        "/" => quote! { (#left_expr / #right_expr) },
        "%" => quote! { (#left_expr % #right_expr) },

        // Unknown operator - generate as method call
        other => {
            let method = format_ident!("{}", to_ident(other));
            quote! { #left_expr.#method(#right_expr) }
        }
    }
}

/// Generate function call
fn gen_call(function: &str, args: &[ExprIR]) -> TokenStream {
    let arg_exprs: Vec<_> = args.iter().map(gen_expr).collect();

    // Handle special built-in functions
    match function {
        // Collection functions
        "len" => {
            if let Some(arg) = arg_exprs.first() {
                quote! { #arg.len() }
            } else {
                quote! { 0 }
            }
        }
        "is_empty" => {
            if let Some(arg) = arg_exprs.first() {
                quote! { #arg.is_empty() }
            } else {
                quote! { true }
            }
        }
        "contains" => {
            if arg_exprs.len() >= 2 {
                let collection = &arg_exprs[0];
                let item = &arg_exprs[1];
                quote! { #collection.contains(&#item) }
            } else {
                quote! { false }
            }
        }

        // Option functions
        "is_some" => {
            if let Some(arg) = arg_exprs.first() {
                quote! { #arg.is_some() }
            } else {
                quote! { false }
            }
        }
        "is_none" => {
            if let Some(arg) = arg_exprs.first() {
                quote! { #arg.is_none() }
            } else {
                quote! { true }
            }
        }
        "unwrap" => {
            if let Some(arg) = arg_exprs.first() {
                quote! { #arg.clone().unwrap() }
            } else {
                quote! { panic!("unwrap on None") }
            }
        }
        "unwrap_or" => {
            if arg_exprs.len() >= 2 {
                let opt = &arg_exprs[0];
                let default = &arg_exprs[1];
                quote! { #opt.clone().unwrap_or(#default) }
            } else if let Some(arg) = arg_exprs.first() {
                quote! { #arg.clone().unwrap_or_default() }
            } else {
                quote! { Default::default() }
            }
        }

        // Math functions
        "abs" => {
            if let Some(arg) = arg_exprs.first() {
                quote! { #arg.abs() }
            } else {
                quote! { 0 }
            }
        }
        "min" => {
            if arg_exprs.len() >= 2 {
                let a = &arg_exprs[0];
                let b = &arg_exprs[1];
                quote! { std::cmp::min(#a, #b) }
            } else {
                quote! { 0 }
            }
        }
        "max" => {
            if arg_exprs.len() >= 2 {
                let a = &arg_exprs[0];
                let b = &arg_exprs[1];
                quote! { std::cmp::max(#a, #b) }
            } else {
                quote! { 0 }
            }
        }

        // Logical functions
        "not" => {
            if let Some(arg) = arg_exprs.first() {
                quote! { !#arg }
            } else {
                quote! { true }
            }
        }

        // Default: generate as method or function call
        _ => {
            let func_name = format_ident!("{}", to_ident(function));
            if arg_exprs.is_empty() {
                quote! { #func_name() }
            } else {
                quote! { #func_name(#(#arg_exprs),*) }
            }
        }
    }
}

/// Generate an expression that evaluates a condition
/// Returns code that evaluates to bool
pub fn gen_condition(expr: &ExprIR) -> TokenStream {
    gen_expr(expr)
}

/// Generate expression with reference to input parameter
pub fn gen_expr_with_input(expr: &ExprIR, input_name: &str) -> TokenStream {
    // Replace "input" references with the actual input name
    rewrite_expr(expr, input_name)
}

fn rewrite_expr(expr: &ExprIR, input_name: &str) -> TokenStream {
    match expr {
        ExprIR::Ident { name } if name == "input" => {
            let ident = format_ident!("{}", input_name);
            quote! { #ident }
        }
        ExprIR::FieldAccess { base, field } => {
            let base_expr = rewrite_expr(base, input_name);
            let field_ident = format_ident!("{}", to_ident(field));
            quote! { #base_expr.#field_ident }
        }
        ExprIR::Binary { left, op, right } => {
            let left_expr = rewrite_expr(left, input_name);
            let right_expr = rewrite_expr(right, input_name);
            match op.as_str() {
                "==" => quote! { (#left_expr == #right_expr) },
                "!=" => quote! { (#left_expr != #right_expr) },
                "<" => quote! { (#left_expr < #right_expr) },
                "<=" => quote! { (#left_expr <= #right_expr) },
                ">" => quote! { (#left_expr > #right_expr) },
                ">=" => quote! { (#left_expr >= #right_expr) },
                "&&" | "and" => quote! { (#left_expr && #right_expr) },
                "||" | "or" => quote! { (#left_expr || #right_expr) },
                "+" => quote! { (#left_expr + #right_expr) },
                "-" => quote! { (#left_expr - #right_expr) },
                "*" => quote! { (#left_expr * #right_expr) },
                "/" => quote! { (#left_expr / #right_expr) },
                _ => gen_expr(expr),
            }
        }
        ExprIR::Call { function, args } => {
            let rewritten_args: Vec<_> = args.iter().map(|a| rewrite_expr(a, input_name)).collect();
            let func_name = format_ident!("{}", to_ident(function));
            quote! { #func_name(#(#rewritten_args),*) }
        }
        _ => gen_expr(expr),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gen_literal() {
        let int_lit = ExprIR::Literal {
            value: LiteralIR::Int { value: 42 },
        };
        assert_eq!(gen_expr(&int_lit).to_string(), "42i64");

        let bool_lit = ExprIR::Literal {
            value: LiteralIR::Bool { value: true },
        };
        assert_eq!(gen_expr(&bool_lit).to_string(), "true");
    }

    #[test]
    fn test_gen_binary() {
        let expr = ExprIR::Binary {
            left: Box::new(ExprIR::Ident {
                name: "x".to_string(),
            }),
            op: "==".to_string(),
            right: Box::new(ExprIR::Literal {
                value: LiteralIR::Int { value: 5 },
            }),
        };
        assert!(gen_expr(&expr).to_string().contains("=="));
    }

    #[test]
    fn test_gen_field_access() {
        let expr = ExprIR::FieldAccess {
            base: Box::new(ExprIR::Ident {
                name: "input".to_string(),
            }),
            field: "position".to_string(),
        };
        let result = gen_expr(&expr).to_string();
        assert!(result.contains("input"));
        assert!(result.contains("position"));
    }
}
