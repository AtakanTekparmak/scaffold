//! Utility functions for code generation

use heck::{ToLowerCamelCase, ToPascalCase, ToSnakeCase};

/// Convert a name to snake_case (for function/variable names)
pub fn to_snake_case(name: &str) -> String {
    name.to_snake_case()
}

/// Convert a name to PascalCase (for type/struct names)
pub fn to_pascal_case(name: &str) -> String {
    name.to_pascal_case()
}

/// Convert a name to lowerCamelCase (for field names in some contexts)
pub fn to_lower_camel_case(name: &str) -> String {
    name.to_lower_camel_case()
}

/// Create a valid Rust identifier from a name
/// Handles reserved keywords by appending underscore
pub fn to_ident(name: &str) -> String {
    let snake = to_snake_case(name);
    if is_rust_keyword(&snake) {
        format!("{}_", snake)
    } else {
        snake
    }
}

/// Create a valid Rust type name from a name
pub fn to_type_name(name: &str) -> String {
    let pascal = to_pascal_case(name);
    if is_rust_keyword(&pascal) {
        format!("{}Type", pascal)
    } else {
        pascal
    }
}

/// Check if a string is a Rust keyword or reserved identifier
fn is_rust_keyword(name: &str) -> bool {
    matches!(
        name,
        // Keywords
        "as" | "async"
            | "await"
            | "break"
            | "const"
            | "continue"
            | "crate"
            | "dyn"
            | "else"
            | "enum"
            | "extern"
            | "false"
            | "fn"
            | "for"
            | "if"
            | "impl"
            | "in"
            | "let"
            | "loop"
            | "match"
            | "mod"
            | "move"
            | "mut"
            | "pub"
            | "ref"
            | "return"
            | "self"
            | "Self"
            | "static"
            | "struct"
            | "super"
            | "trait"
            | "true"
            | "type"
            | "unsafe"
            | "use"
            | "where"
            | "while"
            // Reserved
            | "abstract"
            | "become"
            | "box"
            | "do"
            | "final"
            | "macro"
            | "override"
            | "priv"
            | "try"
            | "typeof"
            | "unsized"
            | "virtual"
            | "yield"
            // Built-in attributes that conflict
            | "path"
            | "test"
            | "cfg"
            | "derive"
    )
}

/// Escape a string for use in Rust code
pub fn escape_string(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\\' => result.push_str("\\\\"),
            '"' => result.push_str("\\\""),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            c => result.push(c),
        }
    }
    result
}

/// Create a doc comment from a description
pub fn make_doc_comment(description: &str) -> String {
    description
        .lines()
        .map(|line| format!("/// {}", line))
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snake_case() {
        assert_eq!(to_snake_case("navigateToTarget"), "navigate_to_target");
        assert_eq!(to_snake_case("NavigateToTarget"), "navigate_to_target");
        assert_eq!(to_snake_case("plan_path"), "plan_path");
    }

    #[test]
    fn test_pascal_case() {
        assert_eq!(to_pascal_case("navigate_to_target"), "NavigateToTarget");
        assert_eq!(to_pascal_case("navigateToTarget"), "NavigateToTarget");
        assert_eq!(to_pascal_case("Position"), "Position");
    }

    #[test]
    fn test_to_ident_keywords() {
        assert_eq!(to_ident("type"), "type_");
        assert_eq!(to_ident("match"), "match_");
        assert_eq!(to_ident("name"), "name");
    }

    #[test]
    fn test_escape_string() {
        assert_eq!(escape_string("hello"), "hello");
        assert_eq!(escape_string("hello\nworld"), "hello\\nworld");
        assert_eq!(escape_string(r#"say "hi""#), r#"say \"hi\""#);
    }
}
