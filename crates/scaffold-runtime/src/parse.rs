//! Parsing helpers for converting shell output into typed values

use crate::error::{Error, Result};

/// Parse a string into i64 with better error messages
pub fn parse_i64(s: &str) -> Result<i64> {
    s.trim()
        .parse::<i64>()
        .map_err(|e| Error::ParseError(format!("failed to parse int from '{}': {}", s, e)))
}

/// Parse a string into f64 with better error messages
pub fn parse_f64(s: &str) -> Result<f64> {
    s.trim()
        .parse::<f64>()
        .map_err(|e| Error::ParseError(format!("failed to parse float from '{}': {}", s, e)))
}

/// Parse a string into bool. Accepts true/false (case-insensitive) and 1/0
pub fn parse_bool(s: &str) -> Result<bool> {
    let t = s.trim().to_ascii_lowercase();
    match t.as_str() {
        "true" | "1" => Ok(true),
        "false" | "0" => Ok(false),
        other => Err(Error::ParseError(format!("failed to parse bool from '{}': expected true/false/1/0", other))),
    }
}

/// Parse JSON string into a type T using serde_json
pub fn parse_json<T: serde::de::DeserializeOwned>(s: &str) -> Result<T> {
    serde_json::from_str::<T>(s.trim())
        .map_err(|e| Error::ParseError(format!("failed to parse json: {}", e)))
}

