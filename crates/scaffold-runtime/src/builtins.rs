//! Built-in tool primitives for scaffold DSL
//!
//! These replace opaque `shell("python3 -c ...")` strings with typed, analyzable
//! functions that the scaffold compiler can reason about and optimize.
//!
//! All functions are synchronous (safe to call from async contexts via spawn_blocking
//! or directly since ureq manages its own I/O).

use crate::error::{Error, Result};

// ─── HTTP ────────────────────────────────────────────────────────────────────

/// Get the configured timeout for builtin operations.
fn builtin_timeout() -> std::time::Duration {
    let secs = std::env::var("SCAFFOLD_SHELL_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(30);
    std::time::Duration::from_secs(secs)
}

/// Build a ureq agent with the configured timeout.
fn agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout(builtin_timeout())
        .user_agent("scaffold-runtime/0.1")
        .build()
}

/// HTTP GET request, returns the response body as a string.
///
/// Respects `SCAFFOLD_SHELL_TIMEOUT_SECS` for the request timeout (default 30s).
pub fn http_get(url: &str) -> Result<String> {
    let body = agent()
        .get(url)
        .call()
        .map_err(|e| Error::ActionFailed {
            action: "http_get".into(),
            message: format!("{}", e),
        })?
        .into_string()
        .map_err(|e| Error::ActionFailed {
            action: "http_get".into(),
            message: format!("failed to read body: {}", e),
        })?;
    Ok(body)
}

/// HTTP GET with custom headers (provided as `key: value` newline-separated pairs).
pub fn http_get_with_headers(url: &str, headers: &str) -> Result<String> {
    let mut req = agent().get(url);
    for line in headers.lines() {
        if let Some((key, value)) = line.split_once(':') {
            req = req.set(key.trim(), value.trim());
        }
    }
    let body = req
        .call()
        .map_err(|e| Error::ActionFailed {
            action: "http_get_with_headers".into(),
            message: format!("{}", e),
        })?
        .into_string()
        .map_err(|e| Error::ActionFailed {
            action: "http_get_with_headers".into(),
            message: format!("failed to read body: {}", e),
        })?;
    Ok(body)
}

/// HTTP POST request with a string body, returns response body.
pub fn http_post(url: &str, body: &str) -> Result<String> {
    let resp = agent()
        .post(url)
        .set("Content-Type", "application/json")
        .send_string(body)
        .map_err(|e| Error::ActionFailed {
            action: "http_post".into(),
            message: format!("{}", e),
        })?
        .into_string()
        .map_err(|e| Error::ActionFailed {
            action: "http_post".into(),
            message: format!("failed to read body: {}", e),
        })?;
    Ok(resp)
}

// ─── JSON ────────────────────────────────────────────────────────────────────

/// Parse a JSON string into a serde_json::Value.
pub fn json_parse(text: &str) -> Result<serde_json::Value> {
    serde_json::from_str(text).map_err(|e| Error::ParseError(format!("json_parse: {}", e)))
}

/// Extract a value from a JSON object by dot-separated path.
///
/// Path syntax: `"query.search.0.title"` — dots descend into objects,
/// numeric segments index into arrays.
pub fn json_get(value: &serde_json::Value, path: &str) -> Result<String> {
    let mut current = value;
    for segment in path.split('.') {
        if segment.is_empty() {
            continue;
        }
        if let Ok(idx) = segment.parse::<usize>() {
            current = current.get(idx).ok_or_else(|| Error::ActionFailed {
                action: "json_get".into(),
                message: format!("array index {} out of bounds at path '{}'", idx, path),
            })?;
        } else {
            current = current.get(segment).ok_or_else(|| Error::ActionFailed {
                action: "json_get".into(),
                message: format!("key '{}' not found at path '{}'", segment, path),
            })?;
        }
    }
    match current {
        serde_json::Value::String(s) => Ok(s.clone()),
        other => Ok(other.to_string()),
    }
}

/// Serialize a value to a JSON string.
pub fn json_stringify(value: &serde_json::Value) -> String {
    serde_json::to_string(value).unwrap_or_default()
}

// ─── Regex ───────────────────────────────────────────────────────────────────

/// Extract the first match of a regex pattern from text.
/// Returns the full match (group 0), or group 1 if a capture group exists.
pub fn regex_extract(text: &str, pattern: &str) -> Result<String> {
    let re = regex::Regex::new(pattern)
        .map_err(|e| Error::ParseError(format!("invalid regex '{}': {}", pattern, e)))?;
    match re.captures(text) {
        Some(caps) => {
            // Return first capture group if it exists, otherwise full match
            let m = caps
                .get(1)
                .or_else(|| caps.get(0))
                .map(|m| m.as_str().to_string())
                .unwrap_or_default();
            Ok(m)
        }
        None => Ok(String::new()),
    }
}

/// Extract all matches of a regex pattern from text.
/// Returns full matches (group 0), or group 1 from each match if capture groups exist.
pub fn regex_extract_all(text: &str, pattern: &str) -> Result<Vec<String>> {
    let re = regex::Regex::new(pattern)
        .map_err(|e| Error::ParseError(format!("invalid regex '{}': {}", pattern, e)))?;
    let results: Vec<String> = re
        .captures_iter(text)
        .map(|caps| {
            caps.get(1)
                .or_else(|| caps.get(0))
                .map(|m| m.as_str().to_string())
                .unwrap_or_default()
        })
        .collect();
    Ok(results)
}

/// Test whether text matches a regex pattern.
pub fn regex_matches(text: &str, pattern: &str) -> Result<bool> {
    let re = regex::Regex::new(pattern)
        .map_err(|e| Error::ParseError(format!("invalid regex '{}': {}", pattern, e)))?;
    Ok(re.is_match(text))
}

/// Replace all matches of a regex pattern in text.
pub fn regex_replace(text: &str, pattern: &str, replacement: &str) -> Result<String> {
    let re = regex::Regex::new(pattern)
        .map_err(|e| Error::ParseError(format!("invalid regex '{}': {}", pattern, e)))?;
    Ok(re.replace_all(text, replacement).to_string())
}

// ─── Text ────────────────────────────────────────────────────────────────────

/// Split text by a delimiter.
pub fn text_split(text: &str, delimiter: &str) -> Vec<String> {
    text.split(delimiter).map(|s| s.to_string()).collect()
}

/// Join strings with a delimiter.
pub fn text_join(items: &[String], delimiter: &str) -> String {
    items.join(delimiter)
}

/// Truncate text to a maximum number of characters.
pub fn text_truncate(text: &str, max_chars: i64) -> String {
    let max = max_chars.max(0) as usize;
    if text.len() <= max {
        text.to_string()
    } else {
        let mut end = max;
        while end > 0 && !text.is_char_boundary(end) { end -= 1; }
        text[..end].to_string()
    }
}

/// Strip HTML tags from text, returning plain text.
pub fn html_strip(text: &str) -> String {
    // Remove script/style blocks first
    let re_script = regex::Regex::new(r"(?is)<script[^>]*>.*?</script>").unwrap();
    let text = re_script.replace_all(text, " ");
    let re_style = regex::Regex::new(r"(?is)<style[^>]*>.*?</style>").unwrap();
    let text = re_style.replace_all(&text, " ");
    // Remove all remaining tags
    let re_tags = regex::Regex::new(r"<[^>]+>").unwrap();
    let text = re_tags.replace_all(&text, " ");
    // Collapse whitespace
    let re_ws = regex::Regex::new(r"\s+").unwrap();
    re_ws.replace_all(&text, " ").trim().to_string()
}

/// Count occurrences of a substring in text.
pub fn text_count(text: &str, substring: &str) -> i64 {
    text.matches(substring).count() as i64
}

/// Check if text contains a substring (case-insensitive).
pub fn text_contains_ci(text: &str, substring: &str) -> bool {
    text.to_lowercase().contains(&substring.to_lowercase())
}

// ─── URL ─────────────────────────────────────────────────────────────────────

/// URL-encode a string.
pub fn url_encode(text: &str) -> String {
    // Percent-encode everything except unreserved characters
    let mut encoded = String::with_capacity(text.len() * 3);
    for byte in text.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                encoded.push(byte as char);
            }
            _ => {
                encoded.push_str(&format!("%{:02X}", byte));
            }
        }
    }
    encoded
}

/// URL-decode a percent-encoded string.
pub fn url_decode(text: &str) -> String {
    let mut decoded = Vec::with_capacity(text.len());
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Ok(byte) = u8::from_str_radix(&text[i + 1..i + 3], 16) {
                decoded.push(byte);
                i += 3;
                continue;
            }
        }
        decoded.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&decoded).to_string()
}

// ─── List of all builtin names (for codegen) ─────────────────────────────────

/// Returns the set of all builtin function names.
/// Used by codegen to distinguish builtins from tool calls.
pub const BUILTIN_NAMES: &[&str] = &[
    "http_get",
    "http_get_with_headers",
    "http_post",
    "json_parse",
    "json_get",
    "json_stringify",
    "regex_extract",
    "regex_extract_all",
    "regex_matches",
    "regex_replace",
    "text_split",
    "text_join",
    "text_truncate",
    "html_strip",
    "text_count",
    "text_contains_ci",
    "url_encode",
    "url_decode",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_url_encode_decode() {
        assert_eq!(url_encode("hello world"), "hello%20world");
        assert_eq!(url_decode("hello%20world"), "hello world");
        assert_eq!(url_encode("a+b=c"), "a%2Bb%3Dc");
    }

    #[test]
    fn test_json_parse_and_get() {
        let val = json_parse(r#"{"a":{"b":[1,2,3]}}"#).unwrap();
        assert_eq!(json_get(&val, "a.b.1").unwrap(), "2");
    }

    #[test]
    fn test_regex_extract() {
        let result = regex_extract("age is 21 years", r"(\d+)").unwrap();
        assert_eq!(result, "21");
    }

    #[test]
    fn test_regex_extract_all() {
        let results = regex_extract_all("a1 b2 c3", r"(\d+)").unwrap();
        assert_eq!(results, vec!["1", "2", "3"]);
    }

    #[test]
    fn test_text_split_join() {
        let parts = text_split("a,b,c", ",");
        assert_eq!(parts, vec!["a", "b", "c"]);
        assert_eq!(text_join(&parts, "-"), "a-b-c");
    }

    #[test]
    fn test_html_strip() {
        let html = "<p>Hello <b>world</b></p>";
        let plain = html_strip(html);
        assert_eq!(plain, "Hello world");
    }

    #[test]
    fn test_text_truncate() {
        assert_eq!(text_truncate("hello world", 5), "hello");
        assert_eq!(text_truncate("hi", 10), "hi");
    }

    #[test]
    fn test_text_contains_ci() {
        assert!(text_contains_ci("Hello World", "hello"));
        assert!(!text_contains_ci("Hello World", "xyz"));
    }

    #[test]
    fn test_regex_replace() {
        let result = regex_replace("hello 123 world 456", r"\d+", "NUM").unwrap();
        assert_eq!(result, "hello NUM world NUM");
    }

    #[test]
    fn test_json_stringify() {
        let val = serde_json::json!({"key": "value"});
        let s = json_stringify(&val);
        assert!(s.contains("key"));
        assert!(s.contains("value"));
    }
}
