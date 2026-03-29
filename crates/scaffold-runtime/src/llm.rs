//! LLM (Language Model) integration via rig.rs
//!
//! This module provides a unified interface for calling LLMs from scaffold.
//!
//! Supported providers:
//! - OpenAI (gpt-4, gpt-4o, gpt-4o-mini, o1, o3, etc.)
//! - Anthropic (claude-3-opus, claude-3-sonnet, claude-3-haiku, etc.)
//!
//! # Configuration
//!
//! API keys can be set via:
//! - Environment variables: OPENAI_API_KEY, ANTHROPIC_API_KEY
//! - Config file: ~/.scaffold/config.toml
//!
//! # Usage
//!
//! ```ignore
//! use scaffold_runtime::llm::{query, query_with_model, AgentBuilder};
//!
//! // Simple query with default model
//! let response = query("What is 2+2?").await?;
//!
//! // Query with specific model
//! let response = query_with_model("claude-3-sonnet", "Explain quantum computing").await?;
//!
//! // Build a custom agent
//! let agent = AgentBuilder::new("gpt-4o")
//!     .system_prompt("You are a helpful coding assistant.")
//!     .temperature(0.7)
//!     .build();
//! let response = agent.prompt("Write a Python hello world").await?;
//! ```

use crate::config::{config, parse_model_id};
use crate::error::{Error, Result};
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::{AssistantContent, CompletionModel};
use rig::providers::{anthropic, openai};
use std::collections::VecDeque;
use std::sync::{Mutex, OnceLock};

fn structured_mock_queue() -> &'static Mutex<VecDeque<serde_json::Value>> {
    static QUEUE: OnceLock<Mutex<VecDeque<serde_json::Value>>> = OnceLock::new();
    QUEUE.get_or_init(|| Mutex::new(VecDeque::new()))
}

/// Configuration for LLM calls
#[derive(Debug, Clone, Default)]
pub struct LlmConfig {
    /// Model to use (e.g., "gpt-4", "claude-3-sonnet")
    pub model: Option<String>,
    /// Temperature for generation (0.0 - 2.0)
    pub temperature: Option<f32>,
    /// Maximum tokens to generate
    pub max_tokens: Option<u32>,
    /// System prompt to prepend
    pub system_prompt: Option<String>,
}

impl LlmConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }

    pub fn with_max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = Some(tokens);
        self
    }

    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }
}

/// Query an LLM with a prompt using the default model
pub async fn query(prompt: &str) -> Result<String> {
    let model = &config().default_model;
    query_with_model(model, prompt).await
}

/// Query an LLM with a specific model
pub async fn query_with_model(model: &str, prompt: &str) -> Result<String> {
    let config = LlmConfig::new().with_model(model);
    query_with_config(prompt, &config).await
}

/// Default LLM request timeout in seconds.
const DEFAULT_LLM_TIMEOUT_SECS: u64 = 120;

/// Resolve the LLM request timeout from env or default.
fn llm_timeout() -> std::time::Duration {
    let secs = std::env::var("SCAFFOLD_LLM_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(DEFAULT_LLM_TIMEOUT_SECS);
    std::time::Duration::from_secs(secs)
}

/// Query an LLM with full configuration
pub async fn query_with_config(prompt: &str, llm_config: &LlmConfig) -> Result<String> {
    // Check for mock mode (for testing)
    if std::env::var("SCAFFOLD_LLM_MOCK").is_ok() {
        return Ok(format!("Mock LLM response for: {}", prompt));
    }

    let timeout = llm_timeout();
    match tokio::time::timeout(timeout, query_with_config_inner(prompt, llm_config)).await {
        Ok(result) => result,
        Err(_) => Err(Error::Timeout(format!(
            "LLM request timed out after {}s",
            timeout.as_secs()
        ))),
    }
}

async fn query_with_config_inner(prompt: &str, llm_config: &LlmConfig) -> Result<String> {
    let model = llm_config
        .model
        .as_deref()
        .unwrap_or(&config().default_model);

    // If OpenRouter API key is available, use OpenRouter for all models
    if std::env::var("OPENROUTER_API_KEY").is_ok() || config().get_api_key("openrouter").is_some() {
        return query_openrouter(model, prompt, llm_config).await;
    }

    // Otherwise, route based on model prefix
    let (provider, model_name) = parse_model_id(model);

    match provider {
        "openai" => query_openai(model_name, prompt, llm_config).await,
        "anthropic" => query_anthropic(model_name, prompt, llm_config).await,
        "openrouter" => query_openrouter(model_name, prompt, llm_config).await,
        other => Err(Error::ConfigError(format!(
            "Unknown LLM provider: {}. Supported: openai, anthropic, openrouter",
            other
        ))),
    }
}

/// Extract text from AssistantContent
fn extract_text<'a, I>(content: I) -> String
where
    I: IntoIterator<Item = &'a AssistantContent>,
{
    content
        .into_iter()
        .filter_map(|item| match item {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect::<String>()
}

fn extract_text_or_error<'a, I>(content: I) -> Result<String>
where
    I: IntoIterator<Item = &'a AssistantContent>,
{
    let text = extract_text(content);
    if text.trim().is_empty() {
        Err(Error::Runtime(
            "LLM response did not contain any text content".to_string(),
        ))
    } else {
        Ok(text)
    }
}

/// Query OpenAI models
async fn query_openai(model: &str, prompt: &str, llm_config: &LlmConfig) -> Result<String> {
    let cfg = config();
    let api_key = cfg.get_api_key("openai").ok_or_else(|| {
        Error::ConfigError(
            "OpenAI API key not found. Set OPENAI_API_KEY env var or add to ~/.scaffold/config.toml"
                .to_string(),
        )
    })?;

    // Set env vars for rig to pick up (supports custom base_url for OpenRouter, etc.)
    std::env::set_var("OPENAI_API_KEY", api_key);
    if let Some(base_url) = cfg.get_base_url("openai") {
        std::env::set_var("OPENAI_BASE_URL", base_url);
    }

    let client: openai::Client = openai::Client::from_env();
    let completion_model = client.completion_model(model);

    // Build the completion request
    let mut request = completion_model.completion_request(prompt);

    if let Some(ref system) = llm_config.system_prompt {
        request = request.preamble(system.clone());
    }

    if let Some(temp) = llm_config.temperature {
        request = request.temperature(temp as f64);
    }

    if let Some(max_tokens) = llm_config.max_tokens {
        request = request.max_tokens(max_tokens as u64);
    }

    let response = request
        .send()
        .await
        .map_err(|e| Error::Runtime(format!("OpenAI API error: {}", e)))?;

    // Extract text from the first choice
    extract_text_or_error(response.choice.iter())
}

/// Query Anthropic models
async fn query_anthropic(model: &str, prompt: &str, llm_config: &LlmConfig) -> Result<String> {
    let cfg = config();
    let api_key = cfg.get_api_key("anthropic").ok_or_else(|| {
        Error::ConfigError(
            "Anthropic API key not found. Set ANTHROPIC_API_KEY env var or add to ~/.scaffold/config.toml"
                .to_string(),
        )
    })?;

    // Set env vars for rig to pick up
    std::env::set_var("ANTHROPIC_API_KEY", api_key);
    if let Some(base_url) = cfg.get_base_url("anthropic") {
        std::env::set_var("ANTHROPIC_BASE_URL", base_url);
    }

    let client: anthropic::Client = anthropic::Client::from_env();
    let completion_model = client.completion_model(model);

    // Build the completion request
    let mut request = completion_model.completion_request(prompt);

    if let Some(ref system) = llm_config.system_prompt {
        request = request.preamble(system.clone());
    }

    if let Some(temp) = llm_config.temperature {
        request = request.temperature(temp as f64);
    }

    if let Some(max_tokens) = llm_config.max_tokens {
        request = request.max_tokens(max_tokens as u64);
    }

    let response = request
        .send()
        .await
        .map_err(|e| Error::Runtime(format!("Anthropic API error: {}", e)))?;

    // Extract text from the first choice
    extract_text_or_error(response.choice.iter())
}

/// Query via OpenRouter (unified API for all models)
///
/// OpenRouter provides access to OpenAI, Anthropic, and many other models
/// through a single API endpoint using the OpenAI SDK format.
async fn query_openrouter(model: &str, prompt: &str, llm_config: &LlmConfig) -> Result<String> {
    let cfg = config();

    // Get API key from env or config
    let api_key = std::env::var("OPENROUTER_API_KEY")
        .ok()
        .or_else(|| cfg.get_api_key("openrouter").map(|s| s.to_string()))
        .ok_or_else(|| {
            Error::ConfigError(
                "OpenRouter API key not found. Set OPENROUTER_API_KEY env var or add to ~/.scaffold/config.toml"
                    .to_string(),
            )
        })?;

    // Build chat completions request body directly (avoids rig's Responses API
    // which sends fields like service_tier that OpenRouter may reject).
    let mut messages = Vec::new();
    if let Some(ref system) = llm_config.system_prompt {
        messages.push(serde_json::json!({"role": "system", "content": system}));
    }
    messages.push(serde_json::json!({"role": "user", "content": prompt}));

    let mut body = serde_json::json!({
        "model": model,
        "messages": messages,
    });

    if let Some(temp) = llm_config.temperature {
        body["temperature"] = serde_json::json!(temp);
    }
    if let Some(max_tokens) = llm_config.max_tokens {
        body["max_tokens"] = serde_json::json!(max_tokens);
    }

    let api_key_owned = api_key.clone();
    let body_string = body.to_string();

    let response_text = tokio::task::spawn_blocking(move || {
        let resp = ureq::post("https://openrouter.ai/api/v1/chat/completions")
            .set("Authorization", &format!("Bearer {}", api_key_owned))
            .set("Content-Type", "application/json")
            .send_string(&body_string)
            .map_err(|e| Error::Runtime(format!("OpenRouter API error: {}", e)))?;

        resp.into_string()
            .map_err(|e| Error::Runtime(format!("OpenRouter read error: {}", e)))
    })
    .await
    .map_err(|e| Error::Runtime(format!("OpenRouter task error: {}", e)))??;

    // Parse the chat completions response
    let parsed: serde_json::Value = serde_json::from_str(&response_text)
        .map_err(|e| Error::Runtime(format!("OpenRouter JSON parse error: {}", e)))?;

    // Check for API errors
    if let Some(err) = parsed.get("error") {
        let msg = err
            .get("message")
            .and_then(|m| m.as_str())
            .unwrap_or("unknown error");
        return Err(Error::Runtime(format!("OpenRouter API error: {}", msg)));
    }

    // Extract content from first choice
    parsed
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|choice| choice.get("message"))
        .and_then(|msg| msg.get("content"))
        .and_then(|content| content.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| Error::Runtime("OpenRouter: no content in response".to_string()))
}

/// Query and parse response as JSON (typed)
pub async fn query_json<T: serde::de::DeserializeOwned>(prompt: &str) -> Result<T> {
    let response = query(prompt).await?;
    serde_json::from_str(&response)
        .map_err(|e| Error::Runtime(format!("Failed to parse LLM response as JSON: {}", e)))
}

/// Query with structured JSON output
///
/// Wraps the prompt with instructions to return JSON matching the given schema,
/// and parses the response into a Value.
///
/// # Example schema
/// ```text
/// { "answer": "string", "confidence": "high|medium|low", "reasoning": "string" }
/// ```
pub async fn query_structured(prompt: &str, schema: &str) -> Result<crate::Value> {
    query_structured_with_config(prompt, schema, &LlmConfig::default()).await
}

/// Query with structured JSON output and explicit LLM configuration.
pub async fn query_structured_with_config(
    prompt: &str,
    schema: &str,
    llm_config: &LlmConfig,
) -> Result<crate::Value> {
    if let Some(mock_value) = structured_mock_queue()
        .lock()
        .map_err(|_| Error::Runtime("failed to lock structured LLM mock queue".to_string()))?
        .pop_front()
    {
        return Ok(json_to_value(mock_value));
    }

    if let Ok(mock_json) = std::env::var("SCAFFOLD_LLM_MOCK_JSON") {
        let json_value: serde_json::Value = serde_json::from_str(&mock_json).map_err(|e| {
            Error::Runtime(format!(
                "Failed to parse SCAFFOLD_LLM_MOCK_JSON as JSON: {}",
                e
            ))
        })?;
        return Ok(json_to_value(json_value));
    }

    let structured_prompt = format!(
        "{}\n\nRespond with ONLY valid JSON matching this schema (no markdown, no explanation):\n{}",
        prompt, schema
    );

    let response = query_with_config(&structured_prompt, llm_config).await?;
    let json_value = match parse_json_with_repairs(&response) {
        Ok(value) => value,
        Err(initial_error) => {
            let repair_prompt = format!(
                "Repair the following malformed JSON so it becomes valid JSON matching this schema.\n\
                 Preserve the exact intended content as much as possible.\n\
                 Return ONLY valid JSON, with no markdown or explanation.\n\n\
                 Schema:\n{}\n\n\
                 Malformed JSON:\n{}",
                schema, response
            );
            let repaired_response = query_with_config(&repair_prompt, llm_config).await?;
            parse_json_with_repairs(&repaired_response).map_err(|repair_error| {
                Error::Runtime(format!(
                    "Failed to parse LLM response as JSON: {}. Failed to repair malformed JSON: {}. Response was: {}",
                    initial_error, repair_error, response
                ))
            })?
        }
    };

    // Convert to our Value type
    Ok(json_to_value(json_value))
}

/// Extract JSON from a response that might be wrapped in markdown code blocks
fn extract_json(response: &str) -> &str {
    let trimmed = response.trim();

    // Check for ```json ... ``` blocks
    if let Some(start) = trimmed.find("```json") {
        let content_start = start + 7;
        if let Some(end) = trimmed[content_start..].find("```") {
            return trimmed[content_start..content_start + end].trim();
        }
    }

    // Check for ``` ... ``` blocks
    if let Some(start) = trimmed.find("```") {
        let content_start = start + 3;
        // Skip optional language identifier on same line
        let newline_pos = trimmed[content_start..].find('\n').unwrap_or(0);
        let actual_start = content_start + newline_pos;
        if let Some(end) = trimmed[actual_start..].find("```") {
            return trimmed[actual_start..actual_start + end].trim();
        }
    }

    // Return as-is if no code blocks found
    trimmed
}

pub(crate) fn parse_json_with_repairs(
    response: &str,
) -> std::result::Result<serde_json::Value, String> {
    let mut attempts = Vec::new();
    let primary = extract_balanced_json_candidate(response)
        .unwrap_or_else(|| extract_json(response).trim().to_string());
    push_json_attempt(&mut attempts, primary);

    let mut index = 0;
    let mut last_error = None;
    while index < attempts.len() {
        let candidate = attempts[index].clone();
        match serde_json::from_str::<serde_json::Value>(&candidate) {
            Ok(value) => return Ok(value),
            Err(error) => {
                last_error = Some(error.to_string());
                push_json_attempt(&mut attempts, strip_trailing_commas(&candidate));
                if let Some(closed) = close_json_delimiters(&candidate) {
                    push_json_attempt(&mut attempts, closed);
                }
                if let Some(closed) = close_json_delimiters(&strip_trailing_commas(&candidate)) {
                    push_json_attempt(&mut attempts, closed);
                }
            }
        }
        index += 1;
    }

    Err(last_error.unwrap_or_else(|| "response was empty".to_string()))
}

fn push_json_attempt(attempts: &mut Vec<String>, candidate: String) {
    let trimmed = candidate.trim();
    if trimmed.is_empty() {
        return;
    }
    if !attempts.iter().any(|existing| existing == trimmed) {
        attempts.push(trimmed.to_string());
    }
}

fn extract_balanced_json_candidate(response: &str) -> Option<String> {
    let trimmed = response.trim();
    let start = trimmed.find(|ch| ['{', '['].contains(&ch))?;
    let slice = &trimmed[start..];
    let mut stack = Vec::new();
    let mut in_string = false;
    let mut escape = false;

    for (idx, ch) in slice.char_indices() {
        if in_string {
            if escape {
                escape = false;
                continue;
            }
            match ch {
                '\\' => escape = true,
                '"' => in_string = false,
                _ => {}
            }
            continue;
        }

        match ch {
            '"' => in_string = true,
            '{' => stack.push('}'),
            '[' => stack.push(']'),
            '}' | ']' => {
                if stack.pop() != Some(ch) {
                    break;
                }
                if stack.is_empty() {
                    return Some(slice[..=idx].trim().to_string());
                }
            }
            _ => {}
        }
    }

    Some(slice.trim().to_string())
}

fn close_json_delimiters(candidate: &str) -> Option<String> {
    let trimmed = candidate.trim();
    if trimmed.is_empty() {
        return None;
    }

    let mut repaired = String::with_capacity(trimmed.len() + 8);
    let mut stack = Vec::new();
    let mut in_string = false;
    let mut escape = false;

    for ch in trimmed.chars() {
        if in_string {
            repaired.push(ch);
            if escape {
                escape = false;
                continue;
            }
            match ch {
                '\\' => escape = true,
                '"' => in_string = false,
                _ => {}
            }
            continue;
        }

        match ch {
            '"' => {
                in_string = true;
                repaired.push(ch);
            }
            '{' => {
                stack.push('}');
                repaired.push(ch);
            }
            '[' => {
                stack.push(']');
                repaired.push(ch);
            }
            '}' | ']' => {
                while let Some(expected) = stack.pop() {
                    if expected == ch {
                        repaired.push(ch);
                        break;
                    }
                    repaired.push(expected);
                }
            }
            _ => repaired.push(ch),
        }
    }

    if in_string {
        repaired.push('"');
    }
    while let Some(ch) = stack.pop() {
        repaired.push(ch);
    }
    Some(repaired)
}

fn strip_trailing_commas(candidate: &str) -> String {
    let mut out = String::with_capacity(candidate.len());
    let chars = candidate.chars().collect::<Vec<_>>();
    let mut index = 0;
    let mut in_string = false;
    let mut escape = false;

    while index < chars.len() {
        let ch = chars[index];
        if in_string {
            out.push(ch);
            if escape {
                escape = false;
            } else {
                match ch {
                    '\\' => escape = true,
                    '"' => in_string = false,
                    _ => {}
                }
            }
            index += 1;
            continue;
        }

        match ch {
            '"' => {
                in_string = true;
                out.push(ch);
            }
            ',' => {
                let mut lookahead = index + 1;
                while lookahead < chars.len() && chars[lookahead].is_whitespace() {
                    lookahead += 1;
                }
                if lookahead < chars.len() && matches!(chars[lookahead], ']' | '}') {
                    index += 1;
                    continue;
                }
                out.push(ch);
            }
            _ => out.push(ch),
        }
        index += 1;
    }

    out
}

/// Convert serde_json::Value to our Value type
fn json_to_value(v: serde_json::Value) -> crate::Value {
    match v {
        serde_json::Value::Null => crate::Value::Null,
        serde_json::Value::Bool(b) => crate::Value::Bool(b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                crate::Value::Int(i)
            } else if let Some(f) = n.as_f64() {
                crate::Value::Float(f)
            } else {
                crate::Value::Null
            }
        }
        serde_json::Value::String(s) => crate::Value::String(s),
        serde_json::Value::Array(arr) => {
            crate::Value::List(arr.into_iter().map(json_to_value).collect())
        }
        serde_json::Value::Object(obj) => crate::Value::Map(
            obj.into_iter()
                .map(|(k, v)| (k, json_to_value(v)))
                .collect(),
        ),
    }
}

/// Trait for custom LLM implementations
pub trait LlmBackend: Send + Sync {
    /// Query the LLM with a prompt
    fn query(&self, prompt: &str) -> impl std::future::Future<Output = Result<String>> + Send;

    /// Query with configuration
    fn query_with_config(
        &self,
        prompt: &str,
        config: &LlmConfig,
    ) -> impl std::future::Future<Output = Result<String>> + Send {
        async move {
            let _ = config;
            self.query(prompt).await
        }
    }
}

/// Agent builder for creating configured LLM agents
#[derive(Debug, Clone)]
pub struct AgentBuilder {
    model: String,
    system_prompt: Option<String>,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
}

impl AgentBuilder {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            system_prompt: None,
            temperature: None,
            max_tokens: None,
        }
    }

    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    pub fn temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }

    pub fn max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = Some(tokens);
        self
    }

    /// Build the agent configuration
    pub fn build_config(&self) -> LlmConfig {
        LlmConfig {
            model: Some(self.model.clone()),
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            system_prompt: self.system_prompt.clone(),
        }
    }

    /// Create an agent that can be used for multiple queries
    pub fn build(self) -> Agent {
        Agent {
            config: self.build_config(),
        }
    }
}

/// A configured agent for making LLM queries
#[derive(Debug, Clone)]
pub struct Agent {
    config: LlmConfig,
}

impl Agent {
    /// Create a new agent with the specified model
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            config: LlmConfig::new().with_model(model),
        }
    }

    /// Query the agent with a prompt
    pub async fn prompt(&self, prompt: &str) -> Result<String> {
        query_with_config(prompt, &self.config).await
    }

    /// Query and parse as JSON
    pub async fn prompt_json<T: serde::de::DeserializeOwned>(&self, prompt: &str) -> Result<T> {
        let response = self.prompt(prompt).await?;
        serde_json::from_str(&response)
            .map_err(|e| Error::Runtime(format!("Failed to parse response as JSON: {}", e)))
    }

    /// Get the model being used
    pub fn model(&self) -> Option<&str> {
        self.config.model.as_deref()
    }

    /// Get the system prompt
    pub fn system_prompt(&self) -> Option<&str> {
        self.config.system_prompt.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rig::completion::message::{Reasoning, Text};

    #[tokio::test]
    async fn test_mock_query() {
        std::env::set_var("SCAFFOLD_LLM_MOCK", "1");
        let result = query("test prompt").await.unwrap();
        assert!(result.contains("Mock LLM response"));
        std::env::remove_var("SCAFFOLD_LLM_MOCK");
    }

    #[test]
    fn test_agent_builder() {
        let agent = AgentBuilder::new("gpt-4o")
            .system_prompt("You are helpful.")
            .temperature(0.7)
            .max_tokens(1000)
            .build();

        assert_eq!(agent.model(), Some("gpt-4o"));
        assert_eq!(agent.system_prompt(), Some("You are helpful."));
    }

    #[test]
    fn test_llm_config_builder() {
        let config = LlmConfig::new()
            .with_model("claude-3-sonnet")
            .with_temperature(0.5)
            .with_max_tokens(2000);

        assert_eq!(config.model, Some("claude-3-sonnet".to_string()));
        assert_eq!(config.temperature, Some(0.5));
        assert_eq!(config.max_tokens, Some(2000));
    }

    #[test]
    fn extract_text_collects_all_text_segments() {
        let content = vec![
            AssistantContent::Reasoning(Reasoning::new("thinking")),
            AssistantContent::Text(Text {
                text: "{\"answer\":".to_string(),
            }),
            AssistantContent::Text(Text {
                text: "\"Paris\"}".to_string(),
            }),
        ];

        assert_eq!(extract_text(content.iter()), "{\"answer\":\"Paris\"}");
    }

    #[test]
    fn extract_text_or_error_rejects_non_text_responses() {
        let content = vec![AssistantContent::Reasoning(Reasoning::new("thinking"))];

        let error = extract_text_or_error(content.iter()).unwrap_err();
        assert!(error
            .to_string()
            .contains("LLM response did not contain any text content"));
    }

    #[test]
    fn parse_json_with_repairs_closes_missing_delimiters() {
        let parsed = parse_json_with_repairs(r#"{"outputs":[[[1,2],[3,4]]}"#).unwrap();
        assert_eq!(
            parsed,
            serde_json::json!({
                "outputs": [[[1, 2], [3, 4]]]
            })
        );
    }

    #[test]
    fn parse_json_with_repairs_strips_trailing_commas() {
        let parsed = parse_json_with_repairs(r#"{"outputs":[[[1,2],[3,4]],],"score":1,}"#).unwrap();
        assert_eq!(
            parsed,
            serde_json::json!({
                "outputs": [[[1, 2], [3, 4]]],
                "score": 1
            })
        );
    }

    #[test]
    fn parse_json_with_repairs_extracts_json_from_surrounding_text() {
        let parsed =
            parse_json_with_repairs(r#"Here is the answer: {"answer":"Paris"} Thanks!"#).unwrap();
        assert_eq!(parsed, serde_json::json!({ "answer": "Paris" }));
    }
}
