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

/// Query an LLM with full configuration
pub async fn query_with_config(prompt: &str, llm_config: &LlmConfig) -> Result<String> {
    // Check for mock mode (for testing)
    if std::env::var("SCAFFOLD_LLM_MOCK").is_ok() {
        return Ok(format!("Mock LLM response for: {}", prompt));
    }

    let model = llm_config
        .model
        .as_deref()
        .unwrap_or(&config().default_model);
    let (provider, model_name) = parse_model_id(model);

    match provider {
        "openai" => query_openai(model_name, prompt, llm_config).await,
        "anthropic" => query_anthropic(model_name, prompt, llm_config).await,
        other => Err(Error::ConfigError(format!(
            "Unknown LLM provider: {}. Supported: openai, anthropic",
            other
        ))),
    }
}

/// Extract text from AssistantContent
fn extract_text(content: AssistantContent) -> String {
    match content {
        AssistantContent::Text(text) => text.text,
        _ => String::new(),
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
    Ok(extract_text(response.choice.first()))
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
    Ok(extract_text(response.choice.first()))
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
    let structured_prompt = format!(
        "{}\n\nRespond with ONLY valid JSON matching this schema (no markdown, no explanation):\n{}",
        prompt, schema
    );

    let response = query(&structured_prompt).await?;

    // Try to extract JSON if wrapped in markdown code blocks
    let json_str = extract_json(&response);

    // Parse into serde_json::Value first
    let json_value: serde_json::Value = serde_json::from_str(json_str).map_err(|e| {
        Error::Runtime(format!(
            "Failed to parse LLM response as JSON: {}. Response was: {}",
            e, response
        ))
    })?;

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
}
