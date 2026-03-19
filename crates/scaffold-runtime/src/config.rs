//! Configuration management for scaffold runtime
//!
//! Configuration is loaded from:
//! 1. Environment variables (highest priority)
//! 2. ~/.scaffold/config.toml
//! 3. ./scaffold.toml (project-local)
//! 4. Default values (lowest priority)

// Config module doesn't use runtime errors directly
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::OnceLock;

/// Global config singleton
static CONFIG: OnceLock<Config> = OnceLock::new();

/// Initialize global config from a specific file path.
/// Must be called before any `config()` access.
pub fn init_from_path(path: &std::path::Path) -> Result<(), String> {
    let content = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    init_from_str(&content)
}

/// Initialize global config from a TOML string.
/// Must be called before any `config()` access.
pub fn init_from_str(toml_str: &str) -> Result<(), String> {
    let mut config: Config = toml::from_str(toml_str).map_err(|e| e.to_string())?;
    config.apply_env_overrides();
    CONFIG
        .set(config)
        .map_err(|_| "Config already initialized".to_string())
}

/// Get the global config (loads on first access)
pub fn config() -> &'static Config {
    CONFIG.get_or_init(Config::load)
}

/// Main configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// LLM provider settings
    #[serde(default)]
    pub llm: LlmConfig,

    /// Default model for tasks without explicit config
    #[serde(default = "default_model")]
    pub default_model: String,

    /// Logging verbosity
    #[serde(default)]
    pub verbose: bool,
}

fn default_model() -> String {
    "gpt-4o-mini".to_string()
}

impl Default for Config {
    fn default() -> Self {
        Self {
            llm: LlmConfig::default(),
            default_model: default_model(),
            verbose: false,
        }
    }
}

/// LLM provider configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LlmConfig {
    /// OpenAI configuration
    #[serde(default)]
    pub openai: ProviderConfig,

    /// Anthropic configuration
    #[serde(default)]
    pub anthropic: ProviderConfig,

    /// OpenRouter configuration (unified API for all models)
    #[serde(default)]
    pub openrouter: ProviderConfig,

    /// Custom providers (name -> config)
    #[serde(default)]
    pub providers: HashMap<String, ProviderConfig>,
}

/// Provider-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProviderConfig {
    /// API key (can also be set via env var)
    pub api_key: Option<String>,

    /// Base URL (for custom endpoints)
    pub base_url: Option<String>,

    /// Default model for this provider
    pub default_model: Option<String>,

    /// Organization ID (OpenAI)
    pub organization: Option<String>,

    /// Max tokens default
    pub max_tokens: Option<u32>,

    /// Temperature default
    pub temperature: Option<f32>,
}

impl Config {
    /// Load configuration from all sources
    pub fn load() -> Self {
        let _ = dotenvy::dotenv();
        let mut config = Config::default();

        // Load from ~/.scaffold/config.toml
        if let Some(home_config) = Self::load_home_config() {
            config = config.merge(home_config);
        }

        // Load from ./scaffold.toml
        if let Some(local_config) = Self::load_local_config() {
            config = config.merge(local_config);
        }

        // Override with environment variables
        config.apply_env_overrides();

        config
    }

    /// Load config from home directory
    fn load_home_config() -> Option<Config> {
        let home = dirs::home_dir()?;
        let config_path = home.join(".scaffold").join("config.toml");
        Self::load_from_file(&config_path)
    }

    /// Load config from current directory
    fn load_local_config() -> Option<Config> {
        Self::load_from_file(&PathBuf::from("scaffold.toml"))
    }

    /// Load config from a specific file
    fn load_from_file(path: &PathBuf) -> Option<Config> {
        let content = std::fs::read_to_string(path).ok()?;
        toml::from_str(&content).ok()
    }

    /// Merge another config into this one (other takes precedence)
    fn merge(mut self, other: Config) -> Config {
        if !other.default_model.is_empty() {
            self.default_model = other.default_model;
        }
        self.verbose = self.verbose || other.verbose;

        // Merge LLM config
        self.llm = self.llm.merge(other.llm);

        self
    }

    /// Apply environment variable overrides
    fn apply_env_overrides(&mut self) {
        // API keys from environment
        if let Ok(key) = std::env::var("OPENAI_API_KEY") {
            self.llm.openai.api_key = Some(key);
        }
        if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
            self.llm.anthropic.api_key = Some(key);
        }
        if let Ok(key) = std::env::var("OPENROUTER_API_KEY") {
            self.llm.openrouter.api_key = Some(key);
        }

        // Base URLs
        if let Ok(url) = std::env::var("OPENAI_BASE_URL") {
            self.llm.openai.base_url = Some(url);
        }
        if let Ok(url) = std::env::var("ANTHROPIC_BASE_URL") {
            self.llm.anthropic.base_url = Some(url);
        }

        // Default model
        if let Ok(model) = std::env::var("SCAFFOLD_DEFAULT_MODEL") {
            self.default_model = model;
        }

        // Verbose
        if std::env::var("SCAFFOLD_VERBOSE").is_ok() {
            self.verbose = true;
        }
    }

    /// Get API key for a provider
    pub fn get_api_key(&self, provider: &str) -> Option<&str> {
        match provider {
            "openai" => self.llm.openai.api_key.as_deref(),
            "anthropic" => self.llm.anthropic.api_key.as_deref(),
            "openrouter" => self.llm.openrouter.api_key.as_deref(),
            other => self
                .llm
                .providers
                .get(other)
                .and_then(|p| p.api_key.as_deref()),
        }
    }

    /// Get base URL for a provider (for OpenRouter, Azure, etc.)
    pub fn get_base_url(&self, provider: &str) -> Option<&str> {
        match provider {
            "openai" => self.llm.openai.base_url.as_deref(),
            "anthropic" => self.llm.anthropic.base_url.as_deref(),
            "openrouter" => self.llm.openrouter.base_url.as_deref(),
            other => self
                .llm
                .providers
                .get(other)
                .and_then(|p| p.base_url.as_deref()),
        }
    }

    /// Get provider config
    pub fn get_provider(&self, provider: &str) -> Option<&ProviderConfig> {
        match provider {
            "openai" => Some(&self.llm.openai),
            "anthropic" => Some(&self.llm.anthropic),
            "openrouter" => Some(&self.llm.openrouter),
            other => self.llm.providers.get(other),
        }
    }
}

impl LlmConfig {
    fn merge(mut self, other: LlmConfig) -> LlmConfig {
        self.openai = self.openai.merge(other.openai);
        self.anthropic = self.anthropic.merge(other.anthropic);
        self.openrouter = self.openrouter.merge(other.openrouter);

        for (name, config) in other.providers {
            self.providers.insert(name, config);
        }

        self
    }
}

impl ProviderConfig {
    fn merge(mut self, other: ProviderConfig) -> ProviderConfig {
        if other.api_key.is_some() {
            self.api_key = other.api_key;
        }
        if other.base_url.is_some() {
            self.base_url = other.base_url;
        }
        if other.default_model.is_some() {
            self.default_model = other.default_model;
        }
        if other.organization.is_some() {
            self.organization = other.organization;
        }
        if other.max_tokens.is_some() {
            self.max_tokens = other.max_tokens;
        }
        if other.temperature.is_some() {
            self.temperature = other.temperature;
        }
        self
    }
}

/// Model ID parsing - determines provider from model name
pub fn parse_model_id(model: &str) -> (&str, &str) {
    if model.starts_with("gpt-") || model.starts_with("o1") || model.starts_with("o3") {
        ("openai", model)
    } else if model.starts_with("claude-") {
        ("anthropic", model)
    } else if let Some((provider, model_name)) = model.split_once('/') {
        (provider, model_name)
    } else {
        // Default to openai
        ("openai", model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_model_id() {
        assert_eq!(parse_model_id("gpt-4"), ("openai", "gpt-4"));
        assert_eq!(parse_model_id("gpt-4o-mini"), ("openai", "gpt-4o-mini"));
        assert_eq!(
            parse_model_id("claude-3-sonnet"),
            ("anthropic", "claude-3-sonnet")
        );
        assert_eq!(
            parse_model_id("anthropic/claude-3-opus"),
            ("anthropic", "claude-3-opus")
        );
        assert_eq!(parse_model_id("ollama/llama3"), ("ollama", "llama3"));
    }

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.default_model, "gpt-4o-mini");
    }
}
