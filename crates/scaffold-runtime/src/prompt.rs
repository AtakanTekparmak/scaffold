//! Prompt templating system using Jinja2-compatible syntax
//!
//! This module provides template rendering for prompts using minijinja,
//! supporting variable interpolation, conditionals, loops, and filters.

use crate::error::{Error, Result};
use crate::value::Value;
use minijinja::Environment;
use std::collections::HashMap;
use std::path::Path;
use std::sync::RwLock;

/// Prompt manager for loading and rendering templates
pub struct PromptManager {
    /// Stored templates (name -> source)
    templates: RwLock<HashMap<String, String>>,
    /// Template directory (if any)
    template_dir: Option<std::path::PathBuf>,
}

impl Default for PromptManager {
    fn default() -> Self {
        Self::new()
    }
}

impl PromptManager {
    /// Create a new prompt manager
    pub fn new() -> Self {
        Self {
            templates: RwLock::new(HashMap::new()),
            template_dir: None,
        }
    }

    /// Create a fresh minijinja environment with custom filters
    fn create_env() -> Environment<'static> {
        let mut env = Environment::new();

        // Add useful built-in filters
        env.add_filter("json", |v: minijinja::Value| {
            serde_json::to_string(&v).unwrap_or_else(|_| "null".to_string())
        });

        env.add_filter("trim", |s: String| s.trim().to_string());
        env.add_filter("upper", |s: String| s.to_uppercase());
        env.add_filter("lower", |s: String| s.to_lowercase());

        env.add_filter("truncate", |s: String, len: usize| {
            if s.len() > len {
                format!("{}...", &s[..len.saturating_sub(3)])
            } else {
                s
            }
        });

        env.add_filter("default", |v: minijinja::Value, default: String| {
            if v.is_undefined() || v.is_none() {
                default
            } else {
                v.to_string()
            }
        });

        env
    }

    /// Create a prompt manager with a template directory
    pub fn with_template_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let mut manager = Self::new();
        manager.template_dir = Some(dir.as_ref().to_path_buf());
        manager.load_templates_from_dir(dir)?;
        Ok(manager)
    }

    /// Load templates from a directory
    pub fn load_templates_from_dir(&self, dir: impl AsRef<Path>) -> Result<()> {
        let dir = dir.as_ref();
        if !dir.exists() {
            return Ok(()); // No templates directory is fine
        }

        for entry in std::fs::read_dir(dir).map_err(|e| Error::ConfigError(e.to_string()))? {
            let entry = entry.map_err(|e| Error::ConfigError(e.to_string()))?;
            let path = entry.path();

            if path.extension().map(|e| e == "jinja" || e == "j2").unwrap_or(false) {
                let name = path.file_stem()
                    .and_then(|s| s.to_str())
                    .ok_or_else(|| Error::ConfigError("Invalid template filename".into()))?;

                let content = std::fs::read_to_string(&path)
                    .map_err(|e| Error::ConfigError(format!("Failed to read template {}: {}", path.display(), e)))?;

                self.add_template(name, &content)?;
            }
        }

        Ok(())
    }

    /// Add a template
    pub fn add_template(&self, name: &str, source: &str) -> Result<()> {
        let mut templates = self.templates.write()
            .map_err(|_| Error::Runtime("Lock poisoned".into()))?;
        templates.insert(name.to_string(), source.to_string());
        Ok(())
    }

    /// Get a template source by name
    pub fn get_template(&self, name: &str) -> Option<String> {
        self.templates.read()
            .ok()
            .and_then(|t| t.get(name).cloned())
    }

    /// Check if a template exists
    pub fn has_template(&self, name: &str) -> bool {
        self.templates.read()
            .map(|t| t.contains_key(name))
            .unwrap_or(false)
    }

    /// Render a named template with a context value
    pub fn render(&self, name: &str, ctx: &Value) -> Result<String> {
        let source = self.get_template(name)
            .ok_or_else(|| Error::Runtime(format!("Template '{}' not found", name)))?;

        self.render_inline(&source, ctx)
    }

    /// Render an inline template string
    pub fn render_inline(&self, template: &str, ctx: &Value) -> Result<String> {
        let env = Self::create_env();
        let jinja_ctx = ctx.to_template_value();

        env.render_str(template, jinja_ctx)
            .map_err(|e| Error::Runtime(format!("Template render error: {}", e)))
    }

    /// Render a string with simple {variable} interpolation
    /// This is for shell commands and simple prompts that use {var} syntax
    pub fn interpolate(&self, template: &str, ctx: &Value) -> Result<String> {
        // Convert {var} to {{ var }} for Jinja2 compatibility
        let jinja_template = convert_simple_interpolation(template);
        self.render_inline(&jinja_template, ctx)
    }

    /// Reload templates from directory
    pub fn reload(&self) -> Result<()> {
        if let Some(ref dir) = self.template_dir {
            // Clear existing templates
            {
                let mut templates = self.templates.write()
                    .map_err(|_| Error::Runtime("Lock poisoned".into()))?;
                templates.clear();
            }
            self.load_templates_from_dir(dir)?;
        }
        Ok(())
    }

    /// Get list of loaded template names
    pub fn template_names(&self) -> Vec<String> {
        self.templates.read()
            .map(|t| t.keys().cloned().collect())
            .unwrap_or_default()
    }
}

/// Convert simple {var} interpolation to Jinja2 {{ var }} syntax
fn convert_simple_interpolation(template: &str) -> String {
    let mut result = String::with_capacity(template.len() * 2);
    let mut chars = template.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '{' {
            // Check if it's already {{ (Jinja2 syntax)
            if chars.peek() == Some(&'{') {
                result.push('{');
                result.push(chars.next().unwrap());
            } else if chars.peek() == Some(&'%') {
                // Jinja2 block syntax {% ... %}
                result.push('{');
            } else {
                // Convert {var} to {{ var }}
                result.push_str("{{ ");
                while let Some(&nc) = chars.peek() {
                    if nc == '}' {
                        chars.next();
                        break;
                    }
                    result.push(chars.next().unwrap());
                }
                result.push_str(" }}");
            }
        } else {
            result.push(c);
        }
    }

    result
}

/// Global prompt manager instance
static GLOBAL_PROMPT_MANAGER: std::sync::OnceLock<PromptManager> = std::sync::OnceLock::new();

/// Get the global prompt manager
pub fn global_prompt_manager() -> &'static PromptManager {
    GLOBAL_PROMPT_MANAGER.get_or_init(PromptManager::new)
}

/// Initialize the global prompt manager with a template directory
pub fn init_prompt_manager(dir: impl AsRef<Path>) -> Result<()> {
    let pm = PromptManager::with_template_dir(dir)?;
    GLOBAL_PROMPT_MANAGER.set(pm)
        .map_err(|_| Error::ConfigError("Prompt manager already initialized".into()))
}

/// Render a template using the global prompt manager
pub fn render_template(name: &str, ctx: &Value) -> Result<String> {
    global_prompt_manager().render(name, ctx)
}

/// Interpolate variables in a string using the global prompt manager
pub fn interpolate(template: &str, ctx: &Value) -> Result<String> {
    global_prompt_manager().interpolate(template, ctx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_interpolation() {
        assert_eq!(
            convert_simple_interpolation("cat {path}"),
            "cat {{ path }}"
        );
        assert_eq!(
            convert_simple_interpolation("Hello {name}, your id is {id}"),
            "Hello {{ name }}, your id is {{ id }}"
        );
        // Already Jinja2 syntax should be preserved
        assert_eq!(
            convert_simple_interpolation("{{ already_jinja }}"),
            "{{ already_jinja }}"
        );
    }

    #[test]
    fn test_render_inline() {
        let pm = PromptManager::new();

        let mut ctx_map = HashMap::new();
        ctx_map.insert("name".to_string(), Value::String("World".to_string()));
        let ctx = Value::Map(ctx_map);

        let result = pm.render_inline("Hello {{ name }}!", &ctx).unwrap();
        assert_eq!(result, "Hello World!");
    }

    #[test]
    fn test_interpolate() {
        let pm = PromptManager::new();

        let mut ctx_map = HashMap::new();
        ctx_map.insert("path".to_string(), Value::String("/tmp/test.txt".to_string()));
        let ctx = Value::Map(ctx_map);

        let result = pm.interpolate("cat {path}", &ctx).unwrap();
        assert_eq!(result, "cat /tmp/test.txt");
    }

    #[test]
    fn test_conditionals() {
        let pm = PromptManager::new();

        let mut ctx_map = HashMap::new();
        ctx_map.insert("show_extra".to_string(), Value::Bool(true));
        ctx_map.insert("name".to_string(), Value::String("Test".to_string()));
        let ctx = Value::Map(ctx_map);

        let result = pm.render_inline(
            "Hello{% if show_extra %}, {{ name }}{% endif %}!",
            &ctx
        ).unwrap();
        assert_eq!(result, "Hello, Test!");
    }

    #[test]
    fn test_loops() {
        let pm = PromptManager::new();

        let mut ctx_map = HashMap::new();
        ctx_map.insert("items".to_string(), Value::List(vec![
            Value::String("a".to_string()),
            Value::String("b".to_string()),
            Value::String("c".to_string()),
        ]));
        let ctx = Value::Map(ctx_map);

        let result = pm.render_inline(
            "{% for item in items %}{{ item }}{% if not loop.last %}, {% endif %}{% endfor %}",
            &ctx
        ).unwrap();
        assert_eq!(result, "a, b, c");
    }

    #[test]
    fn test_named_template() {
        let pm = PromptManager::new();
        pm.add_template("greeting", "Hello {{ name }}!").unwrap();

        let mut ctx_map = HashMap::new();
        ctx_map.insert("name".to_string(), Value::String("World".to_string()));
        let ctx = Value::Map(ctx_map);

        let result = pm.render("greeting", &ctx).unwrap();
        assert_eq!(result, "Hello World!");
    }

    #[test]
    fn test_struct_context() {
        let pm = PromptManager::new();

        let mut fields = HashMap::new();
        fields.insert("x".to_string(), Value::Int(10));
        fields.insert("y".to_string(), Value::Int(20));
        let ctx = Value::Struct {
            type_name: "Position".to_string(),
            fields,
        };

        let result = pm.render_inline("Position: ({{ x }}, {{ y }})", &ctx).unwrap();
        assert_eq!(result, "Position: (10, 20)");
    }
}