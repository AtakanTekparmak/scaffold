//! Node dispatch: runs a single node (prompt/tool/agent/verify) and returns a Value.

use std::collections::HashMap;

use scaffold_ir::ir::*;

use crate::error::{Error, Result};
use crate::llm::{self, LlmConfig};
use crate::prompt::PromptManager;
use crate::value::Value;

/// Run a single node with the given input value.
///
/// Dispatches based on node kind:
/// - **Prompt**: render template → LLM call → parse output
/// - **Tool**: shell command or json expression evaluation
/// - **Agent**: multi-turn LLM + tools loop
/// - **Verify**: LLM call → parse verdict with `pass` field
pub async fn run_node(
    node: &NodeIR,
    input: Value,
    overrides: &HashMap<String, serde_json::Value>,
    prompt_mgr: &PromptManager,
) -> Result<Value> {
    match node.kind {
        NodeKindIR::Prompt => run_prompt(node, input, overrides, prompt_mgr).await,
        NodeKindIR::Tool => run_tool(node, input, overrides).await,
        NodeKindIR::Agent => run_agent(node, input, overrides, prompt_mgr).await,
        NodeKindIR::Verify => run_verify(node, input, overrides, prompt_mgr).await,
    }
}

/// Resolve a config field, applying overrides if present.
fn resolve_model(config: &NodeConfigIR, overrides: &HashMap<String, serde_json::Value>) -> String {
    if let Some(v) = overrides.get("model") {
        if let Some(s) = v.as_str() {
            return s.to_string();
        }
    }
    config
        .model
        .clone()
        .unwrap_or_else(|| crate::config::config().default_model.clone())
}

fn resolve_temperature(
    config: &NodeConfigIR,
    overrides: &HashMap<String, serde_json::Value>,
) -> Option<f64> {
    if let Some(v) = overrides.get("temperature") {
        return v.as_f64();
    }
    config.temperature
}

fn resolve_max_tokens(
    config: &NodeConfigIR,
    overrides: &HashMap<String, serde_json::Value>,
) -> Option<u64> {
    if let Some(v) = overrides.get("max_tokens") {
        return v.as_u64();
    }
    config.max_tokens
}

/// Load template content from StringOrFileIR.
pub(crate) fn load_template(sof: &StringOrFileIR) -> Result<String> {
    match sof {
        StringOrFileIR::Literal { value } => Ok(value.clone()),
        StringOrFileIR::File { path } => {
            // Try loading from file; look for .jinja/.j2 extensions first, then raw
            let candidates = [
                format!("{}.jinja", path),
                format!("{}.j2", path),
                path.clone(),
                format!("prompts/{}.jinja", path),
                format!("prompts/{}", path),
                format!("examples/prompts/{}/template.jinja", path),
            ];
            for candidate in &candidates {
                if let Ok(content) = std::fs::read_to_string(candidate) {
                    return Ok(content);
                }
            }
            Err(Error::TemplateError(format!(
                "template file not found: {}",
                path
            )))
        }
    }
}

/// Render a template with the given context.
fn render_template(
    template_source: &str,
    ctx: &Value,
    prompt_mgr: &PromptManager,
) -> Result<String> {
    prompt_mgr.render_inline(template_source, ctx)
}

/// Build an LlmConfig from node config + overrides.
fn build_llm_config(
    node_config: &NodeConfigIR,
    overrides: &HashMap<String, serde_json::Value>,
    system_prompt: Option<String>,
) -> LlmConfig {
    let model = resolve_model(node_config, overrides);
    let temperature = resolve_temperature(node_config, overrides);
    let max_tokens = resolve_max_tokens(node_config, overrides);

    let mut config = LlmConfig::new().with_model(model);
    if let Some(t) = temperature {
        config = config.with_temperature(t as f32);
    }
    if let Some(mt) = max_tokens {
        config = config.with_max_tokens(mt as u32);
    }
    if let Some(sp) = system_prompt {
        config = config.with_system_prompt(sp);
    }
    config
}

// ── Prompt Node ──

async fn run_prompt(
    node: &NodeIR,
    input: Value,
    overrides: &HashMap<String, serde_json::Value>,
    prompt_mgr: &PromptManager,
) -> Result<Value> {
    let config = &node.config;

    // Load and render template (overrides take priority)
    let template_source = if let Some(v) = overrides.get("template") {
        match v.as_str() {
            Some(s) => s.to_string(),
            None => match &config.template {
                Some(sof) => load_template(sof)?,
                None => return run_prompt_raw(node, &input.to_string(), overrides).await,
            },
        }
    } else {
        match &config.template {
            Some(sof) => load_template(sof)?,
            None => {
                // If no template, just use the input as the prompt
                return run_prompt_raw(node, &input.to_string(), overrides).await;
            }
        }
    };

    // Build template context from input
    let ctx = match &input {
        Value::Map(_) | Value::Struct { .. } => input.clone(),
        other => {
            let mut map = HashMap::new();
            map.insert("input".to_string(), other.clone());
            Value::Map(map)
        }
    };

    let prompt_text = render_template(&template_source, &ctx, prompt_mgr)?;

    // Load system prompt (overrides take priority)
    let system = if let Some(v) = overrides.get("system") {
        v.as_str().map(|s| s.to_string())
    } else {
        match &config.system {
            Some(sof) => Some(load_template(sof)?),
            None => None,
        }
    };

    let llm_config = build_llm_config(config, overrides, system);
    let response = llm::query_with_config(&prompt_text, &llm_config).await?;

    // If the node has structured json output, try to parse
    if config.json.is_some() {
        let schema = "{}"; // TODO: derive schema from output type
        match llm::query_structured_with_config(&prompt_text, schema, &llm_config).await {
            Ok(val) => return Ok(val),
            Err(_) => {
                // Fall back to string output
            }
        }
    }

    // Try to parse JSON from the response when output type is structured
    if !matches!(node.output, TypeIR::String) {
        if let Ok(json_val) = llm::parse_json_with_repairs(&response) {
            return Ok(Value::from(json_val));
        }
    }

    Ok(Value::String(response))
}

async fn run_prompt_raw(
    node: &NodeIR,
    prompt_text: &str,
    overrides: &HashMap<String, serde_json::Value>,
) -> Result<Value> {
    let llm_config = build_llm_config(&node.config, overrides, None);
    let response = llm::query_with_config(prompt_text, &llm_config).await?;
    Ok(Value::String(response))
}

// ── Tool Node ──

async fn run_tool(
    node: &NodeIR,
    input: Value,
    overrides: &HashMap<String, serde_json::Value>,
) -> Result<Value> {
    let config = &node.config;

    // Shell tool: run a shell command (check override first)
    let shell_override = overrides
        .get("shell")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let shell_source = shell_override.as_deref().or(config.shell.as_deref());
    if let Some(shell_cmd) = shell_source {
        let ctx = match &input {
            Value::Map(m) => Value::Map(m.clone()),
            Value::Struct { fields, .. } => Value::Map(fields.clone()),
            other => {
                let mut map = HashMap::new();
                map.insert("input".to_string(), other.clone());
                Value::Map(map)
            }
        };
        let prompt_mgr = PromptManager::new();
        let rendered_cmd = prompt_mgr.interpolate(shell_cmd, &ctx)?;

        let timeout = overrides
            .get("timeout")
            .and_then(|v| v.as_u64())
            .or(config.timeout);

        // Meta-agent proposed shell commands run sandboxed (no network access).
        let is_meta_proposed = shell_override.is_some();
        let output = match (is_meta_proposed, timeout) {
            (true, Some(t)) => {
                crate::shell::execute_sandboxed_with_timeout(&rendered_cmd, t * 1000)?
            }
            (true, None) => crate::shell::execute_sandboxed(&rendered_cmd)?,
            (false, Some(t)) => crate::shell::execute_with_timeout(&rendered_cmd, t * 1000)?,
            (false, None) => crate::shell::execute(&rendered_cmd)?,
        };

        return Ok(Value::String(output));
    }

    // JSON tool: evaluate json expression fields
    if let Some(ref json_fields) = config.json {
        let mut result = HashMap::new();
        for field in json_fields {
            let val = eval_json_field_expr(&field.value, &input);
            result.insert(field.key.clone(), val);
        }
        return Ok(Value::Map(result));
    }

    Err(Error::NodeFailed {
        node: node.name.clone(),
        message: "tool node has neither shell nor json configuration".into(),
    })
}

/// Evaluate a simple expression for JSON tool fields.
fn eval_json_field_expr(expr: &ExprIR, input: &Value) -> Value {
    match expr {
        ExprIR::LitString { value } => Value::String(value.clone()),
        ExprIR::LitInt { value } => Value::Int(*value),
        ExprIR::LitFloat { value } => Value::Float(*value),
        ExprIR::LitBool { value } => Value::Bool(*value),
        ExprIR::LitNull => Value::Null,
        ExprIR::Ident { name } => {
            if name == "input" {
                input.clone()
            } else {
                input.field(name).cloned().unwrap_or(Value::Null)
            }
        }
        ExprIR::FieldAccess { base, field } => {
            let base_val = eval_json_field_expr(base, input);
            base_val.field(field).cloned().unwrap_or(Value::Null)
        }
        _ => Value::Null,
    }
}

// ── Agent Node ──

async fn run_agent(
    node: &NodeIR,
    input: Value,
    overrides: &HashMap<String, serde_json::Value>,
    prompt_mgr: &PromptManager,
) -> Result<Value> {
    let config = &node.config;
    let max_turns = overrides
        .get("max_turns")
        .and_then(|v| v.as_u64())
        .or(config.max_turns)
        .unwrap_or(5);

    // Load system prompt
    let system = match &config.system {
        Some(sof) => Some(load_template(sof)?),
        None => None,
    };

    // Load initial prompt from template
    let initial_prompt = match &config.template {
        Some(sof) => {
            let template_source = load_template(sof)?;
            let ctx = match &input {
                Value::Map(_) | Value::Struct { .. } => input.clone(),
                other => {
                    let mut map = HashMap::new();
                    map.insert("input".to_string(), other.clone());
                    Value::Map(map)
                }
            };
            render_template(&template_source, &ctx, prompt_mgr)?
        }
        None => input.to_string(),
    };

    // For now, agent is a multi-turn prompt loop.
    // Each turn: send prompt → get response → check if done.
    let llm_config = build_llm_config(config, overrides, system);
    let conversation = initial_prompt;
    let mut last_response = String::new();

    for _turn in 0..max_turns {
        last_response = llm::query_with_config(&conversation, &llm_config).await?;
        // Simple heuristic: if the response doesn't contain a tool call marker, we're done
        // In a full implementation, this would parse tool calls and execute them
        break;
    }

    Ok(Value::String(last_response))
}

// ── Verify Node ──

async fn run_verify(
    node: &NodeIR,
    input: Value,
    overrides: &HashMap<String, serde_json::Value>,
    prompt_mgr: &PromptManager,
) -> Result<Value> {
    let config = &node.config;

    // Load and render template (overrides take priority)
    let template_source = if let Some(v) = overrides.get("template") {
        match v.as_str() {
            Some(s) => s.to_string(),
            None => match &config.template {
                Some(sof) => load_template(sof)?,
                None => {
                    return Err(Error::NodeFailed {
                        node: node.name.clone(),
                        message: "verify node requires a template".into(),
                    });
                }
            },
        }
    } else {
        match &config.template {
            Some(sof) => load_template(sof)?,
            None => {
                return Err(Error::NodeFailed {
                    node: node.name.clone(),
                    message: "verify node requires a template".into(),
                });
            }
        }
    };

    let ctx = match &input {
        Value::Map(_) | Value::Struct { .. } => input.clone(),
        other => {
            let mut map = HashMap::new();
            map.insert("input".to_string(), other.clone());
            Value::Map(map)
        }
    };

    let prompt_text = render_template(&template_source, &ctx, prompt_mgr)?;

    // Load system prompt (overrides take priority)
    let system = if let Some(v) = overrides.get("system") {
        v.as_str().map(|s| s.to_string())
    } else {
        match &config.system {
            Some(sof) => Some(load_template(sof)?),
            None => None,
        }
    };

    let llm_config = build_llm_config(config, overrides, system);

    // Ask for structured output with pass field
    let schema = r#"{"pass": true, "feedback": "string"}"#;
    let result = llm::query_structured_with_config(&prompt_text, schema, &llm_config).await?;

    // Ensure the result has a "pass" field
    match &result {
        Value::Map(m) if m.contains_key("pass") => Ok(result),
        _ => {
            // Try to extract pass from the raw response
            let raw = llm::query_with_config(&prompt_text, &llm_config).await?;
            let pass = raw.to_lowercase().contains("pass")
                || raw.to_lowercase().contains("true")
                || raw.to_lowercase().contains("correct");
            let mut verdict = HashMap::new();
            verdict.insert("pass".to_string(), Value::Bool(pass));
            verdict.insert("feedback".to_string(), Value::String(raw));
            Ok(Value::Map(verdict))
        }
    }
}
