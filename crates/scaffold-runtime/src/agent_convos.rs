//! Agent conversation logging helpers.
//!
//! When `SCAFFOLD_AGENT_CONVOS_PATH` is set, generated agents append one JSON
//! record per invocation. This captures nested agent calls inside pipelines too.

use serde::Serialize;
use serde_json::{json, Value};
use std::fs::OpenOptions;
use std::io::Write;
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

static CONVO_WRITE_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn env_opt(key: &str) -> Option<String> {
    match std::env::var(key) {
        Ok(value) if !value.trim().is_empty() => Some(value),
        _ => None,
    }
}

fn to_value_or_error<T: Serialize>(value: &T) -> Value {
    serde_json::to_value(value).unwrap_or_else(|e| {
        json!({
            "_serialization_error": e.to_string()
        })
    })
}

/// Append one agent conversation record to the JSONL sink specified by
/// `SCAFFOLD_AGENT_CONVOS_PATH`. Errors are intentionally swallowed so logging
/// never impacts execution behavior.
pub fn maybe_log_agent_conversation<InputT: Serialize, OutputT: Serialize>(
    event: &str,
    agent_name: &str,
    model: &str,
    system_prompt: &str,
    tools: &[&str],
    input: &InputT,
    user_message: &str,
    response_text: Option<&str>,
    output: Option<&OutputT>,
    error: Option<&str>,
) {
    let path = match env_opt("SCAFFOLD_AGENT_CONVOS_PATH") {
        Some(path) => path,
        None => return,
    };

    let lock = CONVO_WRITE_LOCK.get_or_init(|| Mutex::new(()));
    let _guard = match lock.lock() {
        Ok(g) => g,
        Err(_) => return,
    };

    let path_ref = std::path::Path::new(&path);
    if let Some(parent) = path_ref.parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    let mut file = match OpenOptions::new().create(true).append(true).open(path_ref) {
        Ok(file) => file,
        Err(_) => return,
    };

    let record = json!({
        "event": event,
        "timestamp_ms": now_ms(),
        "pid": std::process::id(),
        "thread_id": format!("{:?}", std::thread::current().id()),
        "run_id": env_opt("SCAFFOLD_RUN_ID"),
        "case_id": env_opt("SCAFFOLD_CASE_ID"),
        "target_kind": env_opt("SCAFFOLD_TARGET_KIND"),
        "target_name": env_opt("SCAFFOLD_TARGET_NAME"),
        "agent_name": agent_name,
        "model": model,
        "tools": tools,
        "system_prompt": system_prompt,
        "input": to_value_or_error(input),
        "user_message": user_message,
        "response_text": response_text.unwrap_or(""),
        "output": output.map(to_value_or_error),
        "error": error,
    });

    let _ = serde_json::to_writer(&mut file, &record);
    let _ = writeln!(file);
}
