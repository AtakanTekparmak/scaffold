//! Shell command execution utilities

use crate::error::{Error, Result};
use std::process::{Command, Stdio};

/// Build the macOS sandbox profile at runtime.
///
/// Starts from `allow default` then carves out restrictions:
/// - **No network** — prevents data exfiltration
/// - **No file access under /Users except CWD** — can't read/write user files outside the project
/// - **No process execution under /Users except CWD** — can't run scripts from elsewhere
///
/// System paths (/usr, /bin, /opt, etc.) remain accessible so tools (python, jq, etc.) work.
#[cfg(target_os = "macos")]
fn sandbox_profile() -> String {
    let cwd = std::env::current_dir()
        .unwrap_or_else(|_| std::path::PathBuf::from("."))
        .canonicalize()
        .unwrap_or_else(|_| {
            std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."))
        });
    let cwd_str = cwd.to_string_lossy();

    format!(
        concat!(
            "(version 1)\n",
            "(allow default)\n",
            // No network access
            "(deny network*)\n",
            // No file reads under /Users (blocks reading secrets, SSH keys, other projects, etc.)
            "(deny file-read-data (subpath \"/Users\"))\n",
            "(deny file-read-metadata (subpath \"/Users\"))\n",
            // No file writes under /Users
            "(deny file-write* (subpath \"/Users\"))\n",
            // No executing binaries from /Users
            "(deny process-exec (subpath \"/Users\"))\n",
            // Exception: allow full access to the working directory
            "(allow file-read-data (subpath \"{cwd}\"))\n",
            "(allow file-read-metadata (subpath \"{cwd}\"))\n",
            "(allow file-write* (subpath \"{cwd}\"))\n",
            "(allow process-exec (subpath \"{cwd}\"))\n",
        ),
        cwd = cwd_str,
    )
}

/// Build a `Command` for shell execution, optionally sandboxed on macOS.
fn build_command(command: &str, sandboxed: bool) -> Command {
    #[cfg(target_os = "macos")]
    if sandboxed {
        let profile = sandbox_profile();
        let mut cmd = Command::new("sandbox-exec");
        cmd.arg("-p")
            .arg(profile)
            .arg("sh")
            .arg("-c")
            .arg(command);
        return cmd;
    }

    #[cfg(not(target_os = "macos"))]
    let _ = sandboxed;

    let mut cmd = Command::new("sh");
    cmd.arg("-c").arg(command);
    cmd
}

/// Run a command to completion and return stdout.
fn run_to_completion(command: &str, sandboxed: bool) -> Result<String> {
    let output = build_command(command, sandboxed)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()?;

    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let label = if sandboxed { "shell (sandboxed)" } else { "shell" };
        Err(Error::ActionFailed {
            action: label.to_string(),
            message: format!("command failed: {}", stderr),
        })
    }
}

/// Spawn a command and wait with a timeout.
fn run_with_deadline(command: &str, sandboxed: bool, timeout_ms: u64) -> Result<String> {
    use std::time::{Duration, Instant};

    let mut child = build_command(command, sandboxed)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    let deadline = Instant::now() + Duration::from_millis(timeout_ms);

    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let stdout = child
                    .stdout
                    .take()
                    .map(|mut s| {
                        let mut buf = Vec::new();
                        std::io::Read::read_to_end(&mut s, &mut buf).ok();
                        buf
                    })
                    .unwrap_or_default();
                let stderr = child
                    .stderr
                    .take()
                    .map(|mut s| {
                        let mut buf = Vec::new();
                        std::io::Read::read_to_end(&mut s, &mut buf).ok();
                        buf
                    })
                    .unwrap_or_default();

                if status.success() {
                    return Ok(String::from_utf8_lossy(&stdout).to_string());
                } else {
                    let stderr_str = String::from_utf8_lossy(&stderr);
                    let label = if sandboxed { "shell (sandboxed)" } else { "shell" };
                    return Err(Error::ActionFailed {
                        action: label.to_string(),
                        message: format!("command failed: {}", stderr_str),
                    });
                }
            }
            Ok(None) => {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait(); // reap zombie
                    return Err(Error::Timeout(format!(
                        "shell command timed out after {}s",
                        timeout_ms / 1000
                    )));
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(Error::Runtime(format!("failed to wait on child: {}", e)));
            }
        }
    }
}

/// Resolve the environment-level timeout, if set.
fn env_timeout_ms() -> Option<u64> {
    std::env::var("SCAFFOLD_SHELL_TIMEOUT_SECS")
        .ok()
        .and_then(|raw| raw.parse::<u64>().ok())
        .filter(|v| *v > 0)
        .map(|s| s * 1000)
}

// ── Public API ──

/// Execute a shell command and return its output.
///
/// Respects the `SCAFFOLD_SHELL_TIMEOUT_SECS` environment variable.
pub fn execute(command: &str) -> Result<String> {
    if let Some(ms) = env_timeout_ms() {
        return run_with_deadline(command, false, ms);
    }
    run_to_completion(command, false)
}

/// Execute a shell command with a timeout (in milliseconds).
pub fn execute_with_timeout(command: &str, timeout_ms: u64) -> Result<String> {
    run_with_deadline(command, false, timeout_ms)
}

/// Execute a shell command in a sandbox (for meta-agent proposed commands).
///
/// On macOS, wraps the command with `sandbox-exec` to deny network access.
/// On other platforms, executes normally (no OS-level sandbox available).
pub fn execute_sandboxed(command: &str) -> Result<String> {
    if let Some(ms) = env_timeout_ms() {
        return run_with_deadline(command, true, ms);
    }
    run_to_completion(command, true)
}

/// Execute a sandboxed shell command with a timeout (in milliseconds).
pub fn execute_sandboxed_with_timeout(command: &str, timeout_ms: u64) -> Result<String> {
    run_with_deadline(command, true, timeout_ms)
}

/// Execute a shell command and return raw bytes.
pub fn execute_bytes(command: &str) -> Result<Vec<u8>> {
    let output = Command::new("sh")
        .arg("-c")
        .arg(command)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()?;

    if output.status.success() {
        Ok(output.stdout)
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        Err(Error::ActionFailed {
            action: "shell".to_string(),
            message: format!("command failed: {}", stderr),
        })
    }
}

/// Check if a command is available on the system.
pub fn command_exists(name: &str) -> bool {
    Command::new("which")
        .arg(name)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execute_simple() {
        let result = execute("echo hello").unwrap();
        assert!(result.contains("hello"));
    }

    #[test]
    fn test_execute_failure() {
        let result = execute("exit 1");
        assert!(result.is_err());
    }

    #[test]
    fn test_execute_sandboxed_simple() {
        // Sandboxed execution should work for basic commands
        let result = execute_sandboxed("echo sandboxed").unwrap();
        assert!(result.contains("sandboxed"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_execute_sandboxed_denies_network() {
        // Network access should be denied in sandbox
        let result = execute_sandboxed_with_timeout("curl -s https://example.com", 5000);
        assert!(result.is_err(), "sandboxed curl should fail");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_execute_sandboxed_denies_file_read_outside_cwd() {
        // Reading user files outside working directory should be denied
        let result = execute_sandboxed("cat ~/.zshrc 2>&1 || cat ~/.bashrc 2>&1");
        // Either the command fails, or it returns an error about permission denied
        match result {
            Err(_) => {} // command failed — good
            Ok(output) => {
                // If the command "succeeded" it should contain a sandbox denial message
                assert!(
                    output.contains("Operation not permitted")
                        || output.contains("denied")
                        || output.contains("No such file"),
                    "sandbox should deny reading files outside CWD, got: {}",
                    output
                );
            }
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_execute_sandboxed_allows_cwd_read() {
        // Reading files inside the working directory should work
        let result = execute_sandboxed("ls Cargo.toml");
        assert!(result.is_ok(), "sandbox should allow reading CWD files: {:?}", result.err());
    }

    #[test]
    fn test_command_exists() {
        assert!(command_exists("sh"));
        assert!(!command_exists("nonexistent_command_xyz"));
    }
}
