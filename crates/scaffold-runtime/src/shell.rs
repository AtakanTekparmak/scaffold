//! Shell command execution utilities

use crate::error::{Error, Result};
use std::process::{Command, Stdio};

/// Execute a shell command and return its output
///
/// Respects the `SCAFFOLD_SHELL_TIMEOUT_SECS` environment variable.
/// When set to a positive integer, shell commands will be killed after
/// that many seconds. When unset or zero, commands run without a timeout.
///
/// # Arguments
/// * `command` - The shell command to execute
///
/// # Returns
/// The stdout of the command as a string, or an error if execution failed
///
/// # Example
/// ```ignore
/// use scaffold_runtime::shell;
///
/// let output = shell::execute("echo hello")?;
/// assert!(output.contains("hello"));
/// ```
pub fn execute(command: &str) -> Result<String> {
    let timeout_secs = std::env::var("SCAFFOLD_SHELL_TIMEOUT_SECS")
        .ok()
        .and_then(|raw| raw.parse::<u64>().ok())
        .filter(|v| *v > 0);

    if let Some(secs) = timeout_secs {
        return execute_with_timeout(command, secs * 1000);
    }

    let output = Command::new("sh")
        .arg("-c")
        .arg(command)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()?;

    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        Err(Error::ActionFailed {
            action: "shell".to_string(),
            message: format!("command failed: {}", stderr),
        })
    }
}

/// Execute a shell command and return raw bytes
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

/// Execute a shell command with a timeout (in milliseconds).
///
/// Spawns the command as a child process and waits up to `timeout_ms`.
/// If the timeout expires, the child is killed before returning an error.
pub fn execute_with_timeout(command: &str, timeout_ms: u64) -> Result<String> {
    use std::time::{Duration, Instant};

    let mut child = Command::new("sh")
        .arg("-c")
        .arg(command)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    let deadline = Instant::now() + Duration::from_millis(timeout_ms);

    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                // Process exited
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
                    return Err(Error::ActionFailed {
                        action: "shell".to_string(),
                        message: format!("command failed: {}", stderr_str),
                    });
                }
            }
            Ok(None) => {
                // Still running — check deadline
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

/// Check if a command is available on the system
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
    fn test_command_exists() {
        assert!(command_exists("sh"));
        assert!(!command_exists("nonexistent_command_xyz"));
    }
}
