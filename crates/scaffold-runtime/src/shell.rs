//! Shell command execution utilities

use crate::error::{Error, Result};
use std::process::{Command, Stdio};

/// Execute a shell command and return its output
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

/// Execute a shell command with a timeout
pub fn execute_with_timeout(command: &str, timeout_ms: u64) -> Result<String> {
    use std::thread;
    use std::time::Duration;

    let command = command.to_string();
    let handle = thread::spawn(move || execute(&command));

    // Simple timeout implementation
    let timeout = Duration::from_millis(timeout_ms);
    let start = std::time::Instant::now();

    loop {
        if handle.is_finished() {
            return handle
                .join()
                .map_err(|_| Error::Runtime("thread panicked".to_string()))?;
        }
        if start.elapsed() > timeout {
            return Err(Error::Timeout("shell command".to_string()));
        }
        thread::sleep(Duration::from_millis(10));
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
