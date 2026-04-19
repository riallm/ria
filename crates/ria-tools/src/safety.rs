//! Safety policy for tool execution

use serde::{Deserialize, Serialize};

/// Safety policy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyPolicy {
    /// Enable sandboxed execution
    pub sandbox_enabled: bool,
    /// Require human approval for certain operations
    pub approval_required: bool,
    /// Block network access
    pub network_blocked: bool,
    /// Enable seccomp filtering
    pub seccomp_enabled: bool,
    /// Use chroot jail
    pub chroot_enabled: bool,
    /// Maximum execution time in seconds
    pub max_execution_time: u64,
    /// Maximum output size in bytes
    pub max_output_size: u64,
}

impl Default for SafetyPolicy {
    fn default() -> Self {
        Self {
            sandbox_enabled: true,
            approval_required: true,
            network_blocked: true,
            seccomp_enabled: true,
            chroot_enabled: true,
            max_execution_time: 300,
            max_output_size: 10 * 1024 * 1024, // 10MB
        }
    }
}

impl SafetyPolicy {
    /// Check if an operation requires approval
    pub fn requires_approval(&self, command: &str) -> bool {
        // Check for dangerous operations
        let dangerous_patterns = [
            "rm -rf",
            "sudo",
            "chmod 777",
            "curl |",
            "wget |",
            "mkfs",
            "dd if=",
            "> /dev/",
            "> /etc/",
        ];

        dangerous_patterns
            .iter()
            .any(|pattern| command.contains(pattern))
    }
}
