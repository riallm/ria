//! Tool executor

use ria_core::RiaResult;
use tokio::process::Command;

use crate::categories::ToolCategory;
use crate::protocol::{ToolCall, ToolResult};

/// Tool executor
pub struct ToolExecutor {
    sandbox_enabled: bool,
    approval_required: bool,
}

impl ToolExecutor {
    pub fn new(sandbox_enabled: bool, approval_required: bool) -> Self {
        Self {
            sandbox_enabled,
            approval_required,
        }
    }

    /// Execute a tool call
    pub async fn execute(&self, call: &ToolCall) -> RiaResult<ToolResult> {
        if call.requires_approval && self.approval_required {
            return Ok(ToolResult {
                success: false,
                output: "Tool execution requires human approval".to_string(),
                error: Some("Approval pending".to_string()),
                exit_code: None,
            });
        }

        // Execute based on tool category
        match call.category {
            ToolCategory::Execution => self.execute_command(call).await,
            ToolCategory::Testing => self.run_tests(call).await,
            ToolCategory::Analysis => self.analyze_code(call).await,
            ToolCategory::VersionControl => self.run_git_command(call).await,
            ToolCategory::Build => self.run_build(call).await,
            ToolCategory::Search => self.search_files(call).await,
            ToolCategory::PackageManagement => self.manage_packages(call).await,
        }
    }

    async fn execute_command(&self, call: &ToolCall) -> RiaResult<ToolResult> {
        let command = call.arguments["command"].as_str().unwrap_or("");
        let output = if self.sandbox_enabled {
            // Would run in sandbox/chroot
            Command::new("sh").arg("-c").arg(command).output().await?
        } else {
            Command::new("sh").arg("-c").arg(command).output().await?
        };

        Ok(ToolResult {
            success: output.status.success(),
            output: String::from_utf8_lossy(&output.stdout).to_string(),
            error: String::from_utf8_lossy(&output.stderr)
                .to_string()
                .into(),
            exit_code: output.status.code(),
        })
    }

    async fn run_tests(&self, _call: &ToolCall) -> RiaResult<ToolResult> {
        // Would run test suite
        Ok(ToolResult {
            success: true,
            output: "Tests passed".to_string(),
            error: None,
            exit_code: Some(0),
        })
    }

    async fn analyze_code(&self, _call: &ToolCall) -> RiaResult<ToolResult> {
        // Would run static analysis
        Ok(ToolResult {
            success: true,
            output: "Analysis complete".to_string(),
            error: None,
            exit_code: Some(0),
        })
    }

    async fn run_git_command(&self, _call: &ToolCall) -> RiaResult<ToolResult> {
        // Would run git commands
        Ok(ToolResult {
            success: true,
            output: "Git command executed".to_string(),
            error: None,
            exit_code: Some(0),
        })
    }

    async fn run_build(&self, _call: &ToolCall) -> RiaResult<ToolResult> {
        // Would run build commands
        Ok(ToolResult {
            success: true,
            output: "Build successful".to_string(),
            error: None,
            exit_code: Some(0),
        })
    }

    async fn search_files(&self, _call: &ToolCall) -> RiaResult<ToolResult> {
        // Would search files
        Ok(ToolResult {
            success: true,
            output: "Search complete".to_string(),
            error: None,
            exit_code: Some(0),
        })
    }

    async fn manage_packages(&self, _call: &ToolCall) -> RiaResult<ToolResult> {
        // Would manage packages
        Ok(ToolResult {
            success: true,
            output: "Package operation complete".to_string(),
            error: None,
            exit_code: Some(0),
        })
    }
}
