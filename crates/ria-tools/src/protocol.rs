//! Tool call protocol

use serde::{Deserialize, Serialize};

use crate::categories::ToolCategory;

/// Tool call representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub category: ToolCategory,
    pub tool_name: String,
    pub arguments: serde_json::Value,
    pub requires_approval: bool,
}

/// Tool execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub success: bool,
    pub output: String,
    pub error: Option<String>,
    pub exit_code: Option<i32>,
}

/// Tool protocol handler
pub struct ToolProtocol;

impl ToolProtocol {
    /// Format a tool call for model output
    pub fn format_call(call: &ToolCall) -> String {
        format!(
            "<|tool_call|>{}<|tool_result|>",
            serde_json::to_string(call).unwrap_or_default()
        )
    }

    /// Parse a tool call from model output
    pub fn parse_call(text: &str) -> Option<ToolCall> {
        // Parse between tool_call and tool_result markers
        text.find("<|tool_call|>")
            .and_then(|start| text.find("<|tool_result|>").map(|end| (start, end)))
            .and_then(|(start, end)| {
                let json_str = &text[start + 13..end];
                serde_json::from_str(json_str).ok()
            })
    }
}
