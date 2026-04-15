//! API request and response models

use serde::{Deserialize, Serialize};

/// OpenAI-compatible message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

/// Chat completion request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub stop: Option<Vec<String>>,
}

/// Chat completion response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

/// Single choice in a response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Choice {
    pub index: usize,
    pub message: Message,
    pub finish_reason: String,
}

/// Token usage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// Code completion request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeCompletionRequest {
    pub file_path: String,
    pub content: String,
    pub cursor_position: usize,
    pub language: String,
}

/// Code completion response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeCompletionResponse {
    pub completions: Vec<CodeCompletion>,
}

/// Single code completion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeCompletion {
    pub text: String,
    pub score: f64,
}

/// Debug request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugRequest {
    pub code: String,
    pub error: String,
    pub context: Option<String>,
}

/// Debug response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugResponse {
    pub diagnosis: String,
    pub fix: String,
    pub explanation: String,
}

/// Agentic execute request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentExecuteRequest {
    pub task: String,
    pub plan: Option<String>,
    pub context: Option<serde_json::Value>,
}

/// Agentic execute response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentExecuteResponse {
    pub plan: String,
    pub steps: Vec<AgentStep>,
    pub artifacts: Vec<AgentArtifact>,
}

/// Single agent step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStep {
    pub action: String,
    pub output: String,
    pub status: String,
}

/// Agent artifact
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentArtifact {
    pub name: String,
    pub content: String,
    pub artifact_type: String,
}
