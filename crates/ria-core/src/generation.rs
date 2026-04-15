//! Generation configuration and parameters

use serde::{Deserialize, Serialize};

/// Stop sequence marker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StopSequence(pub String);

/// Generation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Maximum number of new tokens to generate
    pub max_new_tokens: usize,
    /// Sampling temperature (0.0 = greedy)
    pub temperature: f64,
    /// Top-p (nucleus) sampling
    pub top_p: f64,
    /// Top-k sampling
    pub top_k: usize,
    /// Repetition penalty
    pub repeat_penalty: f32,
    /// Presence penalty
    pub presence_penalty: f32,
    /// Frequency penalty
    pub frequency_penalty: f32,
    /// Stop sequences
    pub stop_sequences: Vec<StopSequence>,
    /// Whether to return log probabilities
    pub logprobs: bool,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 1024,
            temperature: 0.7,
            top_p: 0.95,
            top_k: 50,
            repeat_penalty: 1.1,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            stop_sequences: vec![StopSequence("<|eos|>".to_string())],
            logprobs: false,
            seed: None,
        }
    }
}
