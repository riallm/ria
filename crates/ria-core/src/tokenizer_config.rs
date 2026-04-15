//! Tokenizer configuration

use serde::{Deserialize, Serialize};

/// Tokenizer configuration for the hybrid BPE + custom token approach
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerConfig {
    /// Total vocabulary size
    pub vocab_size: usize,
    /// Number of BPE subword tokens
    pub bpe_tokens: usize,
    /// Number of identifier tokens
    pub identifier_tokens: usize,
    /// Number of syntax tokens
    pub syntax_tokens: usize,
    /// Number of tool tokens
    pub tool_tokens: usize,
    /// Number of agentic tokens
    pub agentic_tokens: usize,
    /// Special token IDs
    pub special_tokens: SpecialTokens,
}

/// Special token IDs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecialTokens {
    pub pad: usize,
    pub eos: usize,
    pub bos: usize,
    pub unk: usize,
    pub think: usize,
    pub plan_start: usize,
    pub code_start: usize,
    pub test_start: usize,
    pub debug_start: usize,
    pub tool_call: usize,
    pub tool_result: usize,
    pub verify: usize,
    pub file_marker: usize,
    pub error_marker: usize,
    pub success: usize,
    pub failure: usize,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            vocab_size: 106_000,
            bpe_tokens: 98_000,
            identifier_tokens: 5_000,
            syntax_tokens: 2_000,
            tool_tokens: 500,
            agentic_tokens: 500,
            special_tokens: SpecialTokens::default(),
        }
    }
}

impl Default for SpecialTokens {
    fn default() -> Self {
        Self {
            pad: 0,
            eos: 1,
            bos: 2,
            unk: 3,
            think: 4,
            plan_start: 5,
            code_start: 6,
            test_start: 7,
            debug_start: 8,
            tool_call: 9,
            tool_result: 10,
            verify: 11,
            file_marker: 12,
            error_marker: 13,
            success: 14,
            failure: 15,
        }
    }
}
