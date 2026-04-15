//! Token and position embeddings
//!
//! Token embedding follows SPEC-003 Section 2.1: "Token Embedding"
//! Position encoding uses RoPE per SPEC-003: "Position Encoding: RoPE (θ=10,000)"
//!
//! GGUF naming per SPEC-003 Section 8.2:
//! - token_embd.weight: Token embedding matrix (tied with lm_head)
//! - output.weight: LM head (tied)

use candle_core::{Result, Tensor};
use candle_nn::Module;
use ria_core::config::ModelConfig;
use tracing::trace;

use crate::position::RoPE;

/// Combined token and position embedding module
pub struct Embedding {
    pub token_embedding: candle_nn::Embedding,
    pub rope: RoPE,
    pub config: ModelConfig,
}

impl Embedding {
    pub fn new(vs: candle_nn::VarBuilder, config: &ModelConfig) -> Result<Self> {
        let vocab_size = config.tier.vocab_size();
        let hidden_dim = config.tier.hidden_dim();

        let token_embedding = candle_nn::embedding(vocab_size, hidden_dim, vs.pp("token_embd"))?;
        let rope = RoPE::new(
            config.tier.head_dim(),
            config.rope_freq_base as f64,
            config.context_length,
        );

        Ok(Self {
            token_embedding,
            rope,
            config: config.clone(),
        })
    }

    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let embeddings = self.token_embedding.forward(input_ids)?;
        // RoPE is applied per-layer in the attention module
        Ok(embeddings)
    }
}

impl Module for Embedding {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward(xs)
    }
}
