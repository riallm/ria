//! Complete RIA model
//!
//! RIA transformer model per SPEC-003 Section 2.1:
//! ```
//! RIA Model Architecture
//! ├── Token Embedding
//! ├── Position Embedding (RoPE)
//! ├── Transformer Layers (N×)
//! │   ├── RMSNorm (pre-norm)
//! │   ├── Multi-Hop Code Attention (MHCA)
//! │   ├── Tool Integration Router (TIR)
//! │   ├── Feed-Forward Network (SwiGLU)
//! │   │   ├── Planning path
//! │   │   └── Execution path
//! │   └── Residual connections
//! ├── Final RMSNorm
//! └── LM Head (tied with embedding)
//! ```
//!
//! GGUF naming follows SPEC-003 Section 8.2:
//! - token_embd.weight: Embedding
//! - blk.{i}.*: Layer weights
//! - output_norm.weight: Final RMSNorm
//! - output.weight: LM head (tied with embedding)

use candle_core::{Module, Result, Tensor};
use ria_core::config::ModelConfig;
use tracing::{info, trace};

use crate::embedding::Embedding;
use crate::layer::TransformerLayer;
use crate::norm::RMSNorm;

/// Complete RIA transformer model
pub struct RiaModel {
    pub embedding: Embedding,
    pub layers: Vec<TransformerLayer>,
    pub final_norm: RMSNorm,
    pub lm_head: candle_nn::Linear,
    pub config: ModelConfig,
}

impl RiaModel {
    /// Create new RIA model following GGUF naming per SPEC-003 Section 8.2
    pub fn new(vs: candle_nn::VarBuilder, config: &ModelConfig) -> Result<Self> {
        let num_layers = config.tier.num_layers();
        let hidden_dim = config.tier.hidden_dim();
        let vocab_size = config.tier.vocab_size();

        let embedding = Embedding::new(vs.pp("token_embd"), config)?;

        // Create transformer layers following SPEC-003 Section 8.2 naming: blk.{i}.*
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let layer = TransformerLayer::new(vs.pp(&format!("blk.{}", i)), config)?;
            layers.push(layer);
        }

        // Final RMSNorm per SPEC-003 Section 8.2
        let final_norm = RMSNorm::new(hidden_dim, 1e-5, vs.pp("output_norm"))?;

        // LM head tied with embedding (SPEC-003: "Embedding: Tied input/output")
        let lm_head = candle_nn::linear(hidden_dim, vocab_size, vs.pp("output"))?;

        info!(
            "RiaModel created: layers={}, hidden_dim={}, vocab_size={}",
            num_layers, hidden_dim, vocab_size
        );

        Ok(Self {
            embedding,
            layers,
            final_norm,
            lm_head,
            config: config.clone(),
        })
    }

    /// Forward pass through the model
    /// Per SPEC-003 Section 2.2:
    /// ```
    /// h_0 = embed(tokens) + pos_embed(positions)
    /// For l = 1 to N:
    ///     h_l = transformer_layer(h_{l-1})
    /// output = final_norm(h_N) · W_embed^T
    /// ```
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        trace!("RiaModel forward: input shape {:?}", input_ids.shape());

        // Token embedding (SPEC-003: h_0 = embed(tokens))
        let mut h = self.embedding.forward(input_ids)?;
        trace!("After embedding: shape {:?}", h.shape());

        // Transformer layers (SPEC-003: For l = 1 to N)
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, None)?;
            trace!("After layer {}: shape {:?}", i, h.shape());
        }

        // Final RMSNorm (SPEC-003: output = RMSNorm(h_N) · W_embed^T)
        let h = self.final_norm.forward(&h)?;
        trace!("After final_norm: shape {:?}", h.shape());

        // LM head (tied with embedding)
        let logits = self.lm_head.forward(&h)?;
        trace!("RiaModel forward: output shape {:?}", logits.shape());

        Ok(logits)
    }
}

impl Module for RiaModel {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward(xs)
    }
}
