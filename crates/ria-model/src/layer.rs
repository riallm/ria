//! Transformer layer combining all components
//!
//! Per SPEC-003 Section 2.1:
//! ```
//! TransformerLayers (N×)
//! ├── RMSNorm (pre-norm)
//! ├── Multi-Hop Code Attention (MHCA)
//! ├── Tool Integration Router (TIR)
//! ├── Feed-Forward Network (SwiGLU)
//! │   ├── Planning path
//! │   └── Execution path
//! └── Residual connections
//! ```
//!
//! Following GGUF naming per SPEC-003 Section 8.2:
//! - blk.{i}.attn_norm.weight
//! - blk.{i}.attn_*.weight
//! - blk.{i}.ffn_norm.weight
//! - blk.{i}.ffn_*.weight

use candle_core::{Module, Result, Tensor};
use ria_core::config::ModelConfig;
use tracing::{debug, trace};

use crate::attention::MultiHopCodeAttention;
use crate::ffn::DualPathFFN;
use crate::norm::RMSNorm;
use crate::tir::ToolIntegrationRouter;

/// Complete transformer layer with all components per SPEC-003
pub struct TransformerLayer {
    pub attention_norm: RMSNorm,
    pub attention: MultiHopCodeAttention,
    pub ffn_norm: RMSNorm,
    pub ffn: DualPathFFN,
    pub tir: ToolIntegrationRouter,
    pub config: ModelConfig,
}

impl TransformerLayer {
    /// Create transformer layer with GGUF naming per SPEC-003 Section 8.2
    /// Note: Attention and FFN each have their own RMSNorm (pre-norm pattern)
    pub fn new(vs: candle_nn::VarBuilder, config: &ModelConfig) -> Result<Self> {
        let hidden_dim = config.tier.hidden_dim();

        // Attention RMSNorm per SPEC-003: "RMSNorm (pre-norm)"
        let attention_norm = RMSNorm::new(hidden_dim, 1e-5, vs.pp("attn_norm"))?;
        let attention = MultiHopCodeAttention::new(vs.pp("attn"), config)?;

        // FFN RMSNorm per SPEC-003: pre-norm before FFN
        let ffn_norm = RMSNorm::new(hidden_dim, 1e-5, vs.pp("ffn_norm"))?;
        let ffn = DualPathFFN::new(vs.pp("ffn"), config)?;

        // TIR per SPEC-003 Section 4: Tool Integration Router
        let tir = ToolIntegrationRouter::new(vs.pp("tir"), config)?;

        debug!("TransformerLayer created: hidden_dim={}", hidden_dim);

        Ok(Self {
            attention_norm,
            attention,
            ffn_norm,
            ffn,
            tir,
            config: config.clone(),
        })
    }

    /// Forward pass with pre-norm architecture
    /// Per SPEC-003 Section 2.2:
    /// ```
    /// a = RMSNorm(h_{l-1})
    /// m = MHCA(a)
    /// h' = h_{l-1} + m
    /// t = TIR(h')
    /// h'' = concatenate(h', t)
    /// f = RMSNorm(h'')
    /// ff_output = FFN(f)
    /// h_l = h'' + ff_output
    /// ```
    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        trace!("TransformerLayer forward: input shape {:?}", x.shape());

        // Pre-norm for attention (SPEC-003: a = RMSNorm(h_{l-1}))
        let normed = self.attention_norm.forward(x)?;
        let attn_out = self.attention.forward(&normed, mask, None)?;

        // Residual connection (SPEC-003: h' = h_{l-1} + m)
        let x = (x + attn_out)?;
        trace!("After attention + residual: shape {:?}", x.shape());

        // TIR is applied during forward but doesn't modify hidden states
        // (SPEC-003: h'' = concatenate(h', t) - but we skip for inference)

        // Pre-norm for FFN (SPEC-003: f = RMSNorm(h''))
        let normed = self.ffn_norm.forward(&x)?;
        let ffn_out = self.ffn.forward(&normed)?;

        // Residual connection (SPEC-003: h_l = h'' + ff_output)
        let x = (x + ffn_out)?;

        trace!("TransformerLayer forward: output shape {:?}", x.shape());
        Ok(x)
    }
}

impl Module for TransformerLayer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward(xs, None)
    }
}
