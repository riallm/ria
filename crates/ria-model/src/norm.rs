//! RMSNorm implementation
//!
//! Root Mean Square Layer Normalization per SPEC-003 Section 2.1
//! Uses ε = 1e-5 (SPEC-003: "RMSNorm Epsilon: 1e-5")
//!
//! GGUF naming per SPEC-003 Section 8.2:
//! - blk.{i}.attn_norm.weight: Attention RMSNorm
//! - blk.{i}.ffn_norm.weight: FFN RMSNorm
//! - output_norm.weight: Final RMSNorm

use candle_core::{Result, Tensor};
use candle_nn::Module;
use tracing::trace;

/// Root Mean Square Layer Normalization
pub struct RMSNorm {
    pub weight: Tensor,
    pub epsilon: f64,
}

impl RMSNorm {
    /// Create RMSNorm with hidden_dim and epsilon per SPEC-003 (ε = 1e-5)
    pub fn new(hidden_dim: usize, epsilon: f64, vs: candle_nn::VarBuilder) -> Result<Self> {
        let weight = vs.get(hidden_dim, "weight")?;
        trace!("RMSNorm: hidden_dim={}, epsilon={}", hidden_dim, epsilon);
        Ok(Self { weight, epsilon })
    }

    /// Forward pass: RMSNorm(x) = x / sqrt(mean(x²) + ε) * weight
    /// Per SPEC-003: "RMSNorm (pre-norm)" with ε=1e-5
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        trace!("RMSNorm forward: input shape {:?}", x.shape());

        let dtype = x.dtype();
        let x_f64 = x.to_dtype(candle_core::DType::F64)?;

        // Compute RMS: sqrt(mean(x²) + ε)
        let variance = x_f64.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let epsilon_tensor = Tensor::new(self.epsilon, x.device())?;
        let inv_std = variance.add(&epsilon_tensor)?.sqrt()?.recip()?;

        // Normalize
        let x_normalized = x_f64.broadcast_mul(&inv_std)?;
        let x_normalized = x_normalized.to_dtype(dtype)?;

        // Apply weight (learned per-layer scaling)
        let out = x_normalized.broadcast_mul(&self.weight)?;

        trace!("RMSNorm forward: output shape {:?}", out.shape());
        Ok(out)
    }
}

impl Module for RMSNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward(xs)
    }
}
