//! RoPE (Rotary Position Embedding)
//!
//! Per SPEC-003 Section 2.1: "Position Encoding: RoPE (θ=10,000)"
//! RoPE applies rotary transformations to the hidden states to encode position information.

use candle_core::{Result, Tensor, D};
use tracing::trace;

/// Rotary Position Embedding per SPEC-003
pub struct RoPE {
    pub dim: usize,
    pub theta: f64,
    pub max_seq_len: usize,
    /// Cache for frequency computation
    pub freqs_cache: Option<Tensor>,
}

impl RoPE {
    /// Create RoPE with frequency base θ = 10,000 per SPEC-003
    pub fn new(dim: usize, theta: f64, max_seq_len: usize) -> Self {
        trace!(
            "RoPE: dim={}, theta={}, max_seq_len={}",
            dim,
            theta,
            max_seq_len
        );
        Self {
            dim,
            theta,
            max_seq_len,
            freqs_cache: None,
        }
    }

    /// Precompute cosine and sine frequency tables for RoPE.
    fn compute_cos_sin(
        &self,
        seq_len: usize,
        seq_offset: usize,
        device: &candle_core::Device,
    ) -> Result<(Tensor, Tensor)> {
        let half_dim = self.dim / 2;
        let mut cos = Vec::with_capacity(half_dim * seq_len);
        let mut sin = Vec::with_capacity(half_dim * seq_len);

        for pos in seq_offset..(seq_offset + seq_len) {
            for i in 0..half_dim {
                let exponent = -((2 * i) as f64) / self.dim as f64;
                let angle = (pos as f64) * self.theta.powf(exponent);
                cos.push(angle.cos() as f32);
                sin.push(angle.sin() as f32);
            }
        }

        let cos = Tensor::from_vec(cos, (seq_len, half_dim), device)?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        let sin = Tensor::from_vec(sin, (seq_len, half_dim), device)?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        Ok((cos, sin))
    }

    /// Apply RoPE to a tensor
    /// Expected input shape is `(batch, heads, seq, head_dim)`.
    pub fn apply(&self, x: &Tensor, seq_offset: usize) -> Result<Tensor> {
        if x.rank() != 4 {
            candle_core::bail!("RoPE expects input shape (batch, heads, seq, head_dim)");
        }

        let dims = x.dims();
        let batch = dims[0];
        let heads = dims[1];
        let seq_len = dims[2];
        let head_dim = dims[3];
        if head_dim != self.dim {
            candle_core::bail!(
                "RoPE head dim mismatch: input has {}, configured for {}",
                head_dim,
                self.dim
            );
        }
        if head_dim % 2 != 0 {
            candle_core::bail!("RoPE requires an even head dimension");
        }

        trace!("RoPE apply: seq_len={}, offset={}", seq_len, seq_offset);

        let half_dim = head_dim / 2;
        let (cos, sin) = self.compute_cos_sin(seq_len, seq_offset, x.device())?;
        let pairs = x.reshape((batch, heads, seq_len, half_dim, 2))?;
        let even = pairs.narrow(D::Minus1, 0, 1)?.squeeze(D::Minus1)?;
        let odd = pairs.narrow(D::Minus1, 1, 1)?.squeeze(D::Minus1)?;

        let rotated_even = (even.broadcast_mul(&cos)? - odd.broadcast_mul(&sin)?)?;
        let rotated_odd = (even.broadcast_mul(&sin)? + odd.broadcast_mul(&cos)?)?;

        Tensor::stack(&[rotated_even, rotated_odd], D::Minus1)?
            .reshape((batch, heads, seq_len, head_dim))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn rope_preserves_shape() -> Result<()> {
        let rope = RoPE::new(4, 10_000.0, 16);
        let x = Tensor::from_vec(vec![1f32; 2 * 3 * 5 * 4], (2, 3, 5, 4), &Device::Cpu)?;
        let y = rope.apply(&x, 0)?;
        assert_eq!(y.dims(), &[2, 3, 5, 4]);
        Ok(())
    }
}
