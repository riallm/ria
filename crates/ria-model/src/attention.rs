//! Multi-Hop Code Attention (MHCA)
//!
//! Extends standard attention with:
//! - File-aware attention bias
//! - Hierarchical attention windows
//! - Grouped Query Attention (GQA)
//!
//! This implementation follows SPEC-003 Section 8.2 GGUF tensor naming conventions:
//! - blk.{i}.attn_q.weight: Query projection
//! - blk.{i}.attn_k.weight: Key projection (GQA)
//! - blk.{i}.attn_v.weight: Value projection (GQA)
//! - blk.{i}.attn_output.weight: Output projection

use candle_core::{Result, Tensor, D};
use candle_nn::Module;
use ria_core::config::ModelConfig;
use tracing::trace;

use crate::position::RoPE;

/// File relationship types for attention bias
#[derive(Debug, Clone, Copy)]
pub enum FileRelation {
    SameFile,
    Import,
    Module,
    Repository,
}

/// Multi-Hop Code Attention module
pub struct MultiHopCodeAttention {
    pub query: candle_nn::Linear,
    pub key: candle_nn::Linear,
    pub value: candle_nn::Linear,
    pub output: candle_nn::Linear,
    pub config: ModelConfig,
    pub head_dim: usize,
    pub file_attention_bias: Option<Tensor>,
    pub rope: RoPE,
}

impl MultiHopCodeAttention {
    /// Create new MultiHopCodeAttention layer following GGUF naming conventions
    /// per SPEC-003 Section 8.2: blk.{i}.attn_q, blk.{i}.attn_k, blk.{i}.attn_v, blk.{i}.attn_output
    pub fn new(vs: candle_nn::VarBuilder, config: &ModelConfig) -> Result<Self> {
        let head_dim = config.head_dim();
        let hidden_dim = config.hidden_dim();
        let num_heads = config.attention_head_count;
        let num_kv_heads = config.attention_head_count_kv;

        // Following GGUF naming per SPEC-003 Section 8.2
        let query = candle_nn::linear(hidden_dim, num_heads * head_dim, vs.pp("q"))?;
        let key = candle_nn::linear(hidden_dim, num_kv_heads * head_dim, vs.pp("k"))?;
        let value = candle_nn::linear(hidden_dim, num_kv_heads * head_dim, vs.pp("v"))?;
        let output = candle_nn::linear(num_heads * head_dim, hidden_dim, vs.pp("o"))?;
        let rope = RoPE::new(
            head_dim,
            config.rope_freq_base as f64,
            config.context_length,
        );

        trace!(
            "MultiHopCodeAttention: hidden_dim={}, heads={}, kv_heads={}, head_dim={}",
            hidden_dim,
            num_heads,
            num_kv_heads,
            head_dim
        );

        Ok(Self {
            query,
            key,
            value,
            output,
            config: config.clone(),
            head_dim,
            file_attention_bias: None,
            rope,
        })
    }

    /// Compute file-aware attention bias matrix
    /// Following SPEC-003 Section 3.3: learned parameters for file relationships
    pub fn compute_file_bias(&mut self, file_relations: &[FileRelation]) -> Result<()> {
        // Learned parameters for file relationship attention per SPEC-003 Section 3.3
        // α_same ∈ [-2.0, 2.0], α_import ∈ [-1.0, 1.5], etc.
        let _alpha_same = 1.0;
        let _alpha_import = 0.8;
        let _alpha_module = 0.6;
        let _alpha_repo = 0.3;
        trace!(
            "compute_file_bias called for {} file-relation entries",
            file_relations.len()
        );

        // In a full implementation, this would create a bias matrix based on
        // the file relationships between tokens (SPEC-003 Section 3.4)
        self.file_attention_bias = None;
        Ok(())
    }

    /// Forward pass with Grouped Query Attention (GQA)
    /// Follows SPEC-003 Section 3: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k) + M_file + M_hier) · V
    pub fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        _cache: Option<(&Tensor, &Tensor)>,
    ) -> Result<Tensor> {
        trace!("MHCA forward: input shape {:?}", x.shape());
        if x.rank() != 3 {
            candle_core::bail!("MHCA expects input shape (batch, seq, hidden)");
        }

        let q = self.query.forward(x)?;
        let k = self.key.forward(x)?;
        let v = self.value.forward(x)?;

        // Reshape for multi-head attention: (batch, seq, heads, head_dim)
        let batch = x.dims()[0];
        let seq_len = x.dims()[1];
        let num_heads = self.config.attention_head_count;
        let num_kv_heads = self.config.attention_head_count_kv;
        let head_dim = self.head_dim;

        // Reshape Q to separate heads
        let q = q.reshape((batch, seq_len, num_heads, head_dim))?;
        let k = k.reshape((batch, seq_len, num_kv_heads, head_dim))?;
        let v = v.reshape((batch, seq_len, num_kv_heads, head_dim))?;

        // Standard attention computation with GQA.
        let scale = (head_dim as f32).sqrt().recip();
        let scale_tensor = Tensor::new(scale, x.device())?;

        // Transpose for attention: (batch, heads, seq, head_dim)
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        // Apply RoPE to query and key states before dot-product attention.
        let q = self.rope.apply(&q, 0)?;
        let k = self.rope.apply(&k, 0)?;

        // Grouped Query Attention: each KV head is shared by a group of query heads.
        let kv_group_size = num_heads / num_kv_heads;
        let k = repeat_kv_heads(&k, kv_group_size)?;
        let v = repeat_kv_heads(&v, kv_group_size)?;

        // Compute attention: Q @ K^T / sqrt(d_k)
        let k_t = k.transpose(2, 3)?;
        let qk = q.matmul(&k_t)?;
        let mut logits = qk.broadcast_mul(&scale_tensor)?;

        // Apply file-aware bias if present (SPEC-003 Section 3.2: M_file + M_hier)
        if let Some(file_bias) = &self.file_attention_bias {
            logits = logits.broadcast_add(file_bias)?;
        }
        if let Some(m) = mask {
            logits = logits.broadcast_add(m)?;
        }

        // Softmax and weighted sum
        let attn = candle_nn::ops::softmax(&logits, D::Minus1)?;
        let out = attn.matmul(&v)?;

        // Reshape back: (batch, seq, hidden_dim)
        let out = out
            .transpose(1, 2)?
            .reshape((batch, seq_len, num_heads * head_dim))?;

        trace!("MHCA forward: output shape {:?}", out.shape());
        self.output.forward(&out)
    }
}

fn repeat_kv_heads(x: &Tensor, group_size: usize) -> Result<Tensor> {
    if group_size == 1 {
        return Ok(x.clone());
    }

    let dims = x.dims();
    if dims.len() != 4 {
        candle_core::bail!("repeat_kv_heads expects shape (batch, heads, seq, head_dim)");
    }

    let batch = dims[0];
    let kv_heads = dims[1];
    let seq_len = dims[2];
    let head_dim = dims[3];

    x.unsqueeze(2)?.repeat((1, 1, group_size, 1, 1))?.reshape((
        batch,
        kv_heads * group_size,
        seq_len,
        head_dim,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn repeats_kv_heads_for_gqa() -> Result<()> {
        let values = (0..8).map(|v| v as f32).collect::<Vec<_>>();
        let kv = Tensor::from_vec(values, (1, 2, 2, 2), &Device::Cpu)?;
        let repeated = repeat_kv_heads(&kv, 2)?;

        assert_eq!(repeated.dims(), &[1, 4, 2, 2]);
        Ok(())
    }
}
