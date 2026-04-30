//! Language modeling head.
//!
//! RIA ties the output projection to the input token embedding matrix per
//! SPEC-003 Section 6.

use candle_core::{Result, Tensor};

/// Tied output projection for next-token prediction.
pub struct LmHead;

impl LmHead {
    pub fn new() -> Self {
        Self
    }

    /// Project hidden states with the transpose of the token embedding matrix.
    pub fn forward(&self, hidden: &Tensor, embedding_weight: &Tensor) -> Result<Tensor> {
        let weight = match *hidden.dims() {
            [batch, _, _] => embedding_weight.broadcast_left(batch)?.t()?,
            [batch, seq, _, _] => embedding_weight.broadcast_left((batch, seq))?.t()?,
            _ => embedding_weight.t()?,
        };
        hidden.matmul(&weight)
    }
}

impl Default for LmHead {
    fn default() -> Self {
        Self::new()
    }
}
