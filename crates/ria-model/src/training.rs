//! Standard causal language-modeling objective.

use candle_core::{DType, Result, Tensor};

/// Compute next-token prediction loss:
///
/// `L = -sum_t log P(x[t + 1] | x[..=t])`
pub fn causal_lm_loss(logits: &Tensor, token_ids: &Tensor) -> Result<Tensor> {
    if logits.rank() != 3 {
        candle_core::bail!("causal_lm_loss expects logits shaped (batch, seq, vocab)");
    }
    if token_ids.rank() != 2 {
        candle_core::bail!("causal_lm_loss expects token_ids shaped (batch, seq)");
    }

    let dims = logits.dims();
    let batch = dims[0];
    let seq_len = dims[1];
    let vocab = dims[2];
    if seq_len < 2 {
        candle_core::bail!("causal_lm_loss requires at least two sequence positions");
    }
    if token_ids.dims() != [batch, seq_len] {
        candle_core::bail!("token_ids shape must match logits batch and sequence dimensions");
    }

    let prediction_count = batch * (seq_len - 1);
    let shifted_logits = logits
        .narrow(1, 0, seq_len - 1)?
        .reshape((prediction_count, vocab))?;
    let targets = token_ids
        .narrow(1, 1, seq_len - 1)?
        .flatten_all()?
        .to_dtype(DType::U32)?;

    candle_nn::loss::cross_entropy(&shifted_logits, &targets)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn computes_shifted_causal_loss() -> Result<()> {
        let logits = Tensor::from_vec(
            vec![
                0f32, 8., 0., 0., //
                0., 0., 8., 0., 8., 0., 0., 0.,
            ],
            (1, 3, 4),
            &Device::Cpu,
        )?;
        let tokens = Tensor::from_vec(vec![0u32, 1, 2], (1, 3), &Device::Cpu)?;
        let loss = causal_lm_loss(&logits, &tokens)?;
        assert!(loss.to_scalar::<f32>()? < 0.002);
        Ok(())
    }
}
