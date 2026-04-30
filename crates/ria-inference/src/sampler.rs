//! Sampling strategies

use candle_core::{Result, Tensor};
use rand::distributions::Distribution;
use ria_core::generation::GenerationConfig;

/// Sampling strategy
#[derive(Debug, Clone, Copy)]
pub enum SamplingStrategy {
    /// Greedy decoding (argmax)
    Greedy,
    /// Multinomial sampling
    Multinomial,
    /// Top-k sampling
    TopK,
    /// Top-p (nucleus) sampling
    TopP,
    /// Temperature scaling
    Temperature,
}

/// Token sampler
pub struct Sampler {
    pub strategy: SamplingStrategy,
    pub temperature: f64,
    pub top_k: usize,
    pub top_p: f64,
    pub repeat_penalty: f32,
}

impl Sampler {
    pub fn new(config: &GenerationConfig) -> Self {
        let strategy = if config.temperature == 0.0 {
            SamplingStrategy::Greedy
        } else if config.top_k > 0 && config.top_p < 1.0 {
            SamplingStrategy::TopK
        } else if config.top_p < 1.0 {
            SamplingStrategy::TopP
        } else {
            SamplingStrategy::Temperature
        };

        Self {
            strategy,
            temperature: config.temperature,
            top_k: config.top_k,
            top_p: config.top_p,
            repeat_penalty: config.repeat_penalty,
        }
    }

    /// Sample a token from logits
    pub fn sample(&self, logits: &Tensor) -> Result<u32> {
        // Apply temperature scaling
        let scaled = if self.temperature > 0.0 {
            (logits / self.temperature)?
        } else {
            logits.clone()
        };

        // Apply softmax to get probabilities
        let probs = candle_nn::ops::softmax(&scaled, candle_core::D::Minus1)?;

        match self.strategy {
            SamplingStrategy::Greedy => {
                let token = probs.argmax(candle_core::D::Minus1)?;
                if token.rank() == 0 {
                    Ok(token.to_scalar::<u32>()?)
                } else {
                    Ok(token.get(0)?.to_scalar::<u32>()?)
                }
            }
            SamplingStrategy::Multinomial
            | SamplingStrategy::TopK
            | SamplingStrategy::TopP
            | SamplingStrategy::Temperature => {
                // Multinomial sampling from probabilities
                let probs_vec: Vec<f64> = probs.to_vec1()?;
                let dist = rand::distributions::WeightedIndex::new(&probs_vec).unwrap();
                let mut rng = rand::thread_rng();
                Ok(dist.sample(&mut rng) as u32)
            }
        }
    }

    /// Apply repetition penalty to logits
    pub fn apply_penalty(&self, logits: &mut Vec<f64>, token_ids: &[u32]) {
        for &token_id in token_ids {
            let idx = token_id as usize;
            if idx < logits.len() {
                if logits[idx] > 0.0 {
                    logits[idx] /= self.repeat_penalty as f64;
                } else {
                    logits[idx] *= self.repeat_penalty as f64;
                }
            }
        }
    }
}
