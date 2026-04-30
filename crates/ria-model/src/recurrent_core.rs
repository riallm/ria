//! Shared recurrent core for recurrent-depth RIA.

use candle_core::{Result, Tensor};
use ria_core::config::ModelConfig;

use crate::layer::TransformerLayer;

/// A single parameterized transformer block reused across recurrent iterations.
pub struct RecurrentBlock {
    pub layer: TransformerLayer,
    pub residual_scale: f64,
}

impl RecurrentBlock {
    pub fn new(
        vs: candle_nn::VarBuilder,
        config: &ModelConfig,
        residual_scale: f64,
    ) -> Result<Self> {
        Ok(Self {
            layer: TransformerLayer::new(vs, config)?,
            residual_scale,
        })
    }

    pub fn forward(&self, input: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let updated = self.layer.forward(input, mask)?;
        if (self.residual_scale - 1.0).abs() < f64::EPSILON {
            return Ok(updated);
        }

        let scale = Tensor::new(self.residual_scale as f32, input.device())?;
        let delta = updated.sub(input)?;
        input.add(&delta.broadcast_mul(&scale)?)
    }
}

/// Iterative refinement stage that repeatedly applies one shared block.
pub struct RecurrentCore {
    pub block: RecurrentBlock,
    pub default_iterations: usize,
    pub min_iterations: usize,
    pub max_iterations: usize,
}

impl RecurrentCore {
    pub fn new(vs: candle_nn::VarBuilder, config: &ModelConfig) -> Result<Self> {
        let recurrent = &config.recurrent;
        let block = RecurrentBlock::new(vs.pp("block"), config, recurrent.residual_scale)?;
        Ok(Self {
            block,
            default_iterations: recurrent.recurrent_iterations,
            min_iterations: recurrent.min_recurrent_iterations,
            max_iterations: recurrent.max_recurrent_iterations,
        })
    }

    pub fn resolve_iterations(&self, requested: Option<usize>) -> usize {
        requested
            .unwrap_or(self.default_iterations)
            .clamp(self.min_iterations, self.max_iterations)
    }

    pub fn forward(
        &self,
        input: &Tensor,
        mask: Option<&Tensor>,
        iterations: Option<usize>,
    ) -> Result<Tensor> {
        let iterations = self.resolve_iterations(iterations);
        let mut hidden = input.clone();
        for _ in 0..iterations {
            hidden = self.block.forward(&hidden, mask)?;
        }
        Ok(hidden)
    }
}
