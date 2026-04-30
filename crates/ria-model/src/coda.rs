//! Coda stage for recurrent-depth RIA.

use candle_core::{Result, Tensor};
use ria_core::config::ModelConfig;

use crate::layer::TransformerLayer;

/// Distinct final integration blocks after recurrent refinement.
pub struct Coda {
    pub layers: Vec<TransformerLayer>,
}

impl Coda {
    pub fn new(
        vs: candle_nn::VarBuilder,
        config: &ModelConfig,
        layer_count: usize,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(layer_count);
        for i in 0..layer_count {
            layers.push(TransformerLayer::new(
                vs.pp(format!("layer_{}", i)),
                config,
            )?);
        }
        Ok(Self { layers })
    }

    pub fn forward(&self, input: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let mut hidden = input.clone();
        for layer in &self.layers {
            hidden = layer.forward(&hidden, mask)?;
        }
        Ok(hidden)
    }
}
