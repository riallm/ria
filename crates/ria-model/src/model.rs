//! Complete RIA model
//!
//! RIA transformer model per SPEC-003 Section 2.1:
//! ```text
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

use candle_core::{Device, Module, Result, Tensor};
use ria_core::config::{ModelArchitecture, ModelConfig};
use tracing::{info, trace};

use crate::coda::Coda;
use crate::embedding::Embedding;
use crate::layer::TransformerLayer;
use crate::lm_head::LmHead;
use crate::norm::RMSNorm;
use crate::prelude::Prelude;
use crate::recurrent_core::RecurrentCore;

/// Optional forward-pass controls.
#[derive(Debug, Clone, Copy)]
pub struct ForwardOptions {
    /// Override the recurrent loop count for recurrent-depth models.
    pub recurrent_iterations: Option<usize>,
    /// Apply a standard causal attention mask.
    pub use_causal_mask: bool,
}

impl Default for ForwardOptions {
    fn default() -> Self {
        Self {
            recurrent_iterations: None,
            use_causal_mask: true,
        }
    }
}

/// Prelude -> shared recurrent core -> Coda body.
pub struct RecurrentBody {
    pub prelude: Prelude,
    pub recurrent_core: RecurrentCore,
    pub coda: Coda,
}

/// Complete RIA transformer model
pub struct RiaModel {
    pub embedding: Embedding,
    pub layers: Vec<TransformerLayer>,
    pub recurrent_body: Option<RecurrentBody>,
    pub final_norm: RMSNorm,
    pub lm_head: LmHead,
    pub config: ModelConfig,
}

impl RiaModel {
    /// Create new RIA model following GGUF naming per SPEC-003 Section 8.2
    pub fn new(vs: candle_nn::VarBuilder, config: &ModelConfig) -> Result<Self> {
        if let Err(err) = config.validate() {
            candle_core::bail!("invalid RIA model config: {err}");
        }

        let num_layers = config.num_layers();
        let hidden_dim = config.hidden_dim();
        let vocab_size = config.vocab_size();

        let embedding = Embedding::new(vs.clone(), config)?;

        let (layers, recurrent_body) = match config.architecture {
            ModelArchitecture::Transformer => {
                // Create transformer layers following SPEC-003 Section 8.2 naming: blk.{i}.*
                let mut layers = Vec::with_capacity(num_layers);
                for i in 0..num_layers {
                    let layer = TransformerLayer::new(vs.pp(&format!("blk.{}", i)), config)?;
                    layers.push(layer);
                }
                (layers, None)
            }
            ModelArchitecture::RecurrentDepth => {
                let prelude =
                    Prelude::new(vs.pp("prelude"), config, config.recurrent.prelude_layers)?;
                let recurrent_core = RecurrentCore::new(vs.pp("recurrent_core"), config)?;
                let coda = Coda::new(vs.pp("coda"), config, config.recurrent.coda_layers)?;
                (
                    Vec::new(),
                    Some(RecurrentBody {
                        prelude,
                        recurrent_core,
                        coda,
                    }),
                )
            }
        };

        // Final RMSNorm per SPEC-003 Section 8.2
        let final_norm = RMSNorm::new(hidden_dim, config.rms_norm_epsilon(), vs.pp("output_norm"))?;

        // LM head tied with embedding (SPEC-003: "Embedding: Tied input/output")
        let lm_head = LmHead::new();

        info!(
            "RiaModel created: architecture={:?}, layers={}, hidden_dim={}, vocab_size={}",
            config.architecture, num_layers, hidden_dim, vocab_size
        );

        Ok(Self {
            embedding,
            layers,
            recurrent_body,
            final_norm,
            lm_head,
            config: config.clone(),
        })
    }

    /// Create a recurrent-depth RIA model from a config. This is a convenience
    /// constructor for callers that do not want to mutate the config directly.
    pub fn new_recurrent(vs: candle_nn::VarBuilder, config: &ModelConfig) -> Result<Self> {
        let mut config = config.clone();
        config.architecture = ModelArchitecture::RecurrentDepth;
        Self::new(vs, &config)
    }

    /// Forward pass through the model
    /// Per SPEC-003 Section 2.2:
    /// ```text
    /// h_0 = embed(tokens) + pos_embed(positions)
    /// For l = 1 to N:
    ///     h_l = transformer_layer(h_{l-1})
    /// output = final_norm(h_N) · W_embed^T
    /// ```
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.forward_with_options(input_ids, ForwardOptions::default())
    }

    /// Forward pass with explicit recurrent-depth controls.
    pub fn forward_with_options(
        &self,
        input_ids: &Tensor,
        options: ForwardOptions,
    ) -> Result<Tensor> {
        let h = self.forward_hidden_with_options(input_ids, options)?;

        // Final RMSNorm (SPEC-003: output = RMSNorm(h_N) · W_embed^T)
        let h = self.final_norm.forward(&h)?;
        trace!("After final_norm: shape {:?}", h.shape());

        // Tied LM head.
        let logits = self.lm_head.forward(&h, self.embedding.weight())?;
        trace!("RiaModel forward: output shape {:?}", logits.shape());

        Ok(logits)
    }

    /// Forward pass with an explicit recurrent loop count.
    pub fn forward_with_recurrent_iterations(
        &self,
        input_ids: &Tensor,
        recurrent_iterations: usize,
    ) -> Result<Tensor> {
        self.forward_with_options(
            input_ids,
            ForwardOptions {
                recurrent_iterations: Some(recurrent_iterations),
                ..ForwardOptions::default()
            },
        )
    }

    /// Return final hidden states before the output norm and LM head.
    pub fn forward_hidden_with_options(
        &self,
        input_ids: &Tensor,
        options: ForwardOptions,
    ) -> Result<Tensor> {
        trace!("RiaModel forward: input shape {:?}", input_ids.shape());

        // Token embedding (SPEC-003: h_0 = embed(tokens))
        let mut h = self.embedding.forward(input_ids)?;
        trace!("After embedding: shape {:?}", h.shape());

        let mask = if options.use_causal_mask {
            Some(causal_mask(input_ids.dim(1)?, input_ids.device())?)
        } else {
            None
        };
        let mask_ref = mask.as_ref();

        match &self.recurrent_body {
            Some(body) => {
                h = body.prelude.forward(&h, mask_ref)?;
                h = body
                    .recurrent_core
                    .forward(&h, mask_ref, options.recurrent_iterations)?;
                h = body.coda.forward(&h, mask_ref)?;
            }
            None => {
                // Transformer layers (SPEC-003: For l = 1 to N)
                for (i, layer) in self.layers.iter().enumerate() {
                    h = layer.forward(&h, mask_ref)?;
                    trace!("After layer {}: shape {:?}", i, h.shape());
                }
            }
        }

        Ok(h)
    }
}

impl Module for RiaModel {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward(xs)
    }
}

fn causal_mask(seq_len: usize, device: &Device) -> Result<Tensor> {
    let mut values = Vec::with_capacity(seq_len * seq_len);
    for row in 0..seq_len {
        for col in 0..seq_len {
            values.push(if col > row { -1.0e30f32 } else { 0.0 });
        }
    }

    Tensor::from_vec(values, (1, 1, seq_len, seq_len), device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Result};
    use candle_nn::{VarBuilder, VarMap};
    use ria_core::config::RecurrentDepthConfig;

    fn tiny_config() -> ModelConfig {
        ModelConfig::custom(16, 8, 4, 16, 2, 1, 32)
    }

    fn var_builder() -> VarBuilder<'static> {
        let var_map = Box::leak(Box::new(VarMap::new()));
        VarBuilder::from_varmap(var_map, DType::F32, &Device::Cpu)
    }

    #[test]
    fn transformer_forward_produces_vocab_logits() -> Result<()> {
        let config = tiny_config();
        let model = RiaModel::new(var_builder(), &config)?;
        let input = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), &Device::Cpu)?;
        let logits = model.forward(&input)?;

        assert_eq!(logits.dims(), &[1, 4, config.vocab_size()]);
        Ok(())
    }

    #[test]
    fn recurrent_forward_accepts_runtime_iteration_override() -> Result<()> {
        let recurrent = RecurrentDepthConfig {
            prelude_layers: 1,
            recurrent_iterations: 2,
            min_recurrent_iterations: 1,
            max_recurrent_iterations: 4,
            coda_layers: 1,
            residual_scale: 0.5,
        };
        let config = tiny_config().with_recurrent_depth(recurrent);
        let model = RiaModel::new(var_builder(), &config)?;
        let input = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), &Device::Cpu)?;
        let logits = model.forward_with_recurrent_iterations(&input, 3)?;

        assert_eq!(logits.dims(), &[1, 4, config.vocab_size()]);
        assert!(model.recurrent_body.is_some());
        Ok(())
    }
}
