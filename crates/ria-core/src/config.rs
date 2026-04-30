//! Model configuration and tier definitions
//!
//! Per SPEC-003 Section 6: Shared Architecture Across Tiers
//! - Attention Type: Grouped Query Attention (GQA)
//! - KV Heads: 8 (all tiers)
//! - Activation: SwiGLU
//! - Position Encoding: RoPE (θ=10,000)
//! - Normalization: RMSNorm (ε=1e-5)
//! - Embedding: Tied input/output
//! - Vocabulary: 106,000 tokens
//! - Head Dimension: 128

use serde::{Deserialize, Serialize};
use tracing::info;

/// Model tier defining the scale of the RIA model per SPEC-010, SPEC-011, SPEC-012, SPEC-013
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelTier {
    /// RIA-1B: 1.0B parameters, 32K context (SPEC-010)
    /// d=2048, N=24, heads=16, kv_heads=4, ffn=5632
    Ria1B,
    /// RIA-8B: 8.2B parameters, 128K context (SPEC-011)
    /// d=4096, N=36, heads=32, kv_heads=8, ffn=14336
    Ria8B,
    /// RIA-64B: 64.5B parameters, 256K context (SPEC-012)
    /// d=8192, N=64, heads=64, kv_heads=8, ffn=28672
    Ria64B,
    /// RIA-128B: 128.3B parameters, 512K context (SPEC-013)
    /// d=12288, N=80, heads=96, kv_heads=8, ffn=40960
    Ria128B,
}

impl ModelTier {
    /// Hidden dimension (d) per tier specs
    pub fn hidden_dim(&self) -> usize {
        match self {
            ModelTier::Ria1B => 2048,
            ModelTier::Ria8B => 4096,
            ModelTier::Ria64B => 8192,
            ModelTier::Ria128B => 12288,
        }
    }

    /// Number of transformer layers (N) per SPEC-003
    pub fn num_layers(&self) -> usize {
        match self {
            ModelTier::Ria1B => 24,
            ModelTier::Ria8B => 36,
            ModelTier::Ria64B => 64,
            ModelTier::Ria128B => 80,
        }
    }

    /// Number of attention heads per SPEC-010-013
    pub fn num_heads(&self) -> usize {
        match self {
            ModelTier::Ria1B => 16,
            ModelTier::Ria8B => 32,
            ModelTier::Ria64B => 64,
            ModelTier::Ria128B => 96,
        }
    }

    /// Number of KV heads (GQA) per SPEC-003: "KV Heads: 8 (all tiers)"
    /// Note: 1B uses 4 for computational efficiency
    pub fn num_kv_heads(&self) -> usize {
        match self {
            ModelTier::Ria1B => 4,
            ModelTier::Ria8B => 8,
            ModelTier::Ria64B => 8,
            ModelTier::Ria128B => 8,
        }
    }

    /// FFN intermediate dimension per tier specs
    pub fn ffn_dim(&self) -> usize {
        match self {
            ModelTier::Ria1B => 5632,
            ModelTier::Ria8B => 14336,
            ModelTier::Ria64B => 28672,
            ModelTier::Ria128B => 40960,
        }
    }

    /// Maximum context length per tier specs
    pub fn max_context_length(&self) -> usize {
        match self {
            ModelTier::Ria1B => 32_768,
            ModelTier::Ria8B => 131_072,
            ModelTier::Ria64B => 262_144,
            ModelTier::Ria128B => 524_288,
        }
    }

    /// Head dimension: d / heads = 128 per SPEC-003
    pub fn head_dim(&self) -> usize {
        128
    }

    /// Vocabulary size per SPEC-003: "Vocabulary: 106,000 tokens"
    pub fn vocab_size(&self) -> usize {
        106_000
    }

    /// RoPE theta per SPEC-003: 10,000
    pub fn rope_theta(&self) -> f64 {
        10_000.0
    }

    /// RMSNorm epsilon per SPEC-003: 1e-5
    pub fn rms_norm_epsilon(&self) -> f64 {
        1e-5
    }

    /// Estimated parameter count for display purposes
    pub fn estimated_params(&self) -> String {
        match self {
            ModelTier::Ria1B => "1.0B".to_string(),
            ModelTier::Ria8B => "8.2B".to_string(),
            ModelTier::Ria64B => "64.5B".to_string(),
            ModelTier::Ria128B => "128.3B".to_string(),
        }
    }
}

/// High-level model body selection.
///
/// `Transformer` preserves the layer-by-layer RIA architecture described in
/// SPEC-003. `RecurrentDepth` keeps RIA's attention/FFN/TIR block design but
/// organizes the body as Prelude -> shared recurrent core -> Coda.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelArchitecture {
    Transformer,
    RecurrentDepth,
}

impl Default for ModelArchitecture {
    fn default() -> Self {
        Self::Transformer
    }
}

/// Runtime and training configuration for the recurrent-depth RIA body.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecurrentDepthConfig {
    /// Number of distinct transformer blocks before the recurrent core.
    pub prelude_layers: usize,
    /// Number of times to apply the shared recurrent block by default.
    pub recurrent_iterations: usize,
    /// Lower bound for runtime iteration overrides.
    pub min_recurrent_iterations: usize,
    /// Upper bound for runtime iteration overrides.
    pub max_recurrent_iterations: usize,
    /// Number of distinct transformer blocks after the recurrent core.
    pub coda_layers: usize,
    /// Residual delta scale used inside the shared recurrent block.
    pub residual_scale: f64,
}

impl Default for RecurrentDepthConfig {
    fn default() -> Self {
        Self {
            prelude_layers: 2,
            recurrent_iterations: 8,
            min_recurrent_iterations: 1,
            max_recurrent_iterations: 64,
            coda_layers: 2,
            residual_scale: 1.0,
        }
    }
}

impl RecurrentDepthConfig {
    /// Clamp an inference-time loop override to the configured stable range.
    pub fn clamp_iterations(&self, requested: Option<usize>) -> usize {
        let requested = requested.unwrap_or(self.recurrent_iterations);
        requested.clamp(self.min_recurrent_iterations, self.max_recurrent_iterations)
    }

    /// Number of distinct parameterized blocks in the recurrent-depth body.
    pub fn parameterized_block_count(&self) -> usize {
        self.prelude_layers + 1 + self.coda_layers
    }
}

/// Complete model configuration derived from GGUF metadata per SPEC-003 Section 8.1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub tier: ModelTier,
    pub context_length: usize,
    pub embedding_length: usize,
    pub block_count: usize,
    pub feed_forward_length: usize,
    pub attention_head_count: usize,
    pub attention_head_count_kv: usize,
    pub layer_norm_rms_epsilon: f32,
    pub rope_freq_base: f32,
    pub vocab_size: usize,
    #[serde(default)]
    pub architecture: ModelArchitecture,
    #[serde(default)]
    pub recurrent: RecurrentDepthConfig,
}

impl ModelConfig {
    /// Create config from tier specifications
    /// Following GGUF metadata structure per SPEC-003 Section 8.1
    pub fn from_tier(tier: ModelTier) -> Self {
        info!(
            "Creating ModelConfig for {:?}: {} parameters",
            tier,
            tier.estimated_params()
        );
        Self {
            tier,
            context_length: tier.max_context_length(),
            embedding_length: tier.hidden_dim(),
            block_count: tier.num_layers(),
            feed_forward_length: tier.ffn_dim(),
            attention_head_count: tier.num_heads(),
            attention_head_count_kv: tier.num_kv_heads(),
            layer_norm_rms_epsilon: 1e-5,
            rope_freq_base: 10_000.0,
            vocab_size: tier.vocab_size(),
            architecture: ModelArchitecture::Transformer,
            recurrent: RecurrentDepthConfig::default(),
        }
    }

    /// Create a custom config. This is primarily useful for tests and small
    /// local experiments where a full RIA tier would be too large to allocate.
    pub fn custom(
        context_length: usize,
        embedding_length: usize,
        block_count: usize,
        feed_forward_length: usize,
        attention_head_count: usize,
        attention_head_count_kv: usize,
        vocab_size: usize,
    ) -> Self {
        Self {
            tier: ModelTier::Ria1B,
            context_length,
            embedding_length,
            block_count,
            feed_forward_length,
            attention_head_count,
            attention_head_count_kv,
            layer_norm_rms_epsilon: 1e-5,
            rope_freq_base: 10_000.0,
            vocab_size,
            architecture: ModelArchitecture::Transformer,
            recurrent: RecurrentDepthConfig::default(),
        }
    }

    /// Return a copy configured for the recurrent-depth architecture.
    pub fn with_recurrent_depth(mut self, recurrent: RecurrentDepthConfig) -> Self {
        self.architecture = ModelArchitecture::RecurrentDepth;
        self.recurrent = recurrent;
        self
    }

    /// Get the hidden dimension (embedding_length)
    pub fn hidden_dim(&self) -> usize {
        self.embedding_length
    }

    /// Get the number of layers
    pub fn num_layers(&self) -> usize {
        self.block_count
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Get attention head dimension.
    pub fn head_dim(&self) -> usize {
        self.embedding_length / self.attention_head_count
    }

    /// Get RMSNorm epsilon.
    pub fn rms_norm_epsilon(&self) -> f64 {
        self.layer_norm_rms_epsilon as f64
    }

    /// Validate shape-level invariants required by attention and staged bodies.
    pub fn validate(&self) -> Result<(), String> {
        if self.context_length == 0 {
            return Err("context_length must be greater than zero".to_string());
        }
        if self.embedding_length == 0 {
            return Err("embedding_length must be greater than zero".to_string());
        }
        if self.vocab_size == 0 {
            return Err("vocab_size must be greater than zero".to_string());
        }
        if self.block_count == 0 {
            return Err("block_count must be greater than zero".to_string());
        }
        if self.attention_head_count == 0 {
            return Err("attention_head_count must be greater than zero".to_string());
        }
        if self.attention_head_count_kv == 0 {
            return Err("attention_head_count_kv must be greater than zero".to_string());
        }
        if self.embedding_length % self.attention_head_count != 0 {
            return Err("embedding_length must be divisible by attention_head_count".to_string());
        }
        if self.attention_head_count % self.attention_head_count_kv != 0 {
            return Err(
                "attention_head_count must be divisible by attention_head_count_kv".to_string(),
            );
        }
        if matches!(self.architecture, ModelArchitecture::RecurrentDepth) {
            let recurrent = &self.recurrent;
            if recurrent.parameterized_block_count() > self.block_count {
                return Err(
                    "recurrent prelude + shared block + coda exceeds block_count".to_string(),
                );
            }
            if recurrent.min_recurrent_iterations == 0 {
                return Err("min_recurrent_iterations must be greater than zero".to_string());
            }
            if recurrent.min_recurrent_iterations > recurrent.max_recurrent_iterations {
                return Err(
                    "min_recurrent_iterations must be <= max_recurrent_iterations".to_string(),
                );
            }
            if recurrent.residual_scale <= 0.0 {
                return Err("residual_scale must be positive".to_string());
            }
        }

        Ok(())
    }
}
