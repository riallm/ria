//! Dual-Path Feed-Forward Network
//!
//! Implements the planning/execution dual-path FFN with gating per SPEC-003 Section 5:
//! - Planning path: focused on reasoning and strategy
//! - Execution path: focused on implementation details
//! - Gate: learned mixture coefficient
//!
//! Following SPEC-003 Section 8.2 GGUF tensor naming:
//! - blk.{i}.ffn_gate.weight: FFN gate (SwiGLU)
//! - blk.{i}.ffn_down.weight: FFN down projection
//! - blk.{i}.ffn_up.weight: FFN up projection

use candle_core::{Result, Tensor};
use candle_nn::Module;
use ria_core::config::ModelConfig;
use tracing::trace;

/// Dual-Path Feed-Forward Network
pub struct DualPathFFN {
    pub planning_gate: candle_nn::Linear,  // W1_p
    pub planning_up: candle_nn::Linear,    // W2_p
    pub execution_gate: candle_nn::Linear, // W1_e
    pub execution_up: candle_nn::Linear,   // W2_e
    pub gate_proj: candle_nn::Linear,      // W_g
    pub config: ModelConfig,
}

impl DualPathFFN {
    pub fn new(vs: candle_nn::VarBuilder, config: &ModelConfig) -> Result<Self> {
        let hidden_dim = config.tier.hidden_dim();
        let ffn_dim = config.tier.ffn_dim();

        let planning_gate = candle_nn::linear(hidden_dim, ffn_dim, vs.pp("planning_gate"))?;
        let planning_up = candle_nn::linear(ffn_dim, hidden_dim, vs.pp("planning_up"))?;
        let execution_gate = candle_nn::linear(hidden_dim, ffn_dim, vs.pp("execution_gate"))?;
        let execution_up = candle_nn::linear(ffn_dim, hidden_dim, vs.pp("execution_up"))?;
        let gate_proj = candle_nn::linear(hidden_dim, 1, vs.pp("gate"))?;

        Ok(Self {
            planning_gate,
            planning_up,
            execution_gate,
            execution_up,
            gate_proj,
            config: config.clone(),
        })
    }

    /// Forward pass: gate * planning + (1 - gate) * execution
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Compute gate
        let gate_logits = self.gate_proj.forward(x)?;
        let gate = candle_nn::ops::sigmoid(&gate_logits)?;

        // Planning path
        let planning_hidden = self.planning_gate.forward(x)?;
        let planning_hidden = candle_nn::ops::silu(&planning_hidden)?;
        let planning_out = self.planning_up.forward(&planning_hidden)?;

        // Execution path
        let execution_hidden = self.execution_gate.forward(x)?;
        let execution_hidden = candle_nn::ops::silu(&execution_hidden)?;
        let execution_out = self.execution_up.forward(&execution_hidden)?;

        // Gated merge: gate * planning + (1 - gate) * execution
        let one_minus_gate = gate.ones_like()?.sub(&gate)?;
        let gated_planning = planning_out.broadcast_mul(&gate)?;
        let gated_execution = execution_out.broadcast_mul(&one_minus_gate)?;
        gated_planning.add(&gated_execution)
    }
}
