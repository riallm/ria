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

/// Dual-Path Feed-Forward Network
pub struct DualPathFFN {
    pub planning_gate: candle_nn::Linear,
    pub planning_up: candle_nn::Linear,
    pub planning_down: candle_nn::Linear,
    pub execution_gate: candle_nn::Linear,
    pub execution_up: candle_nn::Linear,
    pub execution_down: candle_nn::Linear,
    pub path_gate: candle_nn::Linear,
    pub config: ModelConfig,
}

impl DualPathFFN {
    pub fn new(vs: candle_nn::VarBuilder, config: &ModelConfig) -> Result<Self> {
        let hidden_dim = config.hidden_dim();
        let ffn_dim = config.feed_forward_length;

        let planning_gate = candle_nn::linear(hidden_dim, ffn_dim, vs.pp("planning_gate"))?;
        let planning_up = candle_nn::linear(hidden_dim, ffn_dim, vs.pp("planning_up"))?;
        let planning_down = candle_nn::linear(ffn_dim, hidden_dim, vs.pp("planning_down"))?;
        let execution_gate = candle_nn::linear(hidden_dim, ffn_dim, vs.pp("execution_gate"))?;
        let execution_up = candle_nn::linear(hidden_dim, ffn_dim, vs.pp("execution_up"))?;
        let execution_down = candle_nn::linear(ffn_dim, hidden_dim, vs.pp("execution_down"))?;
        let path_gate = candle_nn::linear(hidden_dim, 1, vs.pp("path_gate"))?;

        Ok(Self {
            planning_gate,
            planning_up,
            planning_down,
            execution_gate,
            execution_up,
            execution_down,
            path_gate,
            config: config.clone(),
        })
    }

    /// Forward pass: gate * planning + (1 - gate) * execution
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Compute mixture gate between the planning and execution paths.
        let gate_logits = self.path_gate.forward(x)?;
        let gate = candle_nn::ops::sigmoid(&gate_logits)?;

        // SwiGLU path: down(silu(gate(x)) * up(x)).
        let planning_out = swiglu(
            x,
            &self.planning_gate,
            &self.planning_up,
            &self.planning_down,
        )?;
        let execution_out = swiglu(
            x,
            &self.execution_gate,
            &self.execution_up,
            &self.execution_down,
        )?;

        // Gated merge: gate * planning + (1 - gate) * execution.
        let one_minus_gate = gate.ones_like()?.sub(&gate)?;
        let gated_planning = planning_out.broadcast_mul(&gate)?;
        let gated_execution = execution_out.broadcast_mul(&one_minus_gate)?;
        gated_planning.add(&gated_execution)
    }
}

fn swiglu(
    x: &Tensor,
    gate: &candle_nn::Linear,
    up: &candle_nn::Linear,
    down: &candle_nn::Linear,
) -> Result<Tensor> {
    let gate = candle_nn::ops::silu(&gate.forward(x)?)?;
    let up = up.forward(x)?;
    down.forward(&gate.mul(&up)?)
}
