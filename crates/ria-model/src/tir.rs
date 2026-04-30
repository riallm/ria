//! Tool Integration Router (TIR)
//!
//! Enables model to decide when/how to invoke external tools per SPEC-003 Section 4:
//! - tool_gate: decides if tool needed
//! - tool_selector: selects which tool
//! - tool_arg_generator: generates arguments
//!
//! Tool token ranges per SPEC-003 Section 4.2:
//! - Execution: 100,000-100,099
//! - Testing: 100,100-100,149
//! - Analysis: 100,150-100,199
//! - Version Control: 100,200-100,249
//! - Build: 100,250-100,299
//! - Search: 100,300-100,349
//! - Package Management: 100,350-100,399

use candle_core::{Result, Tensor};
use candle_nn::Module;
use ria_core::config::ModelConfig;

/// Tool categories supported by RIA
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolCategory {
    Execution,
    Testing,
    Analysis,
    VersionControl,
    Build,
    Search,
    PackageManagement,
}

/// Tool call result
#[derive(Debug, Clone)]
pub struct ToolCall {
    pub category: ToolCategory,
    pub tool_id: usize,
    pub arguments: String,
}

/// Tool Integration Router
pub struct ToolIntegrationRouter {
    pub tool_gate: candle_nn::Linear,
    pub tool_selector: candle_nn::Linear,
    pub tool_arg_generator: candle_nn::Linear,
    pub num_tools: usize,
}

impl ToolIntegrationRouter {
    pub fn new(vs: candle_nn::VarBuilder, config: &ModelConfig) -> Result<Self> {
        let hidden_dim = config.hidden_dim();
        let num_tools = 400; // Tool token range: 100,000-100,399

        let tool_gate = candle_nn::linear(hidden_dim, 1, vs.pp("gate"))?;
        let tool_selector = candle_nn::linear(hidden_dim, num_tools, vs.pp("selector"))?;
        let tool_arg_generator = candle_nn::linear(hidden_dim, hidden_dim, vs.pp("arg_generator"))?;

        Ok(Self {
            tool_gate,
            tool_selector,
            tool_arg_generator,
            num_tools,
        })
    }

    /// Determine if a tool call is needed and which tool to use
    pub fn forward(&self, x: &Tensor, threshold: f64) -> Result<Option<ToolCall>> {
        let gate_logits = self.tool_gate.forward(x)?;
        let tool_prob = candle_nn::ops::sigmoid(&gate_logits)?;

        // Check if tool is needed
        let tool_needed = tool_prob.get(0)?.to_scalar::<f64>()? > threshold;

        if tool_needed {
            let selection_logits = self.tool_selector.forward(x)?;
            let tool_id = selection_logits.argmax(candle_core::D::Minus1)?;
            let tool_id_scalar = tool_id.get(0)?.to_scalar::<u32>()? as usize;

            let arg_hidden = self.tool_arg_generator.forward(x)?;
            let _arg_repr = arg_hidden;

            Ok(Some(ToolCall {
                category: ToolCategory::from_id(tool_id_scalar),
                tool_id: tool_id_scalar,
                arguments: String::new(), // Would be generated from arg_hidden
            }))
        } else {
            Ok(None)
        }
    }
}

impl ToolCategory {
    pub fn from_id(tool_id: usize) -> Self {
        match tool_id {
            0..=56 => ToolCategory::Execution,
            57..=113 => ToolCategory::Testing,
            114..=170 => ToolCategory::Analysis,
            171..=227 => ToolCategory::VersionControl,
            228..=284 => ToolCategory::Build,
            285..=341 => ToolCategory::Search,
            _ => ToolCategory::PackageManagement,
        }
    }
}
