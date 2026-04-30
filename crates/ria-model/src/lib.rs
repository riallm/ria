//! RIA Model - Transformer model architecture
//!
//! Implements the RIA transformer architecture including:
//! - Multi-Hop Code Attention (MHCA)
//! - Tool Integration Router (TIR)
//! - Dual-Path Feed-Forward Network (Planning/Execution)
//! - RMSNorm normalization
//! - RoPE positional encoding

pub mod attention;
pub mod coda;
pub mod embedding;
pub mod ffn;
pub mod layer;
pub mod lm_head;
pub mod model;
pub mod norm;
pub mod position;
pub mod prelude;
pub mod recurrent_core;
pub mod tir;
pub mod training;

pub use attention::MultiHopCodeAttention;
pub use coda::Coda;
pub use ffn::DualPathFFN;
pub use layer::TransformerLayer;
pub use lm_head::LmHead;
pub use model::{ForwardOptions, RecurrentBody, RiaModel};
pub use prelude::Prelude;
pub use recurrent_core::{RecurrentBlock, RecurrentCore};
pub use tir::ToolIntegrationRouter;
pub use training::causal_lm_loss;
