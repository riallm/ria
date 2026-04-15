//! RIA Model - Transformer model architecture
//!
//! Implements the RIA transformer architecture including:
//! - Multi-Hop Code Attention (MHCA)
//! - Tool Integration Router (TIR)
//! - Dual-Path Feed-Forward Network (Planning/Execution)
//! - RMSNorm normalization
//! - RoPE positional encoding

pub mod attention;
pub mod embedding;
pub mod ffn;
pub mod layer;
pub mod model;
pub mod norm;
pub mod position;
pub mod tir;

pub use model::RiaModel;
pub use layer::TransformerLayer;
pub use attention::MultiHopCodeAttention;
pub use ffn::DualPathFFN;
pub use tir::ToolIntegrationRouter;
