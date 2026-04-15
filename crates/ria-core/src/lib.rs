//! RIA Core - Core types, traits, and configuration
//!
//! This crate provides the foundational types and traits used across the RIA ecosystem,
//! including model configuration, device management, and generation parameters.

pub mod config;
pub mod device;
pub mod error;
pub mod generation;
pub mod tokenizer_config;

pub use config::{ModelConfig, ModelTier};
pub use device::{Backend, DeviceConfig};
pub use error::{RiaError, RiaResult};
pub use generation::{GenerationConfig, StopSequence};
pub use tokenizer_config::TokenizerConfig;
