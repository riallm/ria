//! RIA Inference - Generation and sampling
//!
//! Implements the inference engine for the RIA LLM model:
//! - Token generation with various sampling strategies
//! - KV cache management
//! - Batched inference
//! - Streaming generation

pub mod cache;
pub mod generator;
pub mod kv_cache;
pub mod sampler;

pub use generator::TextGenerator;
pub use sampler::{Sampler, SamplingStrategy};
pub use cache::KvCacheManager;
