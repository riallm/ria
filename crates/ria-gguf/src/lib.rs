//! RIA GGUF - GGUF format loader and metadata
//!
//! Implements the GGUF format support specified in the RIA spec:
//! - GGUF v3 parsing
//! - Metadata extraction
//! - Tensor loading with memory mapping
//! - Quantization support

pub mod loader;
pub mod metadata;
pub mod quantization;
pub mod tensor;

pub use loader::GgufLoader;
pub use metadata::GgufMetadata;
pub use tensor::TensorInfo;
