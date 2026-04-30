//! GGUF file loader

use byteorder::{LittleEndian, ReadBytesExt};
use memmap2::Mmap;
use ria_core::{config::ModelConfig, RiaError, RiaResult};
use std::fs::File;
use std::path::Path;

use crate::metadata::{GgufMetadata, MetadataValue};
use crate::tensor::TensorInfo;

/// GGUF magic number: "GGUF" = 0x46554747
const GGUF_MAGIC: u32 = 0x46554747;

/// GGUF file loader
pub struct GgufLoader {
    pub mmap: Mmap,
    pub metadata: GgufMetadata,
    pub tensors: Vec<TensorInfo>,
}

impl GgufLoader {
    /// Load a GGUF file
    pub fn load<P: AsRef<Path>>(path: P) -> RiaResult<Self> {
        let file = File::open(&path).map_err(|e| RiaError::ModelLoad(e.to_string()))?;
        let mmap = unsafe { Mmap::map(&file).map_err(|e| RiaError::ModelLoad(e.to_string()))? };

        let mut metadata = GgufMetadata::new();
        let tensors = Vec::new();

        // Would parse GGUF file here
        // For skeleton, we just initialize

        Ok(Self {
            mmap,
            metadata,
            tensors,
        })
    }

    /// Extract model configuration from metadata
    pub fn extract_config(&self) -> RiaResult<ModelConfig> {
        if !self.metadata.is_ria_model() {
            return Err(RiaError::ModelLoad(
                "GGUF file does not contain a RIA model".to_string(),
            ));
        }

        // Extract required fields
        let context_length = self
            .metadata
            .get_u32("ria.context_length")
            .ok_or_else(|| RiaError::Config("Missing ria.context_length".to_string()))?;

        let embedding_length = self
            .metadata
            .get_u32("ria.embedding_length")
            .ok_or_else(|| RiaError::Config("Missing ria.embedding_length".to_string()))?;

        let block_count = self
            .metadata
            .get_u32("ria.block_count")
            .ok_or_else(|| RiaError::Config("Missing ria.block_count".to_string()))?;

        let feed_forward_length = self
            .metadata
            .get_u32("ria.feed_forward_length")
            .ok_or_else(|| RiaError::Config("Missing ria.feed_forward_length".to_string()))?;

        let attention_head_count = self
            .metadata
            .get_u32("ria.attention.head_count")
            .ok_or_else(|| RiaError::Config("Missing ria.attention.head_count".to_string()))?;

        let attention_head_count_kv = self
            .metadata
            .get_u32("ria.attention.head_count_kv")
            .ok_or_else(|| RiaError::Config("Missing ria.attention.head_count_kv".to_string()))?;

        let layer_norm_rms_epsilon = self
            .metadata
            .kv_pairs
            .get("ria.attention.layer_norm_rms_epsilon")
            .and_then(|v| match v {
                MetadataValue::Float32(f) => Some(*f),
                _ => None,
            })
            .unwrap_or(1e-5);

        let rope_freq_base = self
            .metadata
            .kv_pairs
            .get("ria.rope.freq_base")
            .and_then(|v| match v {
                MetadataValue::Float32(f) => Some(*f),
                _ => None,
            })
            .unwrap_or(10_000.0);

        let vocab_size = self
            .metadata
            .get_u32("ria.vocab_size")
            .ok_or_else(|| RiaError::Config("Missing ria.vocab_size".to_string()))?;

        Ok(ModelConfig {
            tier: ria_core::config::ModelTier::Ria8B, // Would infer from dimensions
            context_length: context_length as usize,
            embedding_length: embedding_length as usize,
            block_count: block_count as usize,
            feed_forward_length: feed_forward_length as usize,
            attention_head_count: attention_head_count as usize,
            attention_head_count_kv: attention_head_count_kv as usize,
            layer_norm_rms_epsilon,
            rope_freq_base,
            vocab_size: vocab_size as usize,
            architecture: Default::default(),
            recurrent: Default::default(),
        })
    }
}
