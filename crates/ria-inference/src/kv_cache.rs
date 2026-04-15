//! KV cache implementation

use candle_core::{DType, Device, Result, Tensor};

/// Key-Value cache for efficient inference
pub struct KVCache {
    pub key_cache: Vec<Tensor>,
    pub value_cache: Vec<Tensor>,
    pub position: usize,
    pub max_seq_len: usize,
}

impl KVCache {
    pub fn new(num_layers: usize, max_seq_len: usize, device: &Device) -> Result<Self> {
        let mut key_cache = Vec::with_capacity(num_layers);
        let mut value_cache = Vec::with_capacity(num_layers);

        for _ in 0..num_layers {
            key_cache.push(Tensor::zeros((1, 1, 128), DType::F32, device)?);
            value_cache.push(Tensor::zeros((1, 1, 128), DType::F32, device)?);
        }

        Ok(Self {
            key_cache,
            value_cache,
            position: 0,
            max_seq_len,
        })
    }

    /// Get key cache for a layer
    pub fn get_key(&self, layer: usize) -> Option<&Tensor> {
        self.key_cache.get(layer)
    }

    /// Get value cache for a layer
    pub fn get_value(&self, layer: usize) -> Option<&Tensor> {
        self.value_cache.get(layer)
    }

    /// Update cache with new key/value
    pub fn update(&mut self, layer: usize, key: Tensor, value: Tensor) -> Result<()> {
        if layer < self.key_cache.len() {
            self.key_cache[layer] = key;
            self.value_cache[layer] = value;
            self.position += 1;
        }
        Ok(())
    }

    /// Reset cache position
    pub fn reset(&mut self) {
        self.position = 0;
    }
}
