//! KV cache management

use candle_core::{Result, Tensor};

/// KV cache entry for a single layer
#[derive(Debug, Clone)]
pub struct CacheEntry {
    pub key: Tensor,
    pub value: Tensor,
    pub seq_len: usize,
}

/// KV cache manager
pub struct KvCacheManager {
    pub caches: Vec<Option<CacheEntry>>,
    pub max_seq_len: usize,
    pub cache_size_bytes: u64,
}

impl KvCacheManager {
    pub fn new(num_layers: usize, max_seq_len: usize, cache_size_bytes: u64) -> Self {
        let mut caches = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            caches.push(None);
        }
        Self {
            caches,
            max_seq_len,
            cache_size_bytes,
        }
    }

    /// Update cache for a layer
    pub fn update(&mut self, layer_idx: usize, key: Tensor, value: Tensor, seq_len: usize) {
        if layer_idx < self.caches.len() {
            self.caches[layer_idx] = Some(CacheEntry {
                key,
                value,
                seq_len,
            });
        }
    }

    /// Get cache for a layer
    pub fn get(&self, layer_idx: usize) -> Option<&CacheEntry> {
        self.caches.get(layer_idx).and_then(|c| c.as_ref())
    }

    /// Clear all caches
    pub fn clear(&mut self) {
        for cache in &mut self.caches {
            *cache = None;
        }
    }

    /// Get current cache size in bytes
    pub fn current_size_bytes(&self) -> u64 {
        // Approximate size calculation
        self.caches
            .iter()
            .filter_map(|c| c.as_ref())
            .map(|c| {
                // Approximate tensor size: seq_len * hidden_dim * sizeof(f32) * 2 (key + value)
                c.seq_len as u64 * 1024 * 4 * 2
            })
            .sum()
    }

    /// Evict old entries if over size limit
    pub fn evict_if_needed(&mut self) {
        while self.current_size_bytes() > self.cache_size_bytes {
            // Evict oldest entry
            if let Some(idx) = self.caches.iter().position(|c| c.is_some()) {
                self.caches[idx] = None;
            } else {
                break;
            }
        }
    }
}
