//! GGUF tensor information

/// Quantization type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationType {
    /// 4-bit quantization
    Q4_0,
    /// 4-bit K-median quantization (recommended default)
    Q4K_M,
    /// 5-bit quantization
    Q5_0,
    /// 5-bit K-median quantization
    Q5K_M,
    /// 6-bit K-median quantization
    Q6K,
    /// 8-bit quantization
    Q8_0,
    /// 16-bit float
    F16,
    /// 32-bit float
    F32,
}

/// Tensor information from GGUF file
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub quantization: QuantizationType,
    pub offset: usize,
    pub size_bytes: usize,
}

impl TensorInfo {
    /// Check if this is an embedding tensor
    pub fn is_embedding(&self) -> bool {
        self.name == "token_embd.weight" || self.name == "output.weight"
    }

    /// Check if this is an attention tensor
    pub fn is_attention(&self) -> bool {
        self.name.contains("attn_")
    }

    /// Check if this is an FFN tensor
    pub fn is_ffn(&self) -> bool {
        self.name.contains("ffn_")
    }

    /// Get the layer number from the tensor name
    pub fn layer_index(&self) -> Option<usize> {
        if let Some(start) = self.name.find("blk.") {
            let rest = &self.name[start + 4..];
            if let Some(end) = rest.find('.') {
                return rest[..end].parse().ok();
            }
        }
        None
    }
}
