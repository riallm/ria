//! GGUF quantization support

use crate::tensor::QuantizationType;

/// Quantization parameters
#[derive(Debug, Clone)]
pub struct QuantizationParams {
    pub qtype: QuantizationType,
    pub block_size: usize,
    pub bits_per_weight: f32,
}

impl QuantizationParams {
    pub fn from_qtype(qtype: QuantizationType) -> Self {
        match qtype {
            QuantizationType::Q4_0 => Self {
                qtype,
                block_size: 32,
                bits_per_weight: 4.0,
            },
            QuantizationType::Q4K_M => Self {
                qtype,
                block_size: 32,
                bits_per_weight: 4.5,
            },
            QuantizationType::Q5_0 => Self {
                qtype,
                block_size: 32,
                bits_per_weight: 5.0,
            },
            QuantizationType::Q5K_M => Self {
                qtype,
                block_size: 32,
                bits_per_weight: 5.5,
            },
            QuantizationType::Q6K => Self {
                qtype,
                block_size: 32,
                bits_per_weight: 6.0,
            },
            QuantizationType::Q8_0 => Self {
                qtype,
                block_size: 32,
                bits_per_weight: 8.0,
            },
            QuantizationType::F16 => Self {
                qtype,
                block_size: 1,
                bits_per_weight: 16.0,
            },
            QuantizationType::F32 => Self {
                qtype,
                block_size: 1,
                bits_per_weight: 32.0,
            },
        }
    }

    /// Calculate size in bytes for a given number of weights
    pub fn size_bytes(&self, num_weights: usize) -> usize {
        (num_weights as f32 * self.bits_per_weight / 8.0).ceil() as usize
    }

    /// Get recommended quantization type
    pub fn recommended() -> QuantizationType {
        QuantizationType::Q4K_M
    }
}
