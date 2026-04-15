//! Device and backend configuration
//!
//! Per SPEC-003 for compute backends: CPU, CUDA, Metal, Vulkan

use serde::{Deserialize, Serialize};
use std::str::FromStr;

/// Backend device type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Backend {
    /// CPU inference
    Cpu,
    /// CUDA GPU inference
    Cuda,
    /// Apple Metal inference
    Metal,
    /// Vulkan inference (beta)
    Vulkan,
}

impl Default for Backend {
    fn default() -> Self {
        Backend::Cpu
    }
}

impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Backend::Cpu => write!(f, "cpu"),
            Backend::Cuda => write!(f, "cuda"),
            Backend::Metal => write!(f, "metal"),
            Backend::Vulkan => write!(f, "vulkan"),
        }
    }
}

impl FromStr for Backend {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "cpu" => Ok(Backend::Cpu),
            "cuda" => Ok(Backend::Cuda),
            "metal" => Ok(Backend::Metal),
            "vulkan" => Ok(Backend::Vulkan),
            _ => Err(format!("Unknown backend: {}", s)),
        }
    }
}

impl Backend {
    pub fn from_str_optional(s: &str) -> Option<Self> {
        s.parse().ok()
    }
}

/// Device configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceConfig {
    pub backend: Backend,
    pub device_id: Option<u32>,
    pub memory_limit_bytes: Option<u64>,
    pub use_mmap: bool,
    pub prefetch_layers: usize,
    pub cache_size_bytes: u64,
}

impl Default for DeviceConfig {
    fn default() -> Self {
        Self {
            backend: Backend::Cpu,
            device_id: None,
            memory_limit_bytes: None,
            use_mmap: true,
            prefetch_layers: 4,
            cache_size_bytes: 256 * 1024 * 1024, // 256MB
        }
    }
}
