//! GGUF metadata extraction

use std::collections::HashMap;
use byteorder::{LittleEndian, ReadBytesExt};

/// GGUF metadata key-value store
#[derive(Debug, Clone)]
pub struct GgufMetadata {
    pub version: u32,
    pub tensor_count: u64,
    pub kv_count: u64,
    pub kv_pairs: HashMap<String, MetadataValue>,
}

/// Metadata value types
#[derive(Debug, Clone)]
pub enum MetadataValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<MetadataValue>),
}

impl GgufMetadata {
    pub fn new() -> Self {
        Self {
            version: 0,
            tensor_count: 0,
            kv_count: 0,
            kv_pairs: HashMap::new(),
        }
    }

    /// Get a string value by key
    pub fn get_string(&self, key: &str) -> Option<&str> {
        match self.kv_pairs.get(key) {
            Some(MetadataValue::String(s)) => Some(s),
            _ => None,
        }
    }

    /// Get a u32 value by key
    pub fn get_u32(&self, key: &str) -> Option<u32> {
        match self.kv_pairs.get(key) {
            Some(MetadataValue::Uint32(v)) => Some(*v),
            _ => None,
        }
    }

    /// Get required RIA architecture key
    pub fn architecture(&self) -> Option<&str> {
        self.get_string("general.architecture")
    }

    /// Validate that metadata is for a RIA model
    pub fn is_ria_model(&self) -> bool {
        self.architecture() == Some("ria")
    }
}

/// Read a metadata value from the GGUF file
pub fn read_value_type<R: std::io::Read>(reader: &mut R) -> std::io::Result<u32> {
    reader.read_u32::<LittleEndian>()
}

/// Read a string from the GGUF file
pub fn read_string<R: std::io::Read>(reader: &mut R) -> std::io::Result<String> {
    let len = reader.read_u64::<LittleEndian>()?;
    let mut buf = vec![0u8; len as usize];
    reader.read_exact(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).to_string())
}
