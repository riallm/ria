//! Error types for the RIA ecosystem

use thiserror::Error;

#[derive(Debug, Error)]
pub enum RiaError {
    #[error("Model loading error: {0}")]
    ModelLoad(String),

    #[error("Inference error: {0}")]
    Inference(String),

    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Device error: {0}")]
    Device(String),

    #[error("GGUF format error: {0}")]
    Gguf(String),

    #[error("API error: {0}")]
    Api(String),

    #[error("Tool execution error: {0}")]
    Tool(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Candle ML error: {0}")]
    Candle(String),

    #[error("Other: {0}")]
    Other(String),
}

pub type RiaResult<T> = Result<T, RiaError>;
