//! RIA Tokenizer - Hybrid BPE + Custom Token tokenizer
//!
//! Implements the tokenizer specified in the RIA spec:
//! - 106,000 total vocabulary
//! - 98K BPE subword tokens
//! - 5K identifier tokens
//! - 2K syntax tokens
//! - 500 tool tokens
//! - 500 agentic tokens

pub mod bpe;
pub mod special_tokens;
pub mod tokenizer;

pub use special_tokens::SpecialTokenRegistry;
pub use tokenizer::RiaTokenizer;
