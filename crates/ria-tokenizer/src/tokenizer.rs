//! Main tokenizer implementation combining BPE and special tokens

use ria_core::{RiaResult, TokenizerConfig};

use crate::bpe::BPECore;
use crate::special_tokens::SpecialTokenRegistry;

/// Complete RIA tokenizer
pub struct RiaTokenizer {
    bpe: BPECore,
    special_tokens: SpecialTokenRegistry,
    config: TokenizerConfig,
}

impl RiaTokenizer {
    pub fn new(config: TokenizerConfig) -> Self {
        let mut bpe = BPECore::new(config.vocab_size);
        let special_tokens = SpecialTokenRegistry::new();

        // Initialize BPE with special token offset
        bpe.vocab.extend(
            (0..special_tokens.len())
                .map(|id| crate::bpe::BPEEntry {
                    token: format!("<special_{}>", id),
                    id,
                    is_special: true,
                })
        );

        Self {
            bpe,
            special_tokens,
            config,
        }
    }

    /// Encode text into token IDs
    pub fn encode(&self, text: &str) -> RiaResult<Vec<usize>> {
        self.bpe.encode(text)
    }

    /// Decode token IDs back to text
    pub fn decode(&self, token_ids: &[usize]) -> RiaResult<String> {
        self.bpe.decode(token_ids)
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    /// Get special token ID
    pub fn get_special_token(&self, token: &str) -> Option<usize> {
        self.special_tokens.get_id(token)
    }
}
