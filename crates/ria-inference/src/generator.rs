//! Text generator

use candle_core::Tensor;
use ria_core::{generation::GenerationConfig, RiaResult};
use ria_model::model::RiaModel;
use ria_tokenizer::RiaTokenizer;

use crate::sampler::Sampler;

/// Generated token with optional logprob
#[derive(Debug, Clone)]
pub struct GeneratedToken {
    pub id: u32,
    pub text: String,
    pub logprob: Option<f64>,
}

/// Generation output
#[derive(Debug, Clone)]
pub struct GenerationOutput {
    pub tokens: Vec<GeneratedToken>,
    pub finish_reason: FinishReason,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
}

/// Reason generation finished
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// Reached max tokens
    Length,
    /// Generated stop sequence
    Stop,
    /// Model predicted EOS
    Eos,
}

/// Text generator
pub struct TextGenerator {
    pub model: RiaModel,
    pub tokenizer: RiaTokenizer,
    pub sampler: Sampler,
    pub config: GenerationConfig,
}

impl TextGenerator {
    pub fn new(model: RiaModel, tokenizer: RiaTokenizer, config: GenerationConfig) -> Self {
        let sampler = Sampler::new(&config);
        Self {
            model,
            tokenizer,
            sampler,
            config,
        }
    }

    /// Generate text from a prompt
    pub fn generate(&self, prompt: &str) -> RiaResult<GenerationOutput> {
        // Encode prompt
        let prompt_tokens: Vec<u32> = self
            .tokenizer
            .encode(prompt)?
            .iter()
            .map(|&x| x as u32)
            .collect();
        let prompt_len = prompt_tokens.len();

        // Generation loop
        let mut generated = Vec::new();
        let mut current_tokens = prompt_tokens.clone();
        let eos_id = self
            .tokenizer
            .get_special_token("<|eos|>")
            .map(|id| id as u32);
        let mut finish_reason = FinishReason::Length;

        for _ in 0..self.config.max_new_tokens {
            // Forward pass
            let input_tensor = Tensor::new(current_tokens.as_slice(), &candle_core::Device::Cpu)
                .and_then(|t| t.unsqueeze(0))
                .map_err(|e| ria_core::RiaError::Inference(e.to_string()))?;
            let logits = self
                .model
                .forward(&input_tensor)
                .map_err(|e| ria_core::RiaError::Inference(e.to_string()))?;
            let seq_len = logits
                .dim(1)
                .map_err(|e| ria_core::RiaError::Inference(e.to_string()))?;
            let last_logits = logits
                .narrow(1, seq_len - 1, 1)
                .and_then(|t| t.squeeze(1))
                .and_then(|t| t.squeeze(0))
                .map_err(|e| ria_core::RiaError::Inference(e.to_string()))?;

            // Sample next token
            let next_token = self
                .sampler
                .sample(&last_logits)
                .map_err(|e| ria_core::RiaError::Inference(e.to_string()))?;
            current_tokens.push(next_token);

            let text = self
                .tokenizer
                .decode(&[next_token as usize])
                .unwrap_or_else(|_| String::new());
            generated.push(GeneratedToken {
                id: next_token,
                text,
                logprob: None,
            });

            if Some(next_token) == eos_id {
                finish_reason = FinishReason::Eos;
                break;
            }

            let generated_ids = generated
                .iter()
                .map(|token| token.id as usize)
                .collect::<Vec<_>>();
            let generated_text = self
                .tokenizer
                .decode(&generated_ids)
                .unwrap_or_else(|_| generated.iter().map(|token| token.text.as_str()).collect());
            if self
                .config
                .stop_sequences
                .iter()
                .any(|stop| generated_text.ends_with(&stop.0))
            {
                finish_reason = FinishReason::Stop;
                break;
            }
        }

        let completion_tokens = generated.len();
        Ok(GenerationOutput {
            tokens: generated,
            finish_reason,
            prompt_tokens: prompt_len,
            completion_tokens,
        })
    }

    /// Generate text streaming (callback-based)
    pub fn generate_stream<F>(&self, _prompt: &str, mut callback: F) -> RiaResult<()>
    where
        F: FnMut(GeneratedToken) -> bool,
    {
        let output = self.generate(_prompt)?;
        for token in output.tokens {
            if !callback(token) {
                break;
            }
        }
        Ok(())
    }
}
