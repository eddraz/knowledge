//! In-process generation with a quantized LFM2.5 GGUF model via candle.
//!
//! Compiled only when the `native` feature is enabled. The public API is
//! intentionally small: `render_chat`, `eos_id`, `GenStats`, and `generate`.

use std::path::Path;
use std::time::Instant;

use candle::quantized::gguf_file;
use candle::{Device, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::quantized_lfm2::ModelWeights;
use candle_transformers::utils::apply_repeat_penalty;
use tokenizers::Tokenizer;

use crate::error::{KnowledgeError, Result};

pub const DEFAULT_TOKENIZER_NAME: &str = "LFM2.5-tokenizer.json";
const EOS_TOKEN: &str = "<|im_end|>";
const SEED: u64 = 1;
const REPEAT_PENALTY: f32 = 1.0;

/// Generation timing statistics returned by `generate`.
#[derive(Debug, Clone, Copy)]
pub struct GenStats {
    pub prompt_tokens: usize,
    pub prompt_tps: f64,
    pub decoded: usize,
    pub decode_tps: f64,
}

/// Render a system+user turn in LFM2.5's ChatML-like format.
///
/// LFM2.5 uses these special tokens (from `tokenizer_config.json` and the
/// model card):
///   * `<|startoftext|>` at the very beginning,
///   * `<|im_start|>` before each role line,
///   * `<|im_end|>` after each role's content,
///   * an open `<|im_start|>assistant\n` prompt to elicit generation.
///
/// This is the exact rendered form for a single system + user turn.
pub fn render_chat(system: &str, user: &str) -> String {
    format!(
        "<|startoftext|><|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    )
}

/// Best-effort EOS token id for LFM2.5 generation.
pub fn eos_id(tokenizer: &Tokenizer) -> Option<u32> {
    tokenizer.token_to_id(EOS_TOKEN)
}

/// Load a quantized LFM2.5 GGUF model and return its context length.
pub fn load_model(model_path: &Path) -> Result<(ModelWeights, usize)> {
    let device = Device::Cpu;
    let mut file = std::fs::File::open(model_path).map_err(KnowledgeError::Io)?;
    let gguf = gguf_file::Content::read(&mut file)
        .map_err(|e| KnowledgeError::Other(format!("failed to read GGUF: {e}")))?;

    let context_length = gguf
        .metadata
        .get("lfm2.context_length")
        .and_then(|v| v.to_u32().ok().map(|v| v as usize))
        .unwrap_or(32_768);

    let model = ModelWeights::from_gguf(gguf, &mut file, &device)
        .map_err(|e| KnowledgeError::Other(format!("failed to load model weights: {e}")))?;

    Ok((model, context_length))
}

/// Load the LFM2.5 tokenizer from a `tokenizer.json` file.
pub fn load_tokenizer(tokenizer_path: &Path) -> Result<Tokenizer> {
    Tokenizer::from_file(tokenizer_path)
        .map_err(|e| KnowledgeError::Other(format!("failed to load tokenizer: {e}")))
}

/// Generate an answer using an already-loaded model and tokenizer.
///
/// This is used by the `native-ask` binary so it can print progress between
/// model load and the actual generation run.
pub fn generate_with_loaded(
    model: &mut ModelWeights,
    tokenizer: &Tokenizer,
    system: &str,
    user: &str,
    max_tokens: usize,
    context_length: usize,
) -> Result<(String, GenStats)> {
    let prompt = render_chat(system, user);
    generate_answer(
        model,
        tokenizer,
        &prompt,
        max_tokens,
        context_length,
        &Device::Cpu,
    )
}

/// Load a tokenizer and a quantized LFM2.5 model and generate an answer.
///
/// `system` and `user` are rendered into the LFM2.5 chat template, trimmed to
/// the model's reported context length, then fed to the model with TopP
/// sampling (p=0.95, temperature=0.2, seed=1) and a repeat penalty of 1.0.
pub fn generate(
    model_path: &Path,
    tokenizer_path: &Path,
    system: &str,
    user: &str,
    max_tokens: usize,
) -> Result<(String, GenStats)> {
    let (mut model, context_length) = load_model(model_path)?;
    let tokenizer = load_tokenizer(tokenizer_path)?;
    generate_with_loaded(
        &mut model,
        &tokenizer,
        system,
        user,
        max_tokens,
        context_length,
    )
}

fn generate_answer(
    model: &mut ModelWeights,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_tokens: usize,
    context_length: usize,
    device: &Device,
) -> Result<(String, GenStats)> {
    let mut tokens = tokenizer
        .encode(prompt, false)
        .map_err(|e| KnowledgeError::Other(format!("failed to encode prompt: {e}")))?
        .get_ids()
        .to_vec();

    if tokens.len() > context_length.saturating_sub(1) {
        tokens.truncate(context_length.saturating_sub(1));
    }
    let prompt_tokens = tokens.len();

    let eos_token = eos_id(tokenizer)
        .ok_or_else(|| KnowledgeError::Other(format!("tokenizer is missing {EOS_TOKEN}")))?;

    let sampling = Sampling::TopP {
        p: 0.95,
        temperature: 0.2,
    };
    let mut logits_processor = LogitsProcessor::from_sampling(SEED, sampling);

    let start_prompt = Instant::now();
    let input = Tensor::new(tokens.as_slice(), device)
        .map_err(|e| KnowledgeError::Other(format!("input tensor error: {e}")))?
        .unsqueeze(0)
        .map_err(|e| KnowledgeError::Other(format!("unsqueeze error: {e}")))?;
    let logits = model
        .forward(&input, 0)
        .map_err(|e| KnowledgeError::Other(format!("model forward error: {e}")))?
        .squeeze(0)
        .map_err(|e| KnowledgeError::Other(format!("logits squeeze error: {e}")))?;
    let prompt_duration = start_prompt.elapsed();

    let mut next_token = logits_processor
        .sample(&logits)
        .map_err(|e| KnowledgeError::Other(format!("sampling error: {e}")))?;
    tokens.push(next_token);

    let start_decode = Instant::now();
    let mut generated_tokens = vec![next_token];

    for _ in 1..max_tokens {
        if next_token == eos_token {
            break;
        }
        let input = Tensor::new(&[next_token], device)
            .map_err(|e| KnowledgeError::Other(format!("input tensor error: {e}")))?
            .unsqueeze(0)
            .map_err(|e| KnowledgeError::Other(format!("unsqueeze error: {e}")))?;
        let logits = model
            .forward(&input, tokens.len() - 1)
            .map_err(|e| KnowledgeError::Other(format!("model forward error: {e}")))?
            .squeeze(0)
            .map_err(|e| KnowledgeError::Other(format!("logits squeeze error: {e}")))?;
        let logits = apply_repeat_penalty(&logits, REPEAT_PENALTY, &tokens)
            .map_err(|e| KnowledgeError::Other(format!("repeat penalty error: {e}")))?;
        next_token = logits_processor
            .sample(&logits)
            .map_err(|e| KnowledgeError::Other(format!("sampling error: {e}")))?;
        tokens.push(next_token);
        generated_tokens.push(next_token);
    }
    let decode_duration = start_decode.elapsed();

    let answer = tokenizer
        .decode(&generated_tokens, true)
        .map_err(|e| KnowledgeError::Other(format!("decode error: {e}")))?;

    let prompt_tps = prompt_tokens as f64 / prompt_duration.as_secs_f64().max(1e-9);
    let decoded = generated_tokens.len();
    let decode_tps = decoded as f64 / decode_duration.as_secs_f64().max(1e-9);

    Ok((
        answer,
        GenStats {
            prompt_tokens,
            prompt_tps,
            decoded,
            decode_tps,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_chat_includes_system_user_and_special_tokens() {
        let out = render_chat("You are helpful.", "What is Rust?");

        assert!(out.contains("You are helpful."), "system text missing");
        assert!(out.contains("What is Rust?"), "user text missing");
        assert!(out.contains("<|startoftext|>"), "BOS token missing");
        assert!(out.contains("<|im_start|>"), "im_start token missing");
        assert!(out.contains("<|im_end|>"), "im_end token missing");
        assert!(out.contains("<|im_start|>system\nYou are helpful."));
        assert!(out.contains("<|im_start|>user\nWhat is Rust?"));
        assert!(out.contains("<|im_start|>assistant\n"));
    }
}
