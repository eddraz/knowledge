//! `native-ask` — in-process retrieval + generation for the knowledge base.
//!
//! Retrieval is identical to `knowledge ask` (embed via the existing embedding
//! sidecar, KNN, score gate).  Generation runs in-process with candle using a
//! quantized LFM2.5 GGUF model instead of calling a generator sidecar.

use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::Parser;

use candle::quantized::gguf_file;
use candle::{Device, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::quantized_lfm2::ModelWeights;
use candle_transformers::utils::apply_repeat_penalty;
use tokenizers::Tokenizer;

use knowledge::ask::{build_prompt, passes_gate};
use knowledge::bootstrap;
use knowledge::config::Config;
use knowledge::db;
use knowledge::error::{KnowledgeError, Result};
use knowledge::llm::{self, EMBED_DIM};
use knowledge::sidecar::{acquire, SidecarRole};

const DEFAULT_TOKENIZER_NAME: &str = "LFM2.5-tokenizer.json";
const EOS_TOKEN: &str = "<|im_end|>";
const SEED: u64 = 1;
const REPEAT_PENALTY: f32 = 1.0;

/// Ask a question using in-process retrieval and generation.
#[derive(Parser)]
#[command(
    name = "native-ask",
    about = "In-process RAG: retrieve with the embedding sidecar, generate with candle"
)]
struct Args {
    /// Question to answer.
    question: String,

    /// Owner namespace (also searches `_shared`).
    #[arg(long)]
    owner: Option<String>,

    /// Number of chunks to retrieve.
    #[arg(long, short = 'k', default_value = "5")]
    k: usize,

    /// Maximum number of tokens to generate.
    #[arg(long, default_value = "512")]
    max_tokens: usize,

    /// Override path to the quantized LFM2.5 GGUF model.
    #[arg(long)]
    model: Option<PathBuf>,

    /// Override path to the LFM2.5 tokenizer.json.
    #[arg(long)]
    tokenizer: Option<PathBuf>,

    /// Print bootstrap/model load progress.
    #[arg(long)]
    verbose: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    run().await.map_err(|e| anyhow::anyhow!(e.to_string()))
}

async fn run() -> Result<()> {
    let args = Args::parse();
    let mut cfg = Config::load()?;

    bootstrap::ensure_ready(&mut cfg, args.verbose)?;

    let model_path = args.model.unwrap_or_else(|| cfg.gen_model_path());
    let tokenizer_path = args.tokenizer.unwrap_or_else(|| {
        let path = cfg.models_dir.join(DEFAULT_TOKENIZER_NAME);
        if !path.exists() {
            // Best-effort lazy download via the same registry used by
            // `ensure_ready`.  Ignoring the result here is fine: if the
            // download fails, `Tokenizer::from_file` will surface it.
            let _ = bootstrap::ensure_registry_file(
                &cfg.models_dir,
                DEFAULT_TOKENIZER_NAME,
                args.verbose,
            );
        }
        path
    });

    eprintln!("[native] loading model...");
    let device = Device::Cpu;
    let (mut model, context_length) = load_model(&model_path, &device)?;
    let tokenizer = load_tokenizer(&tokenizer_path)?;
    eprintln!("[native] model ready (context length: {context_length})");

    let http = llm::http_client(cfg.request_timeout_secs)?;
    let conn = db::open(&cfg.db_path)?;

    eprintln!("[native] embedding question...");
    let _embed_handle = acquire(&cfg, SidecarRole::Embedding).await?;
    let query_vec = embed_query(&http, &cfg.embed_base_url(), &args.question).await?;

    let hits = db::knn_search(&conn, &query_vec, args.k, args.owner.as_deref())?;
    let best = hits.first().map(|h| h.score as f32);
    if !passes_gate(best, cfg.min_score) {
        println!("No related content was found in the knowledge base.");
        return Ok(());
    }

    let (system, user) = build_prompt(&args.question, &hits, 1200);
    let prompt = render_chat(&system, &user);

    eprintln!("[native] generating...");
    let (answer, prompt_tokens, decode_tokens, prompt_tok_s, decode_tok_s) = generate_answer(
        &mut model,
        &tokenizer,
        &prompt,
        args.max_tokens,
        context_length,
        &device,
    )?;

    println!("{answer}\n");
    println!("Sources:");
    for hit in hits {
        let section = hit.section.as_deref().unwrap_or("-");
        println!("  {:.3}  {}::{}", hit.score, hit.source, section);
    }

    eprintln!(
        "[native] prompt: {prompt_tokens} tok ({prompt_tok_s:.1} tok/s), decode: {decode_tokens} tok ({decode_tok_s:.1} tok/s)"
    );

    Ok(())
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

async fn embed_query(http: &reqwest::Client, base_url: &str, question: &str) -> Result<Vec<f32>> {
    let embeddings = llm::embed_texts(http, base_url, &[question.to_string()]).await?;
    let vec = embeddings
        .into_iter()
        .next()
        .unwrap_or_else(|| vec![0.0_f32; EMBED_DIM]);
    if vec.len() != EMBED_DIM {
        return Err(KnowledgeError::Other(format!(
            "expected embedding dimension {EMBED_DIM}, got {}",
            vec.len()
        )));
    }
    Ok(vec)
}

fn load_model(model_path: &Path, device: &Device) -> Result<(ModelWeights, usize)> {
    let mut file = std::fs::File::open(model_path).map_err(KnowledgeError::Io)?;
    let gguf = gguf_file::Content::read(&mut file)
        .map_err(|e| KnowledgeError::Other(format!("failed to read GGUF: {e}")))?;

    let context_length = gguf
        .metadata
        .get("lfm2.context_length")
        .and_then(|v| v.to_u32().ok().map(|v| v as usize))
        .unwrap_or(32_768);

    let model = ModelWeights::from_gguf(gguf, &mut file, device)
        .map_err(|e| KnowledgeError::Other(format!("failed to load model weights: {e}")))?;

    Ok((model, context_length))
}

fn load_tokenizer(tokenizer_path: &Path) -> Result<Tokenizer> {
    Tokenizer::from_file(tokenizer_path)
        .map_err(|e| KnowledgeError::Other(format!("failed to load tokenizer: {e}")))
}

fn generate_answer(
    model: &mut ModelWeights,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_tokens: usize,
    context_length: usize,
    device: &Device,
) -> Result<(String, usize, usize, f64, f64)> {
    let mut tokens = tokenizer
        .encode(prompt, false)
        .map_err(|e| KnowledgeError::Other(format!("failed to encode prompt: {e}")))?
        .get_ids()
        .to_vec();

    if tokens.len() > context_length.saturating_sub(1) {
        tokens.truncate(context_length.saturating_sub(1));
    }
    let prompt_tokens = tokens.len();

    let eos_token = tokenizer
        .token_to_id(EOS_TOKEN)
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

    let prompt_tok_s = prompt_tokens as f64 / prompt_duration.as_secs_f64().max(1e-9);
    let decode_tokens = generated_tokens.len();
    let decode_tok_s = decode_tokens as f64 / decode_duration.as_secs_f64().max(1e-9);

    Ok((
        answer,
        prompt_tokens,
        decode_tokens,
        prompt_tok_s,
        decode_tok_s,
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
