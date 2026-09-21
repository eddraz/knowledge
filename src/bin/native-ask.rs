//! `native-ask` — in-process retrieval + generation for the knowledge base.
//!
//! Retrieval is identical to `knowledge ask` (embed via the existing embedding
//! sidecar, KNN, score gate).  Generation runs in-process with candle using a
//! quantized LFM2.5 GGUF model instead of calling a generator sidecar.

use std::path::PathBuf;

use clap::Parser;

use knowledge::ask::{build_prompt, passes_gate};
use knowledge::bootstrap;
use knowledge::config::Config;
use knowledge::db;
use knowledge::error::{KnowledgeError, Result};
use knowledge::llm::{self, EMBED_DIM};
use knowledge::native::{generate_with_loaded, load_model, load_tokenizer, DEFAULT_TOKENIZER_NAME};
use knowledge::sidecar::{acquire, SidecarRole};

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
            // download fails, `load_tokenizer` will surface it.
            let _ = bootstrap::ensure_registry_file(
                &cfg.models_dir,
                DEFAULT_TOKENIZER_NAME,
                args.verbose,
            );
        }
        path
    });

    eprintln!("[native] loading model...");
    let (mut model, context_length) = load_model(&model_path)?;
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

    eprintln!("[native] generating...");
    let (answer, stats) = generate_with_loaded(
        &mut model,
        &tokenizer,
        &system,
        &user,
        args.max_tokens,
        context_length,
    )?;

    println!("{answer}\n");
    println!("Sources:");
    for hit in hits {
        let section = hit.section.as_deref().unwrap_or("-");
        println!("  {:.3}  {}::{}", hit.score, hit.source, section);
    }

    eprintln!(
        "[native] prompt: {} tok ({:.1} tok/s), decode: {} tok ({:.1} tok/s)",
        stats.prompt_tokens, stats.prompt_tps, stats.decoded, stats.decode_tps
    );

    Ok(())
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
