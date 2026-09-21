//! `knowledge` — local RAG knowledge base CLI.
//!
//! Documents are chunked, embedded with bge-m3 and stored in SQLite with
//! sqlite-vec (semantic KNN) and FTS5 (lexical BM25). `ask` answers strictly
//! from retrieved content using a local LFM2.5 generator sidecar.

mod ask;
mod chunk;
mod config;
mod db;
mod error;
mod ingest;
mod llm;
mod search;
mod sidecar;

use std::io::Read;

use clap::{Parser, Subcommand};

use crate::config::Config;
use crate::error::KnowledgeError;

#[derive(Parser)]
#[command(
    name = "knowledge",
    about = "Local RAG knowledge base: semantic + lexical search over your own documents",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Ingest a text/markdown file into the knowledge base ("-" reads stdin).
    Add { path: String },

    /// Ask a question; answers strictly from ingested content.
    Ask { question: String },

    /// Search the knowledge base without generating an answer.
    Search {
        query: String,
        /// Search mode: vector | lexical | hybrid.
        #[arg(long, default_value = "vector")]
        mode: String,
        /// Maximum number of results.
        #[arg(long, default_value = "5")]
        k: usize,
    },

    /// List ingested documents.
    List,

    /// Remove a document and all of its chunks.
    Rm { source: String },

    /// Show configuration and database status.
    Status,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let cfg = Config::load().map_err(map_err)?;

    match cli.command {
        Commands::Add { path } => cmd_add(&cfg, &path).await?,
        Commands::Ask { question } => cmd_ask(&cfg, &question).await?,
        Commands::Search { query, mode, k } => cmd_search(&cfg, &query, &mode, k).await?,
        Commands::List => cmd_list(&cfg)?,
        Commands::Rm { source } => cmd_rm(&cfg, &source)?,
        Commands::Status => cmd_status(&cfg)?,
    }
    Ok(())
}

fn map_err(e: KnowledgeError) -> anyhow::Error {
    anyhow::anyhow!(e.to_string())
}

fn open_db(cfg: &Config) -> anyhow::Result<rusqlite::Connection> {
    db::open(&cfg.db_path).map_err(map_err)
}

fn read_input(path: &str) -> anyhow::Result<String> {
    if path == "-" {
        let mut buf = String::new();
        std::io::stdin().read_to_string(&mut buf)?;
        Ok(buf)
    } else {
        Ok(std::fs::read_to_string(path)?)
    }
}

fn source_name(path: &str) -> String {
    if path == "-" {
        return "stdin".to_string();
    }
    std::path::Path::new(path)
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string())
}

async fn cmd_add(cfg: &Config, path: &str) -> anyhow::Result<()> {
    let text = read_input(path)?;
    let source = source_name(path);
    let http = llm::http_client(cfg.request_timeout_secs).map_err(map_err)?;

    // `ingest` only performs embeddings; make sure the embedding sidecar is up.
    let _sidecar = sidecar::acquire(cfg, sidecar::SidecarRole::Embedding)
        .await
        .map_err(map_err)?;

    let mut conn = open_db(cfg)?;
    let report = ingest::ingest(&http, cfg, &mut conn, &source, &text)
        .await
        .map_err(map_err)?;
    println!(
        "Ingested {} chunks from {} ({:?})",
        report.chunks, source, report.status
    );
    Ok(())
}

async fn cmd_ask(cfg: &Config, question: &str) -> anyhow::Result<()> {
    let http = llm::http_client(cfg.request_timeout_secs).map_err(map_err)?;
    let conn = open_db(cfg)?;

    match ask::ask(&http, cfg, &conn, question, cfg.top_k).await {
        Ok((answer, hits)) => {
            println!("{answer}\n");
            println!("Sources:");
            for hit in hits {
                let section = hit.section.as_deref().unwrap_or("-");
                println!("  {:.3}  {}::{}", hit.score, hit.source, section);
            }
            Ok(())
        }
        Err(KnowledgeError::NoRelevantContent) => {
            println!("No related content was found in the knowledge base.");
            Ok(())
        }
        Err(e) => Err(map_err(e)),
    }
}

async fn cmd_search(cfg: &Config, query: &str, mode: &str, k: usize) -> anyhow::Result<()> {
    let http = llm::http_client(cfg.request_timeout_secs).map_err(map_err)?;
    let conn = open_db(cfg)?;
    let parsed = search::parse_mode(mode);
    let hits = search::run_search(&http, cfg, &conn, query, parsed, k)
        .await
        .map_err(map_err)?;
    if hits.is_empty() {
        println!("No results.");
    } else {
        search::display(&hits);
    }
    Ok(())
}

fn cmd_list(cfg: &Config) -> anyhow::Result<()> {
    let conn = open_db(cfg)?;
    let docs = db::list_documents(&conn).map_err(map_err)?;
    if docs.is_empty() {
        println!("No documents ingested yet.");
        return Ok(());
    }
    for (id, source, hash) in docs {
        println!("{id}  {source}  {hash}");
    }
    Ok(())
}

fn cmd_rm(cfg: &Config, source: &str) -> anyhow::Result<()> {
    let conn = open_db(cfg)?;
    let removed = db::delete_document(&conn, source).map_err(map_err)?;
    println!("Removed {source}: {removed}");
    Ok(())
}

fn count(conn: &rusqlite::Connection, table: &str) -> anyhow::Result<i64> {
    Ok(conn
        .query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |row| {
            row.get(0)
        })
        .map_err(KnowledgeError::Db)
        .map_err(map_err)?)
}

fn cmd_status(cfg: &Config) -> anyhow::Result<()> {
    let conn = open_db(cfg)?;
    let documents = count(&conn, "documents")?;
    let chunks = count(&conn, "chunks")?;

    println!(
        "db: {} (exists: {})",
        cfg.db_path.display(),
        cfg.db_path.exists()
    );
    println!("documents: {documents}, chunks: {chunks}");
    println!("models dir: {}", cfg.models_dir.display());
    println!(
        "embed model: {} (exists: {})",
        cfg.embed_model,
        cfg.embed_model_path().exists()
    );
    println!(
        "gen model: {} (exists: {})",
        cfg.gen_model,
        cfg.gen_model_path().exists()
    );
    println!(
        "llama-server: {} (exists: {})",
        cfg.llama_server_bin.display(),
        cfg.llama_server_bin.exists()
    );
    println!(
        "sidecars: embeddings :{}, generator :{}",
        cfg.embed_port, cfg.gen_port
    );
    println!(
        "retrieval: top_k = {}, min_score = {}",
        cfg.top_k, cfg.min_score
    );
    Ok(())
}
