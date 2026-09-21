//! `knowledge` — local RAG knowledge base CLI.
//!
//! Documents are chunked, embedded with bge-m3 and stored in SQLite with
//! sqlite-vec (semantic KNN) and FTS5 (lexical BM25). `ask` answers strictly
//! from retrieved content using a local LFM2.5 generator sidecar.

use std::io::Read;
use std::path::Path;

use clap::{Parser, Subcommand};

use knowledge::ask;
use knowledge::bootstrap;
use knowledge::config::Config;
use knowledge::db;
use knowledge::error::KnowledgeError;
use knowledge::ingest;
use knowledge::llm;
use knowledge::meta;
use knowledge::meta::DocMeta;
use knowledge::search;
use knowledge::sidecar::{self, SidecarRole};

#[derive(Parser)]
#[command(
    name = "knowledge",
    about = "Local RAG knowledge base: semantic + lexical search over your own documents",
    version
)]
struct Cli {
    /// Print bootstrap progress even when nothing needs to be installed.
    #[arg(long, global = true)]
    verbose: bool,

    /// Owner namespace: narrows searches to this owner (plus _shared) and
    /// becomes the default owner for new documents. Omit to search everything.
    #[arg(long, global = true)]
    owner: Option<String>,

    /// Disable owner filtering (search across all owners).
    #[arg(long, global = true)]
    all: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Ingest a text/markdown file into the knowledge base ("-" reads stdin).
    Add {
        path: String,
        /// Generate LLM metadata (title/description/keywords) for the document.
        #[arg(long)]
        meta: bool,
    },

    /// Ask a question; answers strictly from ingested content.
    Ask { question: String },

    /// Search the knowledge base without generating an answer.
    Search {
        query: String,
        /// Search mode: vector | lexical | hybrid.
        #[arg(long, default_value = "vector")]
        mode: String,
        /// Maximum number of results.
        #[arg(long, short = 'k', default_value = "5")]
        k: usize,
    },

    /// List ingested documents.
    List,

    /// Remove a document and all of its chunks.
    Rm { source: String },

    /// Change the owner of a document.
    Chown { source: String, owner: String },

    /// Show configuration and database status.
    Status,

    /// Run first-run bootstrap manually and print resolution summary.
    Setup,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let mut cfg = Config::load().map_err(map_err)?;
    let verbose = cli.verbose;
    // No --owner: search everything (pronouns like "my" need an explicit
    // --owner to be meaningful). --all is an explicit synonym.
    let owner = if cli.all { None } else { cli.owner.as_deref() };
    let default_owner = cli.owner.as_deref().unwrap_or("_shared");

    match cli.command {
        Commands::Add { path, meta } => {
            cmd_add(&mut cfg, &path, default_owner, meta, verbose).await?
        }
        Commands::Ask { question } => cmd_ask(&mut cfg, &question, owner, verbose).await?,
        Commands::Search { query, mode, k } => {
            cmd_search(&mut cfg, &query, &mode, k, owner, verbose).await?
        }
        Commands::List => cmd_list(&cfg)?,
        Commands::Rm { source } => cmd_rm(&cfg, &source)?,
        Commands::Chown { source, owner } => cmd_chown(&cfg, &source, &owner)?,
        Commands::Status => cmd_status(&cfg)?,
        Commands::Setup => cmd_setup(&mut cfg)?,
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
    Path::new(path)
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string())
}

fn source_title(path: &str) -> String {
    if path == "-" {
        return "stdin".to_string();
    }
    Path::new(path)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string())
}

async fn cmd_add(
    cfg: &mut Config,
    path: &str,
    owner: &str,
    meta_flag: bool,
    verbose: bool,
) -> anyhow::Result<()> {
    bootstrap::ensure_ready(cfg, verbose).map_err(map_err)?;
    let text = read_input(path)?;
    let source = source_name(path);
    let http = llm::http_client(cfg.request_timeout_secs).map_err(map_err)?;

    let meta = if meta_flag {
        // With the `native` feature, metadata generation runs in-process via
        // candle and no generator sidecar is spawned at all.
        #[cfg(not(feature = "native"))]
        let _gen_sidecar = sidecar::acquire(cfg, SidecarRole::Generator)
            .await
            .map_err(map_err)?;
        #[cfg(feature = "native")]
        let _gen_sidecar = ();
        let generated = meta::generate(&http, &cfg.gen_base_url(), &cfg.gen_model, &text).await;
        let fallback = DocMeta {
            title: source_title(path),
            description: String::new(),
            keywords: Vec::new(),
        };
        let meta = generated.unwrap_or(fallback);

        let keywords = meta.keywords.join(", ");
        if keywords.is_empty() {
            println!("Metadata: {}", meta.title);
        } else {
            println!("Metadata: {} — {}", meta.title, keywords);
        }
        Some(meta)
    } else {
        None
    };

    // `ingest` performs embeddings; make sure the embedding sidecar is up.
    let _embed_sidecar = sidecar::acquire(cfg, SidecarRole::Embedding)
        .await
        .map_err(map_err)?;

    let mut conn = open_db(cfg)?;
    let report = ingest::ingest(&http, cfg, &mut conn, &source, &text, owner, meta)
        .await
        .map_err(map_err)?;
    println!(
        "Ingested {} chunks from {} ({:?})",
        report.chunks, source, report.status
    );
    Ok(())
}

async fn cmd_ask(
    cfg: &mut Config,
    question: &str,
    owner: Option<&str>,
    verbose: bool,
) -> anyhow::Result<()> {
    bootstrap::ensure_ready(cfg, verbose).map_err(map_err)?;
    let http = llm::http_client(cfg.request_timeout_secs).map_err(map_err)?;
    let conn = open_db(cfg)?;

    match ask::ask(&http, cfg, &conn, question, cfg.top_k, owner).await {
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

async fn cmd_search(
    cfg: &mut Config,
    query: &str,
    mode: &str,
    k: usize,
    owner: Option<&str>,
    verbose: bool,
) -> anyhow::Result<()> {
    bootstrap::ensure_ready(cfg, verbose).map_err(map_err)?;
    let http = llm::http_client(cfg.request_timeout_secs).map_err(map_err)?;
    let conn = open_db(cfg)?;
    let parsed = search::parse_mode(mode);
    let hits = search::run_search(&http, cfg, &conn, query, parsed, k, owner)
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
    for doc in docs {
        let title_part = doc
            .title
            .as_deref()
            .filter(|t| !t.is_empty())
            .map(|t| format!("  [{t}]"))
            .unwrap_or_default();
        let keywords_part = doc
            .keywords
            .as_deref()
            .filter(|k| !k.is_empty())
            .map(|k| format!("  {k}"))
            .unwrap_or_default();
        println!(
            "{}  {}  {}{}{}",
            doc.id, doc.owner, doc.source, title_part, keywords_part
        );
    }
    Ok(())
}

fn cmd_rm(cfg: &Config, source: &str) -> anyhow::Result<()> {
    let conn = open_db(cfg)?;
    let removed = db::delete_document(&conn, source).map_err(map_err)?;
    println!("Removed {source}: {removed}");
    Ok(())
}

fn cmd_chown(cfg: &Config, source: &str, owner: &str) -> anyhow::Result<()> {
    let conn = open_db(cfg)?;
    let changed = db::set_owner(&conn, source, owner).map_err(map_err)?;
    println!("Chown {source} -> {owner}: {changed}");
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
    let fork_path = bootstrap::fork_bin_path(&cfg.apps_dir);
    println!(
        "fork build: {} (exists: {})",
        fork_path.display(),
        fork_path.exists()
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

fn cmd_setup(cfg: &mut Config) -> anyhow::Result<()> {
    bootstrap::ensure_ready(cfg, true).map_err(map_err)?;
    println!("llama-server: {}", cfg.llama_server_bin.display());
    println!("embed model: {}", cfg.embed_model_path().display());
    println!("gen model: {}", cfg.gen_model_path().display());
    Ok(())
}
