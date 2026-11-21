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
use knowledge::embed::EmbedderMode;
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
    /// Ingest a text/markdown file or raw string into the knowledge base ("-" reads stdin).
    Add {
        /// Path to file to ingest ("-" reads stdin).
        #[arg(value_name = "PATH")]
        path: Option<String>,

        /// Ingest raw text directly as a string.
        #[arg(short = 't', long = "text")]
        text: Option<String>,

        /// Custom document source name / title (useful with --text or stdin).
        #[arg(long = "name", alias = "title")]
        name: Option<String>,

        /// Generate LLM metadata (title/description/keywords) for the document.
        #[arg(long)]
        meta: bool,
    },

    /// Ask a question; answers strictly from ingested content.
    Ask {
        question: String,
        /// Number of adjacent chunks before and after each hit [default: 1, or KNOWLEDGE_CONTEXT_WINDOW].
        #[arg(long, short = 'w')]
        window: Option<usize>,
        /// Omit adjacent context chunks (equivalent to --window 0).
        #[arg(long)]
        no_context: bool,
    },

    /// Search the knowledge base without generating an answer.
    Search {
        query: String,
        /// Search mode: vector | lexical | hybrid.
        #[arg(long, default_value = "vector")]
        mode: String,
        /// Maximum number of results.
        #[arg(long, short = 'k', default_value = "5")]
        k: usize,
        /// Number of adjacent chunks before and after each hit [default: 1, or KNOWLEDGE_CONTEXT_WINDOW].
        #[arg(long, short = 'w')]
        window: Option<usize>,
        /// Omit adjacent context chunks (equivalent to --window 0).
        #[arg(long)]
        no_context: bool,
    },

    /// List ingested documents.
    List,

    /// List owner namespaces with document and chunk counts.
    Owners,

    /// Remove a document and all of its chunks.
    Rm { source: String },

    /// Change the owner of a document.
    Chown { source: String, owner: String },

    /// Show configuration and database status.
    Status,

    /// Run first-run bootstrap manually and print resolution summary.
    Setup,

    /// Update the knowledge CLI to the latest version.
    Update,
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
        Commands::Add {
            path,
            text,
            name,
            meta,
        } => {
            cmd_add(
                &mut cfg,
                path.as_deref(),
                text.as_deref(),
                name.as_deref(),
                default_owner,
                meta,
                verbose,
            )
            .await?
        }
        Commands::Ask {
            question,
            window,
            no_context,
        } => {
            cfg.context_window = resolve_window(window, no_context, cfg.context_window);
            cmd_ask(&mut cfg, &question, owner, verbose).await?
        }
        Commands::Search {
            query,
            mode,
            k,
            window,
            no_context,
        } => {
            let effective_window = resolve_window(window, no_context, cfg.context_window);
            cmd_search(&mut cfg, &query, &mode, k, effective_window, owner, verbose).await?
        }
        Commands::List => cmd_list(&cfg)?,
        Commands::Owners => cmd_owners(&cfg)?,
        Commands::Rm { source } => cmd_rm(&cfg, &source)?,
        Commands::Chown { source, owner } => cmd_chown(&cfg, &source, &owner)?,
        Commands::Status => cmd_status(&cfg).await?,
        Commands::Setup => cmd_setup(&mut cfg)?,
        Commands::Update => knowledge::update::cmd_update(verbose).map_err(map_err)?,
    }
    Ok(())
}

fn resolve_window(window: Option<usize>, no_context: bool, default: usize) -> usize {
    if no_context {
        0
    } else {
        window.unwrap_or(default)
    }
}

fn map_err(e: KnowledgeError) -> anyhow::Error {
    anyhow::anyhow!(e.to_string())
}

fn open_db(cfg: &Config) -> anyhow::Result<rusqlite::Connection> {
    db::open(&cfg.db_path).map_err(map_err)
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

#[derive(Debug)]
struct IngestInput {
    text: String,
    source: String,
    title: String,
}

fn derive_text_source_and_title(text: &str) -> (String, String) {
    let hash = ingest::hash_text(text);
    let short_hash = &hash[..8];

    for line in text.lines().take(5) {
        let trimmed = line.trim();
        if let Some(header) = trimmed.strip_prefix('#') {
            let header = header.trim_start_matches('#').trim();
            if !header.is_empty() {
                let slug: String = header
                    .chars()
                    .map(|c| {
                        if c.is_alphanumeric() || c == '-' || c == '_' {
                            c
                        } else {
                            '-'
                        }
                    })
                    .collect();
                let clean_slug = slug.trim_matches('-');
                if !clean_slug.is_empty() {
                    return (format!("{clean_slug}-{short_hash}"), header.to_string());
                }
            }
        }
    }

    (
        format!("snippet-{short_hash}"),
        format!("Snippet {short_hash}"),
    )
}

fn resolve_ingest_input(
    path: Option<&str>,
    text_arg: Option<&str>,
    custom_name: Option<&str>,
) -> anyhow::Result<IngestInput> {
    let (raw_text, default_source, default_title) = match (path, text_arg) {
        (Some(_), Some(_)) => {
            anyhow::bail!("Cannot provide both a file PATH and --text / -t; choose one");
        }
        (None, Some(text)) => {
            let trimmed = text.trim();
            if trimmed.is_empty() {
                anyhow::bail!("Text content cannot be empty");
            }
            let (slug, title) = derive_text_source_and_title(trimmed);
            (trimmed.to_string(), slug, title)
        }
        (Some("-"), None) => {
            let mut buf = String::new();
            std::io::stdin().read_to_string(&mut buf)?;
            if buf.trim().is_empty() {
                anyhow::bail!("Stdin content cannot be empty");
            }
            let (slug, title) = derive_text_source_and_title(&buf);
            (buf, slug, title)
        }
        (Some(p), None) => {
            let path_obj = Path::new(p);
            if path_obj.is_file() {
                let content = std::fs::read_to_string(path_obj)?;
                let name = source_name(p);
                let title = source_title(p);
                (content, name, title)
            } else if !path_obj.exists()
                && (p.contains(' ') || p.contains('\n') || p.starts_with('#'))
            {
                let trimmed = p.trim();
                let (slug, title) = derive_text_source_and_title(trimmed);
                (trimmed.to_string(), slug, title)
            } else if !path_obj.exists() {
                anyhow::bail!(
                    "File not found: '{p}'. If you intended to ingest raw text, use --text / -t: knowledge add -t \"{p}\""
                );
            } else {
                anyhow::bail!("Path is not a regular file: '{p}'");
            }
        }
        (None, None) => {
            anyhow::bail!(
                "Nothing to ingest. Specify a file path, '-' for stdin, or --text / -t <CONTENT>"
            );
        }
    };

    let source = custom_name.map(String::from).unwrap_or(default_source);
    let title = custom_name.map(String::from).unwrap_or(default_title);

    Ok(IngestInput {
        text: raw_text,
        source,
        title,
    })
}

async fn cmd_add(
    cfg: &mut Config,
    path: Option<&str>,
    text_arg: Option<&str>,
    name: Option<&str>,
    owner: &str,
    meta_flag: bool,
    verbose: bool,
) -> anyhow::Result<()> {
    bootstrap::ensure_ready(cfg, verbose).map_err(map_err)?;
    let input = resolve_ingest_input(path, text_arg, name)?;
    let text = input.text;
    let source = input.source;
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
            title: input.title,
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

    // `ingest` performs embeddings. In sidecar mode the embedding sidecar
    // must be running; native mode runs in-process and needs no sidecar.
    let _embed_sidecar = if cfg.embedder_mode == EmbedderMode::Sidecar {
        Some(
            sidecar::acquire(cfg, SidecarRole::Embedding)
                .await
                .map_err(map_err)?,
        )
    } else {
        None
    };

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
    window: usize,
    owner: Option<&str>,
    verbose: bool,
) -> anyhow::Result<()> {
    bootstrap::ensure_ready(cfg, verbose).map_err(map_err)?;
    let http = llm::http_client(cfg.request_timeout_secs).map_err(map_err)?;
    let conn = open_db(cfg)?;
    let parsed = search::parse_mode(mode);
    if window == 0 {
        let hits = search::run_search(&http, cfg, &conn, query, parsed, k, owner)
            .await
            .map_err(map_err)?;
        if hits.is_empty() {
            println!("No results.");
        } else {
            search::display(&hits);
        }
    } else {
        let results =
            search::run_search_with_context(&http, cfg, &conn, query, parsed, k, owner, window)
                .await
                .map_err(map_err)?;
        if results.is_empty() {
            println!("No results.");
        } else {
            search::display_with_context(&results);
        }
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

fn cmd_owners(cfg: &Config) -> anyhow::Result<()> {
    let conn = open_db(cfg)?;
    let owners = db::list_owners(&conn).map_err(map_err)?;
    if owners.is_empty() {
        println!("No documents ingested yet.");
        return Ok(());
    }
    for row in owners {
        println!(
            "{owner:<20} {documents:>6} docs  {chunks:>6} chunks",
            owner = row.owner,
            documents = row.documents,
            chunks = row.chunks,
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
    conn.query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |row| {
        row.get(0)
    })
    .map_err(KnowledgeError::Db)
    .map_err(map_err)
}

async fn cmd_status(cfg: &Config) -> anyhow::Result<()> {
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

    match cfg.embedder_mode {
        EmbedderMode::Hybrid => {
            let daemon_online = sidecar::probe_health_fast(&cfg.embed_base_url()).await;
            let status = if daemon_online {
                "active (daemon)"
            } else {
                "active (candle fallback)"
            };
            println!("embedder: hybrid [{status}]");
            println!(
                "  endpoint: {} (responding: {daemon_online})",
                cfg.embed_base_url()
            );
            println!(
                "  candle config: {} (exists: {})",
                cfg.native_embed_config_path().display(),
                cfg.native_embed_config_path().exists()
            );
            println!(
                "  candle tokenizer: {} (exists: {})",
                cfg.native_embed_tokenizer_path().display(),
                cfg.native_embed_tokenizer_path().exists()
            );
            println!(
                "  candle weights: {} (exists: {})",
                cfg.native_embed_weights_path().display(),
                cfg.native_embed_weights_path().exists()
            );
        }
        EmbedderMode::Native => {
            println!("embedder: native candle bge-m3");
            println!(
                "  config: {} (exists: {})",
                cfg.native_embed_config_path().display(),
                cfg.native_embed_config_path().exists()
            );
            println!(
                "  tokenizer: {} (exists: {})",
                cfg.native_embed_tokenizer_path().display(),
                cfg.native_embed_tokenizer_path().exists()
            );
            println!(
                "  weights: {} (exists: {})",
                cfg.native_embed_weights_path().display(),
                cfg.native_embed_weights_path().exists()
            );
        }
        EmbedderMode::Sidecar => {
            println!("embedder: sidecar");
            println!(
                "embed model: {} (exists: {})",
                cfg.embed_model,
                cfg.embed_model_path().exists()
            );
        }
    }

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
        "retrieval: top_k = {}, min_score = {}, context_window = {}",
        cfg.top_k, cfg.min_score, cfg.context_window
    );
    Ok(())
}

fn cmd_setup(cfg: &mut Config) -> anyhow::Result<()> {
    bootstrap::ensure_ready(cfg, true).map_err(map_err)?;
    if cfg.embedder_mode == EmbedderMode::Sidecar || !cfg!(feature = "native") {
        println!("llama-server: {}", cfg.llama_server_bin.display());
    }
    match cfg.embedder_mode {
        EmbedderMode::Hybrid => {
            println!(
                "embedder: hybrid (endpoint on :{} if active, candle fallback)",
                cfg.embed_port
            );
            println!("  endpoint: {}", cfg.embed_base_url());
            println!(
                "  candle config: {}",
                cfg.native_embed_config_path().display()
            );
            println!(
                "  candle tokenizer: {}",
                cfg.native_embed_tokenizer_path().display()
            );
            println!(
                "  candle weights: {}",
                cfg.native_embed_weights_path().display()
            );
        }
        EmbedderMode::Native => {
            println!("embedder: native candle bge-m3");
            println!("  config: {}", cfg.native_embed_config_path().display());
            println!(
                "  tokenizer: {}",
                cfg.native_embed_tokenizer_path().display()
            );
            println!("  weights: {}", cfg.native_embed_weights_path().display());
        }
        EmbedderMode::Sidecar => {
            println!("embed model: {}", cfg.embed_model_path().display());
        }
    }
    println!("gen model: {}", cfg.gen_model_path().display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_ingest_input_with_text_flag() {
        let input = resolve_ingest_input(None, Some("hello world"), None).unwrap();
        assert_eq!(input.text, "hello world");
        assert!(input.source.starts_with("snippet-"));
        assert!(input.title.starts_with("Snippet "));

        let input_named =
            resolve_ingest_input(None, Some("hello world"), Some("greeting")).unwrap();
        assert_eq!(input_named.text, "hello world");
        assert_eq!(input_named.source, "greeting");
        assert_eq!(input_named.title, "greeting");
    }

    #[test]
    fn resolve_ingest_input_extracts_markdown_header_slug() {
        let input = resolve_ingest_input(
            None,
            Some("# Quick Guide\nHere are some instructions."),
            None,
        )
        .unwrap();
        assert!(input.source.starts_with("Quick-Guide-"));
        assert_eq!(input.title, "Quick Guide");
    }

    #[test]
    fn resolve_ingest_input_inline_string_without_flag() {
        let input = resolve_ingest_input(Some("this is a quick note"), None, None).unwrap();
        assert_eq!(input.text, "this is a quick note");
        assert!(input.source.starts_with("snippet-"));
    }

    #[test]
    fn resolve_ingest_input_rejects_missing_file_with_hint() {
        let err = resolve_ingest_input(Some("nonexistent_file.md"), None, None).unwrap_err();
        assert!(err.to_string().contains("File not found"));
        assert!(err.to_string().contains("--text / -t"));
    }

    #[test]
    fn resolve_ingest_input_rejects_both_or_neither() {
        let err_both = resolve_ingest_input(Some("path"), Some("text"), None).unwrap_err();
        assert!(err_both.to_string().contains("Cannot provide both"));

        let err_neither = resolve_ingest_input(None, None, None).unwrap_err();
        assert!(err_neither.to_string().contains("Nothing to ingest"));
    }

    #[test]
    fn resolve_window_handles_defaults_overrides_and_no_context() {
        assert_eq!(resolve_window(None, false, 1), 1);
        assert_eq!(resolve_window(Some(2), false, 1), 2);
        assert_eq!(resolve_window(Some(0), false, 1), 0);
        assert_eq!(resolve_window(None, true, 1), 0);
        assert_eq!(resolve_window(Some(3), true, 1), 0);
    }

    #[test]
    fn cli_parses_search_and_ask_context_flags() {
        let cli = Cli::try_parse_from(["knowledge", "search", "hola"]).unwrap();
        match cli.command {
            Commands::Search {
                window, no_context, ..
            } => {
                assert_eq!(window, None);
                assert!(!no_context);
            }
            _ => panic!("expected Search"),
        }

        let cli = Cli::try_parse_from(["knowledge", "search", "hola", "-w", "2", "--no-context"])
            .unwrap();
        match cli.command {
            Commands::Search {
                window, no_context, ..
            } => {
                assert_eq!(window, Some(2));
                assert!(no_context);
            }
            _ => panic!("expected Search"),
        }

        let cli = Cli::try_parse_from(["knowledge", "ask", "hola", "--window", "0"]).unwrap();
        match cli.command {
            Commands::Ask {
                window, no_context, ..
            } => {
                assert_eq!(window, Some(0));
                assert!(!no_context);
            }
            _ => panic!("expected Ask"),
        }
    }
}
