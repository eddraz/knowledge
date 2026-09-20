// Foundation modules contain APIs that are wired but not yet called by the CLI
// commands in this slice; silence dead-code warnings until later tasks use them.
#![allow(dead_code)]

use std::path::PathBuf;

use anyhow::bail;
use clap::{Parser, Subcommand};

use crate::config::Config;

mod config;
mod db;
mod error;
mod llm;
mod sidecar;

#[derive(Parser)]
#[command(name = "knowledge", about = "Local knowledge-base RAG CLI")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Add a document to the knowledge base.
    Add { path: PathBuf },

    /// Ask a question using the knowledge base.
    Ask { question: String },

    /// Search the knowledge base.
    Search {
        query: String,

        /// Number of results to return.
        #[arg(short, long, default_value_t = 5)]
        k: usize,

        /// Search mode: vector, text, or hybrid.
        #[arg(short, long, default_value = "vector")]
        mode: String,
    },

    /// List indexed documents.
    List,

    /// Remove a document from the knowledge base.
    Rm { source: String },

    /// Print configuration summary and verify the database opens.
    Status,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let cfg = Config::load()?;

    // Smoke-check: open the database early so Status reports real state.
    let _conn = db::open(&cfg.db_path)?;

    match cli.command {
        Command::Add { .. } => bail!("not implemented yet"),
        Command::Ask { .. } => bail!("not implemented yet"),
        Command::Search { .. } => bail!("not implemented yet"),
        Command::List => bail!("not implemented yet"),
        Command::Rm { .. } => bail!("not implemented yet"),
        Command::Status => {
            println!("Database path: {}", cfg.db_path.display());
            println!("Models directory: {}", cfg.models_dir.display());
            println!("Embedding port: {}", cfg.embed_port);
            println!("Generator port: {}", cfg.gen_port);
            println!("Database opened successfully");
            Ok(())
        }
    }
}
