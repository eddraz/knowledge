//! `knowledge` — local RAG knowledge base library and CLI.
//!
//! Documents are chunked, embedded with bge-m3 and stored in SQLite with
//! sqlite-vec (semantic KNN) and FTS5 (lexical BM25). The `knowledge` binary
//! answers strictly from retrieved content using a local LFM2.5 generator
//! sidecar; the `native-ask` binary (enabled with the `native` feature)
//! performs generation in-process with candle.

pub mod ask;
pub mod bootstrap;
pub mod chunk;
pub mod config;
pub mod db;
pub mod error;
pub mod ingest;
pub mod llm;
pub mod meta;
pub mod search;
pub mod sidecar;
