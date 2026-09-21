use reqwest::Client;
use rusqlite::{Connection, OptionalExtension};
use sha2::{Digest, Sha256};

use crate::chunk::chunk_text;
use crate::config::Config;
use crate::db::{delete_document, insert_chunk, insert_document, update_document_meta};
use crate::embed::embed_texts;
use crate::error::{KnowledgeError, Result};
use crate::meta::DocMeta;

const EMBED_BATCH: usize = 32;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IngestStatus {
    Ingested,
    Unchanged,
    Replaced,
}

#[derive(Debug, Clone, PartialEq)]
pub struct IngestReport {
    pub status: IngestStatus,
    pub chunks: usize,
}

/// Compute a stable lowercase hex SHA-256 hash of the trimmed text.
pub fn hash_text(text: &str) -> String {
    let trimmed = text.trim();
    let digest = Sha256::digest(trimmed.as_bytes());
    format!("{:x}", digest)
}

/// Decide the ingest outcome based on the existing document hash, if any.
pub fn decide_status(existing: Option<&str>, new_hash: &str) -> IngestStatus {
    match existing {
        Some(old) if old == new_hash => IngestStatus::Unchanged,
        Some(_) => IngestStatus::Replaced,
        None => IngestStatus::Ingested,
    }
}

/// Compute non-overlapping batch ranges for a total number of items.
pub fn batch_ranges(total: usize, batch: usize) -> Vec<(usize, usize)> {
    if total == 0 || batch == 0 {
        return Vec::new();
    }
    let mut ranges = Vec::new();
    let mut start = 0;
    while start < total {
        let end = (start + batch).min(total);
        ranges.push((start, end));
        start = end;
    }
    ranges
}

/// Build the retrievable summary chunk text from generated metadata.
fn summary_text(meta: &DocMeta) -> String {
    let mut parts = Vec::new();

    let title = meta.title.trim();
    if !title.is_empty() {
        parts.push(format!("{title}."));
    }

    let description = meta.description.trim();
    if !description.is_empty() {
        parts.push(description.to_string());
    }

    let keywords: Vec<&str> = meta
        .keywords
        .iter()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();
    if !keywords.is_empty() {
        parts.push(format!("Keywords: {}", keywords.join(", ")));
    }

    parts.join(" ")
}

/// Ingest a document into the knowledge base, chunking, embedding, and storing
/// everything inside a single transaction.
pub async fn ingest(
    http: &Client,
    cfg: &Config,
    conn: &mut Connection,
    source: &str,
    text: &str,
    owner: &str,
    meta: Option<DocMeta>,
) -> Result<IngestReport> {
    let new_hash = hash_text(text);

    let existing: Option<String> = conn
        .query_row(
            "SELECT hash FROM documents WHERE source = ?1",
            [source],
            |row| row.get(0),
        )
        .optional()
        .map_err(KnowledgeError::Db)?;

    let status = decide_status(existing.as_deref(), &new_hash);
    if status == IngestStatus::Unchanged {
        return Ok(IngestReport {
            status: IngestStatus::Unchanged,
            chunks: 0,
        });
    }

    if status == IngestStatus::Replaced {
        delete_document(conn, source)?;
    }

    let mut chunks: Vec<(Option<String>, String)> = Vec::new();
    if let Some(ref m) = meta {
        let summary = summary_text(m);
        if !summary.is_empty() {
            chunks.push((Some("meta".to_string()), summary));
        }
    }
    chunks.extend(chunk_text(
        text,
        cfg.chunk_target_chars,
        cfg.chunk_overlap_chars,
    ));

    if chunks.is_empty() {
        return Err(KnowledgeError::BadResponse("no content".to_string()));
    }

    // Embed all chunk texts in batches.
    let chunk_texts: Vec<String> = chunks.iter().map(|(_, text)| text.clone()).collect();
    let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(chunk_texts.len());
    for (start, end) in batch_ranges(chunk_texts.len(), EMBED_BATCH) {
        let batch = chunk_texts[start..end].to_vec();
        let mut batch_embeddings = embed_texts(http, cfg, &batch).await?;
        embeddings.append(&mut batch_embeddings);
    }

    if embeddings.len() != chunks.len() {
        return Err(KnowledgeError::BadResponse(format!(
            "embedding count mismatch: expected {}, got {}",
            chunks.len(),
            embeddings.len()
        )));
    }

    let tx = conn.transaction().map_err(KnowledgeError::Db)?;
    let doc_id = insert_document(&tx, source, &new_hash, owner)?;
    if let Some(ref m) = meta {
        update_document_meta(&tx, doc_id, m)?;
    }
    for ((section, text), embedding) in chunks.iter().zip(embeddings.iter()) {
        insert_chunk(&tx, doc_id, section.as_deref(), text, embedding)?;
    }
    tx.commit().map_err(KnowledgeError::Db)?;

    Ok(IngestReport {
        status,
        chunks: chunks.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_is_stable_and_changes_with_input() {
        let h1 = hash_text("hola mundo");
        let h2 = hash_text("hola mundo");
        let h3 = hash_text(" hola mundo ");
        let h4 = hash_text("Hola mundo");
        assert_eq!(h1, h2);
        assert_eq!(h1, h3, "hash should trim input");
        assert_ne!(h1, h4);
        assert_eq!(h1.len(), 64);
    }

    #[test]
    fn decide_status_logic() {
        assert_eq!(decide_status(Some("abc"), "abc"), IngestStatus::Unchanged);
        assert_eq!(decide_status(Some("abc"), "def"), IngestStatus::Replaced);
        assert_eq!(decide_status(None, "abc"), IngestStatus::Ingested);
    }

    #[test]
    fn batch_ranges_maths() {
        assert!(batch_ranges(0, 10).is_empty());
        assert!(batch_ranges(5, 0).is_empty());
        assert_eq!(batch_ranges(5, 10), vec![(0, 5)]);
        assert_eq!(batch_ranges(10, 3), vec![(0, 3), (3, 6), (6, 9), (9, 10)]);
        assert_eq!(batch_ranges(32, 32), vec![(0, 32)]);
        assert_eq!(batch_ranges(33, 32), vec![(0, 32), (32, 33)]);
    }

    #[test]
    fn summary_text_skips_empty_parts() {
        let meta = DocMeta {
            title: "".to_string(),
            description: "A description.".to_string(),
            keywords: vec![],
        };
        assert_eq!(summary_text(&meta), "A description.");

        let meta = DocMeta {
            title: "Title".to_string(),
            description: "".to_string(),
            keywords: vec!["rust".to_string(), "".to_string(), "cli".to_string()],
        };
        assert_eq!(summary_text(&meta), "Title. Keywords: rust, cli");
    }
}
