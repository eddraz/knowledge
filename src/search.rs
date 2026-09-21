use std::collections::HashMap;

use reqwest::Client;
use rusqlite::Connection;

use crate::config::Config;
use crate::db::{fts_search, knn_search, SearchHit};
use crate::error::Result;
use crate::llm::{embed_texts, EMBED_DIM};
use crate::sidecar::{acquire, SidecarRole};

const RRF_K: f64 = 60.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchMode {
    Vector,
    Lexical,
    Hybrid,
}

/// Parse a search mode string.  Defaults to `Vector` for any unknown value.
pub fn parse_mode(input: &str) -> SearchMode {
    match input.to_lowercase().as_str() {
        "lexical" | "text" | "fts" => SearchMode::Lexical,
        "hybrid" => SearchMode::Hybrid,
        _ => SearchMode::Vector,
    }
}

/// Run a search using the requested mode.
///
/// * `Vector`: embed the query and perform a KNN search via sqlite-vec.
/// * `Lexical`: perform an FTS5 search directly (no sidecar required).
/// * `Hybrid`: run both vector and lexical searches and fuse the results with
///   Reciprocal Rank Fusion (RRF).  The returned `score` field is the RRF
///   score, which is higher-is-better and has no fixed upper bound.
pub async fn run_search(
    http: &Client,
    cfg: &Config,
    conn: &Connection,
    query: &str,
    mode: SearchMode,
    k: usize,
    owner: Option<&str>,
) -> Result<Vec<SearchHit>> {
    match mode {
        SearchMode::Vector => vector_search(http, cfg, conn, query, k, owner).await,
        SearchMode::Lexical => fts_search(conn, query, k, owner),
        SearchMode::Hybrid => {
            let mut vector_hits = vector_search(http, cfg, conn, query, k, owner).await?;
            let mut lexical_hits = fts_search(conn, query, k, owner)?;
            Ok(fuse_rrf(&mut vector_hits, &mut lexical_hits, k))
        }
    }
}

async fn vector_search(
    http: &Client,
    cfg: &Config,
    conn: &Connection,
    query: &str,
    k: usize,
    owner: Option<&str>,
) -> Result<Vec<SearchHit>> {
    let _handle = acquire(cfg, SidecarRole::Embedding).await?;
    let base_url = cfg.embed_base_url();
    let embeddings = embed_texts(http, &base_url, &[query.to_string()]).await?;
    let query_vec = embeddings
        .into_iter()
        .next()
        .unwrap_or_else(|| vec![0.0_f32; EMBED_DIM]);
    knn_search(conn, &query_vec, k, owner)
}

/// Fuse two ranked lists using Reciprocal Rank Fusion.
///
/// For each hit present in one or both lists, its RRF score is the sum over
/// every list that contains it of `1.0 / (60.0 + position)`, where position is
/// 1-based.  Hits are keyed by `chunk_id`.  The final vector is sorted by RRF
/// score descending and truncated to `k` items.
fn fuse_rrf(vector: &mut [SearchHit], lexical: &mut [SearchHit], k: usize) -> Vec<SearchHit> {
    let mut scores: HashMap<i64, f64> = HashMap::new();

    for (pos, hit) in vector.iter().enumerate() {
        let rank = pos + 1;
        *scores.entry(hit.chunk_id).or_insert(0.0) += 1.0 / (RRF_K + rank as f64);
    }

    for (pos, hit) in lexical.iter().enumerate() {
        let rank = pos + 1;
        *scores.entry(hit.chunk_id).or_insert(0.0) += 1.0 / (RRF_K + rank as f64);
    }

    // Collect one representative SearchHit per chunk_id with the fused score.
    let mut by_id: HashMap<i64, SearchHit> = HashMap::new();
    for hit in vector.iter().chain(lexical.iter()) {
        by_id.entry(hit.chunk_id).or_insert_with(|| SearchHit {
            chunk_id: hit.chunk_id,
            score: 0.0,
            text: hit.text.clone(),
            section: hit.section.clone(),
            source: hit.source.clone(),
            owner: hit.owner.clone(),
        });
    }

    let mut fused: Vec<SearchHit> = scores
        .into_iter()
        .map(|(chunk_id, score)| {
            let mut hit = by_id.remove(&chunk_id).unwrap_or_else(|| SearchHit {
                chunk_id,
                score: 0.0,
                text: String::new(),
                section: None,
                source: String::new(),
                owner: String::new(),
            });
            hit.score = score;
            hit
        })
        .collect();

    fused.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    fused.truncate(k);
    fused
}

/// Print search hits, one per line.
pub fn display(hits: &[SearchHit]) {
    for hit in hits {
        let section = hit.section.as_deref().unwrap_or("");
        let excerpt = hit
            .text
            .lines()
            .next()
            .unwrap_or("")
            .chars()
            .take(90)
            .collect::<String>();
        println!(
            "{:.4}  {}::{}  {}",
            hit.score,
            hit.source,
            section,
            excerpt.replace('\n', " ")
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hit(chunk_id: i64, score: f64) -> SearchHit {
        SearchHit {
            chunk_id,
            score,
            text: format!("chunk {chunk_id}"),
            section: None,
            source: "test".to_string(),
            owner: String::new(),
        }
    }

    #[test]
    fn parse_mode_defaults_and_variants() {
        assert_eq!(parse_mode("vector"), SearchMode::Vector);
        assert_eq!(parse_mode("VECTOR"), SearchMode::Vector);
        assert_eq!(parse_mode("lexical"), SearchMode::Lexical);
        assert_eq!(parse_mode("text"), SearchMode::Lexical);
        assert_eq!(parse_mode("fts"), SearchMode::Lexical);
        assert_eq!(parse_mode("hybrid"), SearchMode::Hybrid);
        assert_eq!(parse_mode("unknown"), SearchMode::Vector);
    }

    #[test]
    fn rrf_fusion_prefers_hit_in_both_lists() {
        let vector = vec![hit(1, 1.0), hit(2, 0.9), hit(3, 0.8)];
        let lexical = vec![hit(3, 5.0), hit(4, 4.0)];
        let fused = fuse_rrf(&mut vector.clone(), &mut lexical.clone(), 10);

        // chunk 3 appears in both lists: rank 3 in vector, rank 1 in lexical.
        let score_3 = 1.0 / (RRF_K + 3.0) + 1.0 / (RRF_K + 1.0);
        let score_1 = 1.0 / (RRF_K + 1.0);
        let score_2 = 1.0 / (RRF_K + 2.0);
        let score_4 = 1.0 / (RRF_K + 2.0);

        let by_id: HashMap<i64, f64> = fused.iter().map(|h| (h.chunk_id, h.score)).collect();
        assert!((by_id[&3] - score_3).abs() < 1e-9);
        assert!((by_id[&1] - score_1).abs() < 1e-9);
        assert!((by_id[&2] - score_2).abs() < 1e-4);
        assert!((by_id[&4] - score_4).abs() < 1e-4);

        assert_eq!(fused[0].chunk_id, 3, "chunk in both lists should win");
    }

    #[test]
    fn rrf_truncates_to_k() {
        let vector = vec![hit(1, 1.0), hit(2, 0.9)];
        let lexical = vec![hit(3, 1.0), hit(4, 0.9)];
        let fused = fuse_rrf(&mut vector.clone(), &mut lexical.clone(), 2);
        assert_eq!(fused.len(), 2);
    }
}
