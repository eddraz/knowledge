use std::collections::HashMap;

use reqwest::Client;
use rusqlite::Connection;

use crate::config::Config;
use crate::db::{attach_context, fts_search, knn_search, SearchHit, SearchHitContext};
use crate::embed::{embed_texts, EmbedderMode};
use crate::error::Result;
use crate::llm::EMBED_DIM;
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

/// Run a search and enrich each hit with up to `window` adjacent chunks
/// (`N-1`, `N+1`, ...) from the same document.
#[allow(clippy::too_many_arguments)]
pub async fn run_search_with_context(
    http: &Client,
    cfg: &Config,
    conn: &Connection,
    query: &str,
    mode: SearchMode,
    k: usize,
    owner: Option<&str>,
    window: usize,
) -> Result<Vec<SearchHitContext>> {
    let hits = run_search(http, cfg, conn, query, mode, k, owner).await?;
    attach_context(conn, hits, window)
}

async fn vector_search(
    http: &Client,
    cfg: &Config,
    conn: &Connection,
    query: &str,
    k: usize,
    owner: Option<&str>,
) -> Result<Vec<SearchHit>> {
    let _handle = if cfg.embedder_mode == EmbedderMode::Sidecar {
        Some(acquire(cfg, SidecarRole::Embedding).await?)
    } else {
        None
    };
    let embeddings = embed_texts(http, cfg, &[query.to_string()]).await?;
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

fn format_offset_label(offset: isize) -> String {
    if offset < 0 {
        format!("[N{offset}]")
    } else if offset > 0 {
        format!("[N+{offset}]")
    } else {
        "[N]".to_string()
    }
}

fn push_chunk_block(
    out: &mut String,
    label: &str,
    source: &str,
    section: Option<&str>,
    text: &str,
    is_hit: bool,
) {
    let section = section.unwrap_or("");
    let marker = if is_hit { " [hit]" } else { "" };
    out.push_str(&format!("  {label} ({source}::{section}){marker}\n"));
    if text.is_empty() {
        out.push('\n');
        return;
    }
    for line in text.lines() {
        out.push_str("    ");
        out.push_str(line);
        out.push('\n');
    }
}

/// Format search hits together with their adjacent context chunks (`N-1`, `N`, `N+1`).
pub fn format_results_with_context(results: &[SearchHitContext]) -> String {
    let mut out = String::new();
    for (idx, item) in results.iter().enumerate() {
        if idx > 0 {
            out.push('\n');
        }
        let section = item.hit.section.as_deref().unwrap_or("");
        out.push_str(&format!(
            "[{}] {:.4}  {}::{}\n",
            idx + 1,
            item.hit.score,
            item.hit.source,
            section
        ));

        for prev in &item.before {
            let label = format_offset_label(prev.offset);
            push_chunk_block(
                &mut out,
                &label,
                &prev.source,
                prev.section.as_deref(),
                &prev.text,
                false,
            );
        }

        push_chunk_block(
            &mut out,
            "[N]",
            &item.hit.source,
            item.hit.section.as_deref(),
            &item.hit.text,
            true,
        );

        for next in &item.after {
            let label = format_offset_label(next.offset);
            push_chunk_block(
                &mut out,
                &label,
                &next.source,
                next.section.as_deref(),
                &next.text,
                false,
            );
        }
    }
    out
}

/// Print search hits together with their adjacent context chunks (`N-1`, `N`, `N+1`).
pub fn display_with_context(results: &[SearchHitContext]) {
    print!("{}", format_results_with_context(results));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::{insert_chunk, insert_document, open, AdjacentChunk};

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

    #[test]
    fn format_results_with_context_renders_labels_and_full_text() {
        let results = vec![SearchHitContext {
            hit: SearchHit {
                chunk_id: 2,
                score: 0.8123,
                text: "Target line 1\nTarget line 2".to_string(),
                section: Some("Especies".to_string()),
                source: "doc.md".to_string(),
                owner: "_shared".to_string(),
            },
            document_id: 1,
            before: vec![AdjacentChunk {
                chunk_id: 1,
                document_id: 1,
                offset: -1,
                text: "Previous context".to_string(),
                section: Some("Historia".to_string()),
                source: "doc.md".to_string(),
                owner: "_shared".to_string(),
            }],
            after: vec![AdjacentChunk {
                chunk_id: 3,
                document_id: 1,
                offset: 1,
                text: "Next context".to_string(),
                section: Some("Horarios".to_string()),
                source: "doc.md".to_string(),
                owner: "_shared".to_string(),
            }],
        }];

        let rendered = format_results_with_context(&results);
        assert!(rendered.contains("[1] 0.8123  doc.md::Especies"));
        assert!(rendered.contains("  [N-1] (doc.md::Historia)\n    Previous context\n"));
        assert!(rendered
            .contains("  [N] (doc.md::Especies) [hit]\n    Target line 1\n    Target line 2\n"));
        assert!(rendered.contains("  [N+1] (doc.md::Horarios)\n    Next context\n"));
    }

    #[tokio::test]
    async fn run_search_with_context_lexical_returns_adjacent_chunks() {
        let pid = std::process::id();
        let path = std::env::temp_dir().join(format!("knowledge_search_ctx_test_{pid}.db"));
        let _ = std::fs::remove_file(&path);

        let conn = open(&path).unwrap();
        let doc1 = insert_document(&conn, "acuario.md", "h1", "_shared").unwrap();
        let c1 =
            insert_chunk(&conn, doc1, Some("Historia"), "Abrio en 1987", &[0.0; 1024]).unwrap();
        let c2 = insert_chunk(
            &conn,
            doc1,
            Some("Especies"),
            "Pinguinos de Magallanes",
            &[0.0; 1024],
        )
        .unwrap();
        let c3 = insert_chunk(
            &conn,
            doc1,
            Some("Horarios"),
            "Atiende de martes a domingo",
            &[0.0; 1024],
        )
        .unwrap();

        let doc2 = insert_document(&conn, "otro.md", "h2", "_shared").unwrap();
        insert_chunk(
            &conn,
            doc2,
            Some("Otro"),
            "Texto no relacionado",
            &[0.0; 1024],
        )
        .unwrap();

        let cfg = Config::load().unwrap();
        let http = Client::new();
        let results = run_search_with_context(
            &http,
            &cfg,
            &conn,
            "pinguinos",
            SearchMode::Lexical,
            5,
            None,
            1,
        )
        .await
        .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].hit.chunk_id, c2);
        assert_eq!(results[0].before.len(), 1);
        assert_eq!(results[0].before[0].chunk_id, c1);
        assert_eq!(results[0].before[0].offset, -1);
        assert_eq!(results[0].after.len(), 1);
        assert_eq!(results[0].after[0].chunk_id, c3);
        assert_eq!(results[0].after[0].offset, 1);

        drop(conn);
        let _ = std::fs::remove_file(&path);
    }
}
