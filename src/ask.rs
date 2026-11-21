use std::collections::{BTreeMap, HashSet};

use reqwest::Client;
use rusqlite::Connection;

use crate::config::Config;
use crate::db::{attach_context, SearchHit, SearchHitContext};
use crate::error::{KnowledgeError, Result};
use crate::search::{run_search, SearchMode};

#[cfg(not(feature = "native"))]
use crate::sidecar::{acquire, SidecarRole};
#[cfg(feature = "native")]
use crate::{bootstrap, native};

const GROUNDED_SYSTEM_PROMPT: &str = "You are a retrieval-grounded assistant. Answer the question using ONLY the context excerpts below. Synthesize the relevant information into a direct answer in your own words. If the context does not contain enough information, reply exactly that the information is not in the knowledge base. Do not use outside knowledge. Do not mention the excerpt numbers or the word Sources in the reply. Answer in the same language as the question.";

fn push_excerpt(
    context: &mut String,
    num: usize,
    source: &str,
    section: Option<&str>,
    text: &str,
    max_chunk_chars: usize,
) {
    let section = section.unwrap_or("");
    let text = truncate_text(text, max_chunk_chars);
    context.push_str(&format!("[{num}] ({source}::{section})\n{text}\n\n"));
}

/// Build the system and user prompts for a grounded Q&A turn.
///
/// The system prompt instructs the model to answer only from the numbered
/// excerpts and to cite them.  The user prompt contains the truncated excerpt
/// blocks followed by the question.
pub fn build_prompt(
    question: &str,
    hits: &[SearchHit],
    max_chunk_chars: usize,
) -> (String, String) {
    let mut context = String::new();
    for (i, hit) in hits.iter().enumerate() {
        push_excerpt(
            &mut context,
            i + 1,
            &hit.source,
            hit.section.as_deref(),
            &hit.text,
            max_chunk_chars,
        );
    }

    let user = format!("{context}\nQuestion: {question}");
    (GROUNDED_SYSTEM_PROMPT.to_string(), user)
}

struct PromptChunk<'a> {
    chunk_id: i64,
    source: &'a str,
    section: Option<&'a str>,
    text: &'a str,
}

fn collect_prompt_chunks<'a>(hits: &'a [SearchHitContext]) -> Vec<PromptChunk<'a>> {
    struct Cluster<'a> {
        document_id: i64,
        chunks: BTreeMap<i64, PromptChunk<'a>>,
    }

    enum Entry<'a> {
        Cluster(Cluster<'a>),
        Raw(Vec<PromptChunk<'a>>),
    }

    let mut entries: Vec<Entry<'a>> = Vec::new();

    for ctx in hits {
        let mut window_chunks: Vec<PromptChunk<'a>> =
            Vec::with_capacity(ctx.before.len() + 1 + ctx.after.len());
        for adj in &ctx.before {
            window_chunks.push(PromptChunk {
                chunk_id: adj.chunk_id,
                source: &adj.source,
                section: adj.section.as_deref(),
                text: &adj.text,
            });
        }
        window_chunks.push(PromptChunk {
            chunk_id: ctx.hit.chunk_id,
            source: &ctx.hit.source,
            section: ctx.hit.section.as_deref(),
            text: &ctx.hit.text,
        });
        for adj in &ctx.after {
            window_chunks.push(PromptChunk {
                chunk_id: adj.chunk_id,
                source: &adj.source,
                section: adj.section.as_deref(),
                text: &adj.text,
            });
        }

        if ctx.document_id <= 0 || window_chunks.iter().any(|c| c.chunk_id <= 0) {
            entries.push(Entry::Raw(window_chunks));
            continue;
        }

        let window_ids: HashSet<i64> = window_chunks.iter().map(|c| c.chunk_id).collect();
        let matching_indices: Vec<usize> = entries
            .iter()
            .enumerate()
            .filter_map(|(idx, entry)| match entry {
                Entry::Cluster(cluster)
                    if cluster.document_id == ctx.document_id
                        && cluster.chunks.keys().any(|id| window_ids.contains(id)) =>
                {
                    Some(idx)
                }
                _ => None,
            })
            .collect();

        if let Some(&first_idx) = matching_indices.first() {
            // Merge any additional bridged clusters in reverse index order so
            // earlier indices remain valid and rank order is preserved.
            for &idx in matching_indices[1..].iter().rev() {
                if let Entry::Cluster(removed) = entries.remove(idx) {
                    if let Entry::Cluster(ref mut target) = entries[first_idx] {
                        target.chunks.extend(removed.chunks);
                    }
                }
            }
            if let Entry::Cluster(ref mut target) = entries[first_idx] {
                for chunk in window_chunks {
                    target.chunks.insert(chunk.chunk_id, chunk);
                }
            }
        } else {
            let mut chunks = BTreeMap::new();
            for chunk in window_chunks {
                chunks.insert(chunk.chunk_id, chunk);
            }
            entries.push(Entry::Cluster(Cluster {
                document_id: ctx.document_id,
                chunks,
            }));
        }
    }

    let mut flattened = Vec::new();
    for entry in entries {
        match entry {
            Entry::Cluster(cluster) => flattened.extend(cluster.chunks.into_values()),
            Entry::Raw(chunks) => flattened.extend(chunks),
        }
    }
    flattened
}

/// Build the system and user prompts from search hits enriched with adjacent
/// chunks (`N-1`, `N+1`, ...).
///
/// Overlapping windows within the same document are deduplicated and merged in
/// reading order while keeping the highest-ranked passage first. Each chunk is
/// truncated individually to `max_chunk_chars`.
pub fn build_prompt_with_context(
    question: &str,
    hits: &[SearchHitContext],
    max_chunk_chars: usize,
) -> (String, String) {
    let chunks = collect_prompt_chunks(hits);
    let mut context = String::new();
    for (i, chunk) in chunks.iter().enumerate() {
        push_excerpt(
            &mut context,
            i + 1,
            chunk.source,
            chunk.section,
            chunk.text,
            max_chunk_chars,
        );
    }

    let user = format!("{context}\nQuestion: {question}");
    (GROUNDED_SYSTEM_PROMPT.to_string(), user)
}

fn truncate_text(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    text.chars().take(max_chars).collect::<String>() + "…"
}

/// Pure gate check: is the top hit good enough to answer from?
pub fn passes_gate(best: Option<f32>, min: f32) -> bool {
    match best {
        Some(score) => score >= min,
        None => false,
    }
}

/// Ask a question grounded in the knowledge base.
///
/// 1. Runs a vector search for the question.
/// 2. Applies the configured minimum score gate.
/// 3. Builds a citation prompt from the retrieved chunks.
/// 4. Calls the generator sidecar.
/// 5. Returns the answer and the hits that grounded it.
pub async fn ask(
    http: &Client,
    cfg: &Config,
    conn: &Connection,
    question: &str,
    k: usize,
    owner: Option<&str>,
) -> Result<(String, Vec<SearchHit>)> {
    let hits = run_search(http, cfg, conn, question, SearchMode::Vector, k, owner).await?;

    let best = hits.first().map(|h| h.score as f32);
    if !passes_gate(best, cfg.min_score) {
        return Err(KnowledgeError::NoRelevantContent);
    }

    let contextual = attach_context(conn, hits, cfg.context_window)?;
    let (system, user) = build_prompt_with_context(question, &contextual, 1200);
    let hits: Vec<SearchHit> = contextual.into_iter().map(|c| c.hit).collect();

    #[cfg(feature = "native")]
    let answer = {
        let model_path = cfg.gen_model_path();
        let tokenizer_path = bootstrap::ensure_registry_file(
            &cfg.models_dir,
            native::DEFAULT_TOKENIZER_NAME,
            false,
        )?;
        let (answer, _stats) = native::generate(&model_path, &tokenizer_path, &system, &user, 512)?;
        answer
    };

    #[cfg(not(feature = "native"))]
    let answer = {
        let _handle = acquire(cfg, SidecarRole::Generator).await?;
        crate::llm::generate(
            http,
            &cfg.gen_base_url(),
            &cfg.gen_model,
            &system,
            &user,
            512,
        )
        .await?
    };

    Ok((answer, hits))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::AdjacentChunk;

    fn hit(text: &str, source: &str, section: Option<&str>) -> SearchHit {
        SearchHit {
            chunk_id: 0,
            score: 0.0,
            text: text.to_string(),
            section: section.map(String::from),
            source: source.to_string(),
            owner: String::new(),
        }
    }

    fn adj(
        chunk_id: i64,
        document_id: i64,
        offset: isize,
        text: &str,
        source: &str,
    ) -> AdjacentChunk {
        AdjacentChunk {
            chunk_id,
            document_id,
            offset,
            text: text.to_string(),
            section: Some(format!("Sec{chunk_id}")),
            source: source.to_string(),
            owner: String::new(),
        }
    }

    fn ctx(
        chunk_id: i64,
        document_id: i64,
        text: &str,
        source: &str,
        before: Vec<AdjacentChunk>,
        after: Vec<AdjacentChunk>,
    ) -> SearchHitContext {
        SearchHitContext {
            hit: SearchHit {
                chunk_id,
                score: 0.8,
                text: text.to_string(),
                section: Some(format!("Sec{chunk_id}")),
                source: source.to_string(),
                owner: String::new(),
            },
            document_id,
            before,
            after,
        }
    }

    #[test]
    fn build_prompt_contains_context_and_question() {
        let hits = vec![
            hit("first chunk", "src1", Some("Intro")),
            hit("second chunk", "src2", None),
        ];
        let (system, user) = build_prompt("¿Qué es esto?", &hits, 1000);

        assert!(system.contains("retrieval-grounded"));
        assert!(system.contains("Synthesize the relevant information"));
        assert!(system.contains("Do not mention the excerpt numbers"));
        assert!(!system.contains("Sources: [1] [3]"));
        assert!(user.contains("[1] (src1::Intro)\nfirst chunk"));
        assert!(user.contains("[2] (src2::)\nsecond chunk"));
        assert!(user.contains("Question: ¿Qué es esto?"));
    }

    #[test]
    fn build_prompt_truncates_long_chunks() {
        let long = "a".repeat(2000);
        let hits = vec![hit(&long, "src", None)];
        let (_, user) = build_prompt("q", &hits, 100);
        assert!(user.contains("aaaa…"));
        assert!(!user.contains(&"a".repeat(101)));
    }

    #[test]
    fn build_prompt_with_context_includes_adjacent_chunks_in_reading_order() {
        let hits = vec![ctx(
            2,
            10,
            "chunk two",
            "doc.md",
            vec![adj(1, 10, -1, "chunk one", "doc.md")],
            vec![adj(3, 10, 1, "chunk three", "doc.md")],
        )];
        let (_, user) = build_prompt_with_context("q?", &hits, 1000);
        assert!(user.contains("[1] (doc.md::Sec1)\nchunk one"));
        assert!(user.contains("[2] (doc.md::Sec2)\nchunk two"));
        assert!(user.contains("[3] (doc.md::Sec3)\nchunk three"));
    }

    #[test]
    fn build_prompt_with_context_deduplicates_and_merges_overlapping_windows() {
        // Hit 1 (higher ranked) is chunk 6 [5, 6, 7]; Hit 2 is chunk 5 [4, 5, 6].
        // They share 5 and 6 so they must merge into [4, 5, 6, 7] in reading order
        // at Hit 1's position, followed by Hit 3 in another document.
        let hits = vec![
            ctx(
                6,
                1,
                "c6",
                "a.md",
                vec![adj(5, 1, -1, "c5", "a.md")],
                vec![adj(7, 1, 1, "c7", "a.md")],
            ),
            ctx(
                20,
                2,
                "c20",
                "b.md",
                vec![adj(19, 2, -1, "c19", "b.md")],
                vec![adj(21, 2, 1, "c21", "b.md")],
            ),
            ctx(
                5,
                1,
                "c5",
                "a.md",
                vec![adj(4, 1, -1, "c4", "a.md")],
                vec![adj(6, 1, 1, "c6", "a.md")],
            ),
        ];

        let (_, user) = build_prompt_with_context("q", &hits, 1000);
        let expected = "\
[1] (a.md::Sec4)\nc4\n\n\
[2] (a.md::Sec5)\nc5\n\n\
[3] (a.md::Sec6)\nc6\n\n\
[4] (a.md::Sec7)\nc7\n\n\
[5] (b.md::Sec19)\nc19\n\n\
[6] (b.md::Sec20)\nc20\n\n\
[7] (b.md::Sec21)\nc21\n\n\n\
Question: q";
        assert_eq!(user, expected);
    }

    #[test]
    fn build_prompt_with_context_truncates_each_chunk_independently() {
        let long_prev = "p".repeat(200);
        let long_hit = "h".repeat(200);
        let long_next = "n".repeat(200);
        let hits = vec![ctx(
            2,
            1,
            &long_hit,
            "doc.md",
            vec![adj(1, 1, -1, &long_prev, "doc.md")],
            vec![adj(3, 1, 1, &long_next, "doc.md")],
        )];
        let (_, user) = build_prompt_with_context("q", &hits, 20);
        assert!(user.contains(&format!("[1] (doc.md::Sec1)\n{}…", "p".repeat(20))));
        assert!(user.contains(&format!("[2] (doc.md::Sec2)\n{}…", "h".repeat(20))));
        assert!(user.contains(&format!("[3] (doc.md::Sec3)\n{}…", "n".repeat(20))));
    }

    #[test]
    fn passes_gate_boundary() {
        assert!(passes_gate(Some(0.35), 0.35));
        assert!(passes_gate(Some(0.36), 0.35));
        assert!(!passes_gate(Some(0.34), 0.35));
        assert!(!passes_gate(Some(0.34), 0.35));
        assert!(!passes_gate(None, 0.35));
    }
}
