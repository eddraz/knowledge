use reqwest::Client;
use rusqlite::Connection;

use crate::config::Config;
use crate::db::SearchHit;
use crate::error::{KnowledgeError, Result};
use crate::search::{run_search, SearchMode};

#[cfg(not(feature = "native"))]
use crate::sidecar::{acquire, SidecarRole};
#[cfg(feature = "native")]
use crate::{bootstrap, native};

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
    let system = "You are a retrieval-grounded assistant. Answer the question using ONLY the context excerpts below. Synthesize the relevant information into a direct answer in your own words. If the context does not contain enough information, reply exactly that the information is not in the knowledge base. Do not use outside knowledge. Do not mention the excerpt numbers or the word Sources in the reply. Answer in the same language as the question.".to_string();

    let mut context = String::new();
    for (i, hit) in hits.iter().enumerate() {
        let num = i + 1;
        let section = hit.section.as_deref().unwrap_or("");
        let text = truncate_text(&hit.text, max_chunk_chars);
        context.push_str(&format!(
            "[{num}] ({source}::{section})\n{text}\n\n",
            source = hit.source
        ));
    }

    let user = format!("{context}\nQuestion: {question}");
    (system, user)
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

    let (system, user) = build_prompt(question, &hits, 1200);

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
    fn passes_gate_boundary() {
        assert!(passes_gate(Some(0.35), 0.35));
        assert!(passes_gate(Some(0.36), 0.35));
        assert!(!passes_gate(Some(0.34), 0.35));
        assert!(!passes_gate(Some(0.34), 0.35));
        assert!(!passes_gate(None, 0.35));
    }
}
