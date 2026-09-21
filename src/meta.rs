use reqwest::Client;
use serde_json::Value;

#[cfg(feature = "native")]
use crate::{bootstrap, config::Config, native};

/// Metadata extracted from a document, used for the summary chunk and listing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocMeta {
    pub title: String,
    pub description: String,
    pub keywords: Vec<String>,
}

/// Tolerantly extract the first JSON object from raw model output.
///
/// Handles prose before/after the object and code fences.  Validates that the
/// required keys exist with the correct types, trims strings, caps title and
/// description lengths, and deduplicates keywords.  Returns `None` on any
/// failure so that callers can fall back to default metadata.
pub fn parse_metadata(raw: &str) -> Option<DocMeta> {
    let object_text = first_json_object(raw)?;
    let value: Value = serde_json::from_str(object_text).ok()?;
    let object = value.as_object()?;

    let title = object.get("title").and_then(Value::as_str).map(str::trim)?;
    let description = object
        .get("description")
        .and_then(Value::as_str)
        .map(str::trim)?;
    let keywords_value = object.get("keywords")?;
    let keywords_array = keywords_value.as_array()?;

    let title = cap_chars(title, 80);
    let description = cap_chars(description, 300);

    let mut seen = std::collections::HashSet::new();
    let keywords: Vec<String> = keywords_array
        .iter()
        .filter_map(|v| v.as_str().map(str::trim))
        .filter(|s| !s.is_empty())
        .filter(|s| seen.insert(*s))
        .take(8)
        .map(String::from)
        .collect();

    Some(DocMeta {
        title,
        description,
        keywords,
    })
}

fn first_json_object(raw: &str) -> Option<&str> {
    let start = raw.find('{')?;
    let mut depth = 0;
    let mut in_string = false;
    let mut escape = false;

    for (i, b) in raw.bytes().enumerate().skip(start) {
        if in_string {
            if escape {
                escape = false;
                continue;
            }
            match b {
                b'"' => in_string = false,
                b'\\' => escape = true,
                _ => {}
            }
        } else {
            match b {
                b'"' => in_string = true,
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(&raw[start..=i]);
                    }
                }
                _ => {}
            }
        }
    }

    None
}

fn cap_chars(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        s.chars().take(max).collect()
    }
}

/// Generate metadata for a document using an LLM.
///
/// The document text is truncated to 2500 characters and the model is asked to
/// return strict JSON only.  Any error returns `None` so the caller can fall
/// back to default metadata.
pub async fn generate(
    http: &Client,
    gen_base_url: &str,
    model: &str,
    doc_text: &str,
) -> Option<DocMeta> {
    let truncated: String = doc_text.chars().take(2500).collect();

    let system = r#"You are a metadata extractor. Return STRICT JSON only, with no markdown, no code fences, and no prose. The JSON object must have exactly these keys: {"title": "...", "description": "...", "keywords": ["..."]}. Title max 80 chars, description max 300 chars, up to 8 keywords."#;

    let user = format!(
        "Generate a concise title, one-sentence description, and relevant keywords for the following document:\n\n{truncated}"
    );

    #[cfg(feature = "native")]
    let _ = (http, gen_base_url, model);

    #[cfg(feature = "native")]
    let raw = {
        let cfg = Config::load().ok()?;
        let model_path = cfg.gen_model_path();
        let tokenizer_path =
            bootstrap::ensure_registry_file(&cfg.models_dir, native::DEFAULT_TOKENIZER_NAME, false)
                .ok()?;
        native::generate(&model_path, &tokenizer_path, system, &user, 512)
            .map(|(answer, _stats)| answer)
            .ok()?
    };

    #[cfg(not(feature = "native"))]
    let raw = crate::llm::generate(http, gen_base_url, model, system, &user, 512)
        .await
        .ok()?;
    parse_metadata(&raw)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_accepts_valid_json() {
        let raw =
            r#"{"title":"Hello World","description":"A simple document","keywords":["foo","bar"]}"#;
        let meta = parse_metadata(raw).unwrap();
        assert_eq!(meta.title, "Hello World");
        assert_eq!(meta.description, "A simple document");
        assert_eq!(meta.keywords, vec!["foo", "bar"]);
    }

    #[test]
    fn parse_extracts_object_from_fenced_output() {
        let raw = "Here is the metadata you asked for:\n```json\n{\"title\": \"My Doc\", \"description\": \"Short.\", \"keywords\": [\"rust\", \"cli\"]}\n```\nHope this helps.";
        let meta = parse_metadata(raw).unwrap();
        assert_eq!(meta.title, "My Doc");
        assert_eq!(meta.description, "Short.");
        assert_eq!(meta.keywords, vec!["rust", "cli"]);
    }

    #[test]
    fn parse_rejects_missing_fields() {
        let raw = r#"{"title":"Only title","description":"No keywords"}"#;
        assert!(parse_metadata(raw).is_none());
    }

    #[test]
    fn parse_rejects_invalid_types() {
        let raw = r#"{"title":"T","description":"D","keywords":"not an array"}"#;
        assert!(parse_metadata(raw).is_none());
    }

    #[test]
    fn parse_caps_and_dedups_keywords() {
        let title = "a".repeat(100);
        let desc = "b".repeat(400);
        let raw = format!(
            "{{\"title\": \"{title}\", \"description\": \"{desc}\", \"keywords\": [\"one\", \"two\", \"three\", \"four\", \"five\", \"six\", \"seven\", \"eight\", \"nine\", \"one\"]}}"
        );
        let meta = parse_metadata(&raw).unwrap();
        assert_eq!(meta.title.len(), 80);
        assert_eq!(meta.description.len(), 300);
        assert_eq!(meta.keywords.len(), 8);
        assert_eq!(meta.keywords[0], "one");
        assert!(!meta.keywords.contains(&"nine".to_string()));
    }

    #[test]
    fn parse_returns_none_for_garbage() {
        assert!(parse_metadata("not json").is_none());
        assert!(parse_metadata("{\"title\": \"unclosed").is_none());
        assert!(parse_metadata("").is_none());
    }

    #[test]
    fn parse_filters_empty_keywords() {
        let raw = r#"{"title":"T","description":"D","keywords":["","  ","ok","ok"]}}"#;
        let meta = parse_metadata(raw).unwrap();
        assert_eq!(meta.keywords, vec!["ok"]);
    }
}
