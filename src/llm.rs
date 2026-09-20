use std::time::Duration;

use reqwest::Client;
use serde_json::{json, Value};

use crate::error::{KnowledgeError, Result};

pub const EMBED_DIM: usize = 1024;

/// Parse and strictly validate an OpenAI-style `/v1/embeddings` response.
///
/// Checks that every embedding has length `EMBED_DIM` and contains only finite
/// floating point values.  The number of returned embeddings must equal
/// `expected`.
pub fn parse_embeddings_response(body: &Value, expected: usize) -> Result<Vec<Vec<f32>>> {
    let object = body.get("object").and_then(Value::as_str).unwrap_or("list");
    if object != "list" {
        return Err(KnowledgeError::BadResponse(format!(
            "expected object 'list', got '{object}'"
        )));
    }

    let data = body.get("data").and_then(Value::as_array).ok_or_else(|| {
        KnowledgeError::BadResponse("missing or non-array 'data' field".to_string())
    })?;

    if data.len() != expected {
        return Err(KnowledgeError::BadResponse(format!(
            "expected {expected} embeddings, got {}",
            data.len()
        )));
    }

    let mut out = Vec::with_capacity(data.len());
    for (i, item) in data.iter().enumerate() {
        let index =
            item.get("index").and_then(Value::as_u64).ok_or_else(|| {
                KnowledgeError::BadResponse(format!("embedding {i} missing index"))
            })? as usize;
        if index != i {
            return Err(KnowledgeError::BadResponse(format!(
                "embedding index mismatch: expected {i}, got {index}"
            )));
        }

        let arr = item
            .get("embedding")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                KnowledgeError::BadResponse(format!("embedding {i} missing or invalid"))
            })?;

        if arr.len() != EMBED_DIM {
            return Err(KnowledgeError::BadResponse(format!(
                "embedding {i} has dimension {}, expected {EMBED_DIM}",
                arr.len()
            )));
        }

        let mut vec = Vec::with_capacity(EMBED_DIM);
        for (j, val) in arr.iter().enumerate() {
            let f = val.as_f64().ok_or_else(|| {
                KnowledgeError::BadResponse(format!("embedding {i} value {j} is not a number"))
            })? as f32;
            if !f.is_finite() {
                return Err(KnowledgeError::BadResponse(format!(
                    "embedding {i} value {j} is not finite"
                )));
            }
            vec.push(f);
        }
        out.push(vec);
    }

    Ok(out)
}

/// Request embeddings for one or more texts.
pub async fn embed_texts(http: &Client, base_url: &str, texts: &[String]) -> Result<Vec<Vec<f32>>> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }

    let url = format!("{base_url}/v1/embeddings");
    let body = json!({
        "input": texts,
        "model": "bge-m3",
    });

    let response = http
        .post(&url)
        .json(&body)
        .send()
        .await
        .map_err(KnowledgeError::Http)?;

    let status = response.status();
    let value: Value = response.json().await.map_err(KnowledgeError::Http)?;

    if !status.is_success() {
        return Err(KnowledgeError::BadResponse(format!(
            "embeddings endpoint returned {status}: {value}"
        )));
    }

    parse_embeddings_response(&value, texts.len())
}

/// Parse and validate an OpenAI-style `/v1/chat/completions` response.
pub fn parse_chat_response(body: &Value) -> Result<String> {
    let choices = body
        .get("choices")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            KnowledgeError::BadResponse("missing or non-array 'choices' field".to_string())
        })?;

    let first = choices
        .first()
        .ok_or_else(|| KnowledgeError::BadResponse("'choices' array is empty".to_string()))?;

    let content = first
        .get("message")
        .and_then(|m| m.get("content"))
        .and_then(Value::as_str)
        .ok_or_else(|| {
            KnowledgeError::BadResponse("missing choices[0].message.content".to_string())
        })?;

    Ok(content.to_string())
}

/// Generate a chat completion.
pub async fn generate(
    http: &Client,
    base_url: &str,
    model: &str,
    system: &str,
    user: &str,
    max_tokens: u32,
) -> Result<String> {
    let url = format!("{base_url}/v1/chat/completions");
    let body = json!({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
    });

    let response = http
        .post(&url)
        .json(&body)
        .send()
        .await
        .map_err(KnowledgeError::Http)?;

    let status = response.status();
    let value: Value = response.json().await.map_err(KnowledgeError::Http)?;

    if !status.is_success() {
        return Err(KnowledgeError::BadResponse(format!(
            "chat endpoint returned {status}: {value}"
        )));
    }

    parse_chat_response(&value)
}

/// Build an HTTP client with the configured request timeout.
pub fn http_client(timeout_secs: u64) -> Result<Client> {
    Client::builder()
        .timeout(Duration::from_secs(timeout_secs))
        .build()
        .map_err(KnowledgeError::Http)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn embedding_vec(value: f32) -> Vec<f32> {
        let mut v = vec![value; EMBED_DIM];
        v[0] = value + 0.1;
        v
    }

    fn embeddings_fixture() -> Value {
        let e1: Vec<f32> = embedding_vec(0.1);
        let e2: Vec<f32> = embedding_vec(0.2);
        json!({
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": e1
                },
                {
                    "object": "embedding",
                    "index": 1,
                    "embedding": e2
                }
            ],
            "model": "bge-m3",
            "usage": {"prompt_tokens": 42, "total_tokens": 42}
        })
    }

    #[test]
    fn parse_embeddings_accepts_valid_response() {
        let body = embeddings_fixture();
        let parsed = parse_embeddings_response(&body, 2).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].len(), EMBED_DIM);
        assert_eq!(parsed[1].len(), EMBED_DIM);
        assert!(parsed[0][0].is_finite());
    }

    #[test]
    fn parse_embeddings_rejects_wrong_count() {
        let body = embeddings_fixture();
        let err = parse_embeddings_response(&body, 3).unwrap_err();
        assert!(matches!(err, KnowledgeError::BadResponse(_)));
    }

    #[test]
    fn parse_embeddings_rejects_non_finite() {
        let mut e: Vec<f32> = vec![0.0; EMBED_DIM];
        e[0] = f32::NAN;
        let body = json!({
            "object": "list",
            "data": [{"object":"embedding","index":0,"embedding": e}]
        });
        let err = parse_embeddings_response(&body, 1).unwrap_err();
        assert!(matches!(err, KnowledgeError::BadResponse(_)));
    }

    #[test]
    fn parse_chat_extracts_content() {
        let body = json!({
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello, world!"
                    },
                    "finish_reason": "stop"
                }
            ]
        });
        assert_eq!(parse_chat_response(&body).unwrap(), "Hello, world!");
    }

    #[test]
    fn parse_chat_rejects_missing_content() {
        let body = json!({"choices": [{"message": {}}]});
        let err = parse_chat_response(&body).unwrap_err();
        assert!(matches!(err, KnowledgeError::BadResponse(_)));
    }
}
