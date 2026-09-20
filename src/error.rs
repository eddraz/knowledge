use thiserror::Error;

#[derive(Debug, Error)]
pub enum KnowledgeError {
    #[error("database error: {0}")]
    Db(#[from] rusqlite::Error),

    #[error("embedding sidecar error: {0}")]
    EmbeddingSidecar(String),

    #[error("generator sidecar error: {0}")]
    GeneratorSidecar(String),

    #[error("bad response from LLM server: {0}")]
    BadResponse(String),

    #[error("no relevant content found")]
    NoRelevantContent,

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, KnowledgeError>;
