use std::path::Path;
use std::sync::{Mutex, Once};

use rusqlite::{params, Connection};

use crate::error::{KnowledgeError, Result};

static VEC_ONCE: Once = Once::new();
static VEC_ERROR: Mutex<Option<String>> = Mutex::new(None);

const SCHEMA_V1: &str = r#"
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    source TEXT NOT NULL UNIQUE,
    hash TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    section TEXT,
    text TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    content='chunks',
    content_rowid='id',
    tokenize='unicode61 remove_diacritics 2'
);

CREATE TRIGGER IF NOT EXISTS chunks_ai
AFTER INSERT ON chunks
BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad
AFTER DELETE ON chunks
BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text)
    VALUES ('delete', old.id, old.text);
END;

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
    chunk_id INTEGER PRIMARY KEY,
    embedding float[1024]
);
"#;

/// A single search result.
///
/// The meaning of `score` depends on the search mode that produced the hit:
///
/// * `knn_search` returns the cosine similarity between the query vector and
///   the chunk embedding, in the range `[-1, 1]`.
/// * `fts_search` returns a positive lexical relevance score derived from the
///   FTS5 `rank` value.  Higher values indicate a better lexical match.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchHit {
    pub chunk_id: i64,
    pub score: f64,
    pub text: String,
    pub section: Option<String>,
    pub source: String,
}

/// Register the sqlite-vec extension once per process.
///
/// sqlite-vec exposes its entry point as `sqlite3_vec_init`.  The symbol has the
/// classic SQLite auto-extension signature but the Rust crate declares it with
/// no arguments so we can only take its address and transmute it to the expected
/// `RawAutoExtension` type that rusqlite's `register_auto_extension` accepts.
pub fn register_vec_extension() -> Result<()> {
    VEC_ONCE.call_once(|| {
        let result: Result<()> = unsafe {
            let raw: rusqlite::auto_extension::RawAutoExtension =
                std::mem::transmute(sqlite_vec::sqlite3_vec_init as *const ());
            rusqlite::auto_extension::register_auto_extension(raw).map_err(KnowledgeError::Db)
        };
        if let Err(e) = result {
            if let Ok(mut guard) = VEC_ERROR.lock() {
                *guard = Some(e.to_string());
            }
        }
    });

    match VEC_ERROR.lock() {
        Ok(guard) => match guard.as_ref() {
            Some(msg) => Err(KnowledgeError::Other(msg.clone())),
            None => Ok(()),
        },
        Err(_) => Err(KnowledgeError::Other(
            "extension registration mutex poisoned".to_string(),
        )),
    }
}

/// Open a SQLite connection with the vec0/FTS5 schema initialised.
pub fn open<P: AsRef<Path>>(path: P) -> Result<Connection> {
    register_vec_extension()?;

    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(KnowledgeError::Io)?;
    }

    let mut conn = Connection::open(path).map_err(KnowledgeError::Db)?;
    conn.execute_batch("PRAGMA foreign_keys = ON; PRAGMA journal_mode = WAL;")
        .map_err(KnowledgeError::Db)?;
    init_schema(&mut conn)?;
    Ok(conn)
}

fn init_schema(conn: &mut Connection) -> Result<()> {
    let version: i64 = conn
        .query_row("PRAGMA user_version", [], |row| row.get(0))
        .map_err(KnowledgeError::Db)?;
    if version >= 1 {
        return Ok(());
    }

    let tx = conn.transaction().map_err(KnowledgeError::Db)?;
    tx.execute_batch(SCHEMA_V1).map_err(KnowledgeError::Db)?;
    tx.execute("PRAGMA user_version = 1", [])
        .map_err(KnowledgeError::Db)?;
    tx.commit().map_err(KnowledgeError::Db)?;
    Ok(())
}

/// Convert a sqlite-vec KNN distance into a cosine similarity score.
///
/// Empirically sqlite-vec returns the Euclidean distance between L2-normalised
/// vectors.  For unit vectors:
///
///   d^2 = ||u - v||^2 = 2 - 2(u·v)
///
/// so the cosine similarity is `1 - d*d/2`.
pub fn distance_to_cosine_score(distance: f64) -> f64 {
    1.0 - (distance * distance) / 2.0
}

pub fn insert_document(conn: &Connection, source: &str, hash: &str) -> Result<i64> {
    conn.execute(
        "INSERT OR IGNORE INTO documents(source, hash) VALUES (?1, ?2)",
        params![source, hash],
    )
    .map_err(KnowledgeError::Db)?;

    let id: i64 = conn
        .query_row(
            "SELECT id FROM documents WHERE source = ?1",
            params![source],
            |row| row.get(0),
        )
        .map_err(KnowledgeError::Db)?;
    Ok(id)
}

pub fn insert_chunk(
    conn: &Connection,
    document_id: i64,
    section: Option<&str>,
    text: &str,
    embedding: &[f32],
) -> Result<i64> {
    if embedding.len() != 1024 {
        return Err(KnowledgeError::Other(format!(
            "expected embedding dimension 1024, got {}",
            embedding.len()
        )));
    }

    conn.execute(
        "INSERT INTO chunks(document_id, section, text) VALUES (?1, ?2, ?3)",
        params![document_id, section, text],
    )
    .map_err(KnowledgeError::Db)?;

    let chunk_id = conn.last_insert_rowid();
    let blob: &[u8] = bytemuck::cast_slice(embedding);

    conn.execute(
        "INSERT INTO chunks_vec(chunk_id, embedding) VALUES (?1, ?2)",
        params![chunk_id, blob],
    )
    .map_err(KnowledgeError::Db)?;

    Ok(chunk_id)
}

pub fn knn_search(conn: &Connection, query: &[f32], k: usize) -> Result<Vec<SearchHit>> {
    if query.len() != 1024 {
        return Err(KnowledgeError::Other(format!(
            "expected query dimension 1024, got {}",
            query.len()
        )));
    }

    let blob: &[u8] = bytemuck::cast_slice(query);
    let mut stmt = conn
        .prepare(
            "SELECT v.chunk_id, v.distance, c.text, c.section, d.source
             FROM chunks_vec v
             JOIN chunks c ON c.id = v.chunk_id
             JOIN documents d ON d.id = c.document_id
             WHERE v.embedding MATCH ?1 AND v.k = ?2
             ORDER BY v.distance",
        )
        .map_err(KnowledgeError::Db)?;

    let rows = stmt
        .query_map(params![blob, k as i64], |row| {
            let distance: f64 = row.get(1)?;
            Ok(SearchHit {
                chunk_id: row.get(0)?,
                score: distance_to_cosine_score(distance),
                text: row.get(2)?,
                section: row.get(3)?,
                source: row.get(4)?,
            })
        })
        .map_err(KnowledgeError::Db)?;

    rows.collect::<std::result::Result<Vec<_>, _>>()
        .map_err(KnowledgeError::Db)
}

pub fn fts_search(conn: &Connection, query: &str, k: usize) -> Result<Vec<SearchHit>> {
    let mut stmt = conn
        .prepare(
            "SELECT c.id, c.text, c.section, d.source, rank
             FROM chunks c
             JOIN documents d ON d.id = c.document_id
             JOIN chunks_fts f ON f.rowid = c.id
             WHERE chunks_fts MATCH ?1
             ORDER BY rank
             LIMIT ?2",
        )
        .map_err(KnowledgeError::Db)?;

    let rows = stmt
        .query_map(params![query, k as i64], |row| {
            let rank: f64 = row.get(4)?;
            Ok(SearchHit {
                chunk_id: row.get(0)?,
                score: -rank,
                text: row.get(1)?,
                section: row.get(2)?,
                source: row.get(3)?,
            })
        })
        .map_err(KnowledgeError::Db)?;

    rows.collect::<std::result::Result<Vec<_>, _>>()
        .map_err(KnowledgeError::Db)
}

/// Delete a document and all of its chunks and vectors.
pub fn delete_document(conn: &Connection, source: &str) -> Result<bool> {
    let ids: Vec<i64> = conn
        .prepare("SELECT id FROM documents WHERE source = ?1")
        .map_err(KnowledgeError::Db)?
        .query_map(params![source], |row| row.get(0))
        .map_err(KnowledgeError::Db)?
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(KnowledgeError::Db)?;

    if ids.is_empty() {
        return Ok(false);
    }

    conn.execute(
        "DELETE FROM chunks_vec WHERE chunk_id IN (
            SELECT id FROM chunks WHERE document_id = ?1
        )",
        params![ids[0]],
    )
    .map_err(KnowledgeError::Db)?;

    let deleted = conn
        .execute("DELETE FROM documents WHERE source = ?1", params![source])
        .map_err(KnowledgeError::Db)?;

    Ok(deleted > 0)
}

pub fn list_documents(conn: &Connection) -> Result<Vec<(i64, String, String)>> {
    let mut stmt = conn
        .prepare("SELECT id, source, hash FROM documents ORDER BY id")
        .map_err(KnowledgeError::Db)?;

    let rows = stmt
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
        .map_err(KnowledgeError::Db)?;

    rows.collect::<std::result::Result<Vec<_>, _>>()
        .map_err(KnowledgeError::Db)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn in_memory() -> Connection {
        register_vec_extension().unwrap();
        Connection::open_in_memory().unwrap()
    }

    fn schema() -> Connection {
        let mut conn = in_memory();
        init_schema(&mut conn).unwrap();
        conn
    }

    fn unit_vector(idx: usize) -> Vec<f32> {
        let mut v = vec![0.0_f32; 1024];
        v[idx] = 1.0;
        v
    }

    fn opposite(v: &[f32]) -> Vec<f32> {
        v.iter().map(|x| -x).collect()
    }

    #[test]
    fn vec_version_is_available() {
        let conn = in_memory();
        let version: String = conn
            .query_row("SELECT vec_version()", [], |row| row.get(0))
            .unwrap();
        assert!(version.starts_with('v'));
    }

    #[test]
    fn schema_initialises_user_version() {
        let conn = schema();
        let version: i64 = conn
            .query_row("PRAGMA user_version", [], |row| row.get(0))
            .unwrap();
        assert_eq!(version, 1);
    }

    #[test]
    fn knn_roundtrip_and_distance_semantics() {
        let conn = schema();

        let doc = insert_document(&conn, "vectors.txt", "abc123").unwrap();
        let e1 = unit_vector(0);
        let e2 = unit_vector(1);
        let e3 = opposite(&e1);

        insert_chunk(&conn, doc, None, "first axis", &e1).unwrap();
        insert_chunk(&conn, doc, None, "second axis", &e2).unwrap();
        insert_chunk(&conn, doc, None, "opposite first", &e3).unwrap();

        // Inspect raw distances to prove the semantic used by sqlite-vec.
        let mut stmt = conn
            .prepare(
                "SELECT v.distance FROM chunks_vec v
                 WHERE v.embedding MATCH ?1 AND v.k = ?2
                 ORDER BY v.distance",
            )
            .unwrap();
        let distances: Vec<f64> = stmt
            .query_map(params![bytemuck::cast_slice(&e1), 3_i64], |row| {
                row.get::<_, f64>(0)
            })
            .unwrap()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap();

        assert_eq!(distances.len(), 3);
        assert!(
            (distances[0]).abs() < 1e-6,
            "identical vectors must have distance 0"
        );
        assert!(
            (distances[1] - std::f64::consts::SQRT_2).abs() < 1e-4,
            "orthogonal unit vectors have Euclidean distance sqrt(2), got {}",
            distances[1]
        );
        assert!(
            (distances[2] - 2.0).abs() < 1e-4,
            "opposite unit vectors have Euclidean distance 2, got {}",
            distances[2]
        );

        // Converted cosine scores.
        let hits = knn_search(&conn, &e1, 3).unwrap();
        assert_eq!(hits.len(), 3);
        assert!((hits[0].score - 1.0).abs() < 1e-6);
        assert!((hits[1].score - 0.0).abs() < 1e-4);
        assert!((hits[2].score - (-1.0)).abs() < 1e-4);

        // Ordering check.
        assert_eq!(hits[0].text, "first axis");
        assert_eq!(hits[1].text, "second axis");
        assert_eq!(hits[2].text, "opposite first");
    }

    #[test]
    fn fts_matches_with_and_without_accents() {
        let conn = schema();
        let doc = insert_document(&conn, "accents.txt", "hash").unwrap();
        insert_chunk(&conn, doc, None, "cafe au lait", &[0.0; 1024]).unwrap();
        insert_chunk(&conn, doc, None, "cafe solo", &[0.0; 1024]).unwrap();

        let hits = fts_search(&conn, "cafe", 10).unwrap();
        assert_eq!(hits.len(), 2);
        let texts: Vec<_> = hits.iter().map(|h| h.text.as_str()).collect();
        assert!(texts.contains(&"cafe au lait"));
        assert!(texts.contains(&"cafe solo"));

        // A chunk containing the term twice must rank before a chunk containing
        // it once.  This validates that lower FTS5 rank (better match) is
        // surfaced as a higher score.
        let conn = schema();
        let doc = insert_document(&conn, "ranking.txt", "hash").unwrap();
        insert_chunk(&conn, doc, None, "one cafe mention", &[0.0; 1024]).unwrap();
        let two = insert_chunk(&conn, doc, None, "cafe cafe two mentions", &[0.0; 1024]).unwrap();

        let hits = fts_search(&conn, "cafe", 10).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].chunk_id, two);
        assert!(hits[0].score > hits[1].score);
    }

    #[test]
    fn delete_document_removes_chunks_and_vectors() {
        let conn = schema();
        let doc = insert_document(&conn, "remove.txt", "hash").unwrap();
        let chunk = insert_chunk(&conn, doc, None, "delete me", &[1.0; 1024]).unwrap();

        let before = list_documents(&conn).unwrap();
        assert_eq!(before.len(), 1);

        let deleted = delete_document(&conn, "remove.txt").unwrap();
        assert!(deleted);

        let after = list_documents(&conn).unwrap();
        assert!(after.is_empty());

        // Vector row must also be gone.
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM chunks_vec WHERE chunk_id = ?1",
                params![chunk],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 0);

        let deleted_again = delete_document(&conn, "remove.txt").unwrap();
        assert!(!deleted_again);
    }
}
