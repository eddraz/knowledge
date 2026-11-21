use std::path::Path;
use std::sync::{Mutex, Once};

use rusqlite::{params, Connection, OptionalExtension};

use crate::error::{KnowledgeError, Result};
use crate::meta::DocMeta;

static VEC_ONCE: Once = Once::new();
static VEC_ERROR: Mutex<Option<String>> = Mutex::new(None);

#[allow(dead_code)]
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

#[allow(dead_code)]
const SCHEMA_V2: &str = r#"
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    source TEXT NOT NULL UNIQUE,
    hash TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    owner TEXT NOT NULL DEFAULT '_shared',
    title TEXT,
    description TEXT,
    keywords TEXT
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

const SCHEMA_V3: &str = r#"
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    source TEXT NOT NULL UNIQUE,
    hash TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    owner TEXT NOT NULL DEFAULT '_shared',
    title TEXT,
    description TEXT,
    keywords TEXT
);

CREATE INDEX IF NOT EXISTS idx_documents_owner ON documents(owner);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    section TEXT,
    text TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);

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

CREATE TRIGGER IF NOT EXISTS chunks_ad_vec
AFTER DELETE ON chunks
BEGIN
    DELETE FROM chunks_vec WHERE chunk_id = old.id;
END;
"#;

const MIGRATION_V1_TO_V2: &str = r#"
ALTER TABLE documents ADD COLUMN owner TEXT NOT NULL DEFAULT '_shared';
ALTER TABLE documents ADD COLUMN title TEXT;
ALTER TABLE documents ADD COLUMN description TEXT;
ALTER TABLE documents ADD COLUMN keywords TEXT;
"#;

const MIGRATION_V2_TO_V3: &str = r#"
CREATE INDEX IF NOT EXISTS idx_documents_owner ON documents(owner);
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE TRIGGER IF NOT EXISTS chunks_ad_vec
AFTER DELETE ON chunks
BEGIN
    DELETE FROM chunks_vec WHERE chunk_id = old.id;
END;
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
    pub owner: String,
}

/// A chunk adjacent to a search hit within the same document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdjacentChunk {
    pub chunk_id: i64,
    pub document_id: i64,
    pub offset: isize,
    pub text: String,
    pub section: Option<String>,
    pub source: String,
    pub owner: String,
}

/// A search hit enriched with its adjacent chunks (`N-1`, `N+1`, ...) from the same document.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchHitContext {
    pub hit: SearchHit,
    pub document_id: i64,
    pub before: Vec<AdjacentChunk>,
    pub after: Vec<AdjacentChunk>,
}

/// Metadata stored for a document, as returned by `list_documents`.
#[derive(Debug, Clone, PartialEq)]
pub struct DocInfo {
    pub id: i64,
    pub source: String,
    pub hash: String,
    pub owner: String,
    pub title: Option<String>,
    pub description: Option<String>,
    pub keywords: Option<String>,
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
    if version >= 3 {
        return Ok(());
    }

    let tx = conn.transaction().map_err(KnowledgeError::Db)?;
    if version == 0 {
        tx.execute_batch(SCHEMA_V3).map_err(KnowledgeError::Db)?;
    } else {
        if version == 1 {
            tx.execute_batch(MIGRATION_V1_TO_V2)
                .map_err(KnowledgeError::Db)?;
        }
        if version <= 2 {
            tx.execute_batch(MIGRATION_V2_TO_V3)
                .map_err(KnowledgeError::Db)?;
        }
    }
    tx.execute("PRAGMA user_version = 3", [])
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

pub fn insert_document(conn: &Connection, source: &str, hash: &str, owner: &str) -> Result<i64> {
    conn.execute(
        "INSERT OR IGNORE INTO documents(source, hash, owner) VALUES (?1, ?2, ?3)",
        params![source, hash, owner],
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

pub fn update_document_meta(conn: &Connection, doc_id: i64, meta: &DocMeta) -> Result<()> {
    let title = non_empty(&meta.title);
    let description = non_empty(&meta.description);
    let keywords = if meta.keywords.is_empty() {
        None
    } else {
        Some(meta.keywords.join(", "))
    };

    conn.execute(
        "UPDATE documents SET title = ?1, description = ?2, keywords = ?3 WHERE id = ?4",
        params![title, description, keywords.as_deref(), doc_id],
    )
    .map_err(KnowledgeError::Db)?;
    Ok(())
}

fn non_empty(s: &str) -> Option<&str> {
    let t = s.trim();
    if t.is_empty() {
        None
    } else {
        Some(t)
    }
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

pub fn knn_search(
    conn: &Connection,
    query: &[f32],
    k: usize,
    owner: Option<&str>,
) -> Result<Vec<SearchHit>> {
    if query.len() != 1024 {
        return Err(KnowledgeError::Other(format!(
            "expected query dimension 1024, got {}",
            query.len()
        )));
    }

    let blob: &[u8] = bytemuck::cast_slice(query);

    let rows = if let Some(owner) = owner {
        let mut stmt = conn
            .prepare(
                "SELECT v.chunk_id, v.distance, c.text, c.section, d.source, d.owner
                 FROM chunks_vec v
                 JOIN chunks c ON c.id = v.chunk_id
                 JOIN documents d ON d.id = c.document_id
                 WHERE v.embedding MATCH ?1 AND v.k = ?2
                   AND (d.owner = ?3 OR d.owner = '_shared')
                 ORDER BY v.distance",
            )
            .map_err(KnowledgeError::Db)?;
        let rows = stmt
            .query_map(params![blob, k as i64, owner], map_knn_row)
            .map_err(KnowledgeError::Db)?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(KnowledgeError::Db)?
    } else {
        let mut stmt = conn
            .prepare(
                "SELECT v.chunk_id, v.distance, c.text, c.section, d.source, d.owner
                 FROM chunks_vec v
                 JOIN chunks c ON c.id = v.chunk_id
                 JOIN documents d ON d.id = c.document_id
                 WHERE v.embedding MATCH ?1 AND v.k = ?2
                 ORDER BY v.distance",
            )
            .map_err(KnowledgeError::Db)?;
        let rows = stmt
            .query_map(params![blob, k as i64], map_knn_row)
            .map_err(KnowledgeError::Db)?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(KnowledgeError::Db)?
    };

    Ok(rows)
}

fn map_knn_row(row: &rusqlite::Row) -> std::result::Result<SearchHit, rusqlite::Error> {
    let distance: f64 = row.get(1)?;
    Ok(SearchHit {
        chunk_id: row.get(0)?,
        score: distance_to_cosine_score(distance),
        text: row.get(2)?,
        section: row.get(3)?,
        source: row.get(4)?,
        owner: row.get(5)?,
    })
}

pub fn fts_search(
    conn: &Connection,
    query: &str,
    k: usize,
    owner: Option<&str>,
) -> Result<Vec<SearchHit>> {
    let rows = if let Some(owner) = owner {
        let mut stmt = conn
            .prepare(
                "SELECT c.id, c.text, c.section, d.source, d.owner, rank
                 FROM chunks c
                 JOIN documents d ON d.id = c.document_id
                 JOIN chunks_fts f ON f.rowid = c.id
                 WHERE chunks_fts MATCH ?1
                   AND (d.owner = ?3 OR d.owner = '_shared')
                 ORDER BY rank
                 LIMIT ?2",
            )
            .map_err(KnowledgeError::Db)?;
        let rows = stmt
            .query_map(params![query, k as i64, owner], map_fts_row)
            .map_err(KnowledgeError::Db)?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(KnowledgeError::Db)?
    } else {
        let mut stmt = conn
            .prepare(
                "SELECT c.id, c.text, c.section, d.source, d.owner, rank
                 FROM chunks c
                 JOIN documents d ON d.id = c.document_id
                 JOIN chunks_fts f ON f.rowid = c.id
                 WHERE chunks_fts MATCH ?1
                 ORDER BY rank
                 LIMIT ?2",
            )
            .map_err(KnowledgeError::Db)?;
        let rows = stmt
            .query_map(params![query, k as i64], map_fts_row)
            .map_err(KnowledgeError::Db)?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(KnowledgeError::Db)?
    };

    Ok(rows)
}

fn map_fts_row(row: &rusqlite::Row) -> std::result::Result<SearchHit, rusqlite::Error> {
    let rank: f64 = row.get(5)?;
    Ok(SearchHit {
        chunk_id: row.get(0)?,
        score: -rank,
        text: row.get(1)?,
        section: row.get(2)?,
        source: row.get(3)?,
        owner: row.get(4)?,
    })
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

pub fn list_documents(conn: &Connection) -> Result<Vec<DocInfo>> {
    let mut stmt = conn
        .prepare(
            "SELECT id, source, hash, owner, title, description, keywords
             FROM documents ORDER BY id",
        )
        .map_err(KnowledgeError::Db)?;

    let rows = stmt
        .query_map([], |row| {
            Ok(DocInfo {
                id: row.get(0)?,
                source: row.get(1)?,
                hash: row.get(2)?,
                owner: row.get(3)?,
                title: row.get(4)?,
                description: row.get(5)?,
                keywords: row.get(6)?,
            })
        })
        .map_err(KnowledgeError::Db)?;

    rows.collect::<std::result::Result<Vec<_>, _>>()
        .map_err(KnowledgeError::Db)
}

/// Summary of documents and chunks for a single owner namespace.
#[derive(Debug, Clone, PartialEq)]
pub struct OwnerSummary {
    pub owner: String,
    pub documents: i64,
    pub chunks: i64,
}

/// List distinct owner namespaces with their document and chunk counts.
pub fn list_owners(conn: &Connection) -> Result<Vec<OwnerSummary>> {
    let mut stmt = conn
        .prepare(
            "SELECT d.owner, COUNT(DISTINCT d.id), COUNT(c.id)
             FROM documents d
             LEFT JOIN chunks c ON c.document_id = d.id
             GROUP BY d.owner
             ORDER BY d.owner ASC",
        )
        .map_err(KnowledgeError::Db)?;

    let rows = stmt
        .query_map([], |row| {
            Ok(OwnerSummary {
                owner: row.get(0)?,
                documents: row.get(1)?,
                chunks: row.get(2)?,
            })
        })
        .map_err(KnowledgeError::Db)?;

    rows.collect::<std::result::Result<Vec<_>, _>>()
        .map_err(KnowledgeError::Db)
}

/// Change the owner of a document.  Returns `true` if a row was updated.
pub fn set_owner(conn: &Connection, source: &str, owner: &str) -> Result<bool> {
    let changed = conn
        .execute(
            "UPDATE documents SET owner = ?1 WHERE source = ?2",
            params![owner, source],
        )
        .map_err(KnowledgeError::Db)?;
    Ok(changed > 0)
}

#[derive(Clone, Copy)]
enum NeighborDirection {
    Before,
    After,
}

fn query_neighbors(
    conn: &Connection,
    document_id: i64,
    chunk_id: i64,
    window: usize,
    source: &str,
    owner: &str,
    direction: NeighborDirection,
) -> Result<Vec<AdjacentChunk>> {
    if window == 0 {
        return Ok(Vec::new());
    }
    let limit = i64::try_from(window).unwrap_or(i64::MAX);
    let sql = match direction {
        NeighborDirection::Before => {
            "SELECT id, section, text
             FROM chunks
             WHERE document_id = ?1 AND id < ?2
             ORDER BY id DESC
             LIMIT ?3"
        }
        NeighborDirection::After => {
            "SELECT id, section, text
             FROM chunks
             WHERE document_id = ?1 AND id > ?2
             ORDER BY id ASC
             LIMIT ?3"
        }
    };

    let mut stmt = conn.prepare_cached(sql).map_err(KnowledgeError::Db)?;
    let mut raw: Vec<(i64, Option<String>, String)> = stmt
        .query_map(params![document_id, chunk_id, limit], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?))
        })
        .map_err(KnowledgeError::Db)?
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(KnowledgeError::Db)?;

    if matches!(direction, NeighborDirection::Before) {
        raw.reverse();
    }

    let len = raw.len();
    let chunks = raw
        .into_iter()
        .enumerate()
        .map(|(i, (id, section, text))| {
            let offset = match direction {
                NeighborDirection::Before => -((len - i) as isize),
                NeighborDirection::After => (i + 1) as isize,
            };
            AdjacentChunk {
                chunk_id: id,
                document_id,
                offset,
                text,
                section,
                source: source.to_string(),
                owner: owner.to_string(),
            }
        })
        .collect();

    Ok(chunks)
}

/// Fetch up to `window` adjacent chunks before (`N-1`, ...) and after
/// (`N+1`, ...) `chunk_id` within the same document, ordered in document order.
///
/// Returns `(document_id, before, after)`. If `chunk_id` does not exist,
/// returns `(0, Vec::new(), Vec::new())`.
pub fn adjacent_chunks(
    conn: &Connection,
    chunk_id: i64,
    window: usize,
) -> Result<(i64, Vec<AdjacentChunk>, Vec<AdjacentChunk>)> {
    let mut doc_stmt = conn
        .prepare_cached(
            "SELECT c.document_id, d.source, d.owner
             FROM chunks c
             JOIN documents d ON d.id = c.document_id
             WHERE c.id = ?1",
        )
        .map_err(KnowledgeError::Db)?;

    let doc_info: Option<(i64, String, String)> = doc_stmt
        .query_row(params![chunk_id], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?))
        })
        .optional()
        .map_err(KnowledgeError::Db)?;

    let Some((document_id, source, owner)) = doc_info else {
        return Ok((0, Vec::new(), Vec::new()));
    };

    let before = query_neighbors(
        conn,
        document_id,
        chunk_id,
        window,
        &source,
        &owner,
        NeighborDirection::Before,
    )?;
    let after = query_neighbors(
        conn,
        document_id,
        chunk_id,
        window,
        &source,
        &owner,
        NeighborDirection::After,
    )?;

    Ok((document_id, before, after))
}

/// Enrich search hits with up to `window` adjacent chunks before and after each
/// hit within the same document, preserving hit rank order.
pub fn attach_context(
    conn: &Connection,
    hits: Vec<SearchHit>,
    window: usize,
) -> Result<Vec<SearchHitContext>> {
    hits.into_iter()
        .map(|hit| {
            let (document_id, before, after) = adjacent_chunks(conn, hit.chunk_id, window)?;
            Ok(SearchHitContext {
                hit,
                document_id,
                before,
                after,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

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
        assert_eq!(version, 3);
    }

    #[test]
    fn fresh_database_has_owner_column() {
        let conn = schema();
        let mut stmt = conn
            .prepare("SELECT name FROM pragma_table_info('documents') WHERE name = 'owner'")
            .unwrap();
        let names: Vec<String> = stmt
            .query_map([], |row| row.get::<_, String>(0))
            .unwrap()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(names, vec!["owner"]);
    }

    #[test]
    fn migration_from_v1_preserves_data() {
        let pid = std::process::id();
        let path = PathBuf::from(format!(
            "{}/knowledge_v1_migration_test_{pid}.db",
            std::env::temp_dir().display()
        ));
        let _ = std::fs::remove_file(&path);

        {
            register_vec_extension().unwrap();
            let conn = Connection::open(&path).unwrap();
            conn.execute_batch("PRAGMA foreign_keys = ON; PRAGMA journal_mode = WAL;")
                .unwrap();
            conn.execute_batch(SCHEMA_V1).unwrap();
            conn.execute(
                "INSERT INTO documents(source, hash) VALUES (?1, ?2)",
                params!["legacy.txt", "deadbeef"],
            )
            .unwrap();
            conn.execute("PRAGMA user_version = 1", []).unwrap();
        }

        let conn = open(&path).unwrap();
        let version: i64 = conn
            .query_row("PRAGMA user_version", [], |row| row.get(0))
            .unwrap();
        assert_eq!(version, 3);

        let docs = list_documents(&conn).unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].source, "legacy.txt");
        assert_eq!(docs[0].hash, "deadbeef");
        assert_eq!(docs[0].owner, "_shared");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn migration_from_v2_preserves_data_and_adds_indexes_and_trigger() {
        let pid = std::process::id();
        let path = PathBuf::from(format!(
            "{}/knowledge_v2_migration_test_{pid}.db",
            std::env::temp_dir().display()
        ));
        let _ = std::fs::remove_file(&path);

        {
            register_vec_extension().unwrap();
            let conn = Connection::open(&path).unwrap();
            conn.execute_batch("PRAGMA foreign_keys = ON; PRAGMA journal_mode = WAL;")
                .unwrap();
            conn.execute_batch(SCHEMA_V2).unwrap();
            conn.execute(
                "INSERT INTO documents(source, hash, owner, title) VALUES (?1, ?2, ?3, ?4)",
                params!["doc-v2.md", "hashv2", "alice", "Alice Doc"],
            )
            .unwrap();
            conn.execute("PRAGMA user_version = 2", []).unwrap();
        }

        let conn = open(&path).unwrap();
        let version: i64 = conn
            .query_row("PRAGMA user_version", [], |row| row.get(0))
            .unwrap();
        assert_eq!(version, 3);

        let docs = list_documents(&conn).unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].source, "doc-v2.md");
        assert_eq!(docs[0].owner, "alice");
        assert_eq!(docs[0].title.as_deref(), Some("Alice Doc"));

        // Verify index on chunks(document_id) exists
        let mut stmt = conn
            .prepare("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_chunks_document_id'")
            .unwrap();
        let idx_count = stmt.query_map([], |_| Ok(())).unwrap().count();
        assert_eq!(idx_count, 1);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn owner_filter_applies_to_knn_and_fts() {
        let conn = schema();

        let eddraz_doc = insert_document(&conn, "eddraz.txt", "h1", "eddraz").unwrap();
        let laura_doc = insert_document(&conn, "laura.txt", "h2", "laura").unwrap();
        let shared_doc = insert_document(&conn, "shared.txt", "h3", "_shared").unwrap();

        insert_chunk(
            &conn,
            eddraz_doc,
            None,
            "eddraz private content",
            &unit_vector(0),
        )
        .unwrap();
        insert_chunk(
            &conn,
            laura_doc,
            None,
            "laura private content",
            &unit_vector(1),
        )
        .unwrap();
        insert_chunk(
            &conn,
            shared_doc,
            None,
            "shared public content",
            &unit_vector(0),
        )
        .unwrap();

        let hits = knn_search(&conn, &unit_vector(0), 10, Some("eddraz")).unwrap();
        let sources: Vec<&str> = hits.iter().map(|h| h.source.as_str()).collect();
        assert!(sources.contains(&"eddraz.txt"));
        assert!(sources.contains(&"shared.txt"));
        assert!(!sources.contains(&"laura.txt"));

        let all = knn_search(&conn, &unit_vector(0), 10, None).unwrap();
        assert_eq!(all.len(), 3);

        let fts = fts_search(&conn, "shared", 10, Some("laura")).unwrap();
        assert_eq!(fts.len(), 1);
        assert_eq!(fts[0].source, "shared.txt");

        let fts_all = fts_search(&conn, "content", 10, None).unwrap();
        assert_eq!(fts_all.len(), 3);
    }

    #[test]
    fn set_owner_moves_document() {
        let conn = schema();
        let doc = insert_document(&conn, " movable.txt ", "hash", "eddraz").unwrap();
        insert_chunk(&conn, doc, None, "content", &[0.0; 1024]).unwrap();

        let before = list_documents(&conn).unwrap();
        assert_eq!(before[0].owner, "eddraz");

        let changed = set_owner(&conn, " movable.txt ", "laura").unwrap();
        assert!(changed);

        let after = list_documents(&conn).unwrap();
        assert_eq!(after[0].owner, "laura");

        let missing = set_owner(&conn, "missing.txt", "laura").unwrap();
        assert!(!missing);
    }

    #[test]
    fn knn_roundtrip_and_distance_semantics() {
        let conn = schema();

        let doc = insert_document(&conn, "vectors.txt", "abc123", "_shared").unwrap();
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
        let hits = knn_search(&conn, &e1, 3, None).unwrap();
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
        let doc = insert_document(&conn, "accents.txt", "hash", "_shared").unwrap();
        insert_chunk(&conn, doc, None, "cafe au lait", &[0.0; 1024]).unwrap();
        insert_chunk(&conn, doc, None, "cafe solo", &[0.0; 1024]).unwrap();

        let hits = fts_search(&conn, "cafe", 10, None).unwrap();
        assert_eq!(hits.len(), 2);
        let texts: Vec<_> = hits.iter().map(|h| h.text.as_str()).collect();
        assert!(texts.contains(&"cafe au lait"));
        assert!(texts.contains(&"cafe solo"));

        // A chunk containing the term twice must rank before a chunk containing
        // it once.  This validates that lower FTS5 rank (better match) is
        // surfaced as a higher score.
        let conn = schema();
        let doc = insert_document(&conn, "ranking.txt", "hash", "_shared").unwrap();
        insert_chunk(&conn, doc, None, "one cafe mention", &[0.0; 1024]).unwrap();
        let two = insert_chunk(&conn, doc, None, "cafe cafe two mentions", &[0.0; 1024]).unwrap();

        let hits = fts_search(&conn, "cafe", 10, None).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].chunk_id, two);
        assert!(hits[0].score > hits[1].score);
    }

    #[test]
    fn delete_document_removes_chunks_and_vectors() {
        let conn = schema();
        let doc = insert_document(&conn, "remove.txt", "hash", "_shared").unwrap();
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

    #[test]
    fn list_owners_groups_counts_and_orders_alphabetically() {
        let conn = schema();

        // _shared: one document, no chunks.
        insert_document(&conn, "shared.txt", "h1", "_shared").unwrap();

        // alice: two documents, one with chunks.
        let alice_doc1 = insert_document(&conn, "alice1.txt", "h2", "alice").unwrap();
        let alice_doc2 = insert_document(&conn, "alice2.txt", "h3", "alice").unwrap();
        insert_chunk(&conn, alice_doc1, None, "alice first", &[0.0; 1024]).unwrap();
        insert_chunk(&conn, alice_doc1, None, "alice second", &[0.0; 1024]).unwrap();
        insert_chunk(&conn, alice_doc2, None, "alice third", &[0.0; 1024]).unwrap();

        // bob: one document, no chunks.
        insert_document(&conn, "bob.txt", "h4", "bob").unwrap();

        let owners = list_owners(&conn).unwrap();
        assert_eq!(owners.len(), 3);

        assert_eq!(owners[0].owner, "_shared");
        assert_eq!(owners[0].documents, 1);
        assert_eq!(owners[0].chunks, 0);

        assert_eq!(owners[1].owner, "alice");
        assert_eq!(owners[1].documents, 2);
        assert_eq!(owners[1].chunks, 3);

        assert_eq!(owners[2].owner, "bob");
        assert_eq!(owners[2].documents, 1);
        assert_eq!(owners[2].chunks, 0);
    }

    #[test]
    fn adjacent_chunks_middle_hit_returns_prev_and_next() {
        let conn = schema();
        let doc = insert_document(&conn, "guide.md", "h1", "alice").unwrap();
        let c1 = insert_chunk(&conn, doc, Some("S1"), "chunk one", &[0.0; 1024]).unwrap();
        let c2 = insert_chunk(&conn, doc, Some("S2"), "chunk two", &[0.0; 1024]).unwrap();
        let c3 = insert_chunk(&conn, doc, Some("S3"), "chunk three", &[0.0; 1024]).unwrap();
        let c4 = insert_chunk(&conn, doc, Some("S4"), "chunk four", &[0.0; 1024]).unwrap();
        let _c5 = insert_chunk(&conn, doc, Some("S5"), "chunk five", &[0.0; 1024]).unwrap();

        let (doc_id, before, after) = adjacent_chunks(&conn, c3, 1).unwrap();
        assert_eq!(doc_id, doc);
        assert_eq!(before.len(), 1);
        assert_eq!(before[0].chunk_id, c2);
        assert_eq!(before[0].offset, -1);
        assert_eq!(before[0].text, "chunk two");
        assert_eq!(before[0].section.as_deref(), Some("S2"));
        assert_eq!(before[0].source, "guide.md");
        assert_eq!(before[0].owner, "alice");

        assert_eq!(after.len(), 1);
        assert_eq!(after[0].chunk_id, c4);
        assert_eq!(after[0].offset, 1);
        assert_eq!(after[0].text, "chunk four");
        assert_eq!(after[0].section.as_deref(), Some("S4"));

        // Window of 2 returns [N-2, N-1] and [N+1, N+2] in document order.
        let (_, before2, after2) = adjacent_chunks(&conn, c3, 2).unwrap();
        assert_eq!(
            before2
                .iter()
                .map(|c| (c.chunk_id, c.offset))
                .collect::<Vec<_>>(),
            vec![(c1, -2), (c2, -1)]
        );
        assert_eq!(
            after2
                .iter()
                .map(|c| (c.chunk_id, c.offset))
                .collect::<Vec<_>>(),
            vec![(c4, 1), (_c5, 2)]
        );

        // Window of 0 returns empty neighbors while preserving document_id.
        let (doc0, before0, after0) = adjacent_chunks(&conn, c3, 0).unwrap();
        assert_eq!(doc0, doc);
        assert!(before0.is_empty());
        assert!(after0.is_empty());
    }

    #[test]
    fn adjacent_chunks_boundaries_never_cross_documents() {
        let conn = schema();
        let doc_a = insert_document(&conn, "a.md", "h1", "alice").unwrap();
        let a1 = insert_chunk(&conn, doc_a, None, "a first", &[0.0; 1024]).unwrap();
        let a2 = insert_chunk(&conn, doc_a, None, "a second", &[0.0; 1024]).unwrap();

        let doc_b = insert_document(&conn, "b.md", "h2", "bob").unwrap();
        let b1 = insert_chunk(&conn, doc_b, None, "b first", &[0.0; 1024]).unwrap();
        let b2 = insert_chunk(&conn, doc_b, None, "b second", &[0.0; 1024]).unwrap();

        // First chunk of doc A has no N-1, only N+1.
        let (_, before, after) = adjacent_chunks(&conn, a1, 1).unwrap();
        assert!(before.is_empty());
        assert_eq!(
            after.iter().map(|c| c.chunk_id).collect::<Vec<_>>(),
            vec![a2]
        );

        // Last chunk of doc A must NOT leak into doc B.
        let (_, before, after) = adjacent_chunks(&conn, a2, 1).unwrap();
        assert_eq!(
            before.iter().map(|c| c.chunk_id).collect::<Vec<_>>(),
            vec![a1]
        );
        assert!(after.is_empty());

        // First chunk of doc B must NOT leak back into doc A.
        let (_, before, after) = adjacent_chunks(&conn, b1, 1).unwrap();
        assert!(before.is_empty());
        assert_eq!(
            after.iter().map(|c| c.chunk_id).collect::<Vec<_>>(),
            vec![b2]
        );
    }

    #[test]
    fn attach_context_enriches_hits_and_handles_missing_chunk_gracefully() {
        let conn = schema();
        let doc = insert_document(&conn, "vec.md", "h1", "_shared").unwrap();
        let c1 = insert_chunk(&conn, doc, Some("Intro"), "before text", &unit_vector(1)).unwrap();
        let c2 = insert_chunk(&conn, doc, Some("Core"), "match text", &unit_vector(0)).unwrap();
        let c3 = insert_chunk(&conn, doc, Some("Outro"), "after text", &unit_vector(2)).unwrap();

        let hits = knn_search(&conn, &unit_vector(0), 1, None).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].chunk_id, c2);

        let enriched = attach_context(&conn, hits, 1).unwrap();
        assert_eq!(enriched.len(), 1);
        assert_eq!(enriched[0].document_id, doc);
        assert_eq!(enriched[0].before.len(), 1);
        assert_eq!(enriched[0].before[0].chunk_id, c1);
        assert_eq!(enriched[0].after.len(), 1);
        assert_eq!(enriched[0].after[0].chunk_id, c3);

        // Missing chunk_id degrades gracefully without error.
        let missing = vec![SearchHit {
            chunk_id: 9999,
            score: 0.5,
            text: "ghost".to_string(),
            section: None,
            source: "ghost.md".to_string(),
            owner: "_shared".to_string(),
        }];
        let enriched_missing = attach_context(&conn, missing, 1).unwrap();
        assert_eq!(enriched_missing.len(), 1);
        assert_eq!(enriched_missing[0].document_id, 0);
        assert!(enriched_missing[0].before.is_empty());
        assert!(enriched_missing[0].after.is_empty());
    }
}
