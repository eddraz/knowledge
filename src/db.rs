use std::path::Path;
use std::sync::{Mutex, Once};

use rusqlite::{params, Connection};

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

const MIGRATION_V1_TO_V2: &str = r#"
ALTER TABLE documents ADD COLUMN owner TEXT NOT NULL DEFAULT '_shared';
ALTER TABLE documents ADD COLUMN title TEXT;
ALTER TABLE documents ADD COLUMN description TEXT;
ALTER TABLE documents ADD COLUMN keywords TEXT;
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
    if version >= 2 {
        return Ok(());
    }

    let tx = conn.transaction().map_err(KnowledgeError::Db)?;
    if version == 1 {
        tx.execute_batch(MIGRATION_V1_TO_V2)
            .map_err(KnowledgeError::Db)?;
    } else {
        tx.execute_batch(SCHEMA_V2).map_err(KnowledgeError::Db)?;
    }
    tx.execute("PRAGMA user_version = 2", [])
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
        assert_eq!(version, 2);
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
        assert_eq!(version, 2);

        let docs = list_documents(&conn).unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].source, "legacy.txt");
        assert_eq!(docs[0].hash, "deadbeef");
        assert_eq!(docs[0].owner, "_shared");

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
}
