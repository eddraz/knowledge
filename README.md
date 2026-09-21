# knowledge

Local RAG knowledge base CLI: ingest your own documents and ask questions that are
answered **strictly** from their content. Everything runs on your machine.

```
│ docs ──► chunk ──► bge-m3 ──► SQLite + sqlite-vec (KNN) + FTS5 (BM25)
│                                                                    │
└─ query ─► retrieve top-k ─► score gate ─► LFM2.5 ─► cited answer
```

## Requirements

- Rust 1.85+ (crate uses edition 2021)
- `llama-server` on PATH or `~/.local/bin/llama-server`
- Models in `~/models` (or `KNOWLEDGE_MODELS_DIR`):
  - `bge-m3-Q8_0.gguf` (embeddings, 1024-dim, L2-normalized)
  - `LFM2.5-230M-F16.gguf` (grounded answer generation)

## Install (crates.io)

```bash
cargo install knowledge-cli   # installs the `knowledge` command
```

## Usage

```bash
knowledge add doc.md            # ingest a file ("-" reads stdin)
knowledge list                  # list documents
knowledge search "pregunta parafraseada" --mode vector   # semantic (default)
knowledge search "sensores" --mode lexical                        # FTS5 BM25
knowledge search "financiamiento" --mode hybrid                   # RRF fusion
knowledge ask "¿Quién financia el Proyecto Aurora?"               # grounded answer + sources
knowledge rm doc.md             # remove document
knowledge status                # config + db summary
knowledge update                # update the CLI (cargo install or GitHub Release)
```

`knowledge update` checks the installed source: when the binary lives in the Cargo bin directory it runs `cargo install knowledge-cli --force` (preserving the `native` feature if `native-ask` is present), otherwise it downloads the matching GitHub Release archive for the current platform and replaces the `knowledge` and `native-ask` binaries in place.

`ask` refuses politely when retrieval confidence is below `KNOWLEDGE_MIN_SCORE`
(default 0.35): it answers only from your content, never from model memory.

## Configuration (env vars)

| Variable | Default | Meaning |
| --- | --- | --- |
| `KNOWLEDGE_DB` | `~/.local/share/knowledge/knowledge.db` | SQLite database path |
| `KNOWLEDGE_LLAMA_SERVER` | `~/.local/bin/llama-server` | sidecar binary |
| `KNOWLEDGE_MODELS_DIR` | `~/models` | GGUF models directory |
| `KNOWLEDGE_EMBED_PORT` | `8098` | embeddings sidecar port |
| `KNOWLEDGE_GEN_PORT` | `8099` | generator sidecar port |
| `KNOWLEDGE_TOP_K` | `5` | chunks retrieved per query |
| `KNOWLEDGE_MIN_SCORE` | `0.35` | minimum cosine to trust retrieval |
| `KNOWLEDGE_TIMEOUT_SECS` | `120` | HTTP + sidecar health timeout |

## How it works

- **Chunking**: sentence-aware, ~600 chars target with word-complete overlap,
  split on markdown section boundaries (`#`).
- **Embeddings**: bge-m3 served by a `llama-server --embedding` sidecar;
  the CLI spawns it on demand and kills it on exit (reuses a healthy one if
  the port already serves).
- **Storage**: SQLite with three access paths over the same chunks:
  - `chunks_vec`: sqlite-vec `vec0` float[1024], exact KNN with Euclidean
    distance converted to cosine (`score = 1 - d²/2`, valid because bge-m3
    vectors are L2-normalized).
  - `chunks_fts`: FTS5 with `unicode61 remove_diacritics=2` (é matches e).
  - hybrid: Reciprocal Rank Fusion over both ranked lists.
- **Grounding**: numbered context excerpts, instructions to answer only from
  context and cite excerpt numbers, plus a retrieval score gate.
- **Native generation (`--features native`)**: when built with the `native`
  feature, `ask` answers and `add --meta` metadata extraction run in-process
  via candle against the quantized LFM2.5 GGUF model. Embeddings always run
  through the bge-m3 sidecar, so only the generator sidecar is skipped. The
  trade-off is convenience (no generator sidecar startup) vs. raw decode speed:
  candle CPU decoding is slower than the llama.cpp generator sidecar used by
  the default build.

## Development

```bash
cargo test                     # 45 unit tests (no sidecars required)
cargo test --features native   # 46 unit tests, including candle in-process generation
cargo run -- status
cargo build --release --features native
```

Task history lives in `odd/tasks/knowledge-base-rag.md`.
