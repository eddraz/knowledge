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

### Default build (sidecar mode)

- `llama-server` on PATH or `~/.local/bin/llama-server`
- Models in `~/models` (or `KNOWLEDGE_MODELS_DIR`):
  - `bge-m3-Q8_0.gguf` (embeddings, 1024-dim, L2-normalized)
  - `LFM2.5-230M-F16.gguf` (grounded answer generation)

### Native build (`--features native`)

- No `llama-server` needed — embeddings and generation run in-process via candle
- Still needs `LFM2.5-230M-F16.gguf` (candle loads it for generation)
- bge-m3 F32 weights auto-download on first native use (see [Hybrid & Native embeddings](#hybrid--native-embeddings-features-native) below)

## Install

### crates.io

```bash
cargo install knowledge-cli                     # default build (sidecar mode)
cargo install knowledge-cli --features native   # native in-process embeddings + generation
```

### GitHub Releases

Prebuilt native binaries are published as `knowledge-<version>-<target>.tar.gz`
for:

- `x86_64-unknown-linux-gnu`
- `aarch64-unknown-linux-gnu`
- `aarch64-apple-darwin`
- `x86_64-apple-darwin`

Each tarball contains `knowledge`, `native-ask`, `README`, and `LICENSE`.

```bash
curl -L -o knowledge.tar.gz https://github.com/eddraz/knowledge/releases/download/<version>/knowledge-<version>-<target>.tar.gz
tar -xzf knowledge.tar.gz
mv knowledge native-ask ~/.local/bin/
```

See https://github.com/eddraz/knowledge/releases.

### native-ask

`native-ask` (shipped in release tarballs) provides fully in-process RAG:
both retrieval (via candle bge-m3) and generation run in-process via candle
with the quantized LFM2.5 GGUF.

```bash
native-ask "question" --owner alice
```

Flags: `question`, `--owner`, `-k` (default 5), `-w` / `--window` (default 1),
`--no-context`, `--max-tokens` (default 512), `--model`, `--tokenizer`, `--verbose`.

## Usage

```bash
knowledge add doc.md                      # ingest a file ("-" reads stdin)
knowledge add -t "nota rápida o texto"    # ingest a raw string directly (-t / --text)
knowledge add -t "nota" --name "mi-nota"  # ingest a string with a custom name
knowledge add doc.md --meta               # LLM title/keywords extraction
knowledge list                            # list documents
knowledge owners                          # list owner namespaces
knowledge search "pregunta parafraseada" --mode vector   # semantic (default, includes N-1/N+1)
knowledge search "sensores" --mode lexical               # FTS5 BM25
knowledge search "financiamiento" --mode hybrid          # RRF fusion
knowledge search "sensores" --no-context                 # compact 1-line results (or --window 0)
knowledge ask "¿Quién financia el Proyecto Aurora?"      # grounded answer + sources
knowledge rm doc.md                       # remove document
knowledge chown doc.md alice              # change document owner
knowledge status                          # config + db summary
knowledge update                          # update the CLI (cargo install or GitHub Release)
knowledge setup                           # create/check data directory and config
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
| `KNOWLEDGE_EMBED_PORT` | `28488` | embeddings sidecar port |
| `KNOWLEDGE_GEN_PORT` | `8099` | generator sidecar port |
| `KNOWLEDGE_TOP_K` | `5` | chunks retrieved per query |
| `KNOWLEDGE_MIN_SCORE` | `0.35` | minimum cosine to trust retrieval |
| `KNOWLEDGE_CONTEXT_WINDOW` | `1` | adjacent chunks (`N-1`, `N+1`) included per hit |
| `KNOWLEDGE_TIMEOUT_SECS` | `120` | HTTP + sidecar health timeout |
| `KNOWLEDGE_APPS_DIR` | `~/apps` | installed sidecar / app binaries directory |
| `KNOWLEDGE_EMBEDDER` | `hybrid` on native builds / `sidecar` | embedding backend selector (`hybrid`, `native`, `sidecar`) |
| `KNOWLEDGE_EMBED_MODEL` | `bge-m3-Q8_0.gguf` | default embedding GGUF |
| `KNOWLEDGE_GEN_MODEL` | `LFM2.5-230M-F16.gguf` | default generation GGUF |

## How it works

- **Chunking**: sentence-aware, ~600 chars target with word-complete overlap,
  split on markdown section boundaries (`#`).
- **Embeddings** (sidecar mode; native builds use opportunistic hybrid embeddings, see
  [Hybrid & Native embeddings](#hybrid--native-embeddings-features-native) below): bge-m3 served
  by a `llama-server --embedding` sidecar; the CLI spawns it on demand and kills
  it on exit (reuses a healthy one if the port already serves).
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
  via candle against the quantized LFM2.5 GGUF model. Embeddings default to
  opportunistic hybrid mode (daemon check on port 28488 with in-process candle bge-m3 fallback).
  Set `KNOWLEDGE_EMBEDDER=sidecar` to force the llama.cpp embedding sidecar, or
  `KNOWLEDGE_EMBEDDER=native` to force in-process candle only.

## Hybrid & Native embeddings (`--features native`)

When the binary is built with `--features native`, embeddings default to
**Opportunistic Hybrid Mode** (`EmbedderMode::Hybrid`):
1. **Daemon Probe**: If an embedding daemon or server is already running on the configured port (`127.0.0.1:28488`), queries are sent via HTTP for near-instant resolution (~5–10 ms).
2. **Candle Fallback**: If no daemon is running, embeddings run in-process via `candle` against the official BAAI bge-m3 checkpoint completely offline and self-contained, with zero configuration.

To explicitly force a specific backend:
- `KNOWLEDGE_EMBEDDER=hybrid`: (Default) Probe daemon on port 28488; fallback to in-process Candle.
- `KNOWLEDGE_EMBEDDER=native`: Force in-process Candle (never probe daemon).
- `KNOWLEDGE_EMBEDDER=sidecar`: Force `llama-server --embedding` sidecar.

On first use with in-process candle, bootstrap downloads ~2.3 GB of F32
PyTorch weights as `pytorch_model.bin` (plus `config.json` and `tokenizer.json`)
into `$KNOWLEDGE_MODELS_DIR/bge-m3/`. The sidecar-only GGUF
(`bge-m3-Q8_0.gguf`) is not downloaded in native mode unless sidecar mode is forced.

Vectors produced by the native F32 model differ slightly from vectors produced
by the Q8_0 sidecar (typical cosine drift ~0.001–0.01). This is expected and
acceptable: the default `KNOWLEDGE_MIN_SCORE=0.35` gate has enough margin, and
retrieval remains stable.

## Development

```bash
cargo test                     # 74 unit tests (no sidecars required)
cargo test --features native   # 78 unit tests, including candle in-process generation
cargo run -- status
cargo build --release --features native
```

Task history lives in `odd/tasks/` (one document per feature).
