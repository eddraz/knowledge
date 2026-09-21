# Feature: native-candle-embeddings

Run bge-m3 embeddings in-process with candle (XLM-RoBERTa) instead of the
llama-server `--embedding` sidecar, behind the existing `native` feature.

## Locked decisions (2026-09-21)

- Backend: candle 0.9 `candle_transformers::models::xlm_roberta::XLMRobertaModel`
  (verified present in 0.9.2) + `tokenizers` with bge-m3 `tokenizer.json`.
  No quantized_bert in candle -> weights are the official F32 safetensors
  (~2.27 GB) from BAAI/bge-m3, loaded via VarBuilder safetensors mmap, DType F32,
  Device::Cpu (same policy as native-ask LFM2.5).
- Pooling: CLS token + L2 normalize, 1024-dim f32 -> same vector contract as the
  sidecar path (which is already L2-normalized per bootstrap docs).
- Activation: when compiled with `--features native`, native embedder is the
  DEFAULT; llama-server is not spawned for embeddings. `KNOWLEDGE_EMBEDDER=sidecar`
  env escape hatch forces the sidecar path. Non-native builds: sidecar only.
- Drift: stored doc vectors were produced by GGUF Q8_0 via llama.cpp; native F32
  queries drift ~0.001-0.01 cosine. Accepted and documented (README + status
  output showing active embedder). Score gate 0.35 has margin. No forced reindex.
- Downloads: bootstrap gains native-embed artifacts (config.json, tokenizer.json,
  model.safetensors) via existing curl + .part-rename pattern, only fetched when
  the native embedder is active; bge-m3-Q8_0.gguf stays sidecar-only.
- Zero new dependencies: candle/candle-transformers/tokenizers are already
  optional deps of the `native` feature. No hf-hub; direct curl downloads.

## Non-goals

- No GPU/CUDA/Metal device support (Cpu only, like native-ask).
- No reindex command in this feature.
- No GGUF support in candle (upstream lacks quantized_bert).
- native-ask (generation) path unchanged.

## Tasks

### [x] 1. Native embedder module + selection logic
DONE (delegated to gentle-ai-worker, 2 continuations). src/embed.rs: EmbedderMode +
resolve_embedder_mode (unit-tested 5 rules), sidecar HTTP reuse, native candle impl
(OnceLock, spawn_blocking, pad/truncate 8192, CLS get_on_dim(1,0) + l2_normalize,
F32/CPU). Weights: pytorch_model.bin via VarBuilder::from_pth — NO model.safetensors
exists upstream (writer confirmed 404; spec deviation approved by parent). Key-prefix
bug found by E2E: checkpoint has NO roberta. prefix (verified via pickle inspection);
fixed to root VarBuilder. candle-nn added as optional dep (parent-owned Cargo.toml
edit) replacing mimi re-export hack.
- [x] commit: feat native candle embeddings (evidence: afe4108)

### [x] 2. Wire ingest/search + bootstrap downloads
DONE (same delegation). ingest/search through embed::embed_texts; sidecar spawn only
in Sidecar mode; bootstrap native registry only in Native mode (GGUF skipped);
status shows active embedder + resolved paths.
- [x] commit: feat wire embedder selection (evidence: folded into afe4108)

### [x] 3. Verify + docs
DONE. Verified by gentle-ai-verify (8/8): build/test both configs (47/50 green),
clippy clean in changed files, sidecar E2E regression (score 0.8912 via llama-server
HTTP), native E2E with real weights (ingest + search 0.8463), throwaway DBs only,
cleaned up. README native section + drift note.
- [x] commit: docs native embeddings (evidence: folded into afe4108)
