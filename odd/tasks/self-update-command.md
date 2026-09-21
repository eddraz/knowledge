# Feature: self-update-command

`knowledge update`: one command that updates the CLI itself. Auto-detects the
install channel: if the running binary lives in the cargo bin dir, delegate to
`cargo install knowledge-cli --force`; otherwise self-update from GitHub
Releases (download the platform `.tar.gz`, replace `knowledge` + `native-ask`).

## Locked decisions (2026-09-21)

- Channel: auto-detect. Cargo mode when `current_exe()` parent == cargo bin dir
  (`$CARGO_HOME/bin`, else `~/.cargo/bin`); release mode otherwise. User chose
  this over GitHub-only and cargo-only.
- Version check: cargo mode queries crates.io API and skips install when equal;
  release mode queries GitHub `/releases/latest` and compares `tag_name` against
  `CARGO_PKG_VERSION` (numeric major.minor.patch triple compare, string fallback).
- Release-mode platform map from `std::env::consts::{OS, ARCH}`:
  x86_64+linux → x86_64-unknown-linux-gnu, aarch64+linux → aarch64-unknown-linux-gnu,
  aarch64+macos → aarch64-apple-darwin, x86_64+macos → x86_64-apple-darwin.
  Windows/other → explicit error suggesting `cargo install knowledge-cli`.
- Download/extract follow existing bootstrap.rs style: `curl -L -f -o` subprocess
  (streamed output), rename `.part`, then `tar -xzf` subprocess. Zero new deps.
- Atomic-ish replace: `.new` + `.old` backup rename per binary; chmod 0o755.
  Both binaries replaced (release archives always ship both; replace local
  `native-ask` whenever the archive contains it).
- Cargo mode passes `--features native` only when a `native-ask` sibling exists
  in the cargo bin dir (required-features means native installs have it).
- Post-update verification: run updated binary with `--version`, print old → new.
- Errors via `KnowledgeError` (module returns `Result<()>`), matching bootstrap.rs.
- `--verbose` global flag controls progress output.

## Non-goals

- No checksum/signature verification (release workflow publishes no checksums yet).
- No Windows release-mode support (workflow builds no Windows artifacts).
- No auto-update on startup; explicit command only.
- No rollback beyond the `.old` backup rename.

## Tasks

### [x] 1. Implement src/update.rs + CLI wiring
DONE (delegated to gentle-ai-worker). Detection pure + parameterized, numeric version
triple compare with string fallback, curl UA headers, tar -xzf, .new/.old dance with
0o755, .old retained until post-update --version verification (reconciled spec ambiguity:
remove-after-verify). 7 new unit tests, no network. Wired lib.rs + main.rs.
- [x] commit: feat update command (evidence: a791e7d)

### [x] 2. Verify + docs
DONE (verified by gentle-ai-verify, parent-observed). cargo build/test (52 green:
45 pre-existing + 7 new), cargo build --features native, `knowledge update --help` OK.
README usage block + prose documented. Folded into the same work-unit commit a791e7d
(4 files, 452 insertions: docs+tests travel with behavior). Clippy: 1 pre-existing
warning in main.rs:307, untouched by this diff. Pending review: native review candidate
a791e7d under RDD switch.
- [x] commit: docs update command (evidence: folded into a791e7d)
