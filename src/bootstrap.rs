//! First-run bootstrap for the `knowledge` runtime.
//!
//! Verifies (and installs if missing):
//!   * the MBZUAI-IFM llama.cpp fork built as `llama-server`, and
//!   * the GGUF models referenced by the active `Config`.

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use crate::config::Config;
use crate::error::{KnowledgeError, Result};

const MODEL_REGISTRY: &[(&str, &str)] = &[
    (
        "bge-m3-Q8_0.gguf",
        "https://huggingface.co/ggml-org/bge-m3-Q8_0-GGUF/resolve/main/bge-m3-q8_0.gguf?download=true",
    ),
    (
        "LFM2.5-230M-F16.gguf",
        "https://huggingface.co/LiquidAI/LFM2.5-230M-GGUF/resolve/main/LFM2.5-230M-F16.gguf",
    ),
];

/// Path to the fork build of `llama-server` derived from the configured apps
/// directory.
pub fn fork_bin_path(apps_dir: &Path) -> PathBuf {
    apps_dir
        .join("llama.cpp")
        .join("build")
        .join("bin")
        .join("llama-server")
}

/// Pure resolution rule for the llama-server binary.
///
/// Preference order:
/// 1. `fork_bin` (the MBZUAI-IFM fork build),
/// 2. `configured` (user override or system default),
/// 3. `None` (install required).
pub fn resolve_llama_server(fork_bin: &Path, configured: &Path) -> Option<PathBuf> {
    if fork_bin.exists() {
        Some(fork_bin.to_path_buf())
    } else if configured.exists() {
        Some(configured.to_path_buf())
    } else {
        None
    }
}

/// Look up a known model name in the bootstrap registry.
pub fn model_registry_lookup(name: &str) -> Option<&'static str> {
    MODEL_REGISTRY
        .iter()
        .find(|(registered, _)| registered == &name)
        .map(|(_, url)| *url)
}

/// Returns `true` when the file at `path` does not yet exist and must be
/// downloaded.
pub fn needs_download(path: &Path) -> bool {
    !path.exists()
}

/// Blocking, idempotent bootstrap routine.
///
/// Ensures the configured llama-server binary and the configured GGUF models
/// are present, installing or downloading only when necessary. When `verbose`
/// is `false`, progress lines (prefixed `[bootstrap]`) are emitted only while
/// doing work.
pub fn ensure_ready(cfg: &mut Config, verbose: bool) -> Result<()> {
    resolve_llama_server_binary(cfg, verbose)?;
    ensure_models(cfg, verbose)?;
    Ok(())
}

fn resolve_llama_server_binary(cfg: &mut Config, verbose: bool) -> Result<()> {
    let fork_bin = fork_bin_path(&cfg.apps_dir);

    if let Some(bin) = resolve_llama_server(&fork_bin, &cfg.llama_server_bin) {
        cfg.llama_server_bin = bin.clone();
        if verbose {
            eprintln!("[bootstrap] llama-server resolved at {}", bin.display());
        }
        return Ok(());
    }

    eprintln!(
        "[bootstrap] llama-server not found; installing MBZUAI-IFM fork into {}",
        cfg.apps_dir.display()
    );
    install_llama_server(cfg)?;
    cfg.llama_server_bin = fork_bin.clone();
    eprintln!(
        "[bootstrap] llama-server installed at {}",
        fork_bin.display()
    );
    Ok(())
}

fn install_llama_server(cfg: &Config) -> Result<()> {
    fs::create_dir_all(&cfg.apps_dir)?;
    let repo_dir = cfg.apps_dir.join("llama.cpp");

    if !repo_dir.exists() {
        eprintln!("[bootstrap] cloning MBZUAI-IFM/llama.cpp (branch model/K2Horizon) ...");
        let status = Command::new("git")
            .arg("clone")
            .arg("--depth")
            .arg("1")
            .arg("-b")
            .arg("model/K2Horizon")
            .arg("https://github.com/MBZUAI-IFM/llama.cpp.git")
            .arg(&repo_dir)
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .status()?;
        if !status.success() {
            return Err(KnowledgeError::Other(format!(
                "git clone failed ({status})"
            )));
        }
    }

    let build_dir = repo_dir.join("build");
    fs::create_dir_all(&build_dir)?;

    eprintln!("[bootstrap] configuring llama.cpp build ...");
    let mut configure = Command::new("cmake");
    configure
        .arg("-S")
        .arg(&repo_dir)
        .arg("-B")
        .arg(&build_dir)
        .arg("-DCMAKE_BUILD_TYPE=Release");
    run_with_stderr_tee(&mut configure, "cmake configure")?;

    let parallelism = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    eprintln!("[bootstrap] building llama.cpp (jobs={parallelism}) ...");
    let mut build = Command::new("cmake");
    build
        .arg("--build")
        .arg(&build_dir)
        .arg("--config")
        .arg("Release")
        .arg("-j")
        .arg(parallelism.to_string());
    run_with_stderr_tee(&mut build, "cmake build")?;

    Ok(())
}

/// Run a command with stdout inherited and stderr tee'd to the terminal while
/// keeping the tail of stderr for error messages.
fn run_with_stderr_tee(cmd: &mut Command, label: &str) -> Result<()> {
    let mut child = cmd
        .stdout(Stdio::inherit())
        .stderr(Stdio::piped())
        .spawn()?;
    let mut stderr = child
        .stderr
        .take()
        .ok_or_else(|| KnowledgeError::Other(format!("{label}: could not capture stderr")))?;

    let mut buf = [0u8; 1024];
    let mut rolling = Vec::with_capacity(4096);

    loop {
        let n = stderr.read(&mut buf)?;
        if n == 0 {
            break;
        }
        std::io::stderr().write_all(&buf[..n])?;
        rolling.extend_from_slice(&buf[..n]);
        if rolling.len() > 4096 {
            rolling.drain(..rolling.len() - 4096);
        }
    }

    let status = child.wait()?;
    if !status.success() {
        let tail = String::from_utf8_lossy(&rolling);
        return Err(KnowledgeError::Other(format!(
            "{label} failed ({status})\nlast stderr:\n{tail}"
        )));
    }
    Ok(())
}

fn ensure_models(cfg: &Config, verbose: bool) -> Result<()> {
    fs::create_dir_all(&cfg.models_dir)?;

    let model_names = [&cfg.embed_model, &cfg.gen_model];
    for name in model_names {
        let path = cfg.models_dir.join(name);
        if !needs_download(&path) {
            if verbose {
                eprintln!("[bootstrap] model {name} already present");
            }
            continue;
        }

        match model_registry_lookup(name) {
            Some(url) => {
                eprintln!("[bootstrap] downloading model {name} ...");
                download_model(&cfg.models_dir, name, url)?;
                eprintln!("[bootstrap] model {name} ready");
            }
            None => {
                return Err(KnowledgeError::Other(format!(
                    "Model file {name} is not in the bootstrap registry; place it manually at {}",
                    path.display()
                )));
            }
        }
    }

    Ok(())
}

fn download_model(models_dir: &Path, name: &str, url: &str) -> Result<()> {
    let part_path = models_dir.join(format!("{name}.part"));

    if part_path.exists() {
        fs::remove_file(&part_path)?;
    }

    let status = Command::new("curl")
        .arg("-L")
        .arg("-f")
        .arg("-o")
        .arg(&part_path)
        .arg(url)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()?;

    if !status.success() {
        let _ = fs::remove_file(&part_path);
        return Err(KnowledgeError::Other(format!(
            "curl download of {name} failed ({status})"
        )));
    }

    let final_path = models_dir.join(name);
    fs::rename(&part_path, &final_path).map_err(|e| {
        let _ = fs::remove_file(&part_path);
        KnowledgeError::Io(e)
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_llama_server_prefers_fork_then_configured() {
        let tmp =
            std::env::temp_dir().join(format!("knowledge-resolve-test-{}", std::process::id()));
        fs::create_dir_all(&tmp).unwrap();

        let fork = tmp.join("fork").join("llama-server");
        let configured = tmp.join("configured").join("llama-server");
        fs::create_dir_all(fork.parent().unwrap()).unwrap();
        fs::create_dir_all(configured.parent().unwrap()).unwrap();

        fs::write(&fork, "").unwrap();
        assert_eq!(resolve_llama_server(&fork, &configured), Some(fork.clone()));

        fs::remove_file(&fork).unwrap();
        fs::write(&configured, "").unwrap();
        assert_eq!(
            resolve_llama_server(&fork, &configured),
            Some(configured.clone())
        );

        fs::remove_file(&configured).unwrap();
        assert_eq!(resolve_llama_server(&fork, &configured), None);

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn model_registry_lookup_is_case_sensitive() {
        assert!(model_registry_lookup("bge-m3-Q8_0.gguf").is_some());
        assert!(model_registry_lookup("LFM2.5-230M-F16.gguf").is_some());
        assert!(model_registry_lookup("BGE-M3-Q8_0.GGUF").is_none());
        assert!(model_registry_lookup("unknown.gguf").is_none());
    }

    #[test]
    fn needs_download_reports_missing_files() {
        let tmp = std::env::temp_dir().join(format!("knowledge-needs-test-{}", std::process::id()));
        fs::create_dir_all(&tmp).unwrap();

        let existing = tmp.join("exists.gguf");
        fs::write(&existing, "x").unwrap();
        assert!(!needs_download(&existing));
        assert!(needs_download(&tmp.join("missing.gguf")));

        let _ = fs::remove_dir_all(&tmp);
    }
}
