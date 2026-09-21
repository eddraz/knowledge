//! Self-update command for the `knowledge` CLI.
//!
//! Detects whether the running binary was installed via `cargo install` or from
//! a GitHub Release, then updates in place using the appropriate source.

use std::cmp::Ordering;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use serde_json::Value;

use crate::error::{KnowledgeError, Result};

/// Update entry point. Dispatches to `cargo install` for cargo installs or to
/// a GitHub Release download-and-replace path for release binaries.
pub fn cmd_update(verbose: bool) -> Result<()> {
    let exe_path = env::current_exe()?;
    let cargo_bin_dir = cargo_bin_dir();
    let mode = detect_install_mode(&exe_path, &cargo_bin_dir);
    let current = env!("CARGO_PKG_VERSION");

    if verbose {
        eprintln!("[update] current version: {current}");
        eprintln!("[update] install mode: {mode:?}");
    }

    match mode {
        InstallMode::Cargo => update_cargo(current, &cargo_bin_dir, verbose),
        InstallMode::Release => update_release(current, &exe_path, verbose),
    }
}

#[derive(Debug, Clone, PartialEq)]
enum InstallMode {
    Cargo,
    Release,
}

/// Pure install-mode detection rule.
///
/// Cargo installs live in `$CARGO_HOME/bin` (or `~/.cargo/bin`), so when the
/// current executable's parent directory matches that path we delegate to
/// `cargo install`. Everything else is treated as a release binary.
fn detect_install_mode(exe_path: &Path, cargo_bin_dir: &Path) -> InstallMode {
    match exe_path.parent() {
        Some(parent) if parent == cargo_bin_dir => InstallMode::Cargo,
        _ => InstallMode::Release,
    }
}

fn cargo_bin_dir() -> PathBuf {
    env::var_os("CARGO_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| dirs::home_dir().unwrap_or_default().join(".cargo"))
        .join("bin")
}

fn update_cargo(current: &str, cargo_bin_dir: &Path, verbose: bool) -> Result<()> {
    let url = "https://crates.io/api/v1/crates/knowledge-cli";
    if verbose {
        eprintln!("[update] querying crates.io ...");
    }
    let json = curl_json(url, current)?;
    let remote = json
        .get("crate")
        .and_then(|c| c.get("max_stable_version"))
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            KnowledgeError::Other("could not parse crates.io response for max_stable_version".into())
        })?;

    if version_compare(current, remote) == Ordering::Equal {
        println!("already up to date ({current})");
        return Ok(());
    }

    let cargo_ok = Command::new("cargo")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !cargo_ok {
        return Err(KnowledgeError::Other(
            "cargo is not on PATH; cannot update a cargo-installed knowledge-cli".into(),
        ));
    }

    if verbose {
        eprintln!("[update] running cargo install knowledge-cli --force ...");
    }
    let mut cmd = Command::new("cargo");
    cmd.arg("install").arg("knowledge-cli").arg("--force");
    if cargo_bin_dir.join("native-ask").exists() {
        if verbose {
            eprintln!("[update] preserving native feature");
        }
        cmd.arg("--features").arg("native");
    }
    let status = cmd.stdout(Stdio::inherit()).stderr(Stdio::inherit()).status()?;
    if !status.success() {
        return Err(KnowledgeError::Other(format!(
            "cargo install knowledge-cli failed ({status})"
        )));
    }
    println!("updated: {current} -> {remote}");
    Ok(())
}

fn update_release(current: &str, exe_path: &Path, verbose: bool) -> Result<()> {
    let target = target_triple()?;
    let url = "https://api.github.com/repos/eddraz/knowledge/releases/latest";
    if verbose {
        eprintln!("[update] querying GitHub releases for {target} ...");
    }
    let json = curl_json(url, current)?;
    let tag = json
        .get("tag_name")
        .and_then(|t| t.as_str())
        .ok_or_else(|| KnowledgeError::Other("could not parse GitHub release tag_name".into()))?;
    let remote_version = tag.strip_prefix('v').unwrap_or(tag);
    let assets = json
        .get("assets")
        .and_then(|a| a.as_array())
        .ok_or_else(|| KnowledgeError::Other("could not parse GitHub release assets".into()))?;

    if version_compare(current, remote_version) != Ordering::Less {
        println!("already up to date ({current})");
        return Ok(());
    }

    let name = asset_name(remote_version, target);
    let found = assets
        .iter()
        .any(|a| a.get("name").and_then(|n| n.as_str()) == Some(&name));
    if !found {
        let available: Vec<String> = assets
            .iter()
            .filter_map(|a| a.get("name").and_then(|n| n.as_str()))
            .map(|s| s.to_string())
            .collect();
        return Err(KnowledgeError::Other(format!(
            "release asset {name} not found; available assets: {}",
            available.join(", ")
        )));
    }

    let tmp_dir = env::temp_dir().join(format!("knowledge-update-{}", std::process::id()));
    fs::create_dir_all(&tmp_dir)?;

    let asset_url = format!(
        "https://github.com/eddraz/knowledge/releases/download/{tag}/{name}"
    );
    let archive_path = tmp_dir.join(&name);

    if verbose {
        eprintln!("[update] downloading {name} ...");
    }
    download_file(&asset_url, &archive_path, verbose)?;

    if verbose {
        eprintln!("[update] extracting archive ...");
    }
    let status = Command::new("tar")
        .arg("-xzf")
        .arg(&archive_path)
        .arg("-C")
        .arg(&tmp_dir)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()?;
    if !status.success() {
        return Err(KnowledgeError::Other(format!(
            "tar extraction failed ({status})"
        )));
    }

    let extracted_top = tmp_dir.join(format!("knowledge-{remote_version}-{target}"));
    let knowledge_new = find_in_dir(&extracted_top, "knowledge")?;
    let native_ask_new = find_in_dir(&extracted_top, "native-ask").ok();

    let knowledge_target = exe_path.to_path_buf();
    let knowledge_old = install_binary(&knowledge_new, &knowledge_target, verbose)?;

    let native_old = if let Some(native_ask_new) = native_ask_new {
        let native_ask_target = exe_path.with_file_name("native-ask");
        install_binary(&native_ask_new, &native_ask_target, verbose)?
    } else {
        None
    };

    match Command::new(&knowledge_target).arg("--version").output() {
        Ok(output) if output.status.success() => {
            let new_version = String::from_utf8_lossy(&output.stdout)
                .trim()
                .to_string();
            println!("updated: {current} -> {new_version}");
            if let Some(old) = knowledge_old {
                let _ = fs::remove_file(old);
            }
            if let Some(old) = native_old {
                let _ = fs::remove_file(old);
            }
            let _ = fs::remove_dir_all(&tmp_dir);
            Ok(())
        }
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let backup = knowledge_old
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_default();
            Err(KnowledgeError::Other(format!(
                "update verification failed ({}); backup kept at {backup}\nstderr: {stderr}",
                output.status
            )))
        }
        Err(e) => {
            let backup = knowledge_old
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_default();
            Err(KnowledgeError::Other(format!(
                "update verification failed to run: {e}; backup kept at {backup}"
            )))
        }
    }
}

fn curl_json(url: &str, current_version: &str) -> Result<Value> {
    let output = Command::new("curl")
        .arg("-s")
        .arg("-f")
        .arg("-H")
        .arg(format!("User-Agent: knowledge-cli/{current_version}"))
        .arg(url)
        .output()?;
    if !output.status.success() {
        return Err(KnowledgeError::Other(format!(
            "curl request failed ({}): {url}",
            output.status
        )));
    }
    let text = String::from_utf8(output.stdout)
        .map_err(|e| KnowledgeError::Other(format!("invalid UTF-8 from curl: {e}")))?;
    serde_json::from_str(&text).map_err(|e| KnowledgeError::Other(format!("JSON parse error: {e}")))
}

fn download_file(url: &str, dest: &Path, _verbose: bool) -> Result<()> {
    let part_path = dest.with_extension("part");
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
            "curl download failed ({status}): {url}"
        )));
    }
    fs::rename(&part_path, dest).map_err(|e| {
        let _ = fs::remove_file(&part_path);
        KnowledgeError::Io(e)
    })?;
    Ok(())
}

fn find_in_dir(dir: &Path, name: &str) -> Result<PathBuf> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() && path.file_name().and_then(|n| n.to_str()) == Some(name) {
            return Ok(path);
        }
    }
    Err(KnowledgeError::Other(format!(
        "could not find {name} in extracted archive"
    )))
}

fn install_binary(new: &Path, target: &Path, verbose: bool) -> Result<Option<PathBuf>> {
    let new_path = target.with_extension("new");
    let old_path = target.with_extension("old");

    if verbose {
        eprintln!("[update] replacing {} ...", target.display());
    }
    fs::copy(new, &new_path)?;
    let had_old = target.exists();
    if had_old {
        fs::rename(target, &old_path)?;
    }
    fs::rename(&new_path, target)?;
    set_executable(target)?;
    Ok(if had_old { Some(old_path) } else { None })
}

#[cfg(unix)]
fn set_executable(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let mut perms = fs::metadata(path)?.permissions();
    perms.set_mode(0o755);
    fs::set_permissions(path, perms)?;
    Ok(())
}

#[cfg(not(unix))]
fn set_executable(_path: &Path) -> Result<()> {
    Ok(())
}

/// Parse a `major.minor.patch` version string into a numeric triple.
fn parse_version(v: &str) -> Option<(u64, u64, u64)> {
    let mut parts = v.split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next()?.parse().ok()?;
    let patch = parts.next()?.parse().ok()?;
    if parts.next().is_some() {
        return None;
    }
    Some((major, minor, patch))
}

/// Compare two version strings. Falls back to string inequality when either
/// side cannot be parsed as `major.minor.patch`.
fn version_compare(current: &str, remote: &str) -> Ordering {
    match (parse_version(current), parse_version(remote)) {
        (Some(c), Some(r)) => c.cmp(&r),
        _ => {
            if current == remote {
                Ordering::Equal
            } else {
                // String fallback: anything different is treated as "remote is
                // newer" for update decisions. This preserves the old behavior
                // of triggering an update when versions are not comparable.
                Ordering::Less
            }
        }
    }
}

fn target_triple() -> Result<&'static str> {
    target_triple_for(env::consts::OS, env::consts::ARCH)
}

fn target_triple_for(os: &str, arch: &str) -> Result<&'static str> {
    match (os, arch) {
        ("linux", "x86_64") => Ok("x86_64-unknown-linux-gnu"),
        ("linux", "aarch64") => Ok("aarch64-unknown-linux-gnu"),
        ("macos", "aarch64") => Ok("aarch64-apple-darwin"),
        ("macos", "x86_64") => Ok("x86_64-apple-darwin"),
        _ => Err(KnowledgeError::Other(format!(
            "release binaries are not available for {arch}-{os}; install with `cargo install knowledge-cli` instead"
        ))),
    }
}

fn asset_name(version: &str, target: &str) -> String {
    format!("knowledge-{version}-{target}.tar.gz")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_install_mode_cargo_when_exe_in_cargo_bin() {
        let cargo_bin = PathBuf::from("/home/user/.cargo/bin");
        let exe = cargo_bin.join("knowledge");
        assert_eq!(detect_install_mode(&exe, &cargo_bin), InstallMode::Cargo);
    }

    #[test]
    fn detect_install_mode_release_when_exe_elsewhere() {
        let cargo_bin = PathBuf::from("/home/user/.cargo/bin");
        let exe = PathBuf::from("/usr/local/bin/knowledge");
        assert_eq!(detect_install_mode(&exe, &cargo_bin), InstallMode::Release);
    }

    #[test]
    fn version_compare_numeric_triple() {
        assert_eq!(version_compare("0.1.0", "0.1.0"), Ordering::Equal);
        assert_eq!(version_compare("0.1.9", "0.1.10"), Ordering::Less);
        assert_eq!(version_compare("0.2.0", "0.1.10"), Ordering::Greater);
        assert_eq!(version_compare("1.0.0", "0.9.9"), Ordering::Greater);
    }

    #[test]
    fn version_compare_malformed_fallback() {
        assert_eq!(version_compare("0.1.0", "0.1.0-alpha"), Ordering::Less);
        assert_eq!(version_compare("0.1.0-alpha", "0.1.0-alpha"), Ordering::Equal);
        assert_eq!(version_compare("v0.1.0", "0.1.0"), Ordering::Less);
    }

    #[test]
    fn target_triple_maps_supported_platforms() {
        assert_eq!(
            target_triple_for("linux", "x86_64").unwrap(),
            "x86_64-unknown-linux-gnu"
        );
        assert_eq!(
            target_triple_for("linux", "aarch64").unwrap(),
            "aarch64-unknown-linux-gnu"
        );
        assert_eq!(
            target_triple_for("macos", "aarch64").unwrap(),
            "aarch64-apple-darwin"
        );
        assert_eq!(
            target_triple_for("macos", "x86_64").unwrap(),
            "x86_64-apple-darwin"
        );
    }

    #[test]
    fn target_triple_rejects_unsupported_platform() {
        assert!(target_triple_for("windows", "x86_64").is_err());
        assert!(target_triple_for("linux", "i686").is_err());
        assert!(target_triple_for("freebsd", "x86_64").is_err());
    }

    #[test]
    fn asset_name_builder() {
        assert_eq!(
            asset_name("0.2.0", "x86_64-unknown-linux-gnu"),
            "knowledge-0.2.0-x86_64-unknown-linux-gnu.tar.gz"
        );
        assert_eq!(
            asset_name("1.0.0-beta.1", "aarch64-apple-darwin"),
            "knowledge-1.0.0-beta.1-aarch64-apple-darwin.tar.gz"
        );
    }
}
