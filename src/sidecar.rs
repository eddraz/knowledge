use std::process::Stdio;
use std::time::Duration;

use serde_json::Value;
use tokio::process::{Child, Command};
use tokio::time::{sleep, timeout};

use crate::config::Config;
use crate::error::{KnowledgeError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SidecarRole {
    Embedding,
    Generator,
}

#[derive(Debug)]
pub struct SidecarHandle {
    /// Base URL of the sidecar; exposed for diagnostics and future commands.
    #[allow(dead_code)]
    pub base_url: String,
    pub child: Option<Child>,
}

impl Drop for SidecarHandle {
    fn drop(&mut self) {
        if let Some(ref mut child) = self.child {
            let _ = child.start_kill();
        }
    }
}

const HEALTH_PATH: &str = "/health";
const POLL_INTERVAL_MS: u64 = 500;
const STARTUP_TIMEOUT_SECS: u64 = 120;
const HEALTH_PROBE_TIMEOUT_SECS: u64 = 2;

/// Acquire a running llama-server sidecar, spawning one if necessary.
///
/// If a server is already listening at the configured base URL and reports
/// `status: "ok"`, it is reused (`child` is `None`).  Otherwise a new child
/// process is started and polled until healthy.
pub async fn acquire(cfg: &Config, role: SidecarRole) -> Result<SidecarHandle> {
    let base_url = match role {
        SidecarRole::Embedding => cfg.embed_base_url(),
        SidecarRole::Generator => cfg.gen_base_url(),
    };

    if probe_health(&base_url).await? {
        return Ok(SidecarHandle {
            base_url,
            child: None,
        });
    }

    let model_path = match role {
        SidecarRole::Embedding => cfg.embed_model_path(),
        SidecarRole::Generator => cfg.gen_model_path(),
    };

    let port = match role {
        SidecarRole::Embedding => cfg.embed_port,
        SidecarRole::Generator => cfg.gen_port,
    };

    let args: Vec<String> = match role {
        SidecarRole::Embedding => vec![
            "-m".to_string(),
            model_path.to_string_lossy().to_string(),
            "--embedding".to_string(),
            "--port".to_string(),
            port.to_string(),
            "-c".to_string(),
            "4096".to_string(),
        ],
        SidecarRole::Generator => vec![
            "-m".to_string(),
            model_path.to_string_lossy().to_string(),
            "--port".to_string(),
            port.to_string(),
            "-c".to_string(),
            "8192".to_string(),
        ],
    };

    let mut child = Command::new(&cfg.llama_server_bin)
        .args(&args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| sidecar_error(role, format!("failed to spawn llama-server: {e}")))?;

    let started = timeout(Duration::from_secs(STARTUP_TIMEOUT_SECS), async {
        loop {
            match probe_health(&base_url).await {
                Ok(true) => return Ok(()),
                Ok(false) => {}
                Err(e) => {
                    // If the child exited early, give up immediately.
                    match child.try_wait() {
                        Ok(Some(status)) => {
                            return Err(sidecar_error(
                                role,
                                format!("llama-server exited early with {status}"),
                            ));
                        }
                        Ok(None) => {}
                        Err(wait_err) => {
                            return Err(sidecar_error(
                                role,
                                format!("could not poll child status: {wait_err}"),
                            ));
                        }
                    }
                    if !is_transient(&e) {
                        return Err(e);
                    }
                }
            }
            sleep(Duration::from_millis(POLL_INTERVAL_MS)).await;
        }
    })
    .await;

    match started {
        Ok(Ok(())) => Ok(SidecarHandle {
            base_url,
            child: Some(child),
        }),
        Ok(Err(e)) => Err(e),
        Err(_) => Err(sidecar_error(
            role,
            format!("llama-server did not become healthy within {STARTUP_TIMEOUT_SECS}s"),
        )),
    }
}

fn sidecar_error(role: SidecarRole, msg: String) -> KnowledgeError {
    match role {
        SidecarRole::Embedding => KnowledgeError::EmbeddingSidecar(msg),
        SidecarRole::Generator => KnowledgeError::GeneratorSidecar(msg),
    }
}

fn is_transient(err: &KnowledgeError) -> bool {
    matches!(
        err,
        KnowledgeError::Http(_) | KnowledgeError::BadResponse(_) | KnowledgeError::Io(_)
    )
}

async fn probe_health(base_url: &str) -> Result<bool> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(HEALTH_PROBE_TIMEOUT_SECS))
        .build()
        .map_err(KnowledgeError::Http)?;

    let url = format!("{base_url}{HEALTH_PATH}");
    // A connection failure simply means the sidecar is not running yet; the
    // caller is expected to spawn it in that case.
    let response = match client.get(&url).send().await {
        Ok(response) => response,
        Err(err) if err.is_connect() || err.is_timeout() => return Ok(false),
        Err(err) => return Err(KnowledgeError::Http(err)),
    };

    if !response.status().is_success() {
        return Ok(false);
    }

    let body = response.text().await.map_err(KnowledgeError::Http)?;
    parse_health(&body)
}

/// Parse a llama-server /health response.
///
/// Returns `true` only when the JSON body contains `"status": "ok"`.
pub fn parse_health(body: &str) -> Result<bool> {
    let value: Value = serde_json::from_str(body)
        .map_err(|e| KnowledgeError::BadResponse(format!("health JSON parse error: {e}")))?;

    match value.get("status") {
        Some(Value::String(s)) => Ok(s == "ok"),
        _ => Ok(false),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_health_accepts_ok() {
        assert!(parse_health(r#"{"status":"ok"}"#).unwrap());
    }

    #[test]
    fn parse_health_rejects_other() {
        assert!(!parse_health(r#"{"status":"loading"}"#).unwrap());
        assert!(!parse_health(r#"{}"#).unwrap());
        assert!(parse_health("not json").is_err());
    }
}
