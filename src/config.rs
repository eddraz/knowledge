use std::env;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use crate::embed::{resolve_embedder_mode, EmbedderMode};
use crate::error::{KnowledgeError, Result};

const DEFAULT_EMBED_MODEL: &str = "bge-m3-Q8_0.gguf";
const DEFAULT_GEN_MODEL: &str = "LFM2.5-230M-F16.gguf";

#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub db_path: PathBuf,
    pub llama_server_bin: PathBuf,
    pub models_dir: PathBuf,
    pub apps_dir: PathBuf,
    pub embed_model: String,
    pub gen_model: String,
    pub embed_port: u16,
    pub gen_port: u16,
    pub chunk_target_chars: usize,
    pub chunk_overlap_chars: usize,
    pub top_k: usize,
    pub min_score: f32,
    pub request_timeout_secs: u64,
    pub embedder_mode: EmbedderMode,
}

impl Config {
    pub fn load() -> Result<Self> {
        let db_path = env_path("KNOWLEDGE_DB").unwrap_or_else(default_db_path);
        let llama_server_bin =
            env_path("KNOWLEDGE_LLAMA_SERVER").unwrap_or_else(default_llama_server_bin);
        let models_dir = env_path("KNOWLEDGE_MODELS_DIR").unwrap_or_else(default_models_dir);
        let apps_dir = env_path("KNOWLEDGE_APPS_DIR").unwrap_or_else(default_apps_dir);

        let embed_port = env_parse("KNOWLEDGE_EMBED_PORT")?.unwrap_or(8098);
        let gen_port = env_parse("KNOWLEDGE_GEN_PORT")?.unwrap_or(8099);
        let top_k = env_parse("KNOWLEDGE_TOP_K")?.unwrap_or(5);
        let min_score = env_parse("KNOWLEDGE_MIN_SCORE")?.unwrap_or(0.35);
        let request_timeout_secs = env_parse("KNOWLEDGE_TIMEOUT_SECS")?.unwrap_or(120);

        let embedder_env = std::env::var("KNOWLEDGE_EMBEDDER").ok();
        #[cfg(feature = "native")]
        let embedder_mode = resolve_embedder_mode(true, embedder_env.as_deref())?;
        #[cfg(not(feature = "native"))]
        let embedder_mode = resolve_embedder_mode(false, embedder_env.as_deref())?;

        Self::from_parts(
            db_path,
            llama_server_bin,
            models_dir,
            env_string("KNOWLEDGE_EMBED_MODEL", DEFAULT_EMBED_MODEL),
            env_string("KNOWLEDGE_GEN_MODEL", DEFAULT_GEN_MODEL),
            embed_port,
            gen_port,
            600,
            90,
            top_k,
            min_score,
            request_timeout_secs,
            apps_dir,
            embedder_mode,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        db_path: impl AsRef<Path>,
        llama_server_bin: impl AsRef<Path>,
        models_dir: impl AsRef<Path>,
        embed_model: impl Into<String>,
        gen_model: impl Into<String>,
        embed_port: u16,
        gen_port: u16,
        chunk_target_chars: usize,
        chunk_overlap_chars: usize,
        top_k: usize,
        min_score: f32,
        request_timeout_secs: u64,
        apps_dir: impl AsRef<Path>,
        embedder_mode: EmbedderMode,
    ) -> Result<Self> {
        Ok(Self {
            db_path: db_path.as_ref().to_path_buf(),
            llama_server_bin: llama_server_bin.as_ref().to_path_buf(),
            models_dir: models_dir.as_ref().to_path_buf(),
            apps_dir: apps_dir.as_ref().to_path_buf(),
            embed_model: embed_model.into(),
            gen_model: gen_model.into(),
            embed_port,
            gen_port,
            chunk_target_chars,
            chunk_overlap_chars,
            top_k,
            min_score,
            request_timeout_secs,
            embedder_mode,
        })
    }

    pub fn embed_model_path(&self) -> PathBuf {
        self.models_dir.join(&self.embed_model)
    }

    pub fn gen_model_path(&self) -> PathBuf {
        self.models_dir.join(&self.gen_model)
    }

    /// Directory holding the native candle bge-m3 files (config.json,
    /// tokenizer.json, pytorch_model.bin).
    pub fn native_embed_dir(&self) -> PathBuf {
        self.models_dir.join("bge-m3")
    }

    pub fn native_embed_config_path(&self) -> PathBuf {
        self.native_embed_dir().join("config.json")
    }

    pub fn native_embed_tokenizer_path(&self) -> PathBuf {
        self.native_embed_dir().join("tokenizer.json")
    }

    pub fn native_embed_weights_path(&self) -> PathBuf {
        self.native_embed_dir().join("pytorch_model.bin")
    }

    pub fn embed_base_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.embed_port)
    }

    pub fn gen_base_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.gen_port)
    }
}

fn default_db_path() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("knowledge")
        .join("knowledge.db")
}

fn default_llama_server_bin() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".local")
        .join("bin")
        .join("llama-server")
}

fn default_models_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("models")
}

fn default_apps_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("apps")
}

fn env_string(name: &str, default: &str) -> String {
    env::var(name).unwrap_or_else(|_| default.to_string())
}

fn env_path(name: &str) -> Option<PathBuf> {
    env::var(name).ok().map(PathBuf::from)
}

fn env_parse<T: FromStr>(name: &str) -> Result<Option<T>> {
    match env::var(name) {
        Ok(v) => v
            .parse::<T>()
            .map(Some)
            .map_err(|_| KnowledgeError::Other(format!("invalid value for {name}: {v}"))),
        Err(env::VarError::NotPresent) => Ok(None),
        Err(e) => Err(KnowledgeError::Other(format!(
            "could not read environment variable {name}: {e}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_parts_honours_values() {
        let cfg = Config::from_parts(
            "/tmp/db",
            "/tmp/llama-server",
            "/tmp/models",
            "embed.gguf",
            "gen.gguf",
            9001,
            9002,
            100,
            10,
            7,
            0.5,
            60,
            "/tmp/apps",
            EmbedderMode::Sidecar,
        )
        .unwrap();

        assert_eq!(cfg.db_path, PathBuf::from("/tmp/db"));
        assert_eq!(cfg.llama_server_bin, PathBuf::from("/tmp/llama-server"));
        assert_eq!(cfg.models_dir, PathBuf::from("/tmp/models"));
        assert_eq!(cfg.apps_dir, PathBuf::from("/tmp/apps"));
        assert_eq!(cfg.embed_model, "embed.gguf");
        assert_eq!(cfg.gen_model, "gen.gguf");
        assert_eq!(cfg.embed_port, 9001);
        assert_eq!(cfg.gen_port, 9002);
        assert_eq!(cfg.chunk_target_chars, 100);
        assert_eq!(cfg.chunk_overlap_chars, 10);
        assert_eq!(cfg.top_k, 7);
        assert_eq!(cfg.min_score, 0.5);
        assert_eq!(cfg.request_timeout_secs, 60);
        assert_eq!(cfg.embedder_mode, EmbedderMode::Sidecar);
    }

    #[test]
    fn helpers_build_paths_and_urls() {
        let cfg = Config::from_parts(
            "/tmp/db",
            "/tmp/llama-server",
            "/tmp/models",
            "embed.gguf",
            "gen.gguf",
            8098,
            8099,
            600,
            90,
            5,
            0.35,
            120,
            "/tmp/apps",
            EmbedderMode::Native,
        )
        .unwrap();

        assert_eq!(
            cfg.embed_model_path(),
            PathBuf::from("/tmp/models/embed.gguf")
        );
        assert_eq!(cfg.gen_model_path(), PathBuf::from("/tmp/models/gen.gguf"));
        assert_eq!(cfg.embed_base_url(), "http://127.0.0.1:8098");
        assert_eq!(cfg.gen_base_url(), "http://127.0.0.1:8099");
        assert_eq!(cfg.native_embed_dir(), PathBuf::from("/tmp/models/bge-m3"));
        assert_eq!(
            cfg.native_embed_config_path(),
            PathBuf::from("/tmp/models/bge-m3/config.json")
        );
        assert_eq!(
            cfg.native_embed_tokenizer_path(),
            PathBuf::from("/tmp/models/bge-m3/tokenizer.json")
        );
        assert_eq!(
            cfg.native_embed_weights_path(),
            PathBuf::from("/tmp/models/bge-m3/pytorch_model.bin")
        );
    }

    #[test]
    fn apps_dir_defaults_to_home_apps() {
        let cfg = Config::load().unwrap();
        let expected = dirs::home_dir().unwrap().join("apps");
        assert_eq!(cfg.apps_dir, expected);
    }

    #[test]
    fn apps_dir_env_override() {
        std::env::set_var("KNOWLEDGE_APPS_DIR", "/tmp/custom-apps");
        let cfg = Config::load().unwrap();
        std::env::remove_var("KNOWLEDGE_APPS_DIR");
        assert_eq!(cfg.apps_dir, PathBuf::from("/tmp/custom-apps"));
    }
}
