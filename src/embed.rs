//! Embeddings abstraction: sidecar HTTP or native candle bge-m3.
//!
//! The public entry point is [`embed_texts`], which resolves the active
//! embedder from the compile-time `native` feature and the `KNOWLEDGE_EMBEDDER`
//! environment variable, then dispatches to the appropriate backend.

use reqwest::Client;

use crate::config::Config;
use crate::error::{KnowledgeError, Result};



/// Active embedding backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbedderMode {
    /// In-process candle bge-m3 (PyTorch checkpoint).
    Native,
    /// llama-server `--embedding` sidecar (GGUF).
    Sidecar,
}

/// Resolve the embedder mode from the compile-time feature flag and the
/// optional `KNOWLEDGE_EMBEDDER` environment variable.
///
/// Rules:
/// * `native` feature present, no env → `Native`
/// * `native` feature present, `sidecar` → `Sidecar`
/// * no `native` feature, no env → `Sidecar`
/// * no `native` feature, `native` → error
/// * any other env value → error
pub fn resolve_embedder_mode(feature_native: bool, env_value: Option<&str>) -> Result<EmbedderMode> {
    match env_value {
        Some("native") => {
            if feature_native {
                Ok(EmbedderMode::Native)
            } else {
                Err(KnowledgeError::Other(
                    "KNOWLEDGE_EMBEDDER=native requires the native feature; \
                     this binary was built without it"
                        .to_string(),
                ))
            }
        }
        Some("sidecar") => Ok(EmbedderMode::Sidecar),
        Some(v) => Err(KnowledgeError::Other(format!(
            "invalid KNOWLEDGE_EMBEDDER value: {v}; expected 'native' or 'sidecar'"
        ))),
        None => {
            if feature_native {
                Ok(EmbedderMode::Native)
            } else {
                Ok(EmbedderMode::Sidecar)
            }
        }
    }
}

/// Embed a batch of texts using the configured backend.
///
/// For the sidecar backend the caller is responsible for ensuring the
/// llama-server embedding sidecar is healthy before calling this function.
/// For the native backend the model and tokenizer are loaded once per process
/// on first use.
pub async fn embed_texts(http: &Client, cfg: &Config, texts: &[String]) -> Result<Vec<Vec<f32>>> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }

    match cfg.embedder_mode {
        EmbedderMode::Sidecar => {
            crate::llm::embed_texts(http, &cfg.embed_base_url(), texts).await
        }
        #[cfg(feature = "native")]
        EmbedderMode::Native => native::native_embed_texts(cfg, texts).await,
        #[cfg(not(feature = "native"))]
        EmbedderMode::Native => Err(KnowledgeError::Other(
            "native embedder selected but binary lacks the `native` feature".to_string(),
        )),
    }
}

#[cfg(feature = "native")]
mod native {
    use std::sync::{Arc, OnceLock};

    use candle::{DType, Device, Tensor};
    use candle_nn::VarBuilder;
    use candle_transformers::models::xlm_roberta::{Config as XlmConfig, XLMRobertaModel};
    use tokenizers::{PaddingParams, Tokenizer, TruncationParams};

    use crate::config::Config;
    use crate::error::{KnowledgeError, Result};
    use crate::llm::EMBED_DIM;

    const MAX_LENGTH: usize = 8192;

    pub struct NativeEmbedder {
        model: XLMRobertaModel,
        tokenizer: Tokenizer,
    }

    impl NativeEmbedder {
        pub fn load(cfg: &Config) -> Result<Self> {
            let device = Device::Cpu;
            let dir = cfg.native_embed_dir();
            let config_path = dir.join("config.json");
            let weights_path = dir.join("pytorch_model.bin");
            let tokenizer_path = dir.join("tokenizer.json");

            let config: XlmConfig = serde_json::from_str(
                &std::fs::read_to_string(&config_path).map_err(KnowledgeError::Io)?,
            )
            .map_err(|e| {
                KnowledgeError::Other(format!("failed to parse bge-m3 config.json: {e}"))
            })?;

            // The official BAAI/bge-m3 repository ships `pytorch_model.bin`, not
            // `model.safetensors`, so we load the PyTorch checkpoint directly.
            let vb = VarBuilder::from_pth(&weights_path, DType::F32, &device).map_err(|e| {
                KnowledgeError::Other(format!("failed to load bge-m3 weights: {e}"))
            })?;

            // The BAAI/bge-m3 checkpoint stores tensors without a `roberta.`
            // prefix (verified against pytorch_model.bin), so the VarBuilder
            // root maps directly onto XLMRobertaModel.
            let model = XLMRobertaModel::new(&config, vb).map_err(|e| {
                KnowledgeError::Other(format!("failed to build bge-m3 model: {e}"))
            })?;

            let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
                KnowledgeError::Other(format!("failed to load bge-m3 tokenizer: {e}"))
            })?;

            Ok(Self { model, tokenizer })
        }

        pub fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            let mut tokenizer = self.tokenizer.clone();
            tokenizer.with_padding(Some(PaddingParams::default()));
            tokenizer
                .with_truncation(Some(TruncationParams {
                    max_length: MAX_LENGTH,
                    ..TruncationParams::default()
                }))
                .map_err(|e| {
                    KnowledgeError::Other(format!("failed to configure truncation: {e}"))
                })?;

            let inputs: Vec<String> = texts.to_vec();
            let encodings = tokenizer
                .encode_batch(inputs, true)
                .map_err(|e| KnowledgeError::Other(format!("tokenization failed: {e}")))?;

            let max_len = encodings.iter().map(|e| e.len()).max().unwrap_or(0);
            let batch_size = encodings.len();

            let mut input_ids = Vec::with_capacity(batch_size * max_len);
            let mut attention_mask = Vec::with_capacity(batch_size * max_len);
            let mut token_type_ids = Vec::with_capacity(batch_size * max_len);

            for enc in &encodings {
                let ids = enc.get_ids();
                let mask = enc.get_attention_mask();
                let types = enc.get_type_ids();
                for i in 0..max_len {
                    if i < ids.len() {
                        input_ids.push(ids[i]);
                        attention_mask.push(mask[i]);
                        token_type_ids.push(types[i]);
                    } else {
                        input_ids.push(0);
                        attention_mask.push(0);
                        token_type_ids.push(0);
                    }
                }
            }

            let input_ids = Tensor::new(input_ids, &Device::Cpu)
                .map_err(|e| KnowledgeError::Other(format!("input_ids tensor error: {e}")))?
                .reshape((batch_size, max_len))
                .map_err(|e| KnowledgeError::Other(format!("input_ids reshape error: {e}")))?
                .to_dtype(DType::U32)
                .map_err(|e| KnowledgeError::Other(format!("input_ids dtype error: {e}")))?;
            let attention_mask = Tensor::new(attention_mask, &Device::Cpu)
                .map_err(|e| KnowledgeError::Other(format!("attention_mask tensor error: {e}")))?
                .reshape((batch_size, max_len))
                .map_err(|e| KnowledgeError::Other(format!("attention_mask reshape error: {e}")))?
                .to_dtype(DType::U32)
                .map_err(|e| KnowledgeError::Other(format!("attention_mask dtype error: {e}")))?;
            let token_type_ids = Tensor::new(token_type_ids, &Device::Cpu)
                .map_err(|e| KnowledgeError::Other(format!("token_type_ids tensor error: {e}")))?
                .reshape((batch_size, max_len))
                .map_err(|e| KnowledgeError::Other(format!("token_type_ids reshape error: {e}")))?
                .to_dtype(DType::U32)
                .map_err(|e| KnowledgeError::Other(format!("token_type_ids dtype error: {e}")))?;

            let hidden_states = self
                .model
                .forward(
                    &input_ids,
                    &attention_mask,
                    &token_type_ids,
                    None,
                    None,
                    None,
                )
                .map_err(|e| KnowledgeError::Other(format!("bge-m3 forward failed: {e}")))?;

            // CLS pooling: position 0 of the last hidden states.
            let cls = hidden_states
                .get_on_dim(1, 0)
                .map_err(|e| KnowledgeError::Other(format!("CLS extraction failed: {e}")))?;

            // L2 normalization so cosine distance semantics in sqlite-vec stay valid.
            let normalized = l2_normalize(&cls)
                .map_err(|e| KnowledgeError::Other(format!("normalization failed: {e}")))?;

            let out = normalized
                .to_vec2::<f32>()
                .map_err(|e| KnowledgeError::Other(format!("tensor conversion failed: {e}")))?;
            if out.first().map(|v| v.len()) != Some(EMBED_DIM) {
                return Err(KnowledgeError::Other(format!(
                    "expected embedding dimension {EMBED_DIM}, got {:?}",
                    out.first().map(|v| v.len())
                )));
            }
            Ok(out)
        }
    }

    /// L2-normalize each row of a 2-D tensor.
    pub(super) fn l2_normalize(x: &Tensor) -> candle::Result<Tensor> {
        let norm = x.sqr()?.sum(1)?.sqrt()?;
        x.broadcast_div(&norm.unsqueeze(1)?)
    }

    static NATIVE: OnceLock<std::result::Result<Arc<NativeEmbedder>, String>> = OnceLock::new();

    fn get_native(cfg: &Config) -> std::result::Result<Arc<NativeEmbedder>, String> {
        NATIVE
            .get_or_init(|| NativeEmbedder::load(cfg).map(Arc::new).map_err(|e| e.to_string()))
            .clone()
    }

    pub async fn native_embed_texts(cfg: &Config, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let cfg = cfg.clone();
        let texts = texts.to_vec();

        let native = tokio::task::spawn_blocking(move || get_native(&cfg))
            .await
            .map_err(|e| KnowledgeError::Other(format!("bge-m3 load panicked: {e}")))?
            .map_err(KnowledgeError::Other)?;

        let native = native.clone();
        tokio::task::spawn_blocking(move || native.embed(&texts))
            .await
            .map_err(|e| KnowledgeError::Other(format!("bge-m3 embed panicked: {e}")))?
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_embedder_mode_defaults_and_env() {
        assert_eq!(resolve_embedder_mode(true, None).unwrap(), EmbedderMode::Native);
        assert_eq!(
            resolve_embedder_mode(true, Some("sidecar")).unwrap(),
            EmbedderMode::Sidecar
        );
        assert_eq!(
            resolve_embedder_mode(false, Some("sidecar")).unwrap(),
            EmbedderMode::Sidecar
        );
        assert_eq!(
            resolve_embedder_mode(false, None).unwrap(),
            EmbedderMode::Sidecar
        );

        let err = resolve_embedder_mode(false, Some("native")).unwrap_err();
        assert!(matches!(err, KnowledgeError::Other(_)));
        assert!(err.to_string().contains("native feature"));

        let err = resolve_embedder_mode(true, Some("invalid")).unwrap_err();
        assert!(matches!(err, KnowledgeError::Other(_)));
        assert!(err.to_string().contains("invalid KNOWLEDGE_EMBEDDER"));
    }

    #[cfg(feature = "native")]
    #[test]
    fn l2_normalize_unit_length() {
        use candle::Device;

        let x = candle::Tensor::new(&[[3.0_f32, 4.0_f32], [1.0_f32, 0.0_f32]], &Device::Cpu)
            .unwrap();
        let normalized = native::l2_normalize(&x).unwrap();
        let out = normalized.to_vec2::<f32>().unwrap();

        assert!((out[0][0] - 0.6).abs() < 1e-6);
        assert!((out[0][1] - 0.8).abs() < 1e-6);
        assert!((out[1][0] - 1.0).abs() < 1e-6);
        assert!(out[1][1].abs() < 1e-6);
    }

    #[cfg(feature = "native")]
    #[test]
    fn cls_selection_takes_first_position() {
        use candle::Device;

        let hidden =
            candle::Tensor::new(
                &[[[1.0_f32, 2.0_f32], [3.0_f32, 4.0_f32]],
                  [[5.0_f32, 6.0_f32], [7.0_f32, 8.0_f32]]],
                &Device::Cpu,
            )
            .unwrap();
        let cls = hidden.get_on_dim(1, 0).unwrap();
        let out = cls.to_vec2::<f32>().unwrap();

        assert_eq!(out, vec![vec![1.0, 2.0], vec![5.0, 6.0]]);
    }
}
