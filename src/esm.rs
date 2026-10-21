//! ESM-2 encoder in candle, matching Hugging Face `EsmModel` (pre-LayerNorm
//! transformer with rotary position embeddings).
//! candle-transformers has no ESM model, so this is written by hand against
//! the tensor names in `facebook/esm2_*` safetensors files.

use candle_core::{D, DType, Device, Module, Tensor};
use candle_nn::{Embedding, LayerNorm, Linear, VarBuilder, embedding, layer_norm, linear};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

use crate::models::{Embedder, Res};

/// ESM-2 was trained with 15% masking, 80% of which used the <mask> token.
/// HF rescales embeddings by this at inference ("token dropout").
const MASK_RATIO_TRAIN: f64 = 0.15 * 0.8;
/// 1024 positions minus <cls> and <eos>.
pub const MAX_RESIDUES: usize = 1022;

#[derive(Deserialize)]
struct Config {
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    intermediate_size: usize,
    layer_norm_eps: f64,
    vocab_size: usize,
    token_dropout: bool,
    mask_token_id: u32,
}

struct Layer {
    attn_ln: LayerNorm,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    ffn_ln: LayerNorm,
    up: Linear,
    down: Linear,
    heads: usize,
}

/// Rotate the two halves of the last dim: (x1, x2) -> (-x2, x1).
fn rotate_half(x: &Tensor) -> candle_core::Result<Tensor> {
    let half = x.dim(D::Minus1)? / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?;
    Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)
}

fn apply_rotary(x: &Tensor, cos: &Tensor, sin: &Tensor) -> candle_core::Result<Tensor> {
    x.broadcast_mul(cos)? + rotate_half(x)?.broadcast_mul(sin)?
}

impl Layer {
    fn load(vb: VarBuilder, cfg: &Config) -> candle_core::Result<Self> {
        let h = cfg.hidden_size;
        let eps = cfg.layer_norm_eps;
        Ok(Self {
            attn_ln: layer_norm(h, eps, vb.pp("attention.LayerNorm"))?,
            q: linear(h, h, vb.pp("attention.self.query"))?,
            k: linear(h, h, vb.pp("attention.self.key"))?,
            v: linear(h, h, vb.pp("attention.self.value"))?,
            o: linear(h, h, vb.pp("attention.output.dense"))?,
            ffn_ln: layer_norm(h, eps, vb.pp("LayerNorm"))?,
            up: linear(h, cfg.intermediate_size, vb.pp("intermediate.dense"))?,
            down: linear(cfg.intermediate_size, h, vb.pp("output.dense"))?,
            heads: cfg.num_attention_heads,
        })
    }

    /// x: [T, H] -> [T, H]
    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> candle_core::Result<Tensor> {
        let (t, h) = x.dims2()?;
        let hd = h / self.heads;
        // [T, H] -> [heads, T, hd]
        let split = |z: Tensor| {
            z.reshape((t, self.heads, hd))?
                .transpose(0, 1)?
                .contiguous()
        };

        let y = self.attn_ln.forward(x)?;
        // HF scales q *before* applying rotary.
        let q = split((self.q.forward(&y)? * (hd as f64).powf(-0.5))?)?;
        let k = split(self.k.forward(&y)?)?;
        let v = split(self.v.forward(&y)?)?;
        let q = apply_rotary(&q, cos, sin)?;
        let k = apply_rotary(&k, cos, sin)?;

        let scores = q.matmul(&k.transpose(1, 2)?.contiguous()?)?;
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs
            .matmul(&v)?
            .transpose(0, 1)?
            .contiguous()?
            .reshape((t, h))?;
        let x = (x + self.o.forward(&ctx)?)?;

        let y = self.ffn_ln.forward(&x)?;
        let y = self.down.forward(&self.up.forward(&y)?.gelu_erf()?)?;
        x + y
    }
}

pub struct Esm {
    cfg: Config,
    vocab: HashMap<String, u32>,
    embed: Embedding,
    layers: Vec<Layer>,
    final_ln: LayerNorm,
    lm_dense: Linear,
    lm_ln: LayerNorm,
    lm_bias: Tensor,
    device: Device,
}

impl Esm {
    /// Load from a directory containing config.json, vocab.txt and model.safetensors.
    pub fn load(dir: &Path) -> Res<Self> {
        let cfg: Config = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
        let vocab = std::fs::read_to_string(dir.join("vocab.txt"))?
            .lines()
            .enumerate()
            .map(|(i, tok)| (tok.trim().to_string(), i as u32))
            .collect();

        let device = Device::Cpu;
        // Memory-map instead of reading the file into a Vec: the OS pages weights
        // in on demand, so the file isn't held in RAM next to the loaded tensors.
        // SAFETY: the file is in the Hugging Face cache and must not be modified
        // while the model is loading; nothing in this app writes to it.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[dir.join("model.safetensors")],
                DType::F32,
                &device,
            )?
        };

        let embed = embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            vb.pp("esm.embeddings.word_embeddings"),
        )?;
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| Layer::load(vb.pp(format!("esm.encoder.layer.{i}")), &cfg))
            .collect::<candle_core::Result<Vec<_>>>()?;
        let final_ln = layer_norm(
            cfg.hidden_size,
            cfg.layer_norm_eps,
            vb.pp("esm.encoder.emb_layer_norm_after"),
        )?;
        let lm_dense = linear(cfg.hidden_size, cfg.hidden_size, vb.pp("lm_head.dense"))?;
        let lm_ln = layer_norm(
            cfg.hidden_size,
            cfg.layer_norm_eps,
            vb.pp("lm_head.layer_norm"),
        )?;
        let lm_bias = vb.get(cfg.vocab_size, "lm_head.bias")?;

        Ok(Self {
            cfg,
            vocab,
            embed,
            layers,
            final_ln,
            lm_dense,
            lm_ln,
            lm_bias,
            device,
        })
    }

    pub fn hidden_size(&self) -> usize {
        self.cfg.hidden_size
    }

    fn token(&self, name: &str) -> u32 {
        // <unk> is id 3 in every ESM-2 vocab.
        self.vocab.get(name).copied().unwrap_or(3)
    }

    /// `<cls> residues… <eos>`, truncated to the model's maximum length.
    pub fn tokenize(&self, seq: &str) -> Vec<u32> {
        let mut ids = vec![self.token("<cls>")];
        ids.extend(
            seq.chars()
                .take(MAX_RESIDUES)
                .map(|c| self.token(&c.to_ascii_uppercase().to_string())),
        );
        ids.push(self.token("<eos>"));
        ids
    }

    pub fn mask_id(&self) -> u32 {
        self.cfg.mask_token_id
    }

    /// Final hidden states, shape [T, H].
    fn hidden(&self, ids: &[u32]) -> candle_core::Result<Tensor> {
        let t = ids.len();
        let input = Tensor::new(ids, &self.device)?;
        let mut x = self.embed.forward(&input)?;

        if self.cfg.token_dropout {
            let is_mask: Vec<f32> = ids
                .iter()
                .map(|&i| (i == self.cfg.mask_token_id) as u8 as f32)
                .collect();
            let observed = is_mask.iter().sum::<f32>() as f64 / t as f64;
            let keep: Vec<f32> = is_mask.iter().map(|m| 1.0 - m).collect();
            let keep = Tensor::from_vec(keep, (t, 1), &self.device)?;
            x = (x.broadcast_mul(&keep)? * ((1.0 - MASK_RATIO_TRAIN) / (1.0 - observed)))?;
        }

        let hd = self.cfg.hidden_size / self.cfg.num_attention_heads;
        let half = hd / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| 1.0 / 10000f32.powf(2.0 * i as f32 / hd as f32))
            .collect();
        let mut cos = Vec::with_capacity(t * hd);
        let mut sin = Vec::with_capacity(t * hd);
        for pos in 0..t {
            for j in 0..hd {
                let angle = pos as f32 * inv_freq[j % half];
                cos.push(angle.cos());
                sin.push(angle.sin());
            }
        }
        let cos = Tensor::from_vec(cos, (t, hd), &self.device)?;
        let sin = Tensor::from_vec(sin, (t, hd), &self.device)?;

        for layer in &self.layers {
            x = layer.forward(&x, &cos, &sin)?;
        }
        self.final_ln.forward(&x)
    }

    /// Per-protein embedding: mean of residue states (excluding <cls>/<eos>).
    pub fn embed(&self, seq: &str) -> Res<Vec<f32>> {
        let ids = self.tokenize(seq);
        let n = ids.len() - 2;
        if n == 0 {
            return Err("empty sequence".into());
        }
        let h = self.hidden(&ids)?;
        Ok(h.narrow(0, 1, n)?.mean(0)?.to_vec1::<f32>()?)
    }

    /// Masked-LM logits at `pos` (used to sanity-check the implementation).
    pub fn logits_at(&self, ids: &[u32], pos: usize) -> Res<Vec<f32>> {
        let h = self.hidden(ids)?.narrow(0, pos, 1)?;
        let y = self
            .lm_ln
            .forward(&self.lm_dense.forward(&h)?.gelu_erf()?)?;
        // Decoder weights are tied to the input embeddings.
        let logits = y
            .matmul(&self.embed.embeddings().t()?)?
            .broadcast_add(&self.lm_bias)?;
        Ok(logits.squeeze(0)?.to_vec1::<f32>()?)
    }
}

impl Embedder for Esm {
    fn embed(&mut self, seq: &str) -> Res<Vec<f32>> {
        Esm::embed(self, seq)
    }

    fn dim(&self) -> usize {
        self.hidden_size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Downloads ESM-2 8M (31 MB) on first run. If the forward pass were wrong,
    /// masked-residue recovery would be near chance (~1/20).
    #[test]
    fn recovers_masked_residues() -> Res<()> {
        let repo = hf_hub::api::sync::Api::new()?.model("facebook/esm2_t6_8M_UR50D".to_string());
        repo.get("config.json")?;
        repo.get("vocab.txt")?;
        let weights = repo.get("model.safetensors")?;
        let dir = weights.parent().ok_or("no cache dir")?;
        let model = Esm::load(dir)?;

        // Human ubiquitin.
        let seq = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG";
        let ids = model.tokenize(seq);
        let mut correct = 0;
        for pos in 1..ids.len() - 1 {
            let mut masked = ids.clone();
            masked[pos] = model.mask_id();
            let logits = model.logits_at(&masked, pos)?;
            let best = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i as u32);
            if best == Some(ids[pos]) {
                correct += 1;
            }
        }
        // Reference: HF transformers EsmForMaskedLM gets 22/76 (0.289) on this test.
        assert_eq!(
            correct, 22,
            "masked recovery differs from the PyTorch reference"
        );

        // First values of the mean-pooled embedding from HF transformers.
        let reference = [-0.18984f32, 0.05686, -0.01698, 0.02148, 0.30577, -0.10136];
        let e = model.embed(seq)?;
        assert_eq!(e.len(), model.hidden_size());
        for (got, want) in e.iter().zip(reference) {
            assert!(
                (got - want).abs() < 1e-3,
                "embedding {got} != reference {want}"
            );
        }
        Ok(())
    }
}
