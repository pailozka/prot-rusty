//! ProtT5-XL (Rostlab/prot_t5_xl_half_uniref50-enc): the encoder half of a
//! T5 model trained on UniRef50. Uses candle's built-in T5 encoder; only the
//! tokenizer and pooling are ProtT5-specific.

use crate::models::{Embedder, Res};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::t5::{Config, T5EncoderModel};
use std::collections::HashMap;
use std::path::Path;

const EOS: u32 = 1;
const UNK: u32 = 2;
/// ProtT5 was trained on sequences up to ~1000 residues; longer ones work but
/// cost O(n²) memory, so cap them like ESM-2.
pub const MAX_RESIDUES: usize = 1022;

pub struct ProtT5 {
    model: T5EncoderModel,
    vocab: HashMap<String, u32>,
    dim: usize,
    device: Device,
}

/// Read the piece strings out of a SentencePiece `spiece.model` file.
/// The file is a protobuf `ModelProto` whose field 1 repeats `SentencePiece`
/// messages, each with the piece text in field 1. Hand-parsed to avoid a
/// protobuf dependency for ~30 lines of work.
fn read_spiece_vocab(bytes: &[u8]) -> Res<HashMap<String, u32>> {
    fn varint(b: &[u8], i: &mut usize) -> Res<u64> {
        let mut v = 0u64;
        for shift in (0..64).step_by(7) {
            let byte = *b.get(*i).ok_or("truncated spiece.model")?;
            *i += 1;
            v |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Ok(v);
            }
        }
        Err("bad varint in spiece.model".into())
    }
    /// Skip a field of the given wire type.
    fn skip(b: &[u8], i: &mut usize, wire: u64) -> Res<()> {
        match wire {
            0 => {
                varint(b, i)?;
            }
            1 => *i += 8,
            2 => {
                let n = varint(b, i)? as usize;
                *i += n;
            }
            5 => *i += 4,
            _ => return Err("unsupported wire type in spiece.model".into()),
        }
        Ok(())
    }

    let mut vocab = HashMap::new();
    let mut i = 0;
    while i < bytes.len() {
        let key = varint(bytes, &mut i)?;
        let (field, wire) = (key >> 3, key & 7);
        if field == 1 && wire == 2 {
            let len = varint(bytes, &mut i)? as usize;
            let msg = bytes.get(i..i + len).ok_or("truncated piece")?;
            i += len;
            // Inside SentencePiece: field 1 = piece string.
            let mut j = 0;
            while j < msg.len() {
                let k = varint(msg, &mut j)?;
                if k >> 3 == 1 && k & 7 == 2 {
                    let n = varint(msg, &mut j)? as usize;
                    let s = msg.get(j..j + n).ok_or("truncated piece string")?;
                    let id = vocab.len() as u32;
                    vocab.insert(String::from_utf8_lossy(s).into_owned(), id);
                    break;
                }
                skip(msg, &mut j, k & 7)?;
            }
        } else {
            skip(bytes, &mut i, wire)?;
        }
    }
    Ok(vocab)
}

impl ProtT5 {
    pub fn load(dir: &Path) -> Res<Self> {
        let cfg: Config = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
        let vocab = read_spiece_vocab(&std::fs::read(dir.join("spiece.model"))?)?;
        let device = Device::Cpu;
        // f16 halves RAM vs f32 (~2.4 GB instead of ~4.8 GB). This checkpoint is
        // Rostlab's official half-precision release, so f16 is how it is meant
        // to run. (candle's CPU matmul has no bf16 kernel.) Tensors are read from
        // the .bin one at a time, so the file is never fully in memory twice.
        let vb = VarBuilder::from_pth(dir.join("pytorch_model.bin"), DType::F16, &device)?;
        let model = T5EncoderModel::load(vb, &cfg)?;
        Ok(Self {
            model,
            vocab,
            dim: cfg.d_model,
            device,
        })
    }

    /// One token per residue (rare U/Z/O/B mapped to X, as in ProtTrans), then </s>.
    pub fn tokenize(&self, seq: &str) -> Vec<u32> {
        let mut ids: Vec<u32> = seq
            .chars()
            .take(MAX_RESIDUES)
            .map(|c| {
                let c = match c.to_ascii_uppercase() {
                    'U' | 'Z' | 'O' | 'B' => 'X',
                    c => c,
                };
                self.vocab
                    .get(&format!("\u{2581}{c}"))
                    .or_else(|| self.vocab.get(&c.to_string()))
                    .copied()
                    .unwrap_or(UNK)
            })
            .collect();
        ids.push(EOS);
        ids
    }
}

impl Embedder for ProtT5 {
    fn embed(&mut self, seq: &str) -> Res<Vec<f32>> {
        let ids = self.tokenize(seq);
        let n = ids.len() - 1;
        if n == 0 {
            return Err("empty sequence".into());
        }
        let input = Tensor::new(ids.as_slice(), &self.device)?.unsqueeze(0)?;
        let h = self.model.forward(&input)?.squeeze(0)?; // [T, d_model]
        // Average in f32: summing hundreds of f16 values loses precision.
        Ok(h.narrow(0, 0, n)?
            .to_dtype(DType::F32)?
            .mean(0)?
            .to_vec1::<f32>()?)
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Downloads ProtT5-XL (2.4 GB) and needs ~5 GB RAM, so it is opt-in:
    /// `cargo test --release -- --ignored prott5`
    #[test]
    #[ignore]
    fn prott5_matches_pytorch() -> Res<()> {
        let repo = hf_hub::api::sync::Api::new()?
            .model("Rostlab/prot_t5_xl_half_uniref50-enc".to_string());
        repo.get("config.json")?;
        repo.get("spiece.model")?;
        let weights = repo.get("pytorch_model.bin")?;
        let dir = weights.parent().ok_or("no cache dir")?;
        let mut model = ProtT5::load(dir)?;

        let seq = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG";
        // Reference: HF transformers T5EncoderModel, fp16, same sequence.
        let ids = model.tokenize(seq);
        assert_eq!(&ids[..12], &[19, 16, 12, 15, 6, 14, 11, 4, 11, 5, 14, 11]);
        assert_eq!(ids.len(), seq.len() + 1);
        assert_eq!(ids.last(), Some(&EOS));

        let reference = [0.20361f32, 0.14986, -0.04692, -0.05018, 0.1299, -0.01528];
        let e = model.embed(seq)?;
        assert_eq!(e.len(), 1024);
        for (got, want) in e.iter().zip(reference) {
            // f16 compute: allow a little more slack than the f32 ESM test.
            assert!(
                (got - want).abs() < 2e-3,
                "embedding {got} != reference {want}"
            );
        }
        Ok(())
    }
}
