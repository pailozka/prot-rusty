//! The catalogue of protein language models the app can run, and the common
//! `Embedder` interface they share. Everything downstream (map, neighbors,
//! RNS, model comparison) only sees `Embedder`, never a concrete model.

use std::path::Path;

pub type Res<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// Anything that turns a protein sequence into one fixed-size vector.
pub trait Embedder: Send {
    /// Mean-pooled per-protein embedding.
    fn embed(&mut self, seq: &str) -> Res<Vec<f32>>;
    fn dim(&self) -> usize;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Model {
    Esm2_8M,
    Esm2_35M,
    Esm2_150M,
    Esm2_650M,
    ProtT5Xl,
}

impl Model {
    pub const ALL: [Model; 5] = [
        Model::Esm2_8M,
        Model::Esm2_35M,
        Model::Esm2_150M,
        Model::Esm2_650M,
        Model::ProtT5Xl,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Model::Esm2_8M => "ESM-2 8M · 31 MB",
            Model::Esm2_35M => "ESM-2 35M · 136 MB",
            Model::Esm2_150M => "ESM-2 150M · 595 MB",
            Model::Esm2_650M => "ESM-2 650M · 2.6 GB",
            Model::ProtT5Xl => "ProtT5-XL · 2.4 GB",
        }
    }

    pub fn repo(self) -> &'static str {
        match self {
            Model::Esm2_8M => "facebook/esm2_t6_8M_UR50D",
            Model::Esm2_35M => "facebook/esm2_t12_35M_UR50D",
            Model::Esm2_150M => "facebook/esm2_t30_150M_UR50D",
            Model::Esm2_650M => "facebook/esm2_t33_650M_UR50D",
            Model::ProtT5Xl => "Rostlab/prot_t5_xl_half_uniref50-enc",
        }
    }

    /// Small files fetched first, then the (large) weights file.
    pub fn small_files(self) -> &'static [&'static str] {
        match self {
            Model::ProtT5Xl => &["config.json", "spiece.model"],
            _ => &["config.json", "vocab.txt"],
        }
    }

    pub fn weights_file(self) -> &'static str {
        match self {
            Model::ProtT5Xl => "pytorch_model.bin",
            _ => "model.safetensors",
        }
    }

    /// Download size in MB, used to warn before large downloads.
    pub fn size_mb(self) -> u32 {
        match self {
            Model::Esm2_8M => 31,
            Model::Esm2_35M => 136,
            Model::Esm2_150M => 595,
            Model::Esm2_650M => 2610,
            Model::ProtT5Xl => 2416,
        }
    }

    /// Approximate peak RAM while embedding, in MB (weights + activations).
    /// ProtT5 measured at 2.4 GB in f16; ESM-2 runs in f32, so ≈ file size.
    pub fn ram_mb(self) -> u32 {
        match self {
            Model::Esm2_8M => 100,
            Model::Esm2_35M => 250,
            Model::Esm2_150M => 800,
            Model::Esm2_650M => 3000,
            Model::ProtT5Xl => 2600,
        }
    }

    /// Load from the directory holding the downloaded files.
    pub fn load(self, dir: &Path) -> Res<Box<dyn Embedder>> {
        Ok(match self {
            Model::ProtT5Xl => Box::new(crate::prott5::ProtT5::load(dir)?),
            _ => Box::new(crate::esm::Esm::load(dir)?),
        })
    }
}

/// Memory the OS says can be used without swapping, from /proc/meminfo (Linux).
/// `None` on other platforms or if the file can't be read.
pub fn available_ram_mb() -> Option<u32> {
    let info = std::fs::read_to_string("/proc/meminfo").ok()?;
    let line = info.lines().find(|l| l.starts_with("MemAvailable:"))?;
    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some((kb / 1024) as u32)
}
