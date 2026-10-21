//! Placeholder data so the UI can be built before real embeddings exist.
//! Replaced by ESM-2 + UMAP output in steps 6-7.

pub const CLUSTER_NAMES: [&str; 8] = [
    "Kinases",
    "Proteases",
    "Transporters",
    "Ribosomal",
    "Transcription factors",
    "Chaperones",
    "Hydrolases",
    "Receptors",
];

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Protein {
    pub id: String,
    pub seq_len: usize,
    pub cluster: usize,
    pub pos: [f64; 2],
}

/// Tiny deterministic PRNG (xorshift) so we need no extra crate.
struct Rng(u64);

impl Rng {
    fn next_f64(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Standard normal via Box-Muller.
    fn gauss(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-12);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

pub fn generate(n: usize) -> Vec<Protein> {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let k = CLUSTER_NAMES.len();
    let centers: Vec<[f64; 2]> = (0..k)
        .map(|i| {
            let a = std::f64::consts::TAU * i as f64 / k as f64;
            let r = 9.0 + 3.0 * rng.next_f64();
            [r * a.cos(), r * a.sin()]
        })
        .collect();
    let spreads: Vec<f64> = (0..k).map(|_| 0.8 + 1.4 * rng.next_f64()).collect();

    (0..n)
        .map(|i| {
            let cluster = (rng.next_f64() * k as f64) as usize % k;
            let [cx, cy] = centers[cluster];
            let s = spreads[cluster];
            Protein {
                id: format!("P{:05}", i + 1),
                seq_len: 80 + (rng.next_f64() * 700.0) as usize,
                cluster,
                pos: [cx + s * rng.gauss(), cy + s * rng.gauss()],
            }
        })
        .collect()
}
