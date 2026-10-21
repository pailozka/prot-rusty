//! Random Neighbor Score (RNS) — Prabakaran & Bromberg, Nature Methods 2026,
//! doi:10.1038/s41592-026-03028-7.
//!
//! For each real protein: the fraction of its k nearest neighbors (cosine
//! distance) that are residue-shuffled, non-biological sequences.
//! 0 = embedding sits among real biology (reliable); 1 = sits in the
//! "junkyard" of random sequences (the model did not learn this protein).
//!
//! Like the paper, each repeat undersamples the random set to the size of the
//! real set, and scores are averaged over repeats.

use crate::fasta::Record;
use rayon::prelude::*;

/// Paper default for the headline results (RNS_k=1000).
pub const PAPER_K: usize = 1000;
pub const DEFAULT_REPEATS: usize = 100;

/// Small deterministic PRNG so results are reproducible across runs.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        // Avoid the all-zero state, which xorshift can never leave.
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15 | 1)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Uniform in 0..n (n > 0).
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// Shuffle residues (Fisher–Yates): same composition, no biological order.
pub fn shuffle(seq: &str, seed: u64) -> String {
    let mut rng = Rng::new(seed);
    let mut chars: Vec<char> = seq.chars().collect();
    for i in (1..chars.len()).rev() {
        chars.swap(i, rng.below(i + 1));
    }
    chars.into_iter().collect()
}

/// One shuffled copy per real protein. The paper's Astral40R used five, but
/// its Proteome4R control (one per protein) gave the same conclusions, and
/// one copy halves the extra embedding time.
pub fn random_set(records: &[Record], seed: u64) -> Vec<Record> {
    records
        .iter()
        .enumerate()
        .map(|(i, r)| Record {
            id: format!("shuffled_{}", r.id),
            description: "residue-shuffled control".into(),
            seq: shuffle(&r.seq, seed.wrapping_add(i as u64)),
        })
        .collect()
}

/// k must be well below the pool size or every protein trivially scores high.
/// The paper used k > 100 on sets of 930+; for small inputs scale it down.
pub fn default_k(n_real: usize) -> usize {
    (n_real / 5)
        .clamp(1, PAPER_K)
        .min(n_real.saturating_sub(1).max(1))
}

fn normalize(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    v.iter().map(|x| x / norm).collect()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// RNS_k for each real embedding, averaged over `repeats` undersampled draws.
/// Runs in parallel over proteins; memory is O(n) per thread.
pub fn scores(
    real: &[Vec<f32>],
    random: &[Vec<f32>],
    k: usize,
    repeats: usize,
    seed: u64,
) -> Vec<f32> {
    let n = real.len();
    if n < 2 || random.is_empty() || k == 0 || repeats == 0 {
        return vec![0.0; n];
    }
    let real: Vec<Vec<f32>> = real.iter().map(|v| normalize(v)).collect();
    let random: Vec<Vec<f32>> = random.iter().map(|v| normalize(v)).collect();
    let m = random.len().min(n); // random proteins drawn per repeat

    // The same draws for every protein, so all are scored against the same pools.
    let draws: Vec<Vec<usize>> = (0..repeats)
        .map(|r| {
            let mut rng = Rng::new(seed.wrapping_add(r as u64));
            let mut idx: Vec<usize> = (0..random.len()).collect();
            for i in 0..m {
                let j = i + rng.below(idx.len() - i);
                idx.swap(i, j);
            }
            idx.truncate(m);
            idx
        })
        .collect();

    real.par_iter()
        .enumerate()
        .map(|(i, q)| {
            // Similarities to every real (except self) and every random protein, once.
            let sim_real: Vec<f32> = real
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, v)| dot(q, v))
                .collect();
            let sim_rand: Vec<f32> = random.iter().map(|v| dot(q, v)).collect();
            let k = k.min(sim_real.len() + m);

            let mut total = 0.0f32;
            let mut pool: Vec<(f32, bool)> = Vec::with_capacity(sim_real.len() + m);
            for draw in &draws {
                pool.clear();
                pool.extend(sim_real.iter().map(|&s| (s, false)));
                pool.extend(draw.iter().map(|&j| (sim_rand[j], true)));
                // Partial sort: only the top k by similarity matter.
                pool.select_nth_unstable_by(k - 1, |a, b| b.0.total_cmp(&a.0));
                let hits = pool[..k].iter().filter(|(_, is_random)| *is_random).count();
                total += hits as f32 / k as f32;
            }
            total / repeats as f32
        })
        .collect()
}

/// Dataset-level summary, as reported per model in the paper.
#[derive(Clone, Debug)]
pub struct Summary {
    pub k: usize,
    pub mean: f32,
    pub median: f32,
    /// Fraction of proteins with RNS == 0 ("completely reliable").
    pub frac_zero: f32,
}

pub fn summarize(scores: &[f32], k: usize) -> Summary {
    if scores.is_empty() {
        return Summary {
            k,
            mean: 0.0,
            median: 0.0,
            frac_zero: 0.0,
        };
    }
    let mut sorted = scores.to_vec();
    sorted.sort_by(f32::total_cmp);
    Summary {
        k,
        mean: scores.iter().sum::<f32>() / scores.len() as f32,
        median: sorted[sorted.len() / 2],
        frac_zero: scores.iter().filter(|&&s| s == 0.0).count() as f32 / scores.len() as f32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shuffle_keeps_composition() {
        let s = "MKVLLAAGGGW";
        let t = shuffle(s, 7);
        assert_ne!(s, t);
        let mut a: Vec<char> = s.chars().collect();
        let mut b: Vec<char> = t.chars().collect();
        a.sort();
        b.sort();
        assert_eq!(a, b);
        assert_eq!(shuffle(s, 7), t, "same seed must give same shuffle");
    }

    /// Real proteins in one tight cluster, random ones elsewhere -> RNS 0.
    /// A real protein placed inside the random cluster -> RNS near 1.
    #[test]
    fn separates_clusters() {
        let mut real: Vec<Vec<f32>> = (0..20).map(|i| vec![1.0, 0.01 * i as f32, 0.0]).collect();
        let random: Vec<Vec<f32>> = (0..20).map(|i| vec![0.0, 0.01 * i as f32, 1.0]).collect();
        real.push(vec![0.0, 0.05, 1.0]); // an "unlearned" protein
        let s = scores(&real, &random, 5, 10, 1);
        assert!(s[..20].iter().all(|&x| x == 0.0), "{s:?}");
        assert!(s[20] > 0.9, "{}", s[20]);
        let sum = summarize(&s, 5);
        assert!((sum.frac_zero - 20.0 / 21.0).abs() < 1e-6);
    }

    #[test]
    fn default_k_stays_below_pool() {
        assert_eq!(default_k(2), 1);
        assert_eq!(default_k(50), 10);
        assert_eq!(default_k(100_000), PAPER_K);
    }
}
