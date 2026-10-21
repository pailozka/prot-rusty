//! Deterministic PCA of unit-normalized embeddings; camera state is independent of similarity.
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Projection {
    pub points: Vec<[f64; 3]>,
    /// Fraction of total centered variance captured by each component.
    pub variance: [f64; 3],
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct View {
    pub three_d: bool,
    pub yaw: f32,
    pub pitch: f32,
    pub zoom: f32,
    pub pan: [f32; 2],
}
impl Default for View {
    fn default() -> Self {
        Self {
            three_d: false,
            yaw: 0.55,
            pitch: 0.35,
            zoom: 1.0,
            pan: [0.0; 2],
        }
    }
}
impl View {
    pub fn valid(&self) -> bool {
        [self.yaw, self.pitch, self.zoom, self.pan[0], self.pan[1]]
            .iter()
            .all(|v| v.is_finite())
            && (0.1..=20.0).contains(&self.zoom)
    }
    pub fn rotate(&self, p: [f64; 3]) -> [f64; 3] {
        let (sy, cy) = (self.yaw as f64).sin_cos();
        let (sp, cp) = (self.pitch as f64).sin_cos();
        let x = cy * p[0] + sy * p[2];
        let z = -sy * p[0] + cy * p[2];
        [x, cp * p[1] - sp * z, sp * p[1] + cp * z]
    }
}
impl Projection {
    pub fn valid(&self, count: usize) -> bool {
        self.points.len() == count
            && self.points.iter().flatten().all(|x| x.is_finite())
            && self
                .variance
                .iter()
                .all(|x| x.is_finite() && (0.0..=1.000001).contains(x))
            && self.variance.iter().sum::<f64>() <= 1.000001
    }
}
/// Matrix-free power iteration: avoids allocating a dimensions² covariance matrix.
/// Unit normalization aligns the input geometry with cosine-based neighbor search.
pub fn pca(vectors: &[Vec<f32>]) -> Result<Projection, String> {
    let dim = vectors.first().map_or(0, Vec::len);
    if vectors.is_empty()
        || dim == 0
        || vectors
            .iter()
            .any(|v| v.len() != dim || v.iter().any(|x| !x.is_finite()))
    {
        return Err("PCA requires finite, equally sized embeddings".into());
    }
    let mut data = Vec::with_capacity(vectors.len());
    let mut mean = vec![0.; dim];
    for v in vectors {
        let norm = v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        if norm == 0. {
            return Err("PCA cannot normalize a zero vector".into());
        }
        let row: Vec<_> = v.iter().map(|x| *x as f64 / norm).collect();
        for (m, x) in mean.iter_mut().zip(&row) {
            *m += x / vectors.len() as f64;
        }
        data.push(row);
    }
    for row in &mut data {
        for (x, m) in row.iter_mut().zip(&mean) {
            *x -= m;
        }
    }
    let total: f64 = data.iter().flatten().map(|x| x * x).sum();
    let mut basis: Vec<Vec<f64>> = Vec::new();
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    let orthogonalize = |v: &mut Vec<f64>, basis: &[Vec<f64>]| {
        for _ in 0..2 {
            for b in basis {
                let d = dot(v, b);
                for (x, y) in v.iter_mut().zip(b) {
                    *x -= d * y;
                }
            }
        }
        let n = dot(v, v).sqrt();
        if n <= 1e-12 {
            return false;
        }
        for x in v {
            *x /= n;
        }
        true
    };
    let mut points = vec![[0.; 3]; vectors.len()];
    let mut variance = [0.; 3];
    if total <= 1e-24 {
        return Ok(Projection { points, variance });
    }
    for axis in 0..3.min(dim).min(vectors.len().saturating_sub(1)) {
        // Multiple deterministic seeds avoid an initial vector orthogonal to the data.
        let mut best = vec![0.; dim];
        let mut best_energy = 0.;
        for seed in 0..4 {
            let mut v: Vec<_> = (0..dim)
                .map(|j| (((j + 1) * (seed + 1) * 17 + axis * 31) as f64).sin())
                .collect();
            if !orthogonalize(&mut v, &basis) {
                continue;
            }
            for _ in 0..96 {
                let mut next = vec![0.; dim];
                for row in &data {
                    let weight = dot(row, &v);
                    for (x, r) in next.iter_mut().zip(row) {
                        *x += weight * r;
                    }
                }
                if !orthogonalize(&mut next, &basis) {
                    break;
                }
                let alignment = dot(&v, &next).abs();
                v = next;
                if 1. - alignment < 1e-10 {
                    break;
                }
            }
            let energy: f64 = data.iter().map(|r| dot(r, &v).powi(2)).sum();
            if energy > best_energy {
                best_energy = energy;
                best = v;
            }
        }
        if best_energy <= total * 1e-14 {
            break;
        }
        // Canonical signs make saved/recomputed maps reproducible.
        let pivot = best
            .iter()
            .max_by(|a, b| a.abs().total_cmp(&b.abs()))
            .copied()
            .unwrap_or(0.);
        if pivot < 0. {
            for x in &mut best {
                *x = -*x;
            }
        }
        for (point, row) in points.iter_mut().zip(&data) {
            point[axis] = dot(row, &best);
        }
        variance[axis] = (best_energy / total).clamp(0., 1.);
        basis.push(best);
    }
    Ok(Projection { points, variance })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pca_recovers_three_axes_and_centering() {
        let vectors = vec![
            vec![1., 0., 0.],
            vec![-1., 0., 0.],
            vec![0., 1., 0.],
            vec![0., -1., 0.],
            vec![0., 0., 1.],
            vec![0., 0., -1.],
        ];
        let p = pca(&vectors).expect("PCA");
        assert!(p.valid(6));
        for v in p.variance {
            assert!((v - 1. / 3.).abs() < 1e-6);
        }
        for axis in 0..3 {
            assert!(p.points.iter().map(|p| p[axis]).sum::<f64>().abs() < 1e-8);
        }
        for (i, a) in vectors.iter().enumerate() {
            for (j, b) in vectors.iter().enumerate() {
                let source: f64 = a
                    .iter()
                    .zip(b)
                    .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
                    .sum();
                let target: f64 = p.points[i]
                    .iter()
                    .zip(p.points[j])
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                assert!((source - target).abs() < 1e-6);
            }
        }
    }
    #[test]
    fn handles_small_degenerate_and_invalid_data() {
        for v in [
            vec![vec![1., 0.]],
            vec![vec![1., 0.]; 3],
            vec![vec![1., 0.], vec![-1., 0.]],
        ] {
            let p = pca(&v).expect("degenerate PCA");
            assert!(p.valid(v.len()));
            assert_eq!(p.points, pca(&v).expect("repeat PCA").points);
        }
        assert!(pca(&[vec![0., 0.]]).is_err());
        assert!(pca(&[vec![1.], vec![1., 2.]]).is_err());
    }
    #[test]
    fn orders_components_and_ignores_vector_magnitude() {
        let mut vectors = Vec::new();
        for (axis, count) in [(0, 4), (1, 2), (2, 1)] {
            for _ in 0..count {
                for sign in [-1., 1.] {
                    let mut v = vec![0.; 3];
                    v[axis] = sign;
                    vectors.push(v);
                }
            }
        }
        let p = pca(&vectors).expect("anisotropic PCA");
        for (got, want) in p.variance.iter().zip([4. / 7., 2. / 7., 1. / 7.]) {
            assert!((got - want).abs() < 1e-6);
        }
        for (i, v) in vectors.iter_mut().enumerate() {
            for x in v {
                *x *= (i + 1) as f32;
            }
        }
        let scaled = pca(&vectors).expect("scaled PCA");
        for (a, b) in p
            .points
            .iter()
            .flatten()
            .zip(scaled.points.iter().flatten())
        {
            assert!((a - b).abs() < 1e-6);
        }
    }
    #[test]
    fn camera_rotation_preserves_lengths() {
        let p = View::default().rotate([1., 2., 3.]);
        assert!((p.iter().map(|x| x * x).sum::<f64>() - 14.).abs() < 1e-8);
    }
}
