//! Portable project files, similarity search, and interoperable exports.
use crate::{demo::Protein, fasta::Record, models::Model};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct Project {
    pub version: u32,
    pub model: Model,
    #[serde(default)]
    pub configured_model: Option<Model>,
    #[serde(default)]
    pub input_records: Vec<Record>,
    pub records: Vec<Record>,
    pub embeddings: Vec<Vec<f32>>,
    pub proteins: Vec<Protein>,
    pub selected: Option<usize>,
    pub search: String,
    pub cluster_on: [bool; 8],
    #[serde(default)]
    pub projection: Option<crate::projection::Projection>,
    #[serde(default)]
    pub view: crate::projection::View,
    #[serde(default)]
    pub external_source: Option<String>,
    #[serde(default)]
    pub annotations: crate::imports::Annotations,
    #[serde(default)]
    pub active_annotation: Option<String>,
    #[serde(default)]
    pub hidden_labels: std::collections::BTreeSet<String>,
}
impl Project {
    pub fn validate(&self) -> Result<(), String> {
        for (records, pending) in [(&self.records, false), (&self.input_records, true)] {
            let mut ids = std::collections::HashSet::new();
            for r in records {
                if r.id.is_empty()
                    || r.id.chars().any(char::is_whitespace)
                    || !ids.insert(&r.id)
                    || (r.seq.is_empty() && (pending || self.external_source.is_none()))
                    || !r
                        .seq
                        .chars()
                        .all(|c| "ACDEFGHIKLMNPQRSTVWYBXZJUO".contains(c))
                {
                    return Err("Invalid or duplicate FASTA records in project".into());
                }
            }
        }
        let ids = self.proteins.iter().map(|p| p.id.as_str()).collect();
        if !self.annotations.valid(&ids)
            || self
                .active_annotation
                .as_ref()
                .is_some_and(|c| !self.annotations.columns.contains(c))
        {
            return Err("Invalid annotation table or selected column".into());
        }
        if !self.view.valid()
            || self
                .projection
                .as_ref()
                .is_some_and(|p| self.embeddings.is_empty() || !p.valid(self.proteins.len()))
        {
            return Err("Invalid projection or camera state".into());
        }
        if self.version != 1 {
            return Err("Unsupported project version".into());
        }
        if self
            .proteins
            .iter()
            .any(|p| p.cluster >= 8 || p.pos.iter().any(|x| !x.is_finite()))
        {
            return Err("Invalid map data".into());
        }
        if self.selected.is_some_and(|i| i >= self.proteins.len()) {
            return Err("Invalid selection".into());
        }
        if !self.records.is_empty() || !self.embeddings.is_empty() {
            validate_vectors(&self.records, &self.embeddings)?;
            if self.records.len() != self.proteins.len()
                || self
                    .records
                    .iter()
                    .zip(&self.proteins)
                    .any(|(r, p)| r.id != p.id || r.seq.len() != p.seq_len)
            {
                return Err("Protein metadata does not match sequences".into());
            }
        }
        Ok(())
    }
}
pub fn validate_vectors(records: &[Record], vectors: &[Vec<f32>]) -> Result<(), String> {
    let dim = vectors.first().map_or(0, Vec::len);
    if records.is_empty()
        || records.len() != vectors.len()
        || dim == 0
        || vectors.iter().any(|v| {
            v.len() != dim || v.iter().any(|x| !x.is_finite()) || v.iter().all(|x| *x == 0.0)
        })
    {
        return Err(
            "Embeddings must have matching counts, equal dimensions, and finite nonzero vectors"
                .into(),
        );
    }
    Ok(())
}
pub fn neighbors(vectors: &[Vec<f32>], selected: usize, k: usize) -> Vec<(usize, f64)> {
    let Some(query) = vectors.get(selected) else {
        return Vec::new();
    };
    let norm = |v: &[f32]| v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let qn = norm(query);
    if qn == 0.0 {
        return Vec::new();
    }
    let mut result: Vec<_> = vectors
        .iter()
        .enumerate()
        .filter_map(|(i, v)| {
            let vn = norm(v);
            if i == selected || v.len() != query.len() || vn == 0.0 {
                return None;
            }
            let dot: f64 = query
                .iter()
                .zip(v)
                .map(|(a, b)| *a as f64 * *b as f64)
                .sum();
            let distance = 1.0 - (dot / qn / vn).clamp(-1.0, 1.0);
            distance.is_finite().then_some((i, distance))
        })
        .collect();
    result.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    result.truncate(k);
    result
}
fn csv(s: &str) -> String {
    format!("\"{}\"", s.replace('"', "\"\""))
}
pub fn embeddings_csv(records: &[Record], vectors: &[Vec<f32>]) -> String {
    let mut out = String::from("id,description,sequence");
    for i in 0..vectors.first().map_or(0, Vec::len) {
        out.push_str(&format!(",embedding_{i}"));
    }
    out.push('\n');
    for (r, v) in records.iter().zip(vectors) {
        out.push_str(&format!(
            "{},{},{}",
            csv(&r.id),
            csv(&r.description),
            csv(&r.seq)
        ));
        for x in v {
            out.push_str(&format!(",{x}"));
        }
        out.push('\n');
    }
    out
}
pub fn neighbors_csv(
    proteins: &[Protein],
    selected: usize,
    neighbors: &[(usize, f64)],
    real: bool,
) -> String {
    let mut out = format!(
        "query_id,neighbor_id,rank,{}\n",
        if real {
            "cosine_distance"
        } else {
            "demo_map_distance"
        }
    );
    for (rank, (i, d)) in neighbors.iter().enumerate() {
        out.push_str(&format!(
            "{},{},{},{}\n",
            csv(&proteins[selected].id),
            csv(&proteins[*i].id),
            rank + 1,
            d
        ));
    }
    out
}
/// SVG is a standalone vector image and preserves all visible points.
pub fn map_svg(
    proteins: &[Protein],
    visible: &[bool],
    selected: Option<usize>,
    colors: &[crate::egui::Color32],
    title: &str,
) -> String {
    let points: Vec<_> = proteins
        .iter()
        .zip(visible)
        .filter(|(_, v)| **v)
        .map(|(p, _)| p)
        .collect();
    let bounds = |axis: usize| {
        let lo = points
            .iter()
            .map(|p| p.pos[axis])
            .reduce(f64::min)
            .unwrap_or(0.0);
        let hi = points
            .iter()
            .map(|p| p.pos[axis])
            .reduce(f64::max)
            .unwrap_or(1.0);
        (lo, (hi - lo).max(1e-12))
    };
    let (x, w) = bounds(0);
    let (y, h) = bounds(1);
    let scale = (1120.0 / w).min(760.0 / h);
    let left = 600.0 - w * scale / 2.0;
    let bottom = 470.0 + h * scale / 2.0;
    let mut out = format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="900" viewBox="0 0 1200 900"><rect width="1200" height="900" fill="#f6f8f7"/><text x="40" y="40" font-family="sans-serif" font-size="22">ProtRust · {}</text>"##,
        title
    );
    for (i, p) in proteins.iter().enumerate().filter(|(i, _)| visible[*i]) {
        let c = colors.get(i).copied().unwrap_or(crate::PALETTE[p.cluster]);
        out.push_str(&format!(
            r#"<circle cx="{}" cy="{}" r="{}" fill="rgb({},{},{})"/>"#,
            left + (p.pos[0] - x) * scale,
            bottom - (p.pos[1] - y) * scale,
            if selected == Some(i) { 6 } else { 3 },
            c.r(),
            c.g(),
            c.b()
        ));
    }
    out.push_str("</svg>");
    out
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cosine_ranking_is_scale_invariant() {
        let v = vec![
            vec![1., 0.],
            vec![20., 0.],
            vec![1., 1.],
            vec![-1., 0.],
            vec![0., 0.],
        ];
        let n = neighbors(&v, 0, 10);
        assert_eq!(n.iter().map(|p| p.0).collect::<Vec<_>>(), vec![1, 2, 3]);
        assert!(n[0].1.abs() < 1e-10);
        assert!((n[2].1 - 2.).abs() < 1e-10);
    }
    #[test]
    fn rejects_misaligned_and_invalid_vectors() {
        let r = vec![Record {
            id: "a".into(),
            description: String::new(),
            seq: "ACD".into(),
        }];
        assert!(validate_vectors(&r, &[vec![f32::NAN]]).is_err());
        assert!(validate_vectors(&r, &[vec![0.]]).is_err());
        assert!(validate_vectors(&r, &[]).is_err());
        assert!(validate_vectors(&r, &[vec![1.]]).is_ok());
    }
    #[test]
    fn project_round_trip_and_validation() {
        let mut p = Project {
            version: 1,
            model: Model::Esm2_8M,
            configured_model: Some(Model::Esm2_35M),
            input_records: Vec::new(),
            records: vec![Record {
                id: "a".into(),
                description: "Some protein".into(),
                seq: "ACD".into(),
            }],
            embeddings: vec![vec![0.25, 1.5]],
            proteins: vec![Protein {
                id: "a".into(),
                seq_len: 3,
                cluster: 0,
                pos: [1., 2.],
            }],
            selected: Some(0),
            search: "Some".into(),
            cluster_on: [true; 8],
            projection: None,
            view: crate::projection::View::default(),
            external_source: None,
            annotations: crate::imports::Annotations::default(),
            active_annotation: None,
            hidden_labels: Default::default(),
        };
        p.projection = Some(crate::projection::pca(&p.embeddings).expect("fixture PCA"));
        p.view.three_d = true;
        p.view.yaw = 1.2;
        p.view.zoom = 2.0;
        let bytes = serde_json::to_vec(&p).expect("serialize fixture");
        let reopened: Project = serde_json::from_slice(&bytes).expect("deserialize fixture");
        assert!(reopened.validate().is_ok());
        assert_eq!(reopened.embeddings, p.embeddings);
        assert!(reopened.view.three_d);
        assert_eq!(reopened.view.yaw, 1.2);
        assert_eq!(reopened.view.zoom, 2.0);
        assert_eq!(
            reopened.projection.as_ref().expect("saved PCA").points,
            p.projection.as_ref().expect("PCA").points
        );
        let mut legacy = serde_json::to_value(&p).expect("legacy fixture");
        legacy.as_object_mut().expect("object").remove("projection");
        legacy.as_object_mut().expect("object").remove("view");
        let legacy: Project = serde_json::from_value(legacy).expect("old project");
        assert!(legacy.validate().is_ok());
        assert!(legacy.projection.is_none());
        assert!(!legacy.view.three_d);
        let mut invalid = serde_json::from_slice::<Project>(&bytes).expect("invalid fixture");
        invalid
            .projection
            .as_mut()
            .expect("projection")
            .points
            .clear();
        assert!(invalid.validate().is_err());
        invalid.projection = None;
        invalid.view.zoom = 0.;
        assert!(invalid.validate().is_err());
        assert_eq!(reopened.records[0].description, "Some protein");
        assert_eq!(reopened.selected, Some(0));
        assert_eq!(reopened.configured_model, Some(Model::Esm2_35M));
        assert_eq!(reopened.search, "Some");
        p.selected = Some(3);
        assert!(p.validate().is_err());
        p.selected = None;
        p.proteins[0].id = "wrong".into();
        assert!(p.validate().is_err());
        p.proteins[0].id = "a".into();
        p.proteins[0].cluster = 8;
        assert!(p.validate().is_err());
        p.proteins[0].cluster = 0;
        p.version = 2;
        assert!(p.validate().is_err());
    }
    #[test]
    fn exports_preserve_values_and_visibility() {
        let records = vec![Record {
            id: "a,1".into(),
            description: "a quote".into(),
            seq: "ACD".into(),
        }];
        let csv = embeddings_csv(&records, &[vec![0.25, -1.5]]);
        assert!(csv.starts_with("id,description,sequence,embedding_0,embedding_1\n"));
        assert!(csv.contains(",0.25,-1.5\n"));
        let proteins = vec![
            Protein {
                id: "a,1".into(),
                seq_len: 3,
                cluster: 0,
                pos: [1., 1.],
            },
            Protein {
                id: "b".into(),
                seq_len: 3,
                cluster: 0,
                pos: [2., 2.],
            },
        ];
        let csv = neighbors_csv(&proteins, 0, &[(1, 0.2)], true);
        assert!(csv.contains("cosine_distance"));
        assert!(csv.contains("\"a,1\",\"b\",1,0.2"));
        let svg = map_svg(&proteins, &[true, false], Some(0), &[], "PCA 2D");
        assert_eq!(svg.matches("<circle ").count(), 1);
        assert!(svg.contains("PCA 2D"));
    }
    #[test]
    fn csv_quotes_metadata() {
        assert_eq!(csv("a,\"b\"\nc"), "\"a,\"\"b\"\"\nc\"");
    }
}
