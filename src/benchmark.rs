//! Exploratory leave-one-out label retrieval, independent of projections and inference.
use crate::{fasta::Record, imports::Annotations, workspace};
use serde::Serialize;
use std::collections::BTreeMap;

#[derive(Debug, Serialize)]
pub struct LabelResult {
    pub queries: usize,
    pub top1_accuracy: f64,
    pub mean_precision_at_k: f64,
}
#[derive(Debug, Serialize)]
pub struct Report {
    pub software: String,
    pub method: &'static str,
    pub label_column: String,
    pub total_proteins: usize,
    pub labeled_proteins: usize,
    pub excluded_unlabeled: usize,
    pub requested_k: usize,
    pub effective_k: usize,
    pub top1_accuracy: f64,
    pub mean_precision_at_k: f64,
    pub macro_precision_at_k: f64,
    pub random_neighbor_expected_precision: f64,
    pub per_label: BTreeMap<String, LabelResult>,
    pub repeated_sequence_rows: usize,
    pub limitations: &'static str,
}
pub fn evaluate(
    records: &[Record],
    vectors: &[Vec<f32>],
    annotations: &Annotations,
    column: &str,
    k: usize,
) -> Result<Report, String> {
    workspace::validate_vectors(records, vectors)?;
    if k == 0 {
        return Err("k must be greater than zero".into());
    }
    if !annotations.columns.iter().any(|c| c == column) {
        return Err(format!("Unknown annotation column {column:?}"));
    }
    let labeled: Vec<_> = records
        .iter()
        .enumerate()
        .filter_map(|(i, r)| {
            annotations
                .label(&r.id, column)
                .filter(|v| !v.is_empty())
                .map(|v| (i, v.to_string()))
        })
        .collect();
    if labeled.len() < 2 {
        return Err("At least two labeled proteins are required".into());
    }
    let candidates: Vec<_> = labeled.iter().map(|(i, _)| vectors[*i].clone()).collect();
    let effective_k = k.min(labeled.len() - 1);
    let mut counts = BTreeMap::<String, usize>::new();
    for (_, label) in &labeled {
        *counts.entry(label.clone()).or_default() += 1;
    }
    let mut per_label = BTreeMap::<String, LabelResult>::new();
    let mut hits = 0.;
    let mut precision = 0.;
    let mut chance = 0.;
    for (i, (_, label)) in labeled.iter().enumerate() {
        let neighbors = workspace::neighbors(&candidates, i, effective_k);
        let correct = neighbors
            .iter()
            .filter(|(j, _)| &labeled[*j].1 == label)
            .count() as f64
            / effective_k as f64;
        let first = if labeled[neighbors[0].0].1 == *label {
            1.
        } else {
            0.
        };
        hits += first;
        precision += correct;
        chance += (counts[label] - 1) as f64 / (labeled.len() - 1) as f64;
        let result = per_label.entry(label.clone()).or_insert(LabelResult {
            queries: 0,
            top1_accuracy: 0.,
            mean_precision_at_k: 0.,
        });
        result.queries += 1;
        result.top1_accuracy += first;
        result.mean_precision_at_k += correct;
    }
    for result in per_label.values_mut() {
        result.top1_accuracy /= result.queries as f64;
        result.mean_precision_at_k /= result.queries as f64;
    }
    let macro_precision = per_label
        .values()
        .map(|r| r.mean_precision_at_k)
        .sum::<f64>()
        / per_label.len() as f64;
    let mut sequences = std::collections::BTreeSet::new();
    let repeated_sequence_rows = records
        .iter()
        .filter(|r| !r.seq.is_empty() && !sequences.insert(&r.seq))
        .count();
    Ok(Report {
        software: format!("ProtRust {}", env!("CARGO_PKG_VERSION")),
        method: "Leave-one-out cosine retrieval among labeled proteins; self excluded; distance ties resolved by CSV row order",
        label_column: column.into(),
        total_proteins: records.len(),
        labeled_proteins: labeled.len(),
        excluded_unlabeled: records.len() - labeled.len(),
        requested_k: k,
        effective_k,
        top1_accuracy: hits / labeled.len() as f64,
        mean_precision_at_k: precision / labeled.len() as f64,
        macro_precision_at_k: macro_precision,
        random_neighbor_expected_precision: chance / labeled.len() as f64,
        per_label,
        repeated_sequence_rows,
        limitations: "Exploratory retrieval, not independent prediction validation. No homology/identity split, uncertainty estimates, or training-overlap controls. Missing labels excluded from queries and candidates. Duplicate counts require supplied sequences; nonidentical homologs are not detected.",
    })
}
pub fn cli(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    if args.len() != 6 {
        return Err(
            "Usage: prot-rust --evaluate EMBEDDINGS.csv ANNOTATIONS.csv LABEL_COLUMN K REPORT.json"
                .into(),
        );
    }
    let embeddings_text = std::fs::read_to_string(&args[1])?;
    let annotation_text = std::fs::read_to_string(&args[2])?;
    let data = crate::imports::embeddings(&embeddings_text).map_err(std::io::Error::other)?;
    let known = data.records.iter().map(|r| r.id.as_str()).collect();
    let annotations =
        crate::imports::annotations(&annotation_text, &known).map_err(std::io::Error::other)?;
    let k = args[4].parse()?;
    let report = evaluate(&data.records, &data.vectors, &annotations, &args[3], k)
        .map_err(std::io::Error::other)?;
    // Include exact input text so reports can be reproduced without the original paths.
    let output = serde_json::json!({"inputs":{"embeddings_path":args[1],"annotations_path":args[2],"embeddings_csv":embeddings_text,"annotations_csv":annotation_text},"evaluation":report});
    std::fs::write(&args[5], serde_json::to_vec_pretty(&output)?)?;
    println!(
        "Report saved to {} ({} labeled proteins; effective k={})",
        args[5], report.labeled_proteins, report.effective_k
    );
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn separates_labels_and_handles_missing_labels() {
        let data = crate::imports::embeddings(
            "id,embedding_0,embedding_1\na,1,0\nb,1,0.1\nc,-1,0\nd,-1,-0.1\ne,0,1",
        )
        .expect("vectors");
        let ids = data.records.iter().map(|r| r.id.as_str()).collect();
        let a = crate::imports::annotations("id,family\na,A\nb,A\nc,B\nd,B\ne,", &ids)
            .expect("annotations");
        let r = evaluate(&data.records, &data.vectors, &a, "family", 1).expect("evaluation");
        assert_eq!(r.top1_accuracy, 1.);
        assert_eq!(r.mean_precision_at_k, 1.);
        assert_eq!(r.excluded_unlabeled, 1);
        assert!((r.random_neighbor_expected_precision - 1. / 3.).abs() < 1e-10);
        let r = evaluate(&data.records, &data.vectors, &a, "family", 100).expect("large k");
        assert_eq!(r.effective_k, 3);
        assert!((r.mean_precision_at_k - 1. / 3.).abs() < 1e-10);
        assert!(evaluate(&data.records, &data.vectors, &a, "family", 0).is_err());
        assert!(evaluate(&data.records, &data.vectors, &a, "unknown", 1).is_err());
    }
}
