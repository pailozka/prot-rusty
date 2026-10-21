//! CSV imports never invoke an embedding model. Blank sequences mean unavailable.
use crate::fasta::Record;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Strict CSV reader supporting quoted commas, escaped quotes, CRLF and multiline cells.
pub fn csv(text: &str) -> Result<Vec<Vec<String>>, String> {
    let mut rows = Vec::new();
    let mut row = Vec::new();
    let mut cell = String::new();
    let mut quoted = false;
    let mut closed = false;
    let mut chars = text.trim_start_matches('\u{feff}').chars().peekable();
    while let Some(c) = chars.next() {
        if quoted {
            if c == '"' {
                if chars.peek() == Some(&'"') {
                    chars.next();
                    cell.push('"');
                } else {
                    quoted = false;
                    closed = true;
                }
            } else {
                cell.push(c);
            }
            continue;
        }
        match c {
            '"' if cell.is_empty() && !closed => quoted = true,
            ',' => {
                row.push(std::mem::take(&mut cell));
                closed = false;
            }
            '\n' | '\r' => {
                if c == '\r' && chars.peek() == Some(&'\n') {
                    chars.next();
                }
                row.push(std::mem::take(&mut cell));
                if row.len() > 1 || closed || row.iter().any(|v| !v.is_empty()) {
                    rows.push(std::mem::take(&mut row));
                } else {
                    row.clear();
                }
                closed = false;
            }
            _ if closed || c == '"' => {
                return Err(format!(
                    "CSV record {}: unexpected character after a quote",
                    rows.len() + 1
                ));
            }
            _ => cell.push(c),
        }
    }
    if quoted {
        return Err("CSV has an unclosed quoted field".into());
    }
    if !cell.is_empty() || !row.is_empty() || closed {
        row.push(cell);
        rows.push(row);
    }
    let Some(header) = rows.first_mut() else {
        return Err("CSV is empty".into());
    };
    for h in header.iter_mut() {
        *h = h.trim().to_string();
    }
    let mut names = BTreeSet::new();
    if header
        .iter()
        .any(|h| h.is_empty() || !names.insert(h.clone()))
    {
        return Err("CSV headers must be nonempty and unique".into());
    }
    let width = header.len();
    for (i, row) in rows.iter().enumerate().skip(1) {
        if row.len() != width {
            return Err(format!(
                "CSV record {}: expected {width} columns, got {}",
                i + 1,
                row.len()
            ));
        }
    }
    if rows.len() < 2 {
        return Err("CSV contains no data rows".into());
    }
    Ok(rows)
}
fn ids(rows: &[Vec<String>], index: usize) -> Result<(), String> {
    let mut seen = BTreeSet::new();
    for (i, row) in rows.iter().enumerate().skip(1) {
        let id = &row[index];
        if id.is_empty() || id.chars().any(char::is_whitespace) || !seen.insert(id) {
            return Err(format!(
                "CSV record {}: missing, duplicate, or whitespace-containing ID {id:?}",
                i + 1
            ));
        }
    }
    Ok(())
}
pub struct Embeddings {
    pub records: Vec<Record>,
    pub vectors: Vec<Vec<f32>>,
}
pub fn embeddings(text: &str) -> Result<Embeddings, String> {
    let rows = csv(text)?;
    let header = &rows[0];
    let id = header
        .iter()
        .position(|h| h == "id")
        .ok_or("Missing id column")?;
    ids(&rows, id)?;
    let description = header.iter().position(|h| h == "description");
    let sequence = header.iter().position(|h| h == "sequence");
    let mut dims = Vec::new();
    for (column, h) in header.iter().enumerate() {
        if ["id", "description", "sequence"].contains(&h.as_str()) {
            continue;
        }
        let n = h
            .strip_prefix("embedding_")
            .and_then(|s| s.parse::<usize>().ok())
            .ok_or_else(|| format!("Unknown column {h:?}; use embedding_0, embedding_1, …"))?;
        if h != &format!("embedding_{n}") {
            return Err(format!("Invalid dimension header {h:?}"));
        }
        dims.push((n, column));
    }
    dims.sort_unstable();
    if dims.is_empty() || dims.iter().enumerate().any(|(i, (n, _))| i != *n) {
        return Err("Embedding dimensions must be contiguous from embedding_0".into());
    }
    let mut records = Vec::new();
    let mut vectors = Vec::new();
    for (i, row) in rows.iter().enumerate().skip(1) {
        let seq = sequence
            .map(|j| row[j].trim().to_ascii_uppercase())
            .unwrap_or_default();
        if !seq
            .chars()
            .all(|c| "ACDEFGHIKLMNPQRSTVWYBXZJUO".contains(c))
        {
            return Err(format!("CSV record {}: invalid protein sequence", i + 1));
        }
        let v: Vec<f32> =
            dims.iter()
                .map(|(_, j)| {
                    row[*j].trim().parse::<f32>().map_err(|_| {
                        format!("CSV record {}: invalid number in {}", i + 1, header[*j])
                    })
                })
                .collect::<Result<_, _>>()?;
        records.push(Record {
            id: row[id].clone(),
            description: description.map(|j| row[j].clone()).unwrap_or_default(),
            seq,
        });
        vectors.push(v);
    }
    crate::workspace::validate_vectors(&records, &vectors)?;
    Ok(Embeddings { records, vectors })
}
#[derive(Clone, Default, Debug, Serialize, Deserialize)]
pub struct Annotations {
    pub columns: Vec<String>,
    pub rows: BTreeMap<String, Vec<String>>,
}
impl Annotations {
    pub fn label(&self, id: &str, column: &str) -> Option<&str> {
        let j = self.columns.iter().position(|c| c == column)?;
        self.rows.get(id)?.get(j).map(String::as_str)
    }
    pub fn valid(&self, ids: &BTreeSet<&str>) -> bool {
        let names: BTreeSet<_> = self.columns.iter().collect();
        (self.rows.is_empty() || !self.columns.is_empty())
            && names.len() == self.columns.len()
            && self.columns.iter().all(|c| !c.is_empty() && c != "id")
            && self
                .rows
                .iter()
                .all(|(id, v)| ids.contains(id.as_str()) && v.len() == self.columns.len())
    }
}
pub fn annotations(text: &str, known_ids: &BTreeSet<&str>) -> Result<Annotations, String> {
    let rows = csv(text)?;
    let header = &rows[0];
    let id = header
        .iter()
        .position(|h| h == "id")
        .ok_or("Missing id column")?;
    ids(&rows, id)?;
    let columns: Vec<_> = header
        .iter()
        .enumerate()
        .filter(|(j, _)| *j != id)
        .map(|(_, h)| h.clone())
        .collect();
    if columns.is_empty() {
        return Err("Annotations need at least one column besides id".into());
    }
    let mut values = BTreeMap::new();
    for row in rows.iter().skip(1) {
        if !known_ids.contains(row[id].as_str()) {
            return Err(format!(
                "Annotation ID {:?} does not occur in the loaded embeddings",
                row[id]
            ));
        }
        values.insert(
            row[id].clone(),
            row.iter()
                .enumerate()
                .filter(|(j, _)| *j != id)
                .map(|(_, v)| v.trim().to_string())
                .collect(),
        );
    }
    Ok(Annotations {
        columns,
        rows: values,
    })
}
pub fn attach_sequences(records: &[Record], sequences: &[Record]) -> Result<Vec<Record>, String> {
    let by_id: BTreeMap<_, _> = sequences.iter().map(|r| (r.id.as_str(), r)).collect();
    if by_id.len() != records.len() || sequences.len() != records.len() {
        return Err("FASTA must contain exactly the loaded protein IDs".into());
    }
    records
        .iter()
        .map(|r| {
            by_id
                .get(r.id.as_str())
                .map(|s| (*s).clone())
                .ok_or_else(|| format!("FASTA is missing {}", r.id))
        })
        .collect()
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn csv_quoting_and_failures() {
        let r = csv("\u{feff}id,description\r\na,\"line 1\nline 2, \"\"quote\"\"\"\r\n");
        assert!(r.is_ok());
        for text in ["id,x\na,\"open", "id,x\na,\"a\"b", "id,id\na,b", "id,x\na"] {
            assert!(csv(text).is_err(), "{text}");
        }
    }
    #[test]
    fn embedding_import_round_trips_and_reorders_dimensions() {
        let r = vec![Record {
            id: "a,1".into(),
            description: "a \"quote\"\nline".into(),
            seq: "ACD".into(),
        }];
        let text = crate::workspace::embeddings_csv(&r, &[vec![1., 2.]]);
        let e = embeddings(&text).expect("round trip");
        assert_eq!(e.records[0].description, r[0].description);
        assert_eq!(e.vectors, vec![vec![1., 2.]]);
        let e = embeddings("id,embedding_1,embedding_0\na,2,1\n").expect("no sequences");
        assert_eq!(e.vectors, vec![vec![1., 2.]]);
        assert!(e.records[0].seq.is_empty());
        for t in [
            "id,embedding_0\na,NaN",
            "id,embedding_0\na,0",
            "id,embedding_1\na,1",
            "id,embedding_0\na,1\na,2",
            "id,embedding_0\na,no",
        ] {
            assert!(embeddings(t).is_err(), "{t}");
        }
    }
    #[test]
    fn annotations_join_by_id_and_allow_missing_labels() {
        let ids = BTreeSet::from(["a", "b"]);
        let a = annotations("id,family,taxon\nb,B,yeast\na,,human", &ids).expect("annotations");
        assert_eq!(a.label("a", "family"), Some(""));
        assert_eq!(a.label("b", "taxon"), Some("yeast"));
        assert!(annotations("id,family\nx,X", &ids).is_err());
        assert!(annotations("id,family\na,A\na,B", &ids).is_err());
    }
    #[test]
    fn fasta_attachment_matches_ids_not_row_order() {
        let r = vec![
            Record {
                id: "a".into(),
                description: String::new(),
                seq: String::new(),
            },
            Record {
                id: "b".into(),
                description: String::new(),
                seq: String::new(),
            },
        ];
        let seq = crate::fasta::preview(">b second\nMK\n>a first\nACD").records;
        let result = attach_sequences(&r, &seq).expect("attach");
        assert_eq!(result[0].seq, "ACD");
        assert!(attach_sequences(&r, &seq[..1]).is_err());
    }
}
