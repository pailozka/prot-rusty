//! Minimal FASTA reader: `>header` lines followed by sequence lines.

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Record {
    /// First whitespace-separated word of the header, e.g. `sp|P0CG48|UBC_HUMAN`.
    pub id: String,
    /// Rest of the header line (may be empty).
    pub description: String,
    pub seq: String,
}

pub fn parse(text: &str) -> Vec<Record> {
    let mut records = Vec::new();
    let mut current: Option<Record> = None;
    for line in text.lines() {
        let line = line.trim();
        if let Some(header) = line.strip_prefix('>') {
            if let Some(r) = current.take() {
                records.push(r);
            }
            let (id, description) = header
                .split_once(char::is_whitespace)
                .unwrap_or((header, ""));
            current = Some(Record {
                id: id.to_string(),
                description: description.trim().to_string(),
                seq: String::new(),
            });
        } else if let Some(r) = current.as_mut() {
            r.seq.extend(line.chars().filter(|c| !c.is_whitespace()));
        }
    }
    if let Some(r) = current {
        records.push(r);
    }
    records.retain(|r| !r.seq.is_empty());
    records
}

#[derive(Default)]
pub struct Preview {
    pub records: Vec<Record>,
    pub errors: Vec<String>,
}
impl Preview {
    pub fn valid(&self) -> bool {
        !self.records.is_empty() && self.errors.is_empty()
    }
    pub fn summary(&self) -> String {
        let lengths: Vec<_> = self.records.iter().map(|r| r.seq.len()).collect();
        let total: usize = lengths.iter().sum();
        format!(
            "{} sequences · {} residues · {}–{} aa · mean {:.1} aa",
            lengths.len(),
            total,
            lengths.iter().min().copied().unwrap_or(0),
            lengths.iter().max().copied().unwrap_or(0),
            total as f64 / lengths.len().max(1) as f64
        )
    }
}
pub fn preview(text: &str) -> Preview {
    let mut errors = Vec::new();
    let mut header = false;
    let mut residues = 0;
    let mut ids = std::collections::HashSet::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(h) = line.strip_prefix('>') {
            if header && residues == 0 {
                errors.push(format!("Line {}: preceding sequence is empty", i + 1));
            }
            let id = h.split_whitespace().next().unwrap_or("");
            if id.is_empty() {
                errors.push(format!("Line {}: missing sequence ID", i + 1));
            } else if !ids.insert(id.to_string()) {
                errors.push(format!("Line {}: duplicate ID {id}", i + 1));
            }
            header = true;
            residues = 0;
        } else {
            if !header {
                errors.push(format!("Line {}: sequence before a FASTA header", i + 1));
            }
            for c in line.chars().filter(|c| !c.is_whitespace()) {
                residues += 1;
                if !"ACDEFGHIKLMNPQRSTVWYBXZJUO".contains(c.to_ascii_uppercase()) {
                    errors.push(format!("Line {}: invalid residue {c:?}", i + 1));
                    break;
                }
            }
        }
    }
    if header && residues == 0 {
        errors.push("Last sequence is empty".into());
    }
    let mut records = parse(text);
    for r in &mut records {
        r.seq.make_ascii_uppercase();
    }
    if records.is_empty() {
        errors.push("No sequences found".into());
    }
    Preview { records, errors }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_imports() {
        assert!(!preview("ACD\n>a\nACD").valid());
        assert!(!preview(">a\nACD\n>a\nMK").valid());
        assert!(!preview(">a\nACD*\n>b\n").valid());
        assert!(!preview(">\nACD").valid());
        let p = preview(">a desc\nac d\n>b\nMK");
        assert!(p.valid());
        assert_eq!(p.records[0].seq, "ACD");
        assert!(p.summary().contains("5 residues"));
    }
    #[test]
    fn parses_multiline_records() {
        let recs = parse(">a first one\nMKV\nLL\n\n>b\nACD\n>empty\n");
        assert_eq!(recs.len(), 2);
        assert_eq!(recs[0].id, "a");
        assert_eq!(recs[0].description, "first one");
        assert_eq!(recs[0].seq, "MKVLL");
        assert_eq!(recs[1].seq, "ACD");
    }
}
