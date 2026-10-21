mod benchmark;
mod demo;
mod esm;
mod fasta;
mod imports;
mod models;
mod projection;
mod prott5;
mod rns;
mod workspace;

use models::Model;

use demo::{CLUSTER_NAMES, Protein};
use eframe::egui::{self, Color32, RichText};
use egui_plot::{Line, Plot, Points};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use tokio::runtime::Runtime;

const PALETTE: [Color32; 8] = [
    Color32::from_rgb(86, 156, 247),
    Color32::from_rgb(244, 114, 94),
    Color32::from_rgb(98, 209, 160),
    Color32::from_rgb(250, 204, 92),
    Color32::from_rgb(178, 132, 247),
    Color32::from_rgb(240, 138, 196),
    Color32::from_rgb(94, 214, 226),
    Color32::from_rgb(244, 166, 80),
];

const BG: Color32 = Color32::from_rgb(246, 248, 247);
const PANEL: Color32 = Color32::from_rgb(255, 255, 255);
const MUTED: Color32 = Color32::from_rgb(112, 128, 125);
const ACCENT: Color32 = Color32::from_rgb(20, 119, 101);

/// Messages sent from the background download task to the UI thread.
enum Msg {
    Status(String),
    Progress(f32),
    Done(String),
    Embedded(
        Model,
        Vec<fasta::Record>,
        Vec<Vec<f32>>,
        projection::Projection,
    ),
    Projected(Result<projection::Projection, String>),
    Imported(Result<(imports::Embeddings, projection::Projection, String), String>),
}

struct ProtRustApp {
    fasta_path: Option<String>,
    selected_model: Model,
    status: String,
    progress: f32,
    runtime: Arc<Runtime>,
    tx: Sender<Msg>,
    rx: Receiver<Msg>,
    downloading: bool,
    confirm_big: bool,
    proteins: Vec<Protein>,
    cluster_on: [bool; 8],
    search: String,
    selected: Option<usize>,
    /// k nearest visible-or-not neighbors of `selected`: (index, distance).
    neighbors: Vec<(usize, f64)>,
    /// Real data from the last run (FASTA records and their ESM embeddings).
    records: Vec<fasta::Record>,
    embeddings: Vec<Vec<f32>>,
    reset_map: bool,
    preview: fasta::Preview,
    embedding_model: Option<Model>,
    projection: Option<projection::Projection>,
    view: projection::View,
    external_source: Option<String>,
    annotations: imports::Annotations,
    active_annotation: Option<String>,
    hidden_labels: BTreeSet<String>,
}

const K_NEIGHBORS: usize = 10;

/// Reports byte-level progress of one big file to the UI thread.
/// hf-hub downloads in parallel chunks, so the counters are shared atomics.
#[derive(Clone)]
struct ByteProgress {
    tx: Sender<Msg>,
    ctx: egui::Context,
    total: Arc<AtomicUsize>,
    seen: Arc<AtomicUsize>,
}

impl hf_hub::api::tokio::Progress for ByteProgress {
    async fn init(&mut self, size: usize, _filename: &str) {
        self.total.store(size, Ordering::Relaxed);
    }

    async fn update(&mut self, size: usize) {
        let seen = self.seen.fetch_add(size, Ordering::Relaxed) + size;
        let total = self.total.load(Ordering::Relaxed).max(1);
        let _ = self.tx.send(Msg::Progress(seen as f32 / total as f32));
        let _ = self.tx.send(Msg::Status(format!(
            "Downloading weights… {:.0} / {:.0} MB",
            seen as f64 / 1e6,
            total as f64 / 1e6
        )));
        self.ctx.request_repaint();
    }

    async fn finish(&mut self) {}
}

/// Download the files needed for inference; returns the local cache dir on success.
/// Files already in the Hugging Face cache are reused, so a second run is instant.
async fn download_model(
    model: Model,
    tx: &Sender<Msg>,
    ctx: &egui::Context,
) -> Result<String, Box<dyn std::error::Error>> {
    let repo = hf_hub::api::tokio::Api::new()?.model(model.repo().to_string());
    // Small files first: they are tiny, and failing early on a bad repo is cheap.
    for file in model.small_files() {
        repo.get(file).await?;
    }

    let weights = match repo.get(model.weights_file()).await {
        Ok(path) => path, // cached
        Err(_) => {
            let progress = ByteProgress {
                tx: tx.clone(),
                ctx: ctx.clone(),
                total: Arc::new(AtomicUsize::new(0)),
                seen: Arc::new(AtomicUsize::new(0)),
            };
            repo.download_with_progress(model.weights_file(), progress)
                .await?
        }
    };
    let _ = tx.send(Msg::Progress(1.0));
    Ok(weights
        .parent()
        .map(|p| p.display().to_string())
        .unwrap_or_default())
}

/// Load the model and embed every FASTA record. CPU-heavy, so it runs on a
/// blocking thread rather than inside an async task.
fn embed_fasta(
    which: Model,
    model_dir: &std::path::Path,
    records: Vec<fasta::Record>,
    tx: &Sender<Msg>,
    ctx: &egui::Context,
) -> Result<(Vec<fasta::Record>, Vec<Vec<f32>>), Box<dyn std::error::Error + Send + Sync>> {
    let _ = tx.send(Msg::Status("Loading model…".into()));
    let mut model = which.load(model_dir)?;
    if records.is_empty() {
        return Err("no sequences found in FASTA file".into());
    }
    let mut vectors = Vec::with_capacity(records.len());
    for (i, r) in records.iter().enumerate() {
        let vector = model.embed(&r.seq)?;
        if vector.len() != model.dim() {
            return Err("Model returned an unexpected embedding dimension".into());
        }
        vectors.push(vector);
        let _ = tx.send(Msg::Progress((i + 1) as f32 / records.len() as f32));
        let _ = tx.send(Msg::Status(format!(
            "Embedding {}/{} · {}",
            i + 1,
            records.len(),
            r.id
        )));
        ctx.request_repaint();
    }
    Ok((records, vectors))
}

fn apply_theme(ctx: &egui::Context) {
    let mut v = egui::Visuals::light();
    v.panel_fill = PANEL;
    v.window_fill = PANEL;
    v.extreme_bg_color = BG;
    v.selection.bg_fill = Color32::from_rgb(208, 235, 225);
    v.override_text_color = Some(Color32::from_rgb(36, 56, 51));
    v.widgets.noninteractive.bg_stroke.color = Color32::from_rgb(225, 232, 228);
    let r = egui::Rounding::same(6.0);
    v.widgets.noninteractive.rounding = r;
    v.widgets.inactive.rounding = r;
    v.widgets.hovered.rounding = r;
    v.widgets.active.rounding = r;
    ctx.set_visuals(v);

    let mut style = (*ctx.style()).clone();
    style.spacing.item_spacing = egui::vec2(10.0, 10.0);
    style
        .text_styles
        .insert(egui::TextStyle::Body, egui::FontId::proportional(14.0));
    style
        .text_styles
        .insert(egui::TextStyle::Button, egui::FontId::proportional(14.0));
    style.spacing.button_padding = egui::vec2(12.0, 8.0);
    ctx.set_style(style);
}

fn section(ui: &mut egui::Ui, title: &str) {
    ui.add_space(6.0);
    ui.label(
        RichText::new(title.to_uppercase())
            .size(11.0)
            .color(MUTED)
            .strong(),
    );
}

impl ProtRustApp {
    fn new(cc: &eframe::CreationContext<'_>, runtime: Arc<Runtime>) -> Self {
        apply_theme(&cc.egui_ctx);
        let (tx, rx) = channel();
        Self {
            fasta_path: None,
            selected_model: Model::Esm2_8M,
            status: String::from("Ready"),
            progress: 0.0,
            runtime,
            tx,
            rx,
            downloading: false,
            confirm_big: false,
            proteins: demo::generate(6000),
            cluster_on: [true; 8],
            search: String::new(),
            selected: None,
            neighbors: Vec::new(),
            records: Vec::new(),
            embeddings: Vec::new(),
            reset_map: false,
            preview: fasta::Preview::default(),
            embedding_model: None,
            projection: None,
            view: projection::View::default(),
            external_source: None,
            annotations: imports::Annotations::default(),
            active_annotation: None,
            hidden_labels: BTreeSet::new(),
        }
    }

    fn select(&mut self, idx: Option<usize>) {
        self.selected = idx.filter(|i| *i < self.proteins.len());
        self.neighbors.clear();
        let Some(i) = self.selected else { return };
        if !self.embeddings.is_empty() {
            self.neighbors = workspace::neighbors(&self.embeddings, i, K_NEIGHBORS);
        } else {
            let me = &self.proteins[i];
            let mut all: Vec<_> = self
                .proteins
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(j, p)| {
                    (
                        j,
                        ((p.pos[0] - me.pos[0]).powi(2) + (p.pos[1] - me.pos[1]).powi(2)).sqrt(),
                    )
                })
                .collect();
            all.sort_by(|a, b| a.1.total_cmp(&b.1));
            all.truncate(K_NEIGHBORS);
            self.neighbors = all;
        }
    }

    fn apply_import(
        &mut self,
        data: imports::Embeddings,
        projection: projection::Projection,
        source: String,
    ) {
        self.proteins = data
            .records
            .iter()
            .zip(&projection.points)
            .map(|(r, p)| Protein {
                id: r.id.clone(),
                seq_len: r.seq.len(),
                cluster: 0,
                pos: [p[0], p[1]],
            })
            .collect();
        self.records = data.records;
        self.embeddings = data.vectors;
        self.projection = Some(projection);
        self.external_source = Some(source);
        self.embedding_model = None;
        self.annotations = imports::Annotations::default();
        self.active_annotation = None;
        self.hidden_labels.clear();
        self.search.clear();
        self.cluster_on = [true; 8];
        self.select(None);
        self.view = projection::View::default();
        self.reset_map = true;
        self.status = format!(
            "Imported {} proteins · {} dimensions · no model execution",
            self.records.len(),
            self.embeddings[0].len()
        );
    }

    fn import_embeddings(&mut self, ctx: &egui::Context) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("Embeddings CSV", &["csv"])
            .pick_file()
        else {
            return;
        };
        self.downloading = true;
        self.progress = 0.0;
        self.status = "Reading embeddings and computing PCA…".into();
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        self.runtime.spawn_blocking(move || {
            let result = (|| -> Result<_, String> {
                let data = imports::embeddings(
                    &std::fs::read_to_string(&path).map_err(|e| e.to_string())?,
                )?;
                let projection = projection::pca(&data.vectors)?;
                Ok((
                    data,
                    projection,
                    path.file_name()
                        .map(|p| p.to_string_lossy().to_string())
                        .unwrap_or_default(),
                ))
            })();
            let _ = tx.send(Msg::Imported(result));
            ctx.request_repaint();
        });
    }

    fn load_example(&mut self) {
        let result = (|| -> Result<_, String> {
            let data = imports::embeddings(include_str!("../examples/embeddings.csv"))?;
            let projection = projection::pca(&data.vectors)?;
            let ids = data.records.iter().map(|r| r.id.as_str()).collect();
            let annotations =
                imports::annotations(include_str!("../examples/annotations.csv"), &ids)?;
            Ok((data, projection, annotations))
        })();
        match result {
            Ok((data, projection, annotations)) => {
                self.apply_import(
                    data,
                    projection,
                    "Synthetic example vectors (not PLM output)".into(),
                );
                self.annotations = annotations;
                self.active_annotation = Some("family".into());
                self.status =
                    "Synthetic example loaded · 8 artificial vectors, not biological evidence"
                        .into();
            }
            Err(e) => self.status = e,
        }
    }

    fn import_annotations(&mut self) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("Annotations CSV", &["csv"])
            .pick_file()
        else {
            return;
        };
        let result = std::fs::read_to_string(path)
            .map_err(|e| e.to_string())
            .and_then(|text| {
                let ids = self.proteins.iter().map(|p| p.id.as_str()).collect();
                imports::annotations(&text, &ids)
            });
        match result {
            Ok(annotations) => {
                let count = annotations.rows.len();
                self.active_annotation = annotations.columns.first().cloned();
                self.annotations = annotations;
                self.hidden_labels.clear();
                self.status = format!(
                    "Annotations matched to {count}/{} proteins by ID; unmatched proteins remain unannotated",
                    self.proteins.len()
                );
            }
            Err(e) => self.status = format!("Annotation import failed: {e}"),
        }
    }

    fn attach_fasta(&mut self) {
        match imports::attach_sequences(&self.records, &self.preview.records) {
            Ok(records) => {
                self.records = records;
                for (p, r) in self.proteins.iter_mut().zip(&self.records) {
                    p.seq_len = r.seq.len();
                }
                self.status = "Attached FASTA by protein ID; embeddings unchanged".into();
            }
            Err(e) => self.status = format!("FASTA attachment failed: {e}"),
        }
    }

    fn annotation_label(&self, i: usize) -> &str {
        self.active_annotation
            .as_deref()
            .and_then(|c| self.annotations.label(&self.proteins[i].id, c))
            .unwrap_or("")
    }
    fn annotation_categories(&self) -> Vec<String> {
        (0..self.proteins.len())
            .map(|i| self.annotation_label(i).to_string())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }
    fn category_color(index: usize) -> Color32 {
        egui::ecolor::Hsva::new((index as f32 * 0.618034) % 1.0, 0.65, 0.75, 1.0).into()
    }
    fn point_colors(&self) -> Vec<Color32> {
        let categories: BTreeMap<_, _> = self
            .annotation_categories()
            .into_iter()
            .enumerate()
            .map(|(i, c)| (c, Self::category_color(i)))
            .collect();
        (0..self.proteins.len())
            .map(|i| {
                if self.active_annotation.is_some() {
                    if self.annotation_label(i).is_empty() {
                        MUTED
                    } else {
                        categories[self.annotation_label(i)]
                    }
                } else {
                    PALETTE[self.proteins[i].cluster]
                }
            })
            .collect()
    }

    fn start_projection(&mut self) {
        self.downloading = true;
        self.status = "Computing PCA projection…".into();
        let vectors = self.embeddings.clone();
        let tx = self.tx.clone();
        self.runtime.spawn_blocking(move || {
            let _ = tx.send(Msg::Projected(projection::pca(&vectors)));
        });
    }

    fn project(&self) -> workspace::Project {
        workspace::Project {
            version: 1,
            model: self.embedding_model.unwrap_or(self.selected_model),
            configured_model: Some(self.selected_model),
            input_records: if self.preview.valid() {
                self.preview.records.clone()
            } else {
                Vec::new()
            },
            records: self.records.clone(),
            embeddings: self.embeddings.clone(),
            proteins: self.proteins.clone(),
            selected: self.selected,
            search: self.search.clone(),
            cluster_on: self.cluster_on,
            projection: self.projection.clone(),
            view: self.view.clone(),
            external_source: self.external_source.clone(),
            annotations: self.annotations.clone(),
            active_annotation: self.active_annotation.clone(),
            hidden_labels: self.hidden_labels.clone(),
        }
    }

    fn save_project(&mut self) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("ProtRust project", &["protjson"])
            .set_file_name("workspace.protjson")
            .save_file()
        else {
            return;
        };
        let project = self.project();
        let result = (|| -> Result<(), Box<dyn std::error::Error>> {
            let bytes = serde_json::to_vec(&project)?;
            std::fs::write(path, bytes)?;
            Ok(())
        })();
        self.status = match result {
            Ok(()) => "Project saved (sequences and embeddings included)".into(),
            Err(e) => format!("Save failed: {e}"),
        };
    }

    fn open_project(&mut self) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("ProtRust project", &["protjson"])
            .pick_file()
        else {
            return;
        };
        let result = (|| -> Result<workspace::Project, Box<dyn std::error::Error>> {
            let project: workspace::Project = serde_json::from_slice(&std::fs::read(path)?)?;
            project.validate().map_err(std::io::Error::other)?;
            Ok(project)
        })();
        match result {
            Ok(p) => {
                self.selected_model = p.configured_model.unwrap_or(p.model);
                self.embedding_model =
                    (!p.embeddings.is_empty() && p.external_source.is_none()).then_some(p.model);
                self.external_source = p.external_source;
                self.annotations = p.annotations;
                self.active_annotation = p.active_annotation;
                self.hidden_labels = p.hidden_labels;
                self.records = p.records;
                self.embeddings = p.embeddings;
                self.proteins = p.proteins;
                self.projection = p.projection;
                self.view = p.view;
                if let Some(projection) = &self.projection {
                    for (p, point) in self.proteins.iter_mut().zip(&projection.points) {
                        p.pos = [point[0], point[1]];
                    }
                }
                self.search = p.search;
                self.cluster_on = p.cluster_on;
                self.fasta_path = None;
                self.preview = fasta::Preview {
                    records: p.input_records,
                    errors: Vec::new(),
                };
                self.select(p.selected);
                self.reset_map = true;
                self.status = "Project reopened; no inference required".into();
                if self.projection.is_none() && !self.embeddings.is_empty() {
                    self.start_projection();
                }
            }
            Err(e) => self.status = format!("Open failed: {e}"),
        }
    }

    fn export(&mut self, kind: &str) {
        let mut exported_proteins = self.proteins.clone();
        if self.view.three_d {
            for (i, p) in exported_proteins.iter_mut().enumerate() {
                let point = self.view.rotate(self.point_3d(i));
                p.pos = [point[0], point[1]];
            }
        }
        let (name, ext, text) = match kind {
            "evaluation" => {
                let Some(column) = self.active_annotation.as_deref() else {
                    return;
                };
                let report = match benchmark::evaluate(
                    &self.records,
                    &self.embeddings,
                    &self.annotations,
                    column,
                    K_NEIGHBORS,
                ) {
                    Ok(report) => report,
                    Err(e) => {
                        self.status = e;
                        return;
                    }
                };
                let output = serde_json::json!({"evaluation":report,"inputs":{"embeddings_csv":workspace::embeddings_csv(&self.records,&self.embeddings),"annotations":self.annotations,"external_source":self.external_source,"embedding_model":self.embedding_model}});
                ("label-retrieval.json", "json", output.to_string())
            }
            "embeddings" => (
                "embeddings.csv",
                "csv",
                workspace::embeddings_csv(&self.records, &self.embeddings),
            ),
            "neighbors" => {
                let Some(i) = self.selected else { return };
                (
                    "neighbors.csv",
                    "csv",
                    workspace::neighbors_csv(
                        &self.proteins,
                        i,
                        &self.neighbors,
                        !self.embeddings.is_empty(),
                    ),
                )
            }
            _ => (
                "protein-atlas.svg",
                "svg",
                workspace::map_svg(
                    &exported_proteins,
                    &self.visibility(),
                    self.selected,
                    &self.point_colors(),
                    if self.records.is_empty() {
                        "Demo atlas"
                    } else if self.view.three_d {
                        "PCA 3D (rotated view)"
                    } else {
                        "PCA 2D"
                    },
                ),
            ),
        };
        let Some(path) = rfd::FileDialog::new()
            .add_filter(ext, &[ext])
            .set_file_name(name)
            .save_file()
        else {
            return;
        };
        self.status = match std::fs::write(path, text) {
            Ok(()) => format!("Exported {name}"),
            Err(e) => format!("Export failed: {e}"),
        };
    }

    fn start_download(&mut self, ctx: &egui::Context) {
        self.downloading = true;
        self.progress = 0.0;
        let which = self.selected_model;
        let input_records = self.preview.records.clone();
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        self.runtime.spawn(async move {
            let dir = match download_model(which, &tx, &ctx).await {
                Ok(dir) => dir,
                Err(e) => {
                    let _ = tx.send(Msg::Done(format!("Download failed: {e}")));
                    ctx.request_repaint();
                    return;
                }
            };
            if input_records.is_empty() {
                let _ = tx.send(Msg::Done(
                    "Model ready. Pick a FASTA file and Run to embed.".into(),
                ));
                ctx.request_repaint();
                return;
            }
            let (tx2, ctx2) = (tx.clone(), ctx.clone());
            let job = tokio::task::spawn_blocking(move || {
                let (records, vectors) =
                    embed_fasta(which, dir.as_ref(), input_records, &tx2, &ctx2)?;
                let _ = tx2.send(Msg::Status("Computing PCA projection…".into()));
                ctx2.request_repaint();
                let projection = projection::pca(&vectors).map_err(std::io::Error::other)?;
                Ok::<_, Box<dyn std::error::Error + Send + Sync>>((records, vectors, projection))
            });
            let msg = match job.await {
                Ok(Ok((records, vectors, projection))) => {
                    let _ = tx.send(Msg::Embedded(which, records, vectors, projection));
                    Msg::Done(String::new())
                }
                Ok(Err(e)) => Msg::Done(format!("Embedding failed: {e}")),
                Err(e) => Msg::Done(format!("Embedding task crashed: {e}")),
            };
            let _ = tx.send(msg);
            ctx.request_repaint();
        });
    }

    fn drain_messages(&mut self) {
        while let Ok(msg) = self.rx.try_recv() {
            match msg {
                Msg::Status(s) => self.status = s,
                Msg::Progress(p) => self.progress = p,
                Msg::Done(s) => {
                    if !s.is_empty() {
                        self.status = s;
                    }
                    self.downloading = false;
                }
                Msg::Imported(result) => {
                    self.downloading = false;
                    match result {
                        Ok((data, projection, source)) => {
                            self.apply_import(data, projection, source)
                        }
                        Err(e) => self.status = format!("Import failed: {e}"),
                    }
                }
                Msg::Projected(result) => {
                    self.downloading = false;
                    match result {
                        Ok(projection) => {
                            for (p, point) in self.proteins.iter_mut().zip(&projection.points) {
                                p.pos = [point[0], point[1]];
                            }
                            self.projection = Some(projection);
                            self.reset_map = true;
                            self.status = "PCA ready · neighbors use full embeddings".into();
                        }
                        Err(e) => self.status = format!("Projection failed: {e}"),
                    }
                }
                Msg::Embedded(model, records, vectors, projection) => {
                    if let Err(e) = workspace::validate_vectors(&records, &vectors) {
                        self.status = e;
                        continue;
                    }
                    let dim = vectors.first().map_or(0, |v| v.len());
                    self.status = format!(
                        "Embedded {} sequences · {dim}-dim. Neighbors use cosine distance; map shows PCA.",
                        records.len()
                    );
                    self.proteins = records
                        .iter()
                        .enumerate()
                        .map(|(i, r)| {
                            let point = projection.points[i];
                            Protein {
                                id: r.id.clone(),
                                seq_len: r.seq.len(),
                                cluster: 0,
                                pos: [point[0], point[1]],
                            }
                        })
                        .collect();
                    self.projection = Some(projection);
                    self.external_source = None;
                    self.annotations = imports::Annotations::default();
                    self.active_annotation = None;
                    self.hidden_labels.clear();
                    self.view = projection::View::default();
                    self.embedding_model = Some(model);
                    self.select(None);
                    self.search.clear();
                    self.cluster_on = [true; 8];
                    self.reset_map = true;
                    self.records = records;
                    self.embeddings = vectors;
                }
            }
        }
    }

    /// One flag per protein: passes the cluster filter and the search box.
    fn visibility(&self) -> Vec<bool> {
        let q = self.search.trim().to_lowercase();
        self.proteins
            .iter()
            .enumerate()
            .map(|(i, p)| {
                (!self.records.is_empty() || self.cluster_on[p.cluster])
                    && (self.active_annotation.is_none()
                        || !self.hidden_labels.contains(self.annotation_label(i)))
                    && (q.is_empty()
                        || p.id.to_lowercase().contains(&q)
                        || self
                            .annotations
                            .rows
                            .get(&p.id)
                            .is_some_and(|row| row.iter().any(|v| v.to_lowercase().contains(&q)))
                        || (self.records.is_empty()
                            && CLUSTER_NAMES[p.cluster].to_lowercase().contains(&q))
                        || self
                            .records
                            .get(i)
                            .is_some_and(|r| r.description.to_lowercase().contains(&q)))
            })
            .collect()
    }

    fn top_bar(&mut self, ctx: &egui::Context, shown: usize) {
        egui::TopBottomPanel::top("top")
            .frame(
                egui::Frame::none()
                    .fill(PANEL)
                    .inner_margin(egui::Margin::symmetric(24.0, 18.0)),
            )
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new("◈").size(30.0).color(ACCENT));
                    ui.label(RichText::new("ProtRust").size(24.0).strong());
                    ui.add_space(22.0);
                    ui.add_enabled_ui(!self.downloading, |ui| {
                        if ui.button("Open project").clicked() {
                            self.open_project();
                        }
                        if ui.button("Save project").clicked() {
                            self.save_project();
                        }
                        ui.menu_button("Export", |ui| {
                            if ui
                                .add_enabled(
                                    !self.embeddings.is_empty(),
                                    egui::Button::new("Embeddings CSV"),
                                )
                                .clicked()
                            {
                                self.export("embeddings");
                                ui.close_menu();
                            }
                            if ui
                                .add_enabled(
                                    self.selected.is_some(),
                                    egui::Button::new("Selected neighbors CSV"),
                                )
                                .clicked()
                            {
                                self.export("neighbors");
                                ui.close_menu();
                            }
                            if ui
                                .add_enabled(
                                    self.active_annotation.is_some(),
                                    egui::Button::new("Label retrieval report (JSON)"),
                                )
                                .clicked()
                            {
                                self.export("evaluation");
                                ui.close_menu();
                            }
                            if ui.button("Map image (SVG)").clicked() {
                                self.export("map");
                                ui.close_menu();
                            }
                        });
                    });
                    ui.label(
                        RichText::new("WORKSPACE  /  Protein atlas")
                            .size(12.0)
                            .color(MUTED),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(
                            RichText::new("LOCAL WORKSPACE")
                                .size(11.0)
                                .color(ACCENT)
                                .strong(),
                        );
                        ui.label(RichText::new("●").color(ACCENT));
                    });
                });
            });
        egui::TopBottomPanel::bottom("status_bar")
            .frame(
                egui::Frame::none()
                    .fill(PANEL)
                    .inner_margin(egui::Margin::symmetric(20.0, 9.0)),
            )
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new(if self.downloading {
                            "● Processing"
                        } else {
                            "● Ready"
                        })
                        .small()
                        .color(ACCENT),
                    );
                    ui.separator();
                    ui.label(
                        RichText::new(format!(
                            "{shown} of {} proteins visible",
                            self.proteins.len()
                        ))
                        .small()
                        .color(MUTED),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(
                            RichText::new(if self.view.three_d {
                                "Drag to rotate · Shift-drag to pan · Scroll to zoom"
                            } else {
                                "Scroll to zoom  ·  Drag to pan  ·  Click to inspect"
                            })
                            .small()
                            .color(MUTED),
                        );
                    });
                });
            });
    }

    fn annotation_controls(&mut self, ui: &mut egui::Ui) {
        if self.annotations.columns.is_empty() {
            return;
        }
        let before = self.active_annotation.clone();
        egui::ComboBox::from_id_source("annotation_column")
            .selected_text(
                self.active_annotation
                    .as_deref()
                    .unwrap_or("No annotation coloring"),
            )
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut self.active_annotation, None, "No annotation coloring");
                for column in &self.annotations.columns {
                    ui.selectable_value(&mut self.active_annotation, Some(column.clone()), column);
                }
            });
        if before != self.active_annotation {
            self.hidden_labels.clear();
        }
        if self.active_annotation.is_some() {
            egui::ScrollArea::vertical()
                .id_source("annotation_legend")
                .max_height(180.0)
                .show(ui, |ui| {
                    for (index, label) in self.annotation_categories().into_iter().enumerate() {
                        let mut visible = !self.hidden_labels.contains(&label);
                        let color = if label.is_empty() {
                            MUTED
                        } else {
                            Self::category_color(index)
                        };
                        ui.horizontal(|ui| {
                            ui.checkbox(&mut visible, "");
                            ui.colored_label(color, "●");
                            ui.label(if label.is_empty() {
                                "Unannotated"
                            } else {
                                &label
                            });
                        });
                        if visible {
                            self.hidden_labels.remove(&label);
                        } else {
                            self.hidden_labels.insert(label);
                        }
                    }
                });
            if ui.small_button("Show all labels").clicked() {
                self.hidden_labels.clear();
            }
        }
    }

    fn left_panel(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("left").exact_width(280.0)
            .frame(egui::Frame::none().fill(PANEL).inner_margin(18.0)).show(ctx, |ui| {
            egui::ScrollArea::vertical().id_source("left_scroll").show(ui, |ui| {
            ui.label(RichText::new("Your workspace").size(19.0).strong());
            ui.label(RichText::new("From sequence to discovery.").small().color(MUTED));
            ui.add_space(12.0);
            ui.add_space(8.0);

            section(ui, "Precomputed data");
            ui.add_enabled_ui(!self.downloading, |ui| {
                if ui.add_sized([ui.available_width(),38.0],egui::Button::new("Import embeddings CSV")).clicked() { self.import_embeddings(ctx); }
                if ui.add_enabled(!self.embeddings.is_empty(),egui::Button::new("Import annotations CSV")).clicked() { self.import_annotations(); }
                if ui.button("Load small synthetic example").clicked() { self.load_example(); }
            });
            ui.label(RichText::new("Use existing vectors. No model download needed.").small().color(MUTED));
            if let Some(source) = &self.external_source { ui.label(RichText::new(format!("Source: {source} · model unspecified")).small().color(MUTED)); }
            self.annotation_controls(ui);
            section(ui, "01  /  Embedding model");
            egui::ComboBox::from_id_source("model")
                .width(ui.available_width())
                .selected_text(self.selected_model.label())
                .show_ui(ui, |ui| {
                    for m in Model::ALL {
                        ui.selectable_value(&mut self.selected_model, m, m.label());
                    }
                });

            section(ui, "02  /  Sequence input");
            if ui.add_sized([ui.available_width(), 40.0], egui::Button::new("+  Import FASTA file")).clicked() {
                let path = rfd::FileDialog::new()
                    .add_filter("FASTA", &["fasta", "fa"])
                    .pick_file();
                if let Some(path) = path {
                    self.fasta_path = Some(path.display().to_string());
                    self.preview = match std::fs::read_to_string(&path) {
                        Ok(text) => fasta::preview(&text),
                        Err(e) => fasta::Preview { records: Vec::new(), errors: vec![e.to_string()] },
                    };
                }
            }
            match &self.fasta_path {
                Some(p) => ui.label(RichText::new(std::path::Path::new(p).file_name().map(|n| n.to_string_lossy()).unwrap_or_default()).small().color(ACCENT)),
                None => ui.label(RichText::new("Choose a .fasta or .fa sequence file").small().color(MUTED)),
            };

            if self.fasta_path.is_some() || !self.preview.records.is_empty() {
                ui.label(RichText::new(self.preview.summary()).small());
                let limit = match self.selected_model { Model::ProtT5Xl => prott5::MAX_RESIDUES, _ => esm::MAX_RESIDUES };
                let truncated = self.preview.records.iter().filter(|r| r.seq.len() > limit).count();
                if truncated > 0 { ui.colored_label(Color32::DARK_RED, format!("{truncated} sequences exceed {limit} aa; inference uses only their first {limit} residues.")); }
                egui::CollapsingHeader::new("Import preview & validation").show(ui, |ui| {
                    egui::ScrollArea::vertical().id_source("import_preview").max_height(180.0).show(ui, |ui| {
                        for error in &self.preview.errors { ui.colored_label(Color32::DARK_RED,error); }
                        for r in &self.preview.records { ui.label(format!("{} · {} aa",r.id,r.seq.len())); }
                    });
                });
            }
            if !self.embeddings.is_empty() && self.preview.valid() && ui.add_enabled(!self.downloading,egui::Button::new("Attach FASTA to loaded proteins")).clicked() { self.attach_fasta(); }
            ui.add_space(4.0);
            let run = egui::Button::new(RichText::new(if self.downloading { "Processing sequences…" } else { "Run embedding  →" }).strong().color(Color32::WHITE)).fill(ACCENT);
            if ui
                .add_enabled(!self.downloading && self.preview.valid(), run.min_size(egui::vec2(ui.available_width(), 42.0)))
                .clicked()
            {
                if self.selected_model.size_mb() > 1000 && !self.confirm_big {
                    self.confirm_big = true;
                } else {
                    self.confirm_big = false;
                    self.start_download(ctx);
                }
            }
            if self.confirm_big && self.selected_model.size_mb() > 1000 {
                ui.label(
                    RichText::new(format!(
                        "This is a {:.1} GB download. Click Run again to confirm, or pick a smaller model.",
                        self.selected_model.size_mb() as f64 / 1000.0
                    ))
                        .small()
                        .color(Color32::from_rgb(250, 204, 92)),
                );
            }
            if let Some(free) = models::available_ram_mb() {
                let need = self.selected_model.ram_mb();
                if need > free {
                    ui.label(
                        RichText::new(format!(
                            "Needs ~{:.1} GB RAM, {:.1} GB free. Close other apps or pick a smaller model.",
                            need as f64 / 1000.0,
                            free as f64 / 1000.0
                        ))
                        .small()
                        .color(Color32::from_rgb(244, 114, 94)),
                    );
                }
            }
            if self.downloading {
                ui.add(egui::ProgressBar::new(self.progress).animate(true));
            }
            ui.label(RichText::new(&self.status).small().color(MUTED));

            ui.add_space(6.0);
            ui.separator();

            section(ui, "Explore proteins");
            ui.add(
                egui::TextEdit::singleline(&mut self.search)
                    .hint_text("Search protein ID or family…")
                    .desired_width(f32::INFINITY),
            );

            if self.records.is_empty() {
            section(ui, "Protein families");
            let mut counts = [0usize; 8];
            for p in &self.proteins {
                counts[p.cluster] += 1;
            }
            for (i, name) in CLUSTER_NAMES.iter().enumerate() {
                ui.horizontal(|ui| {
                    ui.checkbox(&mut self.cluster_on[i], "");
                    let (rect, _) = ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
                    ui.painter().circle_filled(rect.center(), 5.0, PALETTE[i]);
                    ui.label(*name);
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(RichText::new(counts[i].to_string()).small().color(MUTED));
                    });
                });
            }
            ui.horizontal(|ui| {
                if ui.small_button("All").clicked() {
                    self.cluster_on = [true; 8];
                }
                if ui.small_button("None").clicked() {
                    self.cluster_on = [false; 8];
                }
            });
            } else { ui.label(RichText::new(self.active_annotation.as_ref().map(|c|format!("Colored by {c}")).unwrap_or_else(||"Imported proteins · families unassigned".into())).small().color(MUTED)); }
            });
        });
    }

    fn right_panel(&mut self, ctx: &egui::Context) {
        let mut pick = None;
        let colors = self.point_colors();
        egui::SidePanel::right("right").exact_width(280.0)
            .frame(egui::Frame::none().fill(PANEL).inner_margin(18.0)).show(ctx, |ui| {
            egui::ScrollArea::vertical().id_source("inspector_scroll").show(ui, |ui| {
            ui.add_space(8.0);
            section(ui, "Protein inspector");
            ui.add_space(8.0);
            let Some(p) = self.selected.and_then(|i| self.proteins.get(i)) else {
                ui.add_space(60.0);
                ui.vertical_centered(|ui| {
                    ui.label(RichText::new("◎").size(48.0).color(ACCENT));
                    ui.add_space(12.0);
                    ui.label(RichText::new("Every point, a protein").size(18.0).strong());
                    ui.label(RichText::new("Select a point on the atlas to explore its properties and closest neighbors.").color(MUTED));
                });
                return;
            };
            ui.label(RichText::new(&p.id).size(22.0).strong());
            ui.horizontal(|ui| {
                let (rect, _) = ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
                ui.painter().circle_filled(rect.center(), 5.0, self.selected.map(|i|colors[i]).unwrap_or(ACCENT));
                ui.label(if self.active_annotation.is_some() { self.selected.map(|i|self.annotation_label(i)).filter(|s|!s.is_empty()).unwrap_or("Unannotated") } else if self.records.is_empty() { CLUSTER_NAMES[p.cluster] } else { "Imported protein · family unassigned" });
            });
            ui.add_space(6.0);
            egui::Grid::new("details").num_columns(2).show(ui, |ui| {
                ui.label(RichText::new("Length").color(MUTED));
                ui.label(if p.seq_len == 0 { "Unavailable".into() } else { format!("{} aa",p.seq_len) });
                ui.end_row();
                ui.label(RichText::new(if self.projection.is_some() { "PC1" } else { "Map x" }).color(MUTED));
                ui.label(format!("{:.3}", p.pos[0]));
                ui.end_row();
                ui.label(RichText::new(if self.projection.is_some() { "PC2" } else { "Map y" }).color(MUTED));
                ui.label(format!("{:.3}", p.pos[1]));
                ui.end_row();
                if let Some(point) = self.selected.and_then(|i| self.projection.as_ref()?.points.get(i)) {
                    ui.label(RichText::new("PC3").color(MUTED)); ui.label(format!("{:.3}",point[2])); ui.end_row();
                }
            });

            if let Some(values) = self.annotations.rows.get(&p.id) {
                section(ui,"Annotations");
                for (column,value) in self.annotations.columns.iter().zip(values) { ui.label(format!("{column}: {}",if value.is_empty() { "Unannotated" } else { value })); }
            }
            if let Some(r) = self.selected.and_then(|i| self.records.get(i)).filter(|r| !r.seq.is_empty()) {
                section(ui, "Sequence details");
                ui.label(&r.description);
                ui.horizontal(|ui| {
                    if ui.small_button("Copy sequence").clicked() { ui.output_mut(|o| o.copied_text = r.seq.clone()); }
                    if ui.small_button("Copy FASTA").clicked() { ui.output_mut(|o| o.copied_text = format!(">{} {}\n{}\n",r.id,r.description,r.seq)); }
                });
                egui::ScrollArea::vertical().id_source("sequence").max_height(150.0).show(ui, |ui| {
                    for chunk in r.seq.as_bytes().chunks(24) { ui.monospace(String::from_utf8_lossy(chunk)); }
                });
            } else { ui.label(RichText::new("Sequence unavailable · attach matching FASTA to inspect it").small().color(MUTED)); }
            ui.add_space(4.0);
            ui.separator();
            section(ui, &format!("Nearest {} neighbors", self.neighbors.len()));
            ui.label(RichText::new(if self.embeddings.is_empty() { "Demo map distance · lower is closer" } else { "Cosine distance · lower is closer" }).small().color(MUTED));
            egui::ScrollArea::vertical().id_source("neighbors").max_height(350.0).show(ui, |ui| {
                for &(j, d) in &self.neighbors {
                    let n = &self.proteins[j];
                    ui.horizontal(|ui| {
                        let (rect, _) = ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
                        ui.painter().circle_filled(rect.center(), 5.0, colors[j]);
                        if ui.link(&n.id).clicked() {
                            pick = Some(j);
                        }

                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.label(RichText::new(format!("{d:.4}")).small().color(MUTED));
                        });
                    });
                }
            });
            });
        });
        if pick.is_some() {
            self.select(pick);
        }
    }

    fn reset_view(&mut self) {
        self.view = projection::View {
            three_d: self.view.three_d,
            ..Default::default()
        };
        self.reset_map = true;
    }

    fn point_3d(&self, i: usize) -> [f64; 3] {
        self.projection
            .as_ref()
            .and_then(|p| p.points.get(i))
            .copied()
            .unwrap_or([self.proteins[i].pos[0], self.proteins[i].pos[1], 0.0])
    }

    fn plot_3d(&mut self, ui: &mut egui::Ui, visible: &[bool]) {
        self.reset_map = false;
        let (rect, response) = ui.allocate_exact_size(
            egui::vec2(
                ui.available_width(),
                (ui.available_height() - 36.0).max(100.0),
            ),
            egui::Sense::click_and_drag(),
        );
        let delta = response.drag_delta();
        if response.dragged_by(egui::PointerButton::Secondary)
            || (response.dragged() && ui.input(|i| i.modifiers.shift))
        {
            self.view.pan[0] += delta.x / rect.width().max(1.0);
            self.view.pan[1] += delta.y / rect.height().max(1.0);
        } else if response.dragged_by(egui::PointerButton::Primary) {
            self.view.yaw += delta.x * 0.008;
            self.view.pitch = (self.view.pitch + delta.y * 0.008).clamp(-1.55, 1.55);
        }
        if response.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            self.view.zoom = (self.view.zoom * (scroll * 0.002).exp()).clamp(0.1, 20.0);
        }
        if response.double_clicked() {
            self.reset_view();
        }
        let painter = ui.painter_at(rect);
        let colors = self.point_colors();
        let radius = (0..self.proteins.len())
            .map(|i| self.point_3d(i).iter().map(|x| x * x).sum::<f64>().sqrt())
            .fold(0.0_f64, f64::max)
            .max(1e-9);
        let scale = rect.width().min(rect.height()) * 0.42 * self.view.zoom / radius as f32;
        let center = rect.center()
            + egui::vec2(
                self.view.pan[0] * rect.width(),
                self.view.pan[1] * rect.height(),
            );
        let positions: Vec<_> = (0..self.proteins.len())
            .map(|i| {
                let p = self.view.rotate(self.point_3d(i));
                (
                    center + egui::vec2(p[0] as f32 * scale, -p[1] as f32 * scale),
                    p[2],
                )
            })
            .collect();
        let mut order: Vec<_> = (0..self.proteins.len()).filter(|i| visible[*i]).collect();
        order.sort_by(|a, b| positions[*a].1.total_cmp(&positions[*b].1));
        if let Some(i) = self.selected.filter(|i| visible[*i]) {
            for &(j, _) in &self.neighbors {
                if visible[j] {
                    painter.line_segment(
                        [positions[i].0, positions[j].0],
                        egui::Stroke::new(1.0, Color32::from_black_alpha(50)),
                    );
                }
            }
        }
        let pointer = response.hover_pos();
        let mut hovered = None;
        let mut best = 100.0;
        for i in order {
            let (point, depth) = positions[i];
            if !rect.contains(point) {
                continue;
            }
            let radius = (3.0 + depth as f32 / radius as f32).clamp(2.0, 4.0);
            painter.circle_filled(point, radius, colors[i]);
            if self.selected == Some(i) || self.neighbors.iter().any(|(j, _)| *j == i) {
                painter.circle_stroke(point, radius + 3.0, egui::Stroke::new(1.5, ACCENT));
            }
            if let Some(pointer) = pointer {
                let d = pointer.distance_sq(point);
                if d <= best {
                    best = d;
                    hovered = Some(i);
                }
            }
        }
        // A fixed camera triad gives the rotated PCA axes a readable orientation.
        let origin = rect.left_bottom() + egui::vec2(46.0, -48.0);
        for (axis, name, color) in [
            (
                [1., 0., 0.],
                if self.projection.is_some() {
                    "PC1"
                } else {
                    "X"
                },
                PALETTE[0],
            ),
            (
                [0., 1., 0.],
                if self.projection.is_some() {
                    "PC2"
                } else {
                    "Y"
                },
                PALETTE[1],
            ),
            (
                [0., 0., 1.],
                if self.projection.is_some() {
                    "PC3"
                } else {
                    "Z"
                },
                ACCENT,
            ),
        ] {
            let p = self.view.rotate(axis);
            let tip = origin + egui::vec2(p[0] as f32 * 30.0, -p[1] as f32 * 30.0);
            painter.line_segment([origin, tip], egui::Stroke::new(2.0, color));
            painter.text(
                tip,
                egui::Align2::CENTER_CENTER,
                name,
                egui::FontId::proportional(10.0),
                color,
            );
        }
        if let Some(i) = hovered {
            response
                .clone()
                .on_hover_cursor(egui::CursorIcon::PointingHand)
                .on_hover_text(format!(
                    "{} · {} aa",
                    self.proteins[i].id, self.proteins[i].seq_len
                ));
        }
        if response.clicked() {
            self.select(hovered);
        }
        ui.label(
            RichText::new(
                "Drag to rotate · Scroll to zoom · Shift/right-drag to pan · Click to inspect",
            )
            .size(11.0)
            .color(MUTED),
        );
    }

    fn plot(&mut self, ctx: &egui::Context, visible: &[bool]) {
        egui::CentralPanel::default()
            .frame(egui::Frame::none().fill(BG))
            .show(ctx, |ui| {
                ui.add_space(14.0);
                ui.horizontal(|ui| {
                    ui.add_space(16.0);
                    ui.vertical(|ui| {
                        ui.label(RichText::new("Protein atlas").size(26.0).strong());
                        ui.label(RichText::new("Explore the relationships hidden in protein space.").color(MUTED));
                    });
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.add_space(16.0);
                        if ui.button("Reset view").clicked() { self.reset_view(); }
                        ui.label(RichText::new(if self.records.is_empty() { "DEMO DATA" } else { "EMBEDDED DATA" }).size(11.0).color(ACCENT).strong());
                    });
                });
                ui.add_space(12.0);
                ui.horizontal(|ui| {
                    ui.add_space(16.0);
                    ui.label(RichText::new(format!("{} proteins", self.proteins.len())).strong());
                    ui.separator();
                    ui.label(RichText::new(if self.records.is_empty() { "8 protein families" } else { self.active_annotation.as_deref().unwrap_or("Families unassigned") }).color(MUTED));
                    ui.separator();
                    ui.label(RichText::new(if self.records.is_empty() { "Synthetic 2D projection" } else { "PCA projection" }).color(MUTED));
                });
                ui.horizontal(|ui| {
                    ui.add_space(16.0);
                    if ui.selectable_label(!self.view.three_d,"2D").clicked() { self.view.three_d = false; self.reset_map = true; }
                    if ui.selectable_label(self.view.three_d,"3D").clicked() { self.view.three_d = true; }
                    if let Some(p) = &self.projection {
                        let axes = if self.view.three_d {3} else {2};
                        ui.label(RichText::new(format!("PCA · {:.1}% variance captured",p.variance[..axes].iter().sum::<f64>()*100.0)).small().color(MUTED));
                        if p.variance[2] < 1e-10 && self.view.three_d { ui.label(RichText::new("No third-component variance").small().color(MUTED)); }
                    } else { ui.label(RichText::new("Planar demo data · import sequences for PCA").small().color(MUTED)); }
                });
                ui.add_space(12.0);
                if !self.records.is_empty() && self.projection.is_none() {
                    if self.downloading { ui.spinner(); ui.label("Computing PCA from saved embeddings…"); }
                    else { ui.label(&self.status); if ui.button("Retry PCA").clicked() { self.start_projection(); ctx.request_repaint(); } }
                    return;
                }
                if self.view.three_d { self.plot_3d(ui,visible); return; }
                let selected = self.selected;
                let proteins = &self.proteins;
                let neighbors = &self.neighbors;
                let colors = self.point_colors();

                let plot = Plot::new("embedding");
                let plot = if self.reset_map { plot.reset() } else { plot };
                let out = plot
                    .show_background(false)
                    .height((ui.available_height() - 30.0).max(100.0))
                    .data_aspect(1.0)
                    .show_axes([false, false])
                    .show_grid(false)
                    .show_x(false)
                    .show_y(false)
                    .label_formatter(|_, _| String::new())
                    .allow_boxed_zoom(true)
                    .show(ui, |plot_ui| {
                        let mut groups: BTreeMap<[u8;4],Vec<[f64;2]>> = BTreeMap::new();
                        for (i,p) in proteins.iter().enumerate().filter(|(i,_)|visible[*i]) { groups.entry(colors[i].to_array()).or_default().push(p.pos); }
                        for (color,points) in groups { plot_ui.points(Points::new(points).color(Color32::from_rgba_unmultiplied(color[0],color[1],color[2],color[3])).radius(2.2)); }

                        // Nearest visible point within ~12 screen pixels of the cursor.
                        let mut hovered = None;
                        if let Some(ptr) = plot_ui.pointer_coordinate() {
                            let width_px = plot_ui.response().rect.width().max(1.0) as f64;
                            let reach = plot_ui.plot_bounds().width() / width_px * 12.0;
                            let mut best = reach * reach;
                            for (i, p) in proteins.iter().enumerate() {
                                if !visible[i] {
                                    continue;
                                }
                                let d = (p.pos[0] - ptr.x).powi(2) + (p.pos[1] - ptr.y).powi(2);
                                if d < best {
                                    best = d;
                                    hovered = Some(i);
                                }
                            }
                        }

                        if let Some(i) = selected.filter(|i| visible[*i]) {
                            for &(j, _) in neighbors {
                                if !visible[j] { continue; }
                                plot_ui.line(
                                    Line::new(vec![proteins[i].pos, proteins[j].pos])
                                        .color(Color32::from_black_alpha(50))
                                        .width(1.0),
                                );
                            }
                            let ring: Vec<[f64; 2]> = neighbors.iter().filter(|(j,_)| visible[*j]).map(|&(j, _)| proteins[j].pos).collect();
                            plot_ui.points(
                                Points::new(ring).color(ACCENT).radius(4.0).filled(false),
                            );
                            plot_ui.points(
                                Points::new(vec![proteins[i].pos])
                                    .color(ACCENT)
                                    .radius(7.0)
                                    .filled(false),
                            );
                        }
                        if let Some(i) = hovered {
                            plot_ui.points(
                                Points::new(vec![proteins[i].pos])
                                    .color(colors[i])
                                    .radius(5.0),
                            );
                        }
                        let clicked = plot_ui.response().clicked();
                        (hovered, clicked)
                    });

                self.reset_map = false;
                ui.label(RichText::new(if self.records.is_empty() { "Demo atlas · Import a FASTA file to analyze your own sequences." } else { "PC1 × PC2 · PCA of normalized embeddings · Neighbors use full embedding cosine distance." }).size(11.0).color(MUTED));
                let (hovered, clicked) = out.inner;
                if let Some(i) = hovered {
                    ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
                    let p = &self.proteins[i];
                    egui::show_tooltip_at_pointer(ui.ctx(), egui::Id::new("tip"), |ui| {
                        ui.label(RichText::new(&p.id).strong());
                        ui.label(RichText::new(if self.active_annotation.is_some() { self.annotation_label(i) } else if self.records.is_empty() { CLUSTER_NAMES[p.cluster] } else { "Unassigned family" }).color(colors[i]));
                        ui.label(RichText::new(if p.seq_len == 0 { "Sequence unavailable".into() } else { format!("{} aa",p.seq_len) }).small().color(MUTED));
                    });
                }
                if clicked {
                    self.select(hovered);
                }
            });
    }
}

impl eframe::App for ProtRustApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.drain_messages();
        if self.downloading {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }
        let visible = self.visibility();
        let shown = visible.iter().filter(|v| **v).count();

        self.top_bar(ctx, shown);
        self.left_panel(ctx);
        self.right_panel(ctx);
        let visible = self.visibility();
        self.plot(ctx, &visible);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.first().is_some_and(|a| a == "--evaluate") {
        return benchmark::cli(&args);
    }
    if !args.is_empty() {
        return Err("Usage: prot-rust [--evaluate EMBEDDINGS.csv ANNOTATIONS.csv LABEL_COLUMN K REPORT.json]".into());
    }
    let runtime = Arc::new(Runtime::new()?);
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1440.0, 920.0])
            .with_min_inner_size([1100.0, 780.0]),
        ..Default::default()
    };
    eframe::run_native(
        "ProtRust",
        options,
        Box::new(|cc| Box::new(ProtRustApp::new(cc, runtime))),
    )?;
    Ok(())
}

#[cfg(test)]
mod ui_tests {
    use super::*;

    #[test]
    fn imported_selection_and_panels_render_without_a_window() {
        let runtime = Arc::new(Runtime::new().expect("test runtime"));
        let (tx, rx) = channel();
        let records =
            fasta::preview(">a first protein\nACD\n>b second protein\nMK\n>c third protein\nGH")
                .records;
        let mut app = ProtRustApp {
            fasta_path: None,
            selected_model: Model::Esm2_8M,
            status: "Test workspace".into(),
            progress: 0.0,
            runtime,
            tx,
            rx,
            downloading: false,
            confirm_big: false,
            proteins: records
                .iter()
                .enumerate()
                .map(|(i, r)| Protein {
                    id: r.id.clone(),
                    seq_len: r.seq.len(),
                    cluster: 0,
                    pos: [i as f64, 0.],
                })
                .collect(),
            cluster_on: [false; 8],
            search: String::new(),
            selected: None,
            neighbors: Vec::new(),
            records,
            embeddings: vec![vec![1., 0.], vec![-1., 0.], vec![5., 0.]],
            reset_map: false,
            preview: fasta::Preview::default(),
            embedding_model: Some(Model::Esm2_8M),
            projection: None,
            view: projection::View::default(),
            external_source: None,
            annotations: imports::Annotations::default(),
            active_annotation: None,
            hidden_labels: BTreeSet::new(),
        };
        app.select(Some(0));
        assert_eq!(app.neighbors[0].0, 2); // Cosine neighbor differs from the closest map point.
        assert!(app.visibility().iter().all(|v| *v)); // Demo family filters do not hide imports.
        app.search = "second".into();
        assert_eq!(app.visibility(), vec![false, true, false]);
        app.search.clear();
        let ctx = egui::Context::default();
        apply_theme(&ctx);
        app.projection = Some(projection::pca(&app.embeddings).expect("test PCA"));
        for (p, point) in app
            .proteins
            .iter_mut()
            .zip(&app.projection.as_ref().expect("projection").points)
        {
            p.pos = [point[0], point[1]];
        }
        let original_neighbors = app.neighbors.clone();
        for (width, three_d) in [
            (1100.0, false),
            (1440.0, false),
            (1100.0, true),
            (1440.0, true),
        ] {
            app.view.three_d = three_d;
            app.view.yaw = 1.23;
            app.reset_map = true;
            let input = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(width, 780.0),
                )),
                ..Default::default()
            };
            let _ = ctx.run(input, |ctx| {
                app.top_bar(ctx, 3);
                app.left_panel(ctx);
                app.right_panel(ctx);
                app.plot(ctx, &app.visibility());
            });
            assert_eq!(app.neighbors, original_neighbors);
            assert_eq!(app.view.yaw, 1.23);
        }
        app.reset_view();
        assert_eq!(app.view.yaw, projection::View::default().yaw);
        app.load_example();
        assert_eq!(app.records.len(), 8);
        assert!(app.records.iter().all(|r| r.seq.is_empty()));
        assert!(app.embedding_model.is_none());
        assert!(app.external_source.is_some());
        let colors = app.point_colors();
        assert_ne!(colors[0], colors[4]);
        app.hidden_labels.insert("synthetic_B".into());
        assert_eq!(app.visibility().iter().filter(|v| **v).count(), 4);
        app.select(Some(0));
        let project = app.project();
        project.validate().expect("external project validation");
        let bytes = serde_json::to_vec(&project).expect("save external project");
        let reopened: workspace::Project =
            serde_json::from_slice(&bytes).expect("reopen external project");
        reopened.validate().expect("external project round trip");
        assert_eq!(reopened.active_annotation.as_deref(), Some("family"));
        assert!(reopened.hidden_labels.contains("synthetic_B"));
        for three_d in [false, true] {
            app.view.three_d = three_d;
            let _ = ctx.run(egui::RawInput::default(), |ctx| {
                app.top_bar(ctx, 4);
                app.left_panel(ctx);
                app.right_panel(ctx);
                app.plot(ctx, &app.visibility());
            });
        }
        app.preview = fasta::preview(include_str!("../examples/sequences.fasta"));
        app.attach_fasta();
        assert!(app.records.iter().all(|r| !r.seq.is_empty()));
        app.project().validate().expect("attached project");
    }
}
