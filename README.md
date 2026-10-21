# ProtRust

Native Rust workspace for protein embeddings with ESM-2 and ProtT5.

```sh
cargo run --release
```

- **Import FASTA**: previews sequence count, total residues, minimum/maximum and mean lengths. Expand the preview for IDs and validation errors. Empty sequences, missing/duplicate IDs, invalid residues, and content before headers block inference. Lowercase residues are normalized. The UI warns when sequences exceed the model's 1,022-residue limit.
- **Run embedding**: downloads the selected model into the Hugging Face cache, then embeds the validated input snapshot. Imported proteins replace the demo points. Positions use PCA of unit-normalized embeddings; protein families remain unassigned until annotation is implemented. PCA runs in a background task.
- **2D / 3D**: 2D is the default (PC1 × PC2). Switch to 3D to explore PC1/PC2/PC3: drag to rotate, scroll to zoom, Shift-drag or right-drag to pan, and click to inspect. Reset view restores the camera. The variance indicator reports how much centered embedding variance the displayed components capture; tiny datasets may have fewer than three nonzero components. Demo data remains planar. PCA uses deterministic iterative approximation; nearby projected points need not be nearest neighbors in the full vectors.
- **Inspect**: click a point to see its description and sequence, copy the sequence or FASTA, and explore its ten nearest neighbors. Real-data neighbors use cosine distance on full embeddings (0 = same direction, 2 = opposite). Demo neighbors use 2D map distance.
- **Save/Open project**: `.protjson` files include sequences, embeddings, map positions, cached PCA coordinates, 2D/3D mode and the 3D camera, the embedding model and configured model, current selection, search, family filters, and valid pending input. Reopening does not require model downloads or inference. Original FASTA files are not required. Older projects without PCA coordinates compute them from saved embeddings on opening.
- **Export**: the toolbar offers embeddings CSV, the selected protein's neighbor list CSV, and a standalone SVG map image. CSV exports include all embeddings or the selected protein's computed neighbors, irrespective of map filters. The SVG fits all currently visible points and highlights the selection; it preserves the current 3D orientation and aspect ratio, but fits visible points rather than retaining zoom/pan or neighbor connection lines.

Projects are JSON and include complete sequences and embeddings. Save after inference finishes; project controls are disabled while a job is running.

## Verification

```sh
cargo test --offline -- --skip recovers_masked_residues
cargo build --offline
```

The ESM masked-residue reference test runs inference using cached/downloaded weights and can be slow. The ProtT5 reference test is ignored by default.

## Work without running a protein model

Choose **Import embeddings CSV** in the left panel, followed by **Import annotations CSV**. Import, cosine neighbors, annotation coloring/filtering, PCA, 2D/3D exploration, project saving, and reports all work without loading model weights. Imports run parsing and PCA in a background task; failed imports leave the current workspace intact.

**Load small synthetic example** opens eight artificial vectors and two label groups immediately. These are interface/test fixtures, not PLM outputs or biological validation. The files are also in [`examples`](examples/README.md).

Embedding CSV requires an `id` column and contiguous dimensions named `embedding_0`, `embedding_1`, etc. Dimension columns may be reordered. Optional columns are `description` and `sequence`. Additional metadata belongs in the annotations CSV. For example:

```csv
id,embedding_0,embedding_1
protein_a,0.2,0.9
protein_b,0.3,0.8
```

IDs must be unique, nonempty, and contain no whitespace. Vectors must be finite, nonzero, and equally sized. Quoted commas, escaped quotes, multiline descriptions, UTF-8 BOM, and CRLF files are supported. Invalid files report the offending record where possible. Missing sequences are explicitly unavailable; importing vectors does not infer their model of origin.

Annotation CSV requires `id` plus one or more named columns:

```csv
id,family,organism
protein_b,kinase,human
protein_a,protease,yeast
```

Rows join by exact ID, regardless of order. Unknown or duplicate IDs reject the import. Partial annotation tables and blank values are allowed; missing values appear as **Unannotated**. Each import replaces the annotation table. Select a column to color the map, toggle categories to filter, and inspect all metadata on a selected protein. Search also matches annotation values. Projects preserve the table, selected column, filters, and external source filename.

To attach sequences afterward, import a FASTA into the preview and choose **Attach FASTA to loaded proteins**. It must contain exactly the same IDs; its order may differ. Attaching FASTA updates sequence metadata and leaves vectors and neighbors unchanged. CSV embedding exports can be imported again.

### Lightweight exploratory evaluation

Select an annotation column and choose **Export → Label retrieval report (JSON)**. This evaluates all labeled proteins, independently of map filters, using up to ten neighbors. It reports leave-one-out top-1 accuracy, precision@k, macro precision@k, per-label scores, and the expected precision of a uniformly random neighbor.

The command-line version accepts a custom k and never starts the GUI or loads a model:

```sh
cargo run --offline -- --evaluate examples/embeddings.csv examples/annotations.csv family 3 /tmp/protrust-report.json
```

Reports include their input data and scoring definitions for reproduction. Missing labels are excluded from both queries and candidates; self-matches are excluded; ties follow embedding CSV row order. k is capped at the number of labeled candidates. These are descriptive retrieval metrics, **not independent prediction validation**: there is no homology-controlled split, training-overlap check, or uncertainty estimate. Public biological datasets and reference inference validation remain separate work. Direct HDF5 import is not implemented; use the [HDF5-to-CSV converter](scripts/README.md) to create compatible CSV files without model execution.

## HDF5-to-CSV conversion

Use `scripts/hdf5_to_csv.py` for precomputed `.h5`/`.hdf5` embeddings. It supports one vector dataset per protein or an explicit matrix plus string IDs, validates float32-compatible vectors, and saves a provenance sidecar with SHA-256 hashes. See [setup, supported layouts, and usage](scripts/README.md).
