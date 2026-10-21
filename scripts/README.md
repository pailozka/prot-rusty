# HDF5 → ProtRust CSV

The converter runs on a CPU, reads one protein vector at a time, and does not load protein language models. Requires Python 3.10 or newer.

## Setup

From the repository root:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r scripts/requirements-hdf5.txt
```

On Windows, use `.venv\Scripts\python.exe` instead of `.venv/bin/python`.

## One dataset per protein

For a file containing `/protein_A` and `/protein_B`, each with shape `(dimensions,)`:

```sh
.venv/bin/python scripts/hdf5_to_csv.py embeddings.h5 embeddings.csv
```

For datasets inside a group such as `/vectors/protein_A`:

```sh
.venv/bin/python scripts/hdf5_to_csv.py embeddings.h5 embeddings.csv --group /vectors
```

Dataset names become IDs exactly as stored, sorted lexicographically. The selected group must contain only one-dimensional vector datasets. Groups and unrelated metadata are not silently skipped. The converter cannot recover original IDs if the producer changed them before writing HDF5.

## Matrix plus IDs

For `/embeddings` with shape `(proteins, dimensions)` and `/ids` containing the corresponding string IDs:

```sh
.venv/bin/python scripts/hdf5_to_csv.py embeddings.h5 embeddings.csv \
  --embeddings /embeddings --ids /ids
```

Rows retain the matrix/ID order. Both fixed-length and variable-length UTF-8 string datasets are supported. Numeric IDs are not inferred, and the matrix is not automatically transposed.

If you do not know the layout, inspect paths, shapes, and dtypes without reading the embedding arrays:

```sh
.venv/bin/python scripts/hdf5_to_csv.py embeddings.h5 --inspect
```

## Output and validation

The output is directly accepted by **Import embeddings CSV** in ProtRust:

```csv
id,embedding_0,embedding_1
protein_A,0.25,-0.5
protein_B,0.75,0.125
```

- Requires unique, nonempty IDs without whitespace, equal vector dimensions, and real numeric values.
- Casts vectors to float32 to match ProtRust, rejecting non-finite or all-zero results, including overflow/complete underflow during conversion. Float64 inputs may lose precision during this explicit cast; values are written to preserve the resulting float32 vectors on import.
- Does not normalize vectors, pool per-residue embeddings, fabricate sequences, or infer model identity. Use existing **per-protein** embeddings. Attach a matching FASTA in the app if sequences are needed.
- Writes `embeddings.csv.provenance.json`, recording layout, row/dimension counts, SHA-256 hashes, dependency versions, and optional user-supplied model/revision information:

```sh
.venv/bin/python scripts/hdf5_to_csv.py embeddings.h5 embeddings.csv \
  --model Rostlab/prot_t5_xl_half_uniref50-enc --revision MODEL_COMMIT
```

Model/revision arguments are recorded as supplied, not verified against vector contents. Keep the provenance file with the CSV; the app does not automatically ingest the sidecar.

Existing CSV/provenance files are never overwritten. Invalid data leaves no CSV or provenance output. Completed temporary files are published with hard links on the destination filesystem; the CSV is published last. The output directory must exist and support hard links. Both files and temporary staging share that directory. The input HDF5 file is opened read-only.

## Tests

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_hdf5_to_csv.py' -v
```

Tests create tiny HDF5 files locally and cover both layouts, ID matching and quoting, float32 values, invalid inputs, output preservation, provenance hashes, and cleanup. No network or model weights are needed after installing dependencies.
