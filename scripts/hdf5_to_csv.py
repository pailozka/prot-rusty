#!/usr/bin/env python3
"""Convert protein-level HDF5 vectors to ProtRust CSV, one vector at a time.

Supports datasets named by protein ID inside a group, or an explicit N×D
matrix paired with a length-N string ID dataset. No model execution, pooling,
normalization, or fabricated sequences. Requires Python >= 3.10, h5py, numpy.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

VERSION = "1.0"


def dependencies():
    try:
        import h5py
        import numpy as np
    except ImportError as error:
        raise ValueError(
            "Missing converter dependencies. Install with: "
            "python -m pip install -r scripts/requirements-hdf5.txt"
        ) from error
    return h5py, np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset(file, name, h5py):
    obj = file[name]
    if not isinstance(obj, h5py.Dataset):
        raise ValueError(f"{name!r} must be an HDF5 dataset")
    return obj


def vector_rows(file, group: str, embeddings: str | None, ids: str | None):
    """Yield IDs and arrays without loading a whole embedding matrix."""
    h5py, _ = dependencies()
    if embeddings is not None:
        matrix = dataset(file, embeddings, h5py)
        identifiers = dataset(file, ids, h5py)
        if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError("Embedding matrix must have nonempty shape (proteins, dimensions)")
        if identifiers.ndim != 1 or identifiers.shape[0] != matrix.shape[0]:
            raise ValueError("ID dataset must be one-dimensional and match the matrix row count")
        if h5py.check_string_dtype(identifiers.dtype) is None:
            raise ValueError("ID dataset must contain strings; numeric IDs are not inferred")
        strings = identifiers.asstr(encoding="utf-8", errors="strict")
        for row in range(matrix.shape[0]):
            yield strings[row], matrix[row]
    else:
        root = file[group]
        if not isinstance(root, h5py.Group):
            raise ValueError(f"{group!r} must be a group containing one vector dataset per protein")
        if not len(root):
            raise ValueError("Selected HDF5 group is empty")
        # Sorting makes conversion order independent of HDF5 creation order.
        for name in sorted(root.keys()):
            obj = root[name]
            if not isinstance(obj, h5py.Dataset):
                raise ValueError(
                    f"{obj.name!r} is a group, not a protein vector. "
                    "Use --inspect to find the vectors, then --group PATH or --embeddings PATH --ids PATH."
                )
            if obj.ndim != 1:
                raise ValueError(
                    f"{obj.name!r} has shape {obj.shape}; expected a one-dimensional protein vector. "
                    "Per-residue arrays are not implicitly pooled. For an N×D matrix, use --embeddings and --ids."
                )
            yield name, obj[()]


def inspect_file(source: Path) -> None:
    h5py, _ = dependencies()
    with h5py.File(source, "r") as file:
        def show(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"/{name}\tshape={obj.shape}\tdtype={obj.dtype}")
            else:
                print(f"/{name}/\tgroup")
        file.visititems(show)


def convert(source: Path, output: Path, *, group="/", embeddings=None, ids=None,
            model=None, revision=None) -> dict:
    h5py, np = dependencies()
    source, output = Path(source), Path(output)
    manifest = output.with_name(output.name + ".provenance.json")
    if (embeddings is None) != (ids is None):
        raise ValueError("--embeddings and --ids must be supplied together")
    if embeddings is not None and group != "/":
        raise ValueError("Choose --group OR --embeddings/--ids, not both")
    if revision and not model:
        raise ValueError("--revision requires --model")
    if output.resolve() == source.resolve() or manifest.resolve() == source.resolve():
        raise ValueError("Output paths must differ from the input HDF5 file")
    for path in [output, manifest]:
        if path.exists() or path.is_symlink():
            raise ValueError(f"Output already exists: {path}. Choose a new output filename.")
    original_stat = source.stat()
    source_hash = sha256(source)
    seen = set()
    count, dimensions = 0, None
    temporary = []
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=output.parent,
                                         prefix=".protrust-csv-", delete=False) as stream:
            csv_temp = Path(stream.name)
            temporary.append(csv_temp)
            writer = csv.writer(stream)
            with h5py.File(source, "r") as file:
                for identifier, raw in vector_rows(file, group, embeddings, ids):
                    if not isinstance(identifier, str) or not identifier or any(c.isspace() for c in identifier) or "\x00" in identifier:
                        raise ValueError(f"Invalid protein ID {identifier!r}: IDs must be nonempty and contain no whitespace")
                    if identifier in seen:
                        raise ValueError(f"Duplicate protein ID: {identifier!r}")
                    seen.add(identifier)
                    raw = np.asarray(raw)
                    if raw.ndim != 1 or raw.size == 0 or raw.dtype.kind not in "iuf":
                        raise ValueError(f"{identifier!r}: expected a nonempty real numeric vector, got shape {raw.shape}, dtype {raw.dtype}")
                    # ProtRust imports f32. Validate after conversion as overflow or
                    # underflow can invalidate otherwise finite/nonzero f64 values.
                    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                        vector = raw.astype(np.float32)
                    if not np.isfinite(vector).all() or not np.any(vector != 0):
                        raise ValueError(f"{identifier!r}: vector must be finite and nonzero after float32 conversion")
                    if dimensions is None:
                        dimensions = len(vector)
                        writer.writerow(["id", *[f"embedding_{i}" for i in range(dimensions)]])
                    elif len(vector) != dimensions:
                        raise ValueError(f"{identifier!r}: expected {dimensions} dimensions, got {len(vector)}")
                    writer.writerow([identifier, *[repr(float(value)) for value in vector]])
                    count += 1
            if count == 0:
                raise ValueError("No protein vectors found")
            stream.flush()
            os.fsync(stream.fileno())
        final_stat = source.stat()
        if (original_stat.st_size, original_stat.st_mtime_ns) != (final_stat.st_size, final_stat.st_mtime_ns):
            raise ValueError("Input file changed during conversion; retry with an unchanged copy")
        metadata = {
            "schema_version": 1,
            "converter": "ProtRust hdf5_to_csv",
            "converter_version": VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source": {"path": str(source.resolve()), "sha256": source_hash},
            "output": {"path": str(output.resolve()), "sha256": sha256(csv_temp)},
            "layout": {"mode": "matrix" if embeddings is not None else "per_protein",
                       "group": None if embeddings else group, "embeddings": embeddings, "ids": ids},
            "proteins": count, "dimensions": dimensions,
            "dtype": "float32",
            "preprocessing": "Cast to float32 only; no pooling or normalization; sequences unavailable",
            "model": {"name": model, "revision": revision, "provenance": "user supplied, not inferred or verified"},
            "dependencies": {"python": sys.version.split()[0], "h5py": h5py.__version__, "numpy": np.__version__},
        }
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                         prefix=".protrust-provenance-", delete=False) as stream:
            json_temp = Path(stream.name)
            temporary.append(json_temp)
            json.dump(metadata, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Hard-linking completed files atomically refuses existing destinations,
        # including a file created after the checks above. Both temps share the
        # destination filesystem. Publish CSV last so it always has provenance.
        os.link(json_temp, manifest)
        try:
            os.link(csv_temp, output)
        except BaseException:
            manifest.unlink()
            raise
        return metadata
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="Source .h5/.hdf5 file")
    parser.add_argument("output", nargs="?", type=Path, help="Destination CSV (must not already exist)")
    parser.add_argument("--inspect", action="store_true", help="List dataset paths/shapes without reading their arrays")
    parser.add_argument("--group", default="/", help="Group containing one vector dataset per protein (default: /)")
    parser.add_argument("--embeddings", help="Path of an N×D embedding matrix")
    parser.add_argument("--ids", help="Path of a length-N string ID dataset, matching matrix row order")
    parser.add_argument("--model", help="Optional model name, recorded as user-supplied provenance")
    parser.add_argument("--revision", help="Optional model revision; requires --model")
    args = parser.parse_args(argv)
    if not args.inspect and args.output is None:
        parser.error("output CSV is required unless --inspect is used")
    if args.inspect and (args.output is not None or args.embeddings or args.ids or args.group != "/" or args.model or args.revision):
        parser.error("--inspect only accepts the input path")
    try:
        if args.inspect:
            inspect_file(args.input)
        else:
            metadata = convert(args.input, args.output, group=args.group, embeddings=args.embeddings,
                               ids=args.ids, model=args.model, revision=args.revision)
            print(f"Converted {metadata['proteins']} proteins × {metadata['dimensions']} dimensions → {args.output}")
            print(f"Provenance → {args.output}.provenance.json")
    except (ValueError, OSError, KeyError, UnicodeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
