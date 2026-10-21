"""Real HDF5 fixtures; no protein models or network access."""
import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
from scripts import hdf5_to_csv as converter


class ConverterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "input.h5"
        self.output = self.root / "output.csv"

    def fixture(self, values):
        with h5py.File(self.source, "w") as file:
            for name, value in values.items():
                file[name] = value

    def read_output(self):
        with self.output.open(newline="", encoding="utf-8") as stream:
            return list(csv.reader(stream))

    def assert_no_outputs(self):
        self.assertFalse(self.output.exists())
        self.assertFalse(self.output.with_name("output.csv.provenance.json").exists())
        self.assertEqual(list(self.root.glob(".protrust-*")), [])

    def test_per_protein_values_order_csv_quoting_and_provenance(self):
        self.fixture({"z": np.array([0.1, -0.2], dtype=np.float64),
                      'a,"β"': np.array([3, 4], dtype=np.int32)})
        metadata = converter.convert(self.source, self.output, model="example-model", revision="revision-1")
        rows = self.read_output()
        self.assertEqual(rows[0], ["id", "embedding_0", "embedding_1"])
        self.assertEqual(rows[1][0], 'a,"β"')
        self.assertEqual(rows[2][0], "z")
        np.testing.assert_array_equal(np.asarray(rows[2][1:], dtype=np.float32), np.array([0.1, -0.2], dtype=np.float32))
        self.assertEqual(metadata["proteins"], 2)
        self.assertEqual(metadata["dimensions"], 2)
        self.assertEqual(metadata["source"]["sha256"], hashlib.sha256(self.source.read_bytes()).hexdigest())
        self.assertEqual(metadata["output"]["sha256"], hashlib.sha256(self.output.read_bytes()).hexdigest())
        self.assertEqual(metadata["model"]["name"], "example-model")
        self.assertEqual(json.loads(self.output.with_name("output.csv.provenance.json").read_text()), metadata)
        self.assertEqual(list(self.root.glob(".protrust-*")), [])

    def test_selected_group_ignores_unrelated_metadata(self):
        with h5py.File(self.source, "w") as file:
            file["metadata"] = "description"
            file["vectors/protein_b"] = [2., 3.]
            file["vectors/protein_a"] = [4., 5.]
        converter.convert(self.source, self.output, group="/vectors")
        self.assertEqual([r[0] for r in self.read_output()[1:]], ["protein_a", "protein_b"])

    def test_matrix_preserves_id_order_and_accepts_fixed_and_variable_strings(self):
        for variable in [False, True]:
            with self.subTest(variable=variable), tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / "matrix.h5"
                output = Path(directory) / "matrix.csv"
                with h5py.File(source, "w") as file:
                    file["data/vectors"] = np.array([[1., 2.], [3., 4.]], dtype=np.float16)
                    file.create_dataset("data/ids", data=["b", "a"], dtype=h5py.string_dtype() if variable else "S8")
                converter.convert(source, output, embeddings="/data/vectors", ids="/data/ids")
                with output.open(newline="") as stream:
                    rows = list(csv.reader(stream))
                self.assertEqual([r[0] for r in rows[1:]], ["b", "a"])
                self.assertEqual([float(x) for x in rows[2][1:]], [3., 4.])

    def test_invalid_vectors_never_publish_partial_output(self):
        for bad in [np.array([np.nan, 1.]), np.array([np.inf, 1.]), np.array([0., 0.]),
                    np.array([1e300, 2.]), np.array([1e-300, 0.]), np.array([], dtype=float),
                    np.array([1.+1.j, 2.]), np.array([True, False]), np.zeros((2, 2)), np.array(1.)]:
            with self.subTest(shape=bad.shape, dtype=bad.dtype):
                self.fixture({"a_valid": [1., 2.], "z_invalid": bad})
                with self.assertRaises(ValueError):
                    converter.convert(self.source, self.output)
                self.assert_no_outputs()

    def test_mismatched_dimensions_and_invalid_ids(self):
        for values in [{"a": [1., 2.], "b": [1., 2., 3.]}, {"has space": [1., 2.]}]:
            self.fixture(values)
            with self.assertRaises(ValueError):
                converter.convert(self.source, self.output)
            self.assert_no_outputs()

    def test_matrix_ids_validation(self):
        for ids in [["a", "a"], ["a", ""], ["a"], ["a", "b c"]]:
            with self.subTest(ids=ids):
                with h5py.File(self.source, "w") as file:
                    file["vectors"] = [[1., 2.], [3., 4.]]
                    file.create_dataset("ids", data=ids, dtype=h5py.string_dtype())
                with self.assertRaises(ValueError):
                    converter.convert(self.source, self.output, embeddings="vectors", ids="ids")
                self.assert_no_outputs()
        with h5py.File(self.source, "w") as file:
            file["vectors"] = [[1., 2.], [3., 4.]]
            file["ids"] = [1, 2]
        with self.assertRaisesRegex(ValueError, "strings"):
            converter.convert(self.source, self.output, embeddings="vectors", ids="ids")

    def test_missing_layout_options_empty_files_and_groups(self):
        self.fixture({"a": [1., 2.]})
        for options in [{"embeddings": "a"}, {"ids": "a"}, {"revision": "abc"},
                        {"group": "/other", "embeddings": "a", "ids": "a"}]:
            with self.assertRaises(ValueError):
                converter.convert(self.source, self.output, **options)
        with h5py.File(self.source, "w") as file:
            file.create_group("nested")
        with self.assertRaisesRegex(ValueError, "group, not a protein vector"):
            converter.convert(self.source, self.output)
        with self.assertRaisesRegex(ValueError, "empty"):
            converter.convert(self.source, self.output, group="nested")
        self.assert_no_outputs()

    def test_existing_outputs_and_input_are_preserved(self):
        self.fixture({"a": [1., 2.]})
        original = self.source.read_bytes()
        with self.assertRaises(ValueError):
            converter.convert(self.source, self.source)
        self.assertEqual(self.source.read_bytes(), original)
        self.output.write_text("keep this")
        with self.assertRaises(ValueError):
            converter.convert(self.source, self.output)
        self.assertEqual(self.output.read_text(), "keep this")
        self.output.unlink()
        manifest = self.output.with_name("output.csv.provenance.json")
        manifest.write_text("keep metadata")
        with self.assertRaises(ValueError):
            converter.convert(self.source, self.output)
        self.assertEqual(manifest.read_text(), "keep metadata")

    def test_publish_failure_rolls_back_manifest_and_cleans_temps(self):
        self.fixture({"a": [1., 2.]})
        real_link = converter.os.link
        def fail_csv(source, target):
            if Path(target) == self.output:
                raise OSError("simulated publication failure")
            return real_link(source, target)
        with patch.object(converter.os, "link", side_effect=fail_csv):
            with self.assertRaises(OSError):
                converter.convert(self.source, self.output)
        self.assert_no_outputs()

    def test_inspection_and_cli_errors(self):
        self.fixture({"vectors/a": [1., 2.]})
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(converter.main([str(self.source), "--inspect"]), 0)
        self.assertIn("/vectors/a", output.getvalue())
        self.assertIn("shape=(2,)", output.getvalue())
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(converter.main([str(self.root / "missing.h5"), str(self.output)]), 1)
        self.assert_no_outputs()


if __name__ == "__main__":
    unittest.main()
