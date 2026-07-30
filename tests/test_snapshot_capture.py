from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any
from unittest.mock import patch

import urbanlens.audit as audit_module
from urbanlens.audit import (
    DEFAULT_CONTRACT,
    DatasetCapture,
    audit_capture,
    canonical_json_bytes,
    capture_dataset,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATASET = REPOSITORY_ROOT / "train.csv"
CHECKED_MANIFEST = REPOSITORY_ROOT / "artifacts" / "data_quality" / "train.quality.json"


class DatasetCaptureTests(unittest.TestCase):
    def test_capture_and_audit_reproduce_the_checked_manifest_exactly(self) -> None:
        capture = capture_dataset(DATASET)
        manifest = audit_capture(capture)

        self.assertIsInstance(capture, DatasetCapture)
        self.assertEqual(capture.source_name, "train.csv")
        self.assertEqual(capture.sha256, DEFAULT_CONTRACT.sha256)
        self.assertEqual(capture.byte_size, DEFAULT_CONTRACT.byte_size)
        self.assertEqual(capture.columns, DEFAULT_CONTRACT.columns)
        self.assertEqual(len(capture.rows), DEFAULT_CONTRACT.data_rows)
        self.assertEqual(canonical_json_bytes(manifest), CHECKED_MANIFEST.read_bytes())

    def test_capture_is_deeply_immutable_and_hides_held_identity_from_repr(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            private_parent = Path(directory) / "private-parent"
            private_parent.mkdir()
            dataset = private_parent / "sample.csv"
            dataset.write_bytes(b"city,id\nAlpha,1\n")
            capture = capture_dataset(dataset)

        with self.assertRaises(FrozenInstanceError):
            capture.source_name = "changed.csv"  # type: ignore[misc]
        with self.assertRaises(TypeError):
            capture.columns[0] = "changed"  # type: ignore[index]
        with self.assertRaises(TypeError):
            capture.rows[0][0] = "changed"  # type: ignore[index]

        rendered = repr(capture)
        self.assertNotIn(str(private_parent), rendered)
        self.assertNotIn("identity=", rendered)
        self.assertEqual(capture.source_name, "sample.csv")

    def test_audit_capture_does_not_reopen_or_recapture_the_source(self) -> None:
        original_capture = audit_module._capture_open_file
        capture_calls = 0

        def count_capture(stream: Any) -> tuple[bytes, str]:
            nonlocal capture_calls
            capture_calls += 1
            return original_capture(stream)

        with (
            patch.object(
                audit_module,
                "_capture_open_file",
                side_effect=count_capture,
            ),
            patch.object(
                audit_module,
                "_open_dataset",
                wraps=audit_module._open_dataset,
            ) as open_dataset,
        ):
            capture = capture_dataset(DATASET)
            first = audit_capture(capture)
            second = audit_capture(capture)

        self.assertEqual(capture_calls, 1)
        self.assertEqual(open_dataset.call_count, 1)
        self.assertEqual(first, second)

    def test_capture_exposes_only_the_source_basename(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            nested = Path(directory) / "sensitive" / "nested"
            nested.mkdir(parents=True)
            dataset = nested / "source.csv"
            dataset.write_bytes(b"city,id\nAlpha,1\n")

            capture = capture_dataset(dataset)
            manifest = audit_capture(capture)

        self.assertEqual(capture.source_name, "source.csv")
        self.assertNotIn(str(nested), repr(capture))
        self.assertNotIn(str(nested), canonical_json_bytes(manifest).decode("utf-8"))
        self.assertIsInstance(capture.identity, os.stat_result)


if __name__ == "__main__":
    unittest.main()
