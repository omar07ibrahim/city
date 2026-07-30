from __future__ import annotations

import csv
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from typing import Any, ClassVar, cast
from unittest.mock import patch

import urbanlens.audit as audit_module
from urbanlens.audit import (
    DEFAULT_CONTRACT,
    MAX_CSV_PHYSICAL_LINE_CHARACTERS,
    MAX_DATASET_BYTES,
    MAX_DATASET_COLUMNS,
    MAX_DATASET_ROWS,
    MAX_MANIFEST_BYTES,
    DatasetInputError,
    audit_dataset,
    canonical_json_bytes,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATASET = REPOSITORY_ROOT / "train.csv"
CHECKED_MANIFEST = REPOSITORY_ROOT / "artifacts" / "data_quality" / "train.quality.json"


class SnapshotAuditTests(unittest.TestCase):
    manifest: ClassVar[dict[str, Any]]
    payload: ClassVar[bytes]

    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = audit_dataset(DATASET)
        cls.payload = canonical_json_bytes(cls.manifest)

    def test_real_snapshot_passes_exact_contract(self) -> None:
        self.assertEqual(self.manifest["quality"]["status"], "pass")
        self.assertEqual(
            self.manifest["dataset"],
            {
                "byte_size": 4_734_682,
                "data_rows": 44_691,
                "encoding": "utf-8-sig",
                "name": "train.csv",
                "sha256": DEFAULT_CONTRACT.sha256,
            },
        )
        self.assertEqual(
            self.manifest["schema"]["columns"], list(DEFAULT_CONTRACT.columns)
        )
        self.assertEqual(self.manifest["quality"]["summary"]["failed"], 0)

    def test_real_snapshot_metrics_are_evidence_backed(self) -> None:
        metrics = self.manifest["quality"]["metrics"]
        self.assertEqual(metrics["missingness"]["population"]["count"], 307)
        self.assertEqual(metrics["missingness"]["admin_name"]["count"], 316)
        self.assertEqual(metrics["missingness"]["capital"]["count"], 33_553)
        self.assertEqual(
            metrics["duplicates"]["city_country_admin"]["duplicate_row_excess_count"],
            301,
        )
        self.assertEqual(
            metrics["duplicates"]["coordinate_tokens"]["duplicate_row_excess_count"],
            136,
        )
        self.assertEqual(
            metrics["duplicates"]["coordinate_e7"]["duplicate_group_count"],
            106,
        )
        self.assertEqual(
            metrics["duplicates"]["coordinate_e7"]["duplicate_row_excess_count"],
            137,
        )
        self.assertEqual(
            metrics["validity"]["coordinate_e7_unrepresentable_count"],
            0,
        )
        self.assertEqual(
            metrics["duplicates"]["exact_rows_excluding_id"][
                "duplicate_row_excess_count"
            ],
            0,
        )
        self.assertEqual(
            metrics["mapping_consistency"][
                "iso2_to_country_label_variant_excess_count"
            ],
            1,
        )

    def test_canonical_bytes_are_stable_and_self_describing(self) -> None:
        self.assertEqual(self.payload, canonical_json_bytes(self.manifest))
        self.assertTrue(self.payload.endswith(b"\n"))
        self.assertFalse(self.payload.endswith(b"\n\n"))
        decoded = json.loads(self.payload)
        self.assertEqual(decoded, self.manifest)
        self.assertEqual(
            decoded["manifest"]["canonicalization"],
            "UTF-8, sorted object keys, two-space indent, LF, final newline",
        )
        self.assertEqual(decoded["manifest"]["schema_version"], 3)

    def test_manifest_contains_no_absolute_repository_path(self) -> None:
        self.assertNotIn(
            os.fsencode(str(REPOSITORY_ROOT.resolve())),
            self.payload,
        )

    def test_manifest_contains_no_city_country_or_admin_source_labels(self) -> None:
        with DATASET.open(encoding="utf-8-sig", newline="") as stream:
            source_labels = {
                row[column].strip()
                for row in csv.DictReader(stream)
                for column in ("city", "city_ascii", "country", "admin_name")
                if row[column].strip()
            }

        manifest_strings: set[str] = set()

        def collect_strings(value: Any) -> None:
            if isinstance(value, str):
                manifest_strings.add(value)
            elif isinstance(value, dict):
                for nested in value.values():
                    collect_strings(nested)
            elif isinstance(value, list):
                for nested in value:
                    collect_strings(nested)

        collect_strings(self.manifest)
        self.assertTrue(source_labels.isdisjoint(manifest_strings))

    def test_summary_and_check_ids_are_internally_consistent(self) -> None:
        checks = self.manifest["quality"]["contract_checks"]
        ids = [check["id"] for check in checks]
        self.assertEqual(len(ids), len(set(ids)))
        passed = sum(check["status"] == "pass" for check in checks)
        summary = self.manifest["quality"]["summary"]
        self.assertEqual(summary["passed"], passed)
        self.assertEqual(summary["failed"], len(checks) - passed)
        self.assertEqual(summary["total"], len(checks))

    def test_non_row_observation_has_no_false_row_rate(self) -> None:
        observations = {
            observation["id"]: observation
            for observation in self.manifest["quality"]["observations"]
        }
        country_variants = observations["country.exact_label_variants_per_iso2"]
        self.assertEqual(country_variants["count"], 1)
        self.assertNotIn("row_rate_ppm", country_variants)

    def test_checked_manifest_is_exact_canonical_output(self) -> None:
        self.assertEqual(CHECKED_MANIFEST.read_bytes(), self.payload)


class FailureDetectionTests(unittest.TestCase):
    header: ClassVar[list[str]]
    real_rows: ClassVar[list[list[str]]]

    @classmethod
    def setUpClass(cls) -> None:
        with DATASET.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            cls.header = next(reader)
            cls.real_rows = [next(reader) for _ in range(3)]

    def _write_rows(
        self,
        directory: Path,
        rows: list[list[str]],
        *,
        header: list[str] | None = None,
        name: str = "train.csv",
    ) -> Path:
        path = directory / name
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(header if header is not None else self.header)
            writer.writerows(rows)
        return path

    def _check_status(self, manifest: dict[str, Any], check_id: str) -> str:
        checks = {
            check["id"]: check["status"]
            for check in manifest["quality"]["contract_checks"]
        }
        return cast(str, checks[check_id])

    def test_duplicate_real_id_is_detected(self) -> None:
        rows = [row.copy() for row in self.real_rows]
        id_index = self.header.index("id")
        rows[1][id_index] = rows[0][id_index]
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_rows(Path(directory), rows)
            manifest = audit_dataset(path)
        self.assertEqual(
            self._check_status(manifest, "id.duplicate_row_excess"), "fail"
        )
        self.assertEqual(
            manifest["quality"]["metrics"]["duplicates"]["id"][
                "duplicate_row_excess_count"
            ],
            1,
        )

    def test_invalid_real_coordinate_is_detected(self) -> None:
        rows = [row.copy() for row in self.real_rows]
        rows[0][self.header.index("lat")] = "91"
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_rows(Path(directory), rows)
            manifest = audit_dataset(path)
        self.assertEqual(
            self._check_status(manifest, "coordinates.invalid_counts"), "fail"
        )
        self.assertEqual(
            manifest["quality"]["metrics"]["validity"]["invalid_latitude_count"],
            1,
        )

    def test_numeric_coordinate_metric_merges_equivalent_decimal_tokens(
        self,
    ) -> None:
        rows = [row.copy() for row in self.real_rows[:2]]
        latitude_index = self.header.index("lat")
        longitude_index = self.header.index("lng")
        rows[0][latitude_index], rows[0][longitude_index] = "28.5700", "77.3200"
        rows[1][latitude_index], rows[1][longitude_index] = "28.57", "77.32"

        with tempfile.TemporaryDirectory() as directory:
            path = self._write_rows(Path(directory), rows)
            metrics = audit_dataset(path)["quality"]["metrics"]

        self.assertEqual(
            metrics["duplicates"]["coordinate_tokens"]["duplicate_row_excess_count"],
            0,
        )
        self.assertEqual(
            metrics["duplicates"]["coordinate_e7"]["duplicate_row_excess_count"],
            1,
        )

    def test_numeric_coordinate_metric_canonicalizes_dateline_and_poles(
        self,
    ) -> None:
        rows = [row.copy() for row in self.real_rows]
        rows.append(self.real_rows[0].copy())
        latitude_index = self.header.index("lat")
        longitude_index = self.header.index("lng")
        id_index = self.header.index("id")
        coordinates = (
            ("90", "180"),
            ("90", "-23"),
            ("0", "-180"),
            ("-0.0000", "180"),
        )
        for index, (row, coordinate) in enumerate(
            zip(rows, coordinates, strict=True),
            start=1,
        ):
            row[latitude_index], row[longitude_index] = coordinate
            row[id_index] = f"9{index:09d}"

        with tempfile.TemporaryDirectory() as directory:
            path = self._write_rows(Path(directory), rows)
            duplicates = audit_dataset(path)["quality"]["metrics"]["duplicates"]

        self.assertEqual(
            duplicates["coordinate_tokens"]["duplicate_group_count"],
            0,
        )
        self.assertEqual(
            duplicates["coordinate_e7"]["duplicate_group_count"],
            2,
        )
        self.assertEqual(
            duplicates["coordinate_e7"]["duplicate_row_excess_count"],
            2,
        )

    def test_noncanonical_coordinate_tokens_fail_spatial_readiness(self) -> None:
        invalid_tokens = (
            "0.00000001",
            "1e0",
            "1_0",
            "+1",
            "00.1",
            " 1",
            "\N{ARABIC-INDIC DIGIT ONE}",
            "0." + "0" * 31,
        )
        for invalid_token in invalid_tokens:
            with self.subTest(token=invalid_token):
                rows = [row.copy() for row in self.real_rows[:1]]
                rows[0][self.header.index("lat")] = invalid_token

                with tempfile.TemporaryDirectory() as directory:
                    path = self._write_rows(Path(directory), rows)
                    manifest = audit_dataset(path)

                self.assertEqual(
                    self._check_status(
                        manifest,
                        "coordinates.e7_unrepresentable",
                    ),
                    "fail",
                )
                self.assertEqual(
                    manifest["quality"]["metrics"]["validity"][
                        "coordinate_e7_unrepresentable_count"
                    ],
                    1,
                )

    def test_missing_column_and_short_row_are_detected(self) -> None:
        shortened_header = self.header[:-1]
        rows = [self.real_rows[0][:-1], self.real_rows[1]]
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_rows(Path(directory), rows, header=shortened_header)
            manifest = audit_dataset(path)
        self.assertEqual(self._check_status(manifest, "schema.columns"), "fail")
        self.assertEqual(
            self._check_status(manifest, "schema.row_width_mismatches"), "fail"
        )
        self.assertEqual(manifest["schema"]["row_width_mismatch_count"], 1)

    def test_malformed_csv_is_an_input_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.csv"
            path.write_bytes(b'city,city_ascii\n"unterminated')
            with self.assertRaisesRegex(DatasetInputError, "malformed CSV"):
                audit_dataset(path)

    def test_symlink_dataset_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            link = Path(directory) / "train.csv"
            link.symlink_to(DATASET)
            with self.assertRaisesRegex(DatasetInputError, "not a symlink"):
                audit_dataset(link)

    def test_oversized_dataset_is_rejected_before_csv_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.csv"
            with path.open("wb") as stream:
                stream.seek(MAX_DATASET_BYTES)
                stream.write(b"x")
            with self.assertRaisesRegex(DatasetInputError, "safety limit"):
                audit_dataset(path)

    def test_same_size_rewrite_with_restored_mtime_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_rows(Path(directory), self.real_rows[:1])
            original_capture = audit_module._capture_open_file

            def rewrite_after_capture(stream: Any) -> tuple[bytes, str]:
                payload, digest = original_capture(stream)
                before = path.stat()
                replacement = bytearray(path.read_bytes())
                replacement[-2] = replacement[-2] ^ 1
                path.write_bytes(replacement)
                os.utime(
                    path,
                    ns=(before.st_atime_ns, before.st_mtime_ns),
                )
                self.assertEqual(path.stat().st_mtime_ns, before.st_mtime_ns)
                self.assertEqual(path.stat().st_size, before.st_size)
                return payload, digest

            with (
                patch.object(
                    audit_module,
                    "_capture_open_file",
                    side_effect=rewrite_after_capture,
                ),
                self.assertRaisesRegex(DatasetInputError, "changed while"),
            ):
                audit_dataset(path)

    def test_dataset_row_limit_is_enforced_before_profile_expansion(self) -> None:
        self.assertGreater(MAX_DATASET_ROWS, DEFAULT_CONTRACT.data_rows)
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_rows(Path(directory), self.real_rows)
            with (
                patch.object(audit_module, "MAX_DATASET_ROWS", 2),
                self.assertRaisesRegex(DatasetInputError, "row safety limit"),
            ):
                audit_dataset(path)

    def test_dataset_column_limit_is_enforced(self) -> None:
        self.assertGreater(MAX_DATASET_COLUMNS, len(DEFAULT_CONTRACT.columns))
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_rows(Path(directory), self.real_rows[:1])
            with (
                patch.object(
                    audit_module,
                    "MAX_DATASET_COLUMNS",
                    len(DEFAULT_CONTRACT.columns) - 1,
                ),
                self.assertRaisesRegex(DatasetInputError, "column safety limit"),
            ):
                audit_dataset(path)

    def test_physical_line_limit_is_enforced_before_csv_expansion(self) -> None:
        self.assertGreater(MAX_CSV_PHYSICAL_LINE_CHARACTERS, len(",".join(self.header)))
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_rows(Path(directory), self.real_rows[:1])
            with (
                patch.object(audit_module, "MAX_CSV_PHYSICAL_LINE_CHARACTERS", 20),
                self.assertRaisesRegex(DatasetInputError, "physical line 1"),
            ):
                audit_dataset(path)

    def test_multiline_csv_record_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.csv"
            path.write_bytes(b'city,city_ascii\n"two\nlines",two\n')
            with self.assertRaisesRegex(DatasetInputError, "multi-line record"):
                audit_dataset(path)

    def test_failed_width_diagnostics_use_profiled_row_denominator(self) -> None:
        rows = [row.copy() for row in self.real_rows[:2]]
        rows[0][self.header.index("population")] = ""
        rows[1] = rows[1][:-1]
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_rows(Path(directory), rows)
            manifest = audit_dataset(path)
        metrics = manifest["quality"]["metrics"]
        self.assertEqual(metrics["profiled_rows"], 1)
        self.assertEqual(metrics["missingness"]["population"]["count"], 1)
        self.assertEqual(
            metrics["missingness"]["population"]["row_rate_ppm"],
            1_000_000,
        )


class CommandLineTests(unittest.TestCase):
    def _run(self, *arguments: str) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            [sys.executable, "-m", "urbanlens.audit", *arguments],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            check=False,
        )

    def test_stdout_is_only_canonical_json(self) -> None:
        completed = self._run("train.csv")
        self.assertEqual(completed.returncode, 0, completed.stderr.decode())
        self.assertEqual(completed.stdout, CHECKED_MANIFEST.read_bytes())
        self.assertEqual(completed.stderr, b"")

    def test_output_and_check_modes_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "quality.json"
            generated = self._run("train.csv", "--output", str(output))
            self.assertEqual(generated.returncode, 0, generated.stderr.decode())
            self.assertEqual(output.read_bytes(), CHECKED_MANIFEST.read_bytes())
            self.assertEqual(stat_mode(output), 0o644)

            checked = self._run("train.csv", "--check-manifest", str(output))
            self.assertEqual(checked.returncode, 0, checked.stderr.decode())
            self.assertIn(b"manifest PASS", checked.stderr)

    def test_stale_manifest_returns_contract_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stale = Path(directory) / "quality.json"
            stale.write_text("{}\n", encoding="utf-8")
            completed = self._run("train.csv", "--check-manifest", str(stale))
        self.assertEqual(completed.returncode, 1)
        self.assertIn(b"stale or non-canonical", completed.stderr)

    def test_semantically_equal_noncanonical_manifest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            noncanonical = Path(directory) / "quality.json"
            content = json.loads(CHECKED_MANIFEST.read_bytes())
            noncanonical.write_text(
                json.dumps(content, sort_keys=False),
                encoding="utf-8",
            )
            completed = self._run("train.csv", "--check-manifest", str(noncanonical))
        self.assertEqual(completed.returncode, 1)
        self.assertIn(b"stale or non-canonical", completed.stderr)

    def test_checked_manifest_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            link = Path(directory) / "quality.json"
            link.symlink_to(CHECKED_MANIFEST)
            completed = self._run("train.csv", "--check-manifest", str(link))
        self.assertEqual(completed.returncode, 3)
        self.assertIn(b"not a symlink", completed.stderr)

    def test_checked_manifest_fifo_is_rejected_without_opening(self) -> None:
        if not hasattr(os, "mkfifo"):
            self.skipTest("FIFO creation is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            fifo = Path(directory) / "quality.pipe"
            os.mkfifo(fifo)
            completed = self._run("train.csv", "--check-manifest", str(fifo))
        self.assertEqual(completed.returncode, 3)
        self.assertIn(b"must be a regular file", completed.stderr)

    def test_checked_manifest_device_is_rejected(self) -> None:
        device = Path("/dev/null")
        if not device.exists():
            self.skipTest("/dev/null is unavailable")
        completed = self._run("train.csv", "--check-manifest", str(device))
        self.assertEqual(completed.returncode, 3)
        self.assertIn(b"must be a regular file", completed.stderr)

    def test_oversized_checked_manifest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            oversized = Path(directory) / "quality.json"
            with oversized.open("wb") as stream:
                stream.seek(MAX_MANIFEST_BYTES)
                stream.write(b"x")
            completed = self._run("train.csv", "--check-manifest", str(oversized))
        self.assertEqual(completed.returncode, 3)
        self.assertIn(b"manifest exceeds", completed.stderr)

    def test_missing_dataset_returns_input_error(self) -> None:
        completed = self._run("does-not-exist.csv")
        self.assertEqual(completed.returncode, 3)
        self.assertIn(b"input error:", completed.stderr)
        self.assertEqual(completed.stdout, b"")

    def test_invalid_usage_returns_argparse_exit_two(self) -> None:
        completed = self._run("--output", "one.json", "--check-manifest", "two.json")
        self.assertEqual(completed.returncode, 2)
        self.assertIn(b"not allowed with argument", completed.stderr)

    def test_missing_output_directory_returns_output_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "missing" / "quality.json"
            completed = self._run("train.csv", "--output", str(output))
        self.assertEqual(completed.returncode, 4)
        self.assertIn(b"cannot anchor output directory", completed.stderr)

    def test_symlink_output_parent_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real_parent = root / "real"
            real_parent.mkdir()
            linked_parent = root / "linked"
            linked_parent.symlink_to(real_parent, target_is_directory=True)
            completed = self._run(
                "train.csv",
                "--output",
                str(linked_parent / "quality.json"),
            )
        self.assertEqual(completed.returncode, 4)
        self.assertIn(b"cannot anchor output directory", completed.stderr)

    def test_output_parent_swap_stays_on_anchored_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset.csv"
            dataset.write_bytes(b"test dataset")
            output_parent = root / "output"
            output_parent.mkdir()
            anchored_parent = root / "anchored-output"
            redirected_parent = root / "redirected"
            redirected_parent.mkdir()
            redirected_target = redirected_parent / "quality.json"
            redirected_target.write_bytes(b"must remain unchanged")
            output = output_parent / "quality.json"

            def swap_parent(
                _dataset: Path,
                *,
                contract: audit_module.SnapshotContract = DEFAULT_CONTRACT,
            ) -> tuple[dict[str, Any], os.stat_result]:
                del contract
                output_parent.rename(anchored_parent)
                output_parent.symlink_to(
                    redirected_parent,
                    target_is_directory=True,
                )
                return {"quality": {"status": "pass"}}, dataset.stat()

            with (
                patch.object(
                    audit_module,
                    "_audit_dataset_with_identity",
                    side_effect=swap_parent,
                ),
                redirect_stderr(io.StringIO()),
            ):
                return_code = audit_module.main([str(dataset), "--output", str(output)])

            self.assertEqual(return_code, 0)
            self.assertEqual(
                json.loads((anchored_parent / "quality.json").read_bytes()),
                {"quality": {"status": "pass"}},
            )
            self.assertEqual(
                redirected_target.read_bytes(),
                b"must remain unchanged",
            )

    def test_cli_refuses_to_overwrite_dataset(self) -> None:
        completed = self._run("train.csv", "--output", "train.csv")
        self.assertEqual(completed.returncode, 4)
        self.assertIn(b"refusing to overwrite", completed.stderr)
        self.assertEqual(
            DATASET.stat().st_size,
            DEFAULT_CONTRACT.byte_size,
        )

    def test_cli_refuses_to_overwrite_dataset_through_hard_link(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            alias = Path(directory) / "dataset-alias.csv"
            os.link(DATASET, alias)
            completed = self._run(
                "train.csv",
                "--output",
                str(alias),
            )
        self.assertEqual(completed.returncode, 4)
        self.assertIn(b"refusing to overwrite", completed.stderr)
        self.assertEqual(DATASET.stat().st_size, DEFAULT_CONTRACT.byte_size)

    def test_mutated_real_rows_emit_diagnostics_and_exit_one(self) -> None:
        with DATASET.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            header = next(reader)
            rows = [next(reader) for _ in range(2)]
        rows[1][header.index("id")] = rows[0][header.index("id")]

        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "train.csv"
            output = Path(directory) / "quality.json"
            with dataset.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream, lineterminator="\n")
                writer.writerow(header)
                writer.writerows(rows)
            completed = self._run(str(dataset), "--output", str(output))
            manifest = json.loads(output.read_bytes())

        self.assertEqual(completed.returncode, 1)
        self.assertEqual(manifest["quality"]["status"], "fail")
        self.assertEqual(
            manifest["quality"]["metrics"]["duplicates"]["id"][
                "duplicate_row_excess_count"
            ],
            1,
        )


def stat_mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


if __name__ == "__main__":
    unittest.main()
