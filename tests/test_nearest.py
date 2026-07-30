from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import unittest
from collections.abc import Callable
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar
from unittest.mock import patch

import urbanlens.nearest as nearest_module
from urbanlens.geo import GeoPoint
from urbanlens.nearest import (
    EXIT_ARGUMENT_ERROR,
    EXIT_INPUT_ERROR,
    EXIT_OUTPUT_ERROR,
    EXIT_SUCCESS,
    EXIT_VERIFICATION_ERROR,
    MAX_RECEIPT_BYTES,
    ReceiptOutputError,
    build_receipt,
    canonical_receipt_bytes,
    main,
)
from urbanlens.snapshot import SourceSnapshotError
from urbanlens.spatial import (
    QueryDiagnostics,
    QueryResult,
    SpatialIndex,
    SpatialRecord,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATASET = REPOSITORY_ROOT / "train.csv"


def record(
    source_id: str,
    latitude: str,
    longitude: str,
    *,
    city_ascii: str,
    country: str,
    iso2: str,
) -> SpatialRecord:
    return SpatialRecord(
        source_id=source_id,
        city_ascii=city_ascii,
        country=country,
        iso2=iso2,
        point=GeoPoint.from_decimal(latitude, longitude),
    )


class _BrokenStdout(io.BytesIO):
    def write(self, value: Any) -> int:
        del value
        raise BrokenPipeError


class NearestCliTests(unittest.TestCase):
    index: ClassVar[SpatialIndex]
    snapshot: ClassVar[SimpleNamespace]

    @classmethod
    def setUpClass(cls) -> None:
        cls.index = SpatialIndex(
            (
                record(
                    "100",
                    "0",
                    "0",
                    city_ascii="Alpha",
                    country="Example",
                    iso2="EX",
                ),
                record(
                    "200",
                    "10",
                    "10",
                    city_ascii="Beta",
                    country="Côte d'Ivoire",
                    iso2="CI",
                ),
            )
        )
        cls.snapshot = SimpleNamespace(
            source_sha256="a" * 64,
            byte_size=321,
            data_rows=2,
            audit_schema="urbanlens.data-quality-manifest",
            audit_schema_version=3,
            records=cls.index._records,
        )

    def run_cli(
        self,
        arguments: list[str],
        *,
        loader_side_effect: Exception | None = None,
        stdout: io.BytesIO | None = None,
    ) -> tuple[int, str, str]:
        output = stdout if stdout is not None else io.BytesIO()
        errors = io.StringIO()
        loader_kwargs: dict[str, Any]
        if loader_side_effect is None:
            loader_kwargs = {"return_value": self.snapshot}
        else:
            loader_kwargs = {"side_effect": loader_side_effect}
        with (
            patch.object(
                nearest_module,
                "load_city_snapshot",
                **loader_kwargs,
            ),
            patch.object(
                nearest_module,
                "build_spatial_index",
                return_value=self.index,
            ),
            patch.object(nearest_module, "_stdout_buffer", return_value=output),
            redirect_stderr(errors),
        ):
            exit_code = main(arguments)
        return exit_code, output.getvalue().decode("ascii"), errors.getvalue()

    def test_success_is_exact_canonical_bounded_json(self) -> None:
        exit_code, output, errors = self.run_cli(
            [
                "--dataset",
                "train.csv",
                "--lat",
                "0.0000000",
                "--lon",
                "-0.0000",
            ]
        )

        topology = self.index.metadata.topology_sha256
        canonicalization = (
            "UTF-8, sorted object keys, two-space indent, LF, final newline"
        )
        expected = f"""\
{{
  "index": {{
    "algorithm": "balanced-unit-sphere-kd-tree",
    "collision_excess": 0,
    "collision_groups": 0,
    "coordinate_scale": 10000000,
    "depth": 2,
    "earth_radius_meters": 6371008.8,
    "max_location_multiplicity": 1,
    "records": 2,
    "topology_sha256": "{topology}",
    "unique_numeric_locations": 2,
    "version": 1
  }},
  "proof": {{
    "accelerated": {{
      "evaluated": 1,
      "pruned": 1,
      "visited": 2
    }},
    "oracle": {{
      "evaluations": 0,
      "status": "not-requested"
    }}
  }},
  "query": {{
    "k": 1,
    "latitude": "0",
    "latitude_e7": 0,
    "longitude": "0",
    "longitude_e7": 0
  }},
  "receipt": {{
    "canonicalization": "{canonicalization}",
    "schema": "urbanlens.nearest-city-receipt",
    "schema_version": 1,
    "tool": "python3 -m urbanlens.nearest",
    "tool_version": "0.3.0"
  }},
  "results": [
    {{
      "city_ascii": "Alpha",
      "country": "Example",
      "distance_m": 0,
      "iso2": "EX",
      "latitude": "0",
      "latitude_e7": 0,
      "longitude": "0",
      "longitude_e7": 0,
      "rank": 1,
      "source_id": "100"
    }}
  ],
  "source": {{
    "audit_schema": "urbanlens.data-quality-manifest",
    "audit_schema_version": 3,
    "byte_size": 321,
    "data_rows": 2,
    "sha256": "{"a" * 64}"
  }}
}}
"""
        self.assertEqual(exit_code, EXIT_SUCCESS)
        self.assertEqual(errors, "")
        self.assertEqual(output, expected)
        self.assertEqual(
            output.encode("ascii"), canonical_receipt_bytes(json.loads(output))
        )
        self.assertLess(len(output.encode("ascii")), MAX_RECEIPT_BYTES)
        self.assertNotIn("-0.0", output)
        self.assertNotIn("train.csv", output)
        self.assertNotIn(str(REPOSITORY_ROOT), output)

    def test_verify_declares_matching_full_scan_and_evaluation_count(self) -> None:
        exit_code, output, errors = self.run_cli(
            [
                "--dataset",
                "train.csv",
                "--lat",
                "0",
                "--lon",
                "0",
                "--k",
                "2",
                "--verify",
            ]
        )

        receipt = json.loads(output)
        self.assertEqual(exit_code, EXIT_SUCCESS)
        self.assertEqual(errors, "")
        self.assertEqual(
            receipt["proof"]["oracle"],
            {"evaluations": 2, "status": "match"},
        )
        self.assertEqual([item["rank"] for item in receipt["results"]], [1, 2])
        self.assertEqual(
            {item["source_id"] for item in receipt["results"]},
            {"100", "200"},
        )
        self.assertIsInstance(receipt["results"][1]["distance_m"], int)
        self.assertNotIn("squared_chord_distance", output)

    def test_display_distance_uses_decimal_half_up_rounding(self) -> None:
        self.assertEqual(nearest_module._distance_meters_half_up(0.49), 0)
        self.assertEqual(nearest_module._distance_meters_half_up(0.5), 1)
        self.assertEqual(nearest_module._distance_meters_half_up(1.5), 2)
        self.assertEqual(nearest_module._distance_meters_half_up(2.49), 2)
        for invalid in (float("nan"), float("inf"), -1.0):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                nearest_module._distance_meters_half_up(invalid)

    def test_argument_failures_are_generic_and_do_not_echo_tokens(self) -> None:
        secret = "SECRET_QUERY_TOKEN"
        private_path = "/private/customer/source.csv"
        cases: tuple[list[str], ...] = (
            [],
            ["--dataset", private_path, "--lat", secret, "--lon", "0"],
            [
                "--dataset",
                private_path,
                "--lat",
                "0",
                "--lon",
                "0",
                "--k",
                "0",
            ],
            [
                "--dataset",
                private_path,
                "--lat",
                "0",
                "--lon",
                "0",
                "--k",
                "101",
            ],
            [
                "--dataset",
                private_path,
                "--lat",
                "0",
                "--lon",
                "0",
                "--k",
                "01",
            ],
            [
                "--dataset",
                private_path,
                "--lat",
                "0",
                "--lon",
                "0",
                "--unknown",
                secret,
            ],
        )
        for arguments in cases:
            with self.subTest(arguments=arguments):
                exit_code, output, errors = self.run_cli(arguments)
                self.assertEqual(
                    exit_code,
                    EXIT_INPUT_ERROR
                    if secret in arguments[:4]
                    else EXIT_ARGUMENT_ERROR,
                )
                self.assertEqual(output, "")
                self.assertNotIn(secret, errors)
                self.assertNotIn(private_path, errors)
                self.assertNotIn("usage:", errors)
                self.assertNotIn("Traceback", errors)

    def test_invalid_coordinate_spellings_are_rejected_without_disclosure(self) -> None:
        invalid_tokens = (
            "NaN",
            "inf",
            "-inf",
            "1e2",
            "1_0",
            " 1",
            "+1",
            "\uff11\uff12",
            "",
            "0\x00",
            "1." + "0" * 40,
            "90.0000001",
        )
        for token in invalid_tokens:
            with self.subTest(token=token):
                exit_code, output, errors = self.run_cli(
                    [
                        "--dataset",
                        "/private/source.csv",
                        f"--lat={token}",
                        "--lon",
                        "0",
                    ]
                )
                self.assertEqual(exit_code, EXIT_INPUT_ERROR)
                self.assertEqual(output, "")
                if token:
                    self.assertNotIn(token, errors)
                self.assertNotIn("/private/source.csv", errors)
                self.assertNotIn("Traceback", errors)

    def test_source_failures_and_unexpected_os_errors_do_not_leak_paths(self) -> None:
        failures = (
            SourceSnapshotError("snapshot.capture_failed"),
            OSError("cannot read /private/tenant-secret.csv"),
        )
        for failure in failures:
            with self.subTest(failure=type(failure).__name__):
                exit_code, output, errors = self.run_cli(
                    [
                        "--dataset",
                        "/private/tenant-secret.csv",
                        "--lat",
                        "1",
                        "--lon",
                        "2",
                    ],
                    loader_side_effect=failure,
                )
                self.assertEqual(exit_code, EXIT_INPUT_ERROR)
                self.assertEqual(output, "")
                self.assertNotIn("tenant-secret", errors)
                self.assertNotIn("Traceback", errors)

    def test_api_rejects_invalid_types_before_source_loading(self) -> None:
        point = GeoPoint.from_decimal("0", "0")
        invalid_calls: tuple[Callable[[], dict[str, object]], ...] = (
            lambda: build_receipt("train.csv", point),  # type: ignore[arg-type]
            lambda: build_receipt(Path("train.csv"), "0,0"),  # type: ignore[arg-type]
            lambda: build_receipt(Path("train.csv"), point, k=True),
            lambda: build_receipt(Path("train.csv"), point, k=0),
            lambda: build_receipt(
                Path("train.csv"),
                point,
                verify=1,  # type: ignore[arg-type]
            ),
        )
        for call in invalid_calls:
            with self.subTest(call=call), self.assertRaises((TypeError, ValueError)):
                call()

    def test_verify_mismatch_fails_closed_without_emitting_receipt(self) -> None:
        false_oracle = QueryResult(
            neighbors=(),
            diagnostics=QueryDiagnostics(evaluated=2, pruned=0, visited=2),
        )
        with patch.object(
            SpatialIndex,
            "query_full_scan",
            return_value=false_oracle,
        ):
            exit_code, output, errors = self.run_cli(
                [
                    "--dataset",
                    "train.csv",
                    "--lat",
                    "0",
                    "--lon",
                    "0",
                    "--verify",
                ]
            )

        self.assertEqual(exit_code, EXIT_VERIFICATION_ERROR)
        self.assertEqual(output, "")
        self.assertEqual(
            errors,
            "error: nearest-neighbor verification failed\n",
        )

    def test_output_cap_is_enforced_before_stdout_write(self) -> None:
        with patch.object(nearest_module, "MAX_RECEIPT_BYTES", 1):
            exit_code, output, errors = self.run_cli(
                [
                    "--dataset",
                    "train.csv",
                    "--lat",
                    "0",
                    "--lon",
                    "0",
                ]
            )
        self.assertEqual(exit_code, EXIT_OUTPUT_ERROR)
        self.assertEqual(output, "")
        self.assertEqual(errors, "error: receipt output failed\n")

        with (
            patch.object(nearest_module, "MAX_RECEIPT_BYTES", 1),
            self.assertRaises(ReceiptOutputError),
        ):
            canonical_receipt_bytes({"large": "value"})

    def test_nonfinite_json_and_broken_stdout_fail_without_traceback(self) -> None:
        with self.assertRaises(ReceiptOutputError):
            canonical_receipt_bytes({"distance": float("nan")})

        broken = _BrokenStdout()
        exit_code, output, errors = self.run_cli(
            [
                "--dataset",
                "train.csv",
                "--lat",
                "0",
                "--lon",
                "0",
            ],
            stdout=broken,
        )
        self.assertEqual(exit_code, EXIT_OUTPUT_ERROR)
        self.assertEqual(output, "")
        self.assertEqual(errors, "error: receipt output failed\n")
        self.assertNotIn("Traceback", errors)

    def test_hash_seed_does_not_change_exact_cli_bytes(self) -> None:
        script = """
from types import SimpleNamespace
import urbanlens.nearest as nearest
from urbanlens.geo import GeoPoint
from urbanlens.spatial import SpatialIndex, SpatialRecord

rows = {
    ("300", "20", "20", "Gamma"),
    ("100", "0", "0", "Alpha"),
    ("200", "10", "10", "Beta"),
}
records = tuple(
    SpatialRecord(
        source_id=source_id,
        city_ascii=city,
        country="Example",
        iso2="EX",
        point=GeoPoint.from_decimal(latitude, longitude),
    )
    for source_id, latitude, longitude, city in rows
)
index = SpatialIndex(records)
snapshot = SimpleNamespace(
    source_sha256="a" * 64,
    byte_size=321,
    data_rows=3,
    audit_schema="urbanlens.data-quality-manifest",
    audit_schema_version=3,
    records=records,
)
nearest.load_city_snapshot = lambda path: snapshot
nearest.build_spatial_index = lambda value: index
raise SystemExit(nearest.main([
    "--dataset", "train.csv", "--lat", "1", "--lon", "1",
    "--k", "3", "--verify",
]))
"""
        outputs: set[bytes] = set()
        for seed in ("1", "2", "77", "random"):
            environment = os.environ.copy()
            environment["PYTHONHASHSEED"] = seed
            completed = subprocess.run(
                [sys.executable, "-c", script],
                cwd=REPOSITORY_ROOT,
                env=environment,
                check=True,
                capture_output=True,
                timeout=20,
            )
            self.assertEqual(completed.stderr, b"")
            outputs.add(completed.stdout)
        self.assertEqual(len(outputs), 1)

    def test_receipt_bytes_ignore_hostile_text_stdout_encoding(self) -> None:
        environment = os.environ.copy()
        environment["PYTHONIOENCODING"] = "utf-16"
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "urbanlens.nearest",
                "--dataset",
                "train.csv",
                "--lat",
                "0",
                "--lon",
                "0",
            ],
            cwd=REPOSITORY_ROOT,
            env=environment,
            check=True,
            capture_output=True,
            timeout=30,
        )

        self.assertEqual(completed.stderr, b"")
        receipt = json.loads(completed.stdout.decode("ascii"))
        self.assertEqual(completed.stdout, canonical_receipt_bytes(receipt))
        self.assertFalse(completed.stdout.startswith((b"\xff\xfe", b"\xfe\xff")))


class RealSnapshotSmokeTests(unittest.TestCase):
    def test_real_snapshot_builds_one_source_bound_receipt(self) -> None:
        receipt = build_receipt(
            DATASET,
            GeoPoint.from_decimal("51.5074", "-0.1278"),
            k=1,
        )
        payload = canonical_receipt_bytes(receipt)

        self.assertEqual(
            receipt["source"],
            {
                "audit_schema": "urbanlens.data-quality-manifest",
                "audit_schema_version": 3,
                "byte_size": 4_734_682,
                "data_rows": 44_691,
                "sha256": (
                    "de941def7faca87c0911abb79c3cbd07672887fd486a9b7bea6c48c12ce0cf18"
                ),
            },
        )
        self.assertEqual(receipt["index"]["records"], 44_691)  # type: ignore[index]
        self.assertEqual(len(receipt["results"]), 1)  # type: ignore[arg-type]
        self.assertLess(len(payload), MAX_RECEIPT_BYTES)
        self.assertNotIn(os.fsencode(str(REPOSITORY_ROOT)), payload)


if __name__ == "__main__":
    unittest.main()
