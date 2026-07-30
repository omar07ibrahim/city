from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

import urbanlens.spatial_evidence as evidence_module
from urbanlens.snapshot import CitySnapshot, build_spatial_index, load_city_snapshot
from urbanlens.spatial import SpatialIndex
from urbanlens.spatial_evidence import (
    EVIDENCE_SCHEMA,
    EVIDENCE_SCHEMA_VERSION,
    EXIT_INPUT_ERROR,
    EXIT_OUTPUT_ERROR,
    EXIT_STALE,
    EXIT_SUCCESS,
    EXPECTED_SOURCE_BYTES,
    EXPECTED_SOURCE_ROWS,
    EXPECTED_SOURCE_SHA256,
    EXPECTED_TOPOLOGY_SHA256,
    MAX_EVIDENCE_BYTES,
    EvidenceArtifactError,
    EvidenceGenerationError,
    artifact_is_current,
    build_evidence,
    canonical_evidence_bytes,
    main,
    write_artifact,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATASET = REPOSITORY_ROOT / "train.csv"
ARTIFACT = REPOSITORY_ROOT / "artifacts" / "spatial" / "phase1-evidence.json"


class SpatialEvidenceTests(unittest.TestCase):
    snapshot: ClassVar[CitySnapshot]
    index: ClassVar[SpatialIndex]
    evidence: ClassVar[dict[str, object]]
    payload: ClassVar[bytes]

    @classmethod
    def setUpClass(cls) -> None:
        cls.snapshot = load_city_snapshot(DATASET)
        cls.index = build_spatial_index(cls.snapshot)
        with (
            patch.object(
                evidence_module,
                "load_city_snapshot",
                return_value=cls.snapshot,
            ),
            patch.object(
                evidence_module,
                "build_spatial_index",
                return_value=cls.index,
            ),
        ):
            cls.evidence = build_evidence(DATASET)
        cls.payload = canonical_evidence_bytes(cls.evidence)

    def run_fast_main(self, arguments: list[str]) -> tuple[int, str]:
        errors = io.StringIO()
        with (
            patch.object(
                evidence_module,
                "build_evidence",
                return_value=self.evidence,
            ),
            redirect_stderr(errors),
        ):
            exit_code = main(arguments)
        return exit_code, errors.getvalue()

    def test_root_schema_source_and_topology_are_exact(self) -> None:
        self.assertEqual(
            set(self.evidence),
            {"aggregate", "cases", "evidence", "index", "source"},
        )
        self.assertEqual(
            self.evidence["evidence"],
            {
                "canonicalization": (
                    "US-ASCII, sorted object keys, two-space indent, LF, final newline"
                ),
                "phase": 1,
                "schema": EVIDENCE_SCHEMA,
                "schema_version": EVIDENCE_SCHEMA_VERSION,
                "tool": "python3 -m urbanlens.spatial_evidence",
                "tool_version": "0.3.0",
            },
        )
        self.assertEqual(
            self.evidence["source"],
            {
                "audit_schema": "urbanlens.data-quality-manifest",
                "audit_schema_version": 3,
                "byte_size": EXPECTED_SOURCE_BYTES,
                "data_rows": EXPECTED_SOURCE_ROWS,
                "sha256": EXPECTED_SOURCE_SHA256,
            },
        )
        index = self.evidence["index"]
        self.assertIsInstance(index, dict)
        assert isinstance(index, dict)
        self.assertEqual(index["records"], EXPECTED_SOURCE_ROWS)
        self.assertEqual(index["topology_sha256"], EXPECTED_TOPOLOGY_SHA256)
        self.assertEqual(index["algorithm"], "balanced-unit-sphere-kd-tree")
        self.assertEqual(index["version"], 1)
        self.assertEqual(index["earth_radius_millimeters"], 6_371_008_800)
        self.assertEqual(index["unique_numeric_locations"], 44_554)
        self.assertEqual(index["collision_groups"], 106)
        self.assertEqual(index["collision_excess"], 137)

    def test_case_order_queries_diagnostics_and_results_are_exact(self) -> None:
        cases = self.evidence["cases"]
        self.assertIsInstance(cases, list)
        assert isinstance(cases, list)
        self.assertEqual(
            [case["case_id"] for case in cases],
            ["co-location", "antimeridian", "north-pole", "london"],
        )

        expected_queries = (
            ("-11.1", "-77.6", -111_000_000, -776_000_000),
            ("-16", "180", -160_000_000, -1_800_000_000),
            ("90", "73", 900_000_000, 0),
            ("51.5074", "-0.1278", 515_074_000, -1_278_000),
        )
        expected_diagnostics = (
            {"evaluated": 16, "pruned": 44_675, "visited": 30},
            {"evaluated": 25, "pruned": 44_666, "visited": 43},
            {"evaluated": 48, "pruned": 44_643, "visited": 84},
            {"evaluated": 16, "pruned": 44_675, "visited": 30},
        )
        expected_results = (
            [
                {
                    "city_ascii": "Huacho",
                    "country": "Peru",
                    "distance_m": 0,
                    "iso2": "PE",
                    "latitude_e7": -111_000_000,
                    "longitude_e7": -776_000_000,
                    "rank": 1,
                    "source_id": "1604316784",
                },
                {
                    "city_ascii": "Huaura",
                    "country": "Peru",
                    "distance_m": 0,
                    "iso2": "PE",
                    "latitude_e7": -111_000_000,
                    "longitude_e7": -776_000_000,
                    "rank": 2,
                    "source_id": "1604503366",
                },
                {
                    "city_ascii": "Barranca",
                    "country": "Peru",
                    "distance_m": 42_289,
                    "iso2": "PE",
                    "latitude_e7": -107_541_000,
                    "longitude_e7": -777_610_000,
                    "rank": 3,
                    "source_id": "1604547675",
                },
            ],
            [
                {
                    "city_ascii": "Labasa",
                    "country": "Fiji",
                    "distance_m": 82_573,
                    "iso2": "FJ",
                    "latitude_e7": -164_311_000,
                    "longitude_e7": 1_793_703_000,
                    "rank": 1,
                    "source_id": "1242740801",
                },
                {
                    "city_ascii": "Nausori",
                    "country": "Fiji",
                    "distance_m": 273_112,
                    "iso2": "FJ",
                    "latitude_e7": -180_244_000,
                    "longitude_e7": 1_785_454_000,
                    "rank": 2,
                    "source_id": "1242640119",
                },
                {
                    "city_ascii": "Leava",
                    "country": "Wallis and Futuna",
                    "distance_m": 274_019,
                    "iso2": "WF",
                    "latitude_e7": -142_933_000,
                    "longitude_e7": -1_781_583_000,
                    "rank": 3,
                    "source_id": "1876731744",
                },
            ],
            [
                {
                    "city_ascii": "Nord",
                    "country": "Greenland",
                    "distance_m": 921_073,
                    "iso2": "GL",
                    "latitude_e7": 817_166_000,
                    "longitude_e7": -178_000_000,
                    "rank": 1,
                    "source_id": "1304217709",
                },
                {
                    "city_ascii": "Longyearbyen",
                    "country": "Svalbard",
                    "distance_m": 1_310_245,
                    "iso2": "XR",
                    "latitude_e7": 782_167_000,
                    "longitude_e7": 156_333_000,
                    "rank": 2,
                    "source_id": "1930654114",
                },
                {
                    "city_ascii": "Qaanaaq",
                    "country": "Greenland",
                    "distance_m": 1_393_641,
                    "iso2": "GL",
                    "latitude_e7": 774_667_000,
                    "longitude_e7": -692_306_000,
                    "rank": 3,
                    "source_id": "1304094556",
                },
            ],
            [
                {
                    "city_ascii": "London",
                    "country": "United Kingdom",
                    "distance_m": 30,
                    "iso2": "GB",
                    "latitude_e7": 515_072_000,
                    "longitude_e7": -1_275_000,
                    "rank": 1,
                    "source_id": "1826645935",
                },
                {
                    "city_ascii": "Westminster",
                    "country": "United Kingdom",
                    "distance_m": 1_505,
                    "iso2": "GB",
                    "latitude_e7": 514_947_000,
                    "longitude_e7": -1_353_000,
                    "rank": 2,
                    "source_id": "1826759011",
                },
                {
                    "city_ascii": "Holborn",
                    "country": "United Kingdom",
                    "distance_m": 1_748,
                    "iso2": "GB",
                    "latitude_e7": 515_204_000,
                    "longitude_e7": -1_136_000,
                    "rank": 3,
                    "source_id": "1826657058",
                },
            ],
        )

        for case, query, diagnostics, results in zip(
            cases,
            expected_queries,
            expected_diagnostics,
            expected_results,
            strict=True,
        ):
            requested_latitude, requested_longitude, latitude_e7, longitude_e7 = query
            self.assertEqual(
                case["query"]["requested"],
                {
                    "latitude": requested_latitude,
                    "longitude": requested_longitude,
                },
            )
            self.assertEqual(case["query"]["canonical"]["latitude_e7"], latitude_e7)
            self.assertEqual(
                case["query"]["canonical"]["longitude_e7"],
                longitude_e7,
            )
            self.assertEqual(case["proof"]["accelerated"], diagnostics)
            self.assertEqual(
                case["proof"]["oracle"],
                {"evaluated": EXPECTED_SOURCE_ROWS, "status": "match"},
            )
            self.assertEqual(case["results"], results)

    def test_every_case_and_aggregate_satisfy_exact_accounting(self) -> None:
        cases = self.evidence["cases"]
        assert isinstance(cases, list)
        for case in cases:
            accelerated = case["proof"]["accelerated"]
            reduction = case["proof"]["work_reduction"]
            self.assertEqual(
                accelerated["evaluated"] + accelerated["pruned"],
                EXPECTED_SOURCE_ROWS,
            )
            self.assertEqual(
                reduction["fraction"],
                {
                    "denominator": EXPECTED_SOURCE_ROWS,
                    "numerator": accelerated["pruned"],
                },
            )

        self.assertEqual(
            self.evidence["aggregate"],
            {
                "accelerated_evaluations": 105,
                "oracle_evaluations": 178_764,
                "pruned_records": 178_659,
                "query_count": 4,
                "visited_nodes": 187,
                "work_reduction": {
                    "avoided_evaluations": 178_659,
                    "basis_points": 9_994,
                    "fraction": {
                        "denominator": 178_764,
                        "numerator": 178_659,
                    },
                },
            },
        )

    def test_payload_is_canonical_ascii_deterministic_and_fresh(self) -> None:
        self.assertLess(len(self.payload), MAX_EVIDENCE_BYTES)
        self.assertTrue(self.payload.endswith(b"\n"))
        self.assertEqual(self.payload.decode("ascii").encode("ascii"), self.payload)
        self.assertEqual(json.loads(self.payload), self.evidence)
        self.assertEqual(canonical_evidence_bytes(self.evidence), self.payload)
        self.assertEqual(ARTIFACT.read_bytes(), self.payload)
        self.assertTrue(artifact_is_current(ARTIFACT, self.payload))

    def test_schema_is_allowlisted_and_contains_no_ambient_or_float_data(self) -> None:
        banned_key_fragments = {
            "benchmark",
            "duration",
            "elapsed",
            "hostname",
            "password",
            "path",
            "raw",
            "secret",
            "timestamp",
            "token",
            "username",
        }
        allowed_result_keys = {
            "city_ascii",
            "country",
            "distance_m",
            "iso2",
            "latitude_e7",
            "longitude_e7",
            "rank",
            "source_id",
        }

        def inspect(value: object) -> None:
            self.assertNotIsInstance(value, float)
            if isinstance(value, dict):
                for key, child in value.items():
                    self.assertIsInstance(key, str)
                    lowered = key.lower()
                    self.assertFalse(
                        any(fragment in lowered for fragment in banned_key_fragments),
                        key,
                    )
                    inspect(child)
            elif isinstance(value, list):
                for child in value:
                    inspect(child)

        inspect(self.evidence)
        cases = self.evidence["cases"]
        assert isinstance(cases, list)
        for case in cases:
            for result in case["results"]:
                self.assertEqual(set(result), allowed_result_keys)

        text = self.payload.decode("ascii")
        for ambient in (
            str(REPOSITORY_ROOT),
            "/home/",
            "train.csv",
            "PYTHONHASHSEED",
        ):
            self.assertNotIn(ambient, text)
        self.assertNotIn("squared_chord_distance", text)

    def test_wrong_topology_fails_closed_before_query_evidence(self) -> None:
        with (
            patch.object(
                evidence_module,
                "EXPECTED_TOPOLOGY_SHA256",
                "0" * 64,
            ),
            patch.object(
                evidence_module,
                "load_city_snapshot",
                return_value=self.snapshot,
            ),
            patch.object(
                evidence_module,
                "build_spatial_index",
                return_value=self.index,
            ),
            patch.object(
                evidence_module,
                "_case_payload",
                side_effect=AssertionError("query should not run"),
            ),
            self.assertRaises(EvidenceGenerationError),
        ):
            build_evidence(DATASET)

    def test_serialization_rejects_nonfinite_and_oversized_output(self) -> None:
        with self.assertRaises(EvidenceArtifactError):
            canonical_evidence_bytes({"unsafe": float("nan")})
        with self.assertRaises(EvidenceArtifactError):
            canonical_evidence_bytes({"oversized": "x" * MAX_EVIDENCE_BYTES})

    def test_check_distinguishes_missing_and_stale_from_unsafe_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            missing = root / "missing.json"
            exit_code, errors = self.run_fast_main(
                ["--artifact", str(missing), "--check"]
            )
            self.assertEqual(exit_code, EXIT_STALE)
            self.assertEqual(
                errors,
                "spatial evidence FAIL: tracked bytes are stale\n",
            )
            self.assertNotIn(str(missing), errors)

            stale = root / "stale.json"
            stale.write_bytes(b"{}\n")
            exit_code, errors = self.run_fast_main(
                ["--artifact", str(stale), "--check"]
            )
            self.assertEqual(exit_code, EXIT_STALE)
            self.assertEqual(
                errors,
                "spatial evidence FAIL: tracked bytes are stale\n",
            )

            target = root / "target.json"
            target.write_bytes(self.payload)
            symlink = root / "symlink.json"
            symlink.symlink_to(target)
            exit_code, errors = self.run_fast_main(
                ["--artifact", str(symlink), "--check"]
            )
            self.assertEqual(exit_code, EXIT_OUTPUT_ERROR)
            self.assertEqual(
                errors,
                "error: spatial evidence artifact operation failed\n",
            )
            self.assertNotIn(str(symlink), errors)

            if hasattr(os, "mkfifo"):
                fifo = root / "evidence.fifo"
                os.mkfifo(fifo)
                exit_code, errors = self.run_fast_main(
                    ["--artifact", str(fifo), "--check"]
                )
                self.assertEqual(exit_code, EXIT_OUTPUT_ERROR)
                self.assertEqual(
                    errors,
                    "error: spatial evidence artifact operation failed\n",
                )

    def test_atomic_write_and_source_overwrite_refusal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            target = Path(temporary_directory) / "evidence.json"
            write_artifact(target, DATASET, self.payload)
            self.assertEqual(target.read_bytes(), self.payload)
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o644)

            target.write_bytes(b"stale\n")
            write_artifact(target, DATASET, self.payload)
            self.assertEqual(target.read_bytes(), self.payload)

        before = hashlib.sha256(DATASET.read_bytes()).hexdigest()
        with self.assertRaises(EvidenceArtifactError):
            write_artifact(DATASET, DATASET, self.payload)
        after = hashlib.sha256(DATASET.read_bytes()).hexdigest()
        self.assertEqual(before, EXPECTED_SOURCE_SHA256)
        self.assertEqual(after, before)

    def test_main_errors_never_echo_rejected_paths_or_tracebacks(self) -> None:
        rejected_path = "/private/omar-secret/evidence-input.csv"
        errors = io.StringIO()
        with redirect_stderr(errors):
            exit_code = main(["--dataset", rejected_path, "--check"])
        self.assertEqual(exit_code, EXIT_INPUT_ERROR)
        self.assertEqual(
            errors.getvalue(),
            "error: spatial evidence input was rejected\n",
        )
        self.assertNotIn(rejected_path, errors.getvalue())
        self.assertNotIn("Traceback", errors.getvalue())

    def test_hash_seed_does_not_change_real_evidence_bytes(self) -> None:
        script = """\
import hashlib
from pathlib import Path
from urbanlens.spatial_evidence import build_evidence, canonical_evidence_bytes
payload = canonical_evidence_bytes(build_evidence(Path("train.csv")))
print(hashlib.sha256(payload).hexdigest())
"""
        expected_digest = hashlib.sha256(self.payload).hexdigest()
        for seed in ("1", "8675309"):
            environment = os.environ.copy()
            environment["PYTHONHASHSEED"] = seed
            result = subprocess.run(
                [sys.executable, "-c", script],
                cwd=REPOSITORY_ROOT,
                env=environment,
                capture_output=True,
                check=False,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stderr, "")
            self.assertEqual(result.stdout, expected_digest + "\n")

    def test_checked_cli_succeeds_on_real_tracked_bytes(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "urbanlens.spatial_evidence",
                "--check",
            ],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            check=False,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, EXIT_SUCCESS, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertEqual(
            result.stderr,
            "spatial evidence PASS: tracked bytes are current\n",
        )


if __name__ == "__main__":
    unittest.main()
