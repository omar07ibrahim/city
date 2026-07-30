from __future__ import annotations

import copy
import io
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from contextlib import redirect_stderr
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import patch

import urbanlens.spatial_visuals as visual_module
from urbanlens.nearest import canonical_receipt_bytes
from urbanlens.spatial_visuals import (
    EVIDENCE_CHECK_COMMAND,
    EXIT_ARGUMENT_ERROR,
    EXIT_INPUT_ERROR,
    EXIT_OUTPUT_ERROR,
    EXIT_STALE,
    EXIT_SUCCESS,
    MAX_SVG_BYTES,
    NEAREST_COMMAND,
    VISUAL_FILENAMES,
    CommandResult,
    NearestCapture,
    SpatialVisualDataError,
    SpatialVisualEvidence,
    SpatialVisualFreshnessError,
    SpatialVisualOutputError,
    collect_evidence,
    evidence_from_document,
    generate_visuals,
    main,
    stale_visuals,
    write_visuals,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
VISUAL_DIRECTORY = REPOSITORY_ROOT / "docs" / "visuals"
EVIDENCE_ARTIFACT = REPOSITORY_ROOT / "artifacts" / "spatial" / "phase1-evidence.json"
EXPECTED_DIMENSIONS = {
    "spatial-index-architecture.svg": ("1500", "860"),
    "spatial-work-reduction.svg": ("1400", "860"),
    "nearest-city-cli-result.svg": ("1400", "900"),
    "spatial-edge-case-proof.svg": ("1400", "850"),
}


def _relative_luminance(color: str) -> float:
    channels = []
    for offset in (1, 3, 5):
        value = int(color[offset : offset + 2], 16) / 255
        channels.append(
            value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
        )
    red, green, blue = channels
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _contrast_ratio(first: str, second: str) -> float:
    light, dark = sorted(
        (_relative_luminance(first), _relative_luminance(second)),
        reverse=True,
    )
    return (light + 0.05) / (dark + 0.05)


class SpatialVisualTests(unittest.TestCase):
    evidence: ClassVar[SpatialVisualEvidence]
    capture: ClassVar[NearestCapture]
    artifacts: ClassVar[dict[str, bytes]]
    document: ClassVar[dict[str, Any]]

    @classmethod
    def setUpClass(cls) -> None:
        cls.evidence, cls.capture = collect_evidence(REPOSITORY_ROOT)
        cls.artifacts = generate_visuals(cls.evidence, cls.capture)
        parsed = json.loads(EVIDENCE_ARTIFACT.read_text(encoding="ascii"))
        if not isinstance(parsed, dict):
            raise TypeError("tracked evidence must be an object")
        cls.document = parsed

    def test_real_collection_is_bound_to_source_index_and_actual_cli(self) -> None:
        evidence = self.evidence
        capture = self.capture

        self.assertEqual(evidence.source_rows, 44_691)
        self.assertEqual(evidence.source_bytes, 4_734_682)
        self.assertEqual(
            evidence.source_sha256,
            "de941def7faca87c0911abb79c3cbd07672887fd486a9b7bea6c48c12ce0cf18",
        )
        self.assertEqual(
            evidence.topology_sha256,
            "5ba1bb95e167d7bdf3b9e28e8e211756eab48888c281c215d1dccf801db3cb72",
        )
        self.assertEqual(evidence.index_records, 44_691)
        self.assertEqual(evidence.unique_numeric_locations, 44_554)
        self.assertEqual(evidence.query_count, 4)
        self.assertEqual(evidence.accelerated_evaluations, 105)
        self.assertEqual(evidence.oracle_evaluations, 178_764)
        self.assertEqual(evidence.pruned_records, 178_659)
        self.assertEqual(evidence.work_reduction_basis_points, 9_994)

        self.assertEqual(capture.display_argv, NEAREST_COMMAND)
        self.assertEqual(capture.returncode, 0)
        self.assertEqual(capture.stdout_bytes, 2_118)
        self.assertEqual(
            (capture.evaluated, capture.pruned, capture.visited),
            (16, 44_675, 30),
        )
        self.assertEqual(capture.oracle_evaluated, 44_691)
        self.assertEqual(capture.oracle_status, "match")
        self.assertEqual(
            [
                (item.source_id, item.city_ascii, item.distance_m)
                for item in capture.results
            ],
            [
                ("1604316784", "Huacho", 0),
                ("1604503366", "Huaura", 0),
                ("1604547675", "Barranca", 42_289),
            ],
        )

    def test_generated_artifact_set_is_ordered_deterministic_and_fresh(self) -> None:
        self.assertEqual(tuple(self.artifacts), VISUAL_FILENAMES)
        self.assertEqual(
            generate_visuals(self.evidence, self.capture),
            self.artifacts,
        )
        self.assertEqual(stale_visuals(VISUAL_DIRECTORY, self.artifacts), ())
        for name, expected in self.artifacts.items():
            self.assertEqual((VISUAL_DIRECTORY / name).read_bytes(), expected)
            self.assertLess(len(expected), MAX_SVG_BYTES)

    def test_every_svg_is_accessible_fixed_and_self_contained(self) -> None:
        namespace = "{http://www.w3.org/2000/svg}"
        for name, payload in self.artifacts.items():
            with self.subTest(name=name):
                root = ET.fromstring(payload)
                width, height = EXPECTED_DIMENSIONS[name]
                self.assertEqual(root.tag, namespace + "svg")
                self.assertEqual(root.attrib["width"], width)
                self.assertEqual(root.attrib["height"], height)
                self.assertEqual(root.attrib["viewBox"], f"0 0 {width} {height}")
                self.assertEqual(root.attrib["role"], "img")

                titles = root.findall(namespace + "title")
                descriptions = root.findall(namespace + "desc")
                self.assertEqual(len(titles), 1)
                self.assertEqual(len(descriptions), 1)
                self.assertTrue((titles[0].text or "").strip())
                self.assertTrue((descriptions[0].text or "").strip())
                labelled = root.attrib["aria-labelledby"].split()
                self.assertEqual(
                    labelled,
                    [titles[0].attrib["id"], descriptions[0].attrib["id"]],
                )

                lowered = payload.lower()
                for forbidden in (
                    b"<script",
                    b"<image",
                    b"<foreignobject",
                    b"xlink:",
                    b"javascript:",
                    b"data:",
                    b"onload=",
                    b"url(",
                ):
                    self.assertNotIn(forbidden, lowered)
                self.assertNotIn(b"<a ", lowered)

    def test_small_badge_text_meets_wcag_aa_contrast(self) -> None:
        self.assertGreaterEqual(
            _contrast_ratio(visual_module.GOLD_DARK, visual_module.WHITE),
            4.5,
        )
        architecture = self.artifacts["spatial-index-architecture.svg"]
        self.assertIn(
            b'width="46" height="28" rx="14" fill="#6F4A0C"',
            architecture,
        )

    def test_visual_text_contains_exact_evidence_and_honest_qualifiers(self) -> None:
        architecture = self.artifacts["spatial-index-architecture.svg"].decode("utf-8")
        work = self.artifacts["spatial-work-reduction.svg"].decode("utf-8")
        cli = self.artifacts["nearest-city-cli-result.svg"].decode("utf-8")
        edges = self.artifacts["spatial-edge-case-proof.svg"].decode("utf-8")

        for value in (
            "Implemented exact nearest-city architecture",
            "Single capture + audit",
            "Strict E7 boundary",
            "Balanced sphere kd-tree",
            "Conservative AABB",
            "Optional full-scan oracle",
            "Canonical JSON receipt",
            self.evidence.source_sha256,
            self.evidence.topology_sha256,
        ):
            self.assertIn(value, architecture)

        for value in (
            "105",
            "178,764",
            "178,659",
            "99.94%",
            "this is not a latency benchmark",
            "worst case",
            "16 evaluated",
            "48 evaluated",
        ):
            self.assertIn(value, work)

        for value in (
            "$ python3 -m urbanlens.nearest --dataset train.csv",
            "proof.accelerated evaluated=16",
            "proof.oracle      MATCH",
            "1604316784",
            "1604503366",
            "Huacho (PE)",
            "Huaura (PE)",
            "42,289 m",
            "stdout.bytes  2,118",
            "EXIT CODE 0",
        ):
            self.assertIn(value, cli)

        for value in (
            "Co-location tie",
            "Antimeridian",
            "North-pole collapse",
            "(-160000000, -1800000000)",
            "(900000000, 0)",
            "44,691 evaluated",
            "ORDERED TOP-3 RANK KEYS AGREE",
        ):
            self.assertIn(value, edges)

    def test_visuals_contain_no_ambient_machine_or_private_metadata(self) -> None:
        joined = b"\n".join(self.artifacts.values()).decode("utf-8")
        for forbidden in (
            str(REPOSITORY_ROOT),
            "/home/",
            "/private/",
            "ubuntu",
            "Omar Ibrahim",
            "localhost",
            "127.0.0.1",
            "process id",
            "PYTHONHASHSEED",
            "gho_",
        ):
            self.assertNotIn(forbidden, joined)
        self.assertNotIn("squared_chord_distance", joined)
        self.assertIsNone(re.search(r"\b20[0-9]{2}-[0-9]{2}-[0-9]{2}[T ]", joined))

    def test_document_validation_rejects_identity_or_accounting_mutation(self) -> None:
        mutations: list[tuple[tuple[str, ...], object]] = [
            (("source", "sha256"), "0" * 64),
            (("index", "topology_sha256"), "f" * 64),
            (("aggregate", "accelerated_evaluations"), 106),
            (("cases", "0", "proof", "oracle", "status"), "mismatch"),
            (("cases", "0", "purpose"), "changed purpose"),
            (("cases", "1", "query", "requested", "longitude"), "-180"),
            (("cases", "2", "query", "requested", "latitude"), "89"),
            (("cases", "3", "query", "k"), 2),
        ]
        for path, value in mutations:
            with self.subTest(path=path):
                document = copy.deepcopy(self.document)
                current: Any = document
                for key in path[:-1]:
                    current = (
                        current[int(key)] if isinstance(current, list) else current[key]
                    )
                current[path[-1]] = value
                with self.assertRaises(SpatialVisualDataError):
                    evidence_from_document(document)

    def test_actual_receipt_mutation_is_rejected_before_rendering(self) -> None:
        real_result = visual_module._run_repository_command(
            REPOSITORY_ROOT,
            NEAREST_COMMAND,
        )
        verified = visual_module._capture_from_receipt(real_result, self.evidence)
        self.assertEqual(verified, self.capture)

        receipt = json.loads(real_result.stdout.decode("ascii"))
        receipt["proof"]["accelerated"]["evaluated"] = 17
        mutated = CommandResult(
            display_argv=real_result.display_argv,
            returncode=0,
            stdout=canonical_receipt_bytes(receipt),
            stderr=b"",
        )
        with self.assertRaises(SpatialVisualDataError):
            visual_module._capture_from_receipt(mutated, self.evidence)

        noncanonical = CommandResult(
            display_argv=real_result.display_argv,
            returncode=0,
            stdout=real_result.stdout.rstrip(b"\n"),
            stderr=b"",
        )
        with self.assertRaises(SpatialVisualDataError):
            visual_module._capture_from_receipt(noncanonical, self.evidence)

    def test_evidence_check_contract_rejects_any_stream_or_exit_change(self) -> None:
        passing = CommandResult(
            display_argv=EVIDENCE_CHECK_COMMAND,
            returncode=0,
            stdout=b"",
            stderr=b"spatial evidence PASS: tracked bytes are current\n",
        )
        visual_module._require_evidence_check(passing)

        failures = (
            CommandResult(EVIDENCE_CHECK_COMMAND, 1, b"", passing.stderr),
            CommandResult(EVIDENCE_CHECK_COMMAND, 0, b"unexpected", passing.stderr),
            CommandResult(EVIDENCE_CHECK_COMMAND, 0, b"", b"unexpected"),
            CommandResult(NEAREST_COMMAND, 0, b"", passing.stderr),
        )
        for result in failures:
            with (
                self.subTest(result=result),
                self.assertRaises(SpatialVisualFreshnessError),
            ):
                visual_module._require_evidence_check(result)

    def test_safe_write_check_stale_and_file_modes(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".spatial-visual-test-",
            dir=REPOSITORY_ROOT,
        ) as temporary_directory:
            directory = Path(temporary_directory)
            self.assertEqual(stale_visuals(directory, self.artifacts), VISUAL_FILENAMES)

            write_visuals(directory, self.artifacts)
            self.assertEqual(stale_visuals(directory, self.artifacts), ())
            for name, payload in self.artifacts.items():
                path = directory / name
                self.assertEqual(path.read_bytes(), payload)
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o644)

            stale_name = VISUAL_FILENAMES[1]
            (directory / stale_name).write_bytes(b"<svg/>\n")
            self.assertEqual(stale_visuals(directory, self.artifacts), (stale_name,))

    def test_symlink_fifo_and_oversized_outputs_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".spatial-visual-unsafe-",
            dir=REPOSITORY_ROOT,
        ) as temporary_directory:
            directory = Path(temporary_directory)
            target = directory / "target.svg"
            target.write_bytes(b"safe-target\n")
            symlink_name = VISUAL_FILENAMES[0]
            (directory / symlink_name).symlink_to(target.name)
            with self.assertRaises(SpatialVisualOutputError):
                write_visuals(directory, self.artifacts)
            self.assertEqual(target.read_bytes(), b"safe-target\n")

            (directory / symlink_name).unlink()
            if hasattr(os, "mkfifo"):
                fifo_name = VISUAL_FILENAMES[2]
                os.mkfifo(directory / fifo_name)
                with self.assertRaises(SpatialVisualOutputError):
                    write_visuals(directory, self.artifacts)

        oversized = dict(self.artifacts)
        oversized[VISUAL_FILENAMES[0]] = b"x" * (MAX_SVG_BYTES + 1)
        with (
            tempfile.TemporaryDirectory(
                prefix=".spatial-visual-oversized-",
                dir=REPOSITORY_ROOT,
            ) as temporary_directory,
            self.assertRaises(SpatialVisualOutputError),
        ):
            write_visuals(Path(temporary_directory), oversized)

    def test_parent_path_swap_cannot_redirect_anchored_writes(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".spatial-visual-anchor-",
            dir=REPOSITORY_ROOT,
        ) as temporary_directory:
            root = Path(temporary_directory)
            visible = root / "visuals"
            held = root / "held"
            diverted = root / "diverted"
            visible.mkdir()
            diverted.mkdir()

            descriptor = visual_module._open_visual_directory(visible)
            try:
                visible.rename(held)
                visible.symlink_to(diverted.name, target_is_directory=True)
                visual_module._write_visuals_to_anchored_directory(
                    descriptor,
                    self.artifacts,
                )
            finally:
                os.close(descriptor)

            for name, payload in self.artifacts.items():
                self.assertEqual((held / name).read_bytes(), payload)
                self.assertFalse((diverted / name).exists())

    def test_main_modes_are_generic_and_do_not_leak_internal_errors(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".spatial-visual-main-",
            dir=REPOSITORY_ROOT,
        ) as temporary_directory:
            root = Path(temporary_directory)
            (root / "docs" / "visuals").mkdir(parents=True)

            def run(arguments: list[str]) -> tuple[int, str]:
                errors = io.StringIO()
                with (
                    patch.object(visual_module, "repository_root", return_value=root),
                    patch.object(
                        visual_module,
                        "collect_evidence",
                        return_value=(self.evidence, self.capture),
                    ),
                    redirect_stderr(errors),
                ):
                    code = main(arguments)
                return code, errors.getvalue()

            code, error_text = run(["--check"])
            self.assertEqual(code, EXIT_STALE)
            self.assertEqual(
                error_text,
                "spatial visuals FAIL: tracked SVG bytes are stale\n",
            )

            code, error_text = run(["--write"])
            self.assertEqual(code, EXIT_SUCCESS)
            self.assertEqual(
                error_text,
                "spatial visuals WRITE: 4 canonical SVGs replaced\n",
            )

            code, error_text = run(["--check"])
            self.assertEqual(code, EXIT_SUCCESS)
            self.assertEqual(
                error_text,
                "spatial visuals PASS: 4 source-bound SVGs are current\n",
            )

        error_stream = io.StringIO()
        with redirect_stderr(error_stream):
            code = main([])
        self.assertEqual(code, EXIT_ARGUMENT_ERROR)
        self.assertEqual(
            error_stream.getvalue(),
            "error: invalid visual command arguments\n",
        )

        private_path = "/private/customer/secret-source.json"
        failure_modes = (
            ("--check", EXIT_INPUT_ERROR),
            ("--write", EXIT_INPUT_ERROR),
        )
        for mode, expected_code in failure_modes:
            error_stream = io.StringIO()
            with (
                patch.object(
                    visual_module,
                    "collect_evidence",
                    side_effect=SpatialVisualFreshnessError(private_path),
                ),
                redirect_stderr(error_stream),
            ):
                code = main([mode])
            self.assertEqual(code, expected_code)
            self.assertEqual(
                error_stream.getvalue(),
                "spatial visuals FAIL: checked input was rejected\n",
            )
            self.assertNotIn(private_path, error_stream.getvalue())

        error_stream = io.StringIO()
        with (
            patch.object(
                visual_module,
                "collect_evidence",
                return_value=(self.evidence, self.capture),
            ),
            patch.object(
                visual_module,
                "write_visuals",
                side_effect=SpatialVisualOutputError(private_path),
            ),
            redirect_stderr(error_stream),
        ):
            code = main(["--write"])
        self.assertEqual(code, EXIT_OUTPUT_ERROR)
        self.assertEqual(
            error_stream.getvalue(),
            "spatial visuals FAIL: artifact operation failed\n",
        )
        self.assertNotIn(private_path, error_stream.getvalue())

    def test_real_checked_cli_succeeds_with_exact_generic_status(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "urbanlens.spatial_visuals",
                "--check",
            ],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            check=False,
            timeout=60,
        )
        self.assertEqual(result.returncode, EXIT_SUCCESS, result.stderr)
        self.assertEqual(result.stdout, b"")
        self.assertEqual(
            result.stderr,
            b"spatial visuals PASS: 4 source-bound SVGs are current\n",
        )


if __name__ == "__main__":
    unittest.main()
