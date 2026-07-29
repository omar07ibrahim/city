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
from typing import ClassVar
from unittest.mock import patch
from xml.etree import ElementTree

import urbanlens.visuals as visual_module
from urbanlens.audit import (
    DEFAULT_CONTRACT,
    MANIFEST_SCHEMA,
    MANIFEST_SCHEMA_VERSION,
    MAX_CSV_PHYSICAL_LINE_CHARACTERS,
    MAX_DATASET_BYTES,
    MAX_DATASET_COLUMNS,
    MAX_DATASET_ROWS,
)
from urbanlens.visuals import (
    CHECK_AUDIT_COMMAND,
    FRESH_AUDIT_COMMAND,
    VISUAL_FILENAMES,
    ChartObservation,
    CommandCapture,
    CommandResult,
    VisualDataError,
    VisualEvidence,
    VisualFreshnessError,
    VisualOutputError,
    collect_evidence,
    generate_visuals,
    stale_visuals,
    write_visuals,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATASET = REPOSITORY_ROOT / "train.csv"
CHECKED_MANIFEST = REPOSITORY_ROOT / "artifacts" / "data_quality" / "train.quality.json"
VISUAL_DIRECTORY = REPOSITORY_ROOT / "docs" / "visuals"
SVG_NAMESPACE = "http://www.w3.org/2000/svg"


def _relative_luminance(color: str) -> float:
    channels = [int(color[index : index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [
        channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4
        for channel in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast_ratio(foreground: str, background: str) -> float:
    lighter, darker = sorted(
        (_relative_luminance(foreground), _relative_luminance(background)),
        reverse=True,
    )
    return (lighter + 0.05) / (darker + 0.05)


class VisualGenerationTests(unittest.TestCase):
    evidence: ClassVar[VisualEvidence]
    capture: ClassVar[CommandCapture]
    artifacts: ClassVar[dict[str, bytes]]

    @classmethod
    def setUpClass(cls) -> None:
        cls.evidence, cls.capture = collect_evidence(REPOSITORY_ROOT)
        cls.artifacts = generate_visuals(cls.evidence, cls.capture)

    def test_source_binding_and_comparable_observations_are_exact(self) -> None:
        self.assertEqual(self.evidence.source_sha256, DEFAULT_CONTRACT.sha256)
        self.assertEqual(self.evidence.profiled_rows, 44_691)
        self.assertEqual(
            self.evidence.observations,
            (
                ChartObservation(
                    "admin_name.missing",
                    "Missing admin name",
                    316,
                    7_071,
                ),
                ChartObservation(
                    "population.missing",
                    "Missing population",
                    307,
                    6_869,
                ),
                ChartObservation(
                    "grain.repeated_city_country_admin",
                    "Repeated city/country/admin (excess rows)",
                    301,
                    6_735,
                ),
                ChartObservation(
                    "grain.repeated_coordinates",
                    "Repeated coordinates (excess rows)",
                    136,
                    3_043,
                ),
            ),
        )

    def test_generation_is_byte_deterministic(self) -> None:
        first = generate_visuals(self.evidence, self.capture)
        second = generate_visuals(self.evidence, self.capture)
        self.assertEqual(first, second)
        self.assertEqual(tuple(first), VISUAL_FILENAMES)

    def test_every_svg_is_well_formed_and_has_fixed_dimensions(self) -> None:
        expected_dimensions = {
            "data-quality-observations.svg": ("1200", "720", "0 0 1200 720"),
            "phase-0-workflow.svg": ("1320", "520", "0 0 1320 520"),
            "audit-cli-result.svg": ("1200", "560", "0 0 1200 560"),
        }
        for name, payload in self.artifacts.items():
            root = ElementTree.fromstring(payload)
            self.assertEqual(root.tag, f"{{{SVG_NAMESPACE}}}svg")
            width, height, view_box = expected_dimensions[name]
            self.assertEqual(root.attrib["width"], width)
            self.assertEqual(root.attrib["height"], height)
            self.assertEqual(root.attrib["viewBox"], view_box)
            self.assertEqual(root.attrib["fill"], visual_module.INK)
            self.assertEqual(root.attrib["role"], "img")
            self.assertTrue(payload.endswith(b"\n"))

    def test_chart_has_sorted_exact_values_labels_and_zero_baseline(self) -> None:
        chart = self.artifacts["data-quality-observations.svg"].decode("utf-8")
        expected_labels = (
            "Missing admin name",
            "Missing population",
            "Repeated city/country/admin (excess rows)",
            "Repeated coordinates (excess rows)",
        )
        positions = [chart.index(label) for label in expected_labels]
        self.assertEqual(positions, sorted(positions))
        for direct_label in (
            "316 · 0.71%",
            "307 · 0.69%",
            "301 · 0.67%",
            "136 · 0.30%",
        ):
            self.assertIn(direct_label, chart)
        self.assertIn("n = 44,691 profiled rows", chart)
        self.assertIn('x1="420"', chart)
        self.assertIn('stroke="#172033" stroke-width="2"', chart)
        self.assertIn(f"{DEFAULT_CONTRACT.sha256[:12]}…", chart)
        self.assertIn(
            f"{MANIFEST_SCHEMA} v{MANIFEST_SCHEMA_VERSION}",
            chart,
        )

    def test_workflow_matches_the_implemented_phase_zero_pipeline(self) -> None:
        workflow = self.artifacts["phase-0-workflow.svg"].decode("utf-8")
        expected_limits = (
            (
                "≤ "
                f"{MAX_DATASET_BYTES // (1024 * 1024)} MiB · "
                f"≤ {MAX_DATASET_ROWS // 1_000}k rows"
            ),
            f"≤ {MAX_DATASET_COLUMNS} columns",
            f"≤ {MAX_CSV_PHYSICAL_LINE_CHARACTERS:,} chars / line",
            f"frozen schema · {len(DEFAULT_CONTRACT.columns)} columns",
        )
        for label in (
            "Captured source bytes",
            "regular file · no-follow",
            "Bounded CSV parser",
            "Contract + metrics",
            "Canonical JSON",
            "Freshness check",
            "fresh bytes == tracked",
            "CURRENT · IMPLEMENTED",
            "No map, model, API, or live data claim",
            *expected_limits,
        ):
            self.assertIn(label, workflow)

    def test_text_cascade_and_stage_number_contrast_are_accessible(self) -> None:
        workflow_root = ElementTree.fromstring(self.artifacts["phase-0-workflow.svg"])
        style = workflow_root.find(f"{{{SVG_NAMESPACE}}}style")
        self.assertIsNotNone(style)
        style_text = "" if style is None or style.text is None else style.text
        self.assertNotIn("fill:", style_text)

        circles = {
            circle.attrib["cx"]: circle.attrib["fill"]
            for circle in workflow_root.findall(f"{{{SVG_NAMESPACE}}}circle")
        }
        stage_numbers = [
            text
            for text in workflow_root.findall(f"{{{SVG_NAMESPACE}}}text")
            if (text.text or "") in {"1", "2", "3", "4", "5"}
        ]
        self.assertEqual(len(stage_numbers), 5)
        for number in stage_numbers:
            foreground = number.attrib["fill"]
            background = circles[number.attrib["x"]]
            expected_foreground = (
                visual_module.WHITE
                if background == visual_module.BLUE
                else visual_module.INK
            )
            self.assertEqual(foreground, expected_foreground)
            self.assertGreaterEqual(
                _contrast_ratio(foreground, background),
                4.5,
            )

        self.assertAlmostEqual(
            _contrast_ratio(visual_module.WHITE, visual_module.GOLD),
            3.032,
            places=3,
        )
        self.assertGreaterEqual(
            _contrast_ratio(visual_module.INK, visual_module.GOLD),
            4.5,
        )
        for foreground, background in (
            (visual_module.MUTED, visual_module.WHITE),
            (visual_module.MUTED, visual_module.BLUE_OPEN),
            (visual_module.MUTED, visual_module.GOLD_OPEN),
            (visual_module.BLUE_DARK, visual_module.BLUE_OPEN),
            (visual_module.GOLD_DARK, visual_module.GOLD_OPEN),
        ):
            self.assertGreaterEqual(
                _contrast_ratio(foreground, background),
                4.5,
            )

    def test_cli_capture_is_the_actual_success_result(self) -> None:
        self.assertEqual(self.capture.display_argv, CHECK_AUDIT_COMMAND)
        self.assertEqual(self.capture.returncode, 0)
        self.assertEqual(self.capture.stdout, "")
        self.assertEqual(
            self.capture.stderr,
            "manifest PASS: artifacts/data_quality/train.quality.json",
        )
        rendered = self.artifacts["audit-cli-result.svg"].decode("utf-8")
        self.assertIn(
            "$ python3 -m urbanlens.audit --check-manifest "
            "artifacts/data_quality/train.quality.json",
            rendered,
        )
        self.assertIn(self.capture.stderr, rendered)
        self.assertIn("EXIT CODE 0", rendered)

    def test_visual_text_contains_no_raw_source_labels(self) -> None:
        with DATASET.open(encoding="utf-8-sig", newline="") as stream:
            source_labels = {
                row[column].strip()
                for row in csv.DictReader(stream)
                for column in ("city", "city_ascii", "country", "admin_name")
                if row[column].strip()
            }
        rendered_strings = {
            text.strip()
            for payload in self.artifacts.values()
            for text in ElementTree.fromstring(payload).itertext()
            if text.strip()
        }
        self.assertTrue(source_labels.isdisjoint(rendered_strings))

    def test_visuals_embed_no_paths_secrets_remote_assets_or_scripts(self) -> None:
        combined = b"\n".join(self.artifacts.values())
        forbidden = (
            os.fsencode(str(REPOSITORY_ROOT.resolve())),
            b"/home/",
            b"BEGIN PRIVATE KEY",
            b"Bearer ",
            b"password=",
            b"token=",
            b"<script",
            b"@font-face",
            b"url(",
            b" href=",
            b" xlink:href=",
        )
        for value in forbidden:
            self.assertNotIn(value, combined)

    def test_manifest_with_wrong_hash_is_rejected(self) -> None:
        manifest = json.loads(CHECKED_MANIFEST.read_bytes())
        manifest["dataset"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(VisualDataError, "frozen dataset hash"):
            visual_module._evidence_from_manifest(manifest)

    def test_wrong_manifest_schema_or_version_is_rejected(self) -> None:
        wrong_schema = json.loads(CHECKED_MANIFEST.read_bytes())
        wrong_schema["manifest"]["schema"] = f"{MANIFEST_SCHEMA}.other"
        with self.assertRaisesRegex(VisualDataError, "manifest schema must equal"):
            visual_module._evidence_from_manifest(wrong_schema)

        wrong_version = json.loads(CHECKED_MANIFEST.read_bytes())
        wrong_version["manifest"]["schema_version"] = MANIFEST_SCHEMA_VERSION + 1
        with self.assertRaisesRegex(
            VisualDataError,
            "manifest schema version must equal",
        ):
            visual_module._evidence_from_manifest(wrong_version)

    def test_freshness_mismatch_fails_before_rendering_or_writing(self) -> None:
        mismatched = CommandResult(
            display_argv=FRESH_AUDIT_COMMAND,
            returncode=0,
            stdout=b"{}\n",
            stderr=b"",
        )
        with (
            patch.object(
                visual_module,
                "_run_repository_command",
                return_value=mismatched,
            ),
            self.assertRaisesRegex(
                VisualFreshnessError,
                "do not equal",
            ),
        ):
            collect_evidence(REPOSITORY_ROOT)

        with (
            patch.object(
                visual_module,
                "collect_evidence",
                side_effect=VisualFreshnessError("stale source"),
            ),
            patch.object(visual_module, "write_visuals") as writer,
            redirect_stderr(io.StringIO()),
        ):
            return_code = visual_module.main(["--write"])
        self.assertEqual(return_code, 1)
        writer.assert_not_called()


class VisualOutputSafetyTests(unittest.TestCase):
    @staticmethod
    def _artifacts() -> dict[str, bytes]:
        return {
            name: (
                '<?xml version="1.0" encoding="UTF-8"?>\n'
                f'<svg xmlns="{SVG_NAMESPACE}"><title>{name}</title></svg>\n'
            ).encode()
            for name in VISUAL_FILENAMES
        }

    def test_safe_write_round_trip_and_modes(self) -> None:
        artifacts = self._artifacts()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_visuals(output, artifacts)
            self.assertEqual(stale_visuals(output, artifacts), ())
            for name, payload in artifacts.items():
                path = output / name
                self.assertEqual(path.read_bytes(), payload)
                self.assertEqual(path.stat().st_mode & 0o777, 0o644)

    def test_symlink_output_is_not_replaced_or_followed(self) -> None:
        artifacts = self._artifacts()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "visuals"
            output.mkdir()
            outside = root / "outside.svg"
            outside.write_bytes(b"must remain unchanged")
            linked_output = output / VISUAL_FILENAMES[0]
            linked_output.symlink_to(outside)
            with self.assertRaisesRegex(
                VisualOutputError,
                "refusing to replace symlink",
            ):
                write_visuals(output, artifacts)
            self.assertTrue(linked_output.is_symlink())
            self.assertEqual(outside.read_bytes(), b"must remain unchanged")

    def test_parent_swap_cannot_redirect_anchored_writes(self) -> None:
        artifacts = self._artifacts()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "visuals"
            output.mkdir()
            anchored = root / "anchored-visuals"
            redirected = root / "redirected"
            redirected.mkdir()
            sentinel = redirected / VISUAL_FILENAMES[0]
            sentinel.write_bytes(b"must remain unchanged")

            descriptor = visual_module._open_visual_directory(output)
            try:
                output.rename(anchored)
                output.symlink_to(redirected, target_is_directory=True)
                visual_module._write_visuals_to_anchored_directory(
                    descriptor,
                    artifacts,
                )
            finally:
                os.close(descriptor)

            for name, payload in artifacts.items():
                self.assertEqual((anchored / name).read_bytes(), payload)
            self.assertEqual(sentinel.read_bytes(), b"must remain unchanged")
            self.assertFalse((redirected / VISUAL_FILENAMES[1]).exists())


class VisualCommandLineTests(unittest.TestCase):
    def _run(self, *arguments: str) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            [sys.executable, "-m", "urbanlens.visuals", *arguments],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            check=False,
            timeout=60,
        )

    def test_checked_visuals_equal_fresh_in_memory_render(self) -> None:
        completed = self._run("--check")
        self.assertEqual(completed.returncode, 0, completed.stderr.decode())
        self.assertEqual(completed.stdout, b"")
        self.assertIn(b"3 source-bound SVGs are current", completed.stderr)

    def test_mode_is_required(self) -> None:
        completed = self._run()
        self.assertEqual(completed.returncode, 2)
        self.assertIn(
            b"one of the arguments --check --write is required", completed.stderr
        )


if __name__ == "__main__":
    unittest.main()
