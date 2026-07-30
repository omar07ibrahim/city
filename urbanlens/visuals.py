"""Generate source-bound, deterministic SVG evidence for UrbanLens phase 0.

The module deliberately uses only the Python standard library.  Every render
starts by running a fresh dataset audit and requiring its stdout bytes to equal
the checked manifest.  The resulting SVGs contain aggregate evidence only.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import secrets
import shlex
import stat
import subprocess
import sys
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

from urbanlens.audit import (
    DEFAULT_CONTRACT,
    MANIFEST_SCHEMA,
    MANIFEST_SCHEMA_VERSION,
    MAX_CSV_PHYSICAL_LINE_CHARACTERS,
    MAX_DATASET_BYTES,
    MAX_DATASET_COLUMNS,
    MAX_DATASET_ROWS,
    CheckedManifestInputError,
    _read_checked_manifest,
)

MANIFEST_RELATIVE_PATH: Final = Path("artifacts/data_quality/train.quality.json")
VISUAL_DIRECTORY_RELATIVE_PATH: Final = Path("docs/visuals")
FRESH_AUDIT_COMMAND: Final = (
    "python3",
    "-m",
    "urbanlens.audit",
    "train.csv",
)
CHECK_AUDIT_COMMAND: Final = (
    "python3",
    "-m",
    "urbanlens.audit",
    "--check-manifest",
    MANIFEST_RELATIVE_PATH.as_posix(),
)
VISUAL_FILENAMES: Final = (
    "data-quality-observations.svg",
    "phase-0-workflow.svg",
    "audit-cli-result.svg",
)
MAX_SVG_BYTES: Final = 512 * 1024

EXIT_SUCCESS: Final = 0
EXIT_STALE: Final = 1
EXIT_INPUT_ERROR: Final = 3
EXIT_OUTPUT_ERROR: Final = 4

INK: Final = "#172033"
MUTED: Final = "#5F6B7A"
GRID: Final = "#DDE3EA"
PANEL: Final = "#F7F9FC"
WHITE: Final = "#FFFFFF"
BLUE: Final = "#2563A6"
BLUE_DARK: Final = "#184A7A"
BLUE_OPEN: Final = "#E8F1FA"
GOLD: Final = "#C18A22"
GOLD_DARK: Final = "#7A5411"
GOLD_OPEN: Final = "#FBF3DF"

OBSERVATION_SPECS: Final = (
    (
        "admin_name.missing",
        "Missing admin name",
        ("quality", "metrics", "missingness", "admin_name", "count"),
        ("quality", "metrics", "missingness", "admin_name", "row_rate_ppm"),
    ),
    (
        "population.missing",
        "Missing population",
        ("quality", "metrics", "missingness", "population", "count"),
        ("quality", "metrics", "missingness", "population", "row_rate_ppm"),
    ),
    (
        "grain.repeated_city_country_admin",
        "Repeated city/country/admin (excess rows)",
        (
            "quality",
            "metrics",
            "duplicates",
            "city_country_admin",
            "duplicate_row_excess_count",
        ),
        None,
    ),
    (
        "grain.repeated_coordinates",
        "Repeated coordinates (excess rows)",
        (
            "quality",
            "metrics",
            "duplicates",
            "coordinates",
            "duplicate_row_excess_count",
        ),
        None,
    ),
)


class VisualFreshnessError(Exception):
    """Raised when fresh source evidence and checked evidence do not agree."""


class VisualDataError(Exception):
    """Raised when checked aggregate evidence violates the visual contract."""


class VisualOutputError(Exception):
    """Raised when SVG artifacts cannot be checked or written safely."""


@dataclass(frozen=True)
class CommandResult:
    """Raw result of one repository-local subprocess invocation."""

    display_argv: tuple[str, ...]
    returncode: int
    stdout: bytes
    stderr: bytes


@dataclass(frozen=True)
class CommandCapture:
    """Normalized command evidence safe to embed in an SVG."""

    display_argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class ChartObservation:
    """One comparable row-derived aggregate."""

    observation_id: str
    label: str
    count: int
    row_rate_ppm: int


@dataclass(frozen=True)
class VisualEvidence:
    """Validated aggregate input shared by every generated visual."""

    source_sha256: str
    profiled_rows: int
    observations: tuple[ChartObservation, ...]


def repository_root() -> Path:
    """Return this checkout's root without exposing it in generated output."""

    return Path(__file__).resolve().parents[1]


def _run_repository_command(
    root: Path,
    display_argv: tuple[str, ...],
) -> CommandResult:
    """Run one fixed Python command with a deterministic, minimal environment."""

    if not display_argv or display_argv[0] != "python3":
        raise VisualDataError("visual commands must use the python3 display name")
    environment = {
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONHASHSEED": "0",
        "PYTHONUTF8": "1",
    }
    try:
        completed = subprocess.run(
            [sys.executable, *display_argv[1:]],
            cwd=root,
            env=environment,
            capture_output=True,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise VisualFreshnessError(
            f"cannot run {' '.join(display_argv)}: {error}"
        ) from error
    return CommandResult(
        display_argv=display_argv,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _normalize_command_stream(payload: bytes, root: Path) -> str:
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise VisualDataError("audit command output is not valid UTF-8") from error
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    root_text = str(root.resolve())
    text = text.replace(f"{root_text}/", "")
    text = text.replace(root_text, ".")
    if any(ord(character) < 32 and character not in {"\n", "\t"} for character in text):
        raise VisualDataError("audit command output contains control characters")
    return text.rstrip("\n")


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise VisualDataError(f"{path} must be a JSON object")
    return cast(dict[str, Any], value)


def _list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise VisualDataError(f"{path} must be a JSON array")
    return value


def _path_value(manifest: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = manifest
    traversed: list[str] = []
    for key in path:
        traversed.append(key)
        value = _mapping(value, ".".join(traversed[:-1]) or "manifest").get(key)
        if value is None:
            raise VisualDataError(f"{'.'.join(traversed)} is missing")
    return value


def _nonnegative_int(value: Any, path: str) -> int:
    if type(value) is not int or value < 0:
        raise VisualDataError(f"{path} must be a non-negative integer")
    return value


def _observation_index(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw_observations = _list(
        _path_value(manifest, ("quality", "observations")),
        "quality.observations",
    )
    indexed: dict[str, dict[str, Any]] = {}
    for index, raw_observation in enumerate(raw_observations):
        observation = _mapping(
            raw_observation,
            f"quality.observations[{index}]",
        )
        observation_id = observation.get("id")
        if not isinstance(observation_id, str):
            raise VisualDataError(f"quality.observations[{index}].id must be a string")
        if observation_id in indexed:
            raise VisualDataError(f"duplicate observation id: {observation_id}")
        indexed[observation_id] = observation
    return indexed


def _evidence_from_manifest(manifest: dict[str, Any]) -> VisualEvidence:
    manifest_metadata = _mapping(manifest.get("manifest"), "manifest")
    if manifest_metadata.get("schema") != MANIFEST_SCHEMA:
        raise VisualDataError(f"manifest schema must equal {MANIFEST_SCHEMA}")
    schema_version = manifest_metadata.get("schema_version")
    if type(schema_version) is not int or schema_version != MANIFEST_SCHEMA_VERSION:
        raise VisualDataError(
            f"manifest schema version must equal {MANIFEST_SCHEMA_VERSION}"
        )

    dataset = _mapping(manifest.get("dataset"), "dataset")
    source_sha256 = dataset.get("sha256")
    if source_sha256 != DEFAULT_CONTRACT.sha256:
        raise VisualDataError("manifest is not bound to the frozen dataset hash")
    if dataset.get("name") != "train.csv":
        raise VisualDataError("manifest dataset name is not train.csv")
    if dataset.get("data_rows") != DEFAULT_CONTRACT.data_rows:
        raise VisualDataError("manifest row count is outside the frozen contract")
    quality = _mapping(manifest.get("quality"), "quality")
    if quality.get("status") != "pass":
        raise VisualDataError("visuals require a passing data-quality contract")

    profiled_rows = _nonnegative_int(
        _path_value(manifest, ("quality", "metrics", "profiled_rows")),
        "quality.metrics.profiled_rows",
    )
    if profiled_rows != DEFAULT_CONTRACT.data_rows:
        raise VisualDataError(
            "profiled row count does not equal the frozen dataset row count"
        )

    observed_by_id = _observation_index(manifest)
    observations: list[ChartObservation] = []
    for observation_id, label, count_path, explicit_rate_path in OBSERVATION_SPECS:
        count = _nonnegative_int(
            _path_value(manifest, count_path),
            ".".join(count_path),
        )
        derived_rate = (count * 1_000_000 + profiled_rows // 2) // profiled_rows
        if explicit_rate_path is None:
            rate = derived_rate
        else:
            rate = _nonnegative_int(
                _path_value(manifest, explicit_rate_path),
                ".".join(explicit_rate_path),
            )
            if rate != derived_rate:
                raise VisualDataError(
                    f"{observation_id} rate does not match its row denominator"
                )

        aggregate_observation = observed_by_id.get(observation_id)
        if aggregate_observation is None:
            raise VisualDataError(f"quality.observations lacks {observation_id}")
        if aggregate_observation.get("count") != count:
            raise VisualDataError(
                f"{observation_id} observation and metric counts disagree"
            )
        if aggregate_observation.get("row_rate_ppm") != rate:
            raise VisualDataError(
                f"{observation_id} observation and row rates disagree"
            )
        observations.append(
            ChartObservation(
                observation_id=observation_id,
                label=label,
                count=count,
                row_rate_ppm=rate,
            )
        )

    observations.sort(key=lambda item: (-item.count, item.label))
    return VisualEvidence(
        source_sha256=cast(str, source_sha256),
        profiled_rows=profiled_rows,
        observations=tuple(observations),
    )


def collect_evidence(root: Path | None = None) -> tuple[VisualEvidence, CommandCapture]:
    """Run both audit paths and return only verified aggregate evidence."""

    effective_root = repository_root() if root is None else root
    fresh_result = _run_repository_command(
        effective_root,
        FRESH_AUDIT_COMMAND,
    )
    if fresh_result.returncode != 0 or fresh_result.stderr:
        raise VisualFreshnessError(
            "fresh audit did not produce canonical stdout with exit code 0"
        )
    try:
        checked_bytes = _read_checked_manifest(effective_root / MANIFEST_RELATIVE_PATH)
    except CheckedManifestInputError as error:
        raise VisualFreshnessError(str(error)) from error
    if fresh_result.stdout != checked_bytes:
        raise VisualFreshnessError(
            "fresh audit bytes do not equal the checked data-quality manifest"
        )

    try:
        decoded = json.loads(checked_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VisualDataError("checked manifest is not valid UTF-8 JSON") from error
    manifest = _mapping(decoded, "manifest root")
    evidence = _evidence_from_manifest(manifest)

    check_result = _run_repository_command(
        effective_root,
        CHECK_AUDIT_COMMAND,
    )
    capture = CommandCapture(
        display_argv=check_result.display_argv,
        returncode=check_result.returncode,
        stdout=_normalize_command_stream(check_result.stdout, effective_root),
        stderr=_normalize_command_stream(check_result.stderr, effective_root),
    )
    expected_stderr = f"manifest PASS: {MANIFEST_RELATIVE_PATH.as_posix()}"
    if (
        capture.returncode != 0
        or capture.stdout != ""
        or capture.stderr != expected_stderr
    ):
        raise VisualFreshnessError(
            "checked-manifest command did not return the expected PASS evidence"
        )

    terminal_result = _run_repository_command(
        effective_root,
        FRESH_AUDIT_COMMAND,
    )
    if (
        terminal_result.returncode != 0
        or terminal_result.stderr
        or terminal_result.stdout != checked_bytes
    ):
        raise VisualFreshnessError(
            "source or manifest changed while visual evidence was collected"
        )
    return evidence, capture


def _escape(value: str) -> str:
    return html.escape(value, quote=True)


def _svg_document(
    *,
    width: int,
    height: int,
    title_id: str,
    title: str,
    description_id: str,
    description: str,
    body: Sequence[str],
) -> bytes:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" '
            f'fill="{INK}" role="img" '
            f'aria-labelledby="{title_id} {description_id}">'
        ),
        f'  <title id="{title_id}">{_escape(title)}</title>',
        f'  <desc id="{description_id}">{_escape(description)}</desc>',
        "  <style>",
        "    text { font-family: Arial, Helvetica, sans-serif; }",
        (
            "    .mono { font-family: ui-monospace, SFMono-Regular, "
            "Consolas, monospace; }"
        ),
        "  </style>",
        *body,
        "</svg>",
        "",
    ]
    payload = "\n".join(lines).encode("utf-8")
    if len(payload) > MAX_SVG_BYTES:
        raise VisualDataError("generated SVG exceeds the safety limit")
    return payload


def _format_integer(value: int) -> str:
    return f"{value:,}"


def _format_binary_limit(value: int) -> str:
    mebibyte = 1024 * 1024
    if value % mebibyte == 0:
        return f"{value // mebibyte} MiB"
    return f"{_format_integer(value)} bytes"


def _format_compact_limit(value: int) -> str:
    if value % 1_000 == 0:
        return f"{value // 1_000}k"
    return _format_integer(value)


def _format_percent(row_rate_ppm: int) -> str:
    hundredths = (row_rate_ppm + 50) // 100
    return f"{hundredths // 100}.{hundredths % 100:02d}%"


def _render_observation_chart(evidence: VisualEvidence) -> bytes:
    width = 1200
    height = 720
    plot_left = 420
    plot_right = 1060
    plot_top = 222
    bar_height = 48
    row_gap = 78
    maximum_count = max(item.count for item in evidence.observations)
    axis_max = ((maximum_count + 49) // 50) * 50
    plot_width = plot_right - plot_left

    body = [
        f'  <rect width="{width}" height="{height}" fill="{WHITE}"/>',
        (
            f'  <rect x="48" y="42" width="1104" height="636" rx="18" '
            f'fill="{WHITE}" stroke="{GRID}" stroke-width="2"/>'
        ),
        (f'  <rect x="72" y="68" width="8" height="54" rx="4" fill="{BLUE}"/>'),
        (
            '  <text x="100" y="92" font-size="28" font-weight="700">'
            "Row-derived data-quality observations</text>"
        ),
        (
            '  <text x="100" y="122" font-size="16" '
            f'fill="{MUTED}">Comparable counts over n = '
            f"{_format_integer(evidence.profiled_rows)} profiled rows; "
            "zero-baseline scale</text>"
        ),
        (
            f'  <rect x="890" y="76" width="224" height="34" rx="17" '
            f'fill="{GOLD_OPEN}" stroke="{GOLD}" stroke-width="1"/>'
        ),
        (
            f'  <text x="1002" y="98" text-anchor="middle" font-size="13" '
            f'font-weight="700" fill="{GOLD_DARK}">SOURCE-BOUND · PHASE 0</text>'
        ),
    ]

    tick_step = 50
    for tick in range(0, axis_max + tick_step, tick_step):
        if tick > axis_max:
            break
        x = plot_left + round(plot_width * tick / axis_max)
        stroke = INK if tick == 0 else GRID
        stroke_width = 2 if tick == 0 else 1
        body.append(
            f'  <line x1="{x}" y1="190" x2="{x}" y2="526" '
            f'stroke="{stroke}" stroke-width="{stroke_width}"/>'
        )
        body.append(
            f'  <text class="mono" x="{x}" y="177" text-anchor="middle" '
            f'font-size="13" fill="{MUTED}">{tick}</text>'
        )

    for index, observation in enumerate(evidence.observations):
        y = plot_top + index * row_gap
        bar_width = round(plot_width * observation.count / axis_max)
        label_y = y + 31
        body.extend(
            [
                (
                    f'  <text x="{plot_left - 22}" y="{label_y}" '
                    f'text-anchor="end" font-size="16" fill="{INK}">'
                    f"{_escape(observation.label)}</text>"
                ),
                (
                    f'  <rect x="{plot_left}" y="{y}" width="{bar_width}" '
                    f'height="{bar_height}" rx="5" fill="{BLUE}" '
                    f'stroke="{BLUE_DARK}" stroke-width="1"/>'
                ),
                (
                    f'  <text class="mono" x="{plot_left + bar_width + 14}" '
                    f'y="{label_y}" font-size="15" font-weight="700">'
                    f"{_format_integer(observation.count)} · "
                    f"{_format_percent(observation.row_rate_ppm)}</text>"
                ),
            ]
        )

    body.extend(
        [
            (
                f'  <line x1="72" y1="568" x2="1128" y2="568" '
                f'stroke="{GRID}" stroke-width="1"/>'
            ),
            (
                f'  <text x="72" y="602" font-size="14" fill="{MUTED}">'
                "Counts are aggregate missing or excess-row observations; "
                "categories can overlap.</text>"
            ),
            (
                f'  <text x="72" y="630" font-size="14" fill="{MUTED}">'
                "Source: train.csv · SHA-256 "
                f"{evidence.source_sha256[:12]}… · "
                f"{MANIFEST_SCHEMA} v{MANIFEST_SCHEMA_VERSION}"
                "</text>"
            ),
        ]
    )
    return _svg_document(
        width=width,
        height=height,
        title_id="chart-title",
        title="Row-derived data-quality observations",
        description_id="chart-description",
        description=(
            "A sorted horizontal zero-baseline bar chart comparing four "
            f"aggregate observations across {evidence.profiled_rows} rows."
        ),
        body=body,
    )


def _render_workflow(evidence: VisualEvidence) -> bytes:
    width = 1320
    height = 520
    parser_limits = (
        (
            "≤ "
            f"{_format_binary_limit(MAX_DATASET_BYTES)} · "
            f"≤ {_format_compact_limit(MAX_DATASET_ROWS)} rows"
        ),
        f"≤ {_format_integer(MAX_DATASET_COLUMNS)} columns",
        f"≤ {_format_integer(MAX_CSV_PHYSICAL_LINE_CHARACTERS)} chars / line",
        "one record / physical line",
    )
    frozen_schema_label = (
        f"frozen schema · {_format_integer(len(DEFAULT_CONTRACT.columns))} columns"
    )
    cards = (
        (
            "1",
            "Captured source bytes",
            ("regular file · no-follow", "unchanged-file identity", "train.csv"),
            BLUE_OPEN,
            BLUE,
        ),
        (
            "2",
            "Bounded CSV parser",
            parser_limits,
            BLUE_OPEN,
            BLUE,
        ),
        (
            "3",
            "Contract + metrics",
            (
                "identity · shape",
                frozen_schema_label,
                "completeness · grain",
            ),
            BLUE_OPEN,
            BLUE,
        ),
        (
            "4",
            "Canonical JSON",
            (
                "sorted keys · UTF-8",
                "LF · final newline",
                MANIFEST_RELATIVE_PATH.name,
            ),
            GOLD_OPEN,
            GOLD,
        ),
        (
            "5",
            "Freshness check",
            ("regenerate in memory", "fresh bytes == tracked", "otherwise fail closed"),
            GOLD_OPEN,
            GOLD,
        ),
    )
    card_width = 218
    card_height = 218
    gap = 28
    start_x = 58
    card_y = 190

    body = [
        f'  <rect width="{width}" height="{height}" fill="{WHITE}"/>',
        (
            f'  <rect x="34" y="34" width="1252" height="452" rx="18" '
            f'fill="{WHITE}" stroke="{GRID}" stroke-width="2"/>'
        ),
        (
            '  <text x="62" y="88" font-size="28" font-weight="700">'
            "Current phase-0 data-quality evidence workflow</text>"
        ),
        (
            f'  <text x="62" y="120" font-size="16" fill="{MUTED}">'
            "Implemented path from one captured source snapshot to checked, "
            "canonical aggregate evidence</text>"
        ),
        (
            f'  <rect x="1018" y="70" width="224" height="38" rx="19" '
            f'fill="{BLUE_OPEN}" stroke="{BLUE}" stroke-width="1"/>'
        ),
        (
            f'  <text x="1130" y="95" text-anchor="middle" font-size="13" '
            f'font-weight="700" fill="{BLUE_DARK}">CURRENT · IMPLEMENTED</text>'
        ),
    ]

    for index, (number, title, details, fill, stroke) in enumerate(cards):
        x = start_x + index * (card_width + gap)
        number_fill = WHITE if stroke == BLUE else INK
        if index > 0:
            arrow_start = x - gap + 4
            arrow_end = x - 8
            arrow_y = card_y + card_height // 2
            body.extend(
                [
                    (
                        f'  <line x1="{arrow_start}" y1="{arrow_y}" '
                        f'x2="{arrow_end}" y2="{arrow_y}" '
                        f'stroke="{MUTED}" stroke-width="2"/>'
                    ),
                    (
                        f'  <polygon points="{arrow_end},{arrow_y} '
                        f"{arrow_end - 8},{arrow_y - 6} "
                        f'{arrow_end - 8},{arrow_y + 6}" fill="{MUTED}"/>'
                    ),
                ]
            )
        body.extend(
            [
                (
                    f'  <rect x="{x}" y="{card_y}" width="{card_width}" '
                    f'height="{card_height}" rx="14" fill="{fill}" '
                    f'stroke="{stroke}" stroke-width="2"/>'
                ),
                (
                    f'  <circle cx="{x + 28}" cy="{card_y + 30}" r="16" '
                    f'fill="{stroke}"/>'
                ),
                (
                    f'  <text class="mono" x="{x + 28}" y="{card_y + 35}" '
                    f'text-anchor="middle" font-size="13" font-weight="700" '
                    f'fill="{number_fill}">{number}</text>'
                ),
                (
                    f'  <text x="{x + 18}" y="{card_y + 78}" font-size="17" '
                    f'font-weight="700">{_escape(title)}</text>'
                ),
            ]
        )
        for line_index, detail in enumerate(details):
            body.append(
                f'  <text x="{x + 18}" y="{card_y + 116 + line_index * 27}" '
                f'font-size="14" fill="{MUTED}">{_escape(detail)}</text>'
            )

    body.extend(
        [
            (
                f'  <text x="62" y="449" font-size="14" fill="{MUTED}">'
                f"Frozen input: {_format_integer(evidence.profiled_rows)} rows · "
                f"SHA-256 {evidence.source_sha256[:12]}…</text>"
            ),
            (
                f'  <text x="1258" y="449" text-anchor="end" font-size="14" '
                f'fill="{MUTED}">No map, model, API, or live data claim</text>'
            ),
        ]
    )
    return _svg_document(
        width=width,
        height=height,
        title_id="workflow-title",
        title="Current phase-0 data-quality evidence workflow",
        description_id="workflow-description",
        description=(
            "The implemented pipeline captures source bytes without following "
            "the final symlink, parses within fixed bounds, computes contract "
            "metrics, serializes canonical JSON, and checks byte freshness."
        ),
        body=body,
    )


def _render_cli_capture(
    evidence: VisualEvidence,
    capture: CommandCapture,
) -> bytes:
    width = 1200
    height = 560
    command = "$ " + shlex.join(capture.display_argv)
    stdout_text = capture.stdout or "(empty)"
    stderr_text = capture.stderr or "(empty)"

    terminal_lines = (
        ("command", command, BLUE),
        ("stdout", stdout_text, MUTED),
        ("stderr", stderr_text, INK),
    )
    body = [
        f'  <rect width="{width}" height="{height}" fill="{WHITE}"/>',
        (
            f'  <rect x="48" y="42" width="1104" height="476" rx="18" '
            f'fill="{WHITE}" stroke="{GRID}" stroke-width="2"/>'
        ),
        (
            '  <text x="78" y="91" font-size="28" font-weight="700">'
            "Reproducible audit check</text>"
        ),
        (
            f'  <text x="78" y="121" font-size="16" fill="{MUTED}">'
            "Actual subprocess result captured during generation; "
            "machine paths and timestamps are absent</text>"
        ),
        (
            f'  <rect x="78" y="154" width="1044" height="300" rx="12" '
            f'fill="{PANEL}" stroke="{GRID}" stroke-width="1"/>'
        ),
        (f'  <circle cx="104" cy="178" r="5" fill="{BLUE}"/>'),
        (f'  <circle cx="122" cy="178" r="5" fill="{GOLD}"/>'),
    ]

    y = 218
    for label, value, color in terminal_lines:
        body.append(
            f'  <text class="mono" x="104" y="{y}" font-size="13" '
            f'font-weight="700" fill="{MUTED}">{label}</text>'
        )
        y += 26
        for line in value.split("\n"):
            body.append(
                f'  <text class="mono" x="104" y="{y}" font-size="15" '
                f'fill="{color}">{_escape(line)}</text>'
            )
            y += 24
        y += 14

    body.extend(
        [
            (
                f'  <rect x="946" y="472" width="176" height="30" rx="15" '
                f'fill="{GOLD_OPEN}" stroke="{GOLD}" stroke-width="1"/>'
            ),
            (
                f'  <text class="mono" x="1034" y="492" text-anchor="middle" '
                f'font-size="13" font-weight="700" fill="{GOLD_DARK}">'
                f"EXIT CODE {capture.returncode}</text>"
            ),
            (
                f'  <text x="78" y="492" font-size="13" fill="{MUTED}">'
                f"Source SHA-256 {evidence.source_sha256[:12]}… · "
                "captured only after byte-for-byte freshness passed</text>"
            ),
        ]
    )
    return _svg_document(
        width=width,
        height=height,
        title_id="cli-title",
        title="Reproducible audit check",
        description_id="cli-description",
        description=(
            "A deterministic terminal-style rendering of the actual audit "
            "check command, its standard streams, and exit code zero."
        ),
        body=body,
    )


def generate_visuals(
    evidence: VisualEvidence,
    capture: CommandCapture,
) -> dict[str, bytes]:
    """Render every checked SVG fully in memory."""

    return {
        "data-quality-observations.svg": _render_observation_chart(evidence),
        "phase-0-workflow.svg": _render_workflow(evidence),
        "audit-cli-result.svg": _render_cli_capture(evidence, capture),
    }


def _open_visual_directory(path: Path) -> int:
    required_flags = ("O_DIRECTORY", "O_NOFOLLOW")
    if any(not hasattr(os, flag) for flag in required_flags):
        raise VisualOutputError(
            "platform does not support anchored no-follow visual directories"
        )
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    flags |= getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise VisualOutputError(
            f"cannot anchor visual directory {path}: {error}"
        ) from error
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise VisualOutputError("visual output parent is not a directory")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _validate_visual_targets(
    directory_fd: int,
    artifacts: Mapping[str, bytes],
) -> None:
    if tuple(artifacts) != VISUAL_FILENAMES:
        raise VisualOutputError("visual artifact set or order is unexpected")
    for name, payload in artifacts.items():
        if Path(name).name != name or name in {"", ".", ".."}:
            raise VisualOutputError("visual output names must be plain basenames")
        if len(payload) > MAX_SVG_BYTES:
            raise VisualOutputError(f"{name} exceeds the SVG safety limit")
        try:
            metadata = os.stat(
                name,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            continue
        except OSError as error:
            raise VisualOutputError(
                f"cannot inspect visual output {name}: {error}"
            ) from error
        if stat.S_ISLNK(metadata.st_mode):
            raise VisualOutputError(f"refusing to replace symlink output {name}")
        if not stat.S_ISREG(metadata.st_mode):
            raise VisualOutputError(f"visual output {name} is not a regular file")


def _create_visual_temporary(directory_fd: int) -> tuple[int, str]:
    if not hasattr(os, "O_NOFOLLOW"):
        raise VisualOutputError("platform does not support no-follow temp files")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    flags |= getattr(os, "O_CLOEXEC", 0)
    for _ in range(128):
        name = f".urbanlens-visual-{secrets.token_hex(16)}.tmp"
        try:
            descriptor = os.open(
                name,
                flags,
                0o600,
                dir_fd=directory_fd,
            )
        except FileExistsError:
            continue
        except OSError as error:
            raise VisualOutputError(
                f"cannot create temporary visual: {error}"
            ) from error
        return descriptor, name
    raise VisualOutputError("cannot allocate a unique temporary visual")


def _write_visuals_to_anchored_directory(
    directory_fd: int,
    artifacts: Mapping[str, bytes],
) -> None:
    """Stage and replace all outputs relative to one held directory."""

    _validate_visual_targets(directory_fd, artifacts)
    staged: dict[str, str] = {}
    try:
        for output_name, payload in artifacts.items():
            descriptor, temporary_name = _create_visual_temporary(directory_fd)
            staged[output_name] = temporary_name
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
                os.fchmod(stream.fileno(), 0o644)
        for output_name in artifacts:
            os.replace(
                staged[output_name],
                output_name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
            del staged[output_name]
        os.fsync(directory_fd)
    except VisualOutputError:
        raise
    except OSError as error:
        raise VisualOutputError(f"cannot write visual artifacts: {error}") from error
    finally:
        for temporary_name in staged.values():
            with suppress(OSError):
                os.unlink(temporary_name, dir_fd=directory_fd)


def write_visuals(output_directory: Path, artifacts: Mapping[str, bytes]) -> None:
    """Safely write generated artifacts without following the parent leaf."""

    directory_fd = _open_visual_directory(output_directory)
    try:
        _write_visuals_to_anchored_directory(directory_fd, artifacts)
    finally:
        os.close(directory_fd)


def _read_visual_at(
    directory_fd: int,
    name: str,
) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except OSError as error:
        raise VisualOutputError(
            f"cannot open checked visual {name}: {error}"
        ) from error
    try:
        with os.fdopen(descriptor, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise VisualOutputError(f"checked visual {name} is not a regular file")
            if before.st_size > MAX_SVG_BYTES:
                raise VisualOutputError(
                    f"checked visual {name} exceeds the SVG safety limit"
                )
            payload = stream.read(MAX_SVG_BYTES + 1)
            after = os.fstat(stream.fileno())
    except VisualOutputError:
        raise
    except OSError as error:
        raise VisualOutputError(
            f"cannot read checked visual {name}: {error}"
        ) from error
    identity_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_uid",
        "st_gid",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(
        getattr(before, field) != getattr(after, field) for field in identity_fields
    ):
        raise VisualOutputError(f"checked visual {name} changed while read")
    if len(payload) > MAX_SVG_BYTES:
        raise VisualOutputError(f"checked visual {name} exceeds the SVG safety limit")
    return payload


def stale_visuals(
    output_directory: Path,
    artifacts: Mapping[str, bytes],
) -> tuple[str, ...]:
    """Return missing or byte-stale artifact names after safe reads."""

    directory_fd = _open_visual_directory(output_directory)
    try:
        stale: list[str] = []
        for name, expected in artifacts.items():
            try:
                actual = _read_visual_at(directory_fd, name)
            except VisualOutputError:
                stale.append(name)
                continue
            if actual != expected:
                stale.append(name)
        return tuple(stale)
    finally:
        os.close(directory_fd)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Generate source-bound phase-0 SVG evidence after a fresh audit.")
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check",
        action="store_true",
        help="fail unless every checked SVG equals a fresh in-memory render",
    )
    mode.add_argument(
        "--write",
        action="store_true",
        help="atomically replace checked SVGs after the freshness gate",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    root = repository_root()
    try:
        evidence, capture = collect_evidence(root)
        artifacts = generate_visuals(evidence, capture)
    except (VisualFreshnessError, VisualDataError) as error:
        print(f"visual input FAIL: {error}", file=sys.stderr)
        return EXIT_STALE

    output_directory = root / VISUAL_DIRECTORY_RELATIVE_PATH
    if args.write:
        try:
            write_visuals(output_directory, artifacts)
        except VisualOutputError as error:
            print(f"visual output error: {error}", file=sys.stderr)
            return EXIT_OUTPUT_ERROR
        print(
            "visuals PASS: wrote "
            + ", ".join(
                (VISUAL_DIRECTORY_RELATIVE_PATH / name).as_posix()
                for name in VISUAL_FILENAMES
            ),
            file=sys.stderr,
        )
        return EXIT_SUCCESS

    try:
        stale = stale_visuals(output_directory, artifacts)
    except VisualOutputError as error:
        print(f"visual input error: {error}", file=sys.stderr)
        return EXIT_INPUT_ERROR
    if stale:
        print(
            "visuals FAIL: stale or missing " + ", ".join(stale),
            file=sys.stderr,
        )
        return EXIT_STALE
    print(
        f"visuals PASS: {len(VISUAL_FILENAMES)} source-bound SVGs are current",
        file=sys.stderr,
    )
    return EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
