"""Deterministic audit for the repository's immutable world-cities snapshot.

The module intentionally depends only on Python's standard library.  It
profiles the checked-in CSV, evaluates a small set of stable snapshot
invariants, and emits a canonical JSON manifest without timestamps or absolute
paths.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import secrets
import stat
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, BinaryIO, Final

from urbanlens import __version__

MANIFEST_SCHEMA: Final = "urbanlens.data-quality-manifest"
MANIFEST_SCHEMA_VERSION: Final = 2
RATE_SCALE: Final = 1_000_000
MAX_DATASET_BYTES: Final = 16 * 1024 * 1024
MAX_DATASET_ROWS: Final = 50_000
MAX_DATASET_COLUMNS: Final = 32
MAX_CSV_PHYSICAL_LINE_CHARACTERS: Final = 4_096
MAX_MANIFEST_BYTES: Final = 1024 * 1024

EXIT_SUCCESS: Final = 0
EXIT_CONTRACT_FAILED: Final = 1
EXIT_INPUT_ERROR: Final = 3
EXIT_OUTPUT_ERROR: Final = 4


@dataclass(frozen=True)
class SnapshotContract:
    """Stable invariants for the exact repository snapshot."""

    sha256: str
    byte_size: int
    data_rows: int
    columns: tuple[str, ...]
    capital_values: frozenset[str]


@dataclass(frozen=True, slots=True)
class DatasetCapture:
    """One bounded, immutable CSV snapshot captured from an open regular file."""

    source_name: str
    sha256: str
    byte_size: int
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    identity: os.stat_result = field(repr=False, compare=False)


@dataclass(frozen=True)
class AnchoredOutput:
    """Output basename held relative to one already-open directory."""

    display_path: Path
    directory_fd: int
    name: str


DEFAULT_CONTRACT: Final = SnapshotContract(
    sha256="de941def7faca87c0911abb79c3cbd07672887fd486a9b7bea6c48c12ce0cf18",
    byte_size=4_734_682,
    data_rows=44_691,
    columns=(
        "city",
        "city_ascii",
        "lat",
        "lng",
        "country",
        "iso2",
        "iso3",
        "admin_name",
        "capital",
        "population",
        "id",
    ),
    capital_values=frozenset({"", "admin", "minor", "primary"}),
)


class DatasetInputError(Exception):
    """Raised when the source cannot be safely or unambiguously audited."""


class ManifestOutputError(Exception):
    """Raised when an output target cannot be anchored or written safely."""


class CheckedManifestInputError(Exception):
    """Raised when a checked manifest cannot be captured safely."""


def _rate_ppm(count: int, denominator: int) -> int:
    """Return a deterministic nearest-integer parts-per-million rate."""

    if denominator <= 0:
        return 0
    return (count * RATE_SCALE + denominator // 2) // denominator


def _duplicate_metrics(
    values: Counter[tuple[str, ...]] | Counter[str],
) -> dict[str, int]:
    duplicate_counts = [count for count in values.values() if count > 1]
    return {
        "duplicate_group_count": len(duplicate_counts),
        "duplicate_row_excess_count": sum(count - 1 for count in duplicate_counts),
        "rows_in_duplicate_groups": sum(duplicate_counts),
    }


def _is_ascii_upper_alpha(value: str, length: int) -> bool:
    return (
        len(value) == length and value.isascii() and value.isalpha() and value.isupper()
    )


def _decimal_is_in_range(value: str, lower: Decimal, upper: Decimal) -> bool:
    try:
        parsed = Decimal(value)
    except InvalidOperation:
        return False
    return parsed.is_finite() and lower <= parsed <= upper


def _valid_population(value: str) -> bool:
    if value == "":
        return True
    try:
        parsed = Decimal(value)
    except InvalidOperation:
        return False
    return parsed.is_finite() and parsed >= 0


def _same_open_file(before: os.stat_result, after: os.stat_result) -> bool:
    return (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_uid,
        before.st_gid,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) == (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_uid,
        after.st_gid,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )


def _open_dataset(path: Path) -> BinaryIO:
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_NONBLOCK"):
        raise DatasetInputError(
            "platform does not support bounded no-follow dataset reads"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise DatasetInputError(f"cannot open dataset safely: {error}") from error
    return os.fdopen(descriptor, "rb")


def _capture_open_file(stream: BinaryIO) -> tuple[bytes, str]:
    """Capture and hash one bounded byte stream without a second live read."""

    digest = hashlib.sha256()
    chunks: list[bytes] = []
    captured_bytes = 0
    while chunk := stream.read(
        min(1024 * 1024, MAX_DATASET_BYTES + 1 - captured_bytes)
    ):
        captured_bytes += len(chunk)
        if captured_bytes > MAX_DATASET_BYTES:
            raise DatasetInputError(
                f"dataset exceeds the {MAX_DATASET_BYTES}-byte safety limit"
            )
        digest.update(chunk)
        chunks.append(chunk)
    return b"".join(chunks), digest.hexdigest()


def _read_rows(
    payload: bytes,
) -> tuple[list[str], list[list[str]]]:
    """Parse a resource-bounded, one-physical-line-per-record CSV."""

    captured_stream = io.BytesIO(payload)
    wrapper = io.TextIOWrapper(
        captured_stream,
        encoding="utf-8-sig",
        newline="",
    )
    header: list[str] | None = None
    rows: list[list[str]] = []
    try:
        for physical_line_number, physical_line in enumerate(wrapper, start=1):
            if len(physical_line) > MAX_CSV_PHYSICAL_LINE_CHARACTERS:
                raise DatasetInputError(
                    "CSV physical line "
                    f"{physical_line_number} exceeds the "
                    f"{MAX_CSV_PHYSICAL_LINE_CHARACTERS}-character safety limit"
                )
            try:
                parsed_records = list(csv.reader([physical_line], strict=True))
            except csv.Error as error:
                raise DatasetInputError(
                    "malformed CSV or multi-line record at physical line "
                    f"{physical_line_number}: {error}"
                ) from error
            if len(parsed_records) != 1:
                raise DatasetInputError(
                    "physical line "
                    f"{physical_line_number} did not produce one CSV record"
                )
            record = parsed_records[0]
            if len(record) > MAX_DATASET_COLUMNS:
                raise DatasetInputError(
                    "CSV record at physical line "
                    f"{physical_line_number} exceeds the "
                    f"{MAX_DATASET_COLUMNS}-column safety limit"
                )
            if header is None:
                header = record
                continue
            if len(rows) >= MAX_DATASET_ROWS:
                raise DatasetInputError(
                    f"dataset exceeds the {MAX_DATASET_ROWS}-row safety limit"
                )
            rows.append(record)
    except UnicodeDecodeError as error:
        raise DatasetInputError(
            f"dataset is not valid UTF-8 at byte {error.start}"
        ) from error
    finally:
        wrapper.detach()
    if header is None:
        raise DatasetInputError("dataset is empty and has no CSV header")
    return header, rows


def capture_dataset(dataset_path: Path) -> DatasetCapture:
    """Capture and parse one bounded, unchanged regular CSV file exactly once."""

    try:
        path_metadata = dataset_path.lstat()
    except OSError as error:
        raise DatasetInputError(f"cannot inspect dataset: {error}") from error
    if stat.S_ISLNK(path_metadata.st_mode):
        raise DatasetInputError("dataset must be a regular file, not a symlink")
    if not stat.S_ISREG(path_metadata.st_mode):
        raise DatasetInputError("dataset must be a regular file")

    try:
        with _open_dataset(dataset_path) as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise DatasetInputError("opened dataset is not a regular file")
            if before.st_size > MAX_DATASET_BYTES:
                raise DatasetInputError(
                    f"dataset exceeds the {MAX_DATASET_BYTES}-byte safety limit"
                )
            if (
                before.st_dev,
                before.st_ino,
            ) != (
                path_metadata.st_dev,
                path_metadata.st_ino,
            ):
                raise DatasetInputError("dataset path changed before it was opened")
            payload, digest = _capture_open_file(stream)
            after = os.fstat(stream.fileno())
    except DatasetInputError:
        raise
    except OSError as error:
        raise DatasetInputError(f"cannot read dataset: {error}") from error
    if not _same_open_file(before, after):
        raise DatasetInputError("dataset changed while it was being audited")

    header, rows = _read_rows(payload)
    return DatasetCapture(
        source_name=dataset_path.name,
        sha256=digest,
        byte_size=before.st_size,
        columns=tuple(header),
        rows=tuple(tuple(row) for row in rows),
        identity=before,
    )


def _check(check_id: str, actual: Any, expected: Any) -> dict[str, Any]:
    return {
        "actual": actual,
        "expected": expected,
        "id": check_id,
        "status": "pass" if actual == expected else "fail",
    }


def _missingness(
    blank_counts: dict[str, int], profiled_rows: int
) -> dict[str, dict[str, int]]:
    return {
        column: {
            "count": blank_counts[column],
            "row_rate_ppm": _rate_ppm(blank_counts[column], profiled_rows),
        }
        for column in sorted(blank_counts)
    }


def _observation(
    observation_id: str,
    *,
    count: int,
    profiled_rows: int | None,
    severity: str,
    interpretation: str,
) -> dict[str, Any]:
    observation: dict[str, Any] = {
        "count": count,
        "id": observation_id,
        "interpretation": interpretation,
        "severity": severity,
    }
    if profiled_rows is not None:
        observation["row_rate_ppm"] = _rate_ppm(count, profiled_rows)
    return observation


def _profile(
    header: Sequence[str],
    rows: Sequence[Sequence[str]],
    contract: SnapshotContract,
) -> dict[str, Any]:
    row_width_mismatch_count = sum(len(row) != len(header) for row in rows)
    header_positions: dict[str, int] = {}
    for column in contract.columns:
        matches = [index for index, value in enumerate(header) if value == column]
        if len(matches) == 1:
            header_positions[column] = matches[0]

    blank_counts: dict[str, int] = dict.fromkeys(contract.columns, 0)
    distinct_values: dict[str, set[str]] = {
        column: set() for column in contract.columns
    }
    id_counts: Counter[str] = Counter()
    composite_counts: Counter[tuple[str, ...]] = Counter()
    coordinate_counts: Counter[tuple[str, ...]] = Counter()
    exact_without_id_counts: Counter[tuple[str, ...]] = Counter()
    iso2_to_iso3: defaultdict[str, set[str]] = defaultdict(set)
    iso2_to_country: defaultdict[str, set[str]] = defaultdict(set)

    invalid_latitude_count = 0
    invalid_longitude_count = 0
    invalid_population_count = 0
    invalid_iso2_count = 0
    invalid_iso3_count = 0
    invalid_id_format_count = 0
    invalid_capital_count = 0
    non_ascii_city_ascii_count = 0
    profiled_rows = 0

    for row in rows:
        if len(row) != len(header):
            continue
        profiled_rows += 1
        record = {
            column: row[index].strip()
            if (index := header_positions.get(column)) is not None
            else ""
            for column in contract.columns
        }

        for column, value in record.items():
            if value == "":
                blank_counts[column] += 1
            distinct_values[column].add(value)

        identifier = record["id"]
        id_counts[identifier] += 1
        composite_counts[(record["city"], record["country"], record["admin_name"])] += 1
        coordinate_counts[(record["lat"], record["lng"])] += 1
        exact_without_id_counts[
            tuple(record[column] for column in contract.columns if column != "id")
        ] += 1

        if not _decimal_is_in_range(record["lat"], Decimal(-90), Decimal(90)):
            invalid_latitude_count += 1
        if not _decimal_is_in_range(record["lng"], Decimal(-180), Decimal(180)):
            invalid_longitude_count += 1
        if not _valid_population(record["population"]):
            invalid_population_count += 1
        if not _is_ascii_upper_alpha(record["iso2"], 2):
            invalid_iso2_count += 1
        if not _is_ascii_upper_alpha(record["iso3"], 3):
            invalid_iso3_count += 1
        if not (
            len(identifier) == 10 and identifier.isascii() and identifier.isdigit()
        ):
            invalid_id_format_count += 1
        if record["capital"] not in contract.capital_values:
            invalid_capital_count += 1
        if not record["city_ascii"].isascii():
            non_ascii_city_ascii_count += 1
        if record["iso2"]:
            iso2_to_iso3[record["iso2"]].add(record["iso3"])
            iso2_to_country[record["iso2"]].add(record["country"])

    id_duplicates = _duplicate_metrics(id_counts)
    composite_duplicates = _duplicate_metrics(composite_counts)
    coordinate_duplicates = _duplicate_metrics(coordinate_counts)
    exact_without_id_duplicates = _duplicate_metrics(exact_without_id_counts)

    iso2_to_iso3_conflicts = {
        key: values for key, values in iso2_to_iso3.items() if len(values) > 1
    }
    iso2_to_country_conflicts = {
        key: values for key, values in iso2_to_country.items() if len(values) > 1
    }

    return {
        "blank_counts": blank_counts,
        "distinct_counts": {
            column: len(values) for column, values in sorted(distinct_values.items())
        },
        "duplicates": {
            "city_country_admin": composite_duplicates,
            "coordinates": coordinate_duplicates,
            "exact_rows_excluding_id": exact_without_id_duplicates,
            "id": id_duplicates,
        },
        "mapping_consistency": {
            "iso2_to_country_conflicting_code_count": len(iso2_to_country_conflicts),
            "iso2_to_country_label_variant_excess_count": sum(
                len(values) - 1 for values in iso2_to_country_conflicts.values()
            ),
            "iso2_to_iso3_conflicting_code_count": len(iso2_to_iso3_conflicts),
        },
        "profiled_rows": profiled_rows,
        "row_width_mismatch_count": row_width_mismatch_count,
        "validity": {
            "invalid_capital_value_count": invalid_capital_count,
            "invalid_id_format_count": invalid_id_format_count,
            "invalid_iso2_format_count": invalid_iso2_count,
            "invalid_iso3_format_count": invalid_iso3_count,
            "invalid_latitude_count": invalid_latitude_count,
            "invalid_longitude_count": invalid_longitude_count,
            "invalid_population_count": invalid_population_count,
            "non_ascii_city_ascii_count": non_ascii_city_ascii_count,
        },
    }


def audit_capture(
    capture: DatasetCapture,
    *,
    contract: SnapshotContract = DEFAULT_CONTRACT,
) -> dict[str, Any]:
    """Audit one immutable capture without reopening its source path."""

    header = capture.columns
    rows = capture.rows
    profile = _profile(header, rows, contract)
    data_rows = len(rows)
    blank_counts = profile["blank_counts"]
    duplicates = profile["duplicates"]
    validity = profile["validity"]
    mapping_consistency = profile["mapping_consistency"]
    profiled_rows = profile["profiled_rows"]

    required_columns = ("city", "city_ascii", "country", "iso2", "iso3", "id")
    required_blank_counts = {
        column: blank_counts[column] for column in required_columns
    }
    checks = [
        _check("snapshot.sha256", capture.sha256, contract.sha256),
        _check("snapshot.byte_size", capture.byte_size, contract.byte_size),
        _check("snapshot.data_rows", data_rows, contract.data_rows),
        _check("schema.columns", list(header), list(contract.columns)),
        _check("schema.row_width_mismatches", profile["row_width_mismatch_count"], 0),
        _check(
            "required_fields.blank_counts",
            required_blank_counts,
            dict.fromkeys(required_columns, 0),
        ),
        _check(
            "id.duplicate_row_excess",
            duplicates["id"]["duplicate_row_excess_count"],
            0,
        ),
        _check("id.invalid_format", validity["invalid_id_format_count"], 0),
        _check(
            "coordinates.invalid_counts",
            {
                "latitude": validity["invalid_latitude_count"],
                "longitude": validity["invalid_longitude_count"],
            },
            {"latitude": 0, "longitude": 0},
        ),
        _check(
            "population.invalid_nonblank_values",
            validity["invalid_population_count"],
            0,
        ),
        _check(
            "iso.invalid_format_counts",
            {
                "iso2": validity["invalid_iso2_format_count"],
                "iso3": validity["invalid_iso3_format_count"],
            },
            {"iso2": 0, "iso3": 0},
        ),
        _check(
            "iso.iso2_to_iso3_conflicting_codes",
            mapping_consistency["iso2_to_iso3_conflicting_code_count"],
            0,
        ),
        _check(
            "capital.invalid_values",
            validity["invalid_capital_value_count"],
            0,
        ),
        _check(
            "city_ascii.non_ascii_values",
            validity["non_ascii_city_ascii_count"],
            0,
        ),
        _check(
            "rows.exact_duplicates_excluding_id",
            duplicates["exact_rows_excluding_id"]["duplicate_row_excess_count"],
            0,
        ),
    ]
    passed = sum(check["status"] == "pass" for check in checks)

    observations = [
        _observation(
            "admin_name.missing",
            count=blank_counts["admin_name"],
            profiled_rows=profiled_rows,
            severity="medium",
            interpretation=(
                "Administrative joins need an explicit missing-value policy."
            ),
        ),
        _observation(
            "capital.unmarked",
            count=blank_counts["capital"],
            profiled_rows=profiled_rows,
            severity="informational",
            interpretation=(
                "Blank means not marked as a capital; it is an allowed source value."
            ),
        ),
        _observation(
            "population.missing",
            count=blank_counts["population"],
            profiled_rows=profiled_rows,
            severity="medium",
            interpretation=(
                "Population modeling needs an explicit missing-value policy."
            ),
        ),
        _observation(
            "grain.repeated_city_country_admin",
            count=duplicates["city_country_admin"]["duplicate_row_excess_count"],
            profiled_rows=profiled_rows,
            severity="medium",
            interpretation=(
                "City, country, and admin labels are not a safe composite key."
            ),
        ),
        _observation(
            "grain.repeated_coordinates",
            count=duplicates["coordinates"]["duplicate_row_excess_count"],
            profiled_rows=profiled_rows,
            severity="medium",
            interpretation="Coordinates alone are not a safe row key.",
        ),
        _observation(
            "country.exact_label_variants_per_iso2",
            count=mapping_consistency["iso2_to_country_label_variant_excess_count"],
            profiled_rows=None,
            severity="low",
            interpretation=(
                "Exact country labels vary within an ISO2 and should be "
                "normalized deliberately before grouping."
            ),
        ),
    ]

    manifest = {
        "dataset": {
            "byte_size": capture.byte_size,
            "data_rows": data_rows,
            "encoding": "utf-8-sig",
            "name": capture.source_name,
            "sha256": capture.sha256,
        },
        "grain": {
            "candidate_key": ["id"],
            "description": "one source city record per source-provided id",
        },
        "manifest": {
            "canonicalization": (
                "UTF-8, sorted object keys, two-space indent, LF, final newline"
            ),
            "schema": MANIFEST_SCHEMA,
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "tool": "python3 -m urbanlens.audit",
            "tool_version": __version__,
        },
        "quality": {
            "contract_checks": checks,
            "metrics": {
                "distinct_counts": profile["distinct_counts"],
                "duplicates": duplicates,
                "mapping_consistency": mapping_consistency,
                "missingness": _missingness(blank_counts, profiled_rows),
                "profiled_rows": profile["profiled_rows"],
                "validity": validity,
            },
            "observations": observations,
            "status": "pass" if passed == len(checks) else "fail",
            "summary": {
                "failed": len(checks) - passed,
                "passed": passed,
                "total": len(checks),
            },
        },
        "schema": {
            "column_count": len(header),
            "columns": list(header),
            "row_width_mismatch_count": profile["row_width_mismatch_count"],
        },
        "scope": {
            "covers": (
                "snapshot identity, CSV shape, completeness, uniqueness, "
                "stable domains, and basic cross-field consistency"
            ),
            "does_not_establish": [
                "current population values",
                "geopolitical correctness",
                "authenticated upstream release provenance",
                "entity-resolution correctness",
            ],
        },
    }
    return manifest


def _audit_dataset_with_identity(
    dataset_path: Path,
    *,
    contract: SnapshotContract = DEFAULT_CONTRACT,
) -> tuple[dict[str, Any], os.stat_result]:
    """Audit one capture and return its manifest and held open-file identity."""

    capture = capture_dataset(dataset_path)
    return audit_capture(capture, contract=contract), capture.identity


def audit_dataset(
    dataset_path: Path,
    *,
    contract: SnapshotContract = DEFAULT_CONTRACT,
) -> dict[str, Any]:
    """Audit *dataset_path* and return a deterministic manifest object."""

    manifest, _ = _audit_dataset_with_identity(dataset_path, contract=contract)
    return manifest


def canonical_json_bytes(manifest: dict[str, Any]) -> bytes:
    """Serialize a manifest to the repository's canonical JSON form."""

    return (
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _anchor_output(path: Path) -> AnchoredOutput:
    """Open and hold the output parent without following its final component."""

    if path.name in {"", ".", ".."}:
        raise ManifestOutputError("output path must include a file name")
    required_flags = ("O_DIRECTORY", "O_NOFOLLOW")
    if any(not hasattr(os, flag) for flag in required_flags):
        raise ManifestOutputError(
            "platform does not support anchored no-follow output directories"
        )
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    flags |= getattr(os, "O_CLOEXEC", 0)
    try:
        directory_fd = os.open(path.parent, flags)
    except OSError as error:
        raise ManifestOutputError(
            f"cannot anchor output directory {path.parent}: {error}"
        ) from error
    try:
        if not stat.S_ISDIR(os.fstat(directory_fd).st_mode):
            raise ManifestOutputError("anchored output parent is not a directory")
    except BaseException:
        os.close(directory_fd)
        raise
    return AnchoredOutput(
        display_path=path,
        directory_fd=directory_fd,
        name=path.name,
    )


def _anchored_output_matches_dataset(
    output: AnchoredOutput,
    dataset_identity: os.stat_result,
) -> bool:
    """Check an existing anchored destination against the captured dataset."""

    try:
        output_identity = os.stat(
            output.name,
            dir_fd=output.directory_fd,
            follow_symlinks=True,
        )
    except FileNotFoundError:
        return False
    except OSError as error:
        raise ManifestOutputError(
            f"cannot inspect anchored output target: {error}"
        ) from error
    return (
        output_identity.st_dev,
        output_identity.st_ino,
    ) == (
        dataset_identity.st_dev,
        dataset_identity.st_ino,
    )


def _create_temporary_output(output: AnchoredOutput) -> tuple[int, str]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    flags |= getattr(os, "O_CLOEXEC", 0)
    for _ in range(128):
        temporary_name = f".urbanlens-{secrets.token_hex(16)}.tmp"
        try:
            descriptor = os.open(
                temporary_name,
                flags,
                0o600,
                dir_fd=output.directory_fd,
            )
        except FileExistsError:
            continue
        return descriptor, temporary_name
    raise ManifestOutputError("cannot allocate a unique temporary output file")


def _write_atomic(output: AnchoredOutput, payload: bytes) -> None:
    """Write and rename within the already-anchored output directory."""

    temporary_name: str | None = None
    try:
        descriptor, temporary_name = _create_temporary_output(output)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
            os.fchmod(stream.fileno(), 0o644)
        os.replace(
            temporary_name,
            output.name,
            src_dir_fd=output.directory_fd,
            dst_dir_fd=output.directory_fd,
        )
    except ManifestOutputError:
        raise
    except OSError as error:
        if temporary_name is not None:
            with suppress(OSError):
                os.unlink(temporary_name, dir_fd=output.directory_fd)
        raise ManifestOutputError(f"cannot write manifest: {error}") from error


def _read_checked_manifest(path: Path) -> bytes:
    """Capture one bounded, unchanged regular manifest without following links."""

    try:
        path_metadata = path.lstat()
    except OSError as error:
        raise CheckedManifestInputError(
            f"cannot inspect checked manifest: {error}"
        ) from error
    if stat.S_ISLNK(path_metadata.st_mode):
        raise CheckedManifestInputError(
            "checked manifest must be a regular file, not a symlink"
        )
    if not stat.S_ISREG(path_metadata.st_mode):
        raise CheckedManifestInputError("checked manifest must be a regular file")

    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_NONBLOCK"):
        raise CheckedManifestInputError(
            "platform does not support bounded no-follow manifest reads"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise CheckedManifestInputError(
            f"cannot open checked manifest safely: {error}"
        ) from error

    try:
        with os.fdopen(descriptor, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise CheckedManifestInputError(
                    "opened checked manifest is not a regular file"
                )
            if (
                before.st_dev,
                before.st_ino,
            ) != (
                path_metadata.st_dev,
                path_metadata.st_ino,
            ):
                raise CheckedManifestInputError(
                    "checked manifest path changed before it was opened"
                )
            if before.st_size > MAX_MANIFEST_BYTES:
                raise CheckedManifestInputError(
                    "checked manifest exceeds the "
                    f"{MAX_MANIFEST_BYTES}-byte safety limit"
                )

            chunks: list[bytes] = []
            captured_bytes = 0
            while chunk := stream.read(
                min(64 * 1024, MAX_MANIFEST_BYTES + 1 - captured_bytes)
            ):
                captured_bytes += len(chunk)
                if captured_bytes > MAX_MANIFEST_BYTES:
                    raise CheckedManifestInputError(
                        "checked manifest exceeds the "
                        f"{MAX_MANIFEST_BYTES}-byte safety limit"
                    )
                chunks.append(chunk)
            after = os.fstat(stream.fileno())
    except CheckedManifestInputError:
        raise
    except OSError as error:
        raise CheckedManifestInputError(
            f"cannot read checked manifest: {error}"
        ) from error
    if not _same_open_file(before, after):
        raise CheckedManifestInputError(
            "checked manifest changed while it was being read"
        )
    return b"".join(chunks)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Audit the immutable world-cities CSV and emit canonical JSON.")
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        type=Path,
        default=Path("train.csv"),
        help="CSV to audit (default: train.csv)",
    )
    destinations = parser.add_mutually_exclusive_group()
    destinations.add_argument(
        "--output",
        type=Path,
        help="atomically write the canonical manifest to this path",
    )
    destinations.add_argument(
        "--check-manifest",
        type=Path,
        help="require this file to equal the newly generated canonical bytes",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    dataset: Path = args.dataset
    output: Path | None = args.output
    check_manifest: Path | None = args.check_manifest

    anchored_output: AnchoredOutput | None = None
    if output is not None:
        try:
            anchored_output = _anchor_output(output)
        except ManifestOutputError as error:
            print(f"output error: {error}", file=sys.stderr)
            return EXIT_OUTPUT_ERROR

    try:
        try:
            manifest, dataset_identity = _audit_dataset_with_identity(dataset)
        except DatasetInputError as error:
            print(f"input error: {error}", file=sys.stderr)
            return EXIT_INPUT_ERROR

        payload = canonical_json_bytes(manifest)
        contract_passed = manifest["quality"]["status"] == "pass"

        if check_manifest is not None:
            try:
                expected_payload = _read_checked_manifest(check_manifest)
            except CheckedManifestInputError as error:
                print(f"input error: {error}", file=sys.stderr)
                return EXIT_INPUT_ERROR
            if expected_payload != payload:
                print(
                    f"manifest FAIL: {check_manifest} is stale or non-canonical",
                    file=sys.stderr,
                )
                return EXIT_CONTRACT_FAILED
            if not contract_passed:
                print(
                    "manifest matches, but the snapshot contract failed",
                    file=sys.stderr,
                )
                return EXIT_CONTRACT_FAILED
            print(f"manifest PASS: {check_manifest}", file=sys.stderr)
            return EXIT_SUCCESS

        if anchored_output is not None:
            try:
                if _anchored_output_matches_dataset(
                    anchored_output,
                    dataset_identity,
                ):
                    raise ManifestOutputError("refusing to overwrite the dataset")
                _write_atomic(anchored_output, payload)
            except ManifestOutputError as error:
                print(f"output error: {error}", file=sys.stderr)
                return EXIT_OUTPUT_ERROR
            status = "PASS" if contract_passed else "FAIL"
            print(
                f"audit {status}: {dataset} -> {anchored_output.display_path}",
                file=sys.stderr,
            )
        else:
            try:
                sys.stdout.buffer.write(payload)
            except OSError as error:
                print(f"output error: {error}", file=sys.stderr)
                return EXIT_OUTPUT_ERROR

        return EXIT_SUCCESS if contract_passed else EXIT_CONTRACT_FAILED
    finally:
        if anchored_output is not None:
            os.close(anchored_output.directory_fd)


if __name__ == "__main__":
    raise SystemExit(main())
