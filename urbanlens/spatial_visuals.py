"""Generate deterministic SVG evidence for UrbanLens phase-one spatial search.

Every render is gated by two real subprocesses: the checked spatial-evidence
command and the documented nearest-city command.  The latter's canonical
receipt is parsed, compared field-by-field with the checked evidence, and only
then reduced to the allowlisted fields shown in the terminal-style visual.

The renderer is standard-library only.  It emits fixed-size, self-contained
SVG bytes and writes them through a held no-follow directory descriptor.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import secrets
import shlex
import stat
import subprocess
import sys
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, NoReturn, cast

from urbanlens.audit import DEFAULT_CONTRACT
from urbanlens.nearest import (
    MAX_RECEIPT_BYTES,
    RECEIPT_SCHEMA,
    RECEIPT_SCHEMA_VERSION,
    canonical_receipt_bytes,
)
from urbanlens.spatial import INDEX_ALGORITHM, INDEX_VERSION
from urbanlens.spatial_evidence import (
    DEFAULT_ARTIFACT,
    EVIDENCE_SCHEMA,
    EVIDENCE_SCHEMA_VERSION,
    EXPECTED_TOPOLOGY_SHA256,
    MAX_EVIDENCE_BYTES,
    QUERY_CASES,
)

VISUAL_DIRECTORY_RELATIVE_PATH: Final = Path("docs/visuals")
EVIDENCE_ARTIFACT_RELATIVE_PATH: Final = DEFAULT_ARTIFACT
EVIDENCE_CHECK_COMMAND: Final = (
    "python3",
    "-m",
    "urbanlens.spatial_evidence",
    "--check",
)
NEAREST_COMMAND: Final = (
    "python3",
    "-m",
    "urbanlens.nearest",
    "--dataset",
    "train.csv",
    "--lat",
    "-11.1",
    "--lon",
    "-77.6",
    "--k",
    "3",
    "--verify",
)
VISUAL_FILENAMES: Final = (
    "spatial-index-architecture.svg",
    "spatial-work-reduction.svg",
    "nearest-city-cli-result.svg",
    "spatial-edge-case-proof.svg",
)
MAX_SVG_BYTES: Final = 512 * 1024
SUBPROCESS_TIMEOUT_SECONDS: Final = 60

EXIT_SUCCESS: Final = 0
EXIT_STALE: Final = 1
EXIT_ARGUMENT_ERROR: Final = 2
EXIT_INPUT_ERROR: Final = 3
EXIT_OUTPUT_ERROR: Final = 4

INK: Final = "#172033"
MUTED: Final = "#536174"
GRID: Final = "#D7E0EA"
PANEL: Final = "#F5F8FC"
WHITE: Final = "#FFFFFF"
NAVY: Final = "#173B5F"
BLUE: Final = "#2563A6"
BLUE_DARK: Final = "#184A7A"
BLUE_OPEN: Final = "#E8F1FA"
TEAL: Final = "#087F73"
TEAL_DARK: Final = "#075E57"
TEAL_OPEN: Final = "#E2F4F1"
GOLD: Final = "#B7791F"
GOLD_DARK: Final = "#6F4A0C"
GOLD_OPEN: Final = "#FBF1D8"
GREEN: Final = "#247A4D"
GREEN_DARK: Final = "#155A36"
GREEN_OPEN: Final = "#E7F5ED"
RED: Final = "#B43A3A"

_SHA256 = re.compile(r"[0-9a-f]{64}", re.ASCII)
_EXPECTED_EVIDENCE_PASS: Final = "spatial evidence PASS: tracked bytes are current\n"
_ARGUMENT_ERROR_MESSAGE: Final = "error: invalid visual command arguments\n"
_INPUT_ERROR_MESSAGE: Final = "spatial visuals FAIL: checked input was rejected\n"
_OUTPUT_ERROR_MESSAGE: Final = "spatial visuals FAIL: artifact operation failed\n"
_STALE_MESSAGE: Final = "spatial visuals FAIL: tracked SVG bytes are stale\n"
_PASS_MESSAGE: Final = "spatial visuals PASS: 4 source-bound SVGs are current\n"
_WRITE_MESSAGE: Final = "spatial visuals WRITE: 4 canonical SVGs replaced\n"


class SpatialVisualFreshnessError(RuntimeError):
    """Raised when live source evidence differs from the tracked artifact."""


class SpatialVisualDataError(RuntimeError):
    """Raised when checked evidence violates the visual input contract."""


class SpatialVisualOutputError(RuntimeError):
    """Raised when a visual artifact cannot be read or written safely."""


class _ArgumentInputError(Exception):
    """Raised without echoing an untrusted argument token."""


class _QuietArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        del message
        raise _ArgumentInputError


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Raw bounded output from one fixed repository-local command."""

    display_argv: tuple[str, ...]
    returncode: int
    stdout: bytes
    stderr: bytes


@dataclass(frozen=True, slots=True)
class ResultEvidence:
    """Allowlisted nearest-neighbour result used by the visuals."""

    rank: int
    source_id: str
    city_ascii: str
    country: str
    iso2: str
    latitude_e7: int
    longitude_e7: int
    distance_m: int


@dataclass(frozen=True, slots=True)
class CaseEvidence:
    """One checked real query and its exact traversal proof."""

    case_id: str
    purpose: str
    requested_latitude: str
    requested_longitude: str
    canonical_latitude: str
    canonical_longitude: str
    canonical_latitude_e7: int
    canonical_longitude_e7: int
    k: int
    evaluated: int
    pruned: int
    visited: int
    oracle_evaluated: int
    work_reduction_basis_points: int
    results: tuple[ResultEvidence, ...]


@dataclass(frozen=True, slots=True)
class SpatialVisualEvidence:
    """Validated source, index, aggregate, and query evidence."""

    source_sha256: str
    source_bytes: int
    source_rows: int
    audit_schema: str
    audit_schema_version: int
    index_algorithm: str
    index_version: int
    coordinate_scale: int
    earth_radius_millimeters: int
    index_records: int
    unique_numeric_locations: int
    collision_groups: int
    collision_excess: int
    max_location_multiplicity: int
    index_depth: int
    topology_sha256: str
    query_count: int
    accelerated_evaluations: int
    oracle_evaluations: int
    pruned_records: int
    visited_nodes: int
    work_reduction_basis_points: int
    cases: tuple[CaseEvidence, ...]

    def case(self, case_id: str) -> CaseEvidence:
        """Return one declared case or fail closed on a missing identifier."""

        for case in self.cases:
            if case.case_id == case_id:
                return case
        raise SpatialVisualDataError("required evidence case is absent")


@dataclass(frozen=True, slots=True)
class NearestCapture:
    """Validated actual CLI receipt plus safe display metadata."""

    display_argv: tuple[str, ...]
    returncode: int
    stdout_bytes: int
    receipt_schema: str
    receipt_schema_version: int
    query_latitude: str
    query_longitude: str
    evaluated: int
    pruned: int
    visited: int
    oracle_evaluated: int
    oracle_status: str
    results: tuple[ResultEvidence, ...]


def repository_root() -> Path:
    """Return the repository root without serializing it into any visual."""

    return Path(__file__).resolve().parents[1]


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if type(value) is not dict or not all(type(key) is str for key in value):
        raise SpatialVisualDataError(f"{path} must be an object")
    return cast(dict[str, Any], value)


def _array(value: Any, path: str) -> list[Any]:
    if type(value) is not list:
        raise SpatialVisualDataError(f"{path} must be an array")
    return value


def _string(value: Any, path: str) -> str:
    if type(value) is not str:
        raise SpatialVisualDataError(f"{path} must be a string")
    if any(ord(character) < 32 for character in value):
        raise SpatialVisualDataError(f"{path} contains a control character")
    return value


def _integer(value: Any, path: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise SpatialVisualDataError(
            f"{path} must be an integer greater than or equal to {minimum}"
        )
    return value


def _sha256(value: Any, path: str) -> str:
    text = _string(value, path)
    if _SHA256.fullmatch(text) is None:
        raise SpatialVisualDataError(f"{path} must be a lowercase SHA-256")
    return text


def _at(mapping: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = mapping
    traversed: list[str] = []
    for key in path:
        traversed.append(key)
        current = _mapping(
            current,
            ".".join(traversed[:-1]) or "root",
        ).get(key)
        if current is None:
            raise SpatialVisualDataError(f"{'.'.join(traversed)} is absent")
    return current


def _same_file_state(first: os.stat_result, second: os.stat_result) -> bool:
    fields = (
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
    return all(getattr(first, field) == getattr(second, field) for field in fields)


def _read_regular_file(path: Path, *, limit: int) -> bytes:
    """Read one unchanged bounded regular file without following its leaf."""

    if not isinstance(path, Path) or type(limit) is not int or limit < 1:
        raise TypeError("safe file read arguments are invalid")
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_NONBLOCK"):
        raise SpatialVisualOutputError("platform lacks safe no-follow reads")
    try:
        leaf = path.lstat()
    except OSError as error:
        raise SpatialVisualOutputError("checked input cannot be inspected") from error
    if not stat.S_ISREG(leaf.st_mode):
        raise SpatialVisualOutputError("checked input is not a regular file")
    if leaf.st_size > limit:
        raise SpatialVisualOutputError("checked input exceeds its byte limit")

    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    flags |= getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            before = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_dev != leaf.st_dev
                or before.st_ino != leaf.st_ino
                or before.st_size > limit
            ):
                raise SpatialVisualOutputError("checked input identity is unsafe")
            payload = stream.read(limit + 1)
            after = os.fstat(stream.fileno())
    except SpatialVisualOutputError:
        raise
    except OSError as error:
        raise SpatialVisualOutputError("checked input cannot be read safely") from error

    if len(payload) > limit:
        raise SpatialVisualOutputError("checked input exceeds its byte limit")
    if len(payload) != before.st_size or not _same_file_state(before, after):
        raise SpatialVisualOutputError("checked input changed while being read")
    return payload


def _run_repository_command(
    root: Path,
    display_argv: tuple[str, ...],
) -> CommandResult:
    """Execute one fixed Python command with a deterministic minimal environment."""

    if not isinstance(root, Path):
        raise TypeError("root must be a pathlib.Path")
    if not display_argv or display_argv[0] != "python3":
        raise SpatialVisualDataError("visual command must use python3")
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
            timeout=SUBPROCESS_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise SpatialVisualFreshnessError(
            "repository evidence command could not complete"
        ) from error
    if (
        len(completed.stdout) > MAX_EVIDENCE_BYTES
        or len(completed.stderr) > MAX_EVIDENCE_BYTES
    ):
        raise SpatialVisualFreshnessError("repository command output is oversized")
    return CommandResult(
        display_argv=display_argv,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _require_evidence_check(result: CommandResult) -> None:
    if (
        result.display_argv != EVIDENCE_CHECK_COMMAND
        or result.returncode != 0
        or result.stdout != b""
        or result.stderr != _EXPECTED_EVIDENCE_PASS.encode("ascii")
    ):
        raise SpatialVisualFreshnessError(
            "fresh evidence did not match the tracked canonical artifact"
        )


def _result_evidence(value: Any, path: str) -> ResultEvidence:
    result = _mapping(value, path)
    return ResultEvidence(
        rank=_integer(result.get("rank"), f"{path}.rank", minimum=1),
        source_id=_string(result.get("source_id"), f"{path}.source_id"),
        city_ascii=_string(result.get("city_ascii"), f"{path}.city_ascii"),
        country=_string(result.get("country"), f"{path}.country"),
        iso2=_string(result.get("iso2"), f"{path}.iso2"),
        latitude_e7=_integer_signed(
            result.get("latitude_e7"),
            f"{path}.latitude_e7",
        ),
        longitude_e7=_integer_signed(
            result.get("longitude_e7"),
            f"{path}.longitude_e7",
        ),
        distance_m=_integer(result.get("distance_m"), f"{path}.distance_m"),
    )


def _integer_signed(value: Any, path: str) -> int:
    if type(value) is not int:
        raise SpatialVisualDataError(f"{path} must be an integer")
    return value


def _case_evidence(value: Any, index: int) -> CaseEvidence:
    path = f"cases[{index}]"
    case = _mapping(value, path)
    query = _mapping(case.get("query"), f"{path}.query")
    requested = _mapping(query.get("requested"), f"{path}.query.requested")
    canonical = _mapping(query.get("canonical"), f"{path}.query.canonical")
    proof = _mapping(case.get("proof"), f"{path}.proof")
    accelerated = _mapping(proof.get("accelerated"), f"{path}.proof.accelerated")
    oracle = _mapping(proof.get("oracle"), f"{path}.proof.oracle")
    reduction = _mapping(
        proof.get("work_reduction"),
        f"{path}.proof.work_reduction",
    )
    results = tuple(
        _result_evidence(item, f"{path}.results[{result_index}]")
        for result_index, item in enumerate(
            _array(case.get("results"), f"{path}.results")
        )
    )
    case_evidence = CaseEvidence(
        case_id=_string(case.get("case_id"), f"{path}.case_id"),
        purpose=_string(case.get("purpose"), f"{path}.purpose"),
        requested_latitude=_string(
            requested.get("latitude"),
            f"{path}.query.requested.latitude",
        ),
        requested_longitude=_string(
            requested.get("longitude"),
            f"{path}.query.requested.longitude",
        ),
        canonical_latitude=_string(
            canonical.get("latitude"),
            f"{path}.query.canonical.latitude",
        ),
        canonical_longitude=_string(
            canonical.get("longitude"),
            f"{path}.query.canonical.longitude",
        ),
        canonical_latitude_e7=_integer_signed(
            canonical.get("latitude_e7"),
            f"{path}.query.canonical.latitude_e7",
        ),
        canonical_longitude_e7=_integer_signed(
            canonical.get("longitude_e7"),
            f"{path}.query.canonical.longitude_e7",
        ),
        k=_integer(query.get("k"), f"{path}.query.k", minimum=1),
        evaluated=_integer(
            accelerated.get("evaluated"),
            f"{path}.proof.accelerated.evaluated",
        ),
        pruned=_integer(
            accelerated.get("pruned"),
            f"{path}.proof.accelerated.pruned",
        ),
        visited=_integer(
            accelerated.get("visited"),
            f"{path}.proof.accelerated.visited",
        ),
        oracle_evaluated=_integer(
            oracle.get("evaluated"),
            f"{path}.proof.oracle.evaluated",
            minimum=1,
        ),
        work_reduction_basis_points=_integer(
            reduction.get("basis_points"),
            f"{path}.proof.work_reduction.basis_points",
        ),
        results=results,
    )
    if oracle.get("status") != "match":
        raise SpatialVisualDataError(f"{path} is not oracle-verified")
    if len(results) != case_evidence.k:
        raise SpatialVisualDataError(f"{path} result count differs from k")
    if tuple(result.rank for result in results) != tuple(range(1, case_evidence.k + 1)):
        raise SpatialVisualDataError(f"{path} result ranks are not contiguous")
    return case_evidence


def evidence_from_document(document: dict[str, Any]) -> SpatialVisualEvidence:
    """Validate the complete phase-one evidence document used for rendering."""

    metadata = _mapping(document.get("evidence"), "evidence")
    if metadata.get("schema") != EVIDENCE_SCHEMA:
        raise SpatialVisualDataError("evidence schema is not phase one")
    if metadata.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
        raise SpatialVisualDataError("evidence schema version is unsupported")
    if metadata.get("phase") != 1:
        raise SpatialVisualDataError("evidence phase is not one")

    source = _mapping(document.get("source"), "source")
    source_sha256 = _sha256(source.get("sha256"), "source.sha256")
    source_bytes = _integer(source.get("byte_size"), "source.byte_size", minimum=1)
    source_rows = _integer(source.get("data_rows"), "source.data_rows", minimum=1)
    if (
        source_sha256 != DEFAULT_CONTRACT.sha256
        or source_bytes != DEFAULT_CONTRACT.byte_size
        or source_rows != DEFAULT_CONTRACT.data_rows
    ):
        raise SpatialVisualDataError("evidence is not bound to the frozen source")

    index = _mapping(document.get("index"), "index")
    topology_sha256 = _sha256(index.get("topology_sha256"), "index.topology_sha256")
    if topology_sha256 != EXPECTED_TOPOLOGY_SHA256:
        raise SpatialVisualDataError("evidence topology is outside the contract")
    index_algorithm = _string(index.get("algorithm"), "index.algorithm")
    index_version = _integer(index.get("version"), "index.version", minimum=1)
    if index_algorithm != INDEX_ALGORITHM or index_version != INDEX_VERSION:
        raise SpatialVisualDataError("evidence index implementation is unsupported")

    cases = tuple(
        _case_evidence(item, index_value)
        for index_value, item in enumerate(_array(document.get("cases"), "cases"))
    )
    declared_case_ids = tuple(case.case_id for case in cases)
    expected_case_ids = tuple(case.case_id for case in QUERY_CASES)
    if declared_case_ids != expected_case_ids:
        raise SpatialVisualDataError("evidence query case set or order changed")
    for observed, declared in zip(cases, QUERY_CASES, strict=True):
        if (
            observed.purpose != declared.purpose
            or observed.requested_latitude != declared.latitude
            or observed.requested_longitude != declared.longitude
            or observed.k != declared.k
        ):
            raise SpatialVisualDataError("evidence query case definition changed")

    aggregate = _mapping(document.get("aggregate"), "aggregate")
    reduction = _mapping(
        aggregate.get("work_reduction"),
        "aggregate.work_reduction",
    )
    evidence = SpatialVisualEvidence(
        source_sha256=source_sha256,
        source_bytes=source_bytes,
        source_rows=source_rows,
        audit_schema=_string(source.get("audit_schema"), "source.audit_schema"),
        audit_schema_version=_integer(
            source.get("audit_schema_version"),
            "source.audit_schema_version",
            minimum=1,
        ),
        index_algorithm=index_algorithm,
        index_version=index_version,
        coordinate_scale=_integer(
            index.get("coordinate_scale"),
            "index.coordinate_scale",
            minimum=1,
        ),
        earth_radius_millimeters=_integer(
            index.get("earth_radius_millimeters"),
            "index.earth_radius_millimeters",
            minimum=1,
        ),
        index_records=_integer(index.get("records"), "index.records", minimum=1),
        unique_numeric_locations=_integer(
            index.get("unique_numeric_locations"),
            "index.unique_numeric_locations",
            minimum=1,
        ),
        collision_groups=_integer(
            index.get("collision_groups"),
            "index.collision_groups",
        ),
        collision_excess=_integer(
            index.get("collision_excess"),
            "index.collision_excess",
        ),
        max_location_multiplicity=_integer(
            index.get("max_location_multiplicity"),
            "index.max_location_multiplicity",
            minimum=1,
        ),
        index_depth=_integer(index.get("depth"), "index.depth", minimum=1),
        topology_sha256=topology_sha256,
        query_count=_integer(
            aggregate.get("query_count"),
            "aggregate.query_count",
            minimum=1,
        ),
        accelerated_evaluations=_integer(
            aggregate.get("accelerated_evaluations"),
            "aggregate.accelerated_evaluations",
        ),
        oracle_evaluations=_integer(
            aggregate.get("oracle_evaluations"),
            "aggregate.oracle_evaluations",
            minimum=1,
        ),
        pruned_records=_integer(
            aggregate.get("pruned_records"),
            "aggregate.pruned_records",
        ),
        visited_nodes=_integer(
            aggregate.get("visited_nodes"),
            "aggregate.visited_nodes",
        ),
        work_reduction_basis_points=_integer(
            reduction.get("basis_points"),
            "aggregate.work_reduction.basis_points",
        ),
        cases=cases,
    )
    _validate_evidence_accounting(evidence)
    return evidence


def _validate_evidence_accounting(evidence: SpatialVisualEvidence) -> None:
    if evidence.index_records != evidence.source_rows:
        raise SpatialVisualDataError("index and source row counts differ")
    if evidence.query_count != len(evidence.cases):
        raise SpatialVisualDataError("aggregate query count is inconsistent")
    if (
        evidence.accelerated_evaluations
        != sum(case.evaluated for case in evidence.cases)
        or evidence.oracle_evaluations
        != sum(case.oracle_evaluated for case in evidence.cases)
        or evidence.pruned_records != sum(case.pruned for case in evidence.cases)
        or evidence.visited_nodes != sum(case.visited for case in evidence.cases)
    ):
        raise SpatialVisualDataError("aggregate query accounting is inconsistent")
    if (
        evidence.accelerated_evaluations + evidence.pruned_records
        != evidence.oracle_evaluations
        or evidence.oracle_evaluations != evidence.index_records * evidence.query_count
    ):
        raise SpatialVisualDataError("aggregate full-scan denominator is inconsistent")
    for case in evidence.cases:
        if (
            case.evaluated + case.pruned != evidence.index_records
            or case.oracle_evaluated != evidence.index_records
        ):
            raise SpatialVisualDataError("case traversal accounting is inconsistent")


def _parse_document(payload: bytes, label: str) -> dict[str, Any]:
    try:
        decoded = payload.decode("ascii", errors="strict")
        document = json.loads(decoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SpatialVisualDataError(f"{label} is not canonical ASCII JSON") from error
    return _mapping(document, label)


def _receipt_result(value: Any, path: str) -> ResultEvidence:
    return _result_evidence(value, path)


def _capture_from_receipt(
    result: CommandResult,
    evidence: SpatialVisualEvidence,
) -> NearestCapture:
    """Validate actual canonical receipt bytes against checked query evidence."""

    if result.display_argv != NEAREST_COMMAND:
        raise SpatialVisualDataError("nearest command differs from the visual contract")
    if result.returncode != 0 or result.stderr:
        raise SpatialVisualFreshnessError("nearest command did not exit cleanly")
    if len(result.stdout) > MAX_RECEIPT_BYTES:
        raise SpatialVisualDataError("nearest receipt exceeds its output cap")
    receipt = _parse_document(result.stdout, "receipt")
    try:
        canonical = canonical_receipt_bytes(receipt)
    except Exception as error:
        raise SpatialVisualDataError(
            "nearest receipt cannot be canonicalized"
        ) from error
    if canonical != result.stdout:
        raise SpatialVisualDataError("nearest receipt bytes are not canonical")

    receipt_metadata = _mapping(receipt.get("receipt"), "receipt.receipt")
    if (
        receipt_metadata.get("schema") != RECEIPT_SCHEMA
        or receipt_metadata.get("schema_version") != RECEIPT_SCHEMA_VERSION
    ):
        raise SpatialVisualDataError("nearest receipt schema is unsupported")

    source = _mapping(receipt.get("source"), "receipt.source")
    if (
        source.get("sha256") != evidence.source_sha256
        or source.get("byte_size") != evidence.source_bytes
        or source.get("data_rows") != evidence.source_rows
        or source.get("audit_schema") != evidence.audit_schema
        or source.get("audit_schema_version") != evidence.audit_schema_version
    ):
        raise SpatialVisualDataError("nearest receipt source differs from evidence")

    index = _mapping(receipt.get("index"), "receipt.index")
    expected_index = {
        "algorithm": evidence.index_algorithm,
        "collision_excess": evidence.collision_excess,
        "collision_groups": evidence.collision_groups,
        "coordinate_scale": evidence.coordinate_scale,
        "depth": evidence.index_depth,
        "max_location_multiplicity": evidence.max_location_multiplicity,
        "records": evidence.index_records,
        "topology_sha256": evidence.topology_sha256,
        "unique_numeric_locations": evidence.unique_numeric_locations,
        "version": evidence.index_version,
    }
    for field, expected in expected_index.items():
        if index.get(field) != expected:
            raise SpatialVisualDataError(
                "nearest receipt index differs from checked evidence"
            )
    earth_radius = index.get("earth_radius_meters")
    if (
        type(earth_radius) is not float
        or round(earth_radius * 1_000) != evidence.earth_radius_millimeters
    ):
        raise SpatialVisualDataError("nearest receipt Earth radius is inconsistent")

    case = evidence.case("co-location")
    query = _mapping(receipt.get("query"), "receipt.query")
    expected_query = {
        "k": case.k,
        "latitude": case.canonical_latitude,
        "latitude_e7": case.canonical_latitude_e7,
        "longitude": case.canonical_longitude,
        "longitude_e7": case.canonical_longitude_e7,
    }
    if any(query.get(field) != expected for field, expected in expected_query.items()):
        raise SpatialVisualDataError("nearest receipt query differs from evidence")

    proof = _mapping(receipt.get("proof"), "receipt.proof")
    accelerated = _mapping(proof.get("accelerated"), "receipt.proof.accelerated")
    oracle = _mapping(proof.get("oracle"), "receipt.proof.oracle")
    expected_accelerated = {
        "evaluated": case.evaluated,
        "pruned": case.pruned,
        "visited": case.visited,
    }
    if any(
        accelerated.get(field) != expected
        for field, expected in expected_accelerated.items()
    ):
        raise SpatialVisualDataError("nearest receipt proof differs from evidence")
    if (
        oracle.get("status") != "match"
        or oracle.get("evaluations") != case.oracle_evaluated
    ):
        raise SpatialVisualDataError(
            "nearest receipt oracle proof differs from evidence"
        )

    results = tuple(
        _receipt_result(item, f"receipt.results[{index_value}]")
        for index_value, item in enumerate(
            _array(receipt.get("results"), "receipt.results")
        )
    )
    if results != case.results:
        raise SpatialVisualDataError("nearest receipt results differ from evidence")
    return NearestCapture(
        display_argv=result.display_argv,
        returncode=result.returncode,
        stdout_bytes=len(result.stdout),
        receipt_schema=RECEIPT_SCHEMA,
        receipt_schema_version=RECEIPT_SCHEMA_VERSION,
        query_latitude=case.canonical_latitude,
        query_longitude=case.canonical_longitude,
        evaluated=case.evaluated,
        pruned=case.pruned,
        visited=case.visited,
        oracle_evaluated=case.oracle_evaluated,
        oracle_status="match",
        results=results,
    )


def collect_evidence(
    root: Path | None = None,
) -> tuple[SpatialVisualEvidence, NearestCapture]:
    """Collect fresh checked evidence and one real source-bound CLI capture."""

    effective_root = repository_root() if root is None else root
    if not isinstance(effective_root, Path):
        raise TypeError("root must be a pathlib.Path")

    first_check = _run_repository_command(
        effective_root,
        EVIDENCE_CHECK_COMMAND,
    )
    _require_evidence_check(first_check)
    artifact_path = effective_root / EVIDENCE_ARTIFACT_RELATIVE_PATH
    artifact_bytes = _read_regular_file(artifact_path, limit=MAX_EVIDENCE_BYTES)
    evidence = evidence_from_document(
        _parse_document(artifact_bytes, "spatial evidence")
    )

    command_result = _run_repository_command(effective_root, NEAREST_COMMAND)
    capture = _capture_from_receipt(command_result, evidence)

    second_check = _run_repository_command(
        effective_root,
        EVIDENCE_CHECK_COMMAND,
    )
    _require_evidence_check(second_check)
    if _read_regular_file(artifact_path, limit=MAX_EVIDENCE_BYTES) != artifact_bytes:
        raise SpatialVisualFreshnessError(
            "tracked evidence changed during visual collection"
        )
    return evidence, capture


def _escape(value: str) -> str:
    return html.escape(value, quote=True)


def _format_integer(value: int) -> str:
    return f"{value:,}"


def _format_percent_from_basis_points(value: int) -> str:
    if not 0 <= value <= 10_000:
        raise SpatialVisualDataError("basis-point value is outside [0, 10000]")
    return f"{value // 100}.{value % 100:02d}%"


def _format_megabytes(value: int) -> str:
    whole, remainder = divmod(value, 1_000_000)
    if remainder == 0:
        return f"{whole} MB"
    return f"{whole}.{remainder // 100_000} MB"


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
        ("    text { font-family: Arial, Helvetica, Liberation Sans, sans-serif; }"),
        (
            "    .mono { font-family: ui-monospace, SFMono-Regular, "
            "Consolas, Liberation Mono, monospace; }"
        ),
        "  </style>",
        *body,
        "</svg>",
        "",
    ]
    payload = "\n".join(lines).encode("utf-8")
    if len(payload) > MAX_SVG_BYTES:
        raise SpatialVisualDataError("generated SVG exceeds the byte limit")
    return payload


def _render_architecture(evidence: SpatialVisualEvidence) -> bytes:
    width = 1500
    height = 860
    card_width = 218
    card_height = 396
    card_y = 242
    gap = 18
    start_x = 45
    cards = (
        (
            "01",
            "Single capture + audit",
            (
                "regular file · no-follow",
                "same bytes: audit + rows",
                f"manifest v{evidence.audit_schema_version} · PASS",
                f"{_format_integer(evidence.source_rows)} source rows",
            ),
            BLUE_OPEN,
            BLUE,
        ),
        (
            "02",
            "Strict E7 boundary",
            (
                "ASCII decimal · ≤ 7 dp",
                "signed zero → 0",
                "+180 → -180 · poles → 0",
                f"scale {_format_integer(evidence.coordinate_scale)}",
            ),
            BLUE_OPEN,
            BLUE,
        ),
        (
            "03",
            "Balanced sphere kd-tree",
            (
                "3D unit-sphere vectors",
                "greatest-span axis",
                "lower median · stable ID",
                (
                    f"depth {evidence.index_depth} · "
                    f"{_format_integer(evidence.index_records)} nodes"
                ),
            ),
            TEAL_OPEN,
            TEAL,
        ),
        (
            "04",
            "Conservative AABB",
            (
                "down-rounded lower bound",
                "prune only bound > kth",
                "raw chord² + source_id",
                "exact · worst case O(n)",
            ),
            TEAL_OPEN,
            TEAL,
        ),
        (
            "05",
            "Optional full-scan oracle",
            (
                f"scan {_format_integer(evidence.index_records)} records",
                "ordered rank-key proof",
                "--verify → match",
                "mismatch → fail closed",
            ),
            GOLD_OPEN,
            GOLD,
        ),
        (
            "06",
            "Canonical JSON receipt",
            (
                f"schema v{RECEIPT_SCHEMA_VERSION} · sorted keys",
                "ASCII · LF · final newline",
                f"≤ {MAX_RECEIPT_BYTES // 1024} KiB before write",
                "no path · host · time",
            ),
            GREEN_OPEN,
            GREEN,
        ),
    )

    body = [
        f'  <rect width="{width}" height="{height}" fill="{WHITE}"/>',
        (
            f'  <rect x="28" y="28" width="1444" height="804" rx="22" '
            f'fill="{WHITE}" stroke="{GRID}" stroke-width="2"/>'
        ),
        (
            '  <text x="58" y="82" font-size="30" font-weight="700">'
            "Implemented exact nearest-city architecture</text>"
        ),
        (
            f'  <text x="58" y="116" font-size="16" fill="{MUTED}">'
            "One frozen byte capture flows through audited normalization, "
            "exact acceleration, optional proof, and a bounded receipt</text>"
        ),
        (
            f'  <rect x="1168" y="62" width="260" height="38" rx="19" '
            f'fill="{GREEN_OPEN}" stroke="{GREEN}" stroke-width="1"/>'
        ),
        (
            f'  <text x="1298" y="87" text-anchor="middle" font-size="13" '
            f'font-weight="700" fill="{GREEN_DARK}">CURRENT · IMPLEMENTED</text>'
        ),
        (
            f'  <rect x="58" y="150" width="1370" height="58" rx="12" '
            f'fill="{PANEL}" stroke="{GRID}" stroke-width="1"/>'
        ),
        (
            f'  <text class="mono" x="82" y="176" font-size="13" '
            f'font-weight="700" fill="{NAVY}">SOURCE</text>'
        ),
        (
            f'  <text x="152" y="176" font-size="14" fill="{INK}">'
            f"train.csv · {_format_integer(evidence.source_rows)} rows · "
            f"{_format_integer(evidence.source_bytes)} bytes</text>"
        ),
        (
            f'  <text class="mono" x="82" y="197" font-size="12" '
            f'fill="{MUTED}">SHA-256 {_escape(evidence.source_sha256)}</text>'
        ),
        (
            f'  <text class="mono" x="1404" y="184" text-anchor="end" '
            f'font-size="12" fill="{MUTED}">'
            f"{_escape(evidence.audit_schema)} v{evidence.audit_schema_version}"
            "</text>"
        ),
    ]

    for index_value, (number, title, details, fill, stroke) in enumerate(cards):
        x = start_x + index_value * (card_width + gap)
        badge_fill = GOLD_DARK if stroke == GOLD else stroke
        if index_value > 0:
            arrow_y = card_y + card_height // 2
            arrow_start = x - gap + 2
            arrow_end = x - 5
            body.extend(
                [
                    (
                        f'  <line x1="{arrow_start}" y1="{arrow_y}" '
                        f'x2="{arrow_end}" y2="{arrow_y}" stroke="{MUTED}" '
                        'stroke-width="2"/>'
                    ),
                    (
                        f'  <polygon points="{arrow_end},{arrow_y} '
                        f"{arrow_end - 7},{arrow_y - 5} "
                        f'{arrow_end - 7},{arrow_y + 5}" fill="{MUTED}"/>'
                    ),
                ]
            )
        body.extend(
            [
                (
                    f'  <rect x="{x}" y="{card_y}" width="{card_width}" '
                    f'height="{card_height}" rx="15" fill="{fill}" '
                    f'stroke="{stroke}" stroke-width="2"/>'
                ),
                (
                    f'  <rect x="{x + 18}" y="{card_y + 20}" width="46" '
                    f'height="28" rx="14" fill="{badge_fill}"/>'
                ),
                (
                    f'  <text class="mono" x="{x + 41}" y="{card_y + 39}" '
                    f'text-anchor="middle" font-size="12" font-weight="700" '
                    f'fill="{WHITE}">{number}</text>'
                ),
                (
                    f'  <text x="{x + 18}" y="{card_y + 88}" font-size="14" '
                    f'font-weight="700">{_escape(title)}</text>'
                ),
                (
                    f'  <line x1="{x + 18}" y1="{card_y + 109}" '
                    f'x2="{x + card_width - 18}" y2="{card_y + 109}" '
                    f'stroke="{stroke}" stroke-opacity="0.35"/>'
                ),
            ]
        )
        for line_index, detail in enumerate(details):
            body.extend(
                [
                    (
                        f'  <circle cx="{x + 23}" '
                        f'cy="{card_y + 146 + line_index * 48}" r="3" '
                        f'fill="{stroke}"/>'
                    ),
                    (
                        f'  <text x="{x + 34}" '
                        f'y="{card_y + 151 + line_index * 48}" '
                        f'font-size="13" fill="{INK}">{_escape(detail)}</text>'
                    ),
                ]
            )
        footer = (
            "audit + capture"
            if index_value == 0
            else (
                "integer boundary"
                if index_value == 1
                else (
                    "unit-sphere kd-tree"
                    if index_value == 2
                    else (
                        "strict > pruning"
                        if index_value == 3
                        else (
                            "explicit --verify"
                            if index_value == 4
                            else f"receipt schema v{RECEIPT_SCHEMA_VERSION}"
                        )
                    )
                )
            )
        )
        body.append(
            f'  <text class="mono" x="{x + 18}" y="{card_y + 368}" '
            f'font-size="10" fill="{MUTED}">{_escape(footer)}</text>'
        )

    body.extend(
        [
            (
                f'  <rect x="58" y="684" width="1370" height="112" rx="12" '
                f'fill="{PANEL}" stroke="{GRID}" stroke-width="1"/>'
            ),
            (
                f'  <text x="82" y="716" font-size="14" font-weight="700" '
                f'fill="{NAVY}">INDEX IDENTITY</text>'
            ),
            (
                f'  <text class="mono" x="1404" y="716" text-anchor="end" '
                f'font-size="11" fill="{MUTED}">'
                f"{_escape(evidence.index_algorithm)} "
                f"v{evidence.index_version}</text>"
            ),
            (
                f'  <text x="82" y="746" font-size="14" fill="{INK}">'
                f"{_format_integer(evidence.unique_numeric_locations)} unique "
                f"numeric locations · {_format_integer(evidence.collision_groups)} "
                f"collision groups · {_format_integer(evidence.collision_excess)} "
                f"excess records · max multiplicity "
                f"{evidence.max_location_multiplicity}</text>"
            ),
            (
                f'  <text class="mono" x="82" y="774" font-size="12" '
                f'fill="{MUTED}">topology SHA-256 '
                f"{_escape(evidence.topology_sha256)}</text>"
            ),
            (
                f'  <text x="1404" y="748" text-anchor="end" font-size="13" '
                f'fill="{MUTED}">mean sphere radius '
                f"{evidence.earth_radius_millimeters / 1000:,.1f} m</text>"
            ),
            (
                f'  <text x="1404" y="774" text-anchor="end" font-size="13" '
                f'fill="{MUTED}">exact results; acceleration is not an '
                "approximation</text>"
            ),
        ]
    )
    return _svg_document(
        width=width,
        height=height,
        title_id="architecture-title",
        title="Implemented exact nearest-city architecture",
        description_id="architecture-description",
        description=(
            "A six-stage implemented workflow from one audited source capture "
            "through strict E7 parsing, a balanced unit-sphere kd-tree, "
            "conservative AABB pruning, an optional full-scan oracle, and a "
            "bounded canonical receipt."
        ),
        body=body,
    )


def _render_work_reduction(evidence: SpatialVisualEvidence) -> bytes:
    width = 1400
    height = 860
    plot_left = 360
    plot_width = 890
    row_y = 356
    row_gap = 92
    bar_height = 28

    body = [
        f'  <rect width="{width}" height="{height}" fill="{WHITE}"/>',
        (
            f'  <rect x="34" y="32" width="1332" height="796" rx="22" '
            f'fill="{WHITE}" stroke="{GRID}" stroke-width="2"/>'
        ),
        (
            '  <text x="66" y="86" font-size="30" font-weight="700">'
            "Exact query work on four real source-bound cases</text>"
        ),
        (
            f'  <text x="66" y="120" font-size="16" fill="{MUTED}">'
            "Traversal accounting from the checked artifact; full-scan results "
            "matched for every ordered top-3 result</text>"
        ),
        (
            f'  <rect x="66" y="158" width="300" height="122" rx="14" '
            f'fill="{BLUE_OPEN}" stroke="{BLUE}" stroke-width="1"/>'
        ),
        (
            f'  <text x="88" y="188" font-size="13" font-weight="700" '
            f'fill="{BLUE_DARK}">KD-TREE EVALUATIONS</text>'
        ),
        (
            f'  <text class="mono" x="88" y="232" font-size="30" '
            f'font-weight="700">{_format_integer(evidence.accelerated_evaluations)}'
            "</text>"
        ),
        (
            f'  <text x="88" y="258" font-size="13" fill="{MUTED}">'
            f"{_format_integer(evidence.visited_nodes)} nodes visited across "
            f"{evidence.query_count} queries</text>"
        ),
        (
            f'  <rect x="388" y="158" width="300" height="122" rx="14" '
            f'fill="{PANEL}" stroke="{GRID}" stroke-width="1"/>'
        ),
        (
            f'  <text x="410" y="188" font-size="13" font-weight="700" '
            f'fill="{NAVY}">FULL-SCAN EVALUATIONS</text>'
        ),
        (
            f'  <text class="mono" x="410" y="232" font-size="30" '
            f'font-weight="700">{_format_integer(evidence.oracle_evaluations)}'
            "</text>"
        ),
        (
            f'  <text x="410" y="258" font-size="13" fill="{MUTED}">'
            f"{_format_integer(evidence.index_records)} records x "
            f"{evidence.query_count} exact oracle queries</text>"
        ),
        (
            f'  <rect x="710" y="158" width="624" height="122" rx="14" '
            f'fill="{GREEN_OPEN}" stroke="{GREEN}" stroke-width="1"/>'
        ),
        (
            f'  <text x="734" y="188" font-size="13" font-weight="700" '
            f'fill="{GREEN_DARK}">AVOIDED RECORD EVALUATIONS</text>'
        ),
        (
            f'  <text class="mono" x="734" y="232" font-size="30" '
            f'font-weight="700">{_format_integer(evidence.pruned_records)}'
            "</text>"
        ),
        (
            f'  <text x="1068" y="232" font-size="30" font-weight="700" '
            f'fill="{GREEN_DARK}">'
            f"{_format_percent_from_basis_points(evidence.work_reduction_basis_points)}"
            "</text>"
        ),
        (
            f'  <text x="734" y="258" font-size="13" fill="{MUTED}">'
            "Structural work reduction; this is not a latency benchmark</text>"
        ),
        (
            f'  <text x="66" y="325" font-size="13" font-weight="700" '
            f'fill="{MUTED}">QUERY CASE</text>'
        ),
        (
            f'  <text x="{plot_left}" y="325" font-size="13" font-weight="700" '
            f'fill="{MUTED}">RECORD ACCOUNTING · each row totals '
            f"{_format_integer(evidence.index_records)}</text>"
        ),
        (f'  <rect x="1120" y="305" width="14" height="14" rx="2" fill="{BLUE}"/>'),
        (f'  <text x="1142" y="317" font-size="12" fill="{MUTED}">evaluated</text>'),
        (f'  <rect x="1230" y="305" width="14" height="14" rx="2" fill="{GREEN}"/>'),
        (f'  <text x="1252" y="317" font-size="12" fill="{MUTED}">pruned</text>'),
    ]

    display_names = {
        "co-location": "Co-location tie",
        "antimeridian": "Antimeridian",
        "north-pole": "North pole",
        "london": "London reference",
    }
    for index_value, case in enumerate(evidence.cases):
        y = row_y + index_value * row_gap
        evaluated_width = plot_width * case.evaluated / case.oracle_evaluated
        pruned_width = plot_width - evaluated_width
        reduction_percent = _format_percent_from_basis_points(
            case.work_reduction_basis_points
        )
        body.extend(
            [
                (
                    f'  <text x="66" y="{y + 17}" font-size="15" '
                    f'font-weight="700">{_escape(display_names[case.case_id])}</text>'
                ),
                (
                    f'  <text class="mono" x="66" y="{y + 42}" font-size="12" '
                    f'fill="{MUTED}">eval {_format_integer(case.evaluated)} · '
                    f"prune {_format_integer(case.pruned)} · visit "
                    f"{_format_integer(case.visited)}</text>"
                ),
                (
                    f'  <rect x="{plot_left}" y="{y}" width="{plot_width}" '
                    f'height="{bar_height}" rx="5" fill="{PANEL}" '
                    f'stroke="{GRID}" stroke-width="1"/>'
                ),
                (
                    f'  <rect x="{plot_left}" y="{y}" '
                    f'width="{evaluated_width:.4f}" height="{bar_height}" '
                    f'fill="{BLUE}"/>'
                ),
                (
                    f'  <rect x="{plot_left + evaluated_width:.4f}" y="{y}" '
                    f'width="{pruned_width:.4f}" height="{bar_height}" rx="5" '
                    f'fill="{GREEN}"/>'
                ),
                (
                    f'  <text class="mono" x="{plot_left}" y="{y + 50}" '
                    f'font-size="12" fill="{MUTED}">'
                    f"{_format_integer(case.evaluated)} evaluated</text>"
                ),
                (
                    f'  <text class="mono" x="{plot_left + plot_width}" '
                    f'y="{y + 50}" text-anchor="end" font-size="12" '
                    f'fill="{MUTED}">'
                    f"{_format_integer(case.pruned)} pruned · "
                    f"{reduction_percent} "
                    "avoided</text>"
                ),
            ]
        )

    body.extend(
        [
            (
                f'  <line x1="66" y1="738" x2="1334" y2="738" '
                f'stroke="{GRID}" stroke-width="1"/>'
            ),
            (f'  <rect x="66" y="764" width="8" height="32" rx="4" fill="{GOLD}"/>'),
            (
                f'  <text x="90" y="778" font-size="14" font-weight="700" '
                f'fill="{GOLD_DARK}">HONEST LIMIT</text>'
            ),
            (
                f'  <text x="90" y="800" font-size="13" fill="{MUTED}">'
                f"The exact kd-tree may still evaluate all "
                f"{_format_integer(evidence.index_records)} records in the "
                "worst case; pruning depends on the dataset and query.</text>"
            ),
            (
                f'  <text class="mono" x="1334" y="800" text-anchor="end" '
                f'font-size="11" fill="{MUTED}">source '
                f"{_escape(evidence.source_sha256[:12])}… · topology "
                f"{_escape(evidence.topology_sha256[:12])}…</text>"
            ),
        ]
    )
    return _svg_document(
        width=width,
        height=height,
        title_id="work-title",
        title="Exact query work on four real source-bound cases",
        description_id="work-description",
        description=(
            f"A record-accounting chart showing {evidence.accelerated_evaluations} "
            f"accelerated evaluations, {evidence.pruned_records} record "
            f"evaluations avoided by pruning, and {evidence.oracle_evaluations} "
            "full-scan evaluations across four oracle-matched queries, with an "
            "explicit worst-case warning."
        ),
        body=body,
    )


def _render_cli_capture(
    evidence: SpatialVisualEvidence,
    capture: NearestCapture,
) -> bytes:
    width = 1400
    height = 900
    command = "$ " + shlex.join(capture.display_argv)

    body = [
        f'  <rect width="{width}" height="{height}" fill="{WHITE}"/>',
        (
            f'  <rect x="34" y="30" width="1332" height="840" rx="22" '
            f'fill="{WHITE}" stroke="{GRID}" stroke-width="2"/>'
        ),
        (
            '  <text x="66" y="82" font-size="30" font-weight="700">'
            "Actual verified nearest-city CLI result</text>"
        ),
        (
            f'  <text x="66" y="116" font-size="16" fill="{MUTED}">'
            "Selected fields parsed from the real canonical stdout receipt and "
            "cross-checked against the tracked phase-one evidence</text>"
        ),
        (
            f'  <rect x="66" y="150" width="1268" height="650" rx="14" '
            f'fill="{NAVY}" stroke="{NAVY}" stroke-width="1"/>'
        ),
        (f'  <circle cx="94" cy="178" r="6" fill="{RED}"/>'),
        (f'  <circle cx="114" cy="178" r="6" fill="{GOLD}"/>'),
        (f'  <circle cx="134" cy="178" r="6" fill="{GREEN}"/>'),
        (
            '  <text class="mono" x="700" y="183" text-anchor="middle" '
            'font-size="12" fill="#C8D6E5">urbanlens · canonical receipt</text>'
        ),
        (
            f'  <text class="mono" x="92" y="224" font-size="13" '
            f'fill="#8FD3FF">{_escape(command)}</text>'
        ),
        (
            '  <line x1="92" y1="247" x2="1308" y2="247" '
            'stroke="#365979" stroke-width="1"/>'
        ),
        (
            '  <text class="mono" x="92" y="280" font-size="12" '
            'fill="#9DB4C9">stdout  selected canonical JSON fields</text>'
        ),
        (
            f'  <text class="mono" x="92" y="312" font-size="14" '
            f'fill="{WHITE}">receipt.schema   '
            f"{_escape(capture.receipt_schema)} v"
            f"{capture.receipt_schema_version}</text>"
        ),
        (
            f'  <text class="mono" x="92" y="340" font-size="14" '
            f'fill="{WHITE}">source.sha256    '
            f"{_escape(evidence.source_sha256)}</text>"
        ),
        (
            f'  <text class="mono" x="92" y="368" font-size="14" '
            f'fill="{WHITE}">index.topology   '
            f"{_escape(evidence.topology_sha256)}</text>"
        ),
        (
            f'  <text class="mono" x="92" y="396" font-size="14" '
            f'fill="{WHITE}">query             '
            f"lat={_escape(capture.query_latitude)} · "
            f"lon={_escape(capture.query_longitude)} · k="
            f"{len(capture.results)}</text>"
        ),
        (
            f'  <text class="mono" x="92" y="424" font-size="14" '
            'fill="#9BE7D8">proof.accelerated '
            f"evaluated={_format_integer(capture.evaluated)} · "
            f"pruned={_format_integer(capture.pruned)} · "
            f"visited={_format_integer(capture.visited)}</text>"
        ),
        (
            f'  <text class="mono" x="92" y="452" font-size="14" '
            'fill="#9BE7D8">proof.oracle      '
            f"{_escape(capture.oracle_status.upper())} · "
            f"evaluated={_format_integer(capture.oracle_evaluated)}</text>"
        ),
        (
            '  <line x1="92" y1="478" x2="1308" y2="478" '
            'stroke="#365979" stroke-width="1"/>'
        ),
        (
            '  <text class="mono" x="92" y="510" font-size="12" '
            'font-weight="700" fill="#9DB4C9">rank</text>'
        ),
        (
            '  <text class="mono" x="152" y="510" font-size="12" '
            'font-weight="700" fill="#9DB4C9">source_id</text>'
        ),
        (
            '  <text class="mono" x="318" y="510" font-size="12" '
            'font-weight="700" fill="#9DB4C9">city (ISO2)</text>'
        ),
        (
            '  <text class="mono" x="650" y="510" font-size="12" '
            'font-weight="700" fill="#9DB4C9">canonical E7 coordinate</text>'
        ),
        (
            '  <text class="mono" x="1120" y="510" font-size="12" '
            'font-weight="700" fill="#9DB4C9">distance</text>'
        ),
    ]
    for index_value, result in enumerate(capture.results):
        y = 550 + index_value * 62
        row_fill = "#1E486B" if index_value % 2 == 0 else "#1A4262"
        body.extend(
            [
                (
                    f'  <rect x="88" y="{y - 28}" width="1224" height="48" '
                    f'rx="6" fill="{row_fill}"/>'
                ),
                (
                    f'  <text class="mono" x="100" y="{y + 3}" '
                    f'font-size="14" fill="{WHITE}">{result.rank}</text>'
                ),
                (
                    f'  <text class="mono" x="152" y="{y + 3}" '
                    f'font-size="14" fill="{WHITE}">'
                    f"{_escape(result.source_id)}</text>"
                ),
                (
                    f'  <text class="mono" x="318" y="{y + 3}" '
                    f'font-size="14" fill="{WHITE}">'
                    f"{_escape(result.city_ascii)} "
                    f"({_escape(result.iso2)})</text>"
                ),
                (
                    f'  <text class="mono" x="650" y="{y + 3}" '
                    f'font-size="14" fill="{WHITE}">'
                    f"{result.latitude_e7}, {result.longitude_e7}</text>"
                ),
                (
                    f'  <text class="mono" x="1120" y="{y + 3}" '
                    f'font-size="14" fill="{WHITE}">'
                    f"{_format_integer(result.distance_m)} m</text>"
                ),
            ]
        )

    body.extend(
        [
            (
                f'  <text class="mono" x="92" y="758" font-size="12" '
                'fill="#9DB4C9">stdout.bytes  '
                f"{_format_integer(capture.stdout_bytes)} · canonical ASCII JSON "
                "· stderr (empty)</text>"
            ),
            (
                f'  <rect x="1124" y="818" width="210" height="34" rx="17" '
                f'fill="{GREEN_OPEN}" stroke="{GREEN}" stroke-width="1"/>'
            ),
            (
                f'  <text class="mono" x="1229" y="841" text-anchor="middle" '
                f'font-size="13" font-weight="700" fill="{GREEN_DARK}">'
                f"EXIT CODE {capture.returncode} · ORACLE MATCH</text>"
            ),
            (
                f'  <text x="66" y="840" font-size="13" fill="{MUTED}">'
                "Generated only after a second evidence freshness check; "
                "no absolute path, host, user, timestamp, or secret is captured."
                "</text>"
            ),
        ]
    )
    return _svg_document(
        width=width,
        height=height,
        title_id="cli-title",
        title="Actual verified nearest-city CLI result",
        description_id="cli-description",
        description=(
            "A terminal-style rendering of selected fields parsed from the "
            "actual nearest-city subprocess receipt for latitude minus 11.1 "
            "and longitude minus 77.6. It shows two zero-distance results, "
            "one third result, traversal counts, oracle match, and exit code zero."
        ),
        body=body,
    )


def _render_edge_proof(evidence: SpatialVisualEvidence) -> bytes:
    width = 1400
    height = 850
    cases = (
        evidence.case("co-location"),
        evidence.case("antimeridian"),
        evidence.case("north-pole"),
    )
    names = ("Co-location tie", "Antimeridian", "North-pole collapse")
    accents = (
        (BLUE_OPEN, BLUE, BLUE_DARK),
        (TEAL_OPEN, TEAL, TEAL_DARK),
        (GOLD_OPEN, GOLD, GOLD_DARK),
    )
    card_width = 402
    card_height = 532
    start_x = 66
    gap = 30
    card_y = 186

    body = [
        f'  <rect width="{width}" height="{height}" fill="{WHITE}"/>',
        (
            f'  <rect x="34" y="30" width="1332" height="790" rx="22" '
            f'fill="{WHITE}" stroke="{GRID}" stroke-width="2"/>'
        ),
        (
            '  <text x="66" y="84" font-size="30" font-weight="700">'
            "Real geographic edge cases with exact oracle proof</text>"
        ),
        (
            f'  <text x="66" y="118" font-size="16" fill="{MUTED}">'
            "Canonical coordinate rules and ordered top-3 matches are exercised "
            "against the frozen 44,691-record source</text>"
        ),
        (
            f'  <rect x="1130" y="62" width="204" height="38" rx="19" '
            f'fill="{GREEN_OPEN}" stroke="{GREEN}" stroke-width="1"/>'
        ),
        (
            f'  <text x="1232" y="87" text-anchor="middle" font-size="13" '
            f'font-weight="700" fill="{GREEN_DARK}">3 / 3 ORACLE MATCH</text>'
        ),
    ]
    for index_value, case in enumerate(cases):
        x = start_x + index_value * (card_width + gap)
        fill, stroke, dark = accents[index_value]
        canonical_changed = (
            case.requested_latitude != case.canonical_latitude
            or case.requested_longitude != case.canonical_longitude
        )
        first = case.results[0]
        body.extend(
            [
                (
                    f'  <rect x="{x}" y="{card_y}" width="{card_width}" '
                    f'height="{card_height}" rx="16" fill="{WHITE}" '
                    f'stroke="{stroke}" stroke-width="2"/>'
                ),
                (
                    f'  <rect x="{x}" y="{card_y}" width="{card_width}" '
                    f'height="76" rx="15" fill="{fill}"/>'
                ),
                (
                    f'  <text x="{x + 24}" y="{card_y + 32}" font-size="13" '
                    f'font-weight="700" fill="{dark}">CASE {index_value + 1}</text>'
                ),
                (
                    f'  <text x="{x + 24}" y="{card_y + 59}" font-size="20" '
                    f'font-weight="700">{_escape(names[index_value])}</text>'
                ),
                (
                    f'  <text x="{x + 24}" y="{card_y + 112}" font-size="12" '
                    f'font-weight="700" fill="{MUTED}">REQUESTED</text>'
                ),
                (
                    f'  <text class="mono" x="{x + 24}" y="{card_y + 139}" '
                    f'font-size="15">({_escape(case.requested_latitude)}, '
                    f"{_escape(case.requested_longitude)})</text>"
                ),
                (
                    f'  <text x="{x + 24}" y="{card_y + 178}" font-size="12" '
                    f'font-weight="700" fill="{MUTED}">CANONICAL E7</text>'
                ),
                (
                    f'  <text class="mono" x="{x + 24}" y="{card_y + 205}" '
                    f'font-size="15">({case.canonical_latitude_e7}, '
                    f"{case.canonical_longitude_e7})</text>"
                ),
                (
                    f'  <rect x="{x + 24}" y="{card_y + 226}" width="354" '
                    f'height="38" rx="8" fill="{fill}"/>'
                ),
                (
                    f'  <text x="{x + 201}" y="{card_y + 251}" '
                    f'text-anchor="middle" font-size="13" font-weight="700" '
                    f'fill="{dark}">'
                    f"{'NORMALIZED' if canonical_changed else 'EXACT INPUT RETAINED'}"
                    "</text>"
                ),
                (
                    f'  <text x="{x + 24}" y="{card_y + 300}" font-size="12" '
                    f'font-weight="700" fill="{MUTED}">NEAREST REAL RESULT</text>'
                ),
                (
                    f'  <text x="{x + 24}" y="{card_y + 330}" font-size="17" '
                    f'font-weight="700">{_escape(first.city_ascii)} '
                    f"({_escape(first.iso2)}) · "
                    f"{_format_integer(first.distance_m)} m</text>"
                ),
                (
                    f'  <text class="mono" x="{x + 24}" y="{card_y + 356}" '
                    f'font-size="12" fill="{MUTED}">source_id '
                    f"{_escape(first.source_id)}</text>"
                ),
                (
                    f'  <line x1="{x + 24}" y1="{card_y + 381}" '
                    f'x2="{x + 378}" y2="{card_y + 381}" '
                    f'stroke="{GRID}" stroke-width="1"/>'
                ),
                (
                    f'  <text x="{x + 24}" y="{card_y + 414}" font-size="12" '
                    f'font-weight="700" fill="{MUTED}">EXACT PROOF</text>'
                ),
                (
                    f'  <text class="mono" x="{x + 24}" y="{card_y + 442}" '
                    f'font-size="13">kd-tree {_format_integer(case.evaluated)} '
                    f"evaluated · {_format_integer(case.pruned)} pruned</text>"
                ),
                (
                    f'  <text class="mono" x="{x + 24}" y="{card_y + 469}" '
                    f'font-size="13">oracle '
                    f"{_format_integer(case.oracle_evaluated)} evaluated · "
                    "MATCH</text>"
                ),
                (
                    f'  <rect x="{x + 24}" y="{card_y + 490}" width="354" '
                    f'height="24" rx="12" fill="{GREEN_OPEN}"/>'
                ),
                (
                    f'  <text x="{x + 201}" y="{card_y + 507}" '
                    f'text-anchor="middle" font-size="11" font-weight="700" '
                    f'fill="{GREEN_DARK}">ORDERED TOP-{case.k} RANK KEYS AGREE'
                    "</text>"
                ),
            ]
        )

    co_location = cases[0]
    body.extend(
        [
            (
                f'  <rect x="66" y="744" width="1268" height="50" rx="10" '
                f'fill="{PANEL}" stroke="{GRID}" stroke-width="1"/>'
            ),
            (
                f'  <text x="88" y="766" font-size="13" font-weight="700" '
                f'fill="{NAVY}">CO-LOCATION ORDER</text>'
            ),
            (
                f'  <text class="mono" x="88" y="785" font-size="12" '
                f'fill="{MUTED}">'
                f"{_escape(co_location.results[0].source_id)} "
                f"({_escape(co_location.results[0].city_ascii)}, 0 m) → "
                f"{_escape(co_location.results[1].source_id)} "
                f"({_escape(co_location.results[1].city_ascii)}, 0 m) · "
                "stable source_id breaks the exact-distance tie</text>"
            ),
            (
                f'  <text class="mono" x="1312" y="776" text-anchor="end" '
                f'font-size="11" fill="{MUTED}">source '
                f"{_escape(evidence.source_sha256[:12])}…</text>"
            ),
        ]
    )
    return _svg_document(
        width=width,
        height=height,
        title_id="edge-title",
        title="Real geographic edge cases with exact oracle proof",
        description_id="edge-description",
        description=(
            "Three source-bound cards show a real zero-distance co-location "
            "tie, positive 180 longitude canonicalized to minus 180 at the "
            "antimeridian, and longitude collapsed to zero at the north pole. "
            "Every accelerated ordered top-three result matches full scan."
        ),
        body=body,
    )


def generate_visuals(
    evidence: SpatialVisualEvidence,
    capture: NearestCapture,
) -> dict[str, bytes]:
    """Render the complete ordered phase-one SVG artifact set in memory."""

    if not isinstance(evidence, SpatialVisualEvidence):
        raise TypeError("evidence must be SpatialVisualEvidence")
    if not isinstance(capture, NearestCapture):
        raise TypeError("capture must be NearestCapture")
    return {
        "spatial-index-architecture.svg": _render_architecture(evidence),
        "spatial-work-reduction.svg": _render_work_reduction(evidence),
        "nearest-city-cli-result.svg": _render_cli_capture(evidence, capture),
        "spatial-edge-case-proof.svg": _render_edge_proof(evidence),
    }


def _open_visual_directory(path: Path) -> int:
    """Anchor the existing visual directory without following its leaf."""

    if not isinstance(path, Path):
        raise TypeError("output directory must be a pathlib.Path")
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise SpatialVisualOutputError(
            "platform lacks anchored no-follow directory access"
        )
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    flags |= getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise SpatialVisualOutputError(
            "visual output directory cannot be anchored"
        ) from error
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise SpatialVisualOutputError("visual output parent is not a directory")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _validate_artifacts(
    directory_fd: int,
    artifacts: Mapping[str, bytes],
) -> None:
    if tuple(artifacts) != VISUAL_FILENAMES:
        raise SpatialVisualOutputError("visual artifact set or order is invalid")
    for name, payload in artifacts.items():
        if Path(name).name != name or name in {"", ".", ".."}:
            raise SpatialVisualOutputError("visual name is not a plain basename")
        if type(payload) is not bytes or len(payload) > MAX_SVG_BYTES:
            raise SpatialVisualOutputError("visual payload is invalid or oversized")
        try:
            metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            continue
        except OSError as error:
            raise SpatialVisualOutputError(
                "visual output cannot be inspected"
            ) from error
        if stat.S_ISLNK(metadata.st_mode):
            raise SpatialVisualOutputError("refusing to replace a symlink output")
        if not stat.S_ISREG(metadata.st_mode):
            raise SpatialVisualOutputError("visual output is not a regular file")


def _create_temporary_visual(directory_fd: int) -> tuple[int, str]:
    if not hasattr(os, "O_NOFOLLOW"):
        raise SpatialVisualOutputError("platform lacks safe temporary files")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    flags |= getattr(os, "O_CLOEXEC", 0)
    for _ in range(128):
        name = f".urbanlens-spatial-visual-{secrets.token_hex(16)}.tmp"
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
            raise SpatialVisualOutputError(
                "temporary visual cannot be created"
            ) from error
        return descriptor, name
    raise SpatialVisualOutputError("temporary visual name allocation failed")


def _write_visuals_to_anchored_directory(
    directory_fd: int,
    artifacts: Mapping[str, bytes],
) -> None:
    """Stage every payload, then replace targets under one held directory."""

    _validate_artifacts(directory_fd, artifacts)
    staged: dict[str, str] = {}
    try:
        for output_name, payload in artifacts.items():
            descriptor, temporary_name = _create_temporary_visual(directory_fd)
            staged[output_name] = temporary_name
            with os.fdopen(descriptor, "wb") as stream:
                written = stream.write(payload)
                if written != len(payload):
                    raise SpatialVisualOutputError("visual write was short")
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
    except SpatialVisualOutputError:
        raise
    except OSError as error:
        raise SpatialVisualOutputError("visual artifacts cannot be written") from error
    finally:
        for temporary_name in staged.values():
            with suppress(OSError):
                os.unlink(temporary_name, dir_fd=directory_fd)


def write_visuals(
    output_directory: Path,
    artifacts: Mapping[str, bytes],
) -> None:
    """Safely replace the checked SVG set relative to one directory anchor."""

    descriptor = _open_visual_directory(output_directory)
    try:
        _write_visuals_to_anchored_directory(descriptor, artifacts)
    finally:
        os.close(descriptor)


def _read_visual_at(directory_fd: int, name: str) -> bytes:
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_NONBLOCK"):
        raise SpatialVisualOutputError("platform lacks safe visual reads")
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    flags |= getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except OSError as error:
        raise SpatialVisualOutputError("checked visual cannot be opened") from error
    try:
        with os.fdopen(descriptor, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise SpatialVisualOutputError("checked visual is not a regular file")
            if before.st_size > MAX_SVG_BYTES:
                raise SpatialVisualOutputError("checked visual exceeds the byte limit")
            payload = stream.read(MAX_SVG_BYTES + 1)
            after = os.fstat(stream.fileno())
    except SpatialVisualOutputError:
        raise
    except OSError as error:
        raise SpatialVisualOutputError(
            "checked visual cannot be read safely"
        ) from error
    if len(payload) > MAX_SVG_BYTES:
        raise SpatialVisualOutputError("checked visual exceeds the byte limit")
    if len(payload) != before.st_size or not _same_file_state(before, after):
        raise SpatialVisualOutputError("checked visual changed while being read")
    return payload


def stale_visuals(
    output_directory: Path,
    artifacts: Mapping[str, bytes],
) -> tuple[str, ...]:
    """Return names whose tracked bytes do not equal fresh render bytes."""

    descriptor = _open_visual_directory(output_directory)
    try:
        stale: list[str] = []
        for name, expected in artifacts.items():
            try:
                actual = _read_visual_at(descriptor, name)
            except SpatialVisualOutputError:
                stale.append(name)
                continue
            if actual != expected:
                stale.append(name)
        return tuple(stale)
    finally:
        os.close(descriptor)


def _parser() -> argparse.ArgumentParser:
    parser = _QuietArgumentParser(
        prog="python3 -m urbanlens.spatial_visuals",
        allow_abbrev=False,
        description=("Check or write source-bound phase-one spatial SVG evidence."),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    return parser


def _emit_error(message: str) -> None:
    try:
        sys.stderr.write(message)
        sys.stderr.flush()
    except (BrokenPipeError, OSError, UnicodeError):
        pass


def main(argv: Sequence[str] | None = None) -> int:
    """Check or write all visuals without disclosing rejected ambient state."""

    try:
        arguments = _parser().parse_args(argv)
    except _ArgumentInputError:
        _emit_error(_ARGUMENT_ERROR_MESSAGE)
        return EXIT_ARGUMENT_ERROR

    root = repository_root()
    try:
        evidence, capture = collect_evidence(root)
        artifacts = generate_visuals(evidence, capture)
    except (
        SpatialVisualFreshnessError,
        SpatialVisualDataError,
        SpatialVisualOutputError,
        OSError,
    ):
        _emit_error(_INPUT_ERROR_MESSAGE)
        return EXIT_INPUT_ERROR

    output_directory = root / VISUAL_DIRECTORY_RELATIVE_PATH
    if arguments.write:
        try:
            write_visuals(output_directory, artifacts)
        except (SpatialVisualOutputError, OSError):
            _emit_error(_OUTPUT_ERROR_MESSAGE)
            return EXIT_OUTPUT_ERROR
        _emit_error(_WRITE_MESSAGE)
        return EXIT_SUCCESS

    try:
        stale = stale_visuals(output_directory, artifacts)
    except (SpatialVisualOutputError, OSError):
        _emit_error(_OUTPUT_ERROR_MESSAGE)
        return EXIT_OUTPUT_ERROR
    if stale:
        _emit_error(_STALE_MESSAGE)
        return EXIT_STALE
    _emit_error(_PASS_MESSAGE)
    return EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
