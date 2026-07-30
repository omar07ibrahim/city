"""Deterministic, source-bound evidence for phase-one spatial search.

The artifact produced here is intentionally structural rather than a timing
benchmark.  It binds exact nearest-neighbour results and traversal accounting
to the approved dataset digest and kd-tree topology, and verifies every query
against the full-scan oracle before emitting evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import stat
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path
from typing import Final, NoReturn

import urbanlens.audit as audit
from urbanlens import __version__
from urbanlens.geo import GeoPoint
from urbanlens.snapshot import build_spatial_index, load_city_snapshot
from urbanlens.spatial import NearestNeighbor, QueryDiagnostics, SpatialIndex

EVIDENCE_SCHEMA: Final = "urbanlens.spatial-phase1-evidence"
EVIDENCE_SCHEMA_VERSION: Final = 1
MAX_EVIDENCE_BYTES: Final = 256 * 1024
DEFAULT_DATASET: Final = Path("train.csv")
DEFAULT_ARTIFACT: Final = Path("artifacts/spatial/phase1-evidence.json")

EXPECTED_SOURCE_SHA256: Final = audit.DEFAULT_CONTRACT.sha256
EXPECTED_SOURCE_ROWS: Final = audit.DEFAULT_CONTRACT.data_rows
EXPECTED_SOURCE_BYTES: Final = audit.DEFAULT_CONTRACT.byte_size
EXPECTED_TOPOLOGY_SHA256: Final = (
    "5ba1bb95e167d7bdf3b9e28e8e211756eab48888c281c215d1dccf801db3cb72"
)

EXIT_SUCCESS: Final = 0
EXIT_STALE: Final = 1
EXIT_ARGUMENT_ERROR: Final = 2
EXIT_INPUT_ERROR: Final = 3
EXIT_OUTPUT_ERROR: Final = 4

_CANONICALIZATION: Final = (
    "US-ASCII, sorted object keys, two-space indent, LF, final newline"
)
_SHA256 = re.compile(r"[0-9a-f]{64}", re.ASCII)
_ONE_METER: Final = Decimal("1")
_EARTH_RADIUS_SCALE: Final = Decimal("1000")

_ARGUMENT_ERROR_MESSAGE: Final = "error: invalid evidence command arguments\n"
_INPUT_ERROR_MESSAGE: Final = "error: spatial evidence input was rejected\n"
_OUTPUT_ERROR_MESSAGE: Final = "error: spatial evidence artifact operation failed\n"
_STALE_MESSAGE: Final = "spatial evidence FAIL: tracked bytes are stale\n"
_PASS_MESSAGE: Final = "spatial evidence PASS: tracked bytes are current\n"
_WRITE_MESSAGE: Final = "spatial evidence WRITE: canonical bytes replaced\n"


class EvidenceGenerationError(RuntimeError):
    """Raised when real evidence cannot satisfy its source-bound contract."""


class EvidenceArtifactError(RuntimeError):
    """Raised when evidence bytes cannot be checked or written safely."""


class EvidenceArtifactMissing(EvidenceArtifactError):
    """Raised when a checked evidence artifact does not exist yet."""


class _ArgumentInputError(Exception):
    """Raised instead of echoing an untrusted command-line token."""


class _QuietArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        del message
        raise _ArgumentInputError


@dataclass(frozen=True, slots=True)
class QueryCase:
    """One declared, reproducible edge or reference query."""

    case_id: str
    purpose: str
    latitude: str
    longitude: str
    k: int


QUERY_CASES: Final = (
    QueryCase(
        case_id="co-location",
        purpose="stable source-id ordering at an exact shared location",
        latitude="-11.1",
        longitude="-77.6",
        k=3,
    ),
    QueryCase(
        case_id="antimeridian",
        purpose="canonical +180/-180 longitude equivalence",
        latitude="-16",
        longitude="180",
        k=3,
    ),
    QueryCase(
        case_id="north-pole",
        purpose="longitude collapse at the geographic pole",
        latitude="90",
        longitude="73",
        k=3,
    ),
    QueryCase(
        case_id="london",
        purpose="ordinary urban nearest-neighbour reference",
        latitude="51.5074",
        longitude="-0.1278",
        k=3,
    ),
)


def _integer(value: object, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise EvidenceGenerationError("invalid integer evidence")
    return value


def _sha256(value: object) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise EvidenceGenerationError("invalid digest evidence")
    return value


def _distance_meters(value: float) -> int:
    if type(value) is not float or not math.isfinite(value) or value < 0.0:
        raise EvidenceGenerationError("invalid result distance")
    try:
        rounded = Decimal(str(value)).quantize(_ONE_METER, rounding=ROUND_HALF_UP)
    except (InvalidOperation, ValueError) as error:
        raise EvidenceGenerationError("invalid result distance") from error
    return int(rounded)


def _basis_points(numerator: int, denominator: int) -> int:
    """Return nearest-integer basis points with integer-only arithmetic."""

    numerator = _integer(numerator)
    denominator = _integer(denominator, minimum=1)
    if numerator > denominator:
        raise EvidenceGenerationError("invalid work-reduction fraction")
    return (numerator * 10_000 + denominator // 2) // denominator


def _diagnostics_payload(diagnostics: QueryDiagnostics) -> dict[str, int]:
    return {
        "evaluated": _integer(diagnostics.evaluated),
        "pruned": _integer(diagnostics.pruned),
        "visited": _integer(diagnostics.visited),
    }


def _result_payload(neighbor: NearestNeighbor, rank: int) -> dict[str, object]:
    record = neighbor.record
    return {
        "city_ascii": record.city_ascii,
        "country": record.country,
        "distance_m": _distance_meters(neighbor.distance_meters),
        "iso2": record.iso2,
        "latitude_e7": record.point.latitude_e7,
        "longitude_e7": record.point.longitude_e7,
        "rank": rank,
        "source_id": record.source_id,
    }


def _work_reduction_payload(
    *,
    accelerated_evaluations: int,
    oracle_evaluations: int,
    pruned: int,
) -> dict[str, object]:
    accelerated_evaluations = _integer(accelerated_evaluations)
    oracle_evaluations = _integer(oracle_evaluations, minimum=1)
    pruned = _integer(pruned)
    avoided = oracle_evaluations - accelerated_evaluations
    if avoided < 0 or pruned != avoided:
        raise EvidenceGenerationError("traversal work accounting is inconsistent")
    return {
        "avoided_evaluations": avoided,
        "basis_points": _basis_points(avoided, oracle_evaluations),
        "fraction": {
            "denominator": oracle_evaluations,
            "numerator": avoided,
        },
    }


def _case_payload(index: SpatialIndex, case: QueryCase) -> dict[str, object]:
    point = GeoPoint.from_decimal(case.latitude, case.longitude)
    accelerated = index.query(point, k=case.k)
    oracle = index.query_full_scan(point, k=case.k)

    accelerated_keys = tuple(neighbor.rank_key for neighbor in accelerated.neighbors)
    oracle_keys = tuple(neighbor.rank_key for neighbor in oracle.neighbors)
    if accelerated_keys != oracle_keys:
        raise EvidenceGenerationError("accelerated result differs from oracle")

    records = _integer(index.metadata.records, minimum=1)
    accelerated_diagnostics = accelerated.diagnostics
    oracle_diagnostics = oracle.diagnostics
    if (
        accelerated_diagnostics.evaluated + accelerated_diagnostics.pruned != records
        or oracle_diagnostics.evaluated != records
        or oracle_diagnostics.pruned != 0
        or oracle_diagnostics.visited != records
        or len(accelerated.neighbors) != case.k
        or len(oracle.neighbors) != case.k
    ):
        raise EvidenceGenerationError("query accounting is inconsistent")

    return {
        "case_id": case.case_id,
        "proof": {
            "accelerated": _diagnostics_payload(accelerated_diagnostics),
            "oracle": {
                "evaluated": oracle_diagnostics.evaluated,
                "status": "match",
            },
            "work_reduction": _work_reduction_payload(
                accelerated_evaluations=accelerated_diagnostics.evaluated,
                oracle_evaluations=oracle_diagnostics.evaluated,
                pruned=accelerated_diagnostics.pruned,
            ),
        },
        "purpose": case.purpose,
        "query": {
            "canonical": {
                "latitude": point.latitude,
                "latitude_e7": point.latitude_e7,
                "longitude": point.longitude,
                "longitude_e7": point.longitude_e7,
            },
            "k": case.k,
            "requested": {
                "latitude": case.latitude,
                "longitude": case.longitude,
            },
        },
        "results": [
            _result_payload(neighbor, rank)
            for rank, neighbor in enumerate(accelerated.neighbors, start=1)
        ],
    }


def _aggregate_payload(cases: Sequence[dict[str, object]]) -> dict[str, object]:
    accelerated_evaluations = 0
    oracle_evaluations = 0
    pruned = 0
    visited = 0
    for case in cases:
        proof = case.get("proof")
        if type(proof) is not dict:
            raise EvidenceGenerationError("case proof is invalid")
        accelerated = proof.get("accelerated")
        oracle = proof.get("oracle")
        if type(accelerated) is not dict or type(oracle) is not dict:
            raise EvidenceGenerationError("case proof is invalid")
        accelerated_evaluations += _integer(accelerated.get("evaluated"))
        pruned += _integer(accelerated.get("pruned"))
        visited += _integer(accelerated.get("visited"))
        oracle_evaluations += _integer(oracle.get("evaluated"), minimum=1)

    return {
        "accelerated_evaluations": accelerated_evaluations,
        "oracle_evaluations": oracle_evaluations,
        "pruned_records": pruned,
        "query_count": len(cases),
        "visited_nodes": visited,
        "work_reduction": _work_reduction_payload(
            accelerated_evaluations=accelerated_evaluations,
            oracle_evaluations=oracle_evaluations,
            pruned=pruned,
        ),
    }


def _index_payload(index: SpatialIndex) -> dict[str, object]:
    metadata = index.metadata
    if metadata.topology_sha256 != EXPECTED_TOPOLOGY_SHA256:
        raise EvidenceGenerationError("index topology does not match evidence contract")
    try:
        scaled_radius = Decimal(str(metadata.earth_radius_meters)) * _EARTH_RADIUS_SCALE
        if scaled_radius != scaled_radius.to_integral_value():
            raise EvidenceGenerationError("invalid earth radius evidence")
        earth_radius_mm = int(scaled_radius)
    except (InvalidOperation, ValueError, OverflowError) as error:
        raise EvidenceGenerationError("invalid earth radius evidence") from error
    if earth_radius_mm <= 0:
        raise EvidenceGenerationError("invalid earth radius evidence")
    return {
        "algorithm": metadata.algorithm,
        "collision_excess": _integer(metadata.collision_excess),
        "collision_groups": _integer(metadata.collision_groups),
        "coordinate_scale": _integer(metadata.coordinate_scale, minimum=1),
        "depth": _integer(metadata.depth, minimum=1),
        "earth_radius_millimeters": earth_radius_mm,
        "max_location_multiplicity": _integer(
            metadata.max_location_multiplicity,
            minimum=1,
        ),
        "records": _integer(metadata.records, minimum=1),
        "topology_sha256": _sha256(metadata.topology_sha256),
        "unique_numeric_locations": _integer(
            metadata.unique_numeric_locations,
            minimum=1,
        ),
        "version": _integer(metadata.version, minimum=1),
    }


def build_evidence(dataset: Path = DEFAULT_DATASET) -> dict[str, object]:
    """Build real, verified phase-one evidence from the approved source."""

    if not isinstance(dataset, Path):
        raise TypeError("dataset must be a pathlib.Path")
    snapshot = load_city_snapshot(dataset)
    if (
        snapshot.source_sha256 != EXPECTED_SOURCE_SHA256
        or snapshot.data_rows != EXPECTED_SOURCE_ROWS
        or snapshot.byte_size != EXPECTED_SOURCE_BYTES
    ):
        raise EvidenceGenerationError("source does not match evidence contract")

    index = build_spatial_index(snapshot)
    if index.metadata.records != EXPECTED_SOURCE_ROWS:
        raise EvidenceGenerationError("index record count does not match source")
    index_payload = _index_payload(index)
    cases = [_case_payload(index, case) for case in QUERY_CASES]
    return {
        "aggregate": _aggregate_payload(cases),
        "cases": cases,
        "evidence": {
            "canonicalization": _CANONICALIZATION,
            "phase": 1,
            "schema": EVIDENCE_SCHEMA,
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "tool": "python3 -m urbanlens.spatial_evidence",
            "tool_version": __version__,
        },
        "index": index_payload,
        "source": {
            "audit_schema": snapshot.audit_schema,
            "audit_schema_version": _integer(
                snapshot.audit_schema_version,
                minimum=1,
            ),
            "byte_size": _integer(snapshot.byte_size, minimum=1),
            "data_rows": _integer(snapshot.data_rows, minimum=1),
            "sha256": _sha256(snapshot.source_sha256),
        },
    }


def canonical_evidence_bytes(evidence: dict[str, object]) -> bytes:
    """Serialize evidence as bounded canonical ASCII JSON."""

    try:
        payload = (
            json.dumps(
                evidence,
                allow_nan=False,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise EvidenceArtifactError("evidence cannot be serialized") from error
    if len(payload) > MAX_EVIDENCE_BYTES:
        raise EvidenceArtifactError("evidence exceeds the byte limit")
    return payload


def _same_file_state(before: os.stat_result, after: os.stat_result) -> bool:
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


def _read_artifact(path: Path) -> bytes:
    """Read one unchanged, bounded regular artifact without following links."""

    try:
        path_state = path.lstat()
    except FileNotFoundError as error:
        raise EvidenceArtifactMissing("artifact does not exist") from error
    except OSError as error:
        raise EvidenceArtifactError("artifact cannot be inspected") from error
    if not stat.S_ISREG(path_state.st_mode):
        raise EvidenceArtifactError("artifact is not a regular file")
    if path_state.st_size > MAX_EVIDENCE_BYTES:
        raise EvidenceArtifactError("artifact exceeds the byte limit")
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_NONBLOCK"):
        raise EvidenceArtifactError("platform lacks safe artifact reads")

    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    flags |= getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            before = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_dev != path_state.st_dev
                or before.st_ino != path_state.st_ino
                or before.st_size > MAX_EVIDENCE_BYTES
            ):
                raise EvidenceArtifactError("artifact identity is unsafe")
            payload = stream.read(MAX_EVIDENCE_BYTES + 1)
            after = os.fstat(stream.fileno())
    except EvidenceArtifactError:
        raise
    except OSError as error:
        raise EvidenceArtifactError("artifact cannot be read safely") from error
    if len(payload) > MAX_EVIDENCE_BYTES:
        raise EvidenceArtifactError("artifact exceeds the byte limit")
    if not _same_file_state(before, after) or len(payload) != before.st_size:
        raise EvidenceArtifactError("artifact changed while being read")
    return payload


def artifact_is_current(path: Path, payload: bytes) -> bool:
    """Return whether safe tracked bytes exactly match fresh canonical bytes."""

    if not isinstance(path, Path) or type(payload) is not bytes:
        raise TypeError("path and payload types are invalid")
    if len(payload) > MAX_EVIDENCE_BYTES:
        raise EvidenceArtifactError("evidence exceeds the byte limit")
    try:
        return _read_artifact(path) == payload
    except EvidenceArtifactMissing:
        return False


def write_artifact(path: Path, dataset: Path, payload: bytes) -> None:
    """Atomically replace an anchored artifact without overwriting its source."""

    if (
        not isinstance(path, Path)
        or not isinstance(dataset, Path)
        or type(payload) is not bytes
    ):
        raise TypeError("artifact write types are invalid")
    if len(payload) > MAX_EVIDENCE_BYTES:
        raise EvidenceArtifactError("evidence exceeds the byte limit")

    anchored: audit.AnchoredOutput | None = None
    try:
        anchored = audit._anchor_output(path)
        try:
            dataset_identity = os.stat(dataset)
        except OSError as error:
            raise EvidenceArtifactError(
                "source identity cannot be inspected"
            ) from error
        if audit._anchored_output_matches_dataset(anchored, dataset_identity):
            raise EvidenceArtifactError("refusing to overwrite evidence source")
        audit._write_atomic(anchored, payload)
    except EvidenceArtifactError:
        raise
    except audit.ManifestOutputError as error:
        raise EvidenceArtifactError("artifact cannot be written safely") from error
    finally:
        if anchored is not None:
            os.close(anchored.directory_fd)


def _emit_error(message: str) -> None:
    try:
        sys.stderr.write(message)
        sys.stderr.flush()
    except (BrokenPipeError, OSError, UnicodeError):
        pass


def _parser() -> argparse.ArgumentParser:
    parser = _QuietArgumentParser(
        prog="python3 -m urbanlens.spatial_evidence",
        allow_abbrev=False,
        description="Check or write real source-bound phase-one evidence.",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Check or write evidence without disclosing rejected path values."""

    try:
        arguments = _parser().parse_args(argv)
    except _ArgumentInputError:
        _emit_error(_ARGUMENT_ERROR_MESSAGE)
        return EXIT_ARGUMENT_ERROR

    try:
        evidence = build_evidence(arguments.dataset)
    except Exception:
        _emit_error(_INPUT_ERROR_MESSAGE)
        return EXIT_INPUT_ERROR
    try:
        payload = canonical_evidence_bytes(evidence)
    except EvidenceArtifactError:
        _emit_error(_OUTPUT_ERROR_MESSAGE)
        return EXIT_OUTPUT_ERROR

    if arguments.check:
        try:
            current = artifact_is_current(arguments.artifact, payload)
        except (EvidenceArtifactError, OSError):
            _emit_error(_OUTPUT_ERROR_MESSAGE)
            return EXIT_OUTPUT_ERROR
        if not current:
            _emit_error(_STALE_MESSAGE)
            return EXIT_STALE
        _emit_error(_PASS_MESSAGE)
        return EXIT_SUCCESS

    try:
        write_artifact(arguments.artifact, arguments.dataset, payload)
    except (EvidenceArtifactError, OSError):
        _emit_error(_OUTPUT_ERROR_MESSAGE)
        return EXIT_OUTPUT_ERROR
    _emit_error(_WRITE_MESSAGE)
    return EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
