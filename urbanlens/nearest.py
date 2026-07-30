"""Canonical one-query CLI for exact nearest-city search.

The command deliberately emits one bounded JSON receipt and no ambient
metadata.  Dataset paths and rejected tokens never cross the error boundary.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path
from typing import BinaryIO, Final, NoReturn, cast

from urbanlens import __version__
from urbanlens.geo import CoordinateInputError, GeoPoint
from urbanlens.snapshot import (
    SourceSnapshotError,
    build_spatial_index,
    load_city_snapshot,
)
from urbanlens.spatial import (
    MAX_QUERY_K,
    IndexVerificationError,
    NearestNeighbor,
    QueryDiagnostics,
    SpatialIndex,
)

RECEIPT_SCHEMA: Final = "urbanlens.nearest-city-receipt"
RECEIPT_SCHEMA_VERSION: Final = 1
MAX_RECEIPT_BYTES: Final = 256 * 1024

EXIT_SUCCESS: Final = 0
EXIT_ARGPARSE_ERROR: Final = 2
EXIT_ARGUMENT_ERROR: Final = EXIT_ARGPARSE_ERROR
EXIT_INPUT_ERROR: Final = 3
EXIT_OUTPUT_ERROR: Final = 4
EXIT_VERIFICATION_ERROR: Final = 5

_CANONICALIZATION: Final = (
    "UTF-8, sorted object keys, two-space indent, LF, final newline"
)
_SHA256 = re.compile(r"[0-9a-f]{64}", re.ASCII)
_POSITIVE_INTEGER = re.compile(r"[1-9][0-9]{0,2}", re.ASCII)
_ONE_METER = Decimal("1")

_ARGUMENT_ERROR_MESSAGE: Final = "error: invalid command arguments\n"
_INPUT_ERROR_MESSAGE: Final = "error: source or query input was rejected\n"
_OUTPUT_ERROR_MESSAGE: Final = "error: receipt output failed\n"
_VERIFICATION_ERROR_MESSAGE: Final = "error: nearest-neighbor verification failed\n"


class _ArgumentInputError(Exception):
    """Raised instead of allowing argparse to echo an untrusted token."""


class ReceiptOutputError(Exception):
    """Raised when a canonical receipt cannot be safely emitted."""


class _QuietArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        del message
        raise _ArgumentInputError


def _parse_k(token: str) -> int:
    if _POSITIVE_INTEGER.fullmatch(token) is None:
        raise argparse.ArgumentTypeError("invalid k")
    value = int(token)
    if value > MAX_QUERY_K:
        raise argparse.ArgumentTypeError("invalid k")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = _QuietArgumentParser(
        prog="python3 -m urbanlens.nearest",
        allow_abbrev=False,
        description="Emit one canonical exact nearest-city receipt.",
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--lat", required=True)
    parser.add_argument("--lon", required=True)
    parser.add_argument("--k", type=_parse_k, default=1)
    parser.add_argument("--verify", action="store_true")
    return parser


def _integer(value: object, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError("invalid receipt integer")
    return value


def _safe_sha256(value: object) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError("invalid source digest")
    return value


def _distance_meters_half_up(value: float) -> int:
    if type(value) is not float or not math.isfinite(value) or value < 0.0:
        raise ValueError("invalid result distance")
    try:
        rounded = Decimal(str(value)).quantize(_ONE_METER, rounding=ROUND_HALF_UP)
    except (InvalidOperation, ValueError) as error:
        raise ValueError("invalid result distance") from error
    return int(rounded)


def _diagnostics_payload(diagnostics: QueryDiagnostics) -> dict[str, int]:
    return {
        "evaluated": _integer(diagnostics.evaluated),
        "pruned": _integer(diagnostics.pruned),
        "visited": _integer(diagnostics.visited),
    }


def _neighbor_payload(neighbor: NearestNeighbor, rank: int) -> dict[str, object]:
    record = neighbor.record
    point = record.point
    return {
        "city_ascii": record.city_ascii,
        "country": record.country,
        "distance_m": _distance_meters_half_up(neighbor.distance_meters),
        "iso2": record.iso2,
        "latitude": point.latitude,
        "latitude_e7": point.latitude_e7,
        "longitude": point.longitude,
        "longitude_e7": point.longitude_e7,
        "rank": rank,
        "source_id": record.source_id,
    }


def _index_payload(index: SpatialIndex) -> dict[str, object]:
    metadata = index.metadata
    earth_radius = metadata.earth_radius_meters
    if (
        type(earth_radius) is not float
        or not math.isfinite(earth_radius)
        or earth_radius <= 0.0
    ):
        raise ValueError("invalid index metadata")
    return {
        "algorithm": metadata.algorithm,
        "collision_excess": _integer(metadata.collision_excess),
        "collision_groups": _integer(metadata.collision_groups),
        "coordinate_scale": _integer(metadata.coordinate_scale, minimum=1),
        "depth": _integer(metadata.depth, minimum=1),
        "earth_radius_meters": earth_radius,
        "max_location_multiplicity": _integer(
            metadata.max_location_multiplicity,
            minimum=1,
        ),
        "records": _integer(metadata.records, minimum=1),
        "topology_sha256": _safe_sha256(metadata.topology_sha256),
        "unique_numeric_locations": _integer(
            metadata.unique_numeric_locations,
            minimum=1,
        ),
        "version": _integer(metadata.version, minimum=1),
    }


def build_receipt(
    dataset: Path,
    point: GeoPoint,
    *,
    k: int = 1,
    verify: bool = False,
) -> dict[str, object]:
    """Load one source snapshot and return its exact nearest-city receipt."""

    if not isinstance(dataset, Path):
        raise TypeError("dataset must be a pathlib.Path")
    if type(point) is not GeoPoint:
        raise TypeError("point must be a GeoPoint")
    point.__post_init__()
    if type(k) is not int or not 1 <= k <= MAX_QUERY_K:
        raise ValueError(f"k must be in [1, {MAX_QUERY_K}]")
    if type(verify) is not bool:
        raise TypeError("verify must be a boolean")

    snapshot = load_city_snapshot(dataset)
    index = build_spatial_index(snapshot)
    accelerated = index.query(point, k=k)

    oracle_status = "not-requested"
    oracle_evaluations = 0
    if verify:
        oracle = index.query_full_scan(point, k=k)
        accelerated_keys = tuple(
            neighbor.rank_key for neighbor in accelerated.neighbors
        )
        oracle_keys = tuple(neighbor.rank_key for neighbor in oracle.neighbors)
        if accelerated_keys != oracle_keys:
            raise IndexVerificationError(
                "accelerated and oracle ordered rank keys differ"
            )
        oracle_status = "match"
        oracle_evaluations = _integer(oracle.diagnostics.evaluated)

    return {
        "index": _index_payload(index),
        "proof": {
            "accelerated": _diagnostics_payload(accelerated.diagnostics),
            "oracle": {
                "evaluations": oracle_evaluations,
                "status": oracle_status,
            },
        },
        "query": {
            "k": k,
            "latitude": point.latitude,
            "latitude_e7": point.latitude_e7,
            "longitude": point.longitude,
            "longitude_e7": point.longitude_e7,
        },
        "receipt": {
            "canonicalization": _CANONICALIZATION,
            "schema": RECEIPT_SCHEMA,
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "tool": "python3 -m urbanlens.nearest",
            "tool_version": __version__,
        },
        "results": [
            _neighbor_payload(neighbor, rank)
            for rank, neighbor in enumerate(accelerated.neighbors, start=1)
        ],
        "source": {
            "audit_schema": snapshot.audit_schema,
            "audit_schema_version": _integer(
                snapshot.audit_schema_version,
                minimum=1,
            ),
            "byte_size": _integer(snapshot.byte_size, minimum=1),
            "data_rows": _integer(snapshot.data_rows, minimum=1),
            "sha256": _safe_sha256(snapshot.source_sha256),
        },
    }


def canonical_receipt_bytes(receipt: dict[str, object]) -> bytes:
    """Serialize one receipt canonically and enforce the pre-write cap."""

    try:
        payload = (
            json.dumps(
                receipt,
                allow_nan=False,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise ReceiptOutputError("receipt cannot be serialized") from error
    if len(payload) > MAX_RECEIPT_BYTES:
        raise ReceiptOutputError("receipt exceeds the output limit")
    return payload


def _write_all(stream: BinaryIO, payload: bytes) -> None:
    try:
        written = stream.write(payload)
        if written != len(payload):
            raise ReceiptOutputError("short receipt write")
        stream.flush()
    except ReceiptOutputError:
        raise
    except (BrokenPipeError, OSError, UnicodeError) as error:
        raise ReceiptOutputError("receipt stream rejected output") from error


def _stdout_buffer() -> BinaryIO:
    stream = getattr(sys.stdout, "buffer", None)
    if stream is None:
        raise ReceiptOutputError("binary stdout is unavailable")
    return cast(BinaryIO, stream)


def _emit_error(message: str) -> None:
    try:
        sys.stderr.write(message)
        sys.stderr.flush()
    except (BrokenPipeError, OSError, UnicodeError):
        pass


def main(argv: list[str] | None = None) -> int:
    """Run the bounded CLI without disclosing rejected source or query values."""

    try:
        arguments = _parser().parse_args(argv)
    except _ArgumentInputError:
        _emit_error(_ARGUMENT_ERROR_MESSAGE)
        return EXIT_ARGUMENT_ERROR

    try:
        point = GeoPoint.from_decimal(arguments.lat, arguments.lon)
        receipt = build_receipt(
            Path(arguments.dataset),
            point,
            k=arguments.k,
            verify=arguments.verify,
        )
    except IndexVerificationError:
        _emit_error(_VERIFICATION_ERROR_MESSAGE)
        return EXIT_VERIFICATION_ERROR
    except (SourceSnapshotError, CoordinateInputError, OSError):
        _emit_error(_INPUT_ERROR_MESSAGE)
        return EXIT_INPUT_ERROR

    try:
        payload = canonical_receipt_bytes(receipt)
        _write_all(_stdout_buffer(), payload)
    except ReceiptOutputError:
        _emit_error(_OUTPUT_ERROR_MESSAGE)
        return EXIT_OUTPUT_ERROR
    return EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
