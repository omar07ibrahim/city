"""Source-bound bridge from the audited CSV capture to the spatial index.

The loader deliberately consumes the immutable :class:`DatasetCapture`
returned by :mod:`urbanlens.audit`.  Audit, record projection, and snapshot
construction all operate on that one capture; the source path is never opened
again by this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import urbanlens.audit as audit
from urbanlens.geo import GeoPoint
from urbanlens.spatial import SpatialIndex, SpatialRecord

_CAPTURE_FAILED: Final = "snapshot.capture_failed"
_AUDIT_INVALID: Final = "snapshot.audit_invalid"
_CONTRACT_MISMATCH: Final = "snapshot.contract_mismatch"
_RECORD_INVALID: Final = "snapshot.record_invalid"
_DUPLICATE_SOURCE_ID: Final = "snapshot.duplicate_source_id"
_COUNT_MISMATCH: Final = "snapshot.count_mismatch"
_SNAPSHOT_INVALID: Final = "snapshot.invalid"
_INDEX_INVALID: Final = "snapshot.index_invalid"

_ERROR_MESSAGES: Final = {
    _CAPTURE_FAILED: "the dataset could not be captured safely",
    _AUDIT_INVALID: "the dataset audit result is invalid",
    _CONTRACT_MISMATCH: "the dataset does not match the approved snapshot contract",
    _RECORD_INVALID: "a dataset record is unsafe for spatial indexing",
    _DUPLICATE_SOURCE_ID: "dataset source identifiers are not unique",
    _COUNT_MISMATCH: "dataset record counts are inconsistent",
    _SNAPSHOT_INVALID: "the city snapshot is invalid",
    _INDEX_INVALID: "the spatial index could not be constructed safely",
}


class SourceSnapshotError(ValueError):
    """Stable, non-sensitive failure raised by the source snapshot boundary."""

    code: str
    message: str

    def __init__(self, code: str) -> None:
        message = _ERROR_MESSAGES.get(code)
        if message is None:
            raise ValueError("unknown source snapshot error code")
        self.code = code
        self.message = message
        super().__init__(message)


def _snapshot_error(code: str) -> SourceSnapshotError:
    return SourceSnapshotError(code)


def _validate_record_sequence(
    records: object,
    *,
    expected_count: int,
    error_code: str,
) -> tuple[SpatialRecord, ...]:
    if type(records) is not tuple or len(records) != expected_count:
        raise _snapshot_error(error_code)

    previous_source_id: str | None = None
    for record in records:
        if type(record) is not SpatialRecord:
            raise _snapshot_error(error_code)
        try:
            record.__post_init__()
        except (TypeError, ValueError):
            raise _snapshot_error(error_code) from None
        if previous_source_id is not None and record.source_id <= previous_source_id:
            raise _snapshot_error(error_code)
        previous_source_id = record.source_id
    return records


@dataclass(frozen=True, slots=True)
class CitySnapshot:
    """Immutable, path-free projection of the approved source snapshot."""

    source_sha256: str
    byte_size: int
    data_rows: int
    audit_schema: str
    audit_schema_version: int
    records: tuple[SpatialRecord, ...] = field(repr=False)

    def __post_init__(self) -> None:
        contract = audit.DEFAULT_CONTRACT
        if (
            type(self.source_sha256) is not str
            or self.source_sha256 != contract.sha256
            or type(self.byte_size) is not int
            or self.byte_size != contract.byte_size
            or type(self.data_rows) is not int
            or self.data_rows != contract.data_rows
            or type(self.audit_schema) is not str
            or self.audit_schema != audit.MANIFEST_SCHEMA
            or type(self.audit_schema_version) is not int
            or self.audit_schema_version != audit.MANIFEST_SCHEMA_VERSION
        ):
            raise _snapshot_error(_SNAPSHOT_INVALID)
        _validate_record_sequence(
            self.records,
            expected_count=self.data_rows,
            error_code=_SNAPSHOT_INVALID,
        )


def _require_mapping(value: object) -> dict[object, object]:
    if type(value) is not dict:
        raise _snapshot_error(_AUDIT_INVALID)
    return value


def _validate_capture(capture: object) -> audit.DatasetCapture:
    contract = audit.DEFAULT_CONTRACT
    if type(capture) is not audit.DatasetCapture:
        raise _snapshot_error(_CONTRACT_MISMATCH)
    if (
        type(capture.sha256) is not str
        or capture.sha256 != contract.sha256
        or type(capture.byte_size) is not int
        or capture.byte_size != contract.byte_size
        or type(capture.columns) is not tuple
        or capture.columns != contract.columns
        or type(capture.rows) is not tuple
        or len(capture.rows) != contract.data_rows
    ):
        raise _snapshot_error(_CONTRACT_MISMATCH)
    return capture


def _validate_audit_manifest(
    manifest: object,
    *,
    capture: audit.DatasetCapture,
) -> tuple[str, int]:
    root = _require_mapping(manifest)
    manifest_metadata = _require_mapping(root.get("manifest"))
    quality = _require_mapping(root.get("quality"))
    dataset = _require_mapping(root.get("dataset"))
    schema = _require_mapping(root.get("schema"))

    audit_schema = manifest_metadata.get("schema")
    audit_schema_version = manifest_metadata.get("schema_version")
    if (
        type(audit_schema) is not str
        or audit_schema != audit.MANIFEST_SCHEMA
        or type(audit_schema_version) is not int
        or audit_schema_version != audit.MANIFEST_SCHEMA_VERSION
    ):
        raise _snapshot_error(_AUDIT_INVALID)

    if (
        dataset.get("name") != capture.source_name
        or dataset.get("encoding") != "utf-8-sig"
        or dataset.get("sha256") != capture.sha256
        or dataset.get("byte_size") != capture.byte_size
        or dataset.get("data_rows") != len(capture.rows)
        or schema.get("columns") != list(capture.columns)
        or schema.get("column_count") != len(capture.columns)
        or schema.get("row_width_mismatch_count") != 0
        or quality.get("status") != "pass"
    ):
        raise _snapshot_error(_CONTRACT_MISMATCH)

    checks = quality.get("contract_checks")
    summary = quality.get("summary")
    if type(checks) is not list or type(summary) is not dict:
        raise _snapshot_error(_AUDIT_INVALID)
    expected_values: tuple[tuple[str, object], ...] = (
        ("snapshot.sha256", audit.DEFAULT_CONTRACT.sha256),
        ("snapshot.byte_size", audit.DEFAULT_CONTRACT.byte_size),
        ("snapshot.data_rows", audit.DEFAULT_CONTRACT.data_rows),
        ("schema.columns", list(audit.DEFAULT_CONTRACT.columns)),
        ("schema.row_width_mismatches", 0),
        (
            "required_fields.blank_counts",
            dict.fromkeys(("city", "city_ascii", "country", "iso2", "iso3", "id"), 0),
        ),
        ("id.duplicate_row_excess", 0),
        ("id.invalid_format", 0),
        ("coordinates.invalid_counts", {"latitude": 0, "longitude": 0}),
        ("coordinates.e7_unrepresentable", 0),
        ("population.invalid_nonblank_values", 0),
        ("iso.invalid_format_counts", {"iso2": 0, "iso3": 0}),
        ("iso.iso2_to_iso3_conflicting_codes", 0),
        ("capital.invalid_values", 0),
        ("city_ascii.non_ascii_values", 0),
        ("rows.exact_duplicates_excluding_id", 0),
    )
    if len(checks) != len(expected_values) or summary != {
        "failed": 0,
        "passed": len(expected_values),
        "total": len(expected_values),
    }:
        raise _snapshot_error(_AUDIT_INVALID)
    for check, (check_id, expected_value) in zip(checks, expected_values, strict=True):
        if type(check) is not dict:
            raise _snapshot_error(_AUDIT_INVALID)
        if check != {
            "actual": expected_value,
            "expected": expected_value,
            "id": check_id,
            "status": "pass",
        }:
            raise _snapshot_error(_CONTRACT_MISMATCH)
    return audit_schema, audit_schema_version


def _project_records(capture: audit.DatasetCapture) -> tuple[SpatialRecord, ...]:
    positions = {column: position for position, column in enumerate(capture.columns)}
    projected: list[SpatialRecord] = []
    try:
        for row in capture.rows:
            if type(row) is not tuple or len(row) != len(capture.columns):
                raise _snapshot_error(_RECORD_INVALID)
            point = GeoPoint.from_decimal(
                row[positions["lat"]],
                row[positions["lng"]],
            )
            projected.append(
                SpatialRecord(
                    source_id=row[positions["id"]],
                    city_ascii=row[positions["city_ascii"]],
                    country=row[positions["country"]],
                    iso2=row[positions["iso2"]],
                    point=point,
                )
            )
    except SourceSnapshotError:
        raise
    except (IndexError, KeyError, TypeError, ValueError):
        raise _snapshot_error(_RECORD_INVALID) from None

    records = tuple(sorted(projected, key=lambda record: record.source_id))
    if len(records) != len(capture.rows):
        raise _snapshot_error(_COUNT_MISMATCH)
    if len({record.source_id for record in records}) != len(records):
        raise _snapshot_error(_DUPLICATE_SOURCE_ID)
    return records


def load_city_snapshot(path: Path = Path("train.csv")) -> CitySnapshot:
    """Capture, audit, and project the approved dataset without reopening it."""

    try:
        capture = audit.capture_dataset(path)
    except Exception:
        raise _snapshot_error(_CAPTURE_FAILED) from None

    try:
        capture = _validate_capture(capture)
    except SourceSnapshotError:
        raise
    except Exception:
        raise _snapshot_error(_CONTRACT_MISMATCH) from None
    try:
        manifest = audit.audit_capture(capture, contract=audit.DEFAULT_CONTRACT)
    except Exception:
        raise _snapshot_error(_AUDIT_INVALID) from None
    try:
        audit_schema, audit_schema_version = _validate_audit_manifest(
            manifest,
            capture=capture,
        )
    except SourceSnapshotError:
        raise
    except Exception:
        raise _snapshot_error(_AUDIT_INVALID) from None
    try:
        records = _project_records(capture)
    except SourceSnapshotError:
        raise
    except Exception:
        raise _snapshot_error(_RECORD_INVALID) from None
    if len(records) != audit.DEFAULT_CONTRACT.data_rows:
        raise _snapshot_error(_COUNT_MISMATCH)
    try:
        return CitySnapshot(
            source_sha256=capture.sha256,
            byte_size=capture.byte_size,
            data_rows=len(capture.rows),
            audit_schema=audit_schema,
            audit_schema_version=audit_schema_version,
            records=records,
        )
    except SourceSnapshotError:
        raise
    except Exception:
        raise _snapshot_error(_SNAPSHOT_INVALID) from None


def build_spatial_index(snapshot: CitySnapshot) -> SpatialIndex:
    """Build an exact index only after revalidating the snapshot boundary."""

    if type(snapshot) is not CitySnapshot:
        raise _snapshot_error(_SNAPSHOT_INVALID)
    try:
        snapshot.__post_init__()
        index = SpatialIndex(snapshot.records)
    except SourceSnapshotError:
        raise
    except Exception:
        raise _snapshot_error(_SNAPSHOT_INVALID) from None
    if index.metadata.records != snapshot.data_rows:
        raise _snapshot_error(_INDEX_INVALID)
    return index
