from __future__ import annotations

import dataclasses
import unittest
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import patch

import urbanlens.audit as audit
from urbanlens.audit import DEFAULT_CONTRACT, DatasetCapture
from urbanlens.geo import GeoPoint
from urbanlens.snapshot import (
    CitySnapshot,
    SourceSnapshotError,
    build_spatial_index,
    load_city_snapshot,
)
from urbanlens.spatial import SpatialRecord

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATASET = REPOSITORY_ROOT / "train.csv"


def replace_row(
    capture: DatasetCapture,
    row_number: int,
    *,
    column: str,
    value: str,
) -> DatasetCapture:
    rows = list(capture.rows)
    row = list(rows[row_number])
    row[capture.columns.index(column)] = value
    rows[row_number] = tuple(row)
    return dataclasses.replace(capture, rows=tuple(rows))


def passing_manifest(capture: DatasetCapture) -> dict[str, Any]:
    manifest = audit.audit_capture(capture)
    manifest["quality"]["status"] = "pass"
    manifest["quality"]["summary"] = {
        "failed": 0,
        "passed": 16,
        "total": 16,
    }
    for check in manifest["quality"]["contract_checks"]:
        check["status"] = "pass"
        check["actual"] = check["expected"]
    return manifest


class CitySnapshotIntegrationTests(unittest.TestCase):
    capture: ClassVar[DatasetCapture]
    snapshot: ClassVar[CitySnapshot]

    @classmethod
    def setUpClass(cls) -> None:
        cls.capture = audit.capture_dataset(DATASET)
        cls.snapshot = load_city_snapshot(DATASET)

    def test_real_snapshot_identity_records_and_index_metadata(self) -> None:
        snapshot = self.snapshot
        self.assertEqual(snapshot.source_sha256, DEFAULT_CONTRACT.sha256)
        self.assertEqual(snapshot.byte_size, 4_734_682)
        self.assertEqual(snapshot.data_rows, 44_691)
        self.assertEqual(snapshot.audit_schema, audit.MANIFEST_SCHEMA)
        self.assertEqual(snapshot.audit_schema_version, 3)
        self.assertEqual(len(snapshot.records), 44_691)
        self.assertEqual(
            tuple(record.source_id for record in snapshot.records),
            tuple(sorted(record.source_id for record in snapshot.records)),
        )

        index = build_spatial_index(snapshot)
        self.assertEqual(index.metadata.records, 44_691)
        self.assertEqual(index.metadata.unique_numeric_locations, 44_554)
        self.assertEqual(index.metadata.collision_groups, 106)
        self.assertEqual(index.metadata.collision_excess, 137)
        self.assertEqual(index.metadata.max_location_multiplicity, 7)

    def test_real_negative_zero_source_token_is_canonicalized(self) -> None:
        longitude = self.capture.columns.index("lng")
        source_id = self.capture.columns.index("id")
        matching_ids = tuple(
            row[source_id] for row in self.capture.rows if row[longitude] == "-0.0000"
        )

        self.assertEqual(matching_ids, ("1826278903",))
        projected = next(
            record
            for record in self.snapshot.records
            if record.source_id == matching_ids[0]
        )
        self.assertEqual(projected.point.longitude_e7, 0)
        self.assertEqual(projected.point.longitude, "0")

    def test_negative_zero_at_e7_precision_is_canonicalized(self) -> None:
        capture = self.capture
        source_id = capture.rows[0][capture.columns.index("id")]
        capture = replace_row(capture, 0, column="lat", value="-0.0000000")
        manifest = passing_manifest(capture)
        with (
            patch.object(audit, "capture_dataset", return_value=capture),
            patch.object(audit, "audit_capture", return_value=manifest),
        ):
            projected = load_city_snapshot(DATASET)

        record = next(
            record for record in projected.records if record.source_id == source_id
        )
        self.assertEqual(record.point.latitude_e7, 0)
        self.assertEqual(record.point.latitude, "0")

    def test_loader_captures_once_and_audits_the_same_object_without_reopen(
        self,
    ) -> None:
        captured: list[DatasetCapture] = []
        audited: list[DatasetCapture] = []
        original_capture = audit.capture_dataset
        original_audit = audit.audit_capture

        def capture_once(path: Path) -> DatasetCapture:
            result = original_capture(path)
            captured.append(result)
            return result

        def audit_same(
            capture: DatasetCapture,
            *,
            contract: audit.SnapshotContract,
        ) -> dict[str, Any]:
            audited.append(capture)
            return original_audit(capture, contract=contract)

        with (
            patch.object(audit, "capture_dataset", side_effect=capture_once) as capture,
            patch.object(audit, "audit_capture", side_effect=audit_same) as inspect,
        ):
            result = load_city_snapshot(DATASET)

        self.assertEqual(capture.call_count, 1)
        self.assertEqual(inspect.call_count, 1)
        self.assertIs(captured[0], audited[0])
        self.assertEqual(result.source_sha256, DEFAULT_CONTRACT.sha256)

    def test_snapshot_repr_and_serialization_hold_no_path_or_stat(self) -> None:
        rendered = repr(self.snapshot)
        serialized = dataclasses.asdict(self.snapshot)

        self.assertNotIn(str(REPOSITORY_ROOT), rendered)
        self.assertNotIn("path", rendered.lower())
        self.assertNotIn("stat", rendered.lower())
        self.assertNotIn("identity", rendered.lower())
        self.assertNotIn("records=", rendered)
        self.assertNotIn("path", serialized)
        self.assertNotIn("stat", serialized)
        self.assertNotIn("identity", serialized)
        self.assertEqual(
            set(serialized),
            {
                "source_sha256",
                "byte_size",
                "data_rows",
                "audit_schema",
                "audit_schema_version",
                "records",
            },
        )


class CitySnapshotFailureTests(unittest.TestCase):
    capture: ClassVar[DatasetCapture]
    snapshot: ClassVar[CitySnapshot]

    @classmethod
    def setUpClass(cls) -> None:
        cls.capture = audit.capture_dataset(DATASET)
        cls.snapshot = load_city_snapshot(DATASET)

    def assert_safe_failure(
        self,
        callable_: Any,
        *,
        secrets: tuple[str, ...] = (),
    ) -> SourceSnapshotError:
        with self.assertRaises(SourceSnapshotError) as raised:
            callable_()
        error = raised.exception
        self.assertTrue(error.code.startswith("snapshot."))
        self.assertEqual(str(error), error.message)
        for secret in secrets:
            self.assertNotIn(secret, str(error))
            self.assertNotIn(secret, repr(error))
        return error

    def test_capture_failure_hides_absolute_path_and_os_error(self) -> None:
        private_path = Path("/private/customer/never-present.csv")
        error = self.assert_safe_failure(
            lambda: load_city_snapshot(private_path),
            secrets=(str(private_path), "No such file"),
        )
        self.assertEqual(error.code, "snapshot.capture_failed")
        self.assertIsNone(error.__cause__)

    def test_hash_and_schema_mutations_fail_closed(self) -> None:
        bad_hash = dataclasses.replace(self.capture, sha256="0" * 64)
        bad_columns = dataclasses.replace(
            self.capture,
            columns=("country", *self.capture.columns[1:]),
        )
        for capture in (bad_hash, bad_columns):
            with self.subTest(capture=capture):
                with patch.object(audit, "capture_dataset", return_value=capture):
                    error = self.assert_safe_failure(load_city_snapshot)
                self.assertEqual(error.code, "snapshot.contract_mismatch")

    def test_mutated_audit_schema_and_incomplete_checks_fail_closed(self) -> None:
        for mutation in ("schema", "checks", "forged-check"):
            with self.subTest(mutation=mutation):
                manifest = audit.audit_capture(self.capture)
                if mutation == "schema":
                    manifest["manifest"]["schema_version"] = 2
                else:
                    checks = manifest["quality"]["contract_checks"]
                    if mutation == "checks":
                        checks.pop()
                    else:
                        checks[0]["actual"] = "forged"
                        checks[0]["expected"] = "forged"
                with (
                    patch.object(
                        audit,
                        "capture_dataset",
                        return_value=self.capture,
                    ),
                    patch.object(audit, "audit_capture", return_value=manifest),
                ):
                    error = self.assert_safe_failure(load_city_snapshot)
                expected_code = (
                    "snapshot.contract_mismatch"
                    if mutation == "forged-check"
                    else "snapshot.audit_invalid"
                )
                self.assertEqual(error.code, expected_code)

    def test_duplicate_id_noncanonical_coordinate_and_control_label_fail_closed(
        self,
    ) -> None:
        unsafe_cases = (
            (
                replace_row(
                    self.capture,
                    1,
                    column="id",
                    value=self.capture.rows[0][self.capture.columns.index("id")],
                ),
                "snapshot.duplicate_source_id",
                (),
            ),
            (
                replace_row(
                    self.capture,
                    0,
                    column="lat",
                    value="+1.0",
                ),
                "snapshot.record_invalid",
                ("+1.0",),
            ),
            (
                replace_row(
                    self.capture,
                    0,
                    column="city_ascii",
                    value="private\nlabel",
                ),
                "snapshot.record_invalid",
                ("private\nlabel",),
            ),
        )
        for capture, expected_code, secrets in unsafe_cases:
            with self.subTest(expected_code=expected_code):
                manifest = passing_manifest(capture)
                with (
                    patch.object(audit, "capture_dataset", return_value=capture),
                    patch.object(audit, "audit_capture", return_value=manifest),
                ):
                    error = self.assert_safe_failure(
                        load_city_snapshot,
                        secrets=secrets,
                    )
                self.assertEqual(error.code, expected_code)

    def test_forged_snapshot_and_records_are_revalidated(self) -> None:
        forged_snapshot = object.__new__(CitySnapshot)
        for name, value in (
            ("source_sha256", self.snapshot.source_sha256),
            ("byte_size", self.snapshot.byte_size),
            ("data_rows", self.snapshot.data_rows),
            ("audit_schema", self.snapshot.audit_schema),
            ("audit_schema_version", self.snapshot.audit_schema_version),
            ("records", self.snapshot.records[:-1]),
        ):
            object.__setattr__(forged_snapshot, name, value)
        error = self.assert_safe_failure(lambda: build_spatial_index(forged_snapshot))
        self.assertEqual(error.code, "snapshot.invalid")

        forged_point = object.__new__(GeoPoint)
        object.__setattr__(forged_point, "latitude_e7", 900_000_001)
        object.__setattr__(forged_point, "longitude_e7", 0)
        forged_record = object.__new__(SpatialRecord)
        object.__setattr__(forged_record, "source_id", "safe")
        object.__setattr__(forged_record, "city_ascii", "Safe")
        object.__setattr__(forged_record, "country", "Safe")
        object.__setattr__(forged_record, "iso2", "EX")
        object.__setattr__(forged_record, "point", forged_point)
        records = (forged_record, *self.snapshot.records[1:])
        forged_records_snapshot = object.__new__(CitySnapshot)
        for name, value in (
            ("source_sha256", self.snapshot.source_sha256),
            ("byte_size", self.snapshot.byte_size),
            ("data_rows", self.snapshot.data_rows),
            ("audit_schema", self.snapshot.audit_schema),
            ("audit_schema_version", self.snapshot.audit_schema_version),
            ("records", records),
        ):
            object.__setattr__(forged_records_snapshot, name, value)
        error = self.assert_safe_failure(
            lambda: build_spatial_index(forged_records_snapshot)
        )
        self.assertEqual(error.code, "snapshot.invalid")

    def test_city_snapshot_constructor_rejects_wrong_receipt(self) -> None:
        error = self.assert_safe_failure(
            lambda: CitySnapshot(
                source_sha256="private-secret",
                byte_size=self.snapshot.byte_size,
                data_rows=self.snapshot.data_rows,
                audit_schema=self.snapshot.audit_schema,
                audit_schema_version=self.snapshot.audit_schema_version,
                records=self.snapshot.records,
            ),
            secrets=("private-secret",),
        )
        self.assertEqual(error.code, "snapshot.invalid")


if __name__ == "__main__":
    unittest.main()
