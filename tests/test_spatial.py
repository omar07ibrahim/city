from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from urbanlens.geo import EARTH_RADIUS_METERS, GeoPoint, squared_chord_distance
from urbanlens.spatial import (
    MAX_INDEX_RECORDS,
    IndexVerificationError,
    QueryDiagnostics,
    QueryResult,
    SpatialIndex,
    SpatialRecord,
    _conservative_aabb_lower_bound,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def record(
    source_id: str,
    latitude: str,
    longitude: str,
    *,
    city_ascii: str | None = None,
    country: str = "Example Country",
    iso2: str = "EX",
) -> SpatialRecord:
    return SpatialRecord(
        source_id=source_id,
        city_ascii=city_ascii or f"City {source_id}",
        country=country,
        iso2=iso2,
        point=GeoPoint.from_decimal(latitude, longitude),
    )


def ordered_ids(index: SpatialIndex, point: GeoPoint, k: int) -> tuple[str, ...]:
    return tuple(
        neighbor.record.source_id
        for neighbor in index.query(point, k=k, verify=True).neighbors
    )


def random_vector(randomizer: random.Random) -> tuple[float, float, float]:
    return (
        randomizer.uniform(-1.0, 1.0),
        randomizer.uniform(-1.0, 1.0),
        randomizer.uniform(-1.0, 1.0),
    )


class SpatialRecordValidationTests(unittest.TestCase):
    def test_accepts_only_allowlisted_bounded_safe_fields(self) -> None:
        accepted = record(
            "source:001",
            "10",
            "20",
            city_ascii="St. John's",
            country="Côte d'Ivoire",
            iso2="CI",
        )
        self.assertEqual(accepted.source_id, "source:001")
        self.assertEqual(accepted.point.decimal_pair, ("10", "20"))

        invalid_overrides: tuple[dict[str, Any], ...] = (
            {"source_id": ""},
            {"source_id": 1},
            {"source_id": "bad id"},
            {"source_id": "a" * 65},
            {"city_ascii": 1},
            {"city_ascii": "München"},
            {"city_ascii": "bad\ncity"},
            {"city_ascii": " city"},
            {"city_ascii": "a" * 129},
            {"country": 1},
            {"country": ""},
            {"country": "bad\x00country"},
            {"country": "country "},
            {"country": "a" * 129},
            {"iso2": 1},
            {"iso2": "ex"},
            {"iso2": "USA"},
            {"point": "0,0"},
        )
        base: dict[str, Any] = {
            "source_id": "safe",
            "city_ascii": "Safe City",
            "country": "Safe Country",
            "iso2": "EX",
            "point": GeoPoint.from_decimal("0", "0"),
        }
        for override in invalid_overrides:
            with self.subTest(override=override):
                values = base | override
                with self.assertRaises((TypeError, ValueError)):
                    SpatialRecord(**values)

    def test_index_rejects_invalid_container_size_and_duplicate_ids(self) -> None:
        with self.assertRaises(TypeError):
            SpatialIndex([])  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            SpatialIndex(())
        with self.assertRaises(TypeError):
            SpatialIndex(("not-a-record",))  # type: ignore[arg-type]
        duplicate_ids = (
            record("same", "0", "0"),
            record("same", "1", "1"),
        )
        with self.assertRaises(ValueError):
            SpatialIndex(duplicate_ids)

        repeated = record("one", "0", "0")
        with self.assertRaises(ValueError):
            SpatialIndex((repeated,) * (MAX_INDEX_RECORDS + 1))

    def test_diagnostics_reject_negative_accounting(self) -> None:
        with self.assertRaises(ValueError):
            QueryDiagnostics(evaluated=-1, pruned=0, visited=0)

    def test_index_revalidates_forged_records_before_construction(self) -> None:
        forged_point = object.__new__(GeoPoint)
        object.__setattr__(forged_point, "latitude_e7", 900_000_001)
        object.__setattr__(forged_point, "longitude_e7", 0)
        forged_record = object.__new__(SpatialRecord)
        object.__setattr__(forged_record, "source_id", "safe")
        object.__setattr__(forged_record, "city_ascii", "Safe")
        object.__setattr__(forged_record, "country", "Safe")
        object.__setattr__(forged_record, "iso2", "EX")
        object.__setattr__(forged_record, "point", forged_point)

        with self.assertRaises(ValueError):
            SpatialIndex((forged_record,))


class ExactQueryTests(unittest.TestCase):
    def test_aabb_lower_bound_never_exceeds_an_enclosed_point_distance(
        self,
    ) -> None:
        randomizer = random.Random(7_301)
        for _ in range(20_000):
            first = random_vector(randomizer)
            second = random_vector(randomizer)
            minimum = (
                min(first[0], second[0]),
                min(first[1], second[1]),
                min(first[2], second[2]),
            )
            maximum = (
                max(first[0], second[0]),
                max(first[1], second[1]),
                max(first[2], second[2]),
            )
            enclosed = (
                randomizer.uniform(minimum[0], maximum[0]),
                randomizer.uniform(minimum[1], maximum[1]),
                randomizer.uniform(minimum[2], maximum[2]),
            )
            query = random_vector(randomizer)
            lower_bound = _conservative_aabb_lower_bound(
                query,
                (minimum, maximum),
            )
            self.assertLessEqual(
                lower_bound,
                squared_chord_distance(query, enclosed),
            )

    def test_duplicate_locations_are_preserved_and_tied_by_source_id(self) -> None:
        records = (
            record("z-last", "0", "0"),
            record("a-first", "0", "0"),
            record("middle", "1", "1"),
        )
        index = SpatialIndex(records)
        result = index.query(GeoPoint.from_decimal("0", "0"), k=3, verify=True)
        self.assertEqual(
            tuple(neighbor.record.source_id for neighbor in result.neighbors),
            ("a-first", "z-last", "middle"),
        )
        self.assertEqual(result.neighbors[0].squared_chord_distance, 0.0)
        self.assertEqual(result.neighbors[1].squared_chord_distance, 0.0)
        self.assertEqual(index.metadata.records, 3)
        self.assertEqual(index.metadata.unique_numeric_locations, 2)
        self.assertEqual(index.metadata.collision_groups, 1)
        self.assertEqual(index.metadata.collision_excess, 1)
        self.assertEqual(index.metadata.max_location_multiplicity, 2)

    def test_equidistant_ties_survive_pruning_equality(self) -> None:
        records = (
            record("z-west", "0", "-1"),
            record("a-east", "0", "1"),
            record("north", "10", "0"),
            record("south", "-10", "0"),
            record("far", "80", "120"),
        )
        index = SpatialIndex(records)
        query = GeoPoint.from_decimal("0", "0")
        result = index.query(query, k=1, verify=True)
        self.assertEqual(result.neighbors[0].record.source_id, "a-east")
        self.assertEqual(
            result.diagnostics.evaluated + result.diagnostics.pruned,
            index.metadata.records,
        )
        self.assertGreaterEqual(result.diagnostics.visited, 1)

    def test_antimeridian_poles_and_antipode_rank_correctly(self) -> None:
        records = (
            record("dateline-east", "0", "179.9"),
            record("dateline-west", "0", "-179.7"),
            record("north-pole", "90", "0"),
            record("antipode", "0", "0"),
        )
        index = SpatialIndex(records)
        self.assertEqual(
            ordered_ids(index, GeoPoint.from_decimal("0", "-179.95"), 2),
            ("dateline-east", "dateline-west"),
        )
        self.assertEqual(
            ordered_ids(index, GeoPoint.from_decimal("89.999", "-99"), 1),
            ("north-pole",),
        )
        antipode_result = index.query(
            GeoPoint.from_decimal("0", "180"),
            k=4,
            verify=True,
        )
        self.assertLessEqual(
            antipode_result.neighbors[-1].distance_meters,
            3.141592653589793 * EARTH_RADIUS_METERS,
        )

    def test_k_bounds_and_full_scan_diagnostics(self) -> None:
        index = SpatialIndex((record("only", "0", "0"),))
        for invalid in (True, False, 0, -1, 101, 1.0, "1"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                index.query(
                    GeoPoint.from_decimal("0", "0"),
                    k=invalid,  # type: ignore[arg-type]
                )
        with self.assertRaises(TypeError):
            index.query("0,0")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            index.query(GeoPoint.from_decimal("0", "0"), verify=1)  # type: ignore[arg-type]
        oracle = index.query_full_scan(GeoPoint.from_decimal("1", "1"), k=100)
        self.assertEqual(len(oracle.neighbors), 1)
        self.assertEqual(oracle.diagnostics.evaluated, 1)
        self.assertEqual(oracle.diagnostics.pruned, 0)
        self.assertEqual(oracle.diagnostics.visited, 1)

    def test_verify_fails_closed_when_oracle_order_disagrees(self) -> None:
        index = SpatialIndex(
            (
                record("near", "0", "0"),
                record("far", "40", "40"),
            )
        )
        false_oracle = QueryResult(
            neighbors=(),
            diagnostics=QueryDiagnostics(evaluated=2, pruned=0, visited=2),
        )
        with (
            patch.object(
                SpatialIndex,
                "query_full_scan",
                return_value=false_oracle,
            ),
            self.assertRaises(IndexVerificationError),
        ):
            index.query(GeoPoint.from_decimal("0", "0"), verify=True)

    def test_seeded_random_queries_match_full_scan_exactly(self) -> None:
        randomizer = random.Random(2_026_073_0)
        records = tuple(
            record(
                f"id-{index:04d}",
                str(randomizer.randint(-900_000_000, 900_000_000) / 10_000_000),
                str(randomizer.randint(-1_800_000_000, 1_800_000_000) / 10_000_000),
            )
            for index in range(300)
        )
        index = SpatialIndex(records)
        saw_pruning = False
        for query_number in range(100):
            query = GeoPoint.from_e7(
                randomizer.randint(-900_000_000, 900_000_000),
                randomizer.randint(-1_800_000_000, 1_800_000_000),
            )
            k = 1 + query_number % 17
            accelerated = index.query(query, k=k, verify=True)
            oracle = index.query_full_scan(query, k=k)
            self.assertEqual(
                tuple(neighbor.rank_key for neighbor in accelerated.neighbors),
                tuple(neighbor.rank_key for neighbor in oracle.neighbors),
            )
            self.assertEqual(
                accelerated.diagnostics.evaluated + accelerated.diagnostics.pruned,
                index.metadata.records,
            )
            saw_pruning |= accelerated.diagnostics.pruned > 0
        self.assertTrue(saw_pruning)


class DeterminismTests(unittest.TestCase):
    records = (
        record("a", "0", "0"),
        record("b", "0", "0"),
        record("c", "45", "45"),
        record("d", "-45", "-45"),
        record("e", "80", "179"),
        record("f", "-80", "-179"),
        record("g", "12.5", "-100.25"),
    )

    def test_shuffled_input_has_identical_topology_and_results(self) -> None:
        expected = SpatialIndex(self.records)
        expected_ids = ordered_ids(
            expected,
            GeoPoint.from_decimal("1", "1"),
            len(self.records),
        )
        for seed in range(20):
            shuffled = list(self.records)
            random.Random(seed).shuffle(shuffled)
            candidate = SpatialIndex(tuple(shuffled))
            self.assertEqual(
                candidate.metadata.topology_sha256,
                expected.metadata.topology_sha256,
            )
            self.assertEqual(
                ordered_ids(
                    candidate,
                    GeoPoint.from_decimal("1", "1"),
                    len(self.records),
                ),
                expected_ids,
            )
            self.assertEqual(
                candidate.query(
                    GeoPoint.from_decimal("1", "1"),
                    k=3,
                ).diagnostics,
                expected.query(
                    GeoPoint.from_decimal("1", "1"),
                    k=3,
                ).diagnostics,
            )

    def test_hash_seed_does_not_change_topology_or_query_order(self) -> None:
        script = """
import json
from urbanlens.geo import GeoPoint
from urbanlens.spatial import SpatialIndex, SpatialRecord
rows = {
    ("a", "0", "0"), ("b", "0", "0"), ("c", "45", "45"),
    ("d", "-45", "-45"), ("e", "80", "179"), ("f", "-80", "-179"),
    ("g", "12.5", "-100.25"),
}
records = tuple(
    SpatialRecord(
        source_id=source_id,
        city_ascii="City " + source_id,
        country="Example",
        iso2="EX",
        point=GeoPoint.from_decimal(latitude, longitude),
    )
    for source_id, latitude, longitude in rows
)
index = SpatialIndex(records)
result = index.query(GeoPoint.from_decimal("1", "1"), k=7, verify=True)
print(json.dumps({
    "topology": index.metadata.topology_sha256,
    "ids": [neighbor.record.source_id for neighbor in result.neighbors],
}, sort_keys=True))
"""
        outputs: set[str] = set()
        for seed in ("1", "2", "77", "random"):
            environment = os.environ.copy()
            environment["PYTHONHASHSEED"] = seed
            completed = subprocess.run(
                [sys.executable, "-c", script],
                cwd=REPOSITORY_ROOT,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
                timeout=20,
            )
            outputs.add(completed.stdout.strip())
        self.assertEqual(len(outputs), 1)
        payload = json.loads(outputs.pop())
        self.assertEqual(len(payload["topology"]), 64)
        self.assertEqual(len(payload["ids"]), 7)

    def test_metadata_declares_algorithm_numeric_contract(self) -> None:
        metadata = SpatialIndex(self.records).metadata
        self.assertEqual(metadata.records, 7)
        self.assertEqual(metadata.algorithm, "balanced-unit-sphere-kd-tree")
        self.assertEqual(metadata.version, 1)
        self.assertEqual(metadata.earth_radius_meters, 6_371_008.8)
        self.assertEqual(metadata.coordinate_scale, 10_000_000)
        self.assertGreaterEqual(metadata.depth, 3)
        self.assertEqual(len(metadata.topology_sha256), 64)
        int(metadata.topology_sha256, 16)


if __name__ == "__main__":
    unittest.main()
