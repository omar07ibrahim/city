"""Deterministic exact nearest-neighbour search on the unit sphere.

The balanced kd-tree is an acceleration structure, not an approximation.  It
ranks records by the exact binary64 key ``(squared chord distance, source_id)``
and only prunes a subtree when a conservatively rounded AABB lower bound is
strictly greater than an upward-rounded incumbent distance.
"""

from __future__ import annotations

import bisect
import hashlib
import math
import re
import struct
import unicodedata
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final

from urbanlens.geo import (
    COORDINATE_SCALE,
    EARTH_RADIUS_METERS,
    GeoPoint,
    chord_distance_to_meters,
    squared_chord_distance,
    unit_vector,
)

MAX_INDEX_RECORDS: Final = 50_000
MAX_QUERY_K: Final = 100
INDEX_ALGORITHM: Final = "balanced-unit-sphere-kd-tree"
INDEX_VERSION: Final = 1

_MAX_SOURCE_ID_CHARACTERS: Final = 64
_MAX_CITY_ASCII_CHARACTERS: Final = 128
_MAX_COUNTRY_CHARACTERS: Final = 128
_SOURCE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,63}", re.ASCII)
_ISO2 = re.compile(r"[A-Z]{2}", re.ASCII)

_Vector = tuple[float, float, float]
_Bounds = tuple[_Vector, _Vector]
_RankKey = tuple[float, str]


class IndexVerificationError(AssertionError):
    """Raised when accelerated and full-scan ordered results disagree."""


def _has_control_character(value: str) -> bool:
    return any(unicodedata.category(character).startswith("C") for character in value)


def _validate_label(
    value: object,
    *,
    name: str,
    maximum_characters: int,
    ascii_only: bool,
) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if not 1 <= len(value) <= maximum_characters:
        raise ValueError(f"{name} must contain 1..{maximum_characters} characters")
    if value != value.strip():
        raise ValueError(f"{name} must not contain surrounding whitespace")
    if ascii_only and not value.isascii():
        raise ValueError(f"{name} must contain ASCII characters only")
    if _has_control_character(value):
        raise ValueError(f"{name} must not contain control characters")
    return value


@dataclass(frozen=True, slots=True)
class SpatialRecord:
    """Allowlisted source fields attached to one canonical numeric location."""

    source_id: str
    city_ascii: str
    country: str
    iso2: str
    point: GeoPoint

    def __post_init__(self) -> None:
        if (
            type(self.source_id) is not str
            or _SOURCE_ID.fullmatch(self.source_id) is None
        ):
            raise ValueError("source_id must be 1..64 safe ASCII identifier characters")
        _validate_label(
            self.city_ascii,
            name="city_ascii",
            maximum_characters=_MAX_CITY_ASCII_CHARACTERS,
            ascii_only=True,
        )
        _validate_label(
            self.country,
            name="country",
            maximum_characters=_MAX_COUNTRY_CHARACTERS,
            ascii_only=False,
        )
        if type(self.iso2) is not str or _ISO2.fullmatch(self.iso2) is None:
            raise ValueError("iso2 must contain exactly two uppercase ASCII letters")
        if type(self.point) is not GeoPoint:
            raise TypeError("point must be a GeoPoint")
        self.point.__post_init__()


@dataclass(frozen=True, slots=True)
class NearestNeighbor:
    """One ranked spatial match."""

    record: SpatialRecord
    squared_chord_distance: float
    distance_meters: float

    @property
    def rank_key(self) -> _RankKey:
        return self.squared_chord_distance, self.record.source_id


@dataclass(frozen=True, slots=True)
class QueryDiagnostics:
    """Accounting for one exact tree or oracle traversal."""

    evaluated: int
    pruned: int
    visited: int

    def __post_init__(self) -> None:
        if min(self.evaluated, self.pruned, self.visited) < 0:
            raise ValueError("query diagnostics cannot be negative")


@dataclass(frozen=True, slots=True)
class QueryResult:
    """Ordered matches and reproducible traversal diagnostics."""

    neighbors: tuple[NearestNeighbor, ...]
    diagnostics: QueryDiagnostics


@dataclass(frozen=True, slots=True)
class SpatialMetadata:
    """Stable structural and duplicate-location facts for one index."""

    records: int
    unique_numeric_locations: int
    collision_groups: int
    collision_excess: int
    max_location_multiplicity: int
    depth: int
    topology_sha256: str
    algorithm: str = INDEX_ALGORITHM
    version: int = INDEX_VERSION
    earth_radius_meters: float = EARTH_RADIUS_METERS
    coordinate_scale: int = COORDINATE_SCALE


@dataclass(frozen=True, slots=True)
class _IndexedRecord:
    record: SpatialRecord
    vector: _Vector


@dataclass(frozen=True, slots=True)
class _Node:
    item: _IndexedRecord
    axis: int
    left: _Node | None
    right: _Node | None
    bounds: _Bounds
    size: int
    depth: int


def _greatest_span_axis(items: Sequence[_IndexedRecord]) -> int:
    minima = [min(item.vector[axis] for item in items) for axis in range(3)]
    maxima = [max(item.vector[axis] for item in items) for axis in range(3)]
    spans = [maxima[axis] - minima[axis] for axis in range(3)]
    # ``max`` returns the first entry on an exact tie: x, then y, then z.
    return max(range(3), key=spans.__getitem__)


def _merge_bounds(
    item: _IndexedRecord,
    left: _Node | None,
    right: _Node | None,
) -> _Bounds:
    minima = list(item.vector)
    maxima = list(item.vector)
    for child in (left, right):
        if child is None:
            continue
        for axis in range(3):
            minima[axis] = min(minima[axis], child.bounds[0][axis])
            maxima[axis] = max(maxima[axis], child.bounds[1][axis])
    return (
        (minima[0], minima[1], minima[2]),
        (maxima[0], maxima[1], maxima[2]),
    )


def _build(items: Sequence[_IndexedRecord]) -> _Node | None:
    if not items:
        return None
    axis = _greatest_span_axis(items)
    ordered = sorted(
        items,
        key=lambda item: (item.vector[axis], item.record.source_id),
    )
    median = (len(ordered) - 1) // 2
    item = ordered[median]
    left = _build(ordered[:median])
    right = _build(ordered[median + 1 :])
    size = 1
    depth = 1
    for child in (left, right):
        if child is not None:
            size += child.size
            depth = max(depth, child.depth + 1)
    return _Node(
        item=item,
        axis=axis,
        left=left,
        right=right,
        bounds=_merge_bounds(item, left, right),
        size=size,
        depth=depth,
    )


def _down(value: float) -> float:
    return math.nextafter(value, 0.0) if value > 0.0 else 0.0


def _conservative_aabb_lower_bound(query: _Vector, bounds: _Bounds) -> float:
    """Lower-bound squared distance without rounding above the true value."""

    lower_bound = 0.0
    for axis in range(3):
        if query[axis] < bounds[0][axis]:
            gap = bounds[0][axis] - query[axis]
        elif query[axis] > bounds[1][axis]:
            gap = query[axis] - bounds[1][axis]
        else:
            gap = 0.0
        downward_gap = _down(gap)
        downward_square = _down(downward_gap * downward_gap)
        lower_bound = _down(lower_bound + downward_square)
    return lower_bound


def _topology_sha256(root: _Node) -> str:
    digest = hashlib.sha256()
    digest.update(b"urbanlens.spatial.topology\0")
    digest.update(struct.pack(">I", INDEX_VERSION))

    def visit(node: _Node | None) -> None:
        if node is None:
            digest.update(b"\x00")
            return
        digest.update(b"\x01")
        digest.update(bytes((node.axis,)))
        source_id = node.item.record.source_id.encode("ascii")
        digest.update(struct.pack(">H", len(source_id)))
        digest.update(source_id)
        digest.update(
            struct.pack(
                ">qq",
                node.item.record.point.latitude_e7,
                node.item.record.point.longitude_e7,
            )
        )
        visit(node.left)
        visit(node.right)

    visit(root)
    return digest.hexdigest()


def _metadata(root: _Node, records: Sequence[SpatialRecord]) -> SpatialMetadata:
    locations = Counter(
        (record.point.latitude_e7, record.point.longitude_e7) for record in records
    )
    multiplicities = tuple(locations.values())
    return SpatialMetadata(
        records=len(records),
        unique_numeric_locations=len(locations),
        collision_groups=sum(count > 1 for count in multiplicities),
        collision_excess=sum(max(0, count - 1) for count in multiplicities),
        max_location_multiplicity=max(multiplicities),
        depth=root.depth,
        topology_sha256=_topology_sha256(root),
    )


def _validate_query(point: object, k: object) -> tuple[GeoPoint, int]:
    if type(point) is not GeoPoint:
        raise TypeError("query point must be a GeoPoint")
    point.__post_init__()
    if type(k) is not int or not 1 <= k <= MAX_QUERY_K:
        raise ValueError(f"k must be an integer in [1, {MAX_QUERY_K}]")
    return point, k


def _neighbor(item: _IndexedRecord, query: _Vector) -> NearestNeighbor:
    distance = squared_chord_distance(query, item.vector)
    return NearestNeighbor(
        record=item.record,
        squared_chord_distance=distance,
        distance_meters=chord_distance_to_meters(distance),
    )


@dataclass(frozen=True, slots=True, init=False)
class SpatialIndex:
    """Immutable, deterministic, exact balanced kd-tree."""

    _root: _Node = field(repr=False)
    _records: tuple[SpatialRecord, ...] = field(repr=False)
    metadata: SpatialMetadata

    def __init__(self, records: tuple[SpatialRecord, ...]) -> None:
        if type(records) is not tuple:
            raise TypeError("records must be a tuple of SpatialRecord values")
        if not 1 <= len(records) <= MAX_INDEX_RECORDS:
            raise ValueError(f"records must contain 1..{MAX_INDEX_RECORDS} entries")
        for record in records:
            if type(record) is not SpatialRecord:
                raise TypeError("records must contain only SpatialRecord values")
            record.__post_init__()
        source_ids = [record.source_id for record in records]
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("source_id values must be unique within an index")

        canonical_records = tuple(sorted(records, key=lambda record: record.source_id))
        indexed = tuple(
            _IndexedRecord(record=record, vector=unit_vector(record.point))
            for record in canonical_records
        )
        root = _build(indexed)
        if root is None:  # pragma: no cover - constructor rejects empty input
            raise RuntimeError("cannot build an empty spatial index")
        object.__setattr__(self, "_root", root)
        object.__setattr__(self, "_records", canonical_records)
        object.__setattr__(self, "metadata", _metadata(root, canonical_records))

    def query(
        self,
        point: GeoPoint,
        *,
        k: int = 1,
        verify: bool = False,
    ) -> QueryResult:
        """Return the exact nearest records in stable rank-key order."""

        point, k = _validate_query(point, k)
        if type(verify) is not bool:
            raise TypeError("verify must be a boolean")
        query_vector = unit_vector(point)
        best: list[tuple[_RankKey, NearestNeighbor]] = []
        evaluated = 0
        pruned = 0
        visited = 0

        def should_prune(node: _Node) -> bool:
            if len(best) < k:
                return False
            lower_bound = _conservative_aabb_lower_bound(
                query_vector,
                node.bounds,
            )
            upward_threshold = math.nextafter(best[-1][0][0], math.inf)
            # Strict inequality is essential: an equal-distance record may win
            # the declared source-id tie-break.
            return lower_bound > upward_threshold

        def visit(node: _Node | None) -> None:
            nonlocal evaluated, pruned, visited
            if node is None:
                return
            visited += 1
            if should_prune(node):
                pruned += node.size
                return

            candidate = _neighbor(node.item, query_vector)
            evaluated += 1
            entry = (candidate.rank_key, candidate)
            insertion = bisect.bisect_left(
                [existing[0] for existing in best],
                candidate.rank_key,
            )
            best.insert(insertion, entry)
            if len(best) > k:
                best.pop()

            if query_vector[node.axis] <= node.item.vector[node.axis]:
                first, second = node.left, node.right
            else:
                first, second = node.right, node.left
            visit(first)
            visit(second)

        visit(self._root)
        diagnostics = QueryDiagnostics(
            evaluated=evaluated,
            pruned=pruned,
            visited=visited,
        )
        if (  # pragma: no cover - recursive accounting invariant
            diagnostics.evaluated + diagnostics.pruned != self.metadata.records
        ):
            raise RuntimeError("tree traversal record accounting invariant failed")
        result = QueryResult(
            neighbors=tuple(entry[1] for entry in best),
            diagnostics=diagnostics,
        )

        if verify:
            oracle = self.query_full_scan(point, k=k)
            accelerated_keys = tuple(neighbor.rank_key for neighbor in result.neighbors)
            oracle_keys = tuple(neighbor.rank_key for neighbor in oracle.neighbors)
            if accelerated_keys != oracle_keys:
                raise IndexVerificationError(
                    "accelerated result differs from the full-scan ordered rank keys"
                )
        return result

    def query_full_scan(self, point: GeoPoint, *, k: int = 1) -> QueryResult:
        """Return the exact full-scan oracle result without tree pruning."""

        point, k = _validate_query(point, k)
        query_vector = unit_vector(point)
        neighbors = sorted(
            (
                _neighbor(
                    _IndexedRecord(record=record, vector=unit_vector(record.point)),
                    query_vector,
                )
                for record in self._records
            ),
            key=lambda neighbor: neighbor.rank_key,
        )[:k]
        count = self.metadata.records
        return QueryResult(
            neighbors=tuple(neighbors),
            diagnostics=QueryDiagnostics(
                evaluated=count,
                pruned=0,
                visited=count,
            ),
        )
