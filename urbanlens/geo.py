"""Canonical geographic coordinates and robust spherical distances.

Coordinates enter the spatial layer as strict ASCII decimal strings and are
converted to signed E7 integers before any floating-point work.  This gives
callers one unambiguous representation for hashing, equality, and receipts
while retaining sub-metre input precision.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Final

COORDINATE_SCALE: Final = 10_000_000
EARTH_RADIUS_METERS: Final = 6_371_008.8

_LATITUDE_LIMIT_E7: Final = 90 * COORDINATE_SCALE
_LONGITUDE_LIMIT_E7: Final = 180 * COORDINATE_SCALE
_MAX_DECIMAL_CHARACTERS: Final = 32
_ASCII_DECIMAL = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]{1,7})?", re.ASCII)


class CoordinateInputError(ValueError):
    """Raised when a coordinate is not a strict, bounded ASCII decimal."""


def _parse_decimal_e7(value: object, *, name: str, limit_e7: int) -> int:
    if type(value) is not str:
        raise CoordinateInputError(f"{name} must be an ASCII decimal string")
    if len(value) > _MAX_DECIMAL_CHARACTERS or _ASCII_DECIMAL.fullmatch(value) is None:
        raise CoordinateInputError(
            f"{name} must use canonical ASCII decimal notation with at most 7 places"
        )

    negative = value.startswith("-")
    unsigned = value[1:] if negative else value
    whole_text, separator, fractional_text = unsigned.partition(".")
    fractional_e7 = int(fractional_text.ljust(7, "0")) if separator else 0
    parsed = int(whole_text) * COORDINATE_SCALE + fractional_e7
    if negative:
        parsed = -parsed
    if not -limit_e7 <= parsed <= limit_e7:
        raise CoordinateInputError(f"{name} is outside its geographic bounds")
    return parsed


def _format_e7(value: int) -> str:
    if value == 0:
        return "0"
    sign = "-" if value < 0 else ""
    magnitude = abs(value)
    whole, fractional = divmod(magnitude, COORDINATE_SCALE)
    if fractional == 0:
        return f"{sign}{whole}"
    return f"{sign}{whole}.{fractional:07d}".rstrip("0")


@dataclass(frozen=True, slots=True)
class GeoPoint:
    """One canonical WGS84-style latitude/longitude pair at E7 precision."""

    latitude_e7: int
    longitude_e7: int

    def __post_init__(self) -> None:
        if type(self.latitude_e7) is not int or type(self.longitude_e7) is not int:
            raise TypeError("GeoPoint E7 coordinates must be integers")
        if not -_LATITUDE_LIMIT_E7 <= self.latitude_e7 <= _LATITUDE_LIMIT_E7:
            raise ValueError("latitude_e7 is outside [-90e7, 90e7]")
        if not -_LONGITUDE_LIMIT_E7 <= self.longitude_e7 < _LONGITUDE_LIMIT_E7:
            raise ValueError(
                "longitude_e7 must use the canonical [-180e7, 180e7) range"
            )
        if abs(self.latitude_e7) == _LATITUDE_LIMIT_E7 and self.longitude_e7 != 0:
            raise ValueError("longitude_e7 must be zero at either pole")

    @classmethod
    def from_decimal(cls, latitude: object, longitude: object) -> GeoPoint:
        """Parse two strict ASCII decimals and return their canonical E7 point."""

        latitude_e7 = _parse_decimal_e7(
            latitude,
            name="latitude",
            limit_e7=_LATITUDE_LIMIT_E7,
        )
        longitude_e7 = _parse_decimal_e7(
            longitude,
            name="longitude",
            limit_e7=_LONGITUDE_LIMIT_E7,
        )
        return cls.from_e7(latitude_e7, longitude_e7)

    @classmethod
    def from_e7(cls, latitude_e7: int, longitude_e7: int) -> GeoPoint:
        """Canonicalize already-scaled integer coordinates."""

        if type(latitude_e7) is not int or type(longitude_e7) is not int:
            raise TypeError("E7 coordinates must be integers, not booleans or floats")
        if not -_LATITUDE_LIMIT_E7 <= latitude_e7 <= _LATITUDE_LIMIT_E7:
            raise ValueError("latitude_e7 is outside [-90e7, 90e7]")
        if not -_LONGITUDE_LIMIT_E7 <= longitude_e7 <= _LONGITUDE_LIMIT_E7:
            raise ValueError("longitude_e7 is outside [-180e7, 180e7]")
        if longitude_e7 == _LONGITUDE_LIMIT_E7:
            longitude_e7 = -_LONGITUDE_LIMIT_E7
        if abs(latitude_e7) == _LATITUDE_LIMIT_E7:
            longitude_e7 = 0
        return cls(latitude_e7, longitude_e7)

    @property
    def latitude(self) -> str:
        """Canonical latitude decimal without insignificant zeroes."""

        return _format_e7(self.latitude_e7)

    @property
    def longitude(self) -> str:
        """Canonical longitude decimal without insignificant zeroes."""

        return _format_e7(self.longitude_e7)

    @property
    def decimal_pair(self) -> tuple[str, str]:
        return self.latitude, self.longitude


def unit_vector(point: GeoPoint) -> tuple[float, float, float]:
    """Project a canonical point onto the binary64 unit sphere."""

    if type(point) is not GeoPoint:
        raise TypeError("point must be a GeoPoint")
    point.__post_init__()
    if abs(point.latitude_e7) == _LATITUDE_LIMIT_E7:
        return (0.0, 0.0, 1.0 if point.latitude_e7 > 0 else -1.0)
    latitude = math.radians(point.latitude_e7 / COORDINATE_SCALE)
    longitude = math.radians(point.longitude_e7 / COORDINATE_SCALE)
    cos_latitude = math.cos(latitude)
    return (
        cos_latitude * math.cos(longitude),
        cos_latitude * math.sin(longitude),
        math.sin(latitude),
    )


def squared_chord_distance(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> float:
    """Return the binary64 squared chord distance between two unit vectors."""

    dx = first[0] - second[0]
    dy = first[1] - second[1]
    dz = first[2] - second[2]
    return dx * dx + dy * dy + dz * dz


def chord_distance_to_meters(squared_chord: float) -> float:
    """Convert a squared unit-sphere chord to great-circle metres."""

    if not math.isfinite(squared_chord) or squared_chord < 0.0:
        raise ValueError("squared_chord must be a finite non-negative number")
    half_chord = min(1.0, math.sqrt(squared_chord) / 2.0)
    return 2.0 * EARTH_RADIUS_METERS * math.asin(half_chord)


def haversine_meters(first: GeoPoint, second: GeoPoint) -> float:
    """Return a numerically robust great-circle distance in metres."""

    if type(first) is not GeoPoint or type(second) is not GeoPoint:
        raise TypeError("haversine_meters requires two GeoPoint values")
    first.__post_init__()
    second.__post_init__()
    if first == second:
        return 0.0

    first_latitude = math.radians(first.latitude_e7 / COORDINATE_SCALE)
    second_latitude = math.radians(second.latitude_e7 / COORDINATE_SCALE)
    latitude_delta = second_latitude - first_latitude
    longitude_delta_e7 = (
        second.longitude_e7 - first.longitude_e7 + _LONGITUDE_LIMIT_E7
    ) % (2 * _LONGITUDE_LIMIT_E7) - _LONGITUDE_LIMIT_E7
    longitude_delta = math.radians(longitude_delta_e7 / COORDINATE_SCALE)

    sin_half_latitude = math.sin(latitude_delta / 2.0)
    sin_half_longitude = math.sin(longitude_delta / 2.0)
    haversine = math.fsum(
        (
            sin_half_latitude * sin_half_latitude,
            math.cos(first_latitude)
            * math.cos(second_latitude)
            * sin_half_longitude
            * sin_half_longitude,
        )
    )
    haversine = min(1.0, max(0.0, haversine))
    angle = 2.0 * math.atan2(
        math.sqrt(haversine),
        math.sqrt(max(0.0, 1.0 - haversine)),
    )
    return EARTH_RADIUS_METERS * angle
