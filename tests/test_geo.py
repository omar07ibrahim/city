from __future__ import annotations

import math
import unittest

from urbanlens.geo import (
    COORDINATE_SCALE,
    EARTH_RADIUS_METERS,
    CoordinateInputError,
    GeoPoint,
    chord_distance_to_meters,
    haversine_meters,
    squared_chord_distance,
    unit_vector,
)


class CanonicalCoordinateTests(unittest.TestCase):
    def test_decimal_factory_canonicalizes_zero_dateline_and_poles(self) -> None:
        self.assertEqual(
            GeoPoint.from_decimal("-0.0000000", "-0").decimal_pair,
            ("0", "0"),
        )
        self.assertEqual(
            GeoPoint.from_decimal("12.3400000", "180").decimal_pair,
            ("12.34", "-180"),
        )
        self.assertEqual(
            GeoPoint.from_decimal("90.0000000", "73.2").decimal_pair,
            ("90", "0"),
        )
        self.assertEqual(
            GeoPoint.from_decimal("-90", "-180").decimal_pair,
            ("-90", "0"),
        )

    def test_decimal_factory_retains_exact_e7_precision(self) -> None:
        point = GeoPoint.from_decimal("-12.3456789", "45.0000001")
        self.assertEqual(point.latitude_e7, -123_456_789)
        self.assertEqual(point.longitude_e7, 450_000_001)
        self.assertEqual(point.decimal_pair, ("-12.3456789", "45.0000001"))

    def test_rejects_noncanonical_or_ambiguous_coordinate_text(self) -> None:
        invalid_values: tuple[object, ...] = (
            True,
            False,
            1,
            1.0,
            "",
            " 1",
            "1 ",
            "+1",
            ".1",
            "1.",
            "01",
            "--1",
            "1e2",
            "1E2",
            "1_0",
            "\uff11\uff12",
            "\u0661\u0662",
            "NaN",
            "Inf",
            "-Infinity",
            "0.00000000",
            "1.12345678",
        )
        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises(CoordinateInputError):
                    GeoPoint.from_decimal(value, "0")
                with self.assertRaises(CoordinateInputError):
                    GeoPoint.from_decimal("0", value)

    def test_rejects_out_of_bounds_coordinates(self) -> None:
        for latitude in ("90.0000001", "-90.0000001", "91", "-999"):
            with (
                self.subTest(latitude=latitude),
                self.assertRaises(CoordinateInputError),
            ):
                GeoPoint.from_decimal(latitude, "0")
        for longitude in ("180.0000001", "-180.0000001", "181", "-999"):
            with (
                self.subTest(longitude=longitude),
                self.assertRaises(CoordinateInputError),
            ):
                GeoPoint.from_decimal("0", longitude)

    def test_e7_factory_rejects_boolean_and_canonicalizes_boundaries(self) -> None:
        with self.assertRaises(TypeError):
            GeoPoint.from_e7(True, 0)
        with self.assertRaises(TypeError):
            GeoPoint.from_e7(0, False)
        with self.assertRaises(ValueError):
            GeoPoint.from_e7(900_000_001, 0)
        with self.assertRaises(ValueError):
            GeoPoint.from_e7(0, 1_800_000_001)
        self.assertEqual(
            GeoPoint.from_e7(0, 180 * COORDINATE_SCALE).longitude_e7,
            -180 * COORDINATE_SCALE,
        )
        self.assertEqual(
            GeoPoint.from_e7(90 * COORDINATE_SCALE, 10).longitude_e7,
            0,
        )

    def test_direct_e7_constructor_accepts_only_canonical_representation(self) -> None:
        for latitude, longitude in (
            (True, 0),
            (0, False),
            (900_000_001, 0),
            (0, 1_800_000_000),
            (900_000_000, 1),
        ):
            with (
                self.subTest(latitude=latitude, longitude=longitude),
                self.assertRaises((TypeError, ValueError)),
            ):
                GeoPoint(latitude, longitude)

    def test_analysis_boundary_revalidates_forged_and_subclassed_points(
        self,
    ) -> None:
        forged = object.__new__(GeoPoint)
        object.__setattr__(forged, "latitude_e7", 900_000_001)
        object.__setattr__(forged, "longitude_e7", 0)
        with self.assertRaises(ValueError):
            unit_vector(forged)

        class GeoPointSubclass(GeoPoint):
            pass

        subclassed = GeoPointSubclass(0, 0)
        with self.assertRaises(TypeError):
            unit_vector(subclassed)


class SphericalDistanceTests(unittest.TestCase):
    def test_equatorial_degree_matches_mean_earth_radius(self) -> None:
        first = GeoPoint.from_decimal("0", "0")
        second = GeoPoint.from_decimal("0", "1")
        expected = math.pi * EARTH_RADIUS_METERS / 180.0
        self.assertEqual(haversine_meters(first, first), 0.0)
        self.assertAlmostEqual(haversine_meters(first, second), expected, places=7)

    def test_known_city_distance(self) -> None:
        paris = GeoPoint.from_decimal("48.8566", "2.3522")
        new_york = GeoPoint.from_decimal("40.7128", "-74.006")
        self.assertAlmostEqual(
            haversine_meters(paris, new_york) / 1000.0,
            5_837.24,
            delta=0.5,
        )

    def test_antimeridian_and_pole_distances(self) -> None:
        west = GeoPoint.from_decimal("0", "179.9")
        east = GeoPoint.from_decimal("0", "-179.9")
        expected = math.radians(0.2) * EARTH_RADIUS_METERS
        self.assertAlmostEqual(haversine_meters(west, east), expected, places=6)

        pole = GeoPoint.from_decimal("90", "120")
        equator = GeoPoint.from_decimal("0", "-33")
        self.assertAlmostEqual(
            haversine_meters(pole, equator),
            math.pi * EARTH_RADIUS_METERS / 2.0,
            places=7,
        )
        self.assertEqual(unit_vector(pole), (0.0, 0.0, 1.0))

    def test_antipode_is_finite_and_exact_at_radius_scale(self) -> None:
        first = GeoPoint.from_decimal("0", "0")
        antipode = GeoPoint.from_decimal("0", "180")
        distance = haversine_meters(first, antipode)
        self.assertTrue(math.isfinite(distance))
        self.assertAlmostEqual(distance, math.pi * EARTH_RADIUS_METERS, places=7)

    def test_chord_conversion_agrees_with_haversine(self) -> None:
        first = GeoPoint.from_decimal("-33.8688", "151.2093")
        second = GeoPoint.from_decimal("35.6762", "139.6503")
        squared_chord = squared_chord_distance(
            unit_vector(first),
            unit_vector(second),
        )
        self.assertAlmostEqual(
            chord_distance_to_meters(squared_chord),
            haversine_meters(first, second),
            places=7,
        )

    def test_geometry_helpers_reject_invalid_types_and_distances(self) -> None:
        point = GeoPoint.from_decimal("0", "0")
        with self.assertRaises(TypeError):
            unit_vector("0,0")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            haversine_meters(point, "0,0")  # type: ignore[arg-type]
        for squared_chord in (-1.0, math.inf, -math.inf, math.nan):
            with (
                self.subTest(squared_chord=squared_chord),
                self.assertRaises(ValueError),
            ):
                chord_distance_to_meters(squared_chord)


if __name__ == "__main__":
    unittest.main()
