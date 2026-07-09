"""Standalone test for core/matching.py against known RA/Dec pairs and known
pixel separations (rewrite.md Phase 2 step 4). No pytest dependency -- run
directly with `python3 tests/test_core_matching.py`. Validates the adapter
layer over stdpipe.astrometry, not the matching math itself (that's
stdpipe's own test suite's job).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pyrt_transient.core.matching import match_radius


def _assert_pair_matched(idx_a, idx_b, dist, i, j, expected_dist, tol):
    mask = (idx_a == i) & (idx_b == j)
    assert np.any(mask), f"expected pair ({i}, {j}) to be matched, got idx_a={idx_a}, idx_b={idx_b}"
    found_dist = dist[mask][0]
    assert abs(found_dist - expected_dist) < tol, (
        f"pair ({i}, {j}) distance {found_dist} != expected {expected_dist} (tol {tol})"
    )


def _assert_pair_not_matched(idx_a, idx_b, i, j):
    mask = (idx_a == i) & (idx_b == j)
    assert not np.any(mask), f"expected pair ({i}, {j}) NOT to be matched, but it was"


def test_sky_match_known_separations():
    # Pure-declination offsets so the great-circle separation equals the
    # offset exactly, regardless of RA (no cos(dec) factor to account for).
    ra = 180.0
    dec0 = 0.0
    sep_close_arcsec = 2.0
    sep_far_arcsec = 10.0

    coords_a = np.array([[ra, dec0]])
    coords_b = np.array([
        [ra, dec0 + sep_close_arcsec / 3600.0],
        [ra, dec0 + sep_far_arcsec / 3600.0],
    ])

    idx_a, idx_b, dist_deg = match_radius(coords_a, coords_b, radius_arcsec=5.0, coord_system="sky")
    dist_arcsec = dist_deg * 3600.0

    _assert_pair_matched(idx_a, idx_b, dist_arcsec, 0, 0, sep_close_arcsec, tol=1e-3)
    _assert_pair_not_matched(idx_a, idx_b, 0, 1)

    # Widen the radius past both separations -- both should now match.
    idx_a, idx_b, dist_deg = match_radius(coords_a, coords_b, radius_arcsec=15.0, coord_system="sky")
    dist_arcsec = dist_deg * 3600.0
    _assert_pair_matched(idx_a, idx_b, dist_arcsec, 0, 0, sep_close_arcsec, tol=1e-3)
    _assert_pair_matched(idx_a, idx_b, dist_arcsec, 0, 1, sep_far_arcsec, tol=1e-3)

    print("test_sky_match_known_separations: PASS")


def test_pixel_match_known_separations():
    coords_a = np.array([[100.0, 100.0]])
    coords_b = np.array([
        [103.0, 100.0],   # 3 px away
        [200.0, 100.0],   # 100 px away
    ])

    idx_a, idx_b, dist = match_radius(coords_a, coords_b, radius_arcsec=5.0, coord_system="pixel")
    _assert_pair_matched(idx_a, idx_b, dist, 0, 0, 3.0, tol=1e-9)
    _assert_pair_not_matched(idx_a, idx_b, 0, 1)

    idx_a, idx_b, dist = match_radius(coords_a, coords_b, radius_arcsec=150.0, coord_system="pixel")
    _assert_pair_matched(idx_a, idx_b, dist, 0, 0, 3.0, tol=1e-9)
    _assert_pair_matched(idx_a, idx_b, dist, 0, 1, 100.0, tol=1e-9)

    print("test_pixel_match_known_separations: PASS")


def test_invalid_coord_system_raises():
    coords = np.array([[0.0, 0.0]])
    try:
        match_radius(coords, coords, radius_arcsec=1.0, coord_system="bogus")
    except ValueError:
        print("test_invalid_coord_system_raises: PASS")
        return
    raise AssertionError("expected ValueError for invalid coord_system")


if __name__ == "__main__":
    test_sky_match_known_separations()
    test_pixel_match_known_separations()
    test_invalid_coord_system_raises()
    print("All core/matching.py tests passed.")
