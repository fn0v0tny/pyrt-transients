"""Hand-computed Tier 0 unit tests for core/radii.py (rewrite.md Phase 0 step
6 / Phase 2 step 5) -- one of the three "genuinely risky merges" that must
not rely on check_baseline.py alone, since the single fixture may not
exercise every branch (ERRX2/ERRY2 present vs. absent, each fallback, each
coord_system, clipping, total failure).

No pytest dependency -- run directly with `python3 tests/test_core_radii.py`.
"""
import math
import sys
from pathlib import Path

import numpy as np
from astropy.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pyrt_transient.core.radii import compute_adaptive_radius


def _det(**cols):
    n = len(next(iter(cols.values()))) if cols else 1
    t = Table({k: np.atleast_1d(v) for k, v in cols.items()}) if cols else Table({"_dummy": [0] * n})
    return t


def test_primary_errxy_path_pixel():
    # ERRX2=ERRY2=1.0 -> pos_err = sqrt(2), nsigma=3 -> radius = 3*sqrt(2) = 4.242640687...
    det = _det(ERRX2_IMAGE=[1.0], ERRY2_IMAGE=[1.0])
    r = compute_adaptive_radius(det, coord_system="pixel", nsigma=3.0,
                                 idlimit_min_px=0.01, idlimit_max_px=100.0, use_astvar=True)
    expected = 3.0 * math.sqrt(2.0)
    assert abs(r[0] - expected) < 1e-9, f"expected {expected}, got {r[0]}"
    print("test_primary_errxy_path_pixel: PASS")


def test_primary_errxy_path_with_astvar():
    # Same ERRX2/ERRY2, ASTVAR=1.5 -> pos_err *= sqrt(1.5); sqrt(2)*sqrt(1.5) = sqrt(3).
    # radius = 3*sqrt(3) = 5.196152423...
    det = _det(ERRX2_IMAGE=[1.0], ERRY2_IMAGE=[1.0])
    det.meta["ASTVAR"] = 1.5
    r = compute_adaptive_radius(det, coord_system="pixel", nsigma=3.0,
                                 idlimit_min_px=0.01, idlimit_max_px=100.0, use_astvar=True)
    expected = 3.0 * math.sqrt(3.0)
    assert abs(r[0] - expected) < 1e-9, f"expected {expected}, got {r[0]}"

    # use_astvar=False must ignore the ASTVAR meta entirely.
    r2 = compute_adaptive_radius(det, coord_system="pixel", nsigma=3.0,
                                  idlimit_min_px=0.01, idlimit_max_px=100.0, use_astvar=False)
    expected2 = 3.0 * math.sqrt(2.0)
    assert abs(r2[0] - expected2) < 1e-9, f"expected {expected2}, got {r2[0]}"
    print("test_primary_errxy_path_with_astvar: PASS")


def test_clipping():
    # Tiny ERRX2/ERRY2 -> clipped up to idlimit_min_px; huge -> clipped down to idlimit_max_px.
    det_small = _det(ERRX2_IMAGE=[1e-12], ERRY2_IMAGE=[1e-12])
    r_small = compute_adaptive_radius(det_small, coord_system="pixel",
                                       idlimit_min_px=1.0, idlimit_max_px=8.0)
    assert abs(r_small[0] - 1.0) < 1e-9, f"expected clip to 1.0, got {r_small[0]}"

    det_big = _det(ERRX2_IMAGE=[1000.0], ERRY2_IMAGE=[1000.0])
    r_big = compute_adaptive_radius(det_big, coord_system="pixel",
                                     idlimit_min_px=1.0, idlimit_max_px=8.0)
    assert abs(r_big[0] - 8.0) < 1e-9, f"expected clip to 8.0, got {r_big[0]}"
    print("test_clipping: PASS")


def test_pixel_fallback_snr():
    # No ERRX2/ERRY2 -> pixel fallback: pos_err = (FWHM/2.35)/max(SNR, 1e-6).
    # FWHM_IMAGE=2.35 -> fwhm/2.35=1.0; SNR=10 -> pos_err=0.1; nsigma=3 -> radius=0.3.
    det = _det(SNR=[10.0], FWHM_IMAGE=[2.35])
    r = compute_adaptive_radius(det, coord_system="pixel", nsigma=3.0,
                                 idlimit_min_px=0.01, idlimit_max_px=100.0, use_astvar=True)
    assert abs(r[0] - 0.3) < 1e-9, f"expected 0.3, got {r[0]}"
    print("test_pixel_fallback_snr: PASS")


def test_sky_fallback_and_plate_scale():
    # No ERRX2/ERRY2 -> sky fallback: psf_sigma_px = FWHM/2.35 = 1.0 (FWHM_IMAGE=2.35);
    # snr_scale = sqrt(10/SNR); SNR=10 -> snr_scale=1.0; nsigma=3 -> radii_px = 3.0.
    # CD1_1=CD2_2=1/3600 deg/px -> plate_scale = 3600*sqrt(abs(1/3600 * 1/3600)) = 1.0 arcsec/px.
    # radius_arcsec = 3.0 * 1.0 = 3.0.
    det = _det(SNR=[10.0], FWHM_IMAGE=[2.35])
    det.meta["CD1_1"] = 1.0 / 3600.0
    det.meta["CD2_2"] = 1.0 / 3600.0
    r = compute_adaptive_radius(det, coord_system="sky", nsigma=3.0,
                                 idlimit_min_px=0.01, idlimit_max_px=100.0)
    assert abs(r[0] - 3.0) < 1e-9, f"expected 3.0 arcsec, got {r[0]}"
    print("test_sky_fallback_and_plate_scale: PASS")


def test_sky_default_plate_scale_when_no_wcs():
    # No CD matrix/CDELT1 in meta -> falls back to default_plate_scale_arcsec_per_px.
    det = _det(SNR=[10.0], FWHM_IMAGE=[2.35])
    r = compute_adaptive_radius(det, coord_system="sky", nsigma=3.0,
                                 idlimit_min_px=0.01, idlimit_max_px=100.0,
                                 default_plate_scale_arcsec_per_px=0.5)
    # radii_px = 3.0 (as above) * default plate scale 0.5 = 1.5
    assert abs(r[0] - 1.5) < 1e-9, f"expected 1.5, got {r[0]}"
    print("test_sky_default_plate_scale_when_no_wcs: PASS")


def test_pixel_total_failure_returns_empty():
    # No ERRX2/ERRY2, no SNR, no FLUX_ISO/FLUXERR_ISO -- primary and pixel
    # fallback both explicitly return None on missing columns (original
    # catalog.py contract), so the outer failure path returns an empty array.
    det = _det(SOME_UNRELATED_COLUMN=[1.0, 2.0, 3.0])
    r_pixel = compute_adaptive_radius(det, coord_system="pixel")
    assert len(r_pixel) == 0, f"expected empty array for pixel total failure, got {r_pixel}"
    print("test_pixel_total_failure_returns_empty: PASS")


def test_sky_fallback_never_fails_on_missing_columns():
    # Unlike the pixel fallback, transient_analyser.py's original sky-space
    # formula has no "missing data" signal at all -- FWHM defaults to a flat
    # 2.0 px array and SNR to a flat 5.0 when their source columns are absent,
    # so it always produces a real answer rather than falling through to the
    # constant-2.0-arcsec total-failure contract. That contract is preserved
    # in the code (see _sky_fallback_radius's caller) but is only reachable
    # via an actual exception, not merely absent optional columns -- this
    # test documents that asymmetry rather than asserting a false one.
    det = _det(SOME_UNRELATED_COLUMN=[1.0, 2.0, 3.0])
    r_sky = compute_adaptive_radius(det, coord_system="sky",
                                     idlimit_min_px=1.0, idlimit_max_px=8.0,
                                     default_plate_scale_arcsec_per_px=0.33)
    # psf_sigma_px defaults to 2.0, snr defaults to 5.0 -> snr_scale=sqrt(2);
    # nsigma=3.0 (default) -> radii_px = 3*2*sqrt(2) = 8.485... -> clipped to 8.0;
    # arcsec = 8.0 * 0.33 = 2.64.
    expected = 8.0 * 0.33
    assert np.all(np.abs(r_sky - expected) < 1e-9), f"expected {expected}, got {r_sky}"
    print("test_sky_fallback_never_fails_on_missing_columns: PASS")


def test_empty_detections():
    det = Table({"ERRX2_IMAGE": [], "ERRY2_IMAGE": []})
    assert len(compute_adaptive_radius(det, coord_system="pixel")) == 0
    assert len(compute_adaptive_radius(det, coord_system="sky")) == 0
    print("test_empty_detections: PASS")


def test_invalid_coord_system_raises():
    det = _det(ERRX2_IMAGE=[1.0], ERRY2_IMAGE=[1.0])
    try:
        compute_adaptive_radius(det, coord_system="bogus")
    except ValueError:
        print("test_invalid_coord_system_raises: PASS")
        return
    raise AssertionError("expected ValueError for invalid coord_system")


if __name__ == "__main__":
    test_primary_errxy_path_pixel()
    test_primary_errxy_path_with_astvar()
    test_clipping()
    test_pixel_fallback_snr()
    test_sky_fallback_and_plate_scale()
    test_sky_default_plate_scale_when_no_wcs()
    test_pixel_total_failure_returns_empty()
    test_sky_fallback_never_fails_on_missing_columns()
    test_empty_detections()
    test_invalid_coord_system_raises()
    print("All core/radii.py tests passed.")
