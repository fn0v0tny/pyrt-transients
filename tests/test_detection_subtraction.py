"""Hand-computed unit tests for detection/subtraction/ (Phase A: consuming
already-differenced epochs -- see the subtraction-branch plan and
FUTURE_IDEAS.md's "New detection strategy: image subtraction").

Covers the two genuinely new pieces validated only against real data so
far: candidates.py's science-meta borrowing/calibration adapter, and
artifact_filters.py's dipole-rejection filter (the one check with no
catalog-listed negative detections to compare against -- it samples FITS
pixel data directly, so it needs its own synthetic-array test rather than
relying on tests/2026kid/'s real fixture alone).

No pytest dependency -- run directly with
`python3 tests/test_detection_subtraction.py`.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pyrt_transient.detection.subtraction import candidates as sub_candidates
from pyrt_transient.detection.subtraction import artifact_filters
from pyrt_transient.detection.subtraction import extraction as sub_extraction


def _assert_close(actual, expected, label, tol=1e-9):
    assert abs(actual - expected) < tol, f"{label}: expected {expected}, got {actual}"


# ---------------------------------------------------------------------------
# candidates.py: science-meta borrowing / calibration
# ---------------------------------------------------------------------------

def test_borrow_science_meta_only_fills_missing_keys():
    diff_table = Table({"X_IMAGE": [1.0]})
    diff_table.meta["MAGLIM"] = 19.0  # already present -- must not be overwritten
    sci_table = Table({"X_IMAGE": [1.0]})
    sci_table.meta.update({"MAGLIM": 20.5, "MAGZERO": 25.05, "CTRRA": 228.987})

    sub_candidates.borrow_science_meta(diff_table, sci_table)

    _assert_close(diff_table.meta["MAGLIM"], 19.0, "MAGLIM not overwritten")
    _assert_close(diff_table.meta["MAGZERO"], 25.05, "MAGZERO borrowed")
    _assert_close(diff_table.meta["CTRRA"], 228.987, "CTRRA borrowed")
    print("test_borrow_science_meta_only_fills_missing_keys: PASS")


def test_derive_observation_id_from_science_file_directly():
    # Phase B raw mode passes the *science* file itself as input (not a
    # diff file with an 'h'-suffixed sibling) -- derive_observation_id must
    # find OBJECT/TARGET on the given path directly in that case, not only
    # via the science-sibling lookup. Regression test for the bug found
    # while wiring Phase B into pipeline_magic_sn.py: this case silently
    # fell through to the generic OBSID-based ID (which fragments per
    # night, not per campaign) because find_science_sibling only
    # recognizes paths ending in 'h'.
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        sci = Table({"X_IMAGE": [1.0]})
        sci.meta["OBJECT"] = "AT 2026kid"
        _write_ecsv(sci, d / "0425r.ecsv")

        obs_id = sub_candidates.derive_observation_id(d / "0425r.ecsv")
        assert obs_id == "AT_2026kid", f"expected 'AT_2026kid', got {obs_id!r}"
        print("test_derive_observation_id_from_science_file_directly: PASS")


def test_derive_observation_id_falls_back_to_science_sibling():
    # Phase A's diff-file input: the diff table itself has no OBJECT/TARGET
    # (verified against the real fixture), so this must fall back to the
    # 'h'-suffix sibling lookup -- the original behavior, preserved.
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        sci = Table({"X_IMAGE": [1.0]})
        sci.meta["OBJECT"] = "AT 2026kid"
        _write_ecsv(sci, d / "0425r.ecsv")
        diff = Table({"X_IMAGE": [1.0]})  # no OBJECT/TARGET, matches real fixture
        _write_ecsv(diff, d / "0425rh.ecsv")

        obs_id = sub_candidates.derive_observation_id(d / "0425rh.ecsv")
        assert obs_id == "AT_2026kid", f"expected 'AT_2026kid', got {obs_id!r}"
        print("test_derive_observation_id_falls_back_to_science_sibling: PASS")


def test_calibrate_diff_magnitudes_applies_science_zeropoint():
    # MAG_AUTO here is a raw zp=0 instrumental magnitude (verified against
    # the real tests/2026kid/ fixture: MAG_AUTO = -2.5*log10(FLUX) exactly).
    # -2.5*log10(1000) = -7.5; with MAGZERO=25.05 the true MAG_CALIB is 17.55.
    diff_table = Table({"MAG_AUTO": [-7.5], "MAGERR_AUTO": [0.05]})
    diff_table.meta["MAGZERO"] = 25.05

    sub_candidates.calibrate_diff_magnitudes(diff_table)

    assert "MAG_CALIB" in diff_table.colnames
    _assert_close(diff_table["MAG_CALIB"][0], 17.55, "MAG_CALIB = MAG_AUTO + MAGZERO")
    _assert_close(diff_table["MAGERR_CALIB"][0], 0.05, "MAGERR_CALIB = MAGERR_AUTO (additive zp shift)")
    print("test_calibrate_diff_magnitudes_applies_science_zeropoint: PASS")


def test_calibrate_diff_magnitudes_is_noop_if_already_calibrated():
    diff_table = Table({"MAG_CALIB": [18.0], "MAGERR_CALIB": [0.1]})
    sub_candidates.calibrate_diff_magnitudes(diff_table)
    _assert_close(diff_table["MAG_CALIB"][0], 18.0, "unchanged when MAG_CALIB already present")
    print("test_calibrate_diff_magnitudes_is_noop_if_already_calibrated: PASS")


def test_calibrate_diff_magnitudes_noop_without_science_zeropoint():
    # No MAGZERO available (science sibling missing/unreadable) -- must not
    # fabricate a MAG_CALIB from an arbitrary zeropoint.
    diff_table = Table({"MAG_AUTO": [-7.5], "MAGERR_AUTO": [0.05]})
    sub_candidates.calibrate_diff_magnitudes(diff_table)
    assert "MAG_CALIB" not in diff_table.colnames, "no MAG_CALIB fabricated without a zeropoint"
    print("test_calibrate_diff_magnitudes_noop_without_science_zeropoint: PASS")


def test_add_detection_features_from_diff_columns():
    t = Table({
        "ELLIPTICITY": [0.2, 0.0],
        "FWHM_IMAGE": [4.0, 8.0],
        "MAGERR_CALIB": [0.1, 1.0857],
        "FLAGS": [4, 2],  # 4=saturated, 2=blended
    })
    sub_candidates.add_detection_features(t)

    _assert_close(t["axis_ratio"][0], 0.8, "axis_ratio = 1 - ELLIPTICITY")
    _assert_close(t["fwhm_ratio"][0], 4.0 / 6.0, "fwhm_ratio relative to median FWHM")
    _assert_close(t["fwhm_ratio"][1], 8.0 / 6.0, "fwhm_ratio relative to median FWHM")
    _assert_close(t["snr_auto"][0], 10.857, "snr_auto = 1.0857/MAGERR_CALIB", tol=1e-3)
    _assert_close(t["snr_auto"][1], 1.0, "snr_auto = 1.0857/MAGERR_CALIB", tol=1e-3)
    assert bool(t["saturated"][0]) is True and bool(t["blended"][0]) is False
    assert bool(t["saturated"][1]) is False and bool(t["blended"][1]) is True
    print("test_add_detection_features_from_diff_columns: PASS")


def _write_ecsv(table, path):
    table.write(str(path), format="ascii.ecsv", overwrite=True)


def test_build_epoch_candidates_end_to_end_with_science_sibling():
    """Small integration test: diff + science sibling on disk, verifying
    build_epoch_candidates borrows meta, calibrates, applies the
    significance gate, and produces a quality_score -- the same pipeline
    validated against tests/2026kid/'s real fixture, at unit-test scale.
    """
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)

        sci = Table({
            "ALPHA_J2000": [10.0, 10.001],
            "DELTA_J2000": [20.0, 20.001],
            "X_IMAGE": [100.0, 200.0],
            "Y_IMAGE": [100.0, 200.0],
            "MAG_CALIB": [18.0, 18.0],
            "MAGERR_CALIB": [0.1, 0.1],
        })
        sci.meta.update({"MAGZERO": 25.0, "MAGLIM": 20.0, "CTRRA": 10.0, "CTRDEC": 20.0, "FIELD": 0.3})
        _write_ecsv(sci, d / "0101r.ecsv")

        diff = Table({
            "ALPHA_J2000": [10.0, 10.002],
            "DELTA_J2000": [20.0, 20.002],
            "X_IMAGE": [100.0, 300.0],
            "Y_IMAGE": [100.0, 300.0],
            "ELLIPTICITY": [0.1, 0.1],
            "FWHM_IMAGE": [4.0, 4.0],
            "FLAGS": [0, 0],
            # Row 0: MAG_AUTO=-7.0 -> MAG_CALIB=18.0, tight error -> passes
            # the significance gate. Row 1: much noisier -> should be
            # dropped by the new_source_siglim gate.
            "MAG_AUTO": [-7.0, -7.0],
            "MAGERR_AUTO": [0.05, 5.0],
        })
        _write_ecsv(diff, d / "0101rh.ecsv")

        result = sub_candidates.build_epoch_candidates(
            d / "0101rh.ecsv", config=None, template_provenance="ps1_template",
        )

        assert result is not None
        assert len(result) == 1, f"expected the noisy row to be gated out, got {len(result)}"
        _assert_close(result["MAG_CALIB"][0], 18.0, "borrowed MAGZERO applied correctly")
        assert "quality_score" in result.colnames
        assert result["quality_score"][0] > 0
        assert result["candidate_type"][0] == "new"
        assert result["reference_catalog"][0] == "ps1_template"
        print("test_build_epoch_candidates_end_to_end_with_science_sibling: PASS")


# ---------------------------------------------------------------------------
# artifact_filters.py: dipole rejection
# ---------------------------------------------------------------------------

def _make_diff_fits_with_wcs(path, shape=(101, 101), pixscale_arcsec=1.0):
    """A near-zero-noise diff image (with a minimal WCS -- required since
    reject_dipole_artifacts converts radius_arcsec to pixels via the FITS's
    own pixel scale) containing:
    - an isolated positive point source at (30, 30) -- a real transient,
      no negative counterpart nearby.
    - a positive/negative dipole pair at (70, 70)/(70, 76) -- the signature
      of a bad-subtraction artifact (comparable amplitude, close together).
    """
    rng = np.random.default_rng(0)
    data = rng.normal(0, 1.0, size=shape).astype(np.float64)
    data[30, 30] = 500.0
    data[70, 70] = 500.0
    data[70, 76] = -480.0
    hdu = fits.PrimaryHDU(data)
    hdu.header["CTYPE1"] = "RA---TAN"
    hdu.header["CTYPE2"] = "DEC--TAN"
    hdu.header["CRPIX1"] = shape[1] / 2
    hdu.header["CRPIX2"] = shape[0] / 2
    hdu.header["CRVAL1"] = 10.0
    hdu.header["CRVAL2"] = 20.0
    hdu.header["CDELT1"] = -pixscale_arcsec / 3600.0
    hdu.header["CDELT2"] = pixscale_arcsec / 3600.0
    hdu.writeto(str(path), overwrite=True)


def test_reject_dipole_artifacts_with_wcs():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        diff_fits = d / "diff_wcs.fits"
        _make_diff_fits_with_wcs(diff_fits, pixscale_arcsec=1.0)

        candidates = Table({
            "X_IMAGE": [31.0, 71.0],  # 1-indexed -> array (30,30) and (70,70)
            "Y_IMAGE": [31.0, 71.0],
            "FLUX": [500.0, 500.0],
        })

        kept, n_rejected = artifact_filters.reject_dipole_artifacts(
            candidates, diff_fits,
            radius_arcsec=8.0,  # >6px separation at 1"/px, within search box
            flux_ratio_thresh=0.5,
        )
        assert n_rejected == 1, f"expected exactly the dipole candidate rejected, got {n_rejected}"
        assert len(kept) == 1
        _assert_close(float(kept["X_IMAGE"][0]), 31.0, "isolated real source survives")
        print("test_reject_dipole_artifacts_with_wcs: PASS")


def test_reject_dipole_artifacts_empty_table_noop():
    kept, n_rejected = artifact_filters.reject_dipole_artifacts(Table(), "/nonexistent.fits")
    assert len(kept) == 0 and n_rejected == 0
    print("test_reject_dipole_artifacts_empty_table_noop: PASS")


# ---------------------------------------------------------------------------
# differencing.py: science_meta propagation into the diff FITS header
# ---------------------------------------------------------------------------

def test_write_diff_merges_science_meta_into_header():
    # Regression test for the bug found while wiring Phase B into
    # pipeline_magic_sn.py: FIELD/CTRRA/CTRDEC/MAGLIM live only in the
    # science ecsv's table.meta, never in the raw FITS header -- copying
    # just the science FITS header (as _write_diff used to) silently wrote
    # a diff FITS with FIELD=0.0 and friends, which downstream turned into
    # a 0-degree HyperLEDA search radius that returned ~1 million rows and
    # hung the whole pipeline. science_meta must be merged in.
    from pyrt_transient.detection.subtraction import differencing as sub_differencing

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        header = fits.Header({"NAXIS": 2, "NAXIS1": 10, "NAXIS2": 10, "FWHM": 3.0})
        science_meta = {"FIELD": 0.316, "CTRRA": 228.987, "CTRDEC": 56.309, "MAGLIM": 20.5,
                        "OBJECT": "AT 2026kid"}

        out_path = d / "diff.fits"
        sub_differencing._write_diff(
            np.zeros((10, 10)), header, out_path, "own_epoch:test",
            science_meta=science_meta,
        )

        with fits.open(out_path) as hdul:
            out_header = hdul[0].header
        assert out_header["FIELD"] == 0.316
        assert out_header["CTRRA"] == 228.987
        assert out_header["MAGLIM"] == 20.5
        assert out_header["OBJECT"] == "AT 2026kid"
        assert out_header["TEMPLATE"] == "own_epoch:test"
        print("test_write_diff_merges_science_meta_into_header: PASS")


def test_write_diff_without_science_meta_still_works():
    # science_meta is optional -- omitting it must not break anything
    # (matches every call site before this fix existed).
    from pyrt_transient.detection.subtraction import differencing as sub_differencing

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        header = fits.Header({"NAXIS": 2, "NAXIS1": 10, "NAXIS2": 10})
        out_path = d / "diff.fits"
        sub_differencing._write_diff(np.zeros((10, 10)), header, out_path, "ps1_template")
        with fits.open(out_path) as hdul:
            assert hdul[0].header["TEMPLATE"] == "ps1_template"
        print("test_write_diff_without_science_meta_still_works: PASS")


# ---------------------------------------------------------------------------
# extraction.py: _resolve_fwhm_px -- regression for a real bug found while
# validating the stacking feature (pyrt_transient/detection/stacking.py)
# against real GRB151027B data: its original 2015-era astrometry solution
# recorded FWHM=0.0 (present, not missing), which used to flow straight
# through into SEP's aper= parameter -- aper=0.0 found 0 sources where
# aper=3.0 found 117 on the exact same image.
# ---------------------------------------------------------------------------

def test_resolve_fwhm_px_passes_through_a_sane_value():
    header = fits.Header({"FWHM": 2.5})
    _assert_close(sub_extraction._resolve_fwhm_px(header), 2.5, "sane FWHM preserved")
    print("test_resolve_fwhm_px_passes_through_a_sane_value: PASS")


def test_resolve_fwhm_px_falls_back_on_zero():
    header = fits.Header({"FWHM": 0.0})
    _assert_close(sub_extraction._resolve_fwhm_px(header, default=3.0), 3.0, "FWHM=0.0 falls back to default")
    print("test_resolve_fwhm_px_falls_back_on_zero: PASS")


def test_resolve_fwhm_px_falls_back_on_negative_and_missing():
    _assert_close(sub_extraction._resolve_fwhm_px(fits.Header({"FWHM": -1.0}), default=3.0), 3.0,
                  "negative FWHM falls back to default")
    _assert_close(sub_extraction._resolve_fwhm_px(fits.Header({}), default=3.0), 3.0,
                  "missing FWHM falls back to default")
    print("test_resolve_fwhm_px_falls_back_on_negative_and_missing: PASS")


# ---------------------------------------------------------------------------
# extraction.py: _patch_sep_sum_circle_clip_kwargs -- regression for a real
# bug found running the stacking feature on a production host: that host's
# live stdpipe checkout's get_objects_sep always makes one internal
# background-photometry call via the plain sep.sum_circle(..., clip_sigma=,
# clip_iters=) -- kwargs that function has never accepted (only the newer
# sum_circle_optimal does) -- crashing every SEP detection with
# `TypeError: sum_circle() got an unexpected keyword argument 'clip_sigma'`.
# No local sep install reproduces this (it's a stdpipe-version-specific
# bug), so these tests simulate it with a fake sep.sum_circle.
# ---------------------------------------------------------------------------

def test_patch_sep_sum_circle_retries_without_clip_kwargs_on_typeerror():
    import sep

    calls = []

    def fake_sum_circle(*args, **kwargs):
        calls.append(dict(kwargs))
        if "clip_sigma" in kwargs:
            raise TypeError("sum_circle() got an unexpected keyword argument 'clip_sigma'")
        return "OK"

    original = sep.sum_circle
    sep.sum_circle = fake_sum_circle
    sub_extraction._sep_sum_circle_patch_applied = False
    try:
        sub_extraction._patch_sep_sum_circle_clip_kwargs()
        result = sep.sum_circle(1, 2, 3, clip_sigma=3.0, clip_iters=1)
        assert result == "OK"
        assert len(calls) == 2, "must retry once after the TypeError"
        assert "clip_sigma" in calls[0] and "clip_iters" in calls[0]
        assert "clip_sigma" not in calls[1] and "clip_iters" not in calls[1]
    finally:
        sep.sum_circle = original
        sub_extraction._sep_sum_circle_patch_applied = True
    print("test_patch_sep_sum_circle_retries_without_clip_kwargs_on_typeerror: PASS")


def test_patch_sep_sum_circle_noop_when_kwargs_already_supported():
    import sep

    calls = []

    def fake_sum_circle(*args, **kwargs):
        calls.append(dict(kwargs))
        return "OK"  # a sep version that genuinely supports clip_sigma -- never raises

    original = sep.sum_circle
    sep.sum_circle = fake_sum_circle
    sub_extraction._sep_sum_circle_patch_applied = False
    try:
        sub_extraction._patch_sep_sum_circle_clip_kwargs()
        result = sep.sum_circle(1, 2, 3, clip_sigma=3.0)
        assert result == "OK"
        assert len(calls) == 1, "must not retry when the first call already succeeded"
        assert "clip_sigma" in calls[0]
    finally:
        sep.sum_circle = original
        sub_extraction._sep_sum_circle_patch_applied = True
    print("test_patch_sep_sum_circle_noop_when_kwargs_already_supported: PASS")


def test_patch_sep_sum_circle_reraises_unrelated_typeerrors():
    import sep

    def fake_sum_circle(*args, **kwargs):
        raise TypeError("some unrelated argument problem")

    original = sep.sum_circle
    sep.sum_circle = fake_sum_circle
    sub_extraction._sep_sum_circle_patch_applied = False
    try:
        sub_extraction._patch_sep_sum_circle_clip_kwargs()
        try:
            sep.sum_circle(1, 2, 3)
            assert False, "expected TypeError to propagate"
        except TypeError as e:
            assert "unrelated" in str(e)
    finally:
        sep.sum_circle = original
        sub_extraction._sep_sum_circle_patch_applied = True
    print("test_patch_sep_sum_circle_reraises_unrelated_typeerrors: PASS")


if __name__ == "__main__":
    test_borrow_science_meta_only_fills_missing_keys()
    test_derive_observation_id_from_science_file_directly()
    test_derive_observation_id_falls_back_to_science_sibling()
    test_calibrate_diff_magnitudes_applies_science_zeropoint()
    test_calibrate_diff_magnitudes_is_noop_if_already_calibrated()
    test_calibrate_diff_magnitudes_noop_without_science_zeropoint()
    test_add_detection_features_from_diff_columns()
    test_build_epoch_candidates_end_to_end_with_science_sibling()
    test_reject_dipole_artifacts_with_wcs()
    test_reject_dipole_artifacts_empty_table_noop()
    test_write_diff_merges_science_meta_into_header()
    test_write_diff_without_science_meta_still_works()
    test_resolve_fwhm_px_passes_through_a_sane_value()
    test_resolve_fwhm_px_falls_back_on_zero()
    test_resolve_fwhm_px_falls_back_on_negative_and_missing()
    test_patch_sep_sum_circle_retries_without_clip_kwargs_on_typeerror()
    test_patch_sep_sum_circle_noop_when_kwargs_already_supported()
    test_patch_sep_sum_circle_reraises_unrelated_typeerrors()
    print("All detection/subtraction/ tests passed.")
