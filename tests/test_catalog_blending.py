"""Regression tests for the blending fix in catalog.py --
CatTransients._check_magnitude_changes_cached /
_process_detections_for_candidates.

FUTURE_IDEAS.md's "Detection recall" section documented this as a real,
unresolved false-negative mechanism (found via a real GCN-confirmed
afterglow, GRB200410A): a detection blended with a known catalog star
measures the *combined* flux of both, which dilutes a real superimposed
source's excess enough to fail the ordinary `siglim` significance bar even
when genuinely brighter than the catalogued star alone -- previously any
blended detection that didn't clear the strict bar just read as "same
star, unchanged". The fix: for a blended detection (SExtractor FLAGS bit
2) specifically, a real brightening excess (combined flux brighter than
the catalogued star alone predicts) is checked against the same more
permissive bar new-source detections already get (`new_source_siglim`)
instead of the strict one -- both in the admission pre-filter
(`_process_detections_for_candidates`'s `bad_snr_matched`) and in the
per-match significance check (`_check_magnitude_changes_cached`).

No pytest dependency -- run directly with
`python3 tests/test_catalog_blending.py`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from astropy.table import Table

from pyrt_transient.catalog import CatTransients, CatalogOptimizationCache


def _make_cat(catalog_name="gaia", r_mag=18.0, colors=(0.5, 0.3, 0.1, 0.2)):
    """A minimal CatTransients instance with just enough state for
    _check_magnitude_changes_cached / _process_detections_for_candidates --
    CatTransients subclasses astropy.table.Table (via pyrt's Catalog base),
    so real construction needs Table's own __new__, not a bare object.
    response_model='' makes simple_color_model a no-op (cat_mag == r_mag),
    verified directly, so test numbers are exact and easy to reason about.
    """
    cat = Table.__new__(CatTransients)
    cat.meta = {"catalog_props": {"catalog_name": catalog_name}}
    cat._photometric_cache = CatalogOptimizationCache(
        coordinates=np.zeros((1, 2)),
        pixel_coordinates={},
        magnitudes=np.array([[r_mag, r_mag]]),
        colors=np.array([list(colors)]),
        valid_stars=np.array([True]),
        kdtrees={},
    )
    return cat


# ---------------------------------------------------------------------------
# _check_magnitude_changes_cached
# ---------------------------------------------------------------------------

def test_blended_excess_flagged_between_new_source_siglim_and_siglim():
    # diff=-1.2, err=0.3 -> nsigma~4.0: clears new_source_siglim=1.5 but
    # not the strict siglim=5.0 -- exactly the diluted-excess case this fix
    # targets.
    cat = _make_cat()
    is_cand, ctype, mdiff = cat._check_magnitude_changes_cached(
        np.array([0]), 16.8, 0.3, "", mag_change_threshold=1.0, siglim=5.0,
        is_blended=True, new_source_siglim=1.5,
    )
    assert is_cand is True
    assert ctype == "brightening"
    assert abs(mdiff - (-1.2)) < 1e-6
    print("test_blended_excess_flagged_between_new_source_siglim_and_siglim: PASS")


def test_same_moderate_excess_not_flagged_when_not_blended():
    # Identical numbers, is_blended=False -- must NOT be flagged. Proves
    # the relaxed bar is gated on blend status, not applied universally.
    cat = _make_cat()
    is_cand, ctype, mdiff = cat._check_magnitude_changes_cached(
        np.array([0]), 16.8, 0.3, "", mag_change_threshold=1.0, siglim=5.0,
        is_blended=False, new_source_siglim=1.5,
    )
    assert is_cand is False
    assert ctype == "none"
    print("test_same_moderate_excess_not_flagged_when_not_blended: PASS")


def test_ordinary_high_significance_change_unaffected_by_the_fix():
    # A genuinely strong signal (nsigma > siglim) must still be flagged via
    # the original path regardless of is_blended -- regression check that
    # the new branch doesn't change existing behavior.
    cat = _make_cat()
    is_cand, ctype, mdiff = cat._check_magnitude_changes_cached(
        np.array([0]), 16.8, 0.05, "", mag_change_threshold=1.0, siglim=5.0,
        is_blended=False, new_source_siglim=1.5,
    )
    assert is_cand is True
    assert ctype == "brightening"
    print("test_ordinary_high_significance_change_unaffected_by_the_fix: PASS")


def test_blended_moderate_fading_not_flagged_by_the_new_branch():
    # diff=+1.2 (fainter, not brighter) -- the new branch only targets
    # "combined flux brighter than the catalogued star alone would
    # predict" (diff < 0), matching FUTURE_IDEAS.md's exact wording. A
    # moderate fading blend must not be flagged via it.
    cat = _make_cat()
    is_cand, ctype, mdiff = cat._check_magnitude_changes_cached(
        np.array([0]), 19.2, 0.3, "", mag_change_threshold=1.0, siglim=5.0,
        is_blended=True, new_source_siglim=1.5,
    )
    assert is_cand is False
    print("test_blended_moderate_fading_not_flagged_by_the_new_branch: PASS")


def test_blend_still_requires_clearing_mag_change_threshold():
    # A small excess (below mag_change_threshold) must not be flagged even
    # when blended and otherwise significant -- the relaxed significance
    # bar doesn't bypass the magnitude-difference floor.
    cat = _make_cat()
    is_cand, ctype, mdiff = cat._check_magnitude_changes_cached(
        np.array([0]), 17.7, 0.05, "", mag_change_threshold=1.0, siglim=5.0,
        is_blended=True, new_source_siglim=1.5,
    )
    assert is_cand is False
    print("test_blend_still_requires_clearing_mag_change_threshold: PASS")


# ---------------------------------------------------------------------------
# _process_detections_for_candidates: FLAGS -> is_blended threading and the
# admission pre-filter relaxation
# ---------------------------------------------------------------------------

def test_process_detections_flags_blended_source_that_strict_gate_would_drop():
    cat = _make_cat()
    detections = Table({
        "X_IMAGE": [500.0],
        "Y_IMAGE": [500.0],
        "MAG_CALIB": [16.8],
        "MAGERR_CALIB": [0.3],
        "FLAGS": [2],  # SExtractor blend bit
    })
    detections.meta["NAXIS1"] = 1024
    detections.meta["NAXIS2"] = 1024

    result = cat._process_detections_for_candidates(
        detections, matches_list=[np.array([0])],
        mag_change_threshold=1.0, siglim=5.0, frame=10.0,
        new_source_siglim=1.5,
    )
    assert len(result) == 1
    assert result["candidate_type"][0] == "brightening"
    print("test_process_detections_flags_blended_source_that_strict_gate_would_drop: PASS")


def test_process_detections_drops_same_source_when_not_blended():
    cat = _make_cat()
    detections = Table({
        "X_IMAGE": [500.0],
        "Y_IMAGE": [500.0],
        "MAG_CALIB": [16.8],
        "MAGERR_CALIB": [0.3],
        "FLAGS": [0],  # not blended
    })
    detections.meta["NAXIS1"] = 1024
    detections.meta["NAXIS2"] = 1024

    result = cat._process_detections_for_candidates(
        detections, matches_list=[np.array([0])],
        mag_change_threshold=1.0, siglim=5.0, frame=10.0,
        new_source_siglim=1.5,
    )
    assert len(result) == 0
    print("test_process_detections_drops_same_source_when_not_blended: PASS")


def _make_cat_two_stars(mags):
    cat = Table.__new__(CatTransients)
    cat.meta = {"catalog_props": {"catalog_name": "gaia"}}
    cat._photometric_cache = CatalogOptimizationCache(
        coordinates=np.zeros((len(mags), 2)),
        pixel_coordinates={},
        magnitudes=np.array([[m, m] for m in mags]),
        colors=np.array([[0.5, 0.3, 0.1, 0.2]] * len(mags)),
        valid_stars=np.array([True] * len(mags)),
        kdtrees={},
    )
    return cat


def test_blended_ordinary_pair_is_not_flagged_as_brightening():
    # Two catalogue stars at 15.0 and 16.0 blended into one detection:
    # combined flux = 14.64 mag. Measured 14.64 +/- 0.3 is exactly what the
    # pair predicts -- must NOT be "brightening" even though it is 1.36 mag
    # brighter than the fainter member (the per-match comparison used to
    # flag every unresolved pair on every epoch).
    cat = _make_cat_two_stars([15.0, 16.0])
    combined = -2.5 * np.log10(10 ** (-0.4 * 15.0) + 10 ** (-0.4 * 16.0))
    is_cand, ctype, _ = cat._check_magnitude_changes_cached(
        np.array([0, 1]), combined, 0.3, "", mag_change_threshold=1.0, siglim=5.0,
        is_blended=True, new_source_siglim=1.5,
    )
    assert is_cand is False and ctype == "none"
    # ...while a real excess on top of the pair still is.
    is_cand, ctype, mdiff = cat._check_magnitude_changes_cached(
        np.array([0, 1]), combined - 1.2, 0.3, "", mag_change_threshold=1.0, siglim=5.0,
        is_blended=True, new_source_siglim=1.5,
    )
    assert is_cand is True and ctype == "brightening" and abs(mdiff + 1.2) < 1e-6
    print("test_blended_ordinary_pair_is_not_flagged_as_brightening: PASS")


def test_new_source_siglim_above_siglim_does_not_tighten_blended_gate():
    # A config with new_source_siglim >= siglim must leave blended matched
    # detections gated exactly like unblended ones, not stricter.
    cat = _make_cat()
    detections = Table({
        "X_IMAGE": [500.0], "Y_IMAGE": [500.0],
        "MAG_CALIB": [17.9], "MAGERR_CALIB": [0.15], "FLAGS": [2],
    })
    detections.meta["NAXIS1"] = 1024
    detections.meta["NAXIS2"] = 1024
    # magerr 0.15 passes siglim=5 (0.218) but would fail siglim=8 (0.136).
    # 17.9 vs catalogue 18.0: not a candidate either way -- what matters is
    # that the row is *evaluated* (no crash, no change in outcome), i.e.
    # the gate did not silently tighten.
    result = cat._process_detections_for_candidates(
        detections, matches_list=[np.array([0])],
        mag_change_threshold=1.0, siglim=5.0, frame=10.0, new_source_siglim=8.0,
    )
    assert len(result) == 0
    # And a genuine strong change on that same blended row is still found
    # (it would have been dropped at the admission gate at siglim=8).
    detections["MAG_CALIB"] = [16.0]
    detections["MAGERR_CALIB"] = [0.15]
    result = cat._process_detections_for_candidates(
        detections, matches_list=[np.array([0])],
        mag_change_threshold=1.0, siglim=5.0, frame=10.0, new_source_siglim=8.0,
    )
    assert len(result) == 1 and result["candidate_type"][0] == "brightening"
    print("test_new_source_siglim_above_siglim_does_not_tighten_blended_gate: PASS")


if __name__ == "__main__":
    test_blended_excess_flagged_between_new_source_siglim_and_siglim()
    test_same_moderate_excess_not_flagged_when_not_blended()
    test_ordinary_high_significance_change_unaffected_by_the_fix()
    test_blended_moderate_fading_not_flagged_by_the_new_branch()
    test_blend_still_requires_clearing_mag_change_threshold()
    test_process_detections_flags_blended_source_that_strict_gate_would_drop()
    test_process_detections_drops_same_source_when_not_blended()
    test_blended_ordinary_pair_is_not_flagged_as_brightening()
    test_new_source_siglim_above_siglim_does_not_tighten_blended_gate()
    print("All catalog.py blending regression tests passed.")
