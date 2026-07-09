"""Hand-computed Tier 0 unit tests for core/scoring.py (rewrite.md Phase 0
step 6 / Phase 2 step 8) -- one of the three "genuinely risky merges" that
must not rely on check_baseline.py alone, since translating the original
vectorized-array formulas into scalar per-candidate form is exactly the
kind of mechanical step that silently introduces sign/order/clip mistakes.

No pytest dependency -- run directly with `python3 tests/test_core_scoring.py`.
"""
import math
import sys
from dataclasses import dataclass
from pathlib import Path

from astropy.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pyrt_transient.core.scoring import (
    compute_quality_score,
    add_base_quality_scores,
    apply_lightcurve_score_factor,
)


@dataclass
class _W:
    """Minimal stand-in for config.detection, decoupled from
    config_trans.DetectionConfig's actual defaults so this test doesn't
    silently drift if those are legitimately retuned later.
    """
    magnitude_weight: float = 1.0
    significance_weight: float = 2.0
    consistency_weight: float = 1.5
    isolation_weight: float = 1.0
    lc_shape_weight: float = 1.0
    trail_downweight_factor: float = 3.0


DEFAULT_W = _W()


def _assert_close(actual, expected, label, tol=1e-9):
    assert abs(actual - expected) < tol, f"{label}: expected {expected}, got {actual}"


def test_no_features_gives_neutral_score():
    # Every stage's guard is false -> base_score=1, lightcurve_boost=1,
    # mag_range_factor=1 -> product is 1.0.
    r = compute_quality_score({}, DEFAULT_W)
    _assert_close(r, 1.0, "no_features")
    print("test_no_features_gives_neutral_score: PASS")


def test_fwhm_ratio_factor():
    # exp(-((1.0-1)**2)/0.5) * consistency_weight(1.5) = 1.0 * 1.5 = 1.5
    r = compute_quality_score({"fwhm_ratio": 1.0}, DEFAULT_W)
    _assert_close(r, 1.5, "fwhm_ratio=1.0")

    # exp(-((2.0-1)**2)/0.5) = exp(-2) = 0.1353352832...; * 1.5 = 0.2030029249...
    r2 = compute_quality_score({"fwhm_ratio": 2.0}, DEFAULT_W)
    _assert_close(r2, math.exp(-2.0) * 1.5, "fwhm_ratio=2.0")
    print("test_fwhm_ratio_factor: PASS")


def test_axis_ratio_factor_and_clip():
    # clip(0.5,0,1)=0.5 * consistency_weight(1.5) = 0.75
    r = compute_quality_score({"axis_ratio": 0.5}, DEFAULT_W)
    _assert_close(r, 0.75, "axis_ratio=0.5")

    # clip(1.5,0,1)=1.0 (clipped) * 1.5 = 1.5
    r2 = compute_quality_score({"axis_ratio": 1.5}, DEFAULT_W)
    _assert_close(r2, 1.5, "axis_ratio=1.5 (clipped)")
    print("test_axis_ratio_factor_and_clip: PASS")


def test_snr_auto_factor():
    # clip(20/20,0,1)=1.0 * significance_weight(2.0) = 2.0
    r = compute_quality_score({"snr_auto": 20.0}, DEFAULT_W)
    _assert_close(r, 2.0, "snr_auto=20")

    # clip(5/20,0,1)=0.25 * 2.0 = 0.5
    r2 = compute_quality_score({"snr_auto": 5.0}, DEFAULT_W)
    _assert_close(r2, 0.5, "snr_auto=5")
    print("test_snr_auto_factor: PASS")


def test_flags_penalty():
    r_flagged = compute_quality_score({"FLAGS": 4}, DEFAULT_W)
    _assert_close(r_flagged, 0.5, "FLAGS=4")
    r_clean = compute_quality_score({"FLAGS": 0}, DEFAULT_W)
    _assert_close(r_clean, 1.0, "FLAGS=0")
    print("test_flags_penalty: PASS")


def test_nearest_source_dist_factor():
    # clip(5/10,0,1)=0.5 * isolation_weight(1.0) = 0.5
    r = compute_quality_score({"nearest_source_dist": 5.0}, DEFAULT_W)
    _assert_close(r, 0.5, "nearest_source_dist=5")
    # clip(20/10,0,1)=1.0 (clipped) * 1.0 = 1.0
    r2 = compute_quality_score({"nearest_source_dist": 20.0}, DEFAULT_W)
    _assert_close(r2, 1.0, "nearest_source_dist=20 (clipped)")
    print("test_nearest_source_dist_factor: PASS")


def test_mag_calib_with_maglim():
    # delta = 17-14 = 3.0 -> mag_factor = 1+0.2*3 = 1.6 -> clip(1.6,0.5,2.0)=1.6 * magnitude_weight(1.0)
    r = compute_quality_score({"MAG_CALIB": 14.0, "MAGLIM": 17.0}, DEFAULT_W)
    _assert_close(r, 1.6, "MAG_CALIB=14,MAGLIM=17")

    # delta = 15 -> mag_factor = 4.0 -> clipped to 2.0
    r_hi = compute_quality_score({"MAG_CALIB": 5.0, "MAGLIM": 20.0}, DEFAULT_W)
    _assert_close(r_hi, 2.0, "MAG_CALIB=5,MAGLIM=20 (clipped high)")

    # delta = 17-25 = -8 -> mag_factor = 1-1.6 = -0.6 -> clipped to 0.5
    r_lo = compute_quality_score({"MAG_CALIB": 25.0, "MAGLIM": 17.0}, DEFAULT_W)
    _assert_close(r_lo, 0.5, "MAG_CALIB=25,MAGLIM=17 (clipped low)")
    print("test_mag_calib_with_maglim: PASS")


def test_mag_calib_without_maglim():
    # mag_factor = exp(-(15-15)/3) = 1.0 -> clip(1.0,0.1,3.0)=1.0
    r = compute_quality_score({"MAG_CALIB": 15.0}, DEFAULT_W)
    _assert_close(r, 1.0, "MAG_CALIB=15, no MAGLIM")

    # mag_factor = exp(-(9-15)/3) = exp(2) = 7.389... -> clipped to 3.0
    r_hi = compute_quality_score({"MAG_CALIB": 9.0}, DEFAULT_W)
    _assert_close(r_hi, 3.0, "MAG_CALIB=9, no MAGLIM (clipped high)")

    # mag_factor = exp(-(30-15)/3) = exp(-5) = 0.006737947 -> clipped to 0.1
    r_lo = compute_quality_score({"MAG_CALIB": 30.0}, DEFAULT_W)
    _assert_close(r_lo, 0.1, "MAG_CALIB=30, no MAGLIM (clipped low)")
    print("test_mag_calib_without_maglim: PASS")


def test_mag_calib_fallback_is_skipped():
    # mag_calib_is_fallback=True must skip the MAG_CALIB branch entirely,
    # even though MAG_CALIB/MAGLIM are both present.
    r = compute_quality_score(
        {"MAG_CALIB": 14.0, "MAGLIM": 17.0, "mag_calib_is_fallback": True}, DEFAULT_W
    )
    _assert_close(r, 1.0, "mag_calib_is_fallback=True")
    print("test_mag_calib_fallback_is_skipped: PASS")


def test_trail_downweight():
    # fwhm_ratio=1.0 gives base 1.5 (as in test_fwhm_ratio_factor), then
    # candidate_type='trail' divides by trail_downweight_factor(3.0) -> 0.5
    r = compute_quality_score({"fwhm_ratio": 1.0, "candidate_type": "trail"}, DEFAULT_W)
    _assert_close(r, 0.5, "trail downweight")
    # non-trail candidate_type must NOT trigger the downweight
    r2 = compute_quality_score({"fwhm_ratio": 1.0, "candidate_type": "new"}, DEFAULT_W)
    _assert_close(r2, 1.5, "non-trail candidate_type unaffected")
    print("test_trail_downweight: PASS")


def test_lightcurve_boost_and_mag_range_factor_are_independent():
    # weighted_mean_mag=12 -> brightness_factor = exp(-(12-15)/3) = exp(1)
    # mag_range=1.0 -> variability_factor = min(1.0/0.5, 3.0) = 2.0 (not capped)
    # lightcurve_boost = exp(1) * 2.0 * lc_shape_weight(1.0)
    # mag_range_factor (stage 3) = mag_range = 1.0 directly, uncapped
    r = compute_quality_score({"weighted_mean_mag": 12.0, "mag_range": 1.0}, DEFAULT_W)
    expected = math.exp(1.0) * 2.0 * 1.0 * 1.0  # base_score=1 (no base features)
    _assert_close(r, expected, "lightcurve_boost uncapped variability")

    # mag_range=10 -> variability_factor = min(10/0.5, 3.0) = 3.0 (CAPPED in stage 2)
    # but mag_range_factor (stage 3) uses the raw mag_range=10.0, NOT capped.
    # weighted_mean_mag=15 -> brightness_factor = exp(0) = 1.0
    # lightcurve_boost = 1.0 * 3.0 * 1.0 = 3.0; total = 1.0 * 3.0 * 10.0 = 30.0
    r2 = compute_quality_score({"weighted_mean_mag": 15.0, "mag_range": 10.0}, DEFAULT_W)
    _assert_close(r2, 30.0, "stage2 cap vs stage3 uncapped asymmetry")
    print("test_lightcurve_boost_and_mag_range_factor_are_independent: PASS")


def test_lightcurve_stage_requires_both_keys():
    # mag_range alone (no weighted_mean_mag) must NOT trigger lightcurve_boost,
    # but DOES still trigger the independent stage-3 mag_range_factor.
    r = compute_quality_score({"mag_range": 2.0}, DEFAULT_W)
    _assert_close(r, 2.0, "mag_range alone: stage3 fires, stage2 does not")
    print("test_lightcurve_stage_requires_both_keys: PASS")


def test_add_base_quality_scores_uses_table_meta_maglim():
    # Two rows, same MAG_CALIB, table-level MAGLIM in meta (multi-key lookup).
    # delta = 17-14 = 3 -> mag_factor = 1.6 * magnitude_weight(1.0) = 1.6
    t = Table({"MAG_CALIB": [14.0, 14.0]})
    t.meta["MAGLIMIT"] = 17.0  # note: not the canonical 'MAGLIM' key
    add_base_quality_scores(t, DEFAULT_W)
    assert list(t["quality_score"]) == [1.6, 1.6], t["quality_score"]
    print("test_add_base_quality_scores_uses_table_meta_maglim: PASS")


def test_add_base_quality_scores_respects_mag_calib_is_fallback():
    t = Table({"MAG_CALIB": [14.0]})
    t.meta["MAGLIM"] = 17.0
    t.meta["mag_calib_is_fallback"] = True
    add_base_quality_scores(t, DEFAULT_W)
    _assert_close(t["quality_score"][0], 1.0, "mag_calib_is_fallback skips MAG_CALIB branch")
    print("test_add_base_quality_scores_respects_mag_calib_is_fallback: PASS")


def test_add_base_quality_scores_empty_table_noop():
    t = Table({"MAG_CALIB": []})
    add_base_quality_scores(t, DEFAULT_W)  # must not raise
    print("test_add_base_quality_scores_empty_table_noop: PASS")


def test_apply_lightcurve_score_factor_multiplies_existing_score():
    # base quality_score already computed (e.g. 2.0 from the early stage);
    # mag_weighted_mean=15 (brightness_factor=1.0), mag_range=1.0
    # (variability_factor=min(1/0.5,3)=2.0) -> lightcurve_boost=1.0*2.0*1.0=2.0
    # mag_range_factor=1.0 (raw). factor = 2.0*1.0 = 2.0. new score = 2.0*2.0=4.0
    t = Table({"quality_score": [2.0], "mag_weighted_mean": [15.0], "mag_range": [1.0]})
    apply_lightcurve_score_factor(t, DEFAULT_W)
    _assert_close(t["quality_score"][0], 4.0, "apply_lightcurve_score_factor")
    print("test_apply_lightcurve_score_factor_multiplies_existing_score: PASS")


def test_apply_lightcurve_score_factor_ignores_maglim_context():
    # Even with MAG_CALIB present, apply_lightcurve_score_factor must NOT
    # re-derive stage 1 -- it only multiplies in the lightcurve factor,
    # regardless of MAG_CALIB/MAGLIM being present or absent on this row.
    t = Table({"quality_score": [5.0], "MAG_CALIB": [9.0], "mag_weighted_mean": [15.0], "mag_range": [1.0]})
    apply_lightcurve_score_factor(t, DEFAULT_W)
    _assert_close(t["quality_score"][0], 10.0, "MAG_CALIB present but ignored by lightcurve-only factor")
    print("test_apply_lightcurve_score_factor_ignores_maglim_context: PASS")


def test_full_combination_with_custom_weights():
    w = _W(magnitude_weight=2.0, significance_weight=1.0, consistency_weight=1.0,
           isolation_weight=1.0, lc_shape_weight=2.0, trail_downweight_factor=2.0)

    features = {
        "fwhm_ratio": 1.0,             # exp(0)*1.0 (consistency_weight=1.0) = 1.0
        "MAG_CALIB": 14.0,
        "MAGLIM": 17.0,                # delta=3 -> 1.6 * magnitude_weight(2.0) = 3.2
        "candidate_type": "trail",     # /= trail_downweight_factor(2.0)
        "weighted_mean_mag": 15.0,     # brightness_factor=1.0
        "mag_range": 1.0,              # variability_factor=min(1/0.5,3)=2.0
    }
    # base_score = 1.0 (fwhm) * 3.2 (mag) = 3.2, then /2.0 (trail) = 1.6
    # lightcurve_boost = 1.0 * 2.0 * lc_shape_weight(2.0) = 4.0
    # mag_range_factor = 1.0
    expected = 1.6 * 4.0 * 1.0
    r = compute_quality_score(features, w)
    _assert_close(r, expected, "full combination with custom weights")
    print("test_full_combination_with_custom_weights: PASS")


if __name__ == "__main__":
    test_no_features_gives_neutral_score()
    test_fwhm_ratio_factor()
    test_axis_ratio_factor_and_clip()
    test_snr_auto_factor()
    test_flags_penalty()
    test_nearest_source_dist_factor()
    test_mag_calib_with_maglim()
    test_mag_calib_without_maglim()
    test_mag_calib_fallback_is_skipped()
    test_trail_downweight()
    test_lightcurve_boost_and_mag_range_factor_are_independent()
    test_lightcurve_stage_requires_both_keys()
    test_add_base_quality_scores_uses_table_meta_maglim()
    test_add_base_quality_scores_respects_mag_calib_is_fallback()
    test_add_base_quality_scores_empty_table_noop()
    test_apply_lightcurve_score_factor_multiplies_existing_score()
    test_apply_lightcurve_score_factor_ignores_maglim_context()
    test_full_combination_with_custom_weights()
    print("All core/scoring.py tests passed.")
