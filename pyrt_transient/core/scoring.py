"""Quality-score computation -- the single merge of the three places that
used to mutate quality_score independently:

  1. _add_quality_metrics (transient_analyser.py) -- the base per-catalog
     formula: shape/SNR/flag/isolation/magnitude factors, plus trail
     downweighting.
  2. _update_candidate_with_lightcurve_stats's
     `*= brightness_factor * variability_factor * lc_shape_weight`
  3. _combine_with_lightcurves's `*= mag_range`

This merge is covered by hand-computed Tier 0 tests in
tests/test_core_scoring.py, not just the fixture-based check_baseline.py.

`features` is a flat dict of already-prepared, per-candidate values (not a
raw astropy Table/meta -- callers normalize column presence and the
multi-key MAGLIM lookup (`MAGLIM`/`MAGLIMIT`/`maglim`/`maglimit` in the
original code) into a single `MAGLIM` key before calling this). Any key
absent from `features` means that stage's original "if column in table"
guard is false, contributing a neutral factor (skipped, not zero).

`weights` is any object exposing the same attributes as
config_trans.DetectionConfig (magnitude_weight, significance_weight,
consistency_weight, isolation_weight, trail_downweight_factor,
lc_shape_weight) -- normally `config.detection` itself.
"""

import math
from typing import Any, Dict, Optional, Tuple

from astropy.table import Table


def compute_base_score(features: Dict[str, Any], weights) -> float:
    """Stage 1 only, from _add_quality_metrics -- the part computable before
    lightcurve stats exist (uses only detection/catalog-context features).
    Called early, per-catalog, since combine_results' min_quality filtering
    needs a quality_score before any lightcurve exists.
    """
    base_score = 1.0

    if "fwhm_ratio" in features:
        base_score *= math.exp(-((features["fwhm_ratio"] - 1) ** 2) / 0.5) * weights.consistency_weight

    if "axis_ratio" in features:
        base_score *= _clip(features["axis_ratio"], 0, 1) * weights.consistency_weight

    if "snr_auto" in features:
        base_score *= _clip(features["snr_auto"] / 20.0, 0, 1) * weights.significance_weight

    if "FLAGS" in features:
        base_score *= 0.5 if features["FLAGS"] > 0 else 1.0

    if "nearest_source_dist" in features:
        base_score *= _clip(features["nearest_source_dist"] / 10.0, 0, 1) * weights.isolation_weight

    if "MAG_CALIB" in features and not features.get("mag_calib_is_fallback", False):
        maglim = features.get("MAGLIM")
        if maglim is not None:
            # Reward being brighter than the limiting magnitude.
            delta = maglim - features["MAG_CALIB"]
            mag_factor = 1.0 + 0.2 * delta  # each mag brighter than limit boosts by 0.2
            mag_factor = _clip(mag_factor, 0.5, 2.0) * weights.magnitude_weight
            base_score *= mag_factor
        else:
            # No MAGLIM: gentle brightness prior around 15 mag using MAG_CALIB only.
            mag_factor = math.exp(-(features["MAG_CALIB"] - 15.0) / 3.0) * weights.magnitude_weight
            base_score *= _clip(mag_factor, 0.1, 3.0)

    if features.get("candidate_type") == "trail":
        base_score /= weights.trail_downweight_factor

    return base_score


def compute_lightcurve_score_factor(features: Dict[str, Any], weights) -> float:
    """Stages 2+3 only, from _update_candidate_with_lightcurve_stats's
    `*= brightness_factor * variability_factor * lc_shape_weight` and
    _combine_with_lightcurves's `*= mag_range`. Called once, late, per final
    candidate, and MULTIPLIED into the already-computed base_score --
    exactly matching the original code's `quality_score *= factor` shape.
    Deliberately does NOT touch MAG_CALIB/MAGLIM (the final candidate row
    lives in a fresh Table() that doesn't carry the original table-level
    MAGLIM meta, so re-deriving stage 1 here would silently use the wrong
    MAGLIM context -- see clustering.py's combine_with_lightcurves).
    """
    lightcurve_boost = 1.0
    if "weighted_mean_mag" in features and "mag_range" in features:
        brightness_factor = math.exp(-(features["weighted_mean_mag"] - 15.0) / 3.0)
        variability_factor = min(features["mag_range"] / 0.5, 3.0)  # capped at 3x boost
        lightcurve_boost = brightness_factor * variability_factor * weights.lc_shape_weight

    mag_range_factor = 1.0
    if "mag_range" in features:
        mag_range_factor = features["mag_range"]

    return lightcurve_boost * mag_range_factor


def compute_quality_score(features: Dict[str, Any], weights) -> float:
    """The full merge: compute_base_score(...) * compute_lightcurve_score_factor(...).
    Only valid when `features` already has both the detection/catalog-context
    keys AND (if available) the lightcurve keys with correct MAGLIM/
    mag_calib_is_fallback context -- i.e. a single, freshly-built features
    dict, not a candidate row that has passed through separate early/late
    processing stages. See compute_base_score's docstring for why the real
    pipeline calls the two stages separately instead.
    """
    return compute_base_score(features, weights) * compute_lightcurve_score_factor(features, weights)


def _row_features(row, maglim: Optional[float], mag_calib_is_fallback: bool) -> Dict[str, Any]:
    """Build a features dict from one candidate table row."""
    features: Dict[str, Any] = {}
    for col in ("fwhm_ratio", "axis_ratio", "snr_auto", "FLAGS", "nearest_source_dist",
                "MAG_CALIB", "candidate_type"):
        if col in row.colnames:
            features[col] = row[col]
    if "MAG_CALIB" in features:
        features["MAGLIM"] = maglim
        features["mag_calib_is_fallback"] = mag_calib_is_fallback
    # Column names on the candidate table differ from compute_quality_score's
    # feature-dict keys for the lightcurve stage.
    if "mag_weighted_mean" in row.colnames:
        features["weighted_mean_mag"] = row["mag_weighted_mean"]
    if "mag_range" in row.colnames:
        features["mag_range"] = row["mag_range"]
    return features


def _table_maglim_and_fallback(meta) -> Tuple[Optional[float], bool]:
    maglim = None
    for key in ('MAGLIM', 'MAGLIMIT', 'maglim', 'maglimit'):
        if key in meta:
            try:
                maglim = float(meta[key])
            except (TypeError, ValueError):
                maglim = None
            break
    mag_calib_is_fallback = bool(meta.get('mag_calib_is_fallback', False))
    return maglim, mag_calib_is_fallback


def add_base_quality_scores(candidates: Table, weights) -> None:
    """Early stage (catalog_match.py): compute and write
    candidates['quality_score'] for every row using compute_base_score only
    (no lightcurve features exist yet at this point). Replaces
    _add_quality_metrics's body.
    """
    if len(candidates) == 0:
        return
    maglim, mag_calib_is_fallback = _table_maglim_and_fallback(candidates.meta)
    scores = [
        compute_base_score(_row_features(row, maglim, mag_calib_is_fallback), weights)
        for row in candidates
    ]
    candidates['quality_score'] = scores


def apply_lightcurve_score_factor(candidate: Table, weights) -> None:
    """Late stage (clustering.py's combine_with_lightcurves): multiply the
    already-computed base quality_score by compute_lightcurve_score_factor,
    using this single-row candidate's mag_weighted_mean/mag_range columns
    (set by lightcurve.update_candidate_with_lightcurve_stats just before
    this is called). Replaces the deleted `quality_score *=` mutations.
    """
    features = _row_features(candidate[0], maglim=None, mag_calib_is_fallback=False)
    factor = compute_lightcurve_score_factor(features, weights)
    candidate['quality_score'] = candidate['quality_score'] * factor


def _clip(value, lo, hi):
    return min(max(value, lo), hi)
