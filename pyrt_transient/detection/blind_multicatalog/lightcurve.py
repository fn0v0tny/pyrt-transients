"""Lightcurve building and stats. build_lightcurve_for_group keeps its
pre-built-per-epoch-KDTree optimization as-is (unlike clustering.py, which
uses core/matching.match_radius instead).

update_candidate_with_lightcurve_stats does NOT mutate quality_score itself
-- that multiplication lives inside core/scoring.compute_quality_score only.
weighted_mean_mag/mag_range are still computed and stored as candidate
columns since compute_quality_score's lightcurve_boost stage needs them as
input features.
"""

from typing import List

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
import astropy.units as u

from pyrt_transient.core.radii import astrometric_scatter_arcsec
from pyrt_transient.detection.blind_multicatalog.trail_detection import (
    compute_motion_features,
    compute_trail_features,
    compute_trail_score,
)


def build_lightcurve_for_group(
    group_candidates: Table,
    group_epochs: List[int],
    all_epoch_detections: List[Table],
    match_radius: float,
    epoch_kdtrees=None,
) -> Table:
    """Build lightcurve for a group.

    Uses pre-built epoch KDTrees (passed from the caller) for fast nearest-
    neighbour lookup instead of creating SkyCoord arrays per epoch per group.
    Falls back to the SkyCoord approach when KDTrees are not provided.
    """

    # Calculate mean position
    mean_ra = np.mean(group_candidates['ALPHA_J2000'])
    mean_dec = np.mean(group_candidates['DELTA_J2000'])

    # Convert to 3-D cartesian for KDTree chord-length queries
    mean_ra_r = np.radians(mean_ra)
    mean_dec_r = np.radians(mean_dec)
    target_xyz = np.array([
        np.cos(mean_dec_r) * np.cos(mean_ra_r),
        np.cos(mean_dec_r) * np.sin(mean_ra_r),
        np.sin(mean_dec_r)
    ])
    chord = 2 * np.sin(np.radians(match_radius / 3600) / 2)

    # Collect matching detections
    all_detections = []

    for epoch_id, epoch_detections in enumerate(all_epoch_detections):
        if len(epoch_detections) == 0:
            continue

        try:
            # Fast path: use pre-built KDTree if available
            if (epoch_kdtrees is not None
                    and epoch_id < len(epoch_kdtrees)
                    and epoch_kdtrees[epoch_id] is not None):
                tree, xyz, valid_det = epoch_kdtrees[epoch_id]
                indices = tree.query_radius([target_xyz], r=chord)[0]
                if len(indices) > 0:
                    dists = np.linalg.norm(xyz[indices] - target_xyz, axis=1)
                    closest_idx = indices[np.argmin(dists)]
                    all_detections.append(valid_det[closest_idx])
            else:
                # Fallback: original SkyCoord approach
                ra_values = np.array(epoch_detections['ALPHA_J2000'], dtype=float)
                dec_values = np.array(epoch_detections['DELTA_J2000'], dtype=float)

                valid_mask = np.isfinite(ra_values) & np.isfinite(dec_values)
                if not np.any(valid_mask):
                    continue

                ra_values = ra_values[valid_mask]
                dec_values = dec_values[valid_mask]
                valid_detections = epoch_detections[valid_mask]

                target_coord = SkyCoord(ra=mean_ra*u.deg, dec=mean_dec*u.deg)
                epoch_coords = SkyCoord(
                    ra=ra_values*u.deg,
                    dec=dec_values*u.deg
                )

                separations = target_coord.separation(epoch_coords)
                matches = separations < match_radius*u.arcsec

                if np.any(matches):
                    closest_idx = np.argmin(separations[matches])
                    matched_detection = valid_detections[matches][closest_idx]
                    all_detections.append(matched_detection)

        except Exception:
            continue

    if all_detections:
        lightcurve = vstack(all_detections)
        lightcurve.sort('obs_time')
        return lightcurve
    else:
        return Table()


def estimate_magnitude_error_floor(
    all_epoch_detections: List[Table],
    epoch_kdtrees=None,
    match_radius_arcsec: float = 1.5,
    max_err: float = 0.03,
    min_epoch_fraction: float = 0.7,
    min_stars: int = 10,
):
    """Empirical additive magnitude-error floor of this campaign, from the
    repeatability of bright constant stars.

    MAGERR_CALIB is right for faint stars (rms/err ~ 1.0-1.1) and wrong for
    bright ones: on the 15-field GRB replay the rms of 12-15 mag stars is
    0.02-0.04 mag (0.1 on a poor night) against quoted errors of
    0.003-0.01 (FUTURE_IDEAS.md, 2026-09-04). The excess is an additive
    floor (flat field, PSF and zeropoint variations), so the model is
    err_eff^2 = err^2 + floor^2 with floor measured here as the median of
    sqrt(max(rms^2 - err^2, 0)) over unflagged stars with quoted error
    below `max_err` present in at least `min_epoch_fraction` of the epochs.

    Returns (floor, n_stars); floor is NaN when fewer than `min_stars`
    qualify (fewer than 3 epochs, a sparse field, ...).
    """
    tables = [t for t in all_epoch_detections if t is not None and len(t) > 0]
    if len(tables) < 3:
        return float('nan'), 0
    if epoch_kdtrees is None or len(epoch_kdtrees) != len(all_epoch_detections):
        epoch_kdtrees = [_epoch_kdtree(t) for t in all_epoch_detections]
    usable = [(i, epoch_kdtrees[i]) for i, t in enumerate(all_epoch_detections)
              if t is not None and len(t) > 0 and epoch_kdtrees[i] is not None]
    if len(usable) < 3:
        return float('nan'), 0
    ref_i = max(usable, key=lambda p: len(p[1][2]))[0]
    _, ref_xyz, ref_det = epoch_kdtrees[ref_i]
    n_ref = len(ref_det)
    n_ep = len(usable)
    mags = np.full((n_ep, n_ref), np.nan)
    errs = np.full((n_ep, n_ref), np.nan)
    chord = 2 * np.sin(np.radians(match_radius_arcsec / 3600) / 2)
    for row, (i, (tree, xyz, det)) in enumerate(usable):
        if 'MAG_CALIB' not in det.colnames or 'MAGERR_CALIB' not in det.colnames:
            continue
        dist, idx = tree.query(ref_xyz, k=1)
        dist, idx = np.asarray(dist).ravel(), np.asarray(idx).ravel()
        ok = dist < chord
        m = np.ma.filled(np.ma.asarray(det['MAG_CALIB'], dtype=float), np.nan)[idx]
        e = np.ma.filled(np.ma.asarray(det['MAGERR_CALIB'], dtype=float), np.nan)[idx]
        if 'FLAGS' in det.colnames:
            flagged = np.ma.filled(np.ma.asarray(det['FLAGS']), 0).astype(int)[idx] != 0
            ok &= ~flagged
        mags[row, ok] = m[ok]
        errs[row, ok] = e[ok]
    good = np.isfinite(mags) & np.isfinite(errs) & (errs > 0)
    keep = good.sum(axis=0) >= max(3, int(np.ceil(min_epoch_fraction * n_ep)))
    if keep.sum() < min_stars:
        return float('nan'), 0
    m = np.where(good, mags, np.nan)[:, keep]
    e = np.where(good, errs, np.nan)[:, keep]
    with np.errstate(invalid='ignore'):
        rms = np.nanstd(m, axis=0, ddof=1)
        err_mean = np.sqrt(np.nanmean(e ** 2, axis=0))
    bright = np.isfinite(rms) & np.isfinite(err_mean) & (err_mean < max_err)
    if bright.sum() < min_stars:
        return float('nan'), int(bright.sum())
    floor = float(np.median(np.sqrt(np.maximum(rms[bright] ** 2 - err_mean[bright] ** 2, 0.0))))
    return floor, int(bright.sum())


def _epoch_kdtree(det: Table):
    """(KDTree, xyz, valid_rows) for one epoch table, or None."""
    from sklearn.neighbors import KDTree
    if det is None or len(det) == 0:
        return None
    ra = np.ma.filled(np.ma.asarray(det['ALPHA_J2000'], dtype=float), np.nan)
    dec = np.ma.filled(np.ma.asarray(det['DELTA_J2000'], dtype=float), np.nan)
    valid = np.isfinite(ra) & np.isfinite(dec)
    if not np.any(valid):
        return None
    ra_r, dec_r = np.radians(ra[valid]), np.radians(dec[valid])
    xyz = np.column_stack((np.cos(dec_r) * np.cos(ra_r), np.cos(dec_r) * np.sin(ra_r), np.sin(dec_r)))
    return KDTree(xyz), xyz, det[valid]


def update_candidate_with_lightcurve_stats(candidate: Table, lightcurve: Table, config=None,
                                            logger=None, magerr_floor: float = 0.0):
    """Update candidate with lightcurve statistics including motion/trail analysis.

    magerr_floor: additive magnitude-error floor of the campaign (see
    estimate_magnitude_error_floor); applied to the variability statistics
    (mag_chi2_reduced, is_variable), not to the weighted mean, which feeds
    the frozen quality_score and stays as it was.
    """

    # Time statistics
    time_span = (np.max(lightcurve['obs_time']) - np.min(lightcurve['obs_time'])) / 3600.0
    candidate['time_span_hours'] = time_span
    candidate['n_detections'] = len(lightcurve)
    candidate['n_epochs'] = len(np.unique(lightcurve['epoch_id']))

    # Position statistics
    ra_std = np.std(lightcurve['ALPHA_J2000']) * 3600
    dec_std = np.std(lightcurve['DELTA_J2000']) * 3600
    candidate['position_scatter_arcsec'] = np.sqrt(ra_std**2 + dec_std**2)

    # Per-frame astrometric scatter, in arcsec like the motion it is compared
    # with. pyrt writes it in pixels (ASTSCATT, or ASTSIGMA from pyrt before
    # 97101a7); used unconverted, the trail displacement threshold was PIXEL
    # times too tight. 0 disables that criterion.
    astsigma = astrometric_scatter_arcsec(getattr(lightcurve, 'meta', None), default=0.0)

    # Motion analysis - compute motion features
    motion_features = compute_motion_features(lightcurve)
    for key, value in motion_features.items():
        candidate[key] = value

    # Trail features - analyze shape across epochs
    trail_features = compute_trail_features(lightcurve, motion_features)
    for key, value in trail_features.items():
        candidate[key] = value

    # Trail scoring and decision (pass astsigma and time_span for WCS-error check)
    trail_score, is_trail = compute_trail_score(
        motion_features, trail_features, config=config, logger=logger,
        astsigma=astsigma, time_span_hours=float(time_span)
    )
    candidate['trail_score'] = trail_score

    # Set candidate type based on trail analysis (unless already set to strong photometric event)
    # Read candidate_type safely from single-row table
    if 'candidate_type' in candidate.colnames and len(candidate['candidate_type']) > 0:
        ct_val = candidate['candidate_type'][0]
        current_type = ct_val[0] if isinstance(ct_val, (list, np.ndarray)) else ct_val
    else:
        current_type = 'unknown'
    if is_trail and current_type not in ['brightening', 'fading']:
        candidate['candidate_type'] = ['trail']
    elif 'candidate_type' not in candidate.colnames:
        candidate['candidate_type'] = ['unknown']

    # Photometric statistics
    if 'MAG_CALIB' in lightcurve.colnames:
        # Masked/non-finite photometry (a row calibrated in a band with no
        # zero point, or a masked cell) must not propagate: a masked
        # reduced_chi2 makes `is_variable` a masked float, and the final
        # vstack then fails with "columns have incompatible types
        # ['float64', 'bool', ...]" (seen on the i-band GRB 211024B field).
        mags_all = np.ma.filled(np.ma.asarray(lightcurve['MAG_CALIB'], dtype=float), np.nan)
        errs_all = np.ma.filled(np.ma.asarray(lightcurve['MAGERR_CALIB'], dtype=float), np.nan)
        good = np.isfinite(mags_all) & np.isfinite(errs_all) & (errs_all > 0)
        mags = mags_all[good]
        mag_errors = errs_all[good]

        floor = float(magerr_floor) if magerr_floor and np.isfinite(magerr_floor) else 0.0
        candidate['magerr_floor'] = floor
        if len(mags) > 0:
            # Total variation of lightcurve in magnitudes (sum of absolute frame-to-frame changes).
            # Measures how "erratic" the lightcurve is: a flat LC gives 0, a noisy one
            # gives ~sqrt(2)*n*mag_std.  Units: magnitudes.
            candidate['mag_schizo'] = float(np.sum(np.abs(np.diff(mags))))

            # Weighted mean magnitude
            weights = 1.0 / mag_errors**2
            weighted_mean_mag = float(np.sum(mags * weights) / np.sum(weights))
            candidate['mag_weighted_mean'] = weighted_mean_mag

            # Magnitude variability metrics
            candidate['mag_range'] = float(np.max(mags) - np.min(mags))
            candidate['mag_std'] = float(np.std(mags))

            # Chi-squared test for variability, against errors with the
            # campaign's empirical floor added in quadrature (the quoted
            # MAGERR_CALIB of a bright star is several times too small).
            eff_errors = np.sqrt(mag_errors ** 2 + floor ** 2)
            eff_weights = 1.0 / eff_errors ** 2
            eff_mean = float(np.sum(mags * eff_weights) / np.sum(eff_weights))
            dof = len(mags) - 1
            chi2_raw = float(np.sum(((mags - weighted_mean_mag) / mag_errors)**2))
            chi2 = float(np.sum(((mags - eff_mean) / eff_errors)**2))
            reduced_chi2_raw = chi2_raw / dof if dof > 0 else 0.0
            reduced_chi2 = chi2 / dof if dof > 0 else 0.0
        else:
            candidate['mag_schizo'] = 0.0
            candidate['mag_weighted_mean'] = np.nan
            candidate['mag_range'] = 0.0
            candidate['mag_std'] = 0.0
            reduced_chi2 = reduced_chi2_raw = 0.0
        candidate['mag_chi2_reduced_raw'] = float(reduced_chi2_raw)
        candidate['mag_chi2_reduced'] = float(reduced_chi2)
        candidate['is_variable'] = bool(reduced_chi2 > 2.0)

        # NOTE: the brightness/variability quality_score boost that used to be
        # applied right here (`quality_score *= brightness_factor *
        # variability_factor * lc_shape_weight`) is deleted -- it's folded
        # into core/scoring.compute_quality_score's lightcurve_boost stage
        # (wired in at step 11), which reads mag_weighted_mean/mag_range from
        # this candidate row as its input features.

