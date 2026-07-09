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


def update_candidate_with_lightcurve_stats(candidate: Table, lightcurve: Table, config=None,
                                            logger=None, add_strategy_fields_fn=None):
    """Update candidate with lightcurve statistics including motion/trail analysis."""

    # Time statistics
    time_span = (np.max(lightcurve['obs_time']) - np.min(lightcurve['obs_time'])) / 3600.0
    candidate['time_span_hours'] = time_span
    candidate['n_detections'] = len(lightcurve)
    candidate['n_epochs'] = len(np.unique(lightcurve['epoch_id']))

    # Position statistics
    ra_std = np.std(lightcurve['ALPHA_J2000']) * 3600
    dec_std = np.std(lightcurve['DELTA_J2000']) * 3600
    candidate['position_scatter_arcsec'] = np.sqrt(ra_std**2 + dec_std**2)

    # Extract per-frame WCS astrometric error (ASTSIGMA) from lightcurve metadata.
    # Stored by zpnfit.py in the ECSV header; present only when astrometric calibration ran.
    astsigma = 0.0
    if hasattr(lightcurve, 'meta') and lightcurve.meta:
        raw = lightcurve.meta.get('ASTSIGMA')
        if raw is not None:
            try:
                astsigma = float(raw)
            except (TypeError, ValueError):
                pass

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
        mags = lightcurve['MAG_CALIB']
        mag_errors = lightcurve['MAGERR_CALIB']

        # Total variation of lightcurve in magnitudes (sum of absolute frame-to-frame changes).
        # Measures how "erratic" the lightcurve is: a flat LC gives 0, a noisy one
        # gives ~sqrt(2)*n*mag_std.  Units: magnitudes.
        candidate['mag_schizo'] = float(np.sum(np.abs(np.diff(mags))))

        # Weighted mean magnitude
        weights = 1.0 / mag_errors**2
        weighted_mean_mag = np.sum(mags * weights) / np.sum(weights)
        candidate['mag_weighted_mean'] = weighted_mean_mag

        # Magnitude variability metrics
        candidate['mag_range'] = np.max(mags) - np.min(mags)
        candidate['mag_std'] = np.std(mags)

        # Chi-squared test for variability
        chi2 = np.sum(((mags - weighted_mean_mag) / mag_errors)**2)
        reduced_chi2 = chi2 / (len(mags) - 1) if len(mags) > 1 else 0
        candidate['mag_chi2_reduced'] = reduced_chi2
        candidate['is_variable'] = reduced_chi2 > 2.0

        # NOTE: the brightness/variability quality_score boost that used to be
        # applied right here (`quality_score *= brightness_factor *
        # variability_factor * lc_shape_weight`) is deleted -- it's folded
        # into core/scoring.compute_quality_score's lightcurve_boost stage
        # (wired in at step 11), which reads mag_weighted_mean/mag_range from
        # this candidate row as its input features.

    # Add strategy calculation
    if add_strategy_fields_fn is not None:
        add_strategy_fields_fn(candidate, lightcurve)
