"""Motion/trail detection for moving-object candidates -- moved verbatim
from OptimizedMultiDetectionAnalyzer._compute_motion_features,
._compute_trail_features, ._compute_trail_score. config/logger are now
explicit parameters instead of self.config/self.logger.
"""

from typing import Dict, Tuple

import numpy as np


def compute_motion_features(lightcurve) -> Dict[str, float]:
    """
    Compute motion features per group based on linear fit to positions.

    Args:
        lightcurve: Table with ALPHA_J2000, DELTA_J2000, obs_time, epoch_id, SNR columns

    Returns:
        Dictionary of motion features
    """
    if len(lightcurve) < 2:
        return {
            'motion_ra_as_per_hr': 0.0,
            'motion_dec_as_per_hr': 0.0,
            'motion_rate_as_per_hr': 0.0,
            'motion_sigma_as': 0.0,
            'motion_significance': 0.0,
            'n_epochs_moving': 0
        }

    # Choose reference epoch (highest SNR or median MJD)
    if 'SNR' in lightcurve.colnames:
        ref_idx = np.argmax(lightcurve['SNR'])
    elif 'mjd' in lightcurve.colnames:
        median_mjd = np.median(lightcurve['mjd'])
        ref_idx = np.argmin(np.abs(lightcurve['mjd'] - median_mjd))
    else:
        ref_idx = len(lightcurve) // 2

    ref_ra = lightcurve['ALPHA_J2000'][ref_idx]
    ref_dec = lightcurve['DELTA_J2000'][ref_idx]
    ref_time = lightcurve['obs_time'][ref_idx]

    # Convert times to hours relative to reference
    t_hr = (lightcurve['obs_time'] - ref_time) / 3600.0

    # Compute residuals in arcsec
    delta_ra_as = (lightcurve['ALPHA_J2000'] - ref_ra) * 3600.0 * np.cos(np.radians(ref_dec))
    delta_dec_as = (lightcurve['DELTA_J2000'] - ref_dec) * 3600.0

    # Linear fits for motion
    try:
        # RA motion: ΔRA_as = a_ra * t_hr + b_ra
        ra_coeffs = np.polyfit(t_hr, delta_ra_as, 1)
        a_ra = ra_coeffs[0]  # motion_ra_as_per_hr

        # Dec motion: ΔDec_as = a_dec * t_hr + b_dec
        dec_coeffs = np.polyfit(t_hr, delta_dec_as, 1)
        a_dec = dec_coeffs[0]  # motion_dec_as_per_hr

        # Compute residuals from linear fits
        ra_pred = np.polyval(ra_coeffs, t_hr)
        dec_pred = np.polyval(dec_coeffs, t_hr)
        res_ra = delta_ra_as - ra_pred
        res_dec = delta_dec_as - dec_pred

        # Robust scatter via MAD/0.67
        mad_ra = np.median(np.abs(res_ra - np.median(res_ra))) / 0.67
        mad_dec = np.median(np.abs(res_dec - np.median(res_dec))) / 0.67
        motion_sigma_as = np.sqrt(mad_ra**2 + mad_dec**2)

        # Total motion rate
        motion_rate_as_per_hr = np.sqrt(a_ra**2 + a_dec**2)

        # Motion significance
        eps = 1e-6
        motion_significance = motion_rate_as_per_hr / max(motion_sigma_as, eps)

        # Count epochs following motion (residual norm <= 2 * sigma)
        res_norm = np.sqrt(res_ra**2 + res_dec**2)
        n_epochs_moving = np.sum(res_norm <= 2.0 * motion_sigma_as)

    except (np.linalg.LinAlgError, ValueError):
        # Fallback for degenerate cases
        a_ra = 0.0
        a_dec = 0.0
        motion_rate_as_per_hr = 0.0
        motion_sigma_as = 0.0
        motion_significance = 0.0
        n_epochs_moving = 0

    return {
        'motion_ra_as_per_hr': a_ra,
        'motion_dec_as_per_hr': a_dec,
        'motion_rate_as_per_hr': motion_rate_as_per_hr,
        'motion_sigma_as': motion_sigma_as,
        'motion_significance': motion_significance,
        'n_epochs_moving': int(n_epochs_moving)
    }


def compute_trail_features(lightcurve, motion_features: Dict = None) -> Dict[str, float]:
    """
    Compute trail cues from shape parameters across epochs.

    Args:
        lightcurve: Table with shape columns (A_IMAGE, B_IMAGE, THETA_IMAGE, FWHM_IMAGE)
        motion_features: Dictionary from compute_motion_features (for alignment calculation)

    Returns:
        Dictionary of trail shape features
    """
    if len(lightcurve) == 0:
        return {'elongation_mean': 0.0, 'fwhm_ratio_mean': 1.0, 'align_mean': 0.0}

    # Per-epoch shape features
    elongations = []
    fwhm_ratios = []

    # Axis ratio and elongation
    if 'A_IMAGE' in lightcurve.colnames and 'B_IMAGE' in lightcurve.colnames:
        a_img = lightcurve['A_IMAGE']
        b_img = lightcurve['B_IMAGE']

        # Filter out invalid values
        valid_ab = (a_img > 0) & (b_img > 0) & np.isfinite(a_img) & np.isfinite(b_img)
        if np.any(valid_ab):
            axis_ratios = b_img[valid_ab] / a_img[valid_ab]
            elongations = 1.0 - axis_ratios
            elongations = np.clip(elongations, 0.0, 1.0)  # Ensure valid range

    # FWHM ratio relative to median
    if 'FWHM_IMAGE' in lightcurve.colnames:
        fwhm_vals = lightcurve['FWHM_IMAGE']
        valid_fwhm = (fwhm_vals > 0) & np.isfinite(fwhm_vals)

        if np.any(valid_fwhm):
            median_fwhm = np.median(fwhm_vals[valid_fwhm])
            if median_fwhm > 0:
                fwhm_ratios = fwhm_vals[valid_fwhm] / median_fwhm

    # Compute means
    elongation_mean = np.mean(elongations) if len(elongations) > 0 else 0.0
    fwhm_ratio_mean = np.mean(fwhm_ratios) if len(fwhm_ratios) > 0 else 1.0

    # Alignment with motion direction
    align_mean = 0.0
    if (motion_features and 'THETA_IMAGE' in lightcurve.colnames and
        motion_features.get('motion_rate_as_per_hr', 0) > 0):

        # Calculate motion direction angle (East of North)
        motion_ra = motion_features.get('motion_ra_as_per_hr', 0)
        motion_dec = motion_features.get('motion_dec_as_per_hr', 0)

        if motion_ra != 0 or motion_dec != 0:
            motion_angle_deg = np.degrees(np.arctan2(motion_ra, motion_dec))
            # Normalize to [0, 180] range for comparison with THETA_IMAGE
            if motion_angle_deg < 0:
                motion_angle_deg += 180
            elif motion_angle_deg >= 180:
                motion_angle_deg -= 180

            # Calculate alignment scores for each epoch
            theta_vals = lightcurve['THETA_IMAGE']
            valid_theta = np.isfinite(theta_vals)

            if np.any(valid_theta):
                alignment_scores = []
                for theta_img in theta_vals[valid_theta]:
                    # Calculate angular difference between motion and shape orientation
                    angle_diff = abs(motion_angle_deg - theta_img)

                    # Handle wraparound at 180 degrees (position angles are symmetric)
                    if angle_diff > 90:
                        angle_diff = 180 - angle_diff

                    # Convert to alignment score: 1.0 for perfect alignment (0°), 0.0 for perpendicular (90°)
                    alignment_score = 1.0 - (angle_diff / 90.0)
                    alignment_scores.append(alignment_score)

                align_mean = np.mean(alignment_scores) if alignment_scores else 0.0

    return {
        'elongation_mean': elongation_mean,
        'fwhm_ratio_mean': fwhm_ratio_mean,
        'align_mean': align_mean
    }


def compute_trail_score(
    motion_features: Dict,
    trail_features: Dict,
    config=None,
    logger=None,
    astsigma: float = 0.0,
    time_span_hours: float = 0.0,
) -> Tuple[float, bool]:
    """
    Compute trail score and decision based on motion and shape features.

    Args:
        motion_features: Dictionary from compute_motion_features
        trail_features: Dictionary from compute_trail_features
        config: Optional PipelineConfig for trail parameters (defaults used if None)
        logger: Optional logger for debug messages
        astsigma: Per-frame WCS astrometric error in arcsec (ASTSIGMA from header).
                  When > 0, enables a physics-based "displacement vs WCS error" trail
                  criterion that flags sources whose total displacement exceeds
                  trail_astsigma_displacement_threshold × ASTSIGMA, independently of
                  the score-based criterion.
        time_span_hours: Observation time span in hours (needed for displacement).

    Returns:
        Tuple of (trail_score, is_trail_decision)
    """
    # Get config parameters (use defaults if no config)
    if config:
        min_epochs = config.detection.trail_min_epochs
        motion_sigma_min = config.detection.trail_motion_sigma_min
        motion_sig_tau = config.detection.trail_motion_sig_tau
        score_threshold = config.detection.trail_score_threshold
        astsigma_threshold = config.detection.trail_astsigma_displacement_threshold
    else:
        min_epochs = 3
        motion_sigma_min = 0.5
        motion_sig_tau = 3.0
        score_threshold = 0.7
        astsigma_threshold = 3.0

    # Normalize components to [0, 1] for scoring
    elongation_norm = np.clip(trail_features['elongation_mean'], 0.0, 1.0)

    # Motion significance normalised to [0,1] (10σ = max)
    motion_sig_norm = np.clip(motion_features['motion_significance'] / 10.0, 0.0, 1.0)

    # Alignment component
    align_norm = np.clip(trail_features['align_mean'], 0.0, 1.0)

    # WCS-error displacement component: total positional displacement relative to
    # the per-frame astrometric error (ASTSIGMA).  A source that moves more than
    # astsigma_threshold × ASTSIGMA over the whole observation is physically
    # displaced beyond measurement noise.  Normalised: 5 × ASTSIGMA = max score.
    eps = 1e-9
    astsigma_displacement = 0.0
    if astsigma > 0.0 and time_span_hours > 0.0:
        total_displacement = motion_features['motion_rate_as_per_hr'] * time_span_hours
        astsigma_displacement = total_displacement / max(astsigma, eps)
    astsigma_norm = np.clip(astsigma_displacement / 5.0, 0.0, 1.0)

    # Trail score — weights adjusted to include the new ASTSIGMA component.
    # Old: w1=0.4 (elongation), w2=0.4 (motion_sig), w3=0.2 (align)
    # New: split w2 between motion_sig and astsigma so shape+motion still dominate.
    w1, w2, w3, w4 = 0.35, 0.30, 0.15, 0.20
    trail_score = (w1 * elongation_norm + w2 * motion_sig_norm
                   + w3 * align_norm + w4 * astsigma_norm)

    # Primary decision criteria (unchanged from original)
    n_epochs = motion_features['n_epochs_moving']
    motion_sigma = motion_features['motion_sigma_as']
    motion_significance = motion_features['motion_significance']

    is_trail_score = (
        n_epochs >= min_epochs
        and motion_sigma >= motion_sigma_min
        and motion_significance >= motion_sig_tau
        and trail_score >= score_threshold
    )

    # Secondary criterion: displacement > N × ASTSIGMA (physics-based).
    # Triggers independently of the score threshold, so moving objects with
    # point-source morphology (low elongation) are still caught.
    is_trail_astsigma = (
        astsigma > 0.0
        and astsigma_displacement >= astsigma_threshold
        and n_epochs >= min_epochs
    )

    is_trail = is_trail_score or is_trail_astsigma

    # Debug logging
    if is_trail and logger is not None:
        trigger = "score" if is_trail_score else "astsigma"
        logger.debug(
            f"Trail candidate ({trigger}): "
            f"motion_rate={motion_features['motion_rate_as_per_hr']:.2f}\"/hr, "
            f"motion_sig={motion_significance:.1f}, "
            f"astsigma_displacement={astsigma_displacement:.1f}×ASTSIGMA, "
            f"n_epochs={n_epochs}, trail_score={trail_score:.3f}"
        )

    return trail_score, is_trail
