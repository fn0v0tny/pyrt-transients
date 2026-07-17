"""Subtraction-specific candidate filters (see the subtraction-branch plan's
A1). `apply_morphology_filter`/`apply_magnitude_filter` are ported unchanged
from `pipeline_magic_sn.py` (they operate on generic ELLIPTICITY/fwhm_ratio/
MAG_CALIB columns, equally valid for subtraction candidates). The new,
subtraction-specific filter is `reject_dipole_artifacts`.

Real diff-image detection catalogs run at S/N thresholds low enough to admit
mostly noise: the tests/2026kid/ fixture has ~675 raw diff detections in a
single epoch for what should be ~1 real object. A positive+negative flux
pair close together is the standard signature of imperfect subtraction (bad
registration, saturated-star wings, cosmic rays) rather than a real
transient, which subtracts to a clean, single-signed residual. The catalog
itself never lists negative-flux detections (single-polarity SExtractor --
verified against the real fixture: FLUX is always positive there), so this
filter samples the diff FITS pixel data directly around each candidate's own
position rather than relying on a second catalog entry.
"""

import logging
from typing import Optional, Tuple

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales


def apply_morphology_filter(candidates, logger, max_ellipticity=0.4,
                             fwhm_ratio_range=(0.5, 2.0)):
    """Reject non-point-like sources.

    Real transients look like PSF-matched point sources. Elongated objects
    (ELLIPTICITY >= max_ellipticity) are likely cosmic rays, satellite
    trails or diffraction spikes. Sources whose FWHM deviates strongly from
    the image PSF (fwhm_ratio outside fwhm_ratio_range) are blended pairs,
    extended nuclei, or -- for diff images specifically -- convolution
    kernel-mismatch artifacts near bright stars.
    """
    if len(candidates) == 0:
        return candidates

    mask = np.ones(len(candidates), dtype=bool)

    if 'ELLIPTICITY' in candidates.colnames:
        ell = np.array(candidates['ELLIPTICITY'], dtype=float)
        ell_ok = ell < max_ellipticity
        n_removed = int(np.sum(~ell_ok))
        if n_removed:
            logger.info(f"  Morphology: removed {n_removed} elongated sources "
                        f"(ELLIPTICITY >= {max_ellipticity})")
        mask &= ell_ok

    if 'fwhm_ratio' in candidates.colnames:
        fr = np.array(candidates['fwhm_ratio'], dtype=float)
        fr_ok = (fr >= fwhm_ratio_range[0]) & (fr <= fwhm_ratio_range[1])
        n_removed = int(np.sum(~fr_ok))
        if n_removed:
            logger.info(f"  Morphology: removed {n_removed} sources with "
                        f"fwhm_ratio outside {fwhm_ratio_range}")
        mask &= fr_ok

    return candidates[mask]


def apply_magnitude_filter(candidates, logger, bright_limit=14.0, faint_margin=0.0):
    """Remove sources outside a realistic transient brightness range.

    bright_limit: sources brighter than this are saturated stars or
        artifacts near them, not genuine faint transients.
    faint_margin: if > 0, also remove sources fainter than MAGLIM -
        faint_margin (noise spikes close to the detection limit).
    """
    if len(candidates) == 0 or 'MAG_CALIB' not in candidates.colnames:
        return candidates

    mag = np.array(candidates['MAG_CALIB'], dtype=float)
    mask = mag > bright_limit
    n_bright = int(np.sum(~mask))
    if n_bright:
        logger.info(f"  Magnitude: removed {n_bright} sources brighter than "
                    f"{bright_limit} mag")

    if faint_margin > 0:
        maglim = None
        for key in ('MAGLIM', 'MAGLIMIT', 'maglim', 'maglimit'):
            if key in candidates.meta:
                maglim = float(candidates.meta[key])
                break
        if maglim is not None:
            faint_mask = mag <= (maglim - faint_margin)
            n_faint = int(np.sum(~faint_mask))
            if n_faint:
                logger.info(f"  Magnitude: removed {n_faint} sources fainter than "
                            f"MAGLIM-{faint_margin} = {maglim - faint_margin:.1f}")
            mask &= faint_mask

    return candidates[mask]


def reject_dipole_artifacts(
    candidates: Table,
    diff_fits_path,
    radius_arcsec: float = 3.0,
    flux_ratio_thresh: float = 0.5,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Table, int]:
    """Reject candidates with a significant negative-flux pixel close to
    their own position -- imperfect subtraction leaves a positive/negative
    "dipole" (bad registration, saturated-star wings, cosmic rays); a real
    transient subtracts cleanly and has no comparable negative counterpart
    nearby.

    For each candidate, crops a `radius_arcsec`-radius box (converted to
    pixels via the diff FITS's own WCS) around its X_IMAGE/Y_IMAGE and
    compares the most negative pixel value there to the candidate's own
    FLUX. Rejects if |most_negative| / own_flux >= flux_ratio_thresh --
    i.e. a nearby negative feature of comparable brightness, not a
    coincidental faint dip.

    Returns (filtered_candidates, n_rejected).
    """
    if len(candidates) == 0:
        return candidates, 0
    if "X_IMAGE" not in candidates.colnames or "Y_IMAGE" not in candidates.colnames:
        return candidates, 0

    try:
        with fits.open(diff_fits_path) as hdul:
            data = hdul[0].data.astype(float)
            header = hdul[0].header
        wcs = WCS(header)
        pixscale_deg = float(proj_plane_pixel_scales(wcs)[0])
        pixscale_arcsec = pixscale_deg * 3600.0
    except Exception as e:
        if logger:
            logger.warning(f"  Dipole filter: could not open {diff_fits_path} ({e}) -- skipping")
        return candidates, 0

    radius_px = max(int(round(radius_arcsec / pixscale_arcsec)), 1)
    ny, nx = data.shape

    x = np.asarray(candidates["X_IMAGE"], dtype=float)
    y = np.asarray(candidates["Y_IMAGE"], dtype=float)
    flux = (np.asarray(candidates["FLUX"], dtype=float)
            if "FLUX" in candidates.colnames else None)

    keep = np.ones(len(candidates), dtype=bool)
    for i in range(len(candidates)):
        # SExtractor X_IMAGE/Y_IMAGE are 1-indexed (FITS convention).
        cx, cy = int(round(x[i])) - 1, int(round(y[i])) - 1
        x0, x1 = max(0, cx - radius_px), min(nx, cx + radius_px + 1)
        y0, y1 = max(0, cy - radius_px), min(ny, cy + radius_px + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        box = data[y0:y1, x0:x1]
        if not np.any(np.isfinite(box)):
            continue
        min_val = float(np.nanmin(box))
        if min_val >= 0:
            continue
        own_flux = flux[i] if flux is not None else float(np.nanmax(box))
        if own_flux <= 0:
            continue
        if abs(min_val) / own_flux >= flux_ratio_thresh:
            keep[i] = False

    n_rejected = int(np.sum(~keep))
    if logger and n_rejected:
        logger.info(
            f"  Dipole filter: removed {n_rejected}/{len(candidates)} candidates "
            f"(negative counterpart within {radius_arcsec}\", flux ratio >= {flux_ratio_thresh})"
        )
    return candidates[keep], n_rejected
