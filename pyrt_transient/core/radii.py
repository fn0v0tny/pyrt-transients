"""Per-detection adaptive matching radius.

Merges transient_analyser.py:compute_per_detection_radius (sky-space) and
catalog.py:_compute_adaptive_radii (pixel-space). These were NOT the same
formula wearing different units -- confirmed with the user before merging:

  - catalog.py's version PREFERS the empirically-measured SExtractor
    centroid position error (ERRX2_IMAGE/ERRY2_IMAGE), scaled by sqrt(ASTVAR)
    when available, only falling back to a PSF+SNR heuristic when those
    columns are missing/invalid.
  - transient_analyser.py's version never looked at ERRX2/ERRY2 or ASTVAR at
    all -- it always used a PSF(FWHM/A_IMAGE)+SNR heuristic
    (nsigma * psf_sigma_px * sqrt(10/snr)), then converted px -> arcsec via
    WCS plate scale.

Per explicit user decision, this merge UNIFIES the primary path: both
coord_system="pixel" and coord_system="sky" now prefer
ERRX2_IMAGE/ERRY2_IMAGE + ASTVAR when available. This is a deliberate
behavior change for sky-space callers wherever those columns are present --
check_baseline.py is expected to show a diff here, not a silent pass; that
diff needs manual review against the fixture, not to be treated as a
regression.

The PSF+SNR *fallback* (only reached when ERRX2/ERRY2 are missing or
non-finite) was NOT unified -- the two original fallback formulas genuinely
differ (SNR scaling law, floors, FWHM defaults) and the user's decision only
addressed the ERRX2/ASTVAR primary path. Each coord_system keeps its own
original fallback formula and its own original total-failure return
contract (empty array for "pixel", constant-filled array for "sky"), to
avoid bundling in an extra, unreviewed behavior change alongside the
ERRX2/ASTVAR unification above.
"""

import logging
import warnings

import numpy as np


def compute_adaptive_radius(
    detections,
    coord_system="pixel",
    nsigma=3.0,
    idlimit_min_px=1.0,
    idlimit_max_px=8.0,
    use_astvar=True,
    default_plate_scale_arcsec_per_px=0.33,
):
    """Compute per-detection adaptive matching radius.

    Parameters
    ----------
    detections : astropy.table.Table
    coord_system : "pixel" or "sky"
        "pixel" returns radii in pixels (catalog.py's original contract).
        "sky" converts to arcsec via WCS plate scale, or
        default_plate_scale_arcsec_per_px if no WCS info is present
        (transient_analyser.py's original contract).
    nsigma, idlimit_min_px, idlimit_max_px, use_astvar : see module docstring
    default_plate_scale_arcsec_per_px : only used for coord_system="sky"
        when no CD-matrix/CDELT1 is found in detections.meta.

    Returns
    -------
    np.ndarray
        Empty array on total failure for coord_system="pixel" (old
        catalog.py contract -- callers check `len(r) == 0` to disable
        adaptive radii). Full-length array of a constant 2.0 (arcsec) on
        total failure for coord_system="sky" (old transient_analyser.py
        contract -- callers use the array unconditionally).
    """
    if coord_system not in ("pixel", "sky"):
        raise ValueError(f"coord_system must be 'pixel' or 'sky', got {coord_system!r}")

    n_det = len(detections)
    if n_det == 0:
        return np.array([])

    plate_scale = _plate_scale_arcsec_per_px(detections, default_plate_scale_arcsec_per_px)

    try:
        radii_px = _primary_errxy_radius(detections, nsigma, use_astvar)
        if radii_px is None:
            radii_px = (
                _pixel_fallback_radius(detections, nsigma, use_astvar)
                if coord_system == "pixel"
                else _sky_fallback_radius(detections, nsigma)
            )
        if radii_px is None:
            raise ValueError("no usable radius data (ERRX2/ERRY2, SNR, or FWHM all unavailable)")

        radii_px = np.clip(radii_px, idlimit_min_px, idlimit_max_px)
        return radii_px if coord_system == "pixel" else radii_px * plate_scale

    except Exception as exc:
        logging.warning(f"Adaptive radius computation error: {exc}, using defaults")
        return np.array([]) if coord_system == "pixel" else np.full(n_det, 2.0)


def _primary_errxy_radius(detections, nsigma, use_astvar):
    """nsigma * sqrt(ERRX2_IMAGE + ERRY2_IMAGE), optionally x sqrt(ASTVAR).

    Returns None if the columns are missing or all non-finite/negative, so
    callers can fall back to the PSF+SNR heuristic.
    """
    if "ERRX2_IMAGE" not in detections.colnames or "ERRY2_IMAGE" not in detections.colnames:
        return None

    ex2 = detections["ERRX2_IMAGE"].data
    ey2 = detections["ERRY2_IMAGE"].data
    ok = np.isfinite(ex2) & np.isfinite(ey2) & (ex2 >= 0) & (ey2 >= 0)
    if not np.any(ok):
        return None

    pos_err = np.sqrt(ex2 + ey2)
    if use_astvar:
        av = float(detections.meta.get("ASTVAR", 1.0))
        pos_err = pos_err * np.sqrt(av if (np.isfinite(av) and av > 0) else 1.0)

    radii = np.where(ok & np.isfinite(nsigma * pos_err), nsigma * pos_err, np.nan)
    return radii if np.sum(np.isfinite(radii)) > 0 else None


def _pixel_fallback_radius(detections, nsigma, use_astvar):
    """catalog.py's original SNR-based fallback, verbatim: pos_err = (fwhm/2.35)/snr."""
    snr = None
    if "SNR" in detections.colnames:
        snr = detections["SNR"].data
    elif "FLUX_ISO" in detections.colnames and "FLUXERR_ISO" in detections.colnames:
        f = detections["FLUX_ISO"].data
        fe = detections["FLUXERR_ISO"].data
        snr = np.where((fe > 0) & np.isfinite(f) & np.isfinite(fe), f / fe, np.nan)
    if snr is None:
        return None

    ok = np.isfinite(snr) & (snr > 0)
    if not np.any(ok):
        return None

    fwhm = (
        detections["FWHM_IMAGE"].data
        if "FWHM_IMAGE" in detections.colnames
        else np.full(len(detections), float(detections.meta.get("FWHM", 1.2)))
    )
    fwhm = np.where(np.isfinite(fwhm) & (fwhm > 0), fwhm, 1.2)

    pos_err = (fwhm / 2.35) / np.maximum(snr, 1e-6)
    if use_astvar:
        av = float(detections.meta.get("ASTVAR", 1.0))
        pos_err = pos_err * np.sqrt(av if (np.isfinite(av) and av > 0) else 1.0)

    radii = np.where(ok & np.isfinite(nsigma * pos_err), nsigma * pos_err, np.nan)
    return radii if np.sum(np.isfinite(radii)) > 0 else None


def _sky_fallback_radius(detections, nsigma):
    """transient_analyser.py's original PSF+SNR heuristic, verbatim:
    radii_px = nsigma * psf_sigma_px * sqrt(10/snr).
    """
    n_det = len(detections)

    if "FWHM_IMAGE" in detections.colnames:
        psf_sigma_px = detections["FWHM_IMAGE"] / 2.35
    elif "A_IMAGE" in detections.colnames:
        psf_sigma_px = detections["A_IMAGE"] / 2.0
    else:
        psf_sigma_px = np.full(n_det, 2.0)

    snr = np.full(n_det, 5.0)
    if "SNR" in detections.colnames:
        snr = np.maximum(detections["SNR"], 3.0)
    elif "FLUX_AUTO" in detections.colnames and "FLUXERR_AUTO" in detections.colnames:
        flux = detections["FLUX_AUTO"]
        flux_err = detections["FLUXERR_AUTO"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            snr = np.maximum(flux / np.maximum(flux_err, 1e-10), 3.0)

    snr_scale = np.sqrt(10.0 / snr)
    return nsigma * psf_sigma_px * snr_scale


def _plate_scale_arcsec_per_px(detections, default_plate_scale_arcsec_per_px):
    """CD-matrix or CDELT1 plate scale from detections.meta, else the default."""
    if not (hasattr(detections, "meta") and detections.meta):
        return default_plate_scale_arcsec_per_px

    cd11 = detections.meta.get("CD1_1", 0)
    cd22 = detections.meta.get("CD2_2", 0)
    if cd11 != 0 and cd22 != 0:
        return 3600 * np.sqrt(abs(cd11 * cd22))

    cdelt1 = detections.meta.get("CDELT1", 0)
    if cdelt1 != 0:
        return 3600 * abs(cdelt1)

    return default_plate_scale_arcsec_per_px
