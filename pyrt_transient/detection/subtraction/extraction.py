"""Diff-image source extraction and photometric calibration for the
subtraction detection strategy (Phase B -- see the subtraction-branch
plan's B3).

Per the user's decision: uses stdpipe's own SEP-based extraction
(photometry.get_objects_sep) and photometric calibration
(pipeline.calibrate_photometry against a reference survey catalog), not an
external dophot3/phcat CLI -- directly generalizing the one-off
subtract_supernova.py reference script's approach into a reusable function.

Important correctness note found while implementing this (not obvious from
the plan): you cannot calibrate photometry by matching *diff-image*
detections against a star catalog -- by definition, real catalog stars are
constant sources that subtract away, so a diff image's detections are
almost entirely transients/variables/subtraction artifacts, not the same
population as the catalog. Verified directly: matching 33 real diff-image
detections against 1869 PS1 stars found only 3 candidate matches, 1 after
quality cuts -- nowhere near enough for a robust zeropoint fit.
subtract_supernova.py's actual (validated) approach, followed here, is to
calibrate the zeropoint from the *science* image's own detections (where
real catalog stars are actually present), then apply that zeropoint
directly to the diff image's flux measurements -- exactly the same
"borrow the science epoch's zeropoint" principle candidates.py's
calibrate_diff_magnitudes already uses for the real tests/2026kid/ fixture
(which never has its own calibration at all), just derived here instead of
borrowed from an externally-produced ecsv.

The output is adapted into the exact ECSV column/meta schema
transients.open_ecsv_file / candidates.build_epoch_candidates expect
(ALPHA_J2000, DELTA_J2000, MAG_CALIB, MAGERR_CALIB, X_IMAGE, Y_IMAGE,
FWHM_IMAGE, ELLIPTICITY, FLAGS; meta CTRRA/CTRDEC/FIELD/MAGLIM/FWHM/OBSID/
CTIME/EXPTIME), so this module's output plugs into SubtractionStrategy
completely unchanged -- Phase A and Phase B share the exact same downstream
code, only the *source* of the diff ecsv differs.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

logger = logging.getLogger("detection.subtraction.extraction")


def _load_image(path):
    with fits.open(Path(path)) as hdul:
        image = np.array(hdul[0].data, dtype=np.float64)
        header = hdul[0].header.copy()
    return image, header


def calibrate_science_zeropoint(
    science_fits_path,
    photometric_catalog: str = "ps1",
    detect_thresh: float = 5.0,
    aper_px: Optional[float] = None,
):
    """Detect sources in the science image and calibrate their photometry
    against a reference survey catalog -- real catalog stars are actually
    present here, unlike in the diff image (see this module's docstring).

    Returns (zp_value, zp_err) or (nan, nan) if calibration failed.
    """
    try:
        from stdpipe import photometry as stdpipe_photometry
        from stdpipe import pipeline as stdpipe_pipeline
        from stdpipe import catalogs as stdpipe_catalogs
    except ImportError as e:
        logger.warning(f"stdpipe not available ({e})")
        return np.nan, np.nan

    try:
        image, header = _load_image(science_fits_path)
        wcs = WCS(header)
        pixscale_deg = float(proj_plane_pixel_scales(wcs)[0])
    except Exception as e:
        logger.warning(f"{science_fits_path}: could not read image/WCS: {e}")
        return np.nan, np.nan

    fwhm_px = float(header.get("FWHM", 3.0))
    aper_px = aper_px or fwhm_px
    mask = ~np.isfinite(image)

    try:
        obj = stdpipe_photometry.get_objects_sep(
            image, mask=mask, header=header, wcs=wcs,
            thresh=detect_thresh, aper=aper_px, verbose=False,
        )
    except Exception as e:
        logger.warning(f"{science_fits_path}: SEP detection failed: {e}")
        return np.nan, np.nan

    if obj is None or len(obj) == 0:
        logger.warning(f"{science_fits_path}: no sources detected for calibration")
        return np.nan, np.nan

    ra0 = float(header.get("CTRRA", header.get("CRVAL1", 0.0)))
    dec0 = float(header.get("CTRDEC", header.get("CRVAL2", 0.0)))
    field_deg = float(header.get("FIELD", max(image.shape) * pixscale_deg * 1.1))
    sr0 = field_deg / 2.0 * 1.1

    cat_mag_col, cat_mag_err_col = {
        "ps1": ("rmag", "e_rmag"),
        "gaiaedr3": ("Gmag", "e_Gmag"),
    }.get(photometric_catalog, ("rmag", "e_rmag"))

    try:
        cat = stdpipe_catalogs.get_cat_vizier(ra0, dec0, sr0, catalog=photometric_catalog, verbose=False)
        if cat is None or len(cat) == 0:
            logger.warning(f"{science_fits_path}: no {photometric_catalog} catalog stars in field")
            return np.nan, np.nan

        zp_result = stdpipe_pipeline.calibrate_photometry(
            obj, cat, pixscale=pixscale_deg, order=0,
            obj_col_mag="mag", obj_col_mag_err="magerr",
            cat_col_mag=cat_mag_col, cat_col_mag_err=cat_mag_err_col,
            verbose=False,
        )
    except Exception as e:
        logger.warning(f"{science_fits_path}: photometric calibration failed: {e}")
        return np.nan, np.nan

    if not zp_result or "params" not in zp_result:
        logger.warning(f"{science_fits_path}: calibration did not converge "
                        f"(too few matched reference stars)")
        return np.nan, np.nan

    zp_value = float(zp_result["params"][0])
    zp_err = np.nan
    idx = zp_result.get("idx")
    if idx is not None and np.any(idx):
        good_zeros = np.asarray(zp_result["zero"])[np.asarray(idx, dtype=bool)]
        n_good = int(np.sum(idx))
        if n_good > 1:
            zp_err = float(np.nanstd(good_zeros) / np.sqrt(n_good - 1))
    return zp_value, zp_err


def build_diff_ecsv(
    diff_fits_path,
    science_fits_path,
    output_path=None,
    photometric_catalog: str = "ps1",
    detect_thresh: float = 4.0,
    aper_px: Optional[float] = None,
) -> Optional[Path]:
    """Detect sources in a diff-image FITS and calibrate their photometry
    using the matching science image's own zeropoint (see this module's
    docstring for why calibrating directly against the diff image doesn't
    work), writing an ECSV in the schema build_epoch_candidates expects.

    Returns the ECSV path, or None if detection/calibration failed --
    callers should treat this the same as any other unusable epoch.
    """
    diff_fits_path = Path(diff_fits_path)

    zp_value, zp_err = calibrate_science_zeropoint(
        science_fits_path, photometric_catalog=photometric_catalog,
    )
    if not np.isfinite(zp_value):
        logger.warning(f"{diff_fits_path}: no science-epoch zeropoint available, skipping")
        return None

    try:
        from stdpipe import photometry as stdpipe_photometry
    except ImportError as e:
        logger.warning(f"stdpipe not available ({e})")
        return None

    try:
        image, header = _load_image(diff_fits_path)
        wcs = WCS(header)
        pixscale_deg = float(proj_plane_pixel_scales(wcs)[0])
    except Exception as e:
        logger.warning(f"{diff_fits_path}: could not read image/WCS: {e}")
        return None

    fwhm_px = float(header.get("FWHM", 3.0))
    aper_px = aper_px or fwhm_px
    mask = ~np.isfinite(image)

    try:
        obj = stdpipe_photometry.get_objects_sep(
            image, mask=mask, header=header, wcs=wcs,
            thresh=detect_thresh, aper=aper_px, verbose=False,
        )
    except Exception as e:
        logger.warning(f"{diff_fits_path}: SEP detection failed: {e}")
        return None

    if obj is None or len(obj) == 0:
        logger.info(f"{diff_fits_path}: no sources detected above {detect_thresh} sigma")
        return None

    ecsv_table = _adapt_to_ecsv_schema(obj, header, zp_value, zp_err, pixscale_deg, fwhm_px)

    output_path = Path(output_path) if output_path else diff_fits_path.with_suffix(".ecsv")
    ecsv_table.write(str(output_path), format="ascii.ecsv", overwrite=True)
    logger.info(f"{diff_fits_path}: wrote {len(ecsv_table)} calibrated detections "
                f"(science zp={zp_value:.3f}) -> {output_path}")
    return output_path


def _adapt_to_ecsv_schema(obj: Table, header, zp_value: float, zp_err: float,
                           pixscale_deg: float, fwhm_px: float) -> Table:
    """Map stdpipe's SEP object table + a science-derived zeropoint into
    the column/meta schema build_epoch_candidates (and downstream
    matching/scoring) expect.
    """
    n = len(obj)
    ellipticity = 1.0 - np.clip(np.array(obj["b"], dtype=float) / np.maximum(obj["a"], 1e-6), 0.0, 1.0)

    flux = np.array(obj["flux"], dtype=float)
    fluxerr = np.array(obj["fluxerr"], dtype=float) if "fluxerr" in obj.colnames else np.full(n, np.nan)
    good_flux = flux > 0
    mag_calib = np.where(good_flux, -2.5 * np.log10(np.where(good_flux, flux, 1.0)) + zp_value, np.nan)
    # Standard mag-error <-> S/N relation (same one used throughout this
    # codebase, e.g. DetectionConfig.siglim's docstring): sigma_mag ~= 1.0857/SNR,
    # combined in quadrature with the zeropoint's own uncertainty.
    snr = np.where((fluxerr > 0) & good_flux, flux / np.maximum(fluxerr, 1e-10), np.nan)
    sigma_phot = 1.0857 / np.maximum(snr, 0.01)
    magerr_calib = np.sqrt(sigma_phot ** 2 + (zp_err if np.isfinite(zp_err) else 0.0) ** 2)

    out = Table({
        "NUMBER": np.arange(1, n + 1),
        "ALPHA_J2000": np.array(obj["ra"], dtype=float),
        "DELTA_J2000": np.array(obj["dec"], dtype=float),
        # SEP/stdpipe x/y are 0-indexed; the rest of this codebase expects
        # SExtractor's 1-indexed convention (see candidates.py/
        # frontend_generator.py's "-1 to convert to 0-based" comments).
        "X_IMAGE": np.array(obj["x"], dtype=float) + 1.0,
        "Y_IMAGE": np.array(obj["y"], dtype=float) + 1.0,
        "FWHM_IMAGE": np.array(obj["fwhm"], dtype=float) if "fwhm" in obj.colnames
                      else np.full(n, fwhm_px),
        "ELLIPTICITY": ellipticity,
        "FLAGS": np.array(obj["flags"], dtype=int) if "flags" in obj.colnames else np.zeros(n, dtype=int),
        "FLUX": flux,
        "MAG_CALIB": mag_calib,
        "MAGERR_CALIB": magerr_calib,
    })

    maglim = float(header.get("MAGLIM", zp_value - 2.5 * np.log10(5.0)))
    out.meta.update({
        "MAGZERO": zp_value,
        "DMAGZERO": zp_err if np.isfinite(zp_err) else 0.0,
        "MAGLIM": maglim,
        "FWHM": fwhm_px,
        "CTRRA": float(header.get("CTRRA", header.get("CRVAL1", 0.0))),
        "CTRDEC": float(header.get("CTRDEC", header.get("CRVAL2", 0.0))),
        "FIELD": float(header.get("FIELD", 0.0)),
        "PIXEL": pixscale_deg * 3600.0,
    })
    for key in ("OBSID", "CTIME", "EXPTIME", "JD", "MJD-OBS", "TEMPLATE", "OBJECT", "TARGET"):
        if key in header:
            out.meta[key] = header[key]

    return out
