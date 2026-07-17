"""Image differencing for the subtraction detection strategy (Phase B --
see the subtraction-branch plan's B2 and FUTURE_IDEAS.md's "New detection
strategy: image subtraction").

Two engines, config-selectable via DetectionConfig.subtraction_engine:
- "hotpants": stdpipe.subtraction.run_hotpants -- matches how the real
  tests/2026kid/ fixture's diff images were actually produced (confirmed by
  its own FITS header's TEMPLATE keyword and hotpants-specific conventions).
- "zogy": PyZOGY.subtract -- matches the one-off subtract_supernova.py
  reference script's approach (statistically optimal, no free convolution-
  kernel parameters to tune).

Both write a diff FITS carrying the science image's own header plus a
TEMPLATE keyword recording the template's provenance string (from
templates.py) -- the exact convention frontend_generator.py's triplet-
cutout code and candidates.py's find_science_sibling already rely on (see
the subtraction-branch plan's A3), and the one the real 2026kid fixture's
own 'h' files already use.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from astropy.io import fits

from pyrt_transient.detection.subtraction.candidates import _SCIENCE_META_KEYS

logger = logging.getLogger("detection.subtraction.differencing")


def _pixel_scale_arcsec(header) -> Optional[float]:
    try:
        from astropy.wcs import WCS
        from astropy.wcs.utils import proj_plane_pixel_scales
        wcs = WCS(header)
        return float(proj_plane_pixel_scales(wcs)[0]) * 3600.0
    except Exception:
        return None


def _make_gaussian_psf(fwhm_px: float, size: int = 21) -> np.ndarray:
    """Small, centred, sum-normalized Gaussian PSF array -- same
    construction as subtract_supernova.py's make_gaussian_psf, used as a
    PyZOGY PSF proxy when no measured PSF model is available.
    """
    sigma = max(fwhm_px, 0.5) / 2.355
    size = max(int(8 * sigma) | 1, size)
    if size % 2 == 0:
        size += 1
    y, x = np.mgrid[-size // 2:size // 2 + 1, -size // 2:size // 2 + 1]
    psf = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return (psf / psf.sum()).astype(np.float64)


def _load_science_image(science_path):
    science_path = Path(science_path)
    with fits.open(science_path) as hdul:
        image = np.array(hdul[0].data, dtype=np.float64)
        header = hdul[0].header.copy()
    return image, header


def _default_output_path(science_path) -> Path:
    """Same 'h'-suffix-before-extension naming convention as the real
    tests/2026kid/ fixture (`0429r.fits` -> `0429rh.fits`)."""
    science_path = Path(science_path)
    return science_path.with_name(f"{science_path.stem}h.fits")


def _write_diff(diff: np.ndarray, header, output_path: Path, template_provenance: str,
                 science_meta: Optional[dict] = None) -> Path:
    """Write the diff FITS, with the science epoch's own FITS header plus
    a TEMPLATE keyword, plus (critically) FIELD/CTRRA/CTRDEC/MAGLIM/etc. from
    the science ecsv's *table meta* if given.

    These keys live only in the science `.ecsv`'s sidecar meta, not in the
    raw FITS file's own header -- verified directly (tests/2026kid/'s own
    science FITS header has no FIELD/CTRRA/CTRDEC/MAGLIM keys at all, only
    the ecsv's meta does), so copying just the FITS header (as this
    function used to) silently produced a diff image with FIELD=0.0 and
    friends. That in turn made a downstream HyperLEDA query request a
    0-degree search radius, which for reasons on VizieR's end returned
    ~1 million rows and hung the whole pipeline -- a very concrete
    consequence of this gap, not a hypothetical one.
    """
    out_header = header.copy()
    out_header["TEMPLATE"] = template_provenance
    if science_meta:
        for key in _SCIENCE_META_KEYS:
            if key in science_meta and science_meta[key] is not None:
                try:
                    out_header[key] = science_meta[key]
                except Exception:
                    out_header[key] = str(science_meta[key])
    fits.writeto(str(output_path), diff.astype(np.float32), out_header, overwrite=True)
    return output_path


def run_hotpants_diff(
    science_path,
    template_array: np.ndarray,
    template_mask: Optional[np.ndarray] = None,
    output_path=None,
    template_provenance: str = "unknown",
    science_meta: Optional[dict] = None,
) -> Optional[Path]:
    """Difference `science_path`'s image against `template_array` using
    HOTPANTS (stdpipe.subtraction.run_hotpants).

    Returns the diff FITS path, or None if stdpipe/HOTPANTS aren't
    available, the template shape doesn't match, or the subtraction fails
    -- callers should treat this the same as a missing template (skip this
    epoch, don't crash the whole run).
    """
    try:
        image, header = _load_science_image(science_path)
    except Exception as e:
        logger.warning(f"HOTPANTS: could not read science image {science_path}: {e}")
        return None

    if template_array.shape != image.shape:
        logger.warning(f"HOTPANTS: template shape {template_array.shape} != "
                        f"science shape {image.shape} -- skipping")
        return None

    try:
        from stdpipe import subtraction as stdpipe_subtraction
    except ImportError:
        logger.warning("HOTPANTS: stdpipe not available")
        return None

    mask = ~np.isfinite(image)
    template_mask_arr = (~np.isfinite(template_array) if template_mask is None
                          else template_mask.astype(bool) | ~np.isfinite(template_array))

    try:
        diff = stdpipe_subtraction.run_hotpants(
            image, template_array,
            mask=mask, template_mask=template_mask_arr,
            image_fwhm=header.get("FWHM"),
            verbose=False,
        )
    except Exception as e:
        logger.warning(f"HOTPANTS: run_hotpants raised {type(e).__name__}: {e}")
        return None

    if diff is None:
        logger.warning(f"HOTPANTS: subtraction failed for {science_path} "
                        f"(missing `hotpants` binary or bad inputs)")
        return None

    output_path = Path(output_path) if output_path else _default_output_path(science_path)
    _write_diff(diff, header, output_path, template_provenance, science_meta=science_meta)
    logger.info(f"HOTPANTS: wrote diff image {output_path}")
    return output_path


def run_zogy_diff(
    science_path,
    template_array: np.ndarray,
    template_mask: Optional[np.ndarray] = None,
    template_fwhm_arcsec: float = 1.5,
    output_path=None,
    template_provenance: str = "unknown",
    science_meta: Optional[dict] = None,
) -> Optional[Path]:
    """Difference `science_path`'s image against `template_array` using
    PyZOGY (Zackay, Ofek & Gal-Yam 2016), adapted directly from the one-off
    subtract_supernova.py reference script's steps 6-7: background
    pre-subtraction (so ZOGY's gain ratio carries no DC offset), Gaussian
    PSF proxies scaled to each image's own FWHM, normalize-to-science-flux
    -scale. Same output convention as run_hotpants_diff.
    """
    try:
        image, header = _load_science_image(science_path)
    except Exception as e:
        logger.warning(f"ZOGY: could not read science image {science_path}: {e}")
        return None

    if template_array.shape != image.shape:
        logger.warning(f"ZOGY: template shape {template_array.shape} != "
                        f"science shape {image.shape} -- skipping")
        return None

    try:
        from PyZOGY.subtract import (
            calculate_difference_image,
            calculate_difference_image_zero_point,
            normalize_difference_image,
        )
        from PyZOGY.image_class import ImageClass
        from stdpipe import photometry as stdpipe_photometry
    except ImportError as e:
        logger.warning(f"ZOGY: dependency not available ({e})")
        return None

    mask = ~np.isfinite(image)
    template_mask_arr = (~np.isfinite(template_array) if template_mask is None
                          else template_mask.astype(bool) | ~np.isfinite(template_array))

    fwhm_px = float(header.get("FWHM", 3.0))
    pixscale_arcsec = _pixel_scale_arcsec(header)
    template_fwhm_px = (template_fwhm_arcsec / pixscale_arcsec if pixscale_arcsec
                         else fwhm_px)

    try:
        bg_sci = stdpipe_photometry.get_background(image, mask=mask, size=128)
        bg_tmpl = stdpipe_photometry.get_background(template_array, mask=template_mask_arr, size=128)
        image_bgsub = image - bg_sci
        tmpl_bgsub = template_array - bg_tmpl

        psf_sci = _make_gaussian_psf(fwhm_px)
        psf_ref = _make_gaussian_psf(template_fwhm_px)

        sci_obj = ImageClass(image_bgsub.astype(np.float64), psf_sci, mask.astype(bool))
        ref_obj = ImageClass(tmpl_bgsub.astype(np.float64), psf_ref, template_mask_arr.astype(bool))

        D = calculate_difference_image(sci_obj, ref_obj)
        F_D = np.asarray(calculate_difference_image_zero_point(sci_obj, ref_obj), dtype=np.float64)
        diff = normalize_difference_image(D, F_D, sci_obj, ref_obj, normalization="science")
        diff = np.ascontiguousarray(np.real(diff).astype(np.float64))
    except Exception as e:
        logger.warning(f"ZOGY: subtraction failed for {science_path}: {type(e).__name__}: {e}")
        return None

    output_path = Path(output_path) if output_path else _default_output_path(science_path)
    _write_diff(diff, header, output_path, template_provenance, science_meta=science_meta)
    logger.info(f"ZOGY: wrote diff image {output_path}")
    return output_path


def run_diff(
    engine: str,
    science_path,
    template_array: np.ndarray,
    template_mask: Optional[np.ndarray] = None,
    template_fwhm_arcsec: float = 1.5,
    output_path=None,
    template_provenance: str = "unknown",
    science_meta: Optional[dict] = None,
) -> Optional[Path]:
    """Dispatch to run_hotpants_diff/run_zogy_diff by
    DetectionConfig.subtraction_engine's value.

    science_meta: the science epoch's own ecsv table.meta (FIELD/CTRRA/
    CTRDEC/MAGLIM/etc.) -- these live only there, not in the raw science
    FITS header, and the diff FITS needs them so extraction.py's downstream
    schema adapter (and everything after it) has real values instead of
    falling back to 0.0/absent (see _write_diff's docstring for the
    concrete bug this caused when omitted).
    """
    if engine == "hotpants":
        return run_hotpants_diff(
            science_path, template_array, template_mask=template_mask,
            output_path=output_path, template_provenance=template_provenance,
            science_meta=science_meta,
        )
    if engine == "zogy":
        return run_zogy_diff(
            science_path, template_array, template_mask=template_mask,
            template_fwhm_arcsec=template_fwhm_arcsec,
            output_path=output_path, template_provenance=template_provenance,
            science_meta=science_meta,
        )
    raise ValueError(f"Unknown subtraction_engine: {engine!r} (expected 'hotpants' or 'zogy')")
