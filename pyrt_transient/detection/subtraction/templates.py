"""Template acquisition for the subtraction detection strategy (Phase B --
see the subtraction-branch plan's B1 and FUTURE_IDEAS.md's "New detection
strategy: image subtraction").

Two strategies, config-selectable via DetectionConfig.template_source:
- "own_epoch": reuse detection/reference_frame.py's ReferenceFrameSelector
  to pick this campaign's own best-quality prior epoch as the template --
  zero external dependency, but needs enough prior epochs to have a clean
  one, and only makes sense once a field has been observed a few times.
- "ps1"/"legacysurvey": stdpipe.templates.get_ps1_image_and_mask /
  get_ls_image_and_mask, reprojected onto the science epoch's own WCS via
  SWarp -- works on a field's very first observation, but depends on survey
  coverage, network access, and the `swarp` binary being installed (it
  isn't on this dev machine -- get_template_ps1 degrades to returning None
  rather than raising, exactly like stdpipe's own reproject_swarp does when
  it can't find the binary).

Template caching (DetectionConfig.template_cache_dir/
template_cache_max_size_gb): a reprojected survey template is expensive to
build (skycell download + SWarp) and reusable across many nights of the
same field/band -- keyed by rounded field center + band. Expired via the
same LRU disk-budget pattern frontend_generator.py already uses for the
website directory (cleanup_old_files/enforce_disk_budget_strict), rather
than inventing a second cleanup mechanism.
"""

import inspect
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from pyrt_transient.detection.reference_frame import ReferenceFrameSelector

logger = logging.getLogger("detection.subtraction.templates")

_ps1_patch_applied = False


def _patch_normalize_ps1_skycell() -> None:
    """stdpipe.templates.normalize_ps1_skycell (as installed here --
    site-packages/stdpipe, an older checkout) crashes on real PS1 skycell
    downloads with `ValueError: cannot convert float NaN to integer`,
    inside astropy's own compressed-tile decompression of certain
    BLANK-valued integer skycell masks. Verified directly against a real
    download in this environment, not hypothetical.

    This is the exact same bug the one-off subtract_supernova.py reference
    script (lascaux50:~/sn2026kie/) already found and worked around, by
    reading the skycell with `fitsio` (a different FITS reader, unaffected)
    instead of astropy for this one normalization step. Reapplied here
    verbatim rather than re-deriving it, since it's a proven fix for a
    proven bug -- only applied once per process, and only if `fitsio` is
    importable (falls back to stdpipe's own, occasionally-crashing version
    otherwise, since a monkeypatch attempt without fitsio would just be a
    different way of doing nothing).

    The "older checkout" part of that is load-bearing: this patch was
    written against a `normalize_ps1_skycell(filename, outname=None,
    verbose=False)` that reopens a file from disk. A newer stdpipe (verified
    against 0.4.1 from PyPI) changed the signature to
    `normalize_ps1_skycell(image, header, verbose=False)` -- pure in-memory
    array/header normalization, called internally with the already-decoded
    skycell data, no reopening involved. Applying this patch unconditionally
    against that newer signature doesn't restore the old bug -- it breaks
    every PS1 call outright, binding the in-memory `image` array to this
    function's `filename` parameter and crashing when astropy tries to treat
    a numpy array as a path. Directly verified against 0.4.1 on a real
    field: with this patch *not* applied, stdpipe's own current
    normalize_ps1_skycell handles real skycells (including BLANK-valued
    masks) without the original decompression error, so the bug this patch
    targets is apparently already fixed upstream in 0.4.1. So: guard on the
    installed function's parameter names before patching, and fall back to
    stdpipe's own (here, working) implementation if they don't match the
    old `filename`-based form this patch targets.
    """
    global _ps1_patch_applied
    if _ps1_patch_applied:
        return
    try:
        import fitsio
        import stdpipe.templates as _stdpipe_templates
    except ImportError:
        # No fitsio: the replacement below cannot run (it reads the skycell
        # with fitsio), so stdpipe's own implementation stays. That decision
        # is final for this process -- set the flag, or every PS1 call
        # re-attempts the import and re-inspects the signature.
        _ps1_patch_applied = True
        logger.debug("PS1 template: fitsio not available, using stdpipe's own "
                      "normalize_ps1_skycell (may fail on some real skycells)")
        return

    # Missing symbol / unintrospectable callable: same class of problem as the
    # ImportError above, and this function is called from outside the caller's
    # own try (get_template), so anything raised here kills the detection run
    # rather than degrading to "no template".
    original = getattr(_stdpipe_templates, "normalize_ps1_skycell", None)
    try:
        original_params = list(inspect.signature(original).parameters)
    except (TypeError, ValueError):
        _ps1_patch_applied = True
        logger.debug("PS1 template: this stdpipe build has no introspectable "
                     "normalize_ps1_skycell, leaving it alone")
        return

    if original_params[:1] != ["filename"]:
        _ps1_patch_applied = True
        logger.debug(
            "PS1 template: installed stdpipe's normalize_ps1_skycell has a "
            f"different signature ({original_params}) than the old "
            "filename-based one this patch targets -- assuming a newer "
            "stdpipe that already fixed the bug, using its own implementation"
        )
        return

    def _normalize_ps1_skycell_fixed(filename, outname=None, verbose=False):
        log = (verbose if callable(verbose) else print) if verbose else lambda *a, **k: None
        hdr = fits.getheader(filename, -1)
        if "RADESYS" in hdr and "PC001001" not in hdr:
            log("WCS already normalised: %s" % filename)
            return
        log("Normalising WCS in %s (fitsio path)" % filename)
        hdr["RADESYS"] = "FK5"
        for old, new in [("PC001001", "PC1_1"), ("PC001002", "PC1_2"),
                         ("PC002001", "PC2_1"), ("PC002002", "PC2_2")]:
            if old in hdr:
                hdr.rename_keyword(old, new)
        raw = fitsio.read(filename, ext=1)
        is_mask = "BSOFTEN" not in hdr and "BOFFSET" not in hdr
        if is_mask:
            data = raw.astype(np.int16)
        else:
            data = raw.astype(np.float32)
            if "BSOFTEN" in hdr and "BOFFSET" in hdr:
                log("Linearising ASINH scaling in %s" % filename)
                x = data * 0.4 * np.log(10)
                data = hdr["BOFFSET"] + hdr["BSOFTEN"] * (np.exp(x) - np.exp(-x))
                hdr["FLXSCALE"] = 1.0 / hdr["BSOFTEN"]
                for kw in ["BSOFTEN", "BOFFSET", "BLANK"]:
                    hdr.remove(kw, ignore_missing=True)
        out = outname or filename
        log("Writing normalised skycell to %s" % out)
        fits.writeto(out, data, hdr, overwrite=True)

    _stdpipe_templates.normalize_ps1_skycell = _normalize_ps1_skycell_fixed
    _ps1_patch_applied = True
    logger.debug("PS1 template: applied fitsio-based normalize_ps1_skycell patch")


_get_skycells_patch_applied = False


def _patch_get_skycells() -> None:
    """stdpipe.templates.get_skycells (as installed here) crashes on some
    real PS1 skycell *downloads* -- not the same bug/site as
    normalize_ps1_skycell above. This one is in astropy's own tiled-
    compression decompression itself: `hdu[1].data` on the freshly
    downloaded CompImageHDU raises `ValueError: cannot convert float NaN to
    integer` inside decompress_image_data_section, for skycells whose
    BLANK-valued tiles it tries to NaN-fill into an integer-typed array.
    Verified directly against a real download for this campaign's field
    (rings.v3.skycell.2011.080.stk.r.unconv.fits) -- reading the exact same
    cached file via fitsio instead of astropy's `.data` property works
    fine, matching the precedent above: swap readers for the one call site
    astropy can't handle, don't touch anything else.

    get_skycells does a lot around this one line (caching, PS1/LS
    normalization, invvar masking, downscaling) that has to keep working
    identically, so this reimplements the whole function rather than
    patching just the read -- only the `hdu[1].data` line actually differs
    from stdpipe's original (see the "fitsio fallback" comment below).
    Only applied once per process, and only if `fitsio` is importable, same
    guard as the patch above.
    """
    global _get_skycells_patch_applied
    if _get_skycells_patch_applied:
        return
    try:
        import fitsio
        import stdpipe.templates as _stdpipe_templates
    except ImportError:
        logger.debug("PS1 template: fitsio not available, using stdpipe's own "
                      "get_skycells (may fail on some real skycells)")
        return

    import os
    import tempfile
    from astropy.io import fits as _fits

    def _get_skycells_fixed(
        ra0, dec0, sr0, band='r', ext='image', survey='ps1', normalize=True,
        overwrite=False, wcs=None, width=None, height=None, _cachedir=None,
        _cache_downscale=1, _tmpdir=None, verbose=False,
    ):
        log = (verbose if callable(verbose) else print) if verbose else lambda *a, **k: None

        if _cachedir is not None:
            _cachedir = os.path.expanduser(_cachedir)
            log('Cache location is', _cachedir)
        else:
            if _tmpdir is None:
                _tmpdir = tempfile.gettempdir()
            _cachedir = os.path.join(_tmpdir, survey)
            log('Cache location not specified, falling back to %s' % _cachedir)

        try:
            os.makedirs(_cachedir)
        except OSError:
            pass

        filenames = []
        cells = _stdpipe_templates.find_skycells(
            ra0, dec0, sr0, band=band, ext=ext, survey=survey, wcs=wcs,
            width=width, height=height,
        )

        for cell in cells:
            cellname = os.path.basename(cell)
            filename = os.path.join(_cachedir, cellname)

            if survey == 'ls' and filename.endswith('.fz'):
                filename = os.path.splitext(filename)[0]

            if _cache_downscale > 1:
                filename, fext = os.path.splitext(filename)
                filename = filename + ('.x%d' % _cache_downscale) + fext

            if os.path.exists(filename) and not overwrite:
                log('%s already downloaded' % os.path.split(filename)[-1])
            else:
                log('Downloading %s' % cellname)

                hdu = _stdpipe_templates.fits_open_remote(cell)
                if hdu is not None:
                    try:
                        image, header = hdu[1].data, hdu[1].header
                    except ValueError:
                        # fitsio fallback: astropy's compressed-tile decompression
                        # can't handle this skycell's BLANK-into-integer tiles --
                        # read the exact same cached local file with fitsio instead.
                        local_path = hdu.filename()
                        log('astropy decompression failed for %s, retrying with fitsio' % cellname)
                        image = fitsio.read(local_path, ext=1)
                        header = _fits.getheader(local_path, ext=1)

                    if normalize:
                        if survey == 'ps1':
                            # Not _stdpipe_templates.normalize_ps1_skycell here:
                            # _patch_normalize_ps1_skycell above replaces that name
                            # with a *file-based* fixed version (filename, outname)
                            # -- a different calling convention than the
                            # (image, header) -> (image, header) one get_skycells
                            # actually uses at this call site. Calling the patched
                            # name here would pass `image` positionally where it
                            # expects a filename (verified directly: raises
                            # "truth value of an array is ambiguous" inside
                            # fits.getheader). Inlined here instead, straight from
                            # stdpipe's original array-based normalize_ps1_skycell
                            # body, so this call site never touches that patch.
                            if 'RADESYS' not in header and 'PC001001' in header:
                                header = header.copy()
                                header['RADESYS'] = 'FK5'
                                header.rename_keyword('PC001001', 'PC1_1')
                                header.rename_keyword('PC001002', 'PC1_2')
                                header.rename_keyword('PC002001', 'PC2_1')
                                header.rename_keyword('PC002002', 'PC2_2')
                                if 'BSOFTEN' in header and 'BOFFSET' in header:
                                    x = image * 0.4 * np.log(10)
                                    image = header['BOFFSET'] + header['BSOFTEN'] * (np.exp(x) - np.exp(-x))
                                    image /= header['EXPTIME']
                                    for kw in ['BSOFTEN', 'BOFFSET', 'BLANK']:
                                        header.remove(kw, ignore_missing=True)

                        if survey == 'ls' and ext == 'image':
                            try:
                                ihdu = _stdpipe_templates.fits_open_remote(cell.replace('-image-', '-invvar-'))
                                if ihdu is not None:
                                    invvar = ihdu[1].data
                                    image[invvar == 0] = np.nan
                                    ihdu.close()
                            except Exception:
                                pass

                    if _cache_downscale > 1:
                        image, header = _stdpipe_templates.cutouts.downscale_image(
                            image, header=header, scale=_cache_downscale,
                            mode='or' if ext == 'mask' else 'sum',
                        )
                        log("Downscaling the image and storing it as", os.path.split(filename)[-1])

                    _fits.writeto(filename, image, header, overwrite=True)
                    hdu.close()

            if os.path.exists(filename):
                filenames.append(filename)

        return filenames

    _stdpipe_templates.get_skycells = _get_skycells_fixed
    _get_skycells_patch_applied = True
    logger.debug("PS1 template: applied fitsio-based get_skycells patch")


def _field_cache_key(ra: float, dec: float, band: str, grid_deg: float = 0.02) -> str:
    """Filesystem-safe cache key for a field/band template, rounded to a
    coarse grid (0.02 deg ~= 1.2 arcmin) so slightly different pointings of
    the same campaign share one cache entry -- real telescope pointings
    jitter by much less than a field width between nights of the same
    campaign, but treating every pointing as a brand-new field would defeat
    the whole point of caching.
    """
    ra_r = round(ra / grid_deg) * grid_deg
    dec_r = round(dec / grid_deg) * grid_deg
    key = f"{band}_{ra_r:.4f}_{dec_r:+.4f}"
    return key.replace(".", "p").replace("+", "").replace("-", "m")


def _enforce_cache_budget(cache_dir: Path, max_size_gb: Optional[float]) -> None:
    """LRU eviction by last-access time -- same pattern
    frontend_generator.py already uses for the website directory
    (cleanup_old_files/enforce_disk_budget_strict), reused here rather than
    inventing a second cleanup mechanism for what is otherwise the exact
    same problem (a cache of large files that will grow unbounded).
    """
    if not max_size_gb or max_size_gb <= 0:
        return
    max_bytes = max_size_gb * 1024 ** 3
    try:
        files = sorted(cache_dir.glob("*.fits"), key=lambda p: p.stat().st_atime)
    except OSError:
        return
    total = sum(f.stat().st_size for f in files)
    for f in files:
        if total <= max_bytes:
            break
        try:
            size = f.stat().st_size
            f.unlink()
            total -= size
            logger.info(f"Template cache: evicted {f.name} ({size / 1e6:.1f} MB) "
                        f"to stay under {max_size_gb} GB budget")
        except OSError as e:
            logger.debug(f"Template cache: could not evict {f}: {e}")


def get_template_own_epoch(
    detection_tables: List,
    fits_paths: List,
    target_ra: Optional[float] = None,
    target_dec: Optional[float] = None,
    target_exclusion_radius_arcsec: float = 5.0,
) -> Optional[Tuple[np.ndarray, str]]:
    """Pick this campaign's own best-quality prior epoch as the template,
    via ReferenceFrameSelector (previously dead code -- see
    reference_frame.py's own module docstring for why it was never wired
    up before this).

    fits_paths: FITS file paths parallel to detection_tables (same index)
    -- ReferenceFrameSelector only ever worked with detection tables
    (sky-position/quality metadata), not pixel data, so the caller supplies
    the matching FITS paths separately.

    target_ra/target_dec: known target position, optional but strongly
    recommended whenever it's available (i.e. whenever this is called for
    continued monitoring of an already-flagged transient rather than a
    blind first-detection search) -- without it, the selected reference
    epoch could already contain the target near its normal brightness,
    silently under-representing its true magnitude in the resulting diff
    image. See reference_frame.py's module docstring for the full
    reasoning and the real tests/2026kid/ case this was found from.

    Returns (template_array, provenance_string), or None if no epochs are
    available or the selected epoch's FITS can't be read.
    """
    if not detection_tables or not fits_paths or len(detection_tables) != len(fits_paths):
        return None

    selector = ReferenceFrameSelector(
        detection_tables, target_ra=target_ra, target_dec=target_dec,
        target_exclusion_radius_arcsec=target_exclusion_radius_arcsec,
    )
    idx = selector.reference_idx
    ref_path = Path(fits_paths[idx])
    if not ref_path.exists():
        logger.warning(f"own_epoch template: reference FITS {ref_path} not found")
        return None

    try:
        with fits.open(ref_path) as hdul:
            template = np.array(hdul[0].data, dtype=np.float64)
    except Exception as e:
        logger.warning(f"own_epoch template: could not read {ref_path}: {e}")
        return None

    provenance = f"own_epoch:{ref_path.stem}"
    if target_ra is not None and selector.quality_metrics[idx].target_present:
        # Every candidate epoch had the target in it (selector already
        # logged a warning) -- flag it in the provenance string too, so
        # anything downstream displaying/reporting this (e.g. the frontend's
        # "Template source" row) surfaces it without needing its own log access.
        provenance += ":target-contaminated"
    return template, provenance


def _get_survey_template(
    survey: str,
    stdpipe_fn_name: str,
    header,
    band: str = "r",
    cache_dir=None,
    cache_max_size_gb: Optional[float] = 20.0,
    verbose=False,
) -> Optional[Tuple[np.ndarray, Optional[np.ndarray], str]]:
    """Shared implementation for get_template_ps1/get_template_legacysurvey
    -- same caching/error-handling shape, only the survey name and the
    stdpipe.templates function called differ.
    """
    wcs = WCS(header)
    naxis1 = header.get("NAXIS1", 1024)
    naxis2 = header.get("NAXIS2", 1024)
    ra0, dec0 = wcs.all_pix2world([naxis1 / 2.0], [naxis2 / 2.0], 0)
    ra0, dec0 = float(ra0[0]), float(dec0[0])

    cache_path = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{survey}_{_field_cache_key(ra0, dec0, band)}.fits"
        if cache_path.exists():
            logger.info(f"Template cache: using cached {survey} {band}-band template {cache_path}")
            try:
                with fits.open(cache_path) as hdul:
                    image = np.array(hdul[0].data, dtype=np.float64)
                    mask = np.array(hdul[1].data, dtype=bool) if len(hdul) > 1 else None
                cache_path.touch()  # refresh mtime/atime so LRU eviction sees this as recently used
                return image, mask, f"{survey}_template"
            except Exception as e:
                logger.debug(f"Template cache: cached file {cache_path} unreadable ({e}), re-fetching")

    try:
        from stdpipe import templates as stdpipe_templates
        fetch_fn = getattr(stdpipe_templates, stdpipe_fn_name)
    except (ImportError, AttributeError):
        logger.warning(f"{survey} template: this stdpipe build has no {stdpipe_fn_name}")
        return None

    if survey == "ps1":
        _patch_normalize_ps1_skycell()
        _patch_get_skycells()

    try:
        image, mask = fetch_fn(band=band, header=header, verbose=verbose)
    except Exception as e:
        # stdpipe's own retrieval/reprojection code can raise on a real,
        # otherwise-uncontrollable external failure (verified directly: a
        # bug in astropy's compressed-tile decompression of certain PS1
        # skycell masks raises ValueError even after the normalize_ps1_skycell
        # patch above, depending on which skycells cover a given field) --
        # degrade to "no template" the same way a missing `swarp` binary
        # does, rather than letting an external library's bug crash the
        # whole detection run.
        logger.warning(f"{survey} template: retrieval raised {type(e).__name__}: {e} -- skipping")
        return None

    if image is None:
        logger.warning(f"{survey} template: retrieval failed (missing `swarp` binary, "
                        f"no survey coverage, or network unavailable)")
        return None

    if cache_path is not None:
        try:
            hdul = fits.HDUList([fits.PrimaryHDU(image.astype(np.float32))])
            if mask is not None:
                hdul.append(fits.ImageHDU(mask.astype(np.uint8)))
            hdul.writeto(str(cache_path), overwrite=True)
            _enforce_cache_budget(cache_dir, cache_max_size_gb)
        except Exception as e:
            logger.debug(f"Template cache: could not write {cache_path}: {e}")

    return image, mask, f"{survey}_template"


def get_template_ps1(header, band: str = "r", cache_dir=None,
                      cache_max_size_gb: Optional[float] = 20.0, verbose=False):
    """External-survey template from Pan-STARRS1, reprojected onto the
    science epoch's own WCS via SWarp. Returns (image, mask, provenance) or
    None -- see _get_survey_template for the shared caching/error logic.
    """
    return _get_survey_template("ps1", "get_ps1_image_and_mask", header,
                                 band=band, cache_dir=cache_dir,
                                 cache_max_size_gb=cache_max_size_gb, verbose=verbose)


def get_template_legacysurvey(header, band: str = "r", cache_dir=None,
                               cache_max_size_gb: Optional[float] = 20.0, verbose=False):
    """External-survey template from DESI Legacy Survey. Only available if
    the installed stdpipe build has get_ls_image_and_mask -- verified
    against this project's own installed stdpipe, which currently doesn't
    (an older/pinned checkout); returns None rather than raising if so, the
    same way a missing `swarp` binary degrades for get_template_ps1.
    """
    return _get_survey_template("legacysurvey", "get_ls_image_and_mask", header,
                                 band=band, cache_dir=cache_dir,
                                 cache_max_size_gb=cache_max_size_gb, verbose=verbose)
