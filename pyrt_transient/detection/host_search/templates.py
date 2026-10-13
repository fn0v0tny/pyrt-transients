"""Archival survey images for the host search, with a per-field cache.

Sources, in the order a config lists them:

- "legacysurvey": one FITS cutout of the field from the Legacy Survey
  viewer (DR10, 0.5"/px by default; a 20' field is a 25 MB file and 10-30 s
  of server time). Deepest and best characterised; covers most of the
  extragalactic sky south of about +32 deg plus the BASS/MzLS north.
- "panstarrs": Pan-STARRS DR1 through the CDS hips2fits service
  (stdpipe.templates.get_hips_image), a few seconds, dec > -30. About a
  sigma worse than Legacy Survey on EP261006a but the right choice where
  Legacy Survey is missing.
- "hips:<survey id>": any other HiPS survey (e.g. "hips:CDS/P/DES-DR2/r").
  DSS2 is deliberately not a default: photographic plates do not subtract
  to the few-percent level a nuclear excess needs.

The cache is one FITS per field/band/size; its header records the centre
and the size so a template fetched for the *trigger* position (prefetch,
before any frame exists) serves the *pointing* the telescope settled on,
which can differ by an arcminute. find_cached_template looks for any
entry whose centre is close enough and which covers the requested size.
"""
import logging
import subprocess
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from pyrt_transient.detection.host_search.galaxies import sep_arcsec

logger = logging.getLogger("detection.host_search.templates")

LS_CUTOUT_URL = "https://www.legacysurvey.org/viewer/fits-cutout"
LS_MAX_SIZE_PX = 3000
HIPS_IDS = {"panstarrs": "CDS/P/PanSTARRS/DR1/{band}"}


def default_cache_dir() -> Path:
    return Path.home() / "catalog_cache" / "host_templates"


def _tan_wcs(ra: float, dec: float, pixscale_arcsec: float, size_px: int) -> WCS:
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [ra, dec]
    w.wcs.crpix = [size_px / 2.0 + 0.5, size_px / 2.0 + 0.5]
    w.wcs.cdelt = [-pixscale_arcsec / 3600.0, pixscale_arcsec / 3600.0]
    return w


def _strip_third_axis(header):
    h = header.copy()
    for k in list(h):
        if k[-1:] == "3" and k[:-1] in ("NAXIS", "CTYPE", "CRVAL", "CRPIX", "CDELT", "CUNIT"):
            del h[k]
    h["NAXIS"] = 2
    return h


def _fetch_legacysurvey(ra, dec, band, pixscale, size_px, timeout, log) -> Optional[Tuple[np.ndarray, WCS]]:
    size_px = min(int(size_px), LS_MAX_SIZE_PX)
    url = (f"{LS_CUTOUT_URL}?ra={ra:.6f}&dec={dec:.6f}&layer=ls-dr10&pixscale={pixscale:g}"
           f"&size={size_px}&bands={band}")
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as tmp:
        path = Path(tmp.name)
    try:
        r = subprocess.run(["curl", "-sS", "-f", "-m", str(int(timeout)), "-o", str(path), url],
                           capture_output=True, text=True)
        if r.returncode != 0 or not path.exists() or path.stat().st_size < 5000:
            log.warning(f"Host search: Legacy Survey cutout failed ({r.stderr.strip()[:200] or 'empty file'})")
            return None
        with fits.open(path) as hdul:
            data = hdul[0].data
            data = data[0] if data.ndim == 3 else data
            data = np.array(data, dtype=np.float32)
            wcs = WCS(_strip_third_axis(hdul[0].header))
    finally:
        try:
            path.unlink()
        except OSError:
            pass
    # No coverage comes back as an all-zero (or all-NaN) image, not an error.
    finite = np.isfinite(data)
    if not finite.any() or np.count_nonzero(data[finite]) < 0.2 * finite.sum():
        log.info("Host search: Legacy Survey has no coverage here")
        return None
    return data, wcs


def _fetch_hips(hips_id, ra, dec, pixscale, size_px, timeout, log) -> Optional[Tuple[np.ndarray, WCS]]:
    from stdpipe.templates import get_hips_image
    wcs = _tan_wcs(ra, dec, pixscale, size_px)
    try:
        image, header = get_hips_image(hips_id, wcs=wcs, shape=(size_px, size_px), get_header=True, normalize=False)
    except Exception as e:
        log.warning(f"Host search: hips2fits {hips_id} failed ({e})")
        return None
    if image is None:
        return None
    data = np.array(image, dtype=np.float32)
    finite = np.isfinite(data)
    if not finite.any() or np.count_nonzero(data[finite]) < 0.2 * finite.sum():
        log.info(f"Host search: {hips_id} has no coverage here")
        return None
    return data, WCS(header)


def _fetch(source: str, ra, dec, band, pixscale, size_px, timeout, log):
    if source == "legacysurvey":
        return _fetch_legacysurvey(ra, dec, band, pixscale, size_px, timeout, log)
    if source in HIPS_IDS:
        return _fetch_hips(HIPS_IDS[source].format(band=band), ra, dec, pixscale, size_px, timeout, log)
    if source.startswith("hips:"):
        return _fetch_hips(source[5:], ra, dec, pixscale, size_px, timeout, log)
    log.warning(f"Host search: unknown template source '{source}'")
    return None


def find_cached_template(cache_dir: Path, ra: float, dec: float, band: str, min_size_arcmin: float,
                         max_offset_arcmin: float = 3.0) -> Optional[Path]:
    """A cached template whose centre is within `max_offset_arcmin` of
    (ra, dec) and which covers `min_size_arcmin` around that point."""
    cache_dir = Path(cache_dir)
    if not cache_dir.is_dir():
        return None
    best, best_off = None, None
    for path in cache_dir.glob("*.fits"):
        try:
            h = fits.getheader(path)
            if str(h.get("HSBAND", "")) != band:
                continue
            off = sep_arcsec(ra, dec, float(h["HSRA"]), float(h["HSDEC"])) / 60.0
            if off > max_offset_arcmin or float(h["HSSIZE"]) < min_size_arcmin + 2 * off:
                continue
        except (OSError, KeyError, TypeError, ValueError):
            continue
        if best is None or off < best_off:
            best, best_off = path, off
    return best


def load_template(path: Path) -> Tuple[np.ndarray, WCS, str]:
    with fits.open(path) as hdul:
        data = np.array(hdul[0].data, dtype=np.float32)
        return data, WCS(hdul[0].header), str(hdul[0].header.get("HSSRC", "unknown"))


def fetch_template(ra: float, dec: float, size_arcmin: float, band: str = "r",
                   sources: Iterable[str] = ("legacysurvey", "panstarrs"), cache_dir=None,
                   pixscale_arcsec: float = 0.5, timeout: float = 300.0, log=None,
                   use_cache: bool = True) -> Optional[Tuple[np.ndarray, WCS, str]]:
    """(image, wcs, source) for a `size_arcmin` square around (ra, dec), from
    the cache or the first source with coverage. None if no source has
    coverage or all failed."""
    log = log or logger
    cache_dir = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    if use_cache:
        cached = find_cached_template(cache_dir, ra, dec, band, size_arcmin)
        if cached is not None:
            log.info(f"Host search: template from cache {cached.name}")
            try:
                return load_template(cached)
            except Exception as e:
                log.debug(f"Host search: cached template unreadable ({e}), refetching")
    size_px = int(round(size_arcmin * 60.0 / pixscale_arcsec))
    for source in sources:
        t0 = time.monotonic()
        got = _fetch(source, ra, dec, band, pixscale_arcsec, size_px, timeout, log)
        if got is None:
            continue
        data, wcs = got
        log.info(f"Host search: {source} template {data.shape[1]}x{data.shape[0]} px fetched in "
                 f"{time.monotonic() - t0:.1f} s")
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            hdr = wcs.to_header()
            hdr["HSSRC"] = source
            hdr["HSBAND"] = band
            hdr["HSRA"] = ra
            hdr["HSDEC"] = dec
            hdr["HSSIZE"] = (float(size_arcmin), "arcmin, side of the square")
            name = f"{source.replace(':', '_').replace('/', '_')}_{band}_{ra:.4f}_{dec:+.4f}_{size_arcmin:.0f}.fits"
            fits.writeto(cache_dir / name, data, hdr, overwrite=True)
        except Exception as e:
            log.debug(f"Host search: could not cache template ({e})")
        return data, wcs, source
    return None


def prefetch_template(ra: float, dec: float, size_arcmin: float, band: str = "r",
                      sources: Iterable[str] = ("legacysurvey", "panstarrs"), cache_dir=None,
                      pixscale_arcsec: float = 0.5, timeout: float = 300.0, log=None) -> Optional[str]:
    """Make sure a template for this field is in the cache; returns the
    source name, or None if nothing covers the field. Meant to run as
    soon as a trigger position is known, before the first frame."""
    log = log or logger
    cache_dir = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    if find_cached_template(cache_dir, ra, dec, band, size_arcmin) is not None:
        log.info("Host search: template already cached")
        return "cached"
    got = fetch_template(ra, dec, size_arcmin, band, sources, cache_dir, pixscale_arcsec, timeout, log,
                         use_cache=False)
    return None if got is None else got[2]
