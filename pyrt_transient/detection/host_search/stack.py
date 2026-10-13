"""A median stack of one observation's frames on a common pixel grid, in
counts calibrated to one zeropoint.

Built here rather than with pyrt-combine because the host search needs
the stack in memory with a known WCS and zeropoint, not a photometered
ECSV; and because pyrt-combine forks a worker per core (stacking.py).
Needs the `reproject` package
(optional dependency `hostsearch`); without it build_stack raises
ImportError and the search is skipped with a warning.

Only one exposure class is stacked: a D50 GRB sequence opens with 0.1 s
pointing frames before the 300 s science frames, and those add nothing
but noise once scaled to a common zeropoint. The zeropoint is measured on
each frame in FITS counts from its own calibrated stars (the ECSV FLUX
column is not in those units).
"""
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

try:
    import sep
except ImportError:  # stdpipe installs it as sep_x
    import sep_x as sep

logger = logging.getLogger("detection.host_search.stack")


@dataclass
class StackResult:
    data: np.ndarray          # background-subtracted, NaN where fewer than a third of the frames cover
    wcs: WCS
    zp: float                 # mag = zp - 2.5 log10(counts)
    fwhm_px: float
    n_frames: int
    bkg_rms: float
    frames: List[str] = field(default_factory=list)
    seconds: float = 0.0


def _native(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    return a.astype(a.dtype.newbyteorder("=")) if a.dtype.byteorder not in ("=", "|") else a


def frame_zeropoint(image: np.ndarray, table: Table, fwhm_px: float) -> Optional[float]:
    """Zeropoint of `image` in its own counts, from the calibrated stars in
    its ECSV table: median of MAG_CALIB + 2.5 log10(aperture counts) over
    stars with MAGERR_CALIB < 0.04, a 2 FWHM aperture."""
    need = ("MAG_CALIB", "MAGERR_CALIB", "X_IMAGE", "Y_IMAGE")
    if any(c not in table.colnames for c in need):
        return None
    img = _native(image)
    bk = sep.Background(img, bw=64, bh=64)
    m = np.ma.filled(np.ma.asarray(table["MAG_CALIB"], dtype=float), np.nan)
    me = np.ma.filled(np.ma.asarray(table["MAGERR_CALIB"], dtype=float), np.nan)
    x = np.ma.filled(np.ma.asarray(table["X_IMAGE"], dtype=float), np.nan) - 1
    y = np.ma.filled(np.ma.asarray(table["Y_IMAGE"], dtype=float), np.nan) - 1
    ok = np.isfinite(m) & np.isfinite(me) & np.isfinite(x) & np.isfinite(y) & (me < 0.04) & (m > 12)
    if ok.sum() < 5:
        return None
    flux, _, _ = sep.sum_circle(img - bk.back(), x[ok], y[ok], 2.0 * fwhm_px)
    good = flux > 0
    if good.sum() < 5:
        return None
    return float(np.median(m[ok][good] + 2.5 * np.log10(flux[good])))


def select_frames(pairs: Sequence[Tuple[Path, Table]], min_exptime_fraction: float = 0.5,
                  max_frames: int = 40) -> List[Tuple[Path, Table]]:
    """The frames worth stacking: the dominant photometric band, exposures
    at least `min_exptime_fraction` of the longest, newest `max_frames`."""
    usable = []
    for p, t in pairs:
        try:
            exp = float(t.meta.get("EXPTIME"))
        except (TypeError, ValueError):
            continue
        if p is None or not Path(p).exists() or not np.isfinite(exp) or exp <= 0:
            continue
        usable.append((p, t, exp, str(t.meta.get("PHFILTER", ""))))
    if not usable:
        return []
    emax = max(u[2] for u in usable)
    usable = [u for u in usable if u[2] >= min_exptime_fraction * emax]
    bands = {}
    for u in usable:
        bands.setdefault(u[3], []).append(u)
    best = max(bands.values(), key=len)
    best.sort(key=lambda u: float(u[1].meta.get("CTIME", 0) or 0))
    return [(u[0], u[1]) for u in best[-max_frames:]]


def build_stack(pairs: Sequence[Tuple[Path, Table]], log=None) -> Optional[StackResult]:
    """Median-stack `pairs` (fits path, its ECSV table) on the grid of the
    best-seeing frame. None if nothing could be stacked."""
    from reproject import reproject_interp     # optional dependency, imported late on purpose

    log = log or logger
    t0 = time.monotonic()
    frames = []
    for p, t in pairs:
        try:
            with fits.open(p) as hdul:
                hdr = hdul[0].header
                img = _native(hdul[0].data)
            fwhm = float(t.meta.get("FWHM", 3.5))
            zp = frame_zeropoint(img, t, fwhm)
            if zp is None:
                log.debug(f"Host search: no zeropoint for {Path(p).name}, skipped")
                continue
            bk = sep.Background(img, bw=64, bh=64)
            frames.append(dict(path=str(p), img=img - bk.back(), hdr=hdr, fwhm=fwhm, zp=zp, rms=float(bk.globalrms)))
        except Exception as e:
            log.debug(f"Host search: cannot read {p} ({e})")
    if not frames:
        return None
    ref = min(frames, key=lambda f: f["fwhm"])
    ref_wcs = WCS(ref["hdr"]); shape = ref["img"].shape
    cube = np.full((len(frames),) + shape, np.nan, dtype=np.float32)
    for i, fr in enumerate(frames):
        img = fr["img"] * 10 ** (-0.4 * (ref["zp"] - fr["zp"]))
        if fr is ref:
            cube[i] = img
        else:
            cube[i] = reproject_interp((img, WCS(fr["hdr"])), ref_wcs, shape_out=shape, order="bilinear")[0]
    n = len(frames)
    stack = cube[0] if n == 1 else np.nanmedian(cube, axis=0).astype(np.float32)
    cover = np.isfinite(cube).sum(axis=0)
    stack = np.where(cover >= max(1, n // 3), stack, np.nan).astype(np.float32)
    del cube

    # Zeropoint and PSF of the stack itself, from the reference frame's stars
    valid = np.isfinite(stack)
    st = np.where(valid, stack, 0.0).astype(np.float32)
    cen = stack[shape[0] // 4: 3 * shape[0] // 4, shape[1] // 4: 3 * shape[1] // 4]
    cen = cen[np.isfinite(cen)]
    rms = float(1.4826 * np.median(np.abs(cen - np.median(cen)))) if cen.size else float(ref["rms"])
    fwhm_px = float(np.median([f["fwhm"] for f in frames]))
    try:
        objs = sep.extract(st, 8.0, err=max(rms, 1e-3), minarea=5, mask=~valid)
        bright = ((objs["flux"] > 2000 * rms) & (objs["flux"] < 0.3 * objs["flux"].max())
                  & (objs["b"] / objs["a"] > 0.8))
        if bright.sum() >= 5:
            fwhm_px = float(np.median(2.355 * objs["a"][bright]))
    except Exception:
        pass
    zp = ref["zp"]
    ref_table = next(t for p, t in pairs if str(p) == ref["path"])
    zp_stack = frame_zeropoint(st, ref_table, fwhm_px)
    if zp_stack is not None:
        zp = zp_stack
    res = StackResult(data=stack, wcs=ref_wcs, zp=zp, fwhm_px=fwhm_px, n_frames=n, bkg_rms=rms,
                      frames=[f["path"] for f in frames], seconds=time.monotonic() - t0)
    log.info(f"Host search: stacked {n} frames in {res.seconds:.1f} s (FWHM {fwhm_px:.1f} px, "
             f"rms {rms:.1f}, ZP {zp:.2f})")
    return res
