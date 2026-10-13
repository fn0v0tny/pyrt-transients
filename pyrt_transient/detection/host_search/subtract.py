"""PSF-matched subtraction of the error-circle region and photometry of
the galaxies on the difference image.

hotpants is called directly with explicit valid-pixel limits: the stack is
background-subtracted, and both inputs are offset by a constant so they
look like ordinary positive images to it (stdpipe's wrapper derives its
limits from image statistics and rejected every stamp on this data). The
kernel is fitted over the whole region on its stars (6 x 6 sub-regions,
second-order spatial variation, first-order background); subtracting each
galaxy in its own stamp fits the kernel on the galaxy and absorbs the
very excess being measured (EP261006a came out *negative* that way).

Photometry uses a nuclear aperture of half a FWHM: the residual
systematics of a galaxy scale with its extended light, a transient on the
nucleus is a point source, and on EP261006a 0.5 FWHM separates the host
from the other galaxies by 3.1 sigma where 1.5 FWHM gives 2.3.
"""
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import binary_dilation, gaussian_filter

try:
    import sep
except ImportError:
    import sep_x as sep

logger = logging.getLogger("detection.host_search.subtract")

OFFSET = 100.0
TEMPLATE_SCALE = 1000.0


def hotpants_available(exe: str = "hotpants") -> bool:
    return shutil.which(exe) is not None


def match_template(template: np.ndarray, template_wcs: WCS, target_wcs: WCS, shape: Tuple[int, int],
                   template_pixscale_arcsec: float, smooth_fwhm_arcsec: float = 2.4) -> Tuple[np.ndarray, np.ndarray]:
    """The template smoothed to `smooth_fwhm_arcsec` (so it stays well
    sampled on the coarser science grid; hotpants only ever convolves the
    sharper image) and resampled onto `target_wcs`. Returns (image, mask)."""
    from reproject import reproject_interp
    tnan = ~np.isfinite(template) | (template == 0)
    tm = gaussian_filter(np.where(tnan, 0.0, template).astype(np.float32),
                         sigma=(smooth_fwhm_arcsec / 2.355) / template_pixscale_arcsec)
    tr, fp = reproject_interp((tm, template_wcs), target_wcs, shape_out=shape, order="bilinear")
    tn, _ = reproject_interp((tnan.astype(np.float32), template_wcs), target_wcs, shape_out=shape,
                             order="nearest-neighbor")
    tr = np.asarray(tr, dtype=np.float32)
    mask = ~np.isfinite(tr) | (fp == 0) | (np.nan_to_num(tn, nan=1.0) > 0)
    return np.where(mask, 0.0, tr).astype(np.float32), mask


def saturation_mask(sci: np.ndarray, level: float = 45000.0, grow: int = 4) -> np.ndarray:
    return binary_dilation(np.nan_to_num(sci, nan=0.0) > level, iterations=grow)


def run_hotpants(sci: np.ndarray, tmpl: np.ndarray, mask: np.ndarray, fwhm_px: float, n_frames: int,
                 gain: float = 0.81, exe: str = "hotpants", nsx: int = 6, nsy: int = 6, ko: int = 2,
                 bgo: int = 1, log=None) -> Optional[np.ndarray]:
    """Difference image (science minus kernel-convolved template), NaN where
    masked, or None if hotpants failed or is not installed."""
    log = log or logger
    if not hotpants_available(exe):
        log.warning(f"Host search: `{exe}` not found, no subtraction")
        return None
    with tempfile.TemporaryDirectory(prefix="hostsearch_") as d:
        d = Path(d)
        fits.writeto(d / "i.fits", (np.where(mask, 0.0, sci) + OFFSET).astype(np.float32), overwrite=True)
        fits.writeto(d / "t.fits", (np.where(mask, 0.0, tmpl) * TEMPLATE_SCALE + OFFSET).astype(np.float32),
                     overwrite=True)
        fits.writeto(d / "m.fits", mask.astype(np.int16), overwrite=True)
        cmd = [exe, "-inim", str(d / "i.fits"), "-tmplim", str(d / "t.fits"), "-outim", str(d / "d.fits"),
               "-imi", str(d / "m.fits"), "-tmi", str(d / "m.fits"),
               "-il", "-1000", "-iu", "60000", "-tl", "-1e6", "-tu", "1e9", "-tuk", "1e9",
               "-n", "i", "-c", "t", "-ng", "3", "6", "0.5", "4", "1.0", "2", "2.0",
               "-r", f"{max(5, int(round(1.7 * fwhm_px)))}", "-rss", f"{max(7, int(round(2.4 * fwhm_px)))}",
               "-nsx", str(nsx), "-nsy", str(nsy), "-ko", str(ko), "-bgo", str(bgo),
               "-ig", f"{0.64 * max(1, n_frames) * gain:.2f}", "-tg", "10000", "-v", "0"]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        except (OSError, subprocess.SubprocessError) as e:
            log.warning(f"Host search: hotpants did not run ({e})")
            return None
        if r.returncode != 0 or not (d / "d.fits").exists():
            log.warning(f"Host search: hotpants failed: {(r.stdout + r.stderr)[-300:]}")
            return None
        diff = np.asarray(fits.getdata(d / "d.fits"), dtype=np.float32)
    diff[mask] = np.nan
    return diff


def measure_galaxies(diff: np.ndarray, sci: np.ndarray, mask: np.ndarray, wcs: WCS, galaxies: List[dict],
                     fwhm_px: float, zp: float, aperture_fwhm: float = 0.5) -> List[Dict]:
    """Residual flux at each galaxy's catalogue position in an aperture of
    `aperture_fwhm` x FWHM, local background and noise from an annulus of
    3-7 FWHM on the difference image. Adds residual_flux / residual_err /
    residual_snr / excess_frac (residual over the galaxy's own light in
    the same aperture) / residual_mag / residual_lim5 to a copy of each
    galaxy dict; galaxies outside the image or on masked pixels are left
    out."""
    r_ap = aperture_fwhm * fwhm_px
    d0 = np.where(np.isfinite(diff), diff, 0.0).astype(np.float32)
    s0 = np.where(np.isfinite(sci) & ~mask, sci, 0.0).astype(np.float32)
    ny, nx = diff.shape
    yy, xx = np.mgrid[:ny, :nx]
    out = []
    for g in galaxies:
        cx, cy = wcs.all_world2pix([[g["ra"], g["dec"]]], 0)[0]
        if not (3 * fwhm_px < cx < nx - 3 * fwhm_px and 3 * fwhm_px < cy < ny - 3 * fwhm_px):
            continue
        rr = np.hypot(xx - cx, yy - cy)
        if mask[rr <= r_ap].mean() > 0.1:
            continue
        ann = (rr > 3 * fwhm_px) & (rr < 7 * fwhm_px) & ~mask & np.isfinite(diff)
        if ann.sum() < 50:
            continue
        loc = float(np.median(d0[ann]))
        sig = 1.4826 * float(np.median(np.abs(d0[ann] - loc)))
        fd = float(sep.sum_circle(d0, [cx], [cy], r_ap)[0][0]) - loc * np.pi * r_ap ** 2
        fs = float(sep.sum_circle(s0, [cx], [cy], r_ap)[0][0])
        err = max(sig, 1e-6) * np.sqrt(np.pi * r_ap ** 2)
        row = dict(g)
        row.update(residual_flux=fd, residual_err=err, residual_snr=fd / err,
                   excess_frac=(fd / fs) if fs > 0 else np.nan,
                   residual_mag=(zp - 2.5 * np.log10(fd)) if fd > 0 else np.nan,
                   residual_lim5=zp - 2.5 * np.log10(5 * err),
                   galaxy_mag_meas=(zp - 2.5 * np.log10(fs)) if fs > 0 else np.nan,
                   x=float(cx), y=float(cy))
        out.append(row)
    return out


def population_sigma(rows: List[Dict], bright_mag: float = 19.5, min_n: int = 5) -> Tuple[float, float]:
    """The other galaxies set the systematic floor: subtraction leaves a
    few percent of each galaxy's light behind (over- or under-), with a
    few percent scatter (median and MAD of excess_frac over the galaxies
    brighter than `bright_mag`, falling back to all). A galaxy's own
    photon noise adds in quadrature, so a faint galaxy's noisy fraction
    does not outrank a bright one's real excess:

        excess_sigma = (excess_frac - median) / sqrt(MAD^2 + (residual_err / galaxy_flux)^2)

    Returns (median, mad)."""
    fr = np.array([r["excess_frac"] for r in rows if r.get("mag", 99) < bright_mag and np.isfinite(r["excess_frac"])])
    if fr.size < min_n:
        fr = np.array([r["excess_frac"] for r in rows if np.isfinite(r["excess_frac"])])
    if fr.size < 3:
        for r in rows:
            r["excess_sigma"] = np.nan
        return float("nan"), float("nan")
    med = float(np.median(fr))
    mad = max(float(1.4826 * np.median(np.abs(fr - med))), 0.005)
    for r in rows:
        if not np.isfinite(r["excess_frac"]) or r["residual_flux"] == 0 and r["excess_frac"] == 0:
            r["excess_sigma"] = np.nan
            continue
        gflux = r["residual_flux"] / r["excess_frac"] if r["excess_frac"] != 0 else np.inf
        noise_frac = r["residual_err"] / gflux if np.isfinite(gflux) and gflux > 0 else 0.0
        r["excess_sigma"] = (r["excess_frac"] - med) / np.sqrt(mad ** 2 + noise_frac ** 2)
    return med, mad
