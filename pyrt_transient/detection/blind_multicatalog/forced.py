"""Forced-photometry admission of stack-only candidates.

The stack is one epoch to the clustering, so a source that the stack reaches
but single frames mostly do not can never collect min_n_detections
detections. The GRB 190919B afterglow in tests/190919B was the 20-frame
stack's top candidate and still never became a candidate.

Here every stack-epoch candidate that did not become a final candidate gets
aperture photometry at its position in the frames the stack was built from,
and is admitted when that forced lightcurve shows a persistent source (see
DetectionConfig.stack_forced_*). Forced points are never clustered into new
candidates: they only describe a candidate the stack already found, so this
step cannot feed back into itself.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import astropy.units as u
import astropy.wcs
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table, vstack

from pyrt_transient.config_trans import DetectionConfig
from pyrt_transient.core.scoring import add_score_probability, apply_lightcurve_score_factor
from pyrt_transient.core.timeutil import unix_to_mjd
from pyrt_transient.detection.blind_multicatalog.lightcurve import update_candidate_with_lightcurve_stats
from pyrt_transient.io.naming import get_base_filename

logger = logging.getLogger("detection.forced")

# Bright, clean detections used for a frame's aperture correction.
_APCORR_MAX_MAGERR = 0.05
_APCORR_MIN_STARS = 5


def _sep():
    try:
        import sep
    except ImportError:          # stdpipe's fork
        import sep_x as sep
    return sep


def _frame_wcs(meta) -> astropy.wcs.WCS:
    """The epoch's own WCS (SIP/ZPN kept); RTS2's T-suffixed alternate
    keywords and non-finite values (astropy refuses NaN cards) dropped."""
    from pyrt_transient.core.wcs_meta import wcs_header_from_meta
    return astropy.wcs.WCS(wcs_header_from_meta(meta))


def resolve_fits_path(table: Table, data_dir) -> Optional[Path]:
    """The epoch's FITS: its meta filename as given if that exists, else the
    same basename in data_dir (where pipeline_magic copies every epoch)."""
    name = table.meta.get("filename") or table.meta.get("FITSFILE")
    if not name:
        return None
    path = Path(str(name)).with_suffix(".fits")
    for candidate in ([path] if path.is_absolute() else []) + [Path(data_dir) / path.name, path]:
        if candidate.exists():
            return candidate
    return None


def _aperture_sum(sep, image, bkg, x, y, radius, gain):
    """sep.sum_circle over positions (0-based) far enough from the edges;
    NaN elsewhere."""
    ny, nx = image.shape
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    inside = (np.isfinite(x) & np.isfinite(y) & (x > radius + 1) & (y > radius + 1)
              & (x < nx - radius - 2) & (y < ny - radius - 2))
    flux = np.full(len(x), np.nan)
    fluxerr = np.full(len(x), np.nan)
    if inside.any():
        f, fe, _ = sep.sum_circle(image, x[inside], y[inside], radius,
                                  err=bkg.globalrms, gain=gain)
        flux[inside] = f
        fluxerr[inside] = fe
    return flux, fluxerr


def forced_photometry(table: Table, fits_path, ra, dec, aperture_fwhm: float = 1.0) -> Dict:
    """Aperture photometry at (ra, dec) in one epoch.

    Magnitudes use a zeropoint measured on the frame's own well-measured
    detections (their MAG_CALIB against the same aperture), which puts them
    on the MAG_CALIB scale including the aperture correction. MAGZERO is only
    the fallback when too few such stars are on the chip.
    """
    sep = _sep()
    meta = table.meta
    image = np.ascontiguousarray(fits.getdata(fits_path), dtype=np.float64)
    bkg = sep.Background(image)
    image = image - bkg.back()

    fwhm = float(meta.get("FWHM", np.nan))
    radius = aperture_fwhm * (fwhm if np.isfinite(fwhm) and fwhm > 0 else 3.0)
    gain = float(meta.get("GAIN", 1.0) or 1.0)

    x, y = _frame_wcs(meta).all_world2pix(np.atleast_1d(ra), np.atleast_1d(dec), 0)
    flux, fluxerr = _aperture_sum(sep, image, bkg, x, y, radius, gain)

    zeropoint = float(meta.get("MAGZERO", np.nan))
    cols = {"MAG_CALIB", "MAGERR_CALIB", "X_IMAGE", "Y_IMAGE"}
    if cols <= set(table.colnames):
        magerr = np.asarray(table["MAGERR_CALIB"], dtype=float)
        good = np.isfinite(magerr) & (magerr < _APCORR_MAX_MAGERR)
        if "FLAGS" in table.colnames:
            good &= np.asarray(table["FLAGS"]) == 0
        if good.sum() >= _APCORR_MIN_STARS:
            sflux, _ = _aperture_sum(sep, image, bkg,
                                     np.asarray(table["X_IMAGE"], dtype=float)[good] - 1,
                                     np.asarray(table["Y_IMAGE"], dtype=float)[good] - 1,
                                     radius, gain)
            ok = np.isfinite(sflux) & (sflux > 0)
            if ok.sum() >= _APCORR_MIN_STARS:
                zeropoint = float(np.median(
                    np.asarray(table["MAG_CALIB"], dtype=float)[good][ok] + 2.5 * np.log10(sflux[ok])))

    snr = flux / fluxerr
    with np.errstate(invalid="ignore", divide="ignore"):
        mag = np.where(flux > 0, zeropoint - 2.5 * np.log10(np.where(flux > 0, flux, 1.0)), np.nan)
        magerr = np.where(snr > 0, 1.0857 / snr, np.nan)
    return {"x": x + 1, "y": y + 1, "flux": flux, "fluxerr": fluxerr, "snr": snr,
            "mag": mag, "magerr": magerr, "fwhm": fwhm}


def _frame_stem(name) -> str:
    """'/obs/2026...-df.ecsv' and '2026...-dft.fits' -> '2026...-df'."""
    stem = Path(str(name).strip()).stem
    return stem[:-1] if stem.endswith("-dft") else stem


def stack_input_tables(detection_tables: List[Table]) -> List[Tuple[int, Table]]:
    """(epoch index, table) of the frames the current stack was built from:
    its STACK_INPUTS meta, else every real (non-stack) epoch."""
    stacks = [t for t in detection_tables if t.meta.get("IS_STACK")]
    real = [(i, t) for i, t in enumerate(detection_tables) if not t.meta.get("IS_STACK")]
    names = stacks[-1].meta.get("STACK_INPUTS") if stacks else None
    if isinstance(names, str):
        names = [n for n in names.split(",") if n.strip()]
    if not names:
        return real
    # STACK_INPUTS names the FITS images, but an epoch's meta filename is its
    # ECSV (io/ecsv.py), so frames are matched by stem. Comparing whole names
    # never matched, and every epoch was silently used instead.
    wanted = {_frame_stem(n) for n in names}
    picked = [(i, t) for i, t in real if _frame_stem(t.meta.get("filename", "")) in wanted]
    return picked or real


def admission_metrics(snr, flux, snr_threshold: float) -> Tuple[int, float, float]:
    """(frames measured, fraction at SNR >= snr_threshold, largest single-frame
    share of the positive flux)."""
    snr = np.asarray(snr, dtype=float)
    flux = np.asarray(flux, dtype=float)
    measured = np.isfinite(snr) & np.isfinite(flux)
    n = int(measured.sum())
    if n == 0:
        return 0, 0.0, 1.0
    frac = float(np.mean(snr[measured] >= snr_threshold))
    positive = flux[measured][flux[measured] > 0]
    share = float(positive.max() / positive.sum()) if len(positive) else 1.0
    return n, frac, share


def admit_stack_candidates(
    data_dir,
    detection_tables: List[Table],
    final_candidates: Table,
    lightcurves: Dict,
    position_match_radius: float = 2.0,
    config=None,
    log: Optional[logging.Logger] = None,
) -> Tuple[Table, Dict]:
    """Add the stack-only candidates whose forced lightcurve passes to
    (final_candidates, lightcurves). A no-op without a stack epoch, when
    disabled, or when no input frame's FITS can be found."""
    log = log or logger
    det_cfg = config.detection if config is not None else DetectionConfig()
    if not getattr(det_cfg, "stack_forced_admission", False):
        return final_candidates, lightcurves
    stack_idx = [i for i, t in enumerate(detection_tables) if t.meta.get("IS_STACK")]
    if not stack_idx:
        return final_candidates, lightcurves

    si = stack_idx[-1]
    cand_path = Path(data_dir) / f"{get_base_filename(detection_tables[si], si)}_transients.ecsv"
    if not cand_path.exists():
        return final_candidates, lightcurves
    cands = Table.read(str(cand_path), format="ascii.ecsv")
    if len(cands) == 0:
        return final_candidates, lightcurves

    ra = np.asarray(cands["ALPHA_J2000"], dtype=float)
    dec = np.asarray(cands["DELTA_J2000"], dtype=float)
    keep = np.isfinite(ra) & np.isfinite(dec)
    if len(final_candidates):
        known = SkyCoord(np.asarray(final_candidates["ALPHA_J2000"], dtype=float) * u.deg,
                         np.asarray(final_candidates["DELTA_J2000"], dtype=float) * u.deg)
        _, sep2d, _ = SkyCoord(ra * u.deg, dec * u.deg).match_to_catalog_sky(known)
        keep &= sep2d.arcsec > position_match_radius
    cands = cands[keep]
    if len(cands) == 0:
        return final_candidates, lightcurves
    cands.sort("quality_score", reverse=True)
    cands = cands[: int(det_cfg.stack_forced_max_candidates)]
    ra = np.asarray(cands["ALPHA_J2000"], dtype=float)
    dec = np.asarray(cands["DELTA_J2000"], dtype=float)

    # One pass over the frames: a single read + background per frame covers
    # every position, so the cost barely depends on the number of candidates.
    frames = []
    for epoch_id, table in stack_input_tables(detection_tables):
        fits_path = resolve_fits_path(table, data_dir)
        if fits_path is None:
            continue
        try:
            phot = forced_photometry(table, fits_path, ra, dec,
                                     aperture_fwhm=det_cfg.stack_forced_aperture_fwhm)
        except Exception as e:
            log.debug(f"Forced photometry failed on {fits_path}: {e}")
            continue
        mid = float(table.meta.get("CTIME", 0.0)) + float(table.meta.get("EXPTIME", 0.0)) / 2.0
        frames.append((epoch_id, table, mid, phot))
    if not frames:
        log.info("Forced photometry: no stack input frame found on disk, skipping admission")
        return final_candidates, lightcurves

    snr = np.array([f[3]["snr"] for f in frames]).T        # (candidate, frame)
    flux = np.array([f[3]["flux"] for f in frames]).T
    existing_ids = set(str(t) for t in final_candidates["transient_id"]) if len(final_candidates) else set()

    admitted, new_lightcurves = [], {}
    for k in range(len(cands)):
        n, frac, share = admission_metrics(snr[k], flux[k], det_cfg.stack_forced_snr)
        passes = (n >= det_cfg.stack_forced_min_frames
                  and frac >= det_cfg.stack_forced_min_fraction
                  and share <= det_cfg.stack_forced_max_flux_fraction)
        log.info(f"Forced photometry at {ra[k]:.5f} {dec[k]:.5f}: {n} frames, "
                 f"SNR>={det_cfg.stack_forced_snr:g} in {frac:.0%}, largest single-frame "
                 f"flux share {share:.2f} -> {'admitted' if passes else 'rejected'}")
        if not passes:
            continue

        rows = [(epoch_id, table, mid, phot) for epoch_id, table, mid, phot in frames
                if np.isfinite(phot["snr"][k]) and phot["snr"][k] >= det_cfg.stack_forced_lc_snr]
        if not rows:
            continue
        lc = Table({
            "ALPHA_J2000": np.full(len(rows), ra[k]),
            "DELTA_J2000": np.full(len(rows), dec[k]),
            "X_IMAGE": [p["x"][k] for _, _, _, p in rows],
            "Y_IMAGE": [p["y"][k] for _, _, _, p in rows],
            "MAG_CALIB": [p["mag"][k] for _, _, _, p in rows],
            "MAGERR_CALIB": [p["magerr"][k] for _, _, _, p in rows],
            "FLUX": [p["flux"][k] for _, _, _, p in rows],
            "SNR": [p["snr"][k] for _, _, _, p in rows],
            "FWHM_IMAGE": [p["fwhm"] for _, _, _, p in rows],
            "FLAGS": np.zeros(len(rows), dtype=int),
            "epoch_id": [e for e, _, _, _ in rows],
            "obs_time": [m for _, _, m, _ in rows],
            "mjd": [unix_to_mjd(m) for _, _, m, _ in rows],
            "source_file": [str(t.meta.get("filename", f"epoch_{e}")) for e, t, _, _ in rows],
            "FORCED": np.ones(len(rows), dtype=bool),
        })
        lc.sort("obs_time")

        row = Table(cands[k:k + 1], copy=True)
        update_candidate_with_lightcurve_stats(row, lc, config=config, logger=log)
        apply_lightcurve_score_factor(row, det_cfg)
        transient_id = f"transient_{ra[k]:.3f}_{dec[k]:.3f}"
        if transient_id in existing_ids or transient_id in new_lightcurves:
            continue
        row["transient_id"] = [transient_id]
        row["admission"] = ["stack+forced"]
        row["forced_n_frames"] = [n]
        row["forced_frac_snr"] = [frac]
        row["forced_max_flux_share"] = [share]
        admitted.append(row)
        new_lightcurves[transient_id] = lc

    if not admitted:
        return final_candidates, lightcurves

    new = vstack(admitted, metadata_conflicts="silent")
    add_score_probability(new, det_cfg)
    scores = np.ma.filled(np.ma.asarray(new["quality_score"], dtype=float), np.nan)
    new = new[np.isfinite(scores) & (scores >= det_cfg.min_quality)]
    if len(new) == 0:
        return final_candidates, lightcurves
    log.info(f"Forced photometry admitted {len(new)} stack-only candidate(s)")

    if len(final_candidates):
        final_candidates = final_candidates.copy()
        final_candidates["admission"] = "clustered"
        final_candidates["forced_n_frames"] = 0
        final_candidates["forced_frac_snr"] = np.nan
        final_candidates["forced_max_flux_share"] = np.nan
        merged = vstack([final_candidates, new], metadata_conflicts="silent")
    else:
        merged = new
    merged.sort("quality_score", reverse=True)

    lightcurves = dict(lightcurves)
    kept = set(str(t) for t in new["transient_id"])
    lightcurves.update({k: v for k, v in new_lightcurves.items() if k in kept})
    return merged, lightcurves
