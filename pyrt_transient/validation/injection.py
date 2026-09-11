"""Catalogue-level injection of synthetic point sources.

The pipeline consumes per-frame source catalogues, not pixels, so the
cheapest honest injection is at the catalogue level: for every epoch a row
is appended per synthetic source, with

- position from the source's sky coordinates through the epoch's own WCS
  (SIP terms included, read from the ECSV meta), scattered by the frame's
  reported astrometric residual ASTSIGMA;
- MAG_CALIB from a power-law light curve  m(t) = m0 + 2.5 alpha log10(t/t1)
  in time since a trigger t0, scattered by the calibrated magnitude error
  the followup.exposure noise model predicts for that magnitude in that
  frame (with the model's own ~0.04 dex scatter on the error);
- every other column (FWHM, ellipticity, centroid errors, aperture area,
  fluxes ...) copied from a "donor": a real, unflagged detection of the
  same frame whose calibrated magnitude is closest to the injected one, so
  the synthetic row has exactly the shape statistics a real star of that
  brightness has in that frame;
- a detection probability that falls from 1 to 0 across the frame's own
  faintest-detection limit (MAGLIMIT), so a source below the extractor's
  floor is absent from the catalogue as it would be in reality.

What this does NOT test: SExtractor's own detection efficiency, blending
with a neighbour (injected positions are kept clear of real detections by
default), and anything pixel-level (cosmic rays, trails, cutout
morphology). It tests the pipeline's recall and latency *given* a
detection, which is the part of the chain this package owns.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence

import numpy as np
from astropy.coordinates import angular_separation
from astropy.io import fits
from astropy.table import Table, vstack
from astropy.wcs import WCS

from pyrt_transient.core.radii import astrometric_scatter_arcsec
from pyrt_transient.followup.exposure import (
    ExposureModelError,
    ReferenceConditions,
    predict_magerror,
)

# Keys copied from ECSV meta into a FITS header to rebuild the frame WCS.
_WCS_KEY_PREFIXES = (
    "WCSAXES", "CTYPE", "CUNIT", "CRVAL", "CRPIX", "CD1_", "CD2_", "EQUINOX",
    "A_ORDER", "B_ORDER", "A_", "B_", "AP_", "BP_",
    # Zenithal projections (pyrt's refit writes ZPN for FRAM's NF4 camera)
    # are invalid without their PV terms: wcslib refuses to build them.
    "PV",
)

# Scatter (dex) applied to the noise model's predicted magnitude error,
# matching the model's measured rms against real MAGERR_CALIB.
MAGERR_MODEL_SCATTER_DEX = 0.04

# Width (mag) of the logistic detection-probability roll-off around MAGLIMIT.
DETECTION_ROLLOFF_MAG = 0.1


@dataclass
class InjectedSource:
    """One synthetic source: a power-law light curve in time since t0."""
    source_id: str
    ra: float
    dec: float
    m0: float          # magnitude at the reference time t_ref
    alpha: float       # temporal decay index; 0 = constant
    t0: float          # trigger time, unix seconds
    t_ref: float       # time at which m(t) == m0, unix seconds

    def magnitude_at(self, t_unix: float) -> float:
        if self.alpha == 0.0:
            return self.m0
        dt = max(t_unix - self.t0, 1e-3)
        dt_ref = max(self.t_ref - self.t0, 1e-3)
        return self.m0 + 2.5 * self.alpha * np.log10(dt / dt_ref)

    def as_dict(self) -> Dict:
        return asdict(self)


def wcs_from_meta(meta) -> WCS:
    """Rebuild the frame WCS (with SIP distortion) from ECSV meta keys."""
    header = fits.Header()
    for key, value in meta.items():
        k = str(key)
        if any(k.startswith(p) for p in _WCS_KEY_PREFIXES):
            try:
                header[k] = value
            except (ValueError, TypeError):
                continue
    for axis in ("1", "2"):
        n = meta.get(f"NAXIS{axis}", meta.get(f"IMGAXIS{axis}", meta.get("IMAGEW" if axis == "1" else "IMAGEH")))
        if n is not None:
            header[f"NAXIS{axis}"] = int(n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return WCS(header)


def frame_size(meta):
    w = meta.get("NAXIS1", meta.get("IMGAXIS1", meta.get("IMAGEW")))
    h = meta.get("NAXIS2", meta.get("IMGAXIS2", meta.get("IMAGEH")))
    return int(w), int(h)


def epoch_mid_time(meta) -> float:
    """Mid-exposure unix time, the same convention core/epochs.py uses."""
    return float(meta.get("CTIME", 0.0)) + float(meta.get("EXPTIME", 0.0)) / 2.0


def _min_sep_arcsec(ra: float, dec: float, ra_arr, dec_arr) -> float:
    """Smallest great-circle distance in arcsec from (ra, dec) to the array,
    everything in radians. Same formula `SkyCoord.separation` uses, so the
    accept/reject decisions are unchanged."""
    return float(np.degrees(angular_separation(ra, dec, ra_arr, dec_arr).min()) * 3600.0)


def draw_sources(
    reference_table: Table,
    n: int,
    rng: np.random.Generator,
    mag_range=(14.0, 20.0),
    alphas: Sequence[float] = (0.0, 1.0),
    t0: Optional[float] = None,
    t_ref: Optional[float] = None,
    edge_margin_px: float = 40.0,
    min_sep_real_arcsec: float = 10.0,
    min_sep_injected_arcsec: float = 30.0,
    id_prefix: str = "inj",
) -> List[InjectedSource]:
    """Draw n sources at random positions inside the reference frame,
    clear of real detections and of each other, with m0 uniform in
    mag_range and alpha drawn uniformly from `alphas`.

    t_ref defaults to the reference frame's mid-exposure time; t0 defaults
    to 30 s before it (a plausible slew-and-settle delay for a robotic
    telescope) and must be given explicitly when the real trigger time is
    known.
    """
    meta = reference_table.meta
    wcs = wcs_from_meta(meta)
    width, height = frame_size(meta)
    if t_ref is None:
        t_ref = epoch_mid_time(meta)
    if t0 is None:
        t0 = t_ref - 30.0

    real_ra = np.radians(np.asarray(reference_table["ALPHA_J2000"], dtype=float))
    real_dec = np.radians(np.asarray(reference_table["DELTA_J2000"], dtype=float))
    sources: List[InjectedSource] = []
    # Accepted positions as plain radian arrays: a SkyCoord per attempt, plus
    # one rebuilt over the growing accepted list, made the draw quadratic in
    # astropy object construction.
    acc_ra: List[float] = []
    acc_dec: List[float] = []
    attempts = 0
    while len(sources) < n and attempts < 200 * n:
        attempts += 1
        x = rng.uniform(edge_margin_px, width - edge_margin_px)
        y = rng.uniform(edge_margin_px, height - edge_margin_px)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ra, dec = wcs.all_pix2world(x, y, 1)
        ra, dec = float(ra), float(dec)
        ra_r, dec_r = np.radians(ra), np.radians(dec)
        if len(real_ra) and _min_sep_arcsec(ra_r, dec_r, real_ra, real_dec) < min_sep_real_arcsec:
            continue
        if acc_ra and _min_sep_arcsec(ra_r, dec_r, np.asarray(acc_ra), np.asarray(acc_dec)) < min_sep_injected_arcsec:
            continue
        acc_ra.append(ra_r)
        acc_dec.append(dec_r)
        sources.append(InjectedSource(
            source_id=f"{id_prefix}{len(sources):03d}",
            ra=ra, dec=dec,
            m0=float(rng.uniform(*mag_range)),
            alpha=float(rng.choice(list(alphas))),
            t0=float(t0), t_ref=float(t_ref),
        ))
    if len(sources) < n:
        raise RuntimeError(f"could only place {len(sources)}/{n} sources after {attempts} attempts")
    return sources


def detection_probability(mag_obs: float, maglimit: float) -> float:
    return float(1.0 / (1.0 + np.exp((mag_obs - maglimit) / DETECTION_ROLLOFF_MAG)))


def _donor_index(table: Table, mag: float, rng: np.random.Generator, n_pool: int = 5,
                 fwhm_ratio_range=(0.7, 1.4), max_ellipticity: float = 0.3,
                 bright_tolerance: float = 0.5) -> int:
    """Index of a real, unflagged, point-like detection with MAG_CALIB
    close to `mag`. Point-like: FWHM within fwhm_ratio_range of the frame
    median and ellipticity below max_ellipticity, so a galaxy or an
    unflagged blend is never used as the shape template for a point
    source (a donor with fwhm_ratio ~2.5 would give the injected row a
    base score ~0.003 and drop it at the per-epoch quality cut)."""
    n = len(table)
    flags = np.asarray(table["FLAGS"]) if "FLAGS" in table.colnames else np.zeros(n, dtype=int)
    mags = np.asarray(table["MAG_CALIB"], dtype=float)
    ok = (flags == 0) & np.isfinite(mags)
    if "FWHM_IMAGE" in table.colnames:
        fwhm = np.asarray(table["FWHM_IMAGE"], dtype=float)
        med = np.nanmedian(fwhm[fwhm > 0]) if np.any(fwhm > 0) else 0.0
        if med > 0:
            ok &= (fwhm / med >= fwhm_ratio_range[0]) & (fwhm / med <= fwhm_ratio_range[1])
    if "ELLIPTICITY" in table.colnames:
        ok &= np.asarray(table["ELLIPTICITY"], dtype=float) < max_ellipticity
    idx = np.flatnonzero(ok)
    # Bright sources: a real 12-13 mag transient on a 10 s D50 frame is
    # saturated and flagged like the 12-13 mag stars around it (GRB 210619B:
    # 44% flagged epochs). When no unflagged point-like donor lies within
    # bright_tolerance of the requested magnitude, admit saturated (FLAGS
    # bit 4) donors -- but never blended ones (bit 2) -- so injected bright
    # sources carry the same flags and shapes as real bright sources instead
    # of borrowing the profile of a fainter, perfectly clean star.
    if len(idx) == 0 or np.min(np.abs(mags[idx] - mag)) > bright_tolerance:
        ok_sat = (flags & ~4 == 0) & np.isfinite(mags) & (mags < mag + bright_tolerance)
        if "ELLIPTICITY" in table.colnames:
            ok_sat &= np.asarray(table["ELLIPTICITY"], dtype=float) < max_ellipticity
        idx_sat = np.flatnonzero(ok_sat)
        if len(idx_sat) and (len(idx) == 0 or np.min(np.abs(mags[idx_sat] - mag)) < np.min(np.abs(mags[idx] - mag))):
            idx = idx_sat
    if len(idx) == 0:
        idx = np.flatnonzero(np.isfinite(mags))
    order = idx[np.argsort(np.abs(mags[idx] - mag))]
    pool = order[: min(n_pool, len(order))]
    return int(rng.choice(pool))


def inject_epoch(
    table: Table,
    sources: Sequence[InjectedSource],
    rng: np.random.Generator,
    force_detect: bool = False,
):
    """Return (injected_table, truth_rows) for one epoch.

    truth_rows is a list of dicts, one per source, recording the true and
    observed magnitude, the error, and whether the row was actually added.
    """
    meta = table.meta
    wcs = wcs_from_meta(meta)
    t_mid = epoch_mid_time(meta)
    exptime = float(meta.get("EXPTIME", 0.0))
    maglimit = float(meta.get("MAGLIMIT", meta.get("MAGLIM", 99.0)))
    # Astrometric scatter in arcsec (pyrt writes it in pixels); 0.5" if absent.
    astsigma = astrometric_scatter_arcsec(meta, default=0.5) or 0.5
    try:
        conditions = ReferenceConditions.from_meta(meta)
    except ExposureModelError:
        conditions = None

    number_base = int(np.nanmax(np.asarray(table["NUMBER"], dtype=float))) + 1 if "NUMBER" in table.colnames and len(table) else 0
    new_rows = []
    truth = []
    for i, src in enumerate(sources):
        m_true = src.magnitude_at(t_mid)
        if conditions is not None:
            err_true = predict_magerror(m_true, exptime, conditions)
            err_true *= 10 ** rng.normal(0.0, MAGERR_MODEL_SCATTER_DEX)
        else:
            err_true = 0.05
        err_true = float(max(err_true, 0.001))
        # Noise is Gaussian in flux, not in magnitude: at low S/N a
        # magnitude-space Gaussian badly overstates upward fluctuations.
        snr_true = 1.0857 / err_true
        flux_ratio = 1.0 + rng.normal(0.0, 1.0) / snr_true
        if flux_ratio <= 0:
            m_obs, err, p_det, detected = np.nan, np.nan, 0.0, False
        else:
            m_obs = float(m_true - 2.5 * np.log10(flux_ratio))
            err = float(max(1.0857 / (snr_true * flux_ratio), 0.001))
            p_det = detection_probability(m_obs, maglimit)
            detected = force_detect or (rng.uniform() < p_det)

        # Position scattered by the frame astrometric residual, then to pixels.
        cosd = np.cos(np.radians(src.dec))
        ra = src.ra + rng.normal(0.0, astsigma) / 3600.0 / max(cosd, 1e-3)
        dec = src.dec + rng.normal(0.0, astsigma) / 3600.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x, y = wcs.all_world2pix(ra, dec, 1)

        truth.append({
            "source_id": src.source_id, "epoch_file": str(meta.get("filename", "")),
            "t_mid": t_mid, "t_since_t0": t_mid - src.t0,
            "mag_true": float(m_true), "mag_obs": m_obs, "magerr": err, "magerr_true": err_true,
            "maglim": float(meta.get("MAGLIM", np.nan)), "maglimit": maglimit,
            "p_detect": p_det, "detected": bool(detected),
        })
        if not detected:
            continue

        donor = table[_donor_index(table, m_obs, rng)]
        row = {c: donor[c] for c in table.colnames}
        dm = m_obs - float(donor["MAG_CALIB"])
        row.update({
            "NUMBER": number_base + i,
            "ALPHA_J2000": float(ra), "DELTA_J2000": float(dec),
            "X_IMAGE": float(x), "Y_IMAGE": float(y),
            "MAG_CALIB": m_obs, "MAGERR_CALIB": err,
        })
        for col in ("XCENTER",):
            if col in row: row[col] = float(x)
        for col in ("YCENTER",):
            if col in row: row[col] = float(y)
        for col in ("MAG_SEX", "MAG_AUTO"):
            if col in row and np.isfinite(float(row[col])):
                row[col] = float(row[col]) + dm
        for col in ("MAGERR_SEX", "MAGERR_AUTO"):
            if col in row:
                row[col] = err
        for col in ("FLUX", "FLUX_AUTO"):
            if col in row and np.isfinite(float(row[col])):
                row[col] = float(row[col]) * 10 ** (-0.4 * dm)
        if "FLAGS" in row:
            # Keep the donor's saturation bit (4) and clear the rest. Zeroing
            # FLAGS outright undid _donor_index's deliberate use of saturated
            # donors for bright sources: the saturated star's shape was copied
            # but its flag discarded, so injected bright sources looked cleaner
            # than the real ones they stand in for and bright-end completeness
            # came out too high. The other bits (notably blended, 2) are not
            # carried over -- an injected source is a single point source.
            row["FLAGS"] = int(donor["FLAGS"]) & 4
        new_rows.append(row)

    if not new_rows:
        return table.copy(), truth
    extra = Table(rows=new_rows, names=table.colnames)
    for c in table.colnames:
        extra[c] = extra[c].astype(table[c].dtype)
    out = vstack([table, extra], metadata_conflicts="silent")
    out.meta = dict(table.meta)
    out.meta["N_INJECTED"] = len(new_rows)
    return out, truth


def inject_campaign(tables: Sequence[Table], sources: Sequence[InjectedSource], rng: np.random.Generator, **kw):
    """inject_epoch over every epoch. Returns (tables, truth_table)."""
    out_tables, truth = [], []
    for t in tables:
        tt, rows = inject_epoch(t, sources, rng, **kw)
        out_tables.append(tt)
        truth.extend(rows)
    return out_tables, Table(rows=truth) if truth else Table()
