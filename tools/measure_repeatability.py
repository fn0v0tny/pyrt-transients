"""Empirical photometric repeatability of constant stars per field.

For each field: match every epoch to the reference epoch (most detections),
keep unflagged stars present in >= 70% of epochs, and compare the rms of
MAG_CALIB across epochs with the mean MAGERR_CALIB. Fit
    rms^2 = (k * err)^2 + floor^2
per field, and report the median reduced chi^2 of those stars with the
raw errors, with the fitted model, and with a per-epoch zeropoint-jitter
correction (median offset per epoch removed first).
"""
import glob
import sys
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_eval_collect import field_files  # noqa: E402
from pyrt_transient.io.ecsv import open_ecsv_file  # noqa: E402

RAW = Path(sys.argv[1]) if len(sys.argv) > 1 and Path(sys.argv[1]).is_dir() else Path(__file__).resolve().parent.parent / "local_test_output/replay18/raw"
fields = [a for a in sys.argv[1:] if not Path(a).is_dir()] or sorted({open(f).read(20000).split("OBSID: ")[1].split("}")[0]
                                  for f in glob.glob(str(RAW / "*-df.ecsv")) if "OBSID: " in open(f).read(20000)})


def farr(t, c):
    return np.ma.filled(np.ma.asarray(t[c], dtype=float), np.nan)


def fit_model(rms, err):
    """Least squares on variances: rms^2 = a * err^2 + b, a=k^2, b=floor^2 (both >= 0)."""
    A = np.column_stack([err ** 2, np.ones_like(err)])
    y = rms ** 2
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b = max(sol[0], 0.0), max(sol[1], 0.0)
    return np.sqrt(a), np.sqrt(b)


print("| field | epochs | stars | k (err scale) | floor [mag] | chi2r raw | chi2r zp-corrected | chi2r model | chi2r model+zp |")
print("|---|---|---|---|---|---|---|---|---|")
allrows = []
floors = {}
for field in fields:
    files = field_files(RAW, field)
    tabs = [open_ecsv_file(str(f), verbose=False) for f in files]
    tabs = [t for t in tabs if t is not None and len(t) > 10]
    if len(tabs) < 5:
        continue
    ref_i = int(np.argmax([len(t) for t in tabs]))
    ref = tabs[ref_i]
    ref_c = SkyCoord(farr(ref, "ALPHA_J2000") * u.deg, farr(ref, "DELTA_J2000") * u.deg)
    n_ref = len(ref)
    mags = np.full((len(tabs), n_ref), np.nan)
    errs = np.full((len(tabs), n_ref), np.nan)
    flags = np.zeros((len(tabs), n_ref), int)
    for i, t in enumerate(tabs):
        c = SkyCoord(farr(t, "ALPHA_J2000") * u.deg, farr(t, "DELTA_J2000") * u.deg)
        idx, sep, _ = ref_c.match_to_catalog_sky(c)
        ok = sep.arcsec < 1.5
        mags[i, ok] = farr(t, "MAG_CALIB")[idx[ok]]
        errs[i, ok] = farr(t, "MAGERR_CALIB")[idx[ok]]
        flags[i, ok] = np.asarray(t["FLAGS"])[idx[ok]] if "FLAGS" in t.colnames else 0
    good = np.isfinite(mags) & np.isfinite(errs) & (errs > 0) & (flags == 0)
    n_ep = good.sum(axis=0)
    keep = n_ep >= max(3, int(0.7 * len(tabs)))
    if keep.sum() < 10:
        continue
    m = np.where(good, mags, np.nan)[:, keep]
    e = np.where(good, errs, np.nan)[:, keep]
    # per-epoch zeropoint jitter: median offset of each epoch from the star means
    star_mean = np.nanmean(m, axis=0)
    zp = np.nanmedian(m - star_mean[None, :], axis=1)
    m_zp = m - zp[:, None]

    def chi2r(mm, ee):
        w = 1 / ee ** 2
        mu = np.nansum(mm * w, axis=0) / np.nansum(w, axis=0)
        c2 = np.nansum(((mm - mu) / ee) ** 2, axis=0)
        dof = np.sum(np.isfinite(mm), axis=0) - 1
        return c2 / np.maximum(dof, 1)

    rms = np.nanstd(m_zp, axis=0, ddof=1)
    err_mean = np.sqrt(np.nanmean(e ** 2, axis=0))
    mag_mean = np.nanmean(m_zp, axis=0)
    k, floor = fit_model(rms, err_mean)
    e_model = np.sqrt((k * e) ** 2 + floor ** 2)
    r_raw, r_zp = np.nanmedian(chi2r(m, e)), np.nanmedian(chi2r(m_zp, e))
    r_model, r_model_zp = np.nanmedian(chi2r(m, e_model)), np.nanmedian(chi2r(m_zp, e_model))
    bright = err_mean < 0.03
    floor_med = float(np.nanmedian(np.sqrt(np.maximum(rms[bright] ** 2 - err_mean[bright] ** 2, 0)))) if bright.sum() >= 5 else float("nan")
    floors[field] = {"k": float(k), "floor_fit": float(floor), "floor_bright_median": floor_med, "n_bright": int(bright.sum()), "zp_jitter": float(np.std(zp))}
    print(f"| {field} | {len(tabs)} | {keep.sum()} | {k:.2f} | {floor:.3f} (bright-median {floor_med:.3f}, n={bright.sum()}) | {r_raw:.2f} | {r_zp:.2f} | {r_model:.2f} | {r_model_zp:.2f} |")
    # magnitude-binned view
    for lo in range(10, 20):
        sel = (mag_mean >= lo) & (mag_mean < lo + 1)
        if sel.sum() >= 5:
            allrows.append((field, lo, int(sel.sum()), float(np.median(rms[sel])), float(np.median(err_mean[sel])),
                            float(np.std(zp))))
import json
json.dump(floors, open(RAW.parent / "photometric_floors.json", "w"), indent=1)
print()
print("| field | mag bin | n | median rms (zp-corrected) | median MAGERR_CALIB | rms/err | zp jitter rms |")
print("|---|---|---|---|---|---|---|")
for field, lo, n, r, er, zj in allrows:
    print(f"| {field} | {lo}-{lo+1} | {n} | {r:.3f} | {er:.3f} | {r/er:.2f} | {zj:.3f} |")
