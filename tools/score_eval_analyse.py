#!/usr/bin/env python3
"""Compare quality-score formulae offline on the feature dumps written by
tools/score_eval_collect.py.

Every candidate produced by the pipeline (min_quality=0) is labelled:
  afterglow   within 3" of a GCN afterglow position (plain run)
  inj_const   within 3" of an injected constant source (inject run)
  inj_fade    within 3" of an injected fading source (inject run)
  neg         everything else in the plain run
and scored with several formulae computed from the same features:
  q_cur     the shipped quality_score (variability floor off)
  q_floor   the shipped score with new_source_variability_floor on
  q_chi2    shipped score with the mag_range^2 term replaced by a clipped
            reduced chi^2 (variability significance)
  llr       hand-built log-likelihood-ratio evidence (no fitting):
            persistence (binomial detections vs. expected from MAGLIMIT),
            morphology, position scatter, catalogue completeness, variability
            and fading-trend significance
  p_llr     sigmoid(llr - 7): the same thing as a probability
  lr_lofo   logistic regression on the features, leave-one-field-out
  gb_lofo   gradient boosting on the features, leave-one-field-out
  q_cal     logistic calibration of ln q_cur (same ranking, a probability)

Outputs (in --out): dataset.ecsv, metrics.json, report.md, figures.
"""
import argparse
import json
import math
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
from astropy.table import Table

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pyrt_transient.config_trans import DetectionConfig  # noqa: E402
from pyrt_transient.core.scoring import compute_lightcurve_score_factor  # noqa: E402

MATCH_R_ARCSEC = 3.0
PROD_MIN_QUALITY = 0.2
CONFIG_FLOOR = {"default": False, "vetting": True}

SCORES = ["q_cur", "q_floor", "q_chi2", "llr", "p_bayes", "lr_lofo", "lr_lofo_noshape", "gb_lofo"]
PROB_SCORES = ["p_llr", "p_bayes", "llr_cal", "lr_lofo", "lr_lofo_noshape", "gb_lofo", "q_cal"]
FEATURES_NOSHAPE = ["ln_ndet", "frac_det", "llr_persist", "ln_zsnr", "fr_dev", "ellip", "flag_frac",
                    "ln_nearest", "pos_ratio", "margin", "is_new"]
# Mixture prior per field: one transient against ~600 catalogue-depth stars
# (times the incompleteness eps(m) for being uncatalogued) and ~100 noise
# clusters. Hand-set, not fitted.
PRIOR_STARS, PRIOR_NOISE = 600.0, 100.0
# Fraction of bright (< 18 mag) real stars missing from the reference set:
# measured 9% for the historical Gaia calibrator query (FUTURE_IDEAS.md,
# "Constant new sources"), ~1% for gaia_full + ATLAS with a 3" floor.
CAT_INCOMPLETENESS = {"default": 0.09, "vetting": 0.01}
FEATURES_ML = ["ln_ndet", "frac_det", "llr_persist", "ln_zsnr", "z_var", "t_fade", "fr_dev",
               "ellip", "flag_frac", "ln_nearest", "pos_ratio", "margin", "is_new", "ln_chi2r"]


# ----------------------------------------------------------------------------
# helpers

def fv(row, name, default=np.nan):
    if name not in row.colnames:
        return default
    v = row[name]
    if np.ma.is_masked(v):
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def farr(tab, name):
    if name not in tab.colnames:
        return np.full(len(tab), np.nan)
    return np.ma.filled(np.ma.asarray(tab[name], dtype=float), np.nan)


def sep_arcsec(ra1, dec1, ra2, dec2):
    ra1, dec1, ra2, dec2 = map(np.radians, (ra1, dec1, ra2, dec2))
    c = np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2)
    return np.degrees(np.arccos(np.clip(c, -1, 1))) * 3600.0


def wlinfit(x, y, e):
    """Weighted least squares y = a + b x; returns b, sigma_b."""
    w = 1.0 / e ** 2
    sw = w.sum()
    xm = (w * x).sum() / sw
    ym = (w * y).sum() / sw
    sxx = (w * (x - xm) ** 2).sum()
    if sxx <= 0:
        return 0.0, np.inf
    b = (w * (x - xm) * (y - ym)).sum() / sxx
    return b, math.sqrt(1.0 / sxx)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


# ----------------------------------------------------------------------------
# feature extraction

def cand_features(row, lc, epochs_k, t0, magerr_floor=0.0):
    k = len(epochs_k)
    mags = farr(lc, "MAG_CALIB")
    errs = farr(lc, "MAGERR_CALIB")
    if magerr_floor and np.isfinite(magerr_floor):
        errs = np.sqrt(errs ** 2 + magerr_floor ** 2)
    times = farr(lc, "obs_time")
    eids = np.asarray(np.ma.filled(np.ma.asarray(lc["epoch_id"]), -1), dtype=int)
    fwhm = farr(lc, "FWHM_IMAGE")
    ellip = farr(lc, "ELLIPTICITY")
    flags = np.asarray(np.ma.filled(np.ma.asarray(lc["FLAGS"]), 0), dtype=int) if "FLAGS" in lc.colnames else np.zeros(len(lc), int)

    good = np.isfinite(mags) & np.isfinite(errs) & (errs > 0)
    n_det = len(lc)
    f = {"n_det": n_det, "k": k, "frac_det": n_det / max(k, 1), "ln_ndet": math.log(max(n_det, 1))}

    if good.sum() >= 1:
        m_, e_, t_ = mags[good], errs[good], times[good]
        w = 1.0 / e_ ** 2
        m = float((w * m_).sum() / w.sum())
        chi2 = float((((m_ - m) / e_) ** 2).sum())
        dof = max(good.sum() - 1, 0)
        chi2r = chi2 / dof if dof > 0 else 0.0
        z_var = (chi2 - dof) / math.sqrt(2 * dof) if dof > 0 else 0.0
        snr = 1.0857 / e_
        z_snr = float(math.sqrt((snr ** 2).sum()))
        x = np.log10(np.maximum(t_ - t0, 1.0))
        if good.sum() >= 3 and np.ptp(x) > 0:
            b, sb = wlinfit(x, m_, e_)
            t_fade = b / sb if np.isfinite(sb) and sb > 0 else 0.0
        else:
            b, t_fade = 0.0, 0.0
        mag_range = float(m_.max() - m_.min())
    else:
        m, chi2, dof, chi2r, z_var, z_snr, b, t_fade, mag_range = np.nan, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    f.update({"mag": m, "chi2": chi2, "dof": dof, "chi2r": chi2r, "z_var": z_var, "z_snr": z_snr,
              "ln_zsnr": math.log(max(z_snr, 1e-3)), "slope_logt": b, "t_fade": float(np.clip(t_fade, -50, 50)),
              "mag_range": mag_range, "ln_chi2r": math.log(max(chi2r, 1e-3))})

    # persistence: observed detections vs. expected from each epoch's limit
    det = np.zeros(k, bool)
    det[eids[(eids >= 0) & (eids < k)]] = True
    llr_p, n_exp = 0.0, 0.0
    maglims = []
    for i, ep in enumerate(epochs_k):
        lim = ep.get("MAGLIMIT", ep.get("MAGLIM", np.nan))
        if lim is None or not np.isfinite(lim):
            lim = 19.0
        maglims.append(lim)
        p = 1.0 / (1.0 + math.exp(np.clip((m - lim) / 0.3, -50, 50))) if np.isfinite(m) else 0.5
        p = min(max(p, 0.02), 0.98)
        field = ep.get("FIELD", 0.24) or 0.24
        p_bg = min(0.5, ep.get("n_rows", 500) * math.pi * (2.0 / 3600.0) ** 2 / (field ** 2))
        p_bg = max(p_bg, 1e-4)
        n_exp += p
        if det[i]:
            llr_p += math.log(p / p_bg)
        else:
            llr_p += math.log((1 - p) / (1 - p_bg))
    f.update({"llr_persist": llr_p, "n_expected": n_exp, "margin": (m - float(np.median(maglims))) if np.isfinite(m) else 0.0})

    # morphology per epoch relative to that epoch's frame median FWHM
    frs = []
    for i, e in enumerate(eids):
        if 0 <= e < k and np.isfinite(fwhm[i]):
            med = epochs_k[e].get("median_fwhm", np.nan)
            if med and np.isfinite(med) and med > 0:
                frs.append(fwhm[i] / med)
    fr_med = float(np.median(frs)) if frs else fv(row, "fwhm_ratio", 1.0)
    f.update({"fr_med": fr_med, "fr_dev": abs(fr_med - 1.0),
              "ellip": float(np.nanmedian(ellip)) if np.isfinite(ellip).any() else 0.0,
              "flag_frac": float(np.mean(flags > 0)) if len(flags) else 0.0})

    # position scatter relative to the frames' astrometric residual
    astsig = [ep.get("ASTSIGMA", np.nan) for ep in epochs_k]
    astsig = [a for a in astsig if a is not None and np.isfinite(a) and a > 0]
    astsig = float(np.median(astsig)) if astsig else 0.5
    scatter = fv(row, "position_scatter_arcsec", 0.0)
    f.update({"scatter": scatter, "astsig": astsig, "pos_ratio": scatter / max(astsig, 0.3)})

    ctype = str(row["candidate_type"]) if "candidate_type" in row.colnames else "unknown"
    nearest = fv(row, "nearest_source_dist", 100.0)
    if not np.isfinite(nearest):
        nearest = 100.0
    f.update({"ctype": ctype, "is_new": 1.0 if ctype == "new" else 0.0,
              "dmag": fv(row, "magnitude_difference", 0.0), "nearest": nearest,
              "ln_nearest": math.log(max(min(nearest, 100.0), 0.05)),
              "q_pipe": fv(row, "quality_score", np.nan),
              "mag_wmean_row": fv(row, "mag_weighted_mean", np.nan),
              "mag_range_row": fv(row, "mag_range", np.nan),
              "n_det_row": fv(row, "n_detections", np.nan)})
    return f


# ----------------------------------------------------------------------------
# scores

def _weights(floor):
    w = DetectionConfig()
    w.new_source_variability_floor = floor
    return w


W_NOFLOOR, W_FLOOR = _weights(False), _weights(True)


def hand_scores(f, config_name):
    """The shipped score, its two variants, and the hand-built LLR."""
    rowfeat = {"weighted_mean_mag": f["mag_wmean_row"], "mag_range": f["mag_range_row"],
               "n_detections": f["n_det_row"], "candidate_type": f["ctype"]}
    if not np.isfinite(rowfeat["weighted_mean_mag"]):
        rowfeat.pop("weighted_mean_mag")
    fac_used = compute_lightcurve_score_factor(rowfeat, W_FLOOR if CONFIG_FLOOR[config_name] else W_NOFLOOR)
    base = f["q_pipe"] / fac_used if fac_used > 0 and np.isfinite(f["q_pipe"]) else np.nan
    q_cur = base * compute_lightcurve_score_factor(rowfeat, W_NOFLOOR)
    q_floor = base * compute_lightcurve_score_factor(rowfeat, W_FLOOR)

    m = f["mag"] if np.isfinite(f["mag"]) else 18.0
    brightness = math.exp(-(m - 15.0) / 3.0)
    n_factor = math.sqrt(max(f["n_det"], 1) / max(W_NOFLOOR.min_n_detections, 1))
    q_chi2 = base * brightness * W_NOFLOOR.lc_shape_weight * n_factor * float(np.clip(f["chi2r"], 1.0, 10.0))

    l_morph = max(-0.5 * ((f["fr_med"] - 1.0) / 0.15) ** 2 + 2.0, -8.0) - 2.0 * f["flag_frac"]
    l_pos = max(0.5 - 0.5 * (f["pos_ratio"] / 1.5) ** 2, -8.0)
    l_real = f["llr_persist"] + l_morph + l_pos
    eps0 = CAT_INCOMPLETENESS.get(config_name, 0.05)
    eps = eps0 + (1.0 - eps0) / (1.0 + math.exp(-(m - 20.0) / 0.4))
    if f["ctype"] == "new":
        l_cat = -math.log(eps)
    else:
        l_cat = min(0.5 * (f["dmag"] / 0.2) ** 2, 10.0) if np.isfinite(f["dmag"]) else 0.0
    l_var = min(0.5 * max(f["chi2"] - f["dof"], 0.0), 20.0)
    t = f["t_fade"]
    l_trend = min(0.5 * t * t, 20.0) if t > 0 else min(0.25 * t * t, 10.0)
    l_trans = l_cat + l_var + l_trend
    llr = l_real + l_trans
    # Product form: a candidate must be real AND transient. The four
    # constants (offsets 10, 3; temperatures 4, 2) are hand-set, not fitted.
    p_prod = float(sigmoid((l_real - 10.0) / 4.0) * sigmoid((l_trans - 3.0) / 2.0))
    return {"l_trans": l_trans, "p_prod": p_prod, "eps": eps, "base": base, "q_cur": q_cur, "q_floor": q_floor, "q_chi2": q_chi2,
            "l_real": l_real, "l_persist": f["llr_persist"], "l_morph": l_morph, "l_pos": l_pos,
            "l_cat": l_cat, "l_var": l_var, "l_trend": l_trend, "llr": llr, "p_llr": float(sigmoid(llr - 7.0))}


def add_mixture_posterior(tab):
    """p_bayes: posterior probability of the transient hypothesis against
    (a) an uncatalogued constant star and (b) a noise cluster.
      log L_T = l_real + l_lc            (real, point-like, variable/fading)
      log L_S = l_real + ln eps(m)       (real, point-like, constant, missed by the catalogue)
      log L_N = 0                        (the l_* terms are LLRs against noise)
    with l_lc the variability + fading evidence after inflating the errors
    so that the field's typical constant source has reduced chi^2 = 1 (the
    negatives show a median reduced chi^2 of ~2, i.e. underestimated
    MAGERR_CALIB, which would otherwise count as variability)."""
    keys = np.array([f"{a}|{b}|{c}" for a, b, c in zip(tab["field"], tab["config"], tab["k"])])
    chi2r = farr(tab, "chi2r")
    is_neg = np.asarray(tab["label"], int) == 0
    ref = {}
    for key in np.unique(keys):
        m = (keys == key) & is_neg & np.isfinite(chi2r) & (np.asarray(tab["dof"], int) >= 2)
        ref[key] = max(float(np.median(chi2r[m])), 1.0) if m.sum() >= 3 else 1.5
    infl = np.array([ref[k] for k in keys])
    chi2 = farr(tab, "chi2") / infl
    dof = farr(tab, "dof")
    l_var = np.clip(0.5 * np.maximum(chi2 - dof, 0.0), 0, 20)
    t = farr(tab, "t_fade") / np.sqrt(infl)
    l_trend = np.where(t > 0, np.clip(0.5 * t * t, 0, 20), np.clip(0.25 * t * t, 0, 10))
    l_lc = l_var + l_trend
    l_real = farr(tab, "l_real")
    ln_eps = np.log(farr(tab, "eps"))
    is_new = farr(tab, "is_new") > 0.5
    # A matched (brightening/fading-typed) candidate has a catalogue counterpart:
    # the "uncatalogued star" alternative does not apply, the catalogue Delta-m
    # evidence (l_cat) does.
    l_cat = farr(tab, "l_cat")
    log_t = l_real + l_lc + np.where(is_new, 0.0, l_cat)
    log_s = l_real + np.where(is_new, ln_eps, 0.0) + np.log(PRIOR_STARS)
    log_n = np.full(len(tab), np.log(PRIOR_NOISE))
    mx = np.maximum.reduce([log_t, log_s, log_n])
    p = np.exp(log_t - mx) / (np.exp(log_t - mx) + np.exp(log_s - mx) + np.exp(log_n - mx))
    tab["chi2_infl"] = infl
    tab["l_lc_infl"] = l_lc
    tab["p_bayes"] = p


# ----------------------------------------------------------------------------
# dataset assembly

def load_pickle(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def label_candidates(cands, targets, radius):
    """Return (label_id, sep) per candidate for nearest target within radius."""
    n = len(cands)
    ids = [""] * n
    seps = np.full(n, np.nan)
    if n == 0 or not targets:
        return ids, seps
    ra = farr(cands, "ALPHA_J2000")
    dec = farr(cands, "DELTA_J2000")
    for t in targets:
        s = sep_arcsec(ra, dec, t["ra"], t["dec"])
        ok = np.isfinite(s) & (s < radius)
        if ok.any():
            j = int(np.argmin(np.where(ok, s, np.inf)))
            if not ids[j] or s[j] < seps[j]:
                ids[j] = t["id"]
                seps[j] = s[j]
    return ids, seps


def build_records(eval_dir: Path, configs, ks_wanted=None, floors=None):
    records = []
    floors = floors or {}
    for fdir in sorted(p for p in eval_dir.iterdir() if p.is_dir()):
        field = fdir.name
        epochs = json.loads((fdir / "epochs.json").read_text())
        targets = json.loads((fdir / "targets.json").read_text())
        real_targets = [t for t in targets if "late" not in t["id"]]
        t0 = epochs[0]["mid_time"] - 30.0
        magerr_floor = float(floors.get(field, 0.0) or 0.0)
        for cfg in configs:
            for run in ("plain", "inject"):
                rdir = fdir / cfg / run
                if not rdir.exists():
                    continue
                pkls = sorted(rdir.glob("k_*.pkl"))
                if not pkls:
                    continue
                kmax = int(pkls[-1].stem.split("_")[1])
                inj = {}
                if run == "inject":
                    sources = json.loads((rdir / "sources.json").read_text())
                    truth = Table.read(rdir / "truth.ecsv", format="ascii.ecsv")
                    for s in sources:
                        tr = truth[truth["source_id"] == s["source_id"]]
                        inj[s["source_id"]] = dict(s, n_epochs_detected=int(np.sum(tr["detected"])),
                                                   mag_peak=float(np.nanmin(farr(tr, "mag_obs"))) if len(tr) else np.nan)
                    tg = [{"id": s["source_id"], "ra": s["ra"], "dec": s["dec"]} for s in sources]
                else:
                    tg = real_targets
                for pkl in pkls:
                    k = int(pkl.stem.split("_")[1])
                    if ks_wanted is not None and k not in ks_wanted and k != kmax:
                        continue
                    d = load_pickle(pkl)
                    cands, lcs = d["candidates"], d["lightcurves"]
                    if len(cands) == 0:
                        continue
                    ids, seps = label_candidates(cands, tg, MATCH_R_ARCSEC)
                    for j, row in enumerate(cands):
                        tid = str(row["transient_id"])
                        lc = lcs.get(tid)
                        if lc is None or len(lc) == 0:
                            continue
                        if run == "inject" and not ids[j]:
                            continue  # negatives come from the plain run only
                        f = cand_features(row, lc, epochs[:k], t0, magerr_floor=magerr_floor)
                        f["magerr_floor"] = magerr_floor
                        f.update(hand_scores(f, cfg))
                        if run == "plain":
                            kind = "afterglow" if ids[j] else "neg"
                            extra = {}
                        else:
                            s = inj[ids[j]]
                            kind = "inj_const" if s["alpha"] == 0 else "inj_fade"
                            extra = {"inj_m0": s["m0"], "inj_alpha": s["alpha"],
                                     "inj_n_det_true": s["n_epochs_detected"], "inj_mag_peak": s["mag_peak"]}
                        rec = {"field": field, "config": cfg, "run": run, "k": k, "kmax": kmax,
                               "is_final": k == kmax, "cand_id": tid, "target": ids[j], "sep": seps[j],
                               "kind": kind, "label": 0 if kind == "neg" else 1,
                               "ra": fv(row, "ALPHA_J2000"), "dec": fv(row, "DELTA_J2000"),
                               "inj_m0": np.nan, "inj_alpha": np.nan, "inj_n_det_true": -1, "inj_mag_peak": np.nan}
                        rec.update(extra)
                        rec.update(f)
                        records.append(rec)
    return Table(rows=records)


# ----------------------------------------------------------------------------
# learned scores, leave-one-field-out

def add_learned_scores(tab):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    for name in ("lr_lofo", "lr_lofo_noshape", "gb_lofo", "q_cal", "llr_cal"):
        tab[name] = np.full(len(tab), np.nan)
    X_all = np.column_stack([farr(tab, c) for c in FEATURES_ML])
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=50.0, neginf=-50.0)
    X_ns = np.column_stack([farr(tab, c) for c in FEATURES_NOSHAPE])
    X_ns = np.nan_to_num(X_ns, nan=0.0, posinf=50.0, neginf=-50.0)
    lnq = np.log(np.clip(np.nan_to_num(farr(tab, "q_cur"), nan=1e-6), 1e-6, None))[:, None]
    llr1 = np.nan_to_num(farr(tab, "llr"), nan=-50.0)[:, None]
    y = np.asarray(tab["label"], dtype=int)
    fields = np.asarray(tab["field"]).astype(str)
    configs = np.asarray(tab["config"]).astype(str)
    final = np.asarray(tab["is_final"], dtype=bool)
    for cfg in np.unique(configs):
        cmask = configs == cfg
        for fld in np.unique(fields[cmask]):
            test = cmask & (fields == fld)
            train = cmask & (fields != fld) & final
            if y[train].sum() == 0 or (1 - y[train]).sum() == 0:
                continue
            lr = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"))
            lr.fit(X_all[train], y[train])
            tab["lr_lofo"][test] = lr.predict_proba(X_all[test])[:, 1]
            lr2 = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"))
            lr2.fit(X_ns[train], y[train])
            tab["lr_lofo_noshape"][test] = lr2.predict_proba(X_ns[test])[:, 1]
            gb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                                class_weight="balanced", random_state=0)
            gb.fit(X_all[train], y[train])
            tab["gb_lofo"][test] = gb.predict_proba(X_all[test])[:, 1]
            cal = LogisticRegression(max_iter=2000)
            cal.fit(lnq[train], y[train])
            tab["q_cal"][test] = cal.predict_proba(lnq[test])[:, 1]
            cal2 = LogisticRegression(max_iter=2000)
            cal2.fit(llr1[train], y[train])
            tab["llr_cal"][test] = cal2.predict_proba(llr1[test])[:, 1]
    # A single LR fit on everything, reported for its coefficients only.
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"))
    lr.fit(X_all[final], y[final])
    coefs = dict(zip(FEATURES_ML, [float(c) for c in lr[-1].coef_[0]]))
    return coefs


# ----------------------------------------------------------------------------
# metrics

def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    pos = np.asarray(pos, float)
    neg = np.asarray(neg, float)
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return (gt + 0.5 * eq) / (len(pos) * len(neg))


def average_precision(scores, labels):
    order = np.argsort(-scores, kind="stable")
    l = labels[order]
    if l.sum() == 0:
        return np.nan
    tp = np.cumsum(l)
    prec = tp / np.arange(1, len(l) + 1)
    return float((prec * l).sum() / l.sum())


def recall_at_fp(scores, labels, n_fp):
    """Recall when the threshold admits exactly n_fp negatives (global)."""
    neg = np.sort(scores[labels == 0])[::-1]
    if len(neg) == 0:
        thr = -np.inf
    elif n_fp >= len(neg):
        thr = -np.inf
    else:
        thr = neg[n_fp]  # admit the n_fp highest negatives, strictly above the next one
        if n_fp > 0 and neg[n_fp - 1] == thr:
            thr = neg[n_fp - 1]
    return float(np.mean(scores[labels == 1] > thr)) if (labels == 1).any() else np.nan, thr


def evaluate(tab, out_dir: Path):
    final = tab[np.asarray(tab["is_final"], bool)]
    results = {}
    lines = []
    for cfg in sorted(set(np.asarray(final["config"]).astype(str))):
        sub = final[np.asarray(final["config"]).astype(str) == cfg]
        y = np.asarray(sub["label"], int)
        kind = np.asarray(sub["kind"]).astype(str)
        fields = np.asarray(sub["field"]).astype(str)
        q = farr(sub, "q_cur")
        n_fp_budget = int(np.sum((y == 0) & (q >= PROD_MIN_QUALITY)))
        res = {"n_neg": int((y == 0).sum()), "n_pos": int((y == 1).sum()),
               "n_afterglow": int((kind == "afterglow").sum()),
               "n_inj_const": int((kind == "inj_const").sum()), "n_inj_fade": int((kind == "inj_fade").sum()),
               "fp_budget_q_cur_0.2": n_fp_budget, "scores": {}}
        lines.append(f"\n## Config `{cfg}` (final epoch count per field)\n")
        lines.append(f"negatives {res['n_neg']}, positives {res['n_pos']} "
                     f"(afterglows {res['n_afterglow']}, injected constant {res['n_inj_const']}, "
                     f"injected fading {res['n_inj_fade']}); negatives passing the production gate q_cur >= 0.2: {n_fp_budget}\n")
        lines.append("| score | AUC pooled | AUC mean/field | AP | recall @ matched FP | recall afterglow | recall inj_const | recall inj_fade | recall @ 0 FP | afterglows ranked #1 |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for sc in SCORES + ["llr_cal", "q_cal"]:
            s = farr(sub, sc)
            s = np.nan_to_num(s, nan=-1e9)
            pooled = auc(s[y == 1], s[y == 0])
            per_field = []
            for fld in np.unique(fields):
                m = fields == fld
                if (y[m] == 1).any() and (y[m] == 0).any():
                    per_field.append(auc(s[m & (y == 1)], s[m & (y == 0)]))
            ap = average_precision(s, y)
            r_matched, thr = recall_at_fp(s, y, n_fp_budget)
            r_kind = {}
            for kd in ("afterglow", "inj_const", "inj_fade"):
                mk = kind == kd
                r_kind[kd] = float(np.mean(s[mk] > thr)) if mk.any() else np.nan
            r0, _ = recall_at_fp(s, y, 0)
            # afterglow rank within its field
            top1 = []
            for i in np.flatnonzero(kind == "afterglow"):
                m = (fields == fields[i]) & (y == 0)
                top1.append(int(np.sum(s[m] > s[i]) == 0))
            r = {"auc_pooled": pooled, "auc_field_mean": float(np.nanmean(per_field)) if per_field else np.nan,
                 "ap": ap, "recall_matched_fp": r_matched, "threshold_matched_fp": float(thr),
                 "recall_by_kind": r_kind, "recall_0fp": r0,
                 "afterglow_top1": (int(np.sum(top1)), len(top1))}
            res["scores"][sc] = r
            lines.append(f"| {sc} | {pooled:.3f} | {r['auc_field_mean']:.3f} | {ap:.3f} | {r_matched:.3f} | "
                         f"{r_kind['afterglow']:.2f} | {r_kind['inj_const']:.2f} | {r_kind['inj_fade']:.2f} | {r0:.3f} | "
                         f"{r['afterglow_top1'][0]}/{r['afterglow_top1'][1]} |")
        # calibration of the probabilistic scores
        lines.append("\n| probabilistic score | Brier | log loss | reliability (bins: predicted -> observed, n) |")
        lines.append("|---|---|---|---|")
        for sc in PROB_SCORES:
            p = farr(sub, sc)
            ok = np.isfinite(p)
            if ok.sum() == 0:
                continue
            p, yy = np.clip(p[ok], 1e-6, 1 - 1e-6), y[ok]
            brier = float(np.mean((p - yy) ** 2))
            ll = float(-np.mean(yy * np.log(p) + (1 - yy) * np.log(1 - p)))
            bins = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0001]
            rel = []
            for a, b in zip(bins[:-1], bins[1:]):
                m = (p >= a) & (p < b)
                if m.any():
                    rel.append(f"{p[m].mean():.2f}->{yy[m].mean():.2f} (n={m.sum()})")
            res["scores"].setdefault(sc, {})
            res["scores"][sc].update({"brier": brier, "logloss": ll, "reliability": rel})
            lines.append(f"| {sc} | {brier:.3f} | {ll:.3f} | {'; '.join(rel)} |")
        results[cfg] = res

        lines.append("\n| feature medians by kind | " + " | ".join(FEATURES_ML) + " |")
        lines.append("|---|" + "---|" * len(FEATURES_ML))
        for kd in ("neg", "afterglow", "inj_const", "inj_fade"):
            mk = kind == kd
            if mk.any():
                lines.append(f"| {kd} (n={mk.sum()}) | " + " | ".join(f"{np.nanmedian(farr(sub, c)[mk]):.2f}" for c in FEATURES_ML) + " |")
        # per-afterglow table
        lines.append("\n| field | target | sep\" | n_det/k | mag | q_cur | rank q_cur | rank q_floor | rank q_chi2 | llr | rank llr | p_bayes | rank p_bayes | rank lr_lofo | rank lr_noshape | rank gb_lofo |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for i in np.flatnonzero(kind == "afterglow"):
            m = (fields == fields[i]) & (y == 0)

            def rank(sc):
                s = np.nan_to_num(farr(sub, sc), nan=-1e9)
                return int(np.sum(s[m] > s[i]) + 1)
            lines.append(f"| {fields[i]} | {sub['target'][i]} | {sub['sep'][i]:.2f} | {int(sub['n_det'][i])}/{int(sub['k'][i])} | "
                         f"{sub['mag'][i]:.2f} | {sub['q_cur'][i]:.3g} | {rank('q_cur')} | {rank('q_floor')} | {rank('q_chi2')} | "
                         f"{sub['llr'][i]:.1f} | {rank('llr')} | {sub['p_bayes'][i]:.3f} | {rank('p_bayes')} | {rank('lr_lofo')} | {rank('lr_lofo_noshape')} | {rank('gb_lofo')} |")
    return results, lines


def latency_tables(tab):
    """For each real afterglow and each score: first k at which it ranks #1
    in its field, and first k at which it passes the matched-FP threshold."""
    lines = ["\n## Latency: real afterglows, epochs needed\n",
             "First epoch count k at which the afterglow is the top-ranked candidate of its field (rank 1), per score.\n",
             "| config | field | k_final | " + " | ".join(SCORES) + " |",
             "|---|---|---|" + "---|" * len(SCORES)]
    out = {}
    fields = np.asarray(tab["field"]).astype(str)
    configs = np.asarray(tab["config"]).astype(str)
    kinds = np.asarray(tab["kind"]).astype(str)
    ks = np.asarray(tab["k"], int)
    y = np.asarray(tab["label"], int)
    for cfg in sorted(set(configs)):
        for fld in sorted(set(fields[(configs == cfg) & (kinds == "afterglow")])):
            m_field = (configs == cfg) & (fields == fld)
            row = {}
            for sc in SCORES:
                s = np.nan_to_num(farr(tab, sc), nan=-1e9)
                first = None
                for k in sorted(set(ks[m_field])):
                    mk = m_field & (ks == k)
                    ia = np.flatnonzero(mk & (kinds == "afterglow"))
                    if len(ia) == 0:
                        continue
                    i = ia[0]
                    n_above = np.sum(s[mk & (y == 0)] > s[i])
                    if n_above == 0 and s[i] > -1e8:
                        first = k
                        break
                row[sc] = first
            kmax = int(ks[m_field].max())
            out[f"{cfg}/{fld}"] = row
            lines.append(f"| {cfg} | {fld} | {kmax} | " + " | ".join(str(row[sc]) if row[sc] else "-" for sc in SCORES) + " |")
    return out, lines


def make_figures(tab, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    final = tab[np.asarray(tab["is_final"], bool)]
    for cfg in sorted(set(np.asarray(final["config"]).astype(str))):
        sub = final[np.asarray(final["config"]).astype(str) == cfg]
        y = np.asarray(sub["label"], int)
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
        for sc in ["q_cur", "q_floor", "q_chi2", "llr", "p_bayes", "lr_lofo", "gb_lofo"]:
            s = np.nan_to_num(farr(sub, sc), nan=-1e9)
            order = np.argsort(-s)
            tp = np.cumsum(y[order] == 1) / max((y == 1).sum(), 1)
            fp = np.cumsum(y[order] == 0)
            axes[0].plot(fp, tp, label=sc)
        axes[0].set_xscale("symlog")
        axes[0].set_xlabel("false positives admitted (all fields)")
        axes[0].set_ylabel("recall of positives")
        axes[0].set_title(f"{cfg}: recall vs. false positives")
        axes[0].legend(fontsize=7)
        kind = np.asarray(sub["kind"]).astype(str)
        for kd, c in (("neg", "0.6"), ("inj_const", "C0"), ("inj_fade", "C1"), ("afterglow", "C3")):
            m = kind == kd
            if m.any():
                axes[1].scatter(np.log10(np.clip(farr(sub, "q_cur")[m], 1e-4, None)), farr(sub, "llr")[m],
                                s=8, c=c, label=kd, alpha=0.7)
        axes[1].axvline(np.log10(PROD_MIN_QUALITY), color="k", ls=":", lw=0.8)
        axes[1].set_xlabel("log10 q_cur")
        axes[1].set_ylabel("llr")
        axes[1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / f"scores_{cfg}.png", dpi=130)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--configs", nargs="+", default=["default", "vetting"])
    ap.add_argument("--reuse", action="store_true", help="reuse --out/dataset.ecsv instead of re-extracting")
    ap.add_argument("--floor-json", type=Path, default=None,
                    help="{field: {floor_bright_median: mag}} additive magnitude-error floor per field")
    ap.add_argument("--floor-key", default="floor_bright_median")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    warnings.simplefilter("ignore")

    if args.reuse and (args.out / "dataset.ecsv").exists():
        tab = Table.read(args.out / "dataset.ecsv", format="ascii.ecsv")
    else:
        floors = None
        if args.floor_json:
            raw = json.loads(args.floor_json.read_text())
            floors = {k: (v.get(args.floor_key) if isinstance(v, dict) else v) for k, v in raw.items()}
        tab = build_records(args.eval_dir, args.configs, floors=floors)
        print(f"{len(tab)} candidate records", flush=True)
    add_mixture_posterior(tab)
    coefs = add_learned_scores(tab)
    tab.write(args.out / "dataset.ecsv", format="ascii.ecsv", overwrite=True)

    results, lines = evaluate(tab, args.out)
    lat, lat_lines = latency_tables(tab)
    results["latency_first_rank1"] = lat
    results["lr_coefficients_all_fields"] = coefs
    lines += lat_lines
    lines.append("\n## Logistic-regression coefficients (standardised features, all fields)\n")
    lines.append("| feature | coefficient |\n|---|---|")
    for k_, v in sorted(coefs.items(), key=lambda kv: -abs(kv[1])):
        lines.append(f"| {k_} | {v:+.2f} |")
    (args.out / "report.md").write_text("# Score comparison\n" + "\n".join(lines) + "\n")
    (args.out / "metrics.json").write_text(json.dumps(results, indent=1, default=float))
    make_figures(tab, args.out)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
