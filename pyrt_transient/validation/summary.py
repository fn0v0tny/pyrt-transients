"""Turn injection truth + replay snapshots into the statistics a paper
reports: per-source recovery table, completeness vs. magnitude with
binomial errors, latency distribution, spurious-candidate counts vs. epoch
count. Plotting helpers write PDF figures (matplotlib, Agg backend).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from astropy.table import Table

from pyrt_transient.validation.replay import first_recovery


def recovery_table(sources: Sequence[Dict], truth: Table, snapshots: Sequence[Dict],
                   min_quality: float, realisation: int = 0) -> Table:
    """One row per injected source, joining its light-curve truth with the
    replay outcome. `sources` are InjectedSource.as_dict() records."""
    final = snapshots[-1]
    # Rank of every final candidate by quality score (1 = best), so a
    # recovered source's position in the list an observer would read is
    # recorded alongside its raw score.
    scores = np.array([c["quality_score"] for c in final["candidates"]], dtype=float)
    order = np.argsort(-np.nan_to_num(scores, nan=-np.inf))
    rank_of_index = {int(final["candidates"][j]["index"]): r + 1 for r, j in enumerate(order)}
    rows = []
    for src in sources:
        sid = src["source_id"]
        tr = truth[truth["source_id"] == sid] if len(truth) else Table()
        det = tr[tr["detected"]] if len(tr) else tr
        n_det = int(len(det))
        mag_true = np.asarray(tr["mag_true"], dtype=float) if len(tr) else np.array([np.nan])
        maglim = np.asarray(tr["maglim"], dtype=float) if len(tr) else np.array([np.nan])
        m_peak = float(np.nanmin(mag_true))
        m_last = float(mag_true[-1])
        # Brightest observed magnitude relative to that frame's MAGLIM: the
        # natural x-axis for completeness (>0 means fainter than the limit).
        dm_lim = np.asarray(tr["mag_true"] - tr["maglim"], dtype=float) if len(tr) else np.array([np.nan])
        # Epochs in which the source would survive the per-epoch MAGLIM cut.
        n_above = int(np.sum(mag_true <= 1.1 * maglim))
        fr = first_recovery(snapshots, sid)
        frq = first_recovery(snapshots, sid, min_quality=min_quality)
        fin = final["targets"].get(sid)
        rows.append({
            "realisation": realisation, "source_id": sid,
            "ra": src["ra"], "dec": src["dec"], "m0": src["m0"], "alpha": src["alpha"],
            "mag_peak": m_peak, "mag_last": m_last,
            "dmag_peak_vs_maglim": float(np.nanmin(dm_lim)),
            "dmag_median_vs_maglim": float(np.nanmedian(dm_lim)),
            "n_epochs": int(len(tr)), "n_epochs_detected": n_det,
            "n_epochs_within_maglim_filter": n_above,
            "recovered_final": fin is not None,
            "quality_final": fin["quality_score"] if fin else np.nan,
            "rank_final": rank_of_index.get(fin["index"], -1) if fin else -1,
            "n_candidates_final": final["n_candidates"],
            "sep_final_arcsec": fin["sep_arcsec"] if fin else np.nan,
            "n_detections_final": fin["n_detections"] if fin else 0,
            "k_first": fr["k"] if fr else -1,
            "t_first_s": (fr["t_mid"] - src["t0"]) if fr else np.nan,
            "k_first_minq": frq["k"] if frq else -1,
            "t_first_minq_s": (frq["t_mid"] - src["t0"]) if frq else np.nan,
        })
    return Table(rows=rows)


def wilson_interval(k: int, n: int, z: float = 1.0):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def completeness_by_bin(table: Table, x_col: str, bins: Sequence[float],
                        recovered_col: str = "recovered_final", mask=None) -> Table:
    x = np.asarray(table[x_col], dtype=float)
    rec = np.asarray(table[recovered_col], dtype=bool)
    if mask is not None:
        x, rec = x[mask], rec[mask]
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (x >= lo) & (x < hi)
        n, k = int(sel.sum()), int(rec[sel].sum())
        lo_ci, hi_ci = wilson_interval(k, n)
        rows.append({"x_lo": lo, "x_hi": hi, "x_mid": 0.5 * (lo + hi), "n": n, "n_recovered": k,
                     "fraction": k / n if n else np.nan, "ci_lo": lo_ci, "ci_hi": hi_ci})
    return Table(rows=rows)


def spurious_vs_k(all_snapshots: Sequence[Sequence[Dict]]) -> Table:
    """Median / min / max number of unmatched candidates after k epochs,
    across realisations."""
    by_k: Dict[int, List[int]] = {}
    for snaps in all_snapshots:
        for s in snaps:
            by_k.setdefault(s["k"], []).append(s["n_spurious"])
    rows = [{"k": k, "n_runs": len(v), "median": float(np.median(v)), "min": int(min(v)), "max": int(max(v)),
             "mean": float(np.mean(v))} for k, v in sorted(by_k.items())]
    return Table(rows=rows)


def summarise(recovery: Table, spurious: Table, min_quality: float) -> Dict:
    rec = np.asarray(recovery["recovered_final"], dtype=bool)
    detectable = np.asarray(recovery["n_epochs_detected"]) >= 3
    out = {
        "n_sources": int(len(recovery)),
        "n_recovered": int(rec.sum()),
        "fraction_recovered": float(rec.mean()) if len(rec) else np.nan,
        "n_detectable_3plus_epochs": int(detectable.sum()),
        "fraction_recovered_of_detectable": float(rec[detectable].mean()) if detectable.any() else np.nan,
        "fraction_recovered_undetectable": float(rec[~detectable].mean()) if (~detectable).any() else np.nan,
        "min_quality": min_quality,
    }
    kf = np.asarray(recovery["k_first"])
    tf = np.asarray(recovery["t_first_s"], dtype=float)
    ok = kf > 0
    if ok.any():
        out["latency_epochs"] = {"median": float(np.median(kf[ok])), "p16": float(np.percentile(kf[ok], 16)),
                                 "p84": float(np.percentile(kf[ok], 84)), "max": int(kf[ok].max())}
        out["latency_seconds"] = {"median": float(np.nanmedian(tf[ok])), "p16": float(np.nanpercentile(tf[ok], 16)),
                                  "p84": float(np.nanpercentile(tf[ok], 84)), "max": float(np.nanmax(tf[ok]))}
    if len(spurious):
        out["spurious_final"] = {"k": int(spurious["k"][-1]), "median": float(spurious["median"][-1]),
                                 "min": int(spurious["min"][-1]), "max": int(spurious["max"][-1])}
    return out


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_completeness(comp_all: Table, path, comp_split: Optional[Dict[str, Table]] = None,
                      xlabel=r"$m_\mathrm{peak} - m_\mathrm{lim}$ [mag]"):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    def _draw(t, label, **kw):
        ok = np.asarray(t["n"]) > 0
        x = np.asarray(t["x_mid"])[ok]; y = np.asarray(t["fraction"])[ok]
        lo = np.clip(y - np.asarray(t["ci_lo"])[ok], 0, None); hi = np.clip(np.asarray(t["ci_hi"])[ok] - y, 0, None)
        ax.errorbar(x, y, yerr=[lo, hi], label=label, capsize=2, **kw)
    _draw(comp_all, "all", fmt="o-", color="k", ms=3)
    if comp_split:
        for i, (lab, t) in enumerate(comp_split.items()):
            _draw(t, lab, fmt="s--", ms=2.5, alpha=0.8)
    ax.axvline(0, color="0.6", lw=0.8, ls=":")
    ax.set_xlabel(xlabel); ax.set_ylabel("recovered fraction"); ax.set_ylim(-0.03, 1.03)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_latency(recovery: Table, path, min_quality: float):
    plt = _mpl()
    kf = np.asarray(recovery["k_first"]); tf = np.asarray(recovery["t_first_s"], dtype=float)
    kq = np.asarray(recovery["k_first_minq"])
    ok = kf > 0
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))
    kmax = int(max(kf.max(), kq.max(), 3))
    bins = np.arange(0.5, kmax + 1.5, 1.0)
    axes[0].hist(kf[ok], bins=bins, color="0.3", label="first reported")
    axes[0].set_xlabel("epochs to first candidate"); axes[0].set_ylabel("injected sources")
    axes[0].legend(fontsize=7, frameon=False)
    dm = np.asarray(recovery["dmag_peak_vs_maglim"], dtype=float)
    sc = axes[1].scatter(dm[ok], tf[ok], c=np.asarray(recovery["alpha"])[ok], s=8, cmap="coolwarm")
    axes[1].set_xlabel(r"$m_\mathrm{peak} - m_\mathrm{lim}$ [mag]"); axes[1].set_ylabel("time since $t_0$ to first candidate [s]")
    cb = fig.colorbar(sc, ax=axes[1]); cb.set_label(r"decay index $\alpha$")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_spurious(spurious: Table, path):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    k = np.asarray(spurious["k"])
    ax.fill_between(k, spurious["min"], spurious["max"], color="0.85", label="min–max")
    ax.plot(k, spurious["median"], "k-", label="median")
    ax.set_xlabel("epochs accumulated"); ax.set_ylabel("candidates not matching a target")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def dump_json(obj, path):
    def _default(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return None if np.isnan(o) else float(o)
        if isinstance(o, (np.bool_,)): return bool(o)
        if isinstance(o, np.ndarray): return o.tolist()
        raise TypeError(type(o))
    Path(path).write_text(json.dumps(obj, indent=1, default=_default))
