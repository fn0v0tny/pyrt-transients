"""Incremental replay: run the detection strategy on epochs 1..k for every
k and record what the candidate list looked like at each step.

This is what the daemon does in production (every new frame triggers a
full re-clustering over the accumulated campaign), so the k-th snapshot is
exactly the candidate list an observer would have seen after the k-th
frame. From the snapshots come latency (first k at which a target is
reported), and purity (how many reported candidates are neither a target
nor an injected source, as a function of k).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u

from pyrt_transient.catalog import QueryParams
from pyrt_transient.detection.blind_multicatalog import BlindMulticatalogStrategy
from pyrt_transient.extraction_manager import ImageExtractionManager
from pyrt_transient.validation.injection import epoch_mid_time


def filter_epochs_by_pointing(tables: Sequence[Table], max_sep_arcmin: float = 10.0,
                              logger: Optional[logging.Logger] = None):
    """Drop epochs whose frame centre (CTRRA/CTRDEC) is more than
    max_sep_arcmin from the median pointing of the set. A stray frame of
    another field under the same observation ID (GRB 250813B's first
    frame, 7.5 deg away) otherwise becomes the injection reference frame
    and every injected source lands outside the real frames."""
    import numpy as np
    pts = []
    for t in tables:
        m = t.meta or {}
        try:
            pts.append((float(m["CTRRA"]), float(m["CTRDEC"])))
        except (KeyError, TypeError, ValueError):
            pts.append((np.nan, np.nan))
    pts = np.asarray(pts, dtype=float)
    if not np.isfinite(pts).all():
        return list(tables)
    # RA is reduced modulo 360 around a reference epoch before the median and
    # again before the separation: a plain median and a plain difference put a
    # field straddling RA=0 (359.9, 0.1) ~180 deg from its own centre, so every
    # epoch was dropped and the caller got an empty list.
    dec0 = np.median(pts[:, 1])
    ra_ref = pts[0, 0]
    dra_ref = (pts[:, 0] - ra_ref + 180.0) % 360.0 - 180.0
    ra0 = ra_ref + np.median(dra_ref)
    dra = (pts[:, 0] - ra0 + 180.0) % 360.0 - 180.0
    sep = np.hypot(dra * np.cos(np.radians(dec0)), pts[:, 1] - dec0) * 60.0
    keep = sep <= max_sep_arcmin
    if not keep.all():
        dropped = [str((t.meta or {}).get("filename", i)) for i, t in enumerate(tables) if not keep[i]]
        (logger or logging.getLogger("validation.replay")).warning(
            f"dropping {len(dropped)} epoch(s) pointing > {max_sep_arcmin}' from the field: {dropped}")
    return [t for t, k in zip(tables, keep) if k]


def query_params_for(tables: Sequence[Table], mlim: float = 20.0) -> QueryParams:
    """Same field/query box pipeline_magic.py builds."""
    im = ImageExtractionManager(list(tables))
    ra, dec = im.field_center
    field = float(tables[0].meta.get("FIELD", 0.4))
    return QueryParams(ra=ra, dec=dec, width=1.2 * field, height=1.2 * field, mlim=mlim)


def match_targets(candidates: Table, targets: Sequence[Dict], radius_arcsec: float) -> Dict[str, Dict]:
    """For each target {id, ra, dec}: nearest candidate within radius, or None."""
    out = {}
    if len(candidates) == 0:
        return {t["id"]: None for t in targets}
    cand = SkyCoord(
        ra=np.asarray(candidates["ALPHA_J2000"], dtype=float) * u.deg,
        dec=np.asarray(candidates["DELTA_J2000"], dtype=float) * u.deg,
    )
    for t in targets:
        pos = SkyCoord(ra=t["ra"] * u.deg, dec=t["dec"] * u.deg)
        sep = pos.separation(cand).arcsec
        j = int(np.argmin(sep))
        if sep[j] <= radius_arcsec:
            row = candidates[j]
            out[t["id"]] = {
                "index": j,
                "sep_arcsec": float(sep[j]),
                "quality_score": float(row["quality_score"]) if "quality_score" in candidates.colnames else np.nan,
                "n_detections": int(row["n_detections"]) if "n_detections" in candidates.colnames else -1,
                "transient_id": str(row["transient_id"]) if "transient_id" in candidates.colnames else "",
                "candidate_type": str(row["candidate_type"]) if "candidate_type" in candidates.colnames else "",
            }
        else:
            out[t["id"]] = None
    return out


def snapshot(k: int, epoch_table: Table, candidates: Table, targets: Sequence[Dict],
             radius_arcsec: float, elapsed_s: float) -> Dict:
    matches = match_targets(candidates, targets, radius_arcsec)
    matched_idx = {m["index"] for m in matches.values() if m is not None}
    cand_rows = []
    for j, row in enumerate(candidates):
        cand_rows.append({
            "index": j,
            "transient_id": str(row["transient_id"]) if "transient_id" in candidates.colnames else "",
            "ra": float(row["ALPHA_J2000"]), "dec": float(row["DELTA_J2000"]),
            "quality_score": float(row["quality_score"]) if "quality_score" in candidates.colnames else np.nan,
            "n_detections": int(row["n_detections"]) if "n_detections" in candidates.colnames else -1,
            "candidate_type": str(row["candidate_type"]) if "candidate_type" in candidates.colnames else "",
            "matched_target": j in matched_idx,
        })
    return {
        "k": k,
        "epoch_file": str(epoch_table.meta.get("filename", "")),
        "t_mid": epoch_mid_time(epoch_table.meta),
        "n_candidates": int(len(candidates)),
        "n_spurious": int(len(candidates) - len(matched_idx)),
        "targets": matches,
        "candidates": cand_rows,
        "elapsed_s": elapsed_s,
    }


def incremental_replay(
    tables: Sequence[Table],
    config,
    data_dir,
    targets: Sequence[Dict],
    params: Optional[QueryParams] = None,
    match_radius_arcsec: float = 3.0,
    k_values: Optional[Iterable[int]] = None,
    catalog_loader=None,
    logger: Optional[logging.Logger] = None,
    strategy_factory=None,
):
    """Run the strategy on tables[:k] for each k and return
    (snapshots, final_candidates, final_lightcurves).

    catalog_loader: pass one CatalogLoader across realisations so the
    reference catalogues are fetched and pre-computed once per process.
    strategy_factory(data_dir, config) may be given to substitute a test
    double for BlindMulticatalogStrategy.
    """
    logger = logger or logging.getLogger("validation.replay")
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    tables = list(tables)
    params = params or query_params_for(tables)
    if strategy_factory is None:
        strategy = BlindMulticatalogStrategy(data_dir=data_dir, config=config)
    else:
        strategy = strategy_factory(data_dir, config)
    if catalog_loader is not None:
        strategy.catalog_loader = catalog_loader

    ks = list(k_values) if k_values is not None else list(range(1, len(tables) + 1))
    snapshots: List[Dict] = []
    candidates, lightcurves = Table(), {}
    for k in ks:
        t0 = time.monotonic()
        candidates, lightcurves = strategy.run(
            tables[:k], config=config, params=params,
            idlimit=config.detection.idlimit_px,
            radius_check=config.detection.radius_check,
            filter_pattern=config.detection.filter_pattern,
            plot_lightcurves=False,
        )
        elapsed = time.monotonic() - t0
        snap = snapshot(k, tables[k - 1], candidates, targets, match_radius_arcsec, elapsed)
        snapshots.append(snap)
        logger.info(f"k={k}: {snap['n_candidates']} candidates, {snap['n_spurious']} unmatched, {elapsed:.1f}s")
    return snapshots, candidates, lightcurves


def first_recovery(snapshots: Sequence[Dict], target_id: str, min_quality: Optional[float] = None) -> Optional[Dict]:
    """First snapshot in which target_id is matched (and, if given, scores
    at least min_quality). Returns {k, t_mid, quality_score, ...} or None."""
    for snap in snapshots:
        m = snap["targets"].get(target_id)
        if m is None:
            continue
        if min_quality is not None and not (m["quality_score"] >= min_quality):
            continue
        return {"k": snap["k"], "t_mid": snap["t_mid"], **m}
    return None
