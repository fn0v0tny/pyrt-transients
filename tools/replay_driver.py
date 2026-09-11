#!/usr/bin/env python3
"""Replay a real field one epoch at a time and report when a known target
first appears as a candidate, how its score grows, and how many other
candidates accumulate (validation/replay.py).

Outputs (in --out): snapshots.json, target_track.ecsv, candidates_vs_k.ecsv,
summary.json, track.pdf.

Example:
  python tools/replay_driver.py tests/210619B --out local_test_output/replay_210619B \\
      --catalogs gaia usno            # target defaults to GRB_RA/GRB_DEC in the meta
"""
import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
from astropy.table import Table

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from pyrt_transient import PipelineConfig
from pyrt_transient.catalog import setup_catalog_cache
from pyrt_transient.validation import replay, summary
from inject_recover import load_epochs, parse_t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("field_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--target", action="append", default=[], help="id:ra,dec (repeatable); default GRB_RA/GRB_DEC")
    ap.add_argument("--t0", default="auto")
    ap.add_argument("--catalogs", nargs="+", default=["gaia", "usno"])
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--k-start", type=int, default=1)
    ap.add_argument("--match-radius", type=float, default=3.0)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.WARNING)
    log = logging.getLogger("replay_driver"); log.setLevel(logging.INFO)
    logging.getLogger().addHandler(logging.FileHandler(args.out / "replay.log"))
    setup_catalog_cache(str(Path.home() / "catalog_cache"))

    if args.config:
        from pyrt_transient.core.config_loader import load_config_with_yaml_support
        config = load_config_with_yaml_support(args.config)
    else:
        config = PipelineConfig()
    config.detection.catalogs = list(args.catalogs)

    tables = load_epochs(args.field_dir, args.max_epochs)
    targets = []
    for spec in args.target:
        tid, coords = spec.split(":")
        ra, dec = (float(v) for v in coords.split(","))
        targets.append({"id": tid, "ra": ra, "dec": dec})
    if not targets:
        m = tables[0].meta
        targets.append({"id": "grb", "ra": float(m["GRB_RA"]), "dec": float(m["GRB_DEC"])})
    t0 = parse_t0(args.t0, tables)
    if t0 is None:
        t0 = replay.epoch_mid_time(tables[0].meta) - 30.0

    work = args.out / "obs"
    shutil.rmtree(work, ignore_errors=True)
    snaps, cands, _ = replay.incremental_replay(
        tables, config, work, targets, match_radius_arcsec=args.match_radius,
        k_values=range(args.k_start, len(tables) + 1), logger=log,
    )
    summary.dump_json(snaps, args.out / "snapshots.json")
    if len(cands):
        cands.write(args.out / "final_candidates.ecsv", format="ascii.ecsv", overwrite=True)

    rows = []
    for s in snaps:
        for t in targets:
            m = s["targets"].get(t["id"])
            rows.append({"k": s["k"], "t_since_t0_s": s["t_mid"] - t0, "target": t["id"],
                         "matched": m is not None,
                         "quality_score": m["quality_score"] if m else np.nan,
                         "n_detections": m["n_detections"] if m else 0,
                         "sep_arcsec": m["sep_arcsec"] if m else np.nan,
                         "n_candidates": s["n_candidates"], "n_spurious": s["n_spurious"]})
    track = Table(rows=rows)
    track.write(args.out / "target_track.ecsv", format="ascii.ecsv", overwrite=True)

    head = {"field_dir": str(args.field_dir), "n_epochs": len(tables), "catalogs": args.catalogs,
            "t0": args.t0, "targets": {}}
    for t in targets:
        fr = replay.first_recovery(snaps, t["id"])
        frq = replay.first_recovery(snaps, t["id"], min_quality=config.detection.min_quality)
        fin = snaps[-1]["targets"].get(t["id"])
        head["targets"][t["id"]] = {
            "k_first": fr["k"] if fr else None, "t_first_s": (fr["t_mid"] - t0) if fr else None,
            "k_first_minq": frq["k"] if frq else None,
            "quality_final": fin["quality_score"] if fin else None,
            "n_detections_final": fin["n_detections"] if fin else None,
            "sep_final_arcsec": fin["sep_arcsec"] if fin else None,
        }
    head["candidates_final"] = snaps[-1]["n_candidates"]
    head["spurious_final"] = snaps[-1]["n_spurious"]
    head["spurious_vs_k"] = [[s["k"], s["n_spurious"]] for s in snaps]
    summary.dump_json(head, args.out / "summary.json")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots(figsize=(3.5, 2.8))
    for t in targets:
        tt = track[track["target"] == t["id"]]
        ax1.plot(tt["k"], tt["quality_score"], "k.-", label=f"{t['id']} score")
    ax1.axhline(config.detection.min_quality, color="0.6", ls=":", lw=0.8)
    ax1.set_xlabel("epochs accumulated"); ax1.set_ylabel("quality score of target"); ax1.set_yscale("log")
    ax2 = ax1.twinx()
    ax2.step([s["k"] for s in snaps], [s["n_spurious"] for s in snaps], where="post", color="C3", lw=1)
    ax2.set_ylabel("other candidates", color="C3")
    fig.tight_layout(); fig.savefig(args.out / "track.pdf"); plt.close(fig)
    print(json.dumps(head, indent=1, default=str))


if __name__ == "__main__":
    main()
