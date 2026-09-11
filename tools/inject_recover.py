#!/usr/bin/env python3
"""Injection-recovery experiment on a real multi-epoch field.

For each realisation: draw N synthetic sources (random position, m0, decay
index), inject them into every epoch's catalogue (validation/injection.py),
replay the blind-multicatalogue strategy one epoch at a time
(validation/replay.py), and record recovery, latency and the number of
candidates that are neither injected nor a known real target.

Outputs (in --out):
  realisation_XX/{sources.json, truth.ecsv, snapshots.json}
  recovery.ecsv      one row per injected source, all realisations
  spurious.ecsv      unmatched candidates vs. epoch count
  summary.json       headline numbers
  completeness.pdf, latency.pdf, spurious.pdf

Example (the shipped GRB 210619B fixture, 10 x 40 sources, ~3 min each):
  python tools/inject_recover.py tests/210619B --out local_test_output/inject \\
      --realisations 10 --sources 40 --catalogs gaia usno
"""
import argparse
import glob
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from astropy.table import Table, vstack

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pyrt_transient import PipelineConfig
from pyrt_transient.catalog import setup_catalog_cache
from pyrt_transient.detection.blind_multicatalog.catalog_query import CatalogLoader
from pyrt_transient.io.ecsv import open_ecsv_file
from pyrt_transient.validation import injection, replay, summary


def load_epochs(field_dir: Path, max_epochs=None):
    files = sorted(
        f for f in glob.glob(str(field_dir / "*.ecsv"))
        if not f.endswith("_transients.ecsv") and not Path(f).name.startswith("transient_")
    )
    if max_epochs:
        files = files[:max_epochs]
    tables = [open_ecsv_file(f, verbose=False) for f in files]
    tables = [t for t in tables if t is not None]
    from pyrt_transient.validation.replay import filter_epochs_by_pointing
    return filter_epochs_by_pointing(tables)


def parse_t0(value, tables):
    if value in (None, "auto"):
        return None
    from astropy.time import Time
    return float(Time(value, scale="utc").unix)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("field_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--realisations", type=int, default=5)
    ap.add_argument("--sources", type=int, default=40)
    ap.add_argument("--mag-min", type=float, default=14.0)
    ap.add_argument("--mag-max", type=float, default=20.0)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.0, 1.0])
    ap.add_argument("--t0", default="auto", help="trigger time, ISO UTC; 'auto' = first frame - 30 s")
    ap.add_argument("--catalogs", nargs="+", default=["gaia", "usno"])
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--k-start", type=int, default=None, help="first epoch count to snapshot (default min_n_detections)")
    ap.add_argument("--match-radius", type=float, default=3.0)
    ap.add_argument("--exclude", action="append", default=[], help="ra,dec of a known real source to exclude from the spurious count (default: GRB_RA/GRB_DEC from the frame meta)")
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--config", default=None, help="YAML pipeline config (detection section is used)")
    ap.add_argument("--keep-work", action="store_true", help="keep per-realisation obs directories")
    ap.add_argument("--resume", action="store_true", help="reuse realisation_XX/ outputs that already exist (re-aggregate without recomputing)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.WARNING)
    log = logging.getLogger("inject_recover")
    log.setLevel(logging.INFO)
    fh = logging.FileHandler(args.out / "inject_recover.log"); fh.setLevel(logging.INFO)
    logging.getLogger().addHandler(fh)

    setup_catalog_cache(str(Path.home() / "catalog_cache"))
    if args.config:
        from pyrt_transient.core.config_loader import load_config_with_yaml_support
        config = load_config_with_yaml_support(args.config)
    else:
        config = PipelineConfig()
    config.detection.catalogs = list(args.catalogs)
    config.base_data_dir = str(args.out)

    tables = load_epochs(args.field_dir, args.max_epochs)
    if not tables:
        sys.exit(f"no epochs found under {args.field_dir}")
    print(f"[inject_recover] {len(tables)} epochs from {args.field_dir}, catalogs={args.catalogs}", flush=True)

    known = []
    for i, ex in enumerate(args.exclude):
        ra, dec = (float(v) for v in ex.split(","))
        known.append({"id": f"known{i}", "ra": ra, "dec": dec})
    meta0 = tables[0].meta
    if not known and meta0.get("GRB_RA") is not None:
        known.append({"id": "grb", "ra": float(meta0["GRB_RA"]), "dec": float(meta0["GRB_DEC"])})

    t0 = parse_t0(args.t0, tables)
    rng = np.random.default_rng(args.seed)
    params = replay.query_params_for(tables)
    loader = CatalogLoader()
    k_start = args.k_start or config.detection.min_n_detections
    ks = list(range(k_start, len(tables) + 1))

    recoveries, all_snaps = [], []
    for r in range(args.realisations):
        t_start = time.monotonic()
        rdir = args.out / f"realisation_{r:02d}"
        work = rdir / "obs"
        done = all((rdir / f).exists() for f in ("sources.json", "truth.ecsv", "snapshots.json"))
        if args.resume and done:
            source_dicts = json.loads((rdir / "sources.json").read_text())
            truth = Table.read(rdir / "truth.ecsv", format="ascii.ecsv")
            snaps = json.loads((rdir / "snapshots.json").read_text())
            # Keep the random stream aligned with a fresh run so later,
            # non-resumed realisations reproduce the same draws.
            _ = injection.draw_sources(tables[0], args.sources, rng, mag_range=(args.mag_min, args.mag_max),
                                       alphas=args.alphas, t0=t0, id_prefix=f"r{r:02d}s")
            _ = injection.inject_campaign(tables, [injection.InjectedSource(**d) for d in source_dicts], rng)
        else:
            if work.exists():
                shutil.rmtree(work)
            rdir.mkdir(parents=True, exist_ok=True)
            sources = injection.draw_sources(
                tables[0], args.sources, rng, mag_range=(args.mag_min, args.mag_max),
                alphas=args.alphas, t0=t0, id_prefix=f"r{r:02d}s",
            )
            inj_tables, truth = injection.inject_campaign(tables, sources, rng)
            targets = [{"id": s.source_id, "ra": s.ra, "dec": s.dec} for s in sources] + known
            snaps, cands, _ = replay.incremental_replay(
                inj_tables, config, work, targets, params=params,
                match_radius_arcsec=args.match_radius, k_values=ks, catalog_loader=loader, logger=log,
            )
            source_dicts = [s.as_dict() for s in sources]
            summary.dump_json(source_dicts, rdir / "sources.json")
            truth.write(rdir / "truth.ecsv", format="ascii.ecsv", overwrite=True)
            summary.dump_json(snaps, rdir / "snapshots.json")
        rec = summary.recovery_table(source_dicts, truth, snaps, config.detection.min_quality, realisation=r)
        recoveries.append(rec)
        all_snaps.append(snaps)
        n_rec = int(np.sum(rec["recovered_final"]))
        print(f"[inject_recover] realisation {r}: {n_rec}/{len(rec)} recovered, "
              f"{snaps[-1]['n_spurious']} unmatched candidates at k={snaps[-1]['k']}, "
              f"{time.monotonic() - t_start:.0f}s", flush=True)
        if not args.keep_work:
            shutil.rmtree(work, ignore_errors=True)

    recovery = vstack(recoveries)
    recovery.write(args.out / "recovery.ecsv", format="ascii.ecsv", overwrite=True)
    spurious = summary.spurious_vs_k(all_snaps)
    spurious.write(args.out / "spurious.ecsv", format="ascii.ecsv", overwrite=True)

    bins = np.arange(-4.0, 3.01, 0.5)
    comp_all = summary.completeness_by_bin(recovery, "dmag_peak_vs_maglim", bins)
    comp_all.write(args.out / "completeness.ecsv", format="ascii.ecsv", overwrite=True)
    split = {}
    for a in sorted(set(np.asarray(recovery["alpha"]))):
        split[rf"$\alpha={a:g}$"] = summary.completeness_by_bin(
            recovery, "dmag_peak_vs_maglim", bins, mask=np.asarray(recovery["alpha"]) == a)
    summary.plot_completeness(comp_all, args.out / "completeness.pdf", split)
    summary.plot_latency(recovery, args.out / "latency.pdf", config.detection.min_quality)
    summary.plot_spurious(spurious, args.out / "spurious.pdf")

    head = summary.summarise(recovery, spurious, config.detection.min_quality)
    head.update({
        "field_dir": str(args.field_dir), "n_epochs": len(tables), "realisations": args.realisations,
        "sources_per_realisation": args.sources, "mag_range": [args.mag_min, args.mag_max],
        "alphas": args.alphas, "t0": args.t0, "catalogs": args.catalogs, "seed": args.seed,
        "known_targets": known,
    })
    summary.dump_json(head, args.out / "summary.json")
    print(json.dumps(head, indent=1, default=str))


if __name__ == "__main__":
    main()
