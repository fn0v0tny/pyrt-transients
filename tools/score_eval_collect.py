#!/usr/bin/env python3
"""Collect full candidate feature vectors and light curves for scoring
experiments (offline comparison of quality-score formulae).

For one field (all `*-df.ecsv` in --raw-dir whose OBSID meta equals
--field), run BlindMulticatalogStrategy with min_quality=0 on the first k
epochs for every k in --ks, and pickle the complete candidate table plus
the per-candidate light curves at each k. Optionally repeat with a
campaign of injected synthetic sources (validation/injection.py) so the
same field also yields labelled positives of known brightness and decay.

Output layout (under --out/<field>/):
  epochs.json                       per-epoch meta (MAGLIM, CTIME, FWHM, ...)
  targets.json                      known real targets for this field
  <config>/plain/k_XXX.pkl          {"k", "candidates": Table, "lightcurves": {id: Table}}
  <config>/inject/k_XXX.pkl         same, injected campaign
  <config>/inject/{sources.json, truth.ecsv}

Configs: "default" = historical catalogues (gaia, usno locally), no vetting
options; "vetting" = gaia_full + atlas@vizier + usno with the README
"Constant sources" options. Both with min_quality=0 and the VSX filter off
(pure positional veto, irrelevant to the score and a VizieR call per k).
"""
import argparse
import json
import logging
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pyrt_transient import PipelineConfig
from pyrt_transient.catalog import setup_catalog_cache
from pyrt_transient.detection.blind_multicatalog import BlindMulticatalogStrategy
from pyrt_transient.detection.blind_multicatalog.catalog_query import CatalogLoader
from pyrt_transient.io.ecsv import open_ecsv_file
from pyrt_transient.validation import injection, replay


def scan_obsid(path: Path, max_lines=400):
    with open(path) as fh:
        for i, line in enumerate(fh):
            if line.startswith("# - {OBSID:"):
                return line.split("OBSID:", 1)[1].strip(" }\n'\"")
            if i > max_lines:
                break
    return None


def field_files(raw_dir: Path, field: str):
    files = []
    for f in sorted(raw_dir.glob("*-df.ecsv")):
        if scan_obsid(f) == field:
            files.append(f)
    return files


def parse_ks(spec: str, n: int):
    ks = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            ks.update(range(int(a), int(b) + 1))
        else:
            ks.add(int(part))
    ks.add(n)
    return sorted(k for k in ks if 1 <= k <= n)


def make_config(name: str):
    cfg = PipelineConfig()
    d = cfg.detection
    d.min_quality = 0.0
    d.vsx_filter_enabled = False
    if name == "default":
        d.catalogs = ["gaia", "usno"]
    elif name == "vetting":
        d.catalogs = ["gaia_full", "atlas@vizier", "usno"]
        d.unphotometered_match_is_new = False
        d.catalog_match_floor_arcsec = {"gaia": 3.0, "atlas": 3.0, "usno": 3.0}
        d.new_source_variability_floor = True
    else:
        raise ValueError(name)
    return cfg


def epoch_meta(tables):
    out = []
    for i, t in enumerate(tables):
        m = t.meta
        fwhm = np.asarray(t["FWHM_IMAGE"], dtype=float) if "FWHM_IMAGE" in t.colnames else np.array([])
        fwhm = fwhm[np.isfinite(fwhm) & (fwhm > 0)]
        row = {"epoch_id": i, "filename": str(m.get("filename", "")), "n_rows": int(len(t)),
               "mid_time": float(injection.epoch_mid_time(m)),
               "median_fwhm": float(np.median(fwhm)) if len(fwhm) else float("nan")}
        for key in ("CTIME", "EXPTIME", "MAGLIM", "MAGLIMIT", "ASTSIGMA", "FILTER", "PHFILTER",
                    "LIMFLX3", "LIMFLX10", "FIELD", "PIXEL", "FWHM", "MAGZERO", "OBSID", "DATE-OBS"):
            v = m.get(key)
            if v is not None:
                try:
                    row[key] = float(v)
                except (TypeError, ValueError):
                    row[key] = str(v)
        out.append(row)
    return out


def run_series(tables, cfg, params, loader, ks, out_dir: Path, log):
    work = out_dir / "obs"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    strategy = BlindMulticatalogStrategy(data_dir=work, config=cfg)
    strategy.catalog_loader = loader
    for k in ks:
        t0 = time.monotonic()
        candidates, lightcurves = strategy.run(
            tables[:k], config=cfg, params=params,
            idlimit=cfg.detection.idlimit_px, radius_check=cfg.detection.radius_check,
            filter_pattern=cfg.detection.filter_pattern, plot_lightcurves=False,
        )
        with open(out_dir / f"k_{k:03d}.pkl", "wb") as fh:
            pickle.dump({"k": k, "candidates": candidates, "lightcurves": lightcurves}, fh)
        log.info(f"{out_dir.parent.name}/{out_dir.name} k={k}: {len(candidates)} candidates, "
                 f"{time.monotonic() - t0:.1f}s")
    shutil.rmtree(work, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--field", required=True, help="OBSID value, e.g. GRB230818A")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fields-json", type=Path, default=None, help="{field: [{id, ra, dec}]} known targets")
    ap.add_argument("--configs", nargs="+", default=["default", "vetting"])
    ap.add_argument("--ks", default="1-10,12,14,16,18,20,25,30,40,50,60")
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--inject", type=int, default=30, help="injected sources per field (0 = none)")
    ap.add_argument("--mag-min", type=float, default=14.0)
    ap.add_argument("--mag-max", type=float, default=20.0)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    ap.add_argument("--seed", type=int, default=20260903)
    args = ap.parse_args()

    out = args.out / args.field
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.WARNING)
    log = logging.getLogger("score_eval_collect")
    log.setLevel(logging.INFO)
    fh = logging.FileHandler(out / "collect.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    log.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    log.addHandler(sh)

    setup_catalog_cache(str(Path.home() / "catalog_cache"))

    files = field_files(args.raw_dir, args.field)
    if args.max_epochs:
        files = files[: args.max_epochs]
    tables = [open_ecsv_file(str(f), verbose=False) for f in files]
    tables = [t for t in tables if t is not None]
    if not tables:
        sys.exit(f"no epochs with OBSID={args.field} under {args.raw_dir}")
    tables.sort(key=lambda t: injection.epoch_mid_time(t.meta))
    tables = replay.filter_epochs_by_pointing(tables, logger=log)
    log.info(f"{args.field}: {len(tables)} epochs")

    targets = []
    if args.fields_json and args.fields_json.exists():
        targets = json.loads(args.fields_json.read_text()).get(args.field, [])
    (out / "targets.json").write_text(json.dumps(targets, indent=1))
    (out / "epochs.json").write_text(json.dumps(epoch_meta(tables), indent=1))

    ks = parse_ks(args.ks, len(tables))
    params = replay.query_params_for(tables)
    loader = CatalogLoader()

    # Injected campaign: same sources for every config.
    inj_tables, sources, truth = None, None, None
    if args.inject > 0:
        rng = np.random.default_rng(args.seed)
        t0 = injection.epoch_mid_time(tables[0].meta) - 30.0
        sources = injection.draw_sources(
            tables[0], args.inject, rng, mag_range=(args.mag_min, args.mag_max),
            alphas=args.alphas, t0=t0, id_prefix="inj",
        )
        inj_tables, truth = injection.inject_campaign(tables, sources, rng)

    for cname in args.configs:
        cfg = make_config(cname)
        cdir = out / cname
        try:
            t_start = time.monotonic()
            run_series(tables, cfg, params, loader, ks, cdir / "plain", log)
            log.info(f"{args.field}/{cname}/plain done in {time.monotonic() - t_start:.0f}s")
        except Exception as exc:
            log.exception(f"{args.field}/{cname}/plain FAILED: {exc!r}")
            continue
        if inj_tables is not None:
            try:
                t_start = time.monotonic()
                (cdir / "inject").mkdir(parents=True, exist_ok=True)
                (cdir / "inject" / "sources.json").write_text(json.dumps([s.as_dict() for s in sources], indent=1))
                truth.write(cdir / "inject" / "truth.ecsv", format="ascii.ecsv", overwrite=True)
                run_series(inj_tables, cfg, params, loader, ks, cdir / "inject", log)
                log.info(f"{args.field}/{cname}/inject done in {time.monotonic() - t_start:.0f}s")
            except Exception as exc:
                log.exception(f"{args.field}/{cname}/inject FAILED: {exc!r}")
    log.info(f"{args.field}: all done")


if __name__ == "__main__":
    main()
