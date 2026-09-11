#!/usr/bin/env python3
"""Replay every burst of an archive list one epoch at a time and build a
validation summary (JSON + HTML index).

Input list: TSV with columns  name  ra  dec  [t0_utc]  [afterglow_ra  afterglow_dec]
(header optional; '-' for a missing optional value). `ra dec` is the
position the pipeline is scored against with --match-radius (use the
trigger position with the error radius, or the GCN afterglow position with
a few arcsec). Epochs are the ECSV files under --ecsv-root whose OBSID
meta equals the burst name (the "fixed OBSID" convention of the
production replay), in filename order.

Per burst: <out>/<name>/{snapshots.json, target_track.ecsv, summary.json,
track.png}; overall: <out>/summary.json, <out>/summary.ecsv, <out>/index.html.

Example (on lascaux50, staged tree, existing venv):
  ~/bin/pyrt_transient_venv/bin/python tools/replay_archive.py \\
      ~/grb_list_clean.tsv --ecsv-root ~/phdb_fixed_obsid \\
      --out ~/replay_archive/vetting --config ~/replay_vetting.yaml \\
      --match-radius 200 --site-link-root ../../public_html/grb_replay_validation
"""
import argparse
import glob
import html
import json
import logging
import shutil
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from astropy.table import Table

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pyrt_transient import PipelineConfig
from pyrt_transient.catalog import setup_catalog_cache
from pyrt_transient.transients import open_ecsv_file
from pyrt_transient.validation import replay, summary


def read_list(path: Path):
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        p = line.split("\t") if "\t" in line else line.split()
        if p[0].lower() in ("name", "grb"):
            continue
        row = {"name": p[0], "ra": float(p[1]), "dec": float(p[2]), "t0": None, "ag_ra": None, "ag_dec": None}
        if len(p) > 3 and p[3] not in ("-", ""):
            try:
                float(p[3]); row["t0"] = None   # a bare number in col 4 is the old srcdir/obsid layout
            except ValueError:
                row["t0"] = p[3] if "T" in p[3] else None
        if len(p) > 5 and p[4] not in ("-", "") and p[5] not in ("-", ""):
            try:
                row["ag_ra"], row["ag_dec"] = float(p[4]), float(p[5])
            except ValueError:
                pass
        rows.append(row)
    return rows


@lru_cache(maxsize=4)
def _obsid_index(ecsv_root: str):
    """OBSID -> its ECSV files, in filename order.

    Built in one pass over the archive: scanning every header again for
    each burst meant one full scan of the same directory per burst.
    """
    index = {}
    for f in sorted(glob.glob(str(Path(ecsv_root) / "*.ecsv"))):
        base = Path(f).name
        if base.endswith("_transients.ecsv") or base.startswith("transient_") or "lightcurve" in base:
            continue   # derived per-epoch outputs carry the same meta
        with open(f) as fh:
            for line in fh:
                if not line.startswith("#"):
                    break
                if "OBSID" in line:
                    index.setdefault(line.split(":")[-1].strip().strip("}'\""), []).append(f)
                    break
    return index


def epochs_for(name: str, ecsv_root: Path):
    """ECSV files whose OBSID meta equals `name`, sorted by filename."""
    return list(_obsid_index(str(ecsv_root)).get(name, []))


def t0_unix(value, tables):
    if value:
        from astropy.time import Time
        return float(Time(value, scale="utc").unix), "list"
    return replay.epoch_mid_time(tables[0].meta) - 30.0, "first_frame_minus_30s"


def run_burst(row, ecsv_root, out_dir, config, match_radius, k_start, log):
    files = epochs_for(row["name"], ecsv_root)
    if not files:
        return {"name": row["name"], "n_epochs": 0, "error": "no epochs"}
    tables = [t for t in (open_ecsv_file(f, verbose=False) for f in files) if t is not None]
    targets = [{"id": "target", "ra": row["ra"], "dec": row["dec"]}]
    if row["ag_ra"] is not None:
        targets.append({"id": "afterglow", "ra": row["ag_ra"], "dec": row["ag_dec"]})
    t0, t0_source = t0_unix(row["t0"], tables)
    work = out_dir / "obs"
    shutil.rmtree(work, ignore_errors=True)
    tstart = time.monotonic()
    snaps, cands, _ = replay.incremental_replay(
        tables, config, work, targets, match_radius_arcsec=match_radius,
        k_values=range(min(k_start, len(tables)), len(tables) + 1), logger=log)
    elapsed = time.monotonic() - tstart
    summary.dump_json(snaps, out_dir / "snapshots.json")
    if len(cands):
        cands.write(out_dir / "final_candidates.ecsv", format="ascii.ecsv", overwrite=True)
    res = {"name": row["name"], "ra": row["ra"], "dec": row["dec"], "n_epochs": len(tables),
           "t0_source": t0_source, "elapsed_s": elapsed, "candidates_final": snaps[-1]["n_candidates"],
           "spurious_final": snaps[-1]["n_spurious"], "spurious_vs_k": [[s["k"], s["n_spurious"]] for s in snaps],
           "first_frame_after_t0_s": replay.epoch_mid_time(tables[0].meta) - t0}
    for t in targets:
        fr = replay.first_recovery(snaps, t["id"])
        frq = replay.first_recovery(snaps, t["id"], min_quality=config.detection.min_quality)
        fin = snaps[-1]["targets"].get(t["id"])
        res[t["id"]] = {
            "recovered": fin is not None,
            "k_first": fr["k"] if fr else None, "t_first_s": (fr["t_mid"] - t0) if fr else None,
            "k_first_minq": frq["k"] if frq else None,
            "quality_final": fin["quality_score"] if fin else None,
            "n_detections_final": fin["n_detections"] if fin else None,
            "sep_final_arcsec": fin["sep_arcsec"] if fin else None,
            "transient_id": fin["transient_id"] if fin else None,
            "quality_track": [[s["k"], (s["targets"].get(t["id"]) or {}).get("quality_score")] for s in snaps],
        }
    # matched-candidate position, so a trigger-position match can be checked against GCN
    m = snaps[-1]["targets"].get("target")
    if m is not None:
        c = snaps[-1]["candidates"][m["index"]]
        res["matched_candidate"] = {"ra": c["ra"], "dec": c["dec"], "candidate_type": c["candidate_type"]}
    summary.dump_json(res, out_dir / "summary.json")
    _plot_track(res, snaps, out_dir / "track.png", config.detection.min_quality)
    return res


def _plot_track(res, snaps, path, min_quality):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4, 2.8))
    k = [s["k"] for s in snaps]
    q = [v for _, v in res["target"]["quality_track"]]
    ax.plot(k, [v if v else np.nan for v in q], "k.-", label="target score")
    ax.axhline(min_quality, color="0.6", ls=":", lw=0.8)
    ax.set_xlabel("epochs accumulated"); ax.set_ylabel("quality score"); ax.set_yscale("log")
    ax2 = ax.twinx(); ax2.step(k, [s["n_spurious"] for s in snaps], where="post", color="C3", lw=1)
    ax2.set_ylabel("other candidates", color="C3")
    ax.set_title(res["name"], fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def write_index(results, out: Path, label: str, site_link_root: str, config_desc: str):
    rows = []
    for r in results:
        t = r.get("target", {})
        link = f'<a href="{site_link_root}/obs_{r["name"]}/index.html">site</a>' if site_link_root else ""
        mc = r.get("matched_candidate")
        rows.append(
            "<tr><td>{n}</td><td>{ne}</td><td>{rec}</td><td>{k}</td><td>{ts}</td><td>{q}</td><td>{nd}</td>"
            "<td>{sep}</td><td>{pos}</td><td>{nc}</td><td>{sp}</td><td><a href=\"{n}/track.png\">track</a> {link}</td></tr>".format(
                n=html.escape(r["name"]), ne=r.get("n_epochs", 0),
                rec="yes" if t.get("recovered") else ("<i>error</i>" if r.get("error") else "no"),
                k=t.get("k_first") or "", ts=f'{t["t_first_s"]:.0f}' if t.get("t_first_s") is not None else "",
                q=f'{t["quality_final"]:.1f}' if t.get("quality_final") is not None else "",
                nd=t.get("n_detections_final") or "",
                sep=f'{t["sep_final_arcsec"]:.1f}' if t.get("sep_final_arcsec") is not None else "",
                pos=f'{mc["ra"]:.5f} {mc["dec"]:+.5f}' if mc else "",
                nc=r.get("candidates_final", ""), sp=r.get("spurious_final", ""), link=link))
    n_rec = sum(1 for r in results if r.get("target", {}).get("recovered"))
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Archive replay {html.escape(label)}</title>
<style>body{{font-family:sans-serif;margin:2em}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:3px 8px;font-size:90%}}th{{background:#eee}}</style></head>
<body><h1>Archive replay: {html.escape(label)}</h1>
<p>{html.escape(config_desc)}</p>
<p>Recovered {n_rec} of {len(results)} bursts (target within the match radius of the list position; a trigger-position match must be checked against the GCN afterglow position, see the candidate position column).
t<sub>first</sub> is seconds after the trigger where a trigger time was listed, otherwise after (first frame − 30 s).</p>
<table><tr><th>burst</th><th>epochs</th><th>recovered</th><th>k<sub>first</sub></th><th>t<sub>first</sub> [s]</th><th>Q</th><th>n<sub>det</sub></th><th>sep [\"]</th><th>candidate position</th><th>final cand.</th><th>other cand.</th><th></th></tr>
{''.join(rows)}</table>
<p>Generated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}. Full numbers: <a href="summary.json">summary.json</a>, <a href="summary.ecsv">summary.ecsv</a>.</p></body></html>"""
    (out / "index.html").write_text(page)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("list", type=Path)
    ap.add_argument("--ecsv-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--catalogs", nargs="+", default=None)
    ap.add_argument("--match-radius", type=float, default=200.0, help="arcsec; use the trigger error radius with trigger positions")
    ap.add_argument("--k-start", type=int, default=3)
    ap.add_argument("--only", nargs="+", default=None, help="burst names to run")
    ap.add_argument("--label", default=None)
    ap.add_argument("--site-link-root", default=None, help="relative URL to a directory holding obs_<name>/index.html pages")
    ap.add_argument("--catalog-cache", default=str(Path.home() / "catalog_cache"))
    ap.add_argument("--resume", action="store_true", help="skip bursts whose summary.json exists")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.WARNING)
    log = logging.getLogger("replay_archive"); log.setLevel(logging.INFO)
    logging.getLogger().addHandler(logging.FileHandler(args.out / "replay_archive.log"))
    setup_catalog_cache(args.catalog_cache)
    if args.config:
        from pyrt_transient.core.config_loader import load_config_with_yaml_support
        config = load_config_with_yaml_support(args.config)
    else:
        config = PipelineConfig()
    if args.catalogs:
        config.detection.catalogs = list(args.catalogs)
    config.base_data_dir = str(args.out)
    label = args.label or args.out.name
    desc = (f"catalogs={config.detection.catalogs}; unphotometered_match_is_new={config.detection.unphotometered_match_is_new}; "
            f"catalog_match_floor_arcsec={config.detection.catalog_match_floor_arcsec}; "
            f"new_source_variability_floor={config.detection.new_source_variability_floor}; "
            f"min_quality={config.detection.min_quality}; match_radius={args.match_radius}\"")

    results = []
    for row in read_list(args.list):
        if args.only and row["name"] not in args.only:
            continue
        bdir = args.out / row["name"]; bdir.mkdir(exist_ok=True)
        if args.resume and (bdir / "summary.json").exists():
            results.append(json.loads((bdir / "summary.json").read_text()))
            print(f"[replay_archive] {row['name']}: reused", flush=True); continue
        try:
            res = run_burst(row, args.ecsv_root, bdir, config, args.match_radius, args.k_start, log)
        except Exception as e:  # keep the batch going; the index shows the failure
            log.exception(f"{row['name']} failed")
            res = {"name": row["name"], "error": repr(e)}
            summary.dump_json(res, bdir / "summary.json")
        results.append(res)
        t = res.get("target", {})
        print(f"[replay_archive] {row['name']}: epochs={res.get('n_epochs')} recovered={t.get('recovered')} "
              f"k_first={t.get('k_first')} Q={t.get('quality_final')} spurious={res.get('spurious_final')} "
              f"{res.get('elapsed_s', 0):.0f}s", flush=True)
        summary.dump_json(results, args.out / "summary.json")
        write_index(results, args.out, label, args.site_link_root, desc)

    tab = Table(rows=[{"name": r["name"], "n_epochs": r.get("n_epochs", 0),
                       "recovered": bool(r.get("target", {}).get("recovered")),
                       "k_first": r.get("target", {}).get("k_first") or -1,
                       "t_first_s": r.get("target", {}).get("t_first_s") if r.get("target", {}).get("t_first_s") is not None else np.nan,
                       "quality_final": r.get("target", {}).get("quality_final") if r.get("target", {}).get("quality_final") is not None else np.nan,
                       "n_detections": r.get("target", {}).get("n_detections_final") or 0,
                       "candidates_final": r.get("candidates_final", 0), "spurious_final": r.get("spurious_final", 0)}
                      for r in results])
    tab.write(args.out / "summary.ecsv", format="ascii.ecsv", overwrite=True)
    print(f"[replay_archive] done: {sum(tab['recovered'])}/{len(tab)} recovered -> {args.out / 'index.html'}")


if __name__ == "__main__":
    main()
