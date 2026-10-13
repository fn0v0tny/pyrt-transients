"""Fetch the host-search template for a position before the frames exist.

    python -m pyrt_transient.detection.host_search.prefetch --ra 21.806 --dec 12.9
    python -m pyrt_transient.detection.host_search.prefetch --watch

The first form is what search.prefetch_async spawns when a trigger
observation's first frame arrives (the 0.1 s pointing frames, on D50 some
15 min before the first science frame). The second polls RTS2's targets
table for new GRB-type targets (type_id 'G': the automatic GRB / SVOM /
EP / IceCube triggers, created the moment the GCN notice is processed,
before the telescope has even slewed) and prefetches for each new one:
that is as early as a template can be asked for. The database is read
through `psql` exactly as tools/pipeline_entry.py does (PYRT_TARGETS_DB,
default "stars"; read-only session); without it --watch does nothing.
"""
import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from pyrt_transient.detection.host_search import templates as tpl

logger = logging.getLogger("detection.host_search.prefetch")
GRB_TYPES = {"G"}


def recent_grb_targets(db: str, limit: int = 10, timeout: float = 10.0):
    """[{tar_id, name, ra, dec}] for the newest GRB-type RTS2 targets."""
    if not db:
        return []
    query = (f"select tar_id, tar_name, tar_ra, tar_dec from targets where type_id in "
             f"({','.join(repr(t) for t in sorted(GRB_TYPES))}) and tar_ra is not null "
             f"order by tar_id desc limit {int(limit)}")
    env = dict(os.environ, PGCONNECT_TIMEOUT="5", PGOPTIONS="-c default_transaction_read_only=on")
    try:
        out = subprocess.run(["psql", "-XAtq", "-F", "\t", "-d", db, "-c", query],
                             capture_output=True, text=True, timeout=timeout, env=env)
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    if out.returncode == 0:
        for line in out.stdout.splitlines():
            parts = line.split("\t")
            if len(parts) == 4 and parts[0].strip().isdigit():
                try:
                    rows.append({"tar_id": int(parts[0]), "name": parts[1].strip(),
                                 "ra": float(parts[2]), "dec": float(parts[3])})
                except ValueError:
                    continue
    return rows


def watch(args, log):
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    seen_path = cache_dir / "prefetch_seen.json"
    try:
        seen = set(json.loads(seen_path.read_text()))
    except (OSError, ValueError):
        seen = set()
    first = True
    log.info(f"Prefetch watcher: polling RTS2 targets ({args.targets_db}) every {args.interval:g} s")
    while True:
        rows = recent_grb_targets(args.targets_db)
        for r in rows:
            if r["tar_id"] in seen:
                continue
            seen.add(r["tar_id"])
            if first:
                continue        # targets that existed before the watcher started
            log.info(f"Prefetch watcher: new GRB-type target {r['tar_id']} {r['name']!r} at {r['ra']:.4f} {r['dec']:+.4f}")
            src = tpl.prefetch_template(r["ra"], r["dec"], args.size_arcmin, args.band, args.sources, cache_dir,
                                        args.pixscale, args.timeout, log)
            log.info(f"Prefetch watcher: target {r['tar_id']}: {src or 'no coverage'}")
        first = False
        try:
            seen_path.write_text(json.dumps(sorted(seen)[-200:]))
        except OSError:
            pass
        if args.once:
            return
        time.sleep(args.interval)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ra", type=float)
    ap.add_argument("--dec", type=float)
    ap.add_argument("--size-arcmin", type=float, default=20.0)
    ap.add_argument("--band", default="r")
    ap.add_argument("--sources", nargs="+", default=["legacysurvey", "panstarrs"])
    ap.add_argument("--cache-dir", default=str(tpl.default_cache_dir()))
    ap.add_argument("--pixscale", type=float, default=0.5)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--watch", action="store_true", help="poll RTS2 for new GRB-type targets and prefetch for them")
    ap.add_argument("--targets-db", default=os.environ.get("PYRT_TARGETS_DB", "stars"))
    ap.add_argument("--interval", type=float, default=30.0)
    ap.add_argument("--once", action="store_true", help="with --watch: one poll, then exit")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    log = logging.getLogger("host_search.prefetch")
    if args.watch:
        watch(args, log)
        return 0
    if args.ra is None or args.dec is None:
        ap.error("--ra and --dec are required unless --watch")
    src = tpl.prefetch_template(args.ra, args.dec, args.size_arcmin, args.band, args.sources, args.cache_dir,
                                args.pixscale, args.timeout, log)
    log.info(f"Prefetch: {src or 'no coverage'}")
    return 0 if src else 1


if __name__ == "__main__":
    sys.exit(main())
