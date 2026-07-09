"""Shared driver for running pyrt_transient against the Phase 0 local fixture.

Used by both tools/generate_baseline.py (one-time, writes tests/baseline/210619B/)
and tools/check_baseline.py (run after every refactor phase). See rewrite.md's
Environment note: atlas@localhost is not reachable in local dev, so this always
runs with catalogs=[gaia, usno] only.
"""
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from astropy.table import Table

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = REPO_ROOT / "tests" / "210619B"
BASELINE_CATALOGS = ["gaia", "usno"]

_TIMESTAMP_RE = re.compile(r"^(\d{14})")


def _real_time_cadence_seconds(pairs):
    """Median gap between consecutive epochs' observation timestamps.

    This is the actual real-time budget the production pipeline has to keep
    up with -- a steady-state per-epoch processing time that exceeds this
    means the pipeline falls further behind with every new exposure.
    """
    timestamps = []
    for ecsv, _fits in pairs:
        m = _TIMESTAMP_RE.match(ecsv.name)
        if not m:
            return None
        timestamps.append(datetime.strptime(m.group(1), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc))
    if len(timestamps) < 2:
        return None
    deltas = sorted(
        (b - a).total_seconds() for a, b in zip(timestamps, timestamps[1:])
    )
    mid = len(deltas) // 2
    return deltas[mid] if len(deltas) % 2 else (deltas[mid - 1] + deltas[mid]) / 2


def _raw_epoch_pairs():
    """(ecsv, fits) pairs for the fixture's raw per-epoch detections, sorted by
    observation time, excluding pre-existing derived outputs (*_transients.ecsv,
    transient_*_lightcurve.ecsv) that were checked in alongside the raw fixture.
    """
    pairs = []
    for ecsv in sorted(FIXTURE_DIR.glob("*.ecsv")):
        name = ecsv.name
        if name.endswith("_transients.ecsv") or name.startswith("transient_"):
            continue
        fits = ecsv.with_suffix(".fits")
        if not fits.exists():
            raise FileNotFoundError(f"No matching FITS for {ecsv}")
        pairs.append((ecsv, fits))
    if not pairs:
        raise RuntimeError(f"No raw epoch ecsv/fits pairs found under {FIXTURE_DIR}")
    return pairs


def run_fixture(work_dir: Path, catalogs=None, verbose=True):
    """Run the real pipeline_magic CLI once per epoch against a fresh work_dir.

    Returns (candidates_table, lightcurve_summary_dict, obs_dir, timing_dict).

    timing_dict has:
      - per_epoch_seconds: wall-clock seconds for each epoch's pipeline_magic call
      - epoch_files: matching ecsv filenames, same order
      - total_seconds: sum of per_epoch_seconds
      - steady_state_seconds: mean of per_epoch_seconds[1:] (excludes epoch 1,
        whose cold-cache catalog queries make it a poor complexity signal --
        see rewrite.md's cache-widening note)
      - real_time_cadence_seconds: median gap between the fixture's own epoch
        observation timestamps -- the actual real-time budget the production
        pipeline must keep up with
    """
    catalogs = catalogs or BASELINE_CATALOGS
    if work_dir.exists():
        shutil.rmtree(work_dir)
    data_dir = work_dir / "data"
    public_dir = work_dir / "public_html"
    data_dir.mkdir(parents=True)
    public_dir.mkdir(parents=True)

    config_path = work_dir / "run_config.yaml"
    config_path.write_text(
        "global:\n"
        f"  base_data_dir: {data_dir}\n"
        f"  base_public_dir: {public_dir}\n"
        "  generate_frontend: false\n"
        "detection:\n"
        f"  catalogs: {json.dumps(catalogs)}\n"
    )

    pairs = _raw_epoch_pairs()
    if verbose:
        print(
            f"[fixture_runner] running {len(pairs)} epochs through pipeline_magic "
            f"with catalogs={catalogs} (atlas@localhost excluded -- not reachable locally)"
        )

    per_epoch_seconds = []
    for i, (ecsv, fits) in enumerate(pairs, 1):
        if verbose:
            print(f"[fixture_runner] epoch {i}/{len(pairs)}: {ecsv.name}")
        t0 = time.monotonic()
        result = subprocess.run(
            [
                sys.executable, "-m", "pyrt_transient.pipeline_magic",
                str(ecsv), str(fits), f"--config={config_path}",
            ],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        elapsed = time.monotonic() - t0
        per_epoch_seconds.append(elapsed)
        if verbose:
            print(f"[fixture_runner]   epoch {i} took {elapsed:.1f}s")
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr)
            raise RuntimeError(
                f"pipeline_magic failed on epoch {i} ({ecsv.name}), exit {result.returncode}"
            )

    obs_dirs = list(data_dir.glob("obs_*"))
    if len(obs_dirs) != 1:
        raise RuntimeError(f"Expected exactly one obs_* directory, found {obs_dirs}")
    obs_dir = obs_dirs[0]

    candidates_path = obs_dir / "candidates.tbl"
    if not candidates_path.exists():
        raise RuntimeError(f"No candidates.tbl produced at {candidates_path}")
    candidates = Table.read(str(candidates_path), format="ascii.ipac")

    summary_path = obs_dir / "lightcurve_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    steady_state = per_epoch_seconds[1:] if len(per_epoch_seconds) > 1 else per_epoch_seconds
    timing = {
        "per_epoch_seconds": per_epoch_seconds,
        "epoch_files": [ecsv.name for ecsv, _fits in pairs],
        "total_seconds": sum(per_epoch_seconds),
        "steady_state_seconds": sum(steady_state) / len(steady_state) if steady_state else 0.0,
        "real_time_cadence_seconds": _real_time_cadence_seconds(pairs),
    }
    if verbose:
        print(
            f"[fixture_runner] total {timing['total_seconds']:.1f}s, "
            f"steady-state avg {timing['steady_state_seconds']:.1f}s/epoch "
            f"(real-time budget ~{timing['real_time_cadence_seconds']:.1f}s/epoch)"
        )

    return candidates, summary, obs_dir, timing
