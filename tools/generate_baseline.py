#!/usr/bin/env python3
"""One-time Phase 0 baseline generator (rewrite.md).

Run this exactly once, deliberately, against the CURRENT UNMODIFIED pipeline,
to freeze tests/baseline/210619B/. After that, tools/check_baseline.py compares
future runs against this frozen snapshot -- it never regenerates it.

If the baseline itself later needs to move (a deliberately reviewed and
approved behavior change, not a refactor gone wrong), re-run this with
--force and commit the new baseline as its own reviewed change -- never as a
reflex to make a failing check_baseline.py pass.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixture_runner import run_fixture, REPO_ROOT, BASELINE_CATALOGS

BASELINE_DIR = REPO_ROOT / "tests" / "baseline" / "210619B"
WORK_DIR = REPO_ROOT / "local_test_output" / "baseline_gen"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite an existing baseline without prompting",
    )
    args = parser.parse_args()

    if BASELINE_DIR.exists() and not args.force:
        print(f"Baseline already exists at {BASELINE_DIR}. Re-run with --force to overwrite.")
        sys.exit(1)

    candidates, summary, obs_dir, timing = run_fixture(WORK_DIR, catalogs=BASELINE_CATALOGS)

    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    candidates.write(str(BASELINE_DIR / "candidates.tbl"), format="ascii.ipac", overwrite=True)
    (BASELINE_DIR / "lightcurve_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )

    quality_scores = {}
    if "transient_id" in candidates.colnames and "quality_score" in candidates.colnames:
        for row in candidates:
            quality_scores[str(row["transient_id"])] = float(row["quality_score"])
    (BASELINE_DIR / "quality_scores.json").write_text(
        json.dumps(quality_scores, indent=2, sort_keys=True)
    )
    (BASELINE_DIR / "timing.json").write_text(json.dumps(timing, indent=2, sort_keys=True))

    manifest = {
        "catalogs": BASELINE_CATALOGS,
        "n_candidates": len(candidates),
        "note": (
            "Generated with catalogs=[gaia, usno] only -- atlas@localhost is "
            "unreachable in local dev. This is NOT a full ATLAS-inclusive baseline. "
            "See rewrite.md's Environment note."
        ),
    }
    (BASELINE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(
        f"Baseline written to {BASELINE_DIR}: {len(candidates)} candidates, "
        f"steady-state {timing['steady_state_seconds']:.1f}s/epoch."
    )
    shutil.rmtree(WORK_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
