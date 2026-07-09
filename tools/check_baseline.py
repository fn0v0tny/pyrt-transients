#!/usr/bin/env python3
"""Phase 0 safety net (rewrite.md).

Re-runs the real pipeline against the single 210619B GRB fixture (catalogs=
[gaia, usno] -- atlas@localhost is not reachable locally, see rewrite.md's
Environment note) and diffs the result against the frozen baseline in
tests/baseline/210619B/.

Run this after EVERY refactor phase and stop immediately if it fails -- do
not proceed to the next phase, and do not edit the baseline to match new
output. A failure means the refactor changed behavior; find out why before
continuing. Only tools/generate_baseline.py writes the baseline, and only as
a deliberate, reviewed step.

A passing run means "no obvious regression on this one GRB, on gaia/usno" --
not full behavior preservation (single fixture, not the 20-GRB regression
that lands in Phase 9) and NOT confirmation that the ATLAS-specific code
path is unaffected.
"""
import json
import sys
from pathlib import Path

from astropy.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixture_runner import run_fixture, REPO_ROOT, BASELINE_CATALOGS

BASELINE_DIR = REPO_ROOT / "tests" / "baseline" / "210619B"
WORK_DIR = REPO_ROOT / "local_test_output" / "baseline_check"

QUALITY_SCORE_TOL = 1e-9

# Steady-state (epoch 2+, i.e. warm-cache) per-epoch time vs. the baseline's own
# steady-state time. This is the complexity-regression signal: a refactor that
# turns an O(n) step into O(n^2) shows up here even though candidate counts and
# quality_scores are untouched. Ratios below WARN are ignored as normal jitter.
TIMING_WARN_FACTOR = 2.0
TIMING_FAIL_FACTOR = 5.0


def load_baseline():
    candidates_path = BASELINE_DIR / "candidates.tbl"
    if not candidates_path.exists():
        print(f"No baseline found at {BASELINE_DIR}. Run tools/generate_baseline.py once first.")
        sys.exit(2)
    candidates = Table.read(str(candidates_path), format="ascii.ipac")
    quality_scores = json.loads((BASELINE_DIR / "quality_scores.json").read_text())
    timing_path = BASELINE_DIR / "timing.json"
    timing = json.loads(timing_path.read_text()) if timing_path.exists() else None
    return candidates, quality_scores, timing


def main():
    print(
        f"[check_baseline] catalogs={BASELINE_CATALOGS} -- atlas@localhost excluded, "
        f"this is NOT a full ATLAS-inclusive check (see rewrite.md)."
    )

    baseline_candidates, baseline_scores, baseline_timing = load_baseline()
    new_candidates, new_summary, obs_dir, new_timing = run_fixture(WORK_DIR, catalogs=BASELINE_CATALOGS)

    failures = []
    warnings = []

    if len(new_candidates) != len(baseline_candidates):
        failures.append(
            f"candidate count changed: baseline={len(baseline_candidates)} new={len(new_candidates)}"
        )

    baseline_ids = set(baseline_scores.keys())
    new_scores = {}
    if "transient_id" in new_candidates.colnames and "quality_score" in new_candidates.colnames:
        for row in new_candidates:
            new_scores[str(row["transient_id"])] = float(row["quality_score"])
    new_ids = set(new_scores.keys())

    missing = baseline_ids - new_ids
    added = new_ids - baseline_ids
    if missing:
        failures.append(f"candidates in baseline but missing from new output: {sorted(missing)}")
    if added:
        failures.append(f"candidates in new output but not in baseline: {sorted(added)}")

    for tid in sorted(baseline_ids & new_ids):
        old_q = baseline_scores[tid]
        new_q = new_scores[tid]
        if abs(old_q - new_q) > QUALITY_SCORE_TOL:
            failures.append(
                f"quality_score drifted for {tid}: baseline={old_q!r} new={new_q!r} "
                f"(delta={new_q - old_q!r})"
            )

    # Complexity/timing check: steady-state (epoch 2+, warm-cache) per-epoch time
    # vs. the baseline's own steady-state time. Epoch 1 is excluded on both sides
    # since its cold-cache catalog queries are dominated by network variance, not
    # pipeline complexity (see rewrite.md's cache-widening note).
    new_steady = new_timing["steady_state_seconds"]
    cadence = new_timing["real_time_cadence_seconds"]
    print(
        f"[check_baseline] timing: total={new_timing['total_seconds']:.1f}s, "
        f"steady-state={new_steady:.1f}s/epoch"
        + (f", real-time budget~{cadence:.1f}s/epoch" if cadence else "")
    )

    if baseline_timing is None:
        warnings.append(
            "no timing.json in baseline (predates timing tracking) -- "
            "re-run tools/generate_baseline.py --force to start tracking it"
        )
    else:
        base_steady = baseline_timing["steady_state_seconds"]
        if base_steady > 0:
            ratio = new_steady / base_steady
            msg = (
                f"steady-state per-epoch time is {ratio:.1f}x the baseline "
                f"({new_steady:.1f}s vs {base_steady:.1f}s)"
            )
            if ratio >= TIMING_FAIL_FACTOR:
                failures.append(f"{msg} -- likely complexity regression (>= {TIMING_FAIL_FACTOR}x)")
            elif ratio >= TIMING_WARN_FACTOR:
                warnings.append(msg)

    if cadence and new_steady > cadence:
        warnings.append(
            f"steady-state per-epoch time ({new_steady:.1f}s) exceeds the fixture's own "
            f"real-time cadence (~{cadence:.1f}s) -- the pipeline could not keep up with "
            f"live data at this rate"
        )

    if warnings:
        print("Warnings (non-fatal):")
        for w in warnings:
            print(f"  - {w}")

    if failures:
        print("BASELINE CHECK FAILED:")
        for f in failures:
            print(f"  - {f}")
        print()
        print(
            "This means the refactor changed behavior on the fixture. "
            "Find out why before continuing -- do not edit the baseline."
        )
        sys.exit(1)

    print(
        f"Baseline check passed: {len(new_candidates)} candidates match "
        f"(count, transient_id set, and quality_score within {QUALITY_SCORE_TOL})."
    )


if __name__ == "__main__":
    main()
