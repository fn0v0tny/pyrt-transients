"""Regression test for detection/blind_multicatalog/stdpipe_filters.py's
apply_skybot_filter -- found while running the stacking feature against
real, un-cached GRB data (GRB210410A): SkyBoT (via
stdpipe.pipeline.filter_transient_candidates -> stdpipe.catalogs.
xmatch_skybot) crashes with `KeyError: 'RA'` whenever it genuinely finds
zero solar-system objects for a field/time -- `Skybot.cone_search` doesn't
raise in that case (only warns `NoResultsWarning`), it returns a table with
no 'RA'/'DEC' columns, and stdpipe's own xmatch_skybot indexes into it
unguarded on the next line. A field with no currently-known asteroids is an
entirely ordinary result, reproduced directly against real data, not
hypothetical.

apply_skybot_filter now catches that KeyError and treats it the same as
"SkyBoT found nothing to reject" (candidates pass through unfiltered)
rather than propagating and crashing the whole epoch.

No pytest dependency -- run directly with
`python3 tests/test_stdpipe_filters.py`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from astropy.table import Table
from astropy.time import Time

from pyrt_transient.detection.blind_multicatalog import stdpipe_filters


def _candidates_table(n=3):
    return Table({
        "ALPHA_J2000": [180.0, 180.001, 180.002][:n],
        "DELTA_J2000": [30.0, 30.001, 30.002][:n],
        "FLAGS": [0, 0, 0][:n],
        "MAGERR_CALIB": [0.05, 0.05, 0.05][:n],
        "FWHM_IMAGE": [3.0, 3.0, 3.0][:n],
    })


def test_apply_skybot_filter_survives_keyerror_from_empty_skybot_result(monkeypatch=None):
    candidates = _candidates_table()

    def _raise_keyerror(*args, **kwargs):
        raise KeyError("RA")

    original = stdpipe_filters.stdpipe_pipeline.filter_transient_candidates
    stdpipe_filters.stdpipe_pipeline.filter_transient_candidates = _raise_keyerror
    try:
        result, n_removed = stdpipe_filters.apply_skybot_filter(
            candidates, time=Time("2021-04-10T23:00:00"),
        )
    finally:
        stdpipe_filters.stdpipe_pipeline.filter_transient_candidates = original

    assert len(result) == len(candidates), "candidates must pass through unfiltered, not be dropped"
    assert n_removed == 0
    print("test_apply_skybot_filter_survives_keyerror_from_empty_skybot_result: PASS")


def test_apply_skybot_filter_still_removes_real_matches():
    candidates = _candidates_table(n=3)

    def _fake_filter(obj, sr=None, vizier=None, skybot=None, ned=None, flagged=None,
                      time=None, get_candidates=None):
        # Reject the middle candidate, matching filter_transient_candidates'
        # get_candidates=False contract (boolean mask, True = keep).
        import numpy as np
        return np.array([True, False, True])

    original = stdpipe_filters.stdpipe_pipeline.filter_transient_candidates
    stdpipe_filters.stdpipe_pipeline.filter_transient_candidates = _fake_filter
    try:
        result, n_removed = stdpipe_filters.apply_skybot_filter(
            candidates, time=Time("2021-04-10T23:00:00"),
        )
    finally:
        stdpipe_filters.stdpipe_pipeline.filter_transient_candidates = original

    assert n_removed == 1
    assert len(result) == 2
    print("test_apply_skybot_filter_still_removes_real_matches: PASS")


def test_apply_skybot_filter_noop_on_empty_or_no_time():
    candidates = _candidates_table()
    result, n_removed = stdpipe_filters.apply_skybot_filter(candidates[:0], time=Time("2021-04-10T23:00:00"))
    assert len(result) == 0 and n_removed == 0
    result, n_removed = stdpipe_filters.apply_skybot_filter(candidates, time=None)
    assert len(result) == len(candidates) and n_removed == 0
    print("test_apply_skybot_filter_noop_on_empty_or_no_time: PASS")


if __name__ == "__main__":
    test_apply_skybot_filter_survives_keyerror_from_empty_skybot_result()
    test_apply_skybot_filter_still_removes_real_matches()
    test_apply_skybot_filter_noop_on_empty_or_no_time()
    print("All stdpipe_filters.py tests passed.")
