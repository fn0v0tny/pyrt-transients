"""Regression tests for detection/blind_multicatalog/catalog_match.py's
_empty_candidates_table -- found while running the stacking feature
against real, un-cached GRB data (GRB151027B): USNO-B genuinely has zero
coverage for that field (an entirely ordinary per-field catalog-coverage
gap, not a rare corner case), and find_transients_multicatalog's own
except-branch built an empty placeholder table via
`Table(); candidates['candidate_type'] = []` -- astropy infers an empty
Python list column as float64, but a *real*, successful catalog's
candidate_type column is a genuine string. clustering.combine_results then
vstacks all per-catalog tables together and crashed with
`TableMergeError: The 'candidate_type' columns have incompatible types:
['str352', 'float64']` -- reproduced directly against real data, not
hypothetical.

No pytest dependency -- run directly with
`python3 tests/test_catalog_match.py`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from astropy.table import Table, vstack

from pyrt_transient.detection.blind_multicatalog.catalog_match import _empty_candidates_table
from pyrt_transient.detection.blind_multicatalog.clustering import combine_results


def _real_catalog_table(n=2):
    """A minimal but realistic per-catalog result table, matching what
    find_transients_multicatalog's success branch actually produces:
    real string-typed candidate_type/reference_catalog columns.
    """
    return Table({
        "ALPHA_J2000": np.array([180.0, 180.001][:n]),
        "DELTA_J2000": np.array([30.0, 30.001][:n]),
        "X_IMAGE": np.array([100.0, 200.0][:n]),
        "Y_IMAGE": np.array([100.0, 200.0][:n]),
        "quality_score": np.array([1.5, 2.0][:n]),
        "candidate_type": np.array(["new", "new"][:n]),
        "reference_catalog": np.array(["gaia", "gaia"][:n]),
    })


def test_empty_candidates_table_has_string_dtypes_not_float():
    empty = _empty_candidates_table()
    assert len(empty) == 0
    assert np.issubdtype(empty["quality_score"].dtype, np.floating)
    assert empty["candidate_type"].dtype.kind in ("U", "S"), \
        f"candidate_type must be string-typed, got {empty['candidate_type'].dtype}"
    assert empty["reference_catalog"].dtype.kind in ("U", "S"), \
        f"reference_catalog must be string-typed, got {empty['reference_catalog'].dtype}"
    print("test_empty_candidates_table_has_string_dtypes_not_float: PASS")


def test_empty_candidates_table_vstacks_cleanly_against_real_data():
    # This is the exact operation that crashed on real GRB151027B data:
    # vstack-ing a successful catalog's real string columns against an
    # empty placeholder from a catalog with zero coverage.
    real = _real_catalog_table()
    empty = _empty_candidates_table()
    stacked = vstack([real, empty])  # must not raise TableMergeError
    assert len(stacked) == len(real)
    print("test_empty_candidates_table_vstacks_cleanly_against_real_data: PASS")


def test_combine_results_does_not_crash_when_one_catalog_has_no_coverage():
    # Reproduces the real crash end-to-end through combine_results itself
    # (not just the raw vstack call), with the same dict shape
    # find_transients_multicatalog builds: one catalog succeeded (gaia),
    # one had no coverage at all (usno).
    transients = {
        "gaia": _real_catalog_table(n=2),
        "usno": _empty_candidates_table(),
    }
    # Must not raise -- min_quality/min_catalogs_fraction values don't
    # matter for this regression, only that it returns instead of crashing.
    result = combine_results(transients, min_catalogs_fraction=0.5, min_quality=0.0, time=None)
    assert isinstance(result, Table)
    print("test_combine_results_does_not_crash_when_one_catalog_has_no_coverage: PASS")


def test_combine_results_still_works_when_all_catalogs_have_no_coverage():
    transients = {
        "gaia": _empty_candidates_table(),
        "usno": _empty_candidates_table(),
    }
    result = combine_results(transients, min_catalogs_fraction=1.0, min_quality=0.0, time=None)
    assert isinstance(result, Table)
    assert len(result) == 0
    print("test_combine_results_still_works_when_all_catalogs_have_no_coverage: PASS")


if __name__ == "__main__":
    test_empty_candidates_table_has_string_dtypes_not_float()
    test_empty_candidates_table_vstacks_cleanly_against_real_data()
    test_combine_results_does_not_crash_when_one_catalog_has_no_coverage()
    test_combine_results_still_works_when_all_catalogs_have_no_coverage()
    print("All catalog_match.py regression tests passed.")
