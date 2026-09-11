"""A reference catalogue that has no coverage of the field, returned no
rows, or failed to download must be EXCLUDED from the agreement
denominator, not inserted as an empty candidate table (which, under
min_catalogs_fraction=1.0, vetoes every source and silently yields zero
candidates for e.g. any field south of Dec -30 with Pan-STARRS listed)."""
import logging

import numpy as np
import pytest
from astropy.table import Table

from pyrt_transient import PipelineConfig
from pyrt_transient.catalog import CatalogNoCoverageError, CatTransients, QueryParams
from pyrt_transient.detection.blind_multicatalog import catalog_match, clustering


class _StubCatalog:
    def __init__(self, n_rows, candidates=None):
        self._n = n_rows
        self._cands = candidates
    def __len__(self):
        return self._n
    def _generate_image_id(self, det):
        return "img"
    def get_transient_candidates_optimized(self, detections, **kw):
        return self._cands.copy()


def _cands():
    return Table({"ALPHA_J2000": [10.0], "DELTA_J2000": [20.0], "MAG_CALIB": [16.0],
                  "MAGERR_CALIB": [0.02], "X_IMAGE": [100.0], "Y_IMAGE": [100.0],
                  "FWHM_IMAGE": [3.0], "FLAGS": [0], "quality_score": [1.0],
                  "candidate_type": ["new"]})


class _StubLoader:
    def get_optimized_catalog(self, name, params):
        if name == "good":
            return _StubCatalog(1000, _cands())
        if name == "empty":
            return _StubCatalog(0)
        if name == "broken":
            raise RuntimeError("timeout")
        raise KeyError(name)


def _detections():
    t = Table({"ALPHA_J2000": [10.0], "DELTA_J2000": [20.0], "MAG_CALIB": [16.0],
               "MAGERR_CALIB": [0.02], "X_IMAGE": [100.0], "Y_IMAGE": [100.0]})
    t.meta = {"MAGLIM": 18.0, "CD1_1": -1 / 3600, "CD2_2": 1 / 3600}
    return t


def test_unavailable_catalogs_are_excluded_from_results():
    cfg = PipelineConfig()
    cfg.detection.enable_adaptive_idlimit = False
    res, failed = catalog_match.find_transients_multicatalog(
        _StubLoader(), cfg, logging.getLogger("t"), _detections(),
        ["good", "empty", "broken"], params=QueryParams(ra=10, dec=20, width=0.3, height=0.3))
    assert set(res) == {"good"}
    assert len(res["good"]) == 1
    # A download that raised is reported separately from a field the
    # catalogue simply does not cover: only the former makes the epoch
    # degraded and worth recomputing next run.
    assert set(failed) == {"broken"}


def test_unanimity_over_available_catalogs_only():
    """One available catalogue flagging a source is unanimous; the same
    source with an empty table injected for a second catalogue is not."""
    c = _cands()
    c["reference_catalog"] = "good"
    only_good = clustering.combine_results({"good": c}, min_catalogs_fraction=1.0, min_quality=0.2)
    assert len(only_good) == 1
    with_empty = clustering.combine_results(
        {"good": c, "empty": catalog_match._empty_candidates_table()},
        min_catalogs_fraction=1.0, min_quality=0.2)
    assert len(with_empty) == 0   # this is why exclusion matters


def test_ps1_coverage_precheck_without_network():
    assert CatTransients.ps1_covers(33.8, 0.3)
    assert CatTransients.ps1_covers(-29.9, 0.3)
    assert not CatTransients.ps1_covers(-45.0, 0.3)
    assert not CatTransients.ps1_covers(-30.3, 0.3)
    cat = CatTransients.__new__(CatTransients)
    cat._query_params = QueryParams(ra=100.0, dec=-45.0, width=0.3, height=0.3, mlim=20)
    assert cat._get_panstarrs_vizier_data() is None
    assert cat._get_panstarrs_data() is None


def test_ps1_vizier_column_mapping_and_sentinels():
    from astropy.table import MaskedColumn
    t = Table({"RAJ2000": [1.0, 2.0], "DEJ2000": [3.0, 4.0],
               "gmag": MaskedColumn([17.0, 18.0], mask=[False, True]),
               "rmag": [16.5, -999.0], "e_rmag": [0.01, 0.02],
               "Nd": MaskedColumn(np.array([10, 3], dtype=np.int16), mask=[False, True]),
               "Qual": MaskedColumn(np.array([52, 4], dtype=np.int16), mask=[False, False])})
    out = CatTransients._ps1_vizier_to_catalog(t)
    assert list(out["radeg"]) == [1.0, 2.0]
    assert out["Sloan_g"][0] == 17.0 and np.isnan(out["Sloan_g"][1])
    assert out["Sloan_r"][0] == 16.5 and np.isnan(out["Sloan_r"][1])
    assert out["n_detections"][0] == 10 and np.isnan(out["n_detections"][1])
    assert np.all(out["pmra"] == 0) and len(out) == 2
    assert "panstarrs@vizier" in CatTransients.KNOWN_CATALOGS
    assert set(CatTransients.KNOWN_CATALOGS["panstarrs@vizier"]["filters"]) == {"Sloan_g", "Sloan_r", "Sloan_i", "Sloan_z"}


def test_no_coverage_is_not_reported_as_a_failure():
    """A catalogue that has no rows for the field must land in neither
    `results` nor `failed`.  The load is eager (pyrt's Catalog.__init__ runs
    the query), so "no coverage" arrives as CatalogNoCoverageError out of the
    loader, not as a None return -- if that is classified as a failure the
    caller stamps every epoch DEGRADED, epoch_is_cached never returns True,
    and the whole campaign is re-downloaded and re-analysed on every run."""
    class _Loader:
        def get_optimized_catalog(self, name, params):
            if name == "good":
                return _StubCatalog(1000, _cands())
            if name == "uncovered":
                raise CatalogNoCoverageError("No data retrieved from panstarrs")
            raise RuntimeError("timeout")

    cfg = PipelineConfig()
    cfg.detection.enable_adaptive_idlimit = False
    res, failed = catalog_match.find_transients_multicatalog(
        _Loader(), cfg, logging.getLogger("t"), _detections(),
        ["good", "uncovered", "broken"], params=QueryParams(ra=10, dec=20, width=0.3, height=0.3))
    assert set(res) == {"good"}
    assert set(failed) == {"broken"}


def test_no_coverage_error_is_a_value_error():
    """Callers that predate the split still catch it as a ValueError."""
    assert issubclass(CatalogNoCoverageError, ValueError)


def test_fetch_raises_no_coverage_below_the_ps1_southern_limit(monkeypatch):
    """The end the routing above depends on: a helper returning None becomes
    CatalogNoCoverageError, not a bare ValueError indistinguishable from a
    download that blew up."""
    class _NoDiskCache:
        def load_from_cache(self, name, params):
            return None
    monkeypatch.setattr(CatTransients, "get_cache", classmethod(lambda cls: _NoDiskCache()))

    cat = CatTransients.__new__(CatTransients)
    cat._catalog_name = CatTransients.PANSTARRS_VIZIER
    cat._query_params = QueryParams(ra=100.0, dec=-45.0, width=0.3, height=0.3, mlim=20)
    cat._original_query_params = None
    with pytest.raises(CatalogNoCoverageError):
        cat._fetch_catalog_data()
