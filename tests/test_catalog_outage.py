"""A catalogue-server outage must cost at most one failed query per run, and
a field whose disk-cache entry has expired keeps its catalogue.

On lascaux50 (2026-09-11) Gaia TAP answered 500 after ~3 min. The field's
cache entry was 31.8 days old, so it was deleted before the refresh, and
every epoch still marked degraded asked Gaia again: run n cost n timeouts."""
import pickle
import time

import pytest
from astropy.table import Table

from pyrt_transient.catalog import CatalogCache, CatalogNoCoverageError, CatTransients, QueryParams


@pytest.fixture(autouse=True)
def _fresh_state(tmp_path, monkeypatch):
    monkeypatch.setattr(CatTransients, "_failed_queries", {})
    monkeypatch.setattr(CatTransients, "_cache", CatalogCache(str(tmp_path)))


def _params():
    return QueryParams(ra=323.5, dec=50.4, width=0.5, height=0.5, mlim=17.0)


def _stars():
    return Table({"radeg": [323.4, 323.5, 323.6], "decdeg": [50.3, 50.4, 50.5]})


def _gaia():
    cat = CatTransients.__new__(CatTransients)  # no __init__: it would query
    cat._catalog_name = "gaia"
    cat._query_params = _params()
    cat._original_query_params = None
    return cat


def _write_entry(age_days):
    cache = CatTransients.get_cache()
    path = cache.get_cache_path("gaia", _params())
    with open(path, "wb") as fh:
        pickle.dump({"data": _stars(), "timestamp": time.time() - age_days * 86400,
                     "params": vars(_params())}, fh)
    return path


def _parent_fetch(monkeypatch, outcome):
    """Replace the pyrt base class's query; returns the list of calls."""
    calls = []

    def fetch(self):
        calls.append(self._catalog_name)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(CatTransients.__bases__[0], "_fetch_catalog_data", fetch)
    return calls


def test_an_expired_entry_is_kept_and_served_only_as_a_fallback():
    path = _write_entry(age_days=40)
    cache = CatTransients.get_cache()

    assert cache.load_from_cache("gaia", _params()) is None
    assert path.exists()
    assert len(cache.load_from_cache("gaia", _params(), allow_stale=True)) == 3


def test_a_failed_refresh_serves_the_expired_entry(monkeypatch):
    _write_entry(age_days=40)
    calls = _parent_fetch(monkeypatch, RuntimeError("Gaia query failed: 500"))

    result = _gaia()._fetch_catalog_data()

    assert calls == ["gaia"]
    assert len(result) == 3
    assert result.meta["stale"] and result.meta["catalog"] == "gaia"


def test_a_failed_query_is_sent_once_per_run(monkeypatch):
    calls = _parent_fetch(monkeypatch, RuntimeError("Gaia query failed: 500"))

    for _ in range(3):   # the new epoch plus two degraded ones being recomputed
        with pytest.raises(Exception, match="500"):
            _gaia()._fetch_catalog_data()

    assert calls == ["gaia"]


def test_no_coverage_is_an_answer_not_a_failure(monkeypatch):
    calls = _parent_fetch(monkeypatch, None)

    for _ in range(2):
        with pytest.raises(CatalogNoCoverageError):
            _gaia()._fetch_catalog_data()

    assert calls == ["gaia", "gaia"]
    assert CatTransients._failed_queries == {}


def test_a_fresh_answer_is_cached_and_the_widened_query_restored(monkeypatch):
    calls = _parent_fetch(monkeypatch, _stars())
    cat = _gaia()

    assert len(cat._fetch_catalog_data()) == 3
    assert cat._query_params == _params()
    assert len(_gaia()._fetch_catalog_data()) == 3   # from the disk cache
    assert calls == ["gaia"]
