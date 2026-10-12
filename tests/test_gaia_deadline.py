"""CatTransients._get_gaia_data runs under a deadline: astroquery's TAP client
has no timeout of its own, and a Gaia query that never answers must become
a failure (cache fallback, shared failure marker) instead of eating the
frame's whole 900 s daemon budget."""
import socket
import time

import numpy as np
from astropy.table import Table

from pyrt_transient import catalog as cat
from pyrt_transient.catalog import CatTransients, QueryParams


def _gaia():
    c = CatTransients.__new__(CatTransients)  # no __init__: it would query
    c._catalog_name = CatTransients.GAIA
    c._query_params = QueryParams(ra=10.0, dec=20.0, width=0.3, height=0.3, mlim=20, timeout=0)
    c._original_query_params = None
    return c


def test_a_hanging_gaia_query_is_abandoned_at_the_deadline(monkeypatch):
    monkeypatch.setattr(cat, "GAIA_QUERY_DEADLINE_S", 0.3)

    def hang(self):
        time.sleep(3.0)
        return None

    monkeypatch.setattr(CatTransients.__bases__[0], "_get_gaia_data", hang)
    before = socket.getdefaulttimeout()
    t0 = time.time()
    try:
        _gaia()._get_gaia_data()
        raise AssertionError("expected TimeoutError")
    except TimeoutError as e:
        assert "Gaia query" in str(e)
    assert time.time() - t0 < 2.0, "the deadline, not the hang, must end the call"
    assert socket.getdefaulttimeout() == before, "socket default timeout must be restored"


def test_a_prompt_gaia_answer_passes_through_with_sloan_columns(monkeypatch):
    def answer(self):
        return Table({"G": np.array([15.0]), "BP": np.array([15.5]), "RP": np.array([14.5])})

    monkeypatch.setattr(CatTransients.__bases__[0], "_get_gaia_data", answer)
    out = _gaia()._get_gaia_data()
    assert "Sloan_g" in out.colnames and "Sloan_r" in out.colnames
    assert np.isfinite(out["Sloan_r"][0])


def test_query_params_timeout_can_raise_the_deadline(monkeypatch):
    seen = {}

    def spy(fn, seconds, what="call"):
        seen["seconds"] = seconds
        return fn()

    monkeypatch.setattr(cat, "_call_with_deadline", spy)
    monkeypatch.setattr(CatTransients.__bases__[0], "_get_gaia_data", lambda self: None)
    c = _gaia()
    c._query_params.timeout = 900
    c._get_gaia_data()
    assert seen["seconds"] == 900
    c._query_params.timeout = 10
    c._get_gaia_data()
    assert seen["seconds"] == cat.GAIA_QUERY_DEADLINE_S
