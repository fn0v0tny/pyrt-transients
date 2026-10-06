"""Unit tests for detection/blind_multicatalog/galaxy_flag.py -- the
known-galaxy label on the final GRB candidates.

The catalogue services (Legacy Survey TAP, Pan-STARRS on VizieR) are
replaced by fakes through galaxy_flag.QUERY_FUNCTIONS; what's tested is
the classification, the per-candidate cache, and failure handling.

No pytest dependency -- run directly with
`python3 tests/test_galaxy_flag.py`.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from astropy.table import Table

from pyrt_transient.detection.blind_multicatalog import galaxy_flag


def _candidates():
    # 0: a catalogued 17.0 galaxy measured at 17.6 (aperture < total)
    # 1: 15.5 on top of a 17.0 galaxy -- brighter than the whole galaxy
    # 2: nothing extended nearby
    # 3: a galaxy 3" away, outside the 2" match radius
    return Table({
        "transient_id": ["t0", "t1", "t2", "t3"],
        "ALPHA_J2000": [10.0, 10.1, 10.2, 10.3],
        "DELTA_J2000": [20.0, 20.0, 20.0, 20.0],
        "MAG_CALIB": [17.6, 15.5, 18.0, 18.0],
    })


def _src(ra, dec, mag, catalog):
    return {"ra": ra, "dec": dec, "catalog": catalog,
            "mags": {"g": mag + 0.5, "r": mag, "i": mag - 0.3, "z": None}}


SOURCES = [
    _src(10.0, 20.0 + 0.5 / 3600, 17.0, "legacysurvey"),
    _src(10.1, 20.0, 17.0, "panstarrs"),
    _src(10.3, 20.0 + 3.0 / 3600, 16.0, "legacysurvey"),
]


class _Fake:
    def __init__(self, sources, fail=False):
        self.sources, self.fail, self.calls = sources, fail, []

    def __call__(self, coords, radius_arcsec, timeout):
        self.calls.append(list(coords))
        if self.fail:
            raise RuntimeError("service down")
        return [s for s in self.sources
                if any(galaxy_flag._sep_arcsec(ra, dec, s["ra"], s["dec"]) <= radius_arcsec
                       for ra, dec in coords)]


def _with_fakes(ls, ps1):
    saved = dict(galaxy_flag.QUERY_FUNCTIONS)
    galaxy_flag.QUERY_FUNCTIONS.update(legacysurvey=ls, panstarrs=ps1)
    return lambda: (galaxy_flag.QUERY_FUNCTIONS.clear(), galaxy_flag.QUERY_FUNCTIONS.update(saved))


def test_classifies_galaxy_host_and_unmatched():
    ls = _Fake([s for s in SOURCES if s["catalog"] == "legacysurvey"])
    ps1 = _Fake([s for s in SOURCES if s["catalog"] == "panstarrs"])
    restore = _with_fakes(ls, ps1)
    try:
        t = galaxy_flag.flag_known_galaxies(_candidates(), band="r")
    finally:
        restore()
    assert list(t["galaxy_flag"]) == ["galaxy", "host", "none", "none"], list(t["galaxy_flag"])
    assert list(t["galaxy_catalog"]) == ["legacysurvey", "panstarrs", "none", "none"]
    assert abs(t["galaxy_mag"][0] - 17.0) < 1e-9
    assert abs(t["galaxy_sep_arcsec"][0] - 0.5) < 0.01
    assert np.isnan(t["galaxy_mag"][2]) and np.isnan(t["galaxy_sep_arcsec"][3])
    assert len(t) == 4, "labels only, never removes"
    print("test_classifies_galaxy_host_and_unmatched: PASS")


def test_band_selects_the_galaxy_magnitude():
    restore = _with_fakes(_Fake(SOURCES), _Fake([]))
    try:
        # g mag of galaxy 0 is 17.5, so 17.6 is still the galaxy itself;
        # i mag of galaxy 1 is 16.7, so 15.5 is still brighter than it.
        t = galaxy_flag.flag_known_galaxies(_candidates(), band="g")
        assert abs(t["galaxy_mag"][0] - 17.5) < 1e-9
        t = galaxy_flag.flag_known_galaxies(_candidates(), band="i")
        assert list(t["galaxy_flag"][:2]) == ["galaxy", "host"]
    finally:
        restore()
    assert galaxy_flag.band_for_filter("Sloan_r") == "r"
    assert galaxy_flag.band_for_filter("Johnson_B") == "g"
    assert galaxy_flag.band_for_filter("Sloan_i") == "i"
    assert galaxy_flag.band_for_filter(None) == "r"
    assert galaxy_flag.band_for_filter("N") == "r"
    print("test_band_selects_the_galaxy_magnitude: PASS")


def test_cache_queries_only_new_candidates():
    ls, ps1 = _Fake(SOURCES), _Fake([])
    restore = _with_fakes(ls, ps1)
    try:
        with tempfile.TemporaryDirectory() as d:
            galaxy_flag.flag_known_galaxies(_candidates()[:2], cache_dir=Path(d))
            assert len(ls.calls) == 1 and len(ls.calls[0]) == 2
            t = galaxy_flag.flag_known_galaxies(_candidates(), cache_dir=Path(d))
            assert len(ls.calls) == 2 and len(ls.calls[1]) == 2, "only t2, t3 are new"
            assert list(t["galaxy_flag"]) == ["galaxy", "host", "none", "none"]
            cache = json.loads((Path(d) / galaxy_flag.CACHE_FILENAME).read_text())
            assert set(cache) == {"t0", "t1", "t2", "t3"}
            galaxy_flag.flag_known_galaxies(_candidates(), cache_dir=Path(d))
            assert len(ls.calls) == 2, "everything cached"
    finally:
        restore()
    print("test_cache_queries_only_new_candidates: PASS")


def test_one_failed_catalogue_still_uses_the_other():
    restore = _with_fakes(_Fake(SOURCES, fail=True), _Fake(SOURCES))
    try:
        t = galaxy_flag.flag_known_galaxies(_candidates())
    finally:
        restore()
    assert list(t["galaxy_flag"]) == ["galaxy", "host", "none", "none"]
    print("test_one_failed_catalogue_still_uses_the_other: PASS")


def test_all_catalogues_failing_leaves_none_and_does_not_cache():
    restore = _with_fakes(_Fake(SOURCES, fail=True), _Fake(SOURCES, fail=True))
    try:
        with tempfile.TemporaryDirectory() as d:
            t = galaxy_flag.flag_known_galaxies(_candidates(), cache_dir=Path(d))
            assert list(t["galaxy_flag"]) == ["none"] * 4
            assert not (Path(d) / galaxy_flag.CACHE_FILENAME).exists(), "retry next run"
    finally:
        restore()
    print("test_all_catalogues_failing_leaves_none_and_does_not_cache: PASS")


def test_panstarrs_not_asked_south_of_its_footprint():
    ls, ps1 = _Fake([]), _Fake([])
    restore = _with_fakes(ls, ps1)
    try:
        t = _candidates()
        t["DELTA_J2000"] = [-45.0, -45.0, -10.0, -10.0]
        galaxy_flag.flag_known_galaxies(t)
    finally:
        restore()
    assert len(ls.calls[0]) == 4
    assert [c[1] for c in ps1.calls[0]] == [-10.0, -10.0]
    print("test_panstarrs_not_asked_south_of_its_footprint: PASS")


def test_empty_table_gets_columns():
    t = galaxy_flag.flag_known_galaxies(_candidates()[:0])
    for col in ("galaxy_flag", "galaxy_sep_arcsec", "galaxy_mag", "galaxy_catalog"):
        assert col in t.colnames
    print("test_empty_table_gets_columns: PASS")


if __name__ == "__main__":
    test_classifies_galaxy_host_and_unmatched()
    test_band_selects_the_galaxy_magnitude()
    test_cache_queries_only_new_candidates()
    test_one_failed_catalogue_still_uses_the_other()
    test_all_catalogues_failing_leaves_none_and_does_not_cache()
    test_panstarrs_not_asked_south_of_its_footprint()
    test_empty_table_gets_columns()
    print("All galaxy_flag.py tests passed.")
