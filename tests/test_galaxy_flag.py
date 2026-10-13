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


def _src(ra, dec, mag, catalog, radius=None):
    return {"ra": ra, "dec": dec, "catalog": catalog,
            "mags": {"g": mag + 0.5, "r": mag, "i": mag - 0.3, "z": None},
            "radius_arcsec": radius}


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
    assert len(t) == 4, "flag_known_galaxies labels only, never removes"
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
            assert set(cache) == {"t0", "t1", "t2", "t3", galaxy_flag.CACHE_META_KEY}
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
    for col in ("galaxy_flag", "galaxy_sep_arcsec", "galaxy_mag", "galaxy_dmag",
                "galaxy_radius_arcsec", "galaxy_catalog"):
        assert col in t.colnames
    t, n = galaxy_flag.remove_constant_galaxies(t)
    assert n == 0 and len(t) == 0
    print("test_empty_table_gets_columns: PASS")


def test_fainter_than_the_galaxy_is_a_separate_source():
    # 19.0 measured on a 17.0 galaxy: at this pixel scale the galaxy's own
    # light would have come out at ~17, so this is something projected on
    # it, not the galaxy -- labelled, never removed.
    t = _candidates()
    t["MAG_CALIB"] = [19.0, 15.5, 18.0, 18.0]
    restore = _with_fakes(_Fake(SOURCES), _Fake([]))
    try:
        t = galaxy_flag.flag_known_galaxies(t, band="r")
    finally:
        restore()
    assert list(t["galaxy_flag"]) == ["near_galaxy", "host", "none", "none"], list(t["galaxy_flag"])
    assert abs(t["galaxy_dmag"][0] - 2.0) < 1e-9 and abs(t["galaxy_dmag"][1] + 1.5) < 1e-9
    # The window is configurable: 17.6 on 17.0 is the galaxy within 0.7, 17.8 is not.
    restore = _with_fakes(_Fake(SOURCES), _Fake([]))
    try:
        t = galaxy_flag.flag_known_galaxies(_candidates(), band="r", faint_margin_mag=0.7)
        assert t["galaxy_flag"][0] == "galaxy", "17.6 on 17.0 is within 0.7"
        t2 = _candidates(); t2["MAG_CALIB"][0] = 17.8
        t2 = galaxy_flag.flag_known_galaxies(t2, band="r", faint_margin_mag=0.7)
        assert t2["galaxy_flag"][0] == "near_galaxy"
    finally:
        restore()
    # A resolved galaxy (half-light radius above the seeing) has no faint
    # side: its Kron magnitude can fall any amount short of the total.
    big = _src(10.0, 20.0 + 0.5 / 3600, 13.8, "legacysurvey", radius=8.9)
    t = _candidates()[:1]
    t["MAG_CALIB"] = [15.1]
    for seeing, expected in ((4.2, "galaxy"), (10.0, "near_galaxy"), (None, "galaxy")):
        restore = _with_fakes(_Fake([big]), _Fake([]))
        try:
            t = galaxy_flag.flag_known_galaxies(t, band="r", seeing_arcsec=seeing)
        finally:
            restore()
        assert t["galaxy_flag"][0] == expected, (seeing, t["galaxy_flag"][0])
    print("test_fainter_than_the_galaxy_is_a_separate_source: PASS")


def test_match_radius_widens_with_the_half_light_radius():
    # Galaxy 3 sits 3" from candidate t3: outside a 2" cone, but it is a
    # 6" galaxy, so radius_scale=1 matches it (and, resolved, it is "galaxy"
    # however faint the measurement); radius_scale=0 does not match.
    big = _src(10.3, 20.0 + 3.0 / 3600, 16.0, "legacysurvey", radius=6.0)
    sources = [s for s in SOURCES if s["ra"] != 10.3] + [big]
    for scale, expected in ((1.0, "galaxy"), (0.0, "none"), (0.3, "none")):
        restore = _with_fakes(_Fake(sources), _Fake([]))
        try:
            t = galaxy_flag.flag_known_galaxies(_candidates(), band="r", radius_scale=scale)
        finally:
            restore()
        assert t["galaxy_flag"][3] == expected, (scale, t["galaxy_flag"][3])
    assert abs(t["galaxy_radius_arcsec"][3]) != 6.0 or expected != "none"
    # The widening never exceeds the query cone.
    restore = _with_fakes(_Fake(sources), _Fake([]))
    try:
        t = galaxy_flag.flag_known_galaxies(_candidates(), band="r", radius_scale=5.0,
                                            query_radius_arcsec=2.5)
    finally:
        restore()
    assert t["galaxy_flag"][3] == "none"
    # The closest galaxy in units of its own radius wins: a 10" galaxy 3"
    # away over a point-like one 2" away.
    small = _src(10.3, 20.0 + 2.0 / 3600, 18.0, "panstarrs")
    restore = _with_fakes(_Fake([big]), _Fake([small]))
    try:
        t = galaxy_flag.flag_known_galaxies(_candidates()[3:], band="r")
    finally:
        restore()
    assert t["galaxy_catalog"][0] == "legacysurvey" and abs(t["galaxy_radius_arcsec"][0] - 6.0) < 1e-9
    print("test_match_radius_widens_with_the_half_light_radius: PASS")


def test_cache_is_discarded_when_the_query_cone_changes():
    ls = _Fake(SOURCES)
    restore = _with_fakes(ls, _Fake([]))
    try:
        with tempfile.TemporaryDirectory() as d:
            galaxy_flag.flag_known_galaxies(_candidates(), cache_dir=Path(d), query_radius_arcsec=10.0)
            galaxy_flag.flag_known_galaxies(_candidates(), cache_dir=Path(d), query_radius_arcsec=10.0)
            assert len(ls.calls) == 1, "same cone: served from the cache"
            galaxy_flag.flag_known_galaxies(_candidates(), cache_dir=Path(d), query_radius_arcsec=20.0)
            assert len(ls.calls) == 2, "wider cone: the old lists are incomplete"
            # A cache from before the radius was recorded is discarded too.
            (Path(d) / galaxy_flag.CACHE_FILENAME).write_text(json.dumps({"t0": []}))
            galaxy_flag.flag_known_galaxies(_candidates(), cache_dir=Path(d), query_radius_arcsec=20.0)
            assert len(ls.calls) == 3
    finally:
        restore()
    print("test_cache_is_discarded_when_the_query_cone_changes: PASS")


def test_remove_constant_galaxies():
    t = Table({
        "transient_id": ["const", "variable", "host", "near", "unmatched", "forced", "nochi2"],
        "NUMBER": [5, 6, 7, 8, 9, 0, 11],
        "galaxy_flag": ["galaxy", "galaxy", "host", "near_galaxy", "none", "galaxy", "galaxy"],
        "mag_chi2_reduced": [1.2, 4.5, 0.8, 0.9, 1.0, 0.5, np.nan],
        "quality_score": [10.0, 12.0, 20.0, 8.0, 9.0, 7.0, 6.0],
    })
    out, n = galaxy_flag.remove_constant_galaxies(t, max_chi2=2.0)
    assert n == 2, n
    assert list(out["transient_id"]) == ["variable", "host", "near", "unmatched", "forced"], list(out["transient_id"])
    assert len(t) == 7, "input untouched"
    # A tighter chi2 keeps more; without the column every 'galaxy' is constant.
    out, n = galaxy_flag.remove_constant_galaxies(t, max_chi2=1.0)
    assert n == 1 and "const" in list(out["transient_id"])
    t.remove_column("mag_chi2_reduced")
    out, n = galaxy_flag.remove_constant_galaxies(t)
    assert n == 3 and list(out["transient_id"]) == ["host", "near", "unmatched", "forced"]
    print("test_remove_constant_galaxies: PASS")


def test_galaxies_in_the_error_circle_are_kept():
    # EP261006a: the host (a constant 17.6 mag 'galaxy') sits 1.3" from
    # the FXT position, well inside the 3.1' WXT circle; a galaxy 10'
    # away is not a host candidate and goes.
    t = Table({
        "transient_id": ["host_gal", "far_gal", "far_var"],
        "ALPHA_J2000": [21.7917, 21.7917 + 10.0 / 60 / np.cos(np.radians(12.9)), 21.9],
        "DELTA_J2000": [12.8908, 12.8908, 12.9],
        "galaxy_flag": ["galaxy", "galaxy", "galaxy"],
        "mag_chi2_reduced": [1.1, 1.0, 5.0],
    })
    out, n = galaxy_flag.remove_constant_galaxies(t, keep_within=(21.806, 12.9, 0.0519 * 3600))
    assert n == 1 and list(out["transient_id"]) == ["host_gal", "far_var"], list(out["transient_id"])
    out, n = galaxy_flag.remove_constant_galaxies(t, keep_within=None)
    assert n == 2
    out, n = galaxy_flag.remove_constant_galaxies(t, keep_within=(np.nan, 12.9, 100.0))
    assert n == 2, "an unusable circle keeps nothing extra"
    print("test_galaxies_in_the_error_circle_are_kept: PASS")


if __name__ == "__main__":
    test_classifies_galaxy_host_and_unmatched()
    test_band_selects_the_galaxy_magnitude()
    test_cache_queries_only_new_candidates()
    test_one_failed_catalogue_still_uses_the_other()
    test_all_catalogues_failing_leaves_none_and_does_not_cache()
    test_panstarrs_not_asked_south_of_its_footprint()
    test_empty_table_gets_columns()
    test_fainter_than_the_galaxy_is_a_separate_source()
    test_match_radius_widens_with_the_half_light_radius()
    test_cache_is_discarded_when_the_query_cone_changes()
    test_remove_constant_galaxies()
    test_galaxies_in_the_error_circle_are_kept()
    print("All galaxy_flag.py tests passed.")
