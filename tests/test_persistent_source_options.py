"""The four opt-in options that make constant uncatalogued sources
detectable without admitting catalogue-incompleteness bogus (see
FUTURE_IDEAS.md "Constant new sources"): defaults must reproduce the
historical behaviour exactly; the options must do what they say."""
import numpy as np
import pytest
from astropy.table import Table

from pyrt_transient.config_trans import DetectionConfig
from pyrt_transient.core.scoring import compute_lightcurve_score_factor
from pyrt_transient.catalog import CatTransients
from pyrt_transient.detection.blind_multicatalog.catalog_match import _match_floor_px


def _feat(candidate_type, mag_range, n=3):
    return {"weighted_mean_mag": 15.0, "mag_range": mag_range, "n_detections": n,
            "candidate_type": candidate_type}


def test_variability_floor_default_off_reproduces_historical_product():
    w = DetectionConfig()
    f = compute_lightcurve_score_factor(_feat("new", 0.03), w)
    assert f == pytest.approx(min(0.03 / 0.5, 3.0) * 0.03)     # ~1.8e-3


def test_variability_floor_only_lifts_new_sources_and_only_upwards():
    w = DetectionConfig(new_source_variability_floor=True)
    assert compute_lightcurve_score_factor(_feat("new", 0.03), w) == pytest.approx(1.0)
    # A really varying new source keeps its boost.
    assert compute_lightcurve_score_factor(_feat("new", 1.5), w) == pytest.approx(3.0 * 1.5)
    # Catalogued-star types are untouched.
    assert compute_lightcurve_score_factor(_feat("brightening", 0.03), w) == pytest.approx(
        min(0.03 / 0.5, 3.0) * 0.03)


def test_config_dataclass_has_new_fields_with_conservative_defaults():
    d = DetectionConfig()
    assert d.unphotometered_match_is_new is True
    assert d.catalog_match_floor_arcsec == {}
    assert d.new_source_variability_floor is False


def test_strip_gaia_quality_cuts():
    q = """
            SELECT source_id, ra
            FROM gaiadr3.gaia_source
            WHERE 1=CONTAINS(POINT('ICRS', ra, dec), BOX('ICRS', 1, 2, 3, 4))
                AND phot_g_mean_mag < 20.0
                AND ruwe < 1.4
                AND visibility_periods_used >= 8
                -- Ensure we only get complete photometric data
                AND phot_bp_mean_mag IS NOT NULL
                AND phot_rp_mean_flux_over_error > 0
    """
    out = CatTransients._strip_gaia_quality_cuts(q)
    assert "ruwe" not in out and "visibility_periods_used" not in out
    assert "IS NOT NULL" not in out and "flux_over_error" not in out
    assert "phot_g_mean_mag < 20.0" in out and "CONTAINS" in out


def test_gaia_full_registered_like_gaia():
    assert "gaia_full" in CatTransients.KNOWN_CATALOGS
    assert CatTransients.KNOWN_CATALOGS["gaia_full"]["filters"] == CatTransients.KNOWN_CATALOGS["gaia"]["filters"]
    assert CatTransients.KNOWN_CATALOGS["gaia_full"]["cacheable"] is True


class _Cfg:
    def __init__(self, floors):
        self.detection = DetectionConfig(catalog_match_floor_arcsec=floors)


def test_match_floor_px_uses_plate_scale_and_substring_names():
    det = Table({"X_IMAGE": [1.0]}); det.meta = {"CD1_1": -1.0 / 3600, "CD2_2": 1.0 / 3600}  # 1"/px
    assert _match_floor_px(_Cfg({"usno": 2.0}), "usno", det) == pytest.approx(2.0)
    assert _match_floor_px(_Cfg({"usno": 2.0}), "USNO-B1", det) == pytest.approx(2.0)
    assert _match_floor_px(_Cfg({"usno": 2.0}), "gaia", det) is None
    assert _match_floor_px(_Cfg({}), "usno", det) is None


def test_unphotometered_match_is_not_new_when_disabled():
    """_check_magnitude_changes_cached with matches whose photometry is all
    invalid: historical -> ('new'); option off -> matched, not a candidate."""
    from pyrt_transient.catalog import CatalogOptimizationCache
    cat = CatTransients.__new__(CatTransients)
    cat._photometric_cache = CatalogOptimizationCache(
        coordinates=np.zeros((2, 2)), pixel_coordinates={},
        magnitudes=np.full((2, 5), np.nan), colors=np.full((2, 4), np.nan),
        valid_stars=np.array([False, False]), kdtrees={})
    cat.meta = {"catalog_props": {"catalog_name": "usno"}}
    args = (np.array([0, 1]), 16.0, 0.02, "P0=25.0", 1.0, 5.0)
    assert cat._check_magnitude_changes_cached(*args)[:2] == (True, "new")
    is_cand, ctype, _ = cat._check_magnitude_changes_cached(*args, unphotometered_match_is_new=False)
    assert (is_cand, ctype) == (False, "matched_unphotometered")


def _cache_without_photometry(rough):
    from pyrt_transient.catalog import CatalogOptimizationCache
    n = len(rough)
    return CatalogOptimizationCache(
        coordinates=np.zeros((n, 2)), pixel_coordinates={},
        magnitudes=np.full((n, 5), np.nan), colors=np.full((n, 4), np.nan),
        valid_stars=np.zeros(n, dtype=bool), kdtrees={},
        rough_mags=np.asarray(rough, dtype=float))


def test_unphotometered_veto_is_magnitude_aware():
    """A USNO-B star with only plate magnitudes vetoes a detection of
    comparable brightness but not one 4.5 mag brighter (GRB 250813B)."""
    cat = CatTransients.__new__(CatTransients)
    cat._photometric_cache = _cache_without_photometry([19.7, 20.3])
    cat.meta = {"catalog_props": {"catalog_name": "usno"}}
    common = ("P0=25.0", 1.0, 5.0)
    kw = dict(unphotometered_match_is_new=False, unphotometered_veto_max_brightening=2.0)
    matches = np.array([0, 1])
    # 15.5 mag detection: 4.2 mag brighter than the brightest match -> brightening
    is_cand, ctype, diff = cat._check_magnitude_changes_cached(matches, 15.5, 0.05, *common, **kw)
    assert (is_cand, ctype) == (True, "brightening")
    assert diff == pytest.approx(15.5 - 19.7)
    # 19.0 mag detection: within the margin -> still vetoed
    assert cat._check_magnitude_changes_cached(matches, 19.0, 0.05, *common, **kw)[:2] == (False, "matched_unphotometered")
    # option off -> purely positional veto as before
    kw_off = dict(unphotometered_match_is_new=False, unphotometered_veto_max_brightening=None)
    assert cat._check_magnitude_changes_cached(matches, 15.5, 0.05, *common, **kw_off)[:2] == (False, "matched_unphotometered")
    # no rough magnitude available at all -> positional veto
    cat._photometric_cache = _cache_without_photometry([np.nan, np.nan])
    assert cat._check_magnitude_changes_cached(matches, 15.5, 0.05, *common, **kw)[:2] == (False, "matched_unphotometered")
    # historical behaviour untouched
    assert cat._check_magnitude_changes_cached(matches, 15.5, 0.05, *common)[:2] == (True, "new")


def test_rough_magnitudes_prefers_red_and_skips_missing():
    from astropy.table import Table
    t = Table({"R2": [19.7, np.nan, np.nan], "B1": [21.0, 21.5, np.nan], "G": [np.nan, 18.2, np.nan]})
    rough = CatTransients.rough_magnitudes(t)
    assert rough[0] == pytest.approx(19.7)   # R2 preferred over B1
    assert rough[1] == pytest.approx(18.2)   # G before B1
    assert np.isnan(rough[2])
    assert DetectionConfig().unphotometered_veto_max_brightening_mag == pytest.approx(2.0)


def test_magnitude_comparison_uses_frame_band():
    """An i-band frame is compared against the catalogue's Sloan i, not r."""
    from pyrt_transient.catalog import CatalogOptimizationCache
    assert CatTransients.catalog_band_index("Sloan_i") == 2
    assert CatTransients.catalog_band_index("N") == 1 and CatTransients.catalog_band_index(None) == 1
    cat = CatTransients.__new__(CatTransients)
    # one star: g=16.0, r=15.0, i=14.2, z=14.0, J=13.5 (a red star, r-i = 0.8)
    mags = np.array([[16.0, 15.0, 14.2, 14.0, 13.5]])
    cat._photometric_cache = CatalogOptimizationCache(
        coordinates=np.zeros((1, 2)), pixel_coordinates={},
        magnitudes=mags, colors=np.array([[1.0, 0.8, 0.2, 0.5]]),
        valid_stars=np.array([True]), kdtrees={})
    cat.meta = {"catalog_props": {"catalog_name": "atlas"}}
    common = ("Z=25.0", 0.5, 5.0)   # response without colour terms, 0.5 mag threshold
    # detection at i = 14.2 with a tiny error: against r it is a 0.8 mag "brightening"
    assert cat._check_magnitude_changes_cached(np.array([0]), 14.2, 0.02, *common, band_idx=1)[:2] == (True, "brightening")
    # against the frame's own band it is the same star
    assert cat._check_magnitude_changes_cached(np.array([0]), 14.2, 0.02, *common, band_idx=2)[:2] == (False, "none")
    # catalogue without that band falls back to r
    cat._photometric_cache.magnitudes[0, 2] = np.nan
    assert cat._check_magnitude_changes_cached(np.array([0]), 15.0, 0.02, *common, band_idx=2)[:2] == (False, "none")
