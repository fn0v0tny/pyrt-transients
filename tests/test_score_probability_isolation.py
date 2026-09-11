"""p_real calibration column and the magnitude-limited isolation statistic."""
import numpy as np
import pytest
from astropy.table import Table

from pyrt_transient.catalog import CatTransients, CatalogOptimizationCache
from pyrt_transient.config_trans import DetectionConfig
from pyrt_transient.core.scoring import add_score_probability, score_probability


def test_score_probability_is_logistic_in_ln_q():
    p = score_probability([1.0, np.e, 0.0, np.nan], intercept=0.0, slope=1.0)
    assert p[0] == pytest.approx(0.5)
    assert p[1] == pytest.approx(1 / (1 + np.exp(-1)))
    assert np.isnan(p[2]) and np.isnan(p[3])


def test_add_score_probability_only_with_constants():
    t = Table({"quality_score": [0.2, 20.0]})
    d = DetectionConfig()
    assert add_score_probability(t, d) is False and "p_real" not in t.colnames
    d.score_probability_intercept, d.score_probability_slope = 0.840, 0.843
    assert add_score_probability(t, d) is True
    assert 0.3 < t["p_real"][0] < 0.45 and t["p_real"][1] > 0.95


def _catalog_with_pixels(xy, rough):
    n = len(xy)
    cat = CatTransients.__new__(CatTransients)
    cat._photometric_cache = CatalogOptimizationCache(
        coordinates=np.zeros((n, 2)), pixel_coordinates={"img": np.asarray(xy, float)},
        magnitudes=np.full((n, 5), np.nan), colors=np.full((n, 4), np.nan),
        valid_stars=np.zeros(n, bool), kdtrees={}, rough_mags=np.asarray(rough, float))
    cat._cache_enabled = False
    cat._kdtree_cache = {}
    cat.meta = {"catalog_props": {"catalog_name": "gaia"}}
    return cat


def test_isolation_ignores_stars_fainter_than_limit():
    # a 20 mag star 2 px from the detection, a 15 mag star 30 px away
    cat = _catalog_with_pixels([[102.0, 100.0], [130.0, 100.0]], [20.0, 15.0])
    pos = np.array([[100.0, 100.0]])
    hist = cat.compute_local_statistics(pos, radius=50.0, image_id="img")
    assert hist["nearest_source_dist"][0] == pytest.approx(2.0)
    assert hist["nearby_sources"][0] == 2
    lim = cat.compute_local_statistics(pos, radius=50.0, image_id="img", max_mag=19.0)
    assert lim["nearest_source_dist"][0] == pytest.approx(30.0)
    assert lim["nearby_sources"][0] == 1
    # an entry without any magnitude is kept
    cat2 = _catalog_with_pixels([[102.0, 100.0], [130.0, 100.0]], [np.nan, 15.0])
    lim2 = cat2.compute_local_statistics(pos, radius=50.0, image_id="img", max_mag=19.0)
    assert lim2["nearest_source_dist"][0] == pytest.approx(2.0)
    assert DetectionConfig().isolation_max_mag_margin is None
