"""catalog._remember: the per-image pixel-coordinate and KD-tree caches
stay bounded, so a run over hundreds of epochs no longer grows by one
catalogue copy per epoch."""
import numpy as np

from pyrt_transient import catalog as cat


def test_remember_evicts_oldest_beyond_limit():
    d = {}
    for i in range(cat.PER_IMAGE_CACHE_LIMIT + 3):
        out = cat._remember(d, f"img{i}", i)
        assert out == i
    assert len(d) == cat.PER_IMAGE_CACHE_LIMIT
    assert list(d) == [f"img{i}" for i in range(3, cat.PER_IMAGE_CACHE_LIMIT + 3)]
    # Re-inserting an existing key keeps the size.
    cat._remember(d, "img5", 55)
    assert len(d) == cat.PER_IMAGE_CACHE_LIMIT and d["img5"] == 55


def test_build_spatial_index_cache_is_bounded():
    c = cat.Catalog.__new__(cat.Catalog)
    c._photometric_cache = None
    c._kdtree_cache = {}
    c._cache_enabled = True
    for i in range(cat.PER_IMAGE_CACHE_LIMIT + 4):
        pts = np.random.default_rng(i).random((20, 2))
        tree = c.build_spatial_index(pts, index_id=f"frame{i}")
        assert tree is c._kdtree_cache[f"frame{i}"]
    assert len(c._kdtree_cache) == cat.PER_IMAGE_CACHE_LIMIT
