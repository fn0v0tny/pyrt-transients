"""`fill_missing_photometry_batch` must reproduce the per-star scalar version.

The scalar `fill_missing_photometry` is the reference implementation; the
batch one replaced a Python loop over every catalogue star (~35 s of CPU per
subprocess on a million-star Gaia query), so it has to agree row for row.
"""

import numpy as np
import pytest

from pyrt_transient.catalog import CatTransients


def _reference(mags, typical_colors=None):
    """Per-row scalar fill, in the shape the batch version returns."""
    out = np.asarray(mags, dtype=float).copy()
    ok = np.zeros(len(out), dtype=bool)
    for i, row in enumerate(np.asarray(mags, dtype=float)):
        filled = CatTransients.fill_missing_photometry(row.copy(), typical_colors)
        if filled is not None:
            out[i] = filled
            ok[i] = True
    return out, ok


def _check(mags, typical_colors=None):
    got, got_ok = CatTransients.fill_missing_photometry_batch(mags, typical_colors)
    want, want_ok = _reference(mags, typical_colors)
    np.testing.assert_array_equal(got_ok, want_ok)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12, equal_nan=True)
    return got, got_ok


def test_all_nan_patterns_five_bands():
    """Every possible NaN pattern of a 5-band row, at distinct magnitudes."""
    base = np.array([15.0, 16.5, 17.25, 18.0, 19.5])
    rows = []
    for pattern in range(2 ** 5):
        row = base.copy()
        for band in range(5):
            if pattern & (1 << band):
                row[band] = np.nan
        rows.append(row)
    _check(np.array(rows))


def test_random_catalog_matches_scalar():
    rng = np.random.default_rng(20260909)
    mags = rng.uniform(10.0, 21.0, size=(4000, 5))
    mags[rng.random(mags.shape) < 0.55] = np.nan
    got, got_ok = _check(mags)
    # The sample has to exercise both outcomes or it proves nothing.
    assert 0 < got_ok.sum() < len(mags)
    assert not np.isnan(got[got_ok]).any()


def test_rows_that_cannot_be_filled_are_returned_unchanged():
    mags = np.array([
        [np.nan] * 5,               # nothing known
        [np.nan, np.nan, 17.0, np.nan, np.nan],  # one band only
        [15.0, np.nan, np.nan, np.nan, 19.0],    # fillable
    ])
    got, ok = _check(mags)
    assert list(ok) == [False, False, True]
    np.testing.assert_array_equal(got[0], mags[0])
    np.testing.assert_array_equal(got[1], mags[1])


def test_gaia_shaped_catalog():
    """Realistic shape: only the blue end populated, i/z/J extrapolated."""
    rng = np.random.default_rng(7)
    n = 2000
    mags = np.full((n, 5), np.nan)
    mags[:, 0] = rng.uniform(12.0, 20.0, n)
    mags[:, 1] = mags[:, 0] - rng.uniform(0.0, 1.0, n)
    third = rng.random(n) < 0.5
    mags[third, 2] = mags[third, 1] - 0.3
    got, ok = _check(mags)
    assert ok.all()


@pytest.mark.parametrize("typical_colors", [
    [0.6, 0.3, 0.2, 0.8],
    [0.1, 0.2, 0.3, 0.4],
    [1.0, -0.5, 0.25, 0.0],
])
def test_alternative_typical_colors(typical_colors):
    rng = np.random.default_rng(11)
    mags = rng.uniform(10.0, 21.0, size=(1500, 5))
    mags[rng.random(mags.shape) < 0.5] = np.nan
    _check(mags, typical_colors)


@pytest.mark.parametrize("n_bands", [1, 2, 3, 4, 5, 6, 7])
def test_band_counts_other_than_five(n_bands):
    """The colour chain covers at most len(typical_colors) + 1 bands; extra
    bands fall through to the scalar version's carry-forward pass."""
    rng = np.random.default_rng(n_bands)
    mags = rng.uniform(10.0, 21.0, size=(500, n_bands))
    mags[rng.random(mags.shape) < 0.5] = np.nan
    _check(mags)


def test_empty_catalog():
    got, ok = CatTransients.fill_missing_photometry_batch(np.empty((0, 5)))
    assert got.shape == (0, 5)
    assert ok.shape == (0,)


def test_input_is_not_modified():
    mags = np.array([[15.0, np.nan, np.nan, np.nan, 19.0]])
    before = mags.copy()
    CatTransients.fill_missing_photometry_batch(mags)
    np.testing.assert_array_equal(mags, before)


def test_precompute_matches_scalar_loop():
    """End to end through precompute_photometric_data: the cached magnitudes
    and colors must match what the old per-star loop produced."""
    rng = np.random.default_rng(3)
    n = 800
    bands = ["Sloan_g", "Sloan_r", "Sloan_i", "Sloan_z", "J"]
    cols = {"radeg": rng.uniform(0, 360, n), "decdeg": rng.uniform(-90, 90, n)}
    for band in bands:
        vals = rng.uniform(10.0, 21.0, n)
        vals[rng.random(n) < 0.6] = np.nan
        cols[band] = vals
    cat = CatTransients(cols)

    cache = cat.precompute_photometric_data()

    raw = np.column_stack([np.asarray(cols[b], dtype=float) for b in bands])
    raw = np.where(np.isfinite(raw) & (raw < 99), raw, np.nan)
    want_mags = raw.copy()
    want_colors = np.full((n, 4), np.nan)
    valid = np.sum(~np.isnan(raw), axis=1) >= 2
    for i in np.where(valid)[0]:
        filled = CatTransients.fill_missing_photometry(raw[i].copy())
        if filled is not None:
            want_mags[i] = filled
            want_colors[i] = [filled[0] - filled[1], filled[1] - filled[2],
                              filled[2] - filled[3], filled[3] - filled[4]]

    np.testing.assert_array_equal(cache.valid_stars, valid)
    np.testing.assert_allclose(cache.magnitudes, want_mags, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(cache.colors, want_colors, atol=1e-12, equal_nan=True)
