"""Campaign-level magnitude-error floor (lightcurve.estimate_magnitude_error_floor)
and its use in the light-curve variability statistics."""
import numpy as np
from astropy.table import Table

from pyrt_transient.config_trans import DetectionConfig
from pyrt_transient.detection.blind_multicatalog.lightcurve import (
    estimate_magnitude_error_floor,
    update_candidate_with_lightcurve_stats,
)


def _campaign(n_epochs=12, n_bright=30, n_faint=40, floor=0.03, seed=1):
    """Synthetic epochs: bright stars with tiny quoted errors but `floor`
    true scatter, faint stars whose scatter equals their quoted error."""
    rng = np.random.default_rng(seed)
    n = n_bright + n_faint
    ra = 100.0 + rng.uniform(0, 0.2, n)
    dec = 20.0 + rng.uniform(0, 0.2, n)
    mag = np.concatenate([rng.uniform(12, 14, n_bright), rng.uniform(17, 18.5, n_faint)])
    err = np.concatenate([np.full(n_bright, 0.005), np.full(n_faint, 0.12)])
    true_sigma = np.sqrt(err ** 2 + np.where(np.arange(n) < n_bright, floor ** 2, 0.0))
    epochs = []
    for k in range(n_epochs):
        t = Table({
            "ALPHA_J2000": ra + rng.normal(0, 0.3 / 3600, n),
            "DELTA_J2000": dec + rng.normal(0, 0.3 / 3600, n),
            "MAG_CALIB": mag + rng.normal(0, 1, n) * true_sigma,
            "MAGERR_CALIB": err,
            "FLAGS": np.zeros(n, int),
            "obs_time": np.full(n, 1.7e9 + 20.0 * k),
            "epoch_id": np.full(n, k),
        })
        epochs.append(t)
    return epochs


def test_floor_recovers_injected_scatter():
    floor, n = estimate_magnitude_error_floor(_campaign(floor=0.03))
    assert n >= 10
    assert abs(floor - 0.03) < 0.008


def test_floor_is_zero_when_errors_are_right():
    floor, n = estimate_magnitude_error_floor(_campaign(floor=0.0))
    assert n >= 10
    assert floor < 0.006


def test_floor_nan_when_too_few_stars_or_epochs():
    floor, n = estimate_magnitude_error_floor(_campaign(n_epochs=2))
    assert np.isnan(floor)
    floor, n = estimate_magnitude_error_floor(_campaign(n_bright=3, n_faint=3))
    assert np.isnan(floor)


def test_stats_use_floor_for_chi2_but_not_for_weighted_mean():
    rng = np.random.default_rng(3)
    n = 40
    lc = Table({
        "ALPHA_J2000": np.full(n, 100.0), "DELTA_J2000": np.full(n, 20.0),
        "MAG_CALIB": 13.0 + rng.normal(0, 0.03, n),
        "MAGERR_CALIB": np.full(n, 0.005),
        "obs_time": 1.7e9 + 20.0 * np.arange(n), "epoch_id": np.arange(n),
        "FWHM_IMAGE": np.full(n, 2.5), "ELLIPTICITY": np.full(n, 0.1), "FLAGS": np.zeros(n, int),
    })
    raw = Table({"candidate_type": ["new"]})
    update_candidate_with_lightcurve_stats(raw, lc, magerr_floor=0.0)
    floored = Table({"candidate_type": ["new"]})
    update_candidate_with_lightcurve_stats(floored, lc, magerr_floor=0.03)
    assert raw["mag_chi2_reduced"][0] > 10          # constant star looks wildly variable
    assert floored["mag_chi2_reduced"][0] < 3       # ... and not with the floor
    assert floored["mag_chi2_reduced_raw"][0] == raw["mag_chi2_reduced"][0]
    assert floored["magerr_floor"][0] == 0.03
    assert bool(raw["is_variable"][0]) and not bool(floored["is_variable"][0])
    # the score inputs are untouched
    assert floored["mag_weighted_mean"][0] == raw["mag_weighted_mean"][0]
    assert floored["mag_range"][0] == raw["mag_range"][0]


def test_config_defaults():
    d = DetectionConfig()
    assert d.magerr_floor_auto is True
    assert d.magerr_floor_mag == 0.05
