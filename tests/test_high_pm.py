"""Final-candidate veto against propagated high proper-motion Gaia stars
(pyrt_transient/detection/high_pm.py). Uses the per-field cache file, so no
network is needed."""
import logging

import numpy as np
import pytest
from astropy.table import Table

from pyrt_transient.config_trans import PipelineConfig
from pyrt_transient.detection.high_pm import (
    HIGH_PM_CACHE_NAME, _field_geometry, apply_high_pm_veto, reject_high_pm_stars,
)

# LSPM J2003+6241 as Gaia DR3 gives it (epoch 2016.0); by 2026.65 it has moved
# to 300.96318 +62.69680, where the D50 measured it (cross-night page, 2026-09).
GAIA_ROW = {"RA_ICRS": 300.96401875602, "DE_ICRS": 62.69707236352,
            "pmRA": -130.047, "pmDE": -92.208, "Gmag": 13.904}
JD_2026_65 = 2451545.0 + (2026.65 - 2000.0) * 365.25


def _cache(tmp_path):
    path = tmp_path / HIGH_PM_CACHE_NAME
    Table({k: [v] for k, v in GAIA_ROW.items()}).write(str(path), format="ascii.ecsv")
    return path


def _candidates():
    return Table({"NUMBER": [12, 13, 0],
                  "ALPHA_J2000": [300.96319, 300.98000, 300.96319],
                  "DELTA_J2000": [62.69681, 62.70000, 62.69681],
                  "transient_id": ["at_star", "elsewhere", "forced_at_star"]})


def test_reject_high_pm_stars_drops_the_propagated_position_only(tmp_path):
    cache = _cache(tmp_path)
    out, n = reject_high_pm_stars(_candidates(), 300.96, 62.70, 0.4, JD_2026_65, logging.getLogger("t"),
                                  pm_threshold_masyr=50.0, match_radius_arcsec=3.0, cache_path=cache)
    assert n == 1 and out["transient_id"].tolist() == ["elsewhere", "forced_at_star"]
    # At the catalogue epoch the same candidate is 1.7" from the 2016 position:
    # still inside 3", but outside 1".
    out, n = reject_high_pm_stars(_candidates(), 300.96, 62.70, 0.4, JD_2026_65, logging.getLogger("t"),
                                  pm_threshold_masyr=50.0, match_radius_arcsec=1.0, cache_path=cache)
    assert n == 1
    out, n = reject_high_pm_stars(_candidates(), 300.96, 62.70, 0.4, 2451545.0 + 16 * 365.25,
                                  logging.getLogger("t"), match_radius_arcsec=1.0, cache_path=cache)
    assert n == 0, "at epoch 2016 the star has not moved to where the 2026 detection is"


def test_field_geometry_from_header_and_fallbacks():
    det = Table({"ALPHA_J2000": [10.0, 10.2], "DELTA_J2000": [20.0, 20.2]})
    det.meta.update({"CRVAL1": 10.1, "CRVAL2": 20.1, "FIELD": 0.237, "JD": 2461000.0})
    assert _field_geometry(det) == (10.1, 20.1, pytest.approx(0.474), 2461000.0)
    det.meta.clear()
    det.meta.update({"CTIME": 946728000, "CD1_1": -0.00033, "IMAGEW": 1024})
    ra, dec, field, jd = _field_geometry(det)
    assert (ra, dec) == (10.1, 20.1) and field == pytest.approx(0.00033 * 1024 * 1.5) and jd == pytest.approx(2451545.0)
    det.meta.clear()
    assert _field_geometry(det)[3] is None


def test_apply_high_pm_veto_uses_latest_epoch_and_config(tmp_path):
    _cache(tmp_path)
    config = PipelineConfig()
    old = Table({"X_IMAGE": [1.0]}, meta={"CRVAL1": 300.96, "CRVAL2": 62.70, "FIELD": 0.24,
                                          "JD": 2451545.0 + 16 * 365.25})
    new = Table({"X_IMAGE": [1.0]}, meta={"CRVAL1": 300.96, "CRVAL2": 62.70, "FIELD": 0.24, "JD": JD_2026_65})
    stack = Table({"X_IMAGE": [1.0]}, meta={"IS_STACK": True, "JD": JD_2026_65 + 1000})
    out, n = apply_high_pm_veto(_candidates(), [old, new, stack], tmp_path, config)
    assert n == 1 and out["transient_id"].tolist() == ["elsewhere", "forced_at_star"]
    config.detection.high_pm_threshold_masyr = 500.0     # the star is below this
    out, n = apply_high_pm_veto(_candidates(), [old, new], tmp_path, config)
    assert n == 0 and len(out) == 3
    # No header at all: skipped, nothing lost.
    out, n = apply_high_pm_veto(_candidates(), [Table({"X_IMAGE": [1.0]})], tmp_path, config)
    assert n == 0 and len(out) == 3
    assert (tmp_path / HIGH_PM_CACHE_NAME).exists()
