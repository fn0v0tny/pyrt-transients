"""A per-epoch `<epoch>_transients.ecsv` is a permanent cache: Step 1 skips
any epoch that already has one, and nothing ever recomputes it. So an epoch
produced while an external service was down -- a reference catalogue that
failed to download, a SkyBoT outage -- used to freeze that outage into the
observation forever, with no trace that the result was ever incomplete.

These cover the marking (save_epoch_results stamps meta['DEGRADED']), the
re-run decision (epoch_is_cached), and the distinction the marking rests on:
a service that answered "nothing here" is not a service that was down.
"""
import numpy as np
import pytest
from astropy.table import Table
from astropy.time import Time

from pyrt_transient import PipelineConfig
from pyrt_transient.detection.blind_multicatalog import clustering


def _cands(n=1):
    return Table({
        "ALPHA_J2000": np.linspace(10.0, 10.1, n), "DELTA_J2000": np.full(n, 20.0),
        "MAG_CALIB": np.full(n, 16.0), "MAGERR_CALIB": np.full(n, 0.02),
        "X_IMAGE": np.full(n, 100.0), "Y_IMAGE": np.full(n, 100.0),
        "FWHM_IMAGE": np.full(n, 3.0), "FLAGS": np.zeros(n, dtype=int),
        "quality_score": np.full(n, 1.0), "candidate_type": ["new"] * n,
        "reference_catalog": ["gaia"] * n,
    })


def _det_table():
    t = Table({"ALPHA_J2000": [10.0], "DELTA_J2000": [20.0]})
    t.meta = {"CTIME": 1_623_000_000.0, "EXPTIME": 10.0, "filename": "e1.ecsv"}
    return t


def test_epoch_is_cached_only_for_a_clean_result(tmp_path):
    clean = tmp_path / "clean_transients.ecsv"
    _cands().write(str(clean), format="ascii.ecsv")
    assert clustering.epoch_is_cached(clean)

    degraded = tmp_path / "degraded_transients.ecsv"
    t = _cands()
    t.meta["DEGRADED"] = "gaia failed to load"
    t.write(str(degraded), format="ascii.ecsv")
    assert not clustering.epoch_is_cached(degraded)

    assert not clustering.epoch_is_cached(tmp_path / "missing_transients.ecsv")


def test_save_epoch_results_marks_a_failed_catalogue(tmp_path):
    cfg = PipelineConfig()
    cfg.detection.vsx_filter_enabled = False   # keep the test off the network
    clustering.save_epoch_results(
        {"gaia": _cands()}, _det_table(), 0, min_catalogs=1.0, min_quality=0.1,
        data_dir=tmp_path, config=cfg,
        degraded_reason="reference catalogue(s) failed to load: ['usno']",
    )
    written = next(tmp_path.glob("*_transients.ecsv"))
    assert "usno" in Table.read(str(written)).meta["DEGRADED"]
    # ... and that is exactly what makes the next run redo the epoch
    assert not clustering.epoch_is_cached(written)


def test_save_epoch_results_leaves_a_clean_epoch_cached(tmp_path):
    cfg = PipelineConfig()
    cfg.detection.vsx_filter_enabled = False
    clustering.save_epoch_results(
        {"gaia": _cands()}, _det_table(), 0, min_catalogs=1.0, min_quality=0.1,
        data_dir=tmp_path, config=cfg,
    )
    written = next(tmp_path.glob("*_transients.ecsv"))
    assert "DEGRADED" not in Table.read(str(written)).meta
    assert clustering.epoch_is_cached(written)


def test_skybot_outage_marks_the_epoch_but_an_empty_answer_does_not(monkeypatch):
    """stdpipe raises KeyError when SkyBoT genuinely finds no solar-system
    objects -- an ordinary, complete result. Everything else raising means
    the cross-match never happened. Only the second is degradation."""
    from pyrt_transient.detection.blind_multicatalog import stdpipe_filters
    cfg = PipelineConfig()
    cfg.detection.vsx_filter_enabled = True
    time = Time("2021-06-20T00:00:00")

    monkeypatch.setattr(stdpipe_filters.stdpipe_pipeline, "filter_transient_candidates",
                        lambda *a, **k: (_ for _ in ()).throw(KeyError("RA")))
    out = clustering.combine_results({"gaia": _cands()}, min_catalogs_fraction=1.0,
                                     min_quality=0.1, config=cfg, time=time)
    assert "DEGRADED" not in out.meta

    monkeypatch.setattr(stdpipe_filters.stdpipe_pipeline, "filter_transient_candidates",
                        lambda *a, **k: (_ for _ in ()).throw(ValueError("No table found")))
    out = clustering.combine_results({"gaia": _cands()}, min_catalogs_fraction=1.0,
                                     min_quality=0.1, config=cfg, time=time)
    assert "SkyBoT" in out.meta["DEGRADED"]


def test_an_epoch_with_no_time_is_not_degraded(monkeypatch):
    """No CTIME/JD at all is a permanent property of the data, not an
    outage: marking it degraded would recompute the epoch on every run,
    forever."""
    cfg = PipelineConfig()
    cfg.detection.vsx_filter_enabled = True
    out = clustering.combine_results({"gaia": _cands()}, min_catalogs_fraction=1.0,
                                     min_quality=0.1, config=cfg, time=None)
    assert "DEGRADED" not in out.meta


def test_detections_time_falls_back_and_gives_up_honestly():
    """It used to default a missing CTIME to 0, i.e. query SkyBoT at
    1970-01-01: a valid-looking Time that rejected nothing while the caller
    believed the epoch had been cross-matched. The subtraction pipeline's
    difference epochs carry JD/MJD-OBS rather than CTIME."""
    t = Table({"a": [1]})
    t.meta = {"CTIME": 1_623_000_000.0, "EXPTIME": 20.0}
    assert clustering.detections_time(t).unix == pytest.approx(1_623_000_010.0)

    t.meta = {"JD": 2459386.5}
    assert clustering.detections_time(t).jd == pytest.approx(2459386.5)

    t.meta = {"MJD-OBS": 59386.0}
    assert clustering.detections_time(t).mjd == pytest.approx(59386.0)

    t.meta = {"EXPTIME": 10.0}
    assert clustering.detections_time(t) is None
    t.meta = {"CTIME": "not a time"}
    assert clustering.detections_time(t) is None
