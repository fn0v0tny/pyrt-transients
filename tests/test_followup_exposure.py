"""Tests for the follow-up exposure calculator (pyrt_transient/followup/).

The first two tests are the ones that matter: they pin the model against
real photometry from this repo's own fixtures, which is how both deviations
from the standalone `new_exposure_calculator2.py` were found in the first
place (see exposure.py's module docstring).
"""

import glob
import json
import os

import numpy as np
import pytest
from astropy.table import Table

from pyrt_transient.config_trans import PipelineConfig
from pyrt_transient.followup import enrichment
from pyrt_transient.followup import exposure as ex

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "210619B")


def _real_frames(limit=3):
    paths = sorted(glob.glob(os.path.join(FIXTURE_DIR, "*-N-010-df.ecsv")))
    paths = [p for p in paths if not p.endswith("_transients.ecsv")]
    if not paths:
        pytest.skip("210619B fixture not available")
    return paths[:limit]


def _real_table():
    return Table.read(_real_frames(1)[0], format="ascii.ecsv")


# --- model ------------------------------------------------------------------

def test_model_reproduces_real_photometric_errors():
    """Predicted magerror matches measured MAGERR_CALIB across every source
    in real D50 frames, to the calibration's own quoted 0.047 dex RMS.

    This is what caught the missing ZERO_OFFSET: without it the same check
    is off by a median of -2.5 dex (a factor of ~330).
    """
    for path in _real_frames():
        table = Table.read(path, format="ascii.ecsv")
        if "MAG_CALIB" not in table.colnames:
            pytest.skip(f"{path} carries no calibrated photometry")
        conditions = ex.ReferenceConditions.from_meta(table.meta, source=path)

        mag = np.asarray(table["MAG_CALIB"], dtype=float)
        err = np.asarray(table["MAGERR_CALIB"], dtype=float)
        good = np.isfinite(mag) & np.isfinite(err) & (err > 0)
        assert good.sum() > 100, "fixture should have plenty of measured sources"

        predicted = 10 ** ex.log_magerror(
            ex.instrumental_magnitude(mag[good], conditions.magzero),
            conditions.bgsigma_adu, conditions.fwhm_px, conditions.gain,
        )
        residual = np.log10(predicted / err[good])
        assert abs(np.median(residual)) < 0.1, f"{path}: biased by {np.median(residual):+.3f} dex"
        assert np.std(residual) < 0.047, f"{path}: rms {np.std(residual):.3f} dex"


def test_bgsigma_round_trips_at_the_reference_exposure():
    """Scaling the background to the reference frame's own exposure time
    must return the header BGSIGMA. The standalone script's pair of
    conversions returns GAIN*BGSIGMA instead.
    """
    for path in _real_frames():
        table = Table.read(path, format="ascii.ecsv")
        conditions = ex.ReferenceConditions.from_meta(table.meta)
        assert conditions.bgsigma_at(conditions.exptime_s) == pytest.approx(
            conditions.bgsigma_adu, rel=1e-9
        )
        direct = 10 ** ex.log_magerror(
            ex.instrumental_magnitude(17.0, conditions.magzero),
            conditions.bgsigma_adu, conditions.fwhm_px, conditions.gain,
        )
        assert ex.predict_magerror(17.0, conditions.exptime_s, conditions) == pytest.approx(
            direct, rel=1e-9
        )


def test_required_exptime_is_consistent_and_monotonic():
    conditions = ex.ReferenceConditions.from_meta(_real_table().meta)
    target_magerror = ex.snr_to_magerror(10.0)

    exptime = ex.required_exptime(18.0, target_magerror, conditions, max_exptime_s=36000.0)
    assert exptime is not None
    assert ex.predict_magerror(18.0, exptime, conditions) == pytest.approx(target_magerror, rel=1e-3)
    fainter = ex.required_exptime(19.0, target_magerror, conditions, max_exptime_s=36000.0)
    brighter = ex.required_exptime(16.0, target_magerror, conditions, max_exptime_s=36000.0)
    assert brighter < exptime < fainter
    # A source a magnitude below the frame's own MAGLIM needs meaningfully
    # more than the frame's own exposure -- the sanity check the fsolve
    # version failed, answering 0.0 seconds.
    assert ex.required_exptime(conditions.maglim + 1.0, target_magerror, conditions,
                               max_exptime_s=36000.0) > conditions.exptime_s
    assert ex.required_exptime(conditions.maglim - 2.0, target_magerror, conditions,
                               max_exptime_s=36000.0) < conditions.exptime_s


def test_unreachable_target_returns_none_not_a_number():
    conditions = ex.ReferenceConditions.from_meta(_real_table().meta)
    assert ex.required_exptime(30.0, ex.snr_to_magerror(10.0), conditions,
                               max_exptime_s=3600.0) is None


def test_coadded_frame_is_rejected_with_a_reason():
    """A stacked/rescaled frame's BGSIGMA sits below the readout floor, so
    the sky term is non-physical -- must raise, not silently produce a
    number. tests/2026kid's frames are real co-adds and do exactly this.
    """
    paths = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "2026kid", "*.ecsv")))
    paths = [p for p in paths if "h.ecsv" not in os.path.basename(p)]
    if not paths:
        pytest.skip("2026kid fixture not available")
    table = Table.read(paths[0], format="ascii.ecsv")
    with pytest.raises(ex.ExposureModelError, match="readout floor"):
        ex.ReferenceConditions.from_meta(table.meta)


def test_missing_header_keyword_is_reported():
    with pytest.raises(ex.ExposureModelError, match="BGSIGMA"):
        ex.ReferenceConditions.from_meta({"EXPTIME": 10.0, "MAGZERO": 25.0, "FWHM": 3.0})


def test_snr_magerror_round_trip():
    assert ex.magerror_to_snr(ex.snr_to_magerror(10.0)) == pytest.approx(10.0)
    assert ex.snr_to_magerror(10.0) == pytest.approx(0.1086, abs=1e-4)


# --- enrichment -------------------------------------------------------------

def _candidates_table():
    table = Table()
    table["transient_id"] = ["transient_1.000_2.000", "transient_3.000_4.000"]
    table["quality_score"] = [42.0, 3.0]
    table["ALPHA_J2000"] = [1.0, 3.0]
    table["DELTA_J2000"] = [2.0, 4.0]
    table["mag_weighted_mean"] = [17.0, 18.0]
    return table


def _lightcurve(mags, errs, times=None, epoch_ids=None):
    lc = Table()
    lc["MAG_CALIB"] = np.array(mags, dtype=float)
    lc["MAGERR_CALIB"] = np.array(errs, dtype=float)
    lc["obs_time"] = np.array(times if times is not None else range(len(mags)), dtype=float)
    lc["epoch_id"] = np.array(epoch_ids if epoch_ids is not None else range(len(mags)))
    return lc


def test_recommend_exposures_annotates_all_rows_and_reports_the_top(tmp_path):
    reference = _real_table()
    candidates = _candidates_table()
    # A well-measured latest point wins over the candidate row's own column
    # -- and over an earlier, brighter point.
    lightcurves = {"transient_1.000_2.000": _lightcurve(
        [16.0, 16.8, 16.4], [0.05, 0.05, 0.05], times=[100.0, 300.0, 200.0], epoch_ids=[0, 0, 0])}

    report = enrichment.recommend_exposures(candidates, lightcurves, [reference], config=PipelineConfig())

    assert report["status"] == "ok"
    assert len(candidates["followup_exptime_s"]) == 2
    top = report["top_candidate"]
    assert top["transient_id"] == "transient_1.000_2.000"
    assert top["magnitude"] == pytest.approx(16.8)
    assert top["magnitude_source"] == "latest lightcurve point"
    assert top["magnitude_filter"] == reference.meta["FILTER"] and not top["filter_mismatch"]
    assert top["predicted_snr"] == pytest.approx(report["target_snr"], rel=1e-2)
    assert candidates["followup_exptime_s"][0] == pytest.approx(top["exptime_s"])
    lo, hi = top["exptime_range_s"]
    assert lo < top["exptime_s"] < hi
    # Second candidate: no lightcurve, falls back to its own column, fainter
    # -> longer.
    assert candidates["followup_mag"][1] == pytest.approx(18.0)
    assert candidates["followup_exptime_s"][1] > candidates["followup_exptime_s"][0]
    # The model was checked against the reference frame's own photometry.
    assert report["model_check"]["status"] == "ok"
    assert report["model_check"]["rms_dex"] < 0.047

    path = enrichment.write_exposure_report(tmp_path, report)
    written = json.loads(path.read_text())
    assert written["top_candidate"]["exptime_s"] == pytest.approx(top["exptime_s"])


def test_noisy_latest_point_is_averaged_not_trusted():
    """A latest point at the admission floor (magerr ~0.7) must not set the
    exposure on its own: 0.7 mag is a x3-4 uncertainty on the time."""
    lc = _lightcurve([17.0, 17.1, 18.5], [0.05, 0.05, 0.7], epoch_ids=[0, 0, 0])
    info = enrichment.planning_magnitude_from_lightcurve(lc, max_planning_magerr=0.2)
    assert info["source"].startswith("inverse-variance mean")
    # Dominated by the two good points, not dragged to 18.5.
    assert 17.0 < info["mag"] < 17.2
    assert info["mag_err"] < 0.05

    good_latest = _lightcurve([17.0, 17.1, 18.5], [0.05, 0.05, 0.1], epoch_ids=[0, 0, 0])
    assert enrichment.planning_magnitude_from_lightcurve(good_latest)["mag"] == pytest.approx(18.5)


def test_planning_magnitude_prefers_the_reference_filter_and_flags_mismatch():
    epoch_filters = {0: "z", 1: "z", 2: "i"}
    lc = _lightcurve([18.0, 18.1, 17.0], [0.05, 0.05, 0.05], epoch_ids=[0, 1, 2])

    # Reference in z: the newer i-band point is skipped for the latest z one.
    info = enrichment.planning_magnitude_from_lightcurve(lc, reference_filter="z", epoch_filters=epoch_filters)
    assert info["mag"] == pytest.approx(18.1) and info["filter"] == "z" and not info["filter_mismatch"]

    # Reference in r: nothing matches, latest point used but flagged.
    info = enrichment.planning_magnitude_from_lightcurve(lc, reference_filter="r", epoch_filters=epoch_filters)
    assert info["mag"] == pytest.approx(17.0) and info["filter"] == "i" and info["filter_mismatch"]


def test_masked_photometry_is_treated_as_missing():
    lc = _lightcurve([17.0, 17.5], [0.05, 0.05], epoch_ids=[0, 0])
    lc["MAG_CALIB"] = np.ma.masked_array([17.0, 17.5], mask=[False, True])
    info = enrichment.planning_magnitude_from_lightcurve(lc)
    assert info["mag"] == pytest.approx(17.0)

    candidates = _candidates_table()
    candidates["mag_weighted_mean"] = np.ma.masked_array([17.0, 18.0], mask=[True, True])
    report = enrichment.recommend_exposures(candidates, {}, [_real_table()], config=PipelineConfig())
    assert report["status"] == "error" and "no usable magnitude" in report["reason"]


def test_recommend_exposures_reports_why_it_could_not_run():
    candidates = _candidates_table()
    empty = enrichment.recommend_exposures(Table(), {}, [_real_table()], config=PipelineConfig())
    assert empty["status"] == "skipped" and "no candidates" in empty["reason"]

    no_reference = enrichment.recommend_exposures(candidates, {}, [], config=PipelineConfig())
    assert no_reference["status"] == "skipped"

    unusable = Table()
    unusable["x"] = [1]
    unusable.meta.update({"EXPTIME": 10.0, "MAGZERO": 25.0, "FWHM": 3.0})
    broken = enrichment.recommend_exposures(candidates, {}, [unusable], config=PipelineConfig())
    assert broken["status"] == "error" and "BGSIGMA" in broken["reason"]
    assert "followup_exptime_s" not in candidates.colnames

    bad_config = PipelineConfig()
    bad_config.followup.target_snr = 0.0
    invalid = enrichment.recommend_exposures(candidates, {}, [_real_table()], config=bad_config)
    assert invalid["status"] == "error" and "target_snr" in invalid["reason"]


def test_model_check_flags_a_frame_the_model_does_not_describe():
    """Scale the measured errors of a real frame by 3x: same header, so the
    conditions load fine, but the model no longer matches the photometry."""
    table = _real_table()
    table["MAGERR_CALIB"] = np.asarray(table["MAGERR_CALIB"], dtype=float) * np.where(
        np.arange(len(table)) % 2 == 0, 3.0, 1.0)
    conditions = ex.ReferenceConditions.from_meta(table.meta)
    check = enrichment.check_model_against_frame(table, conditions)
    assert check["status"] == "poor" and check["rms_dex"] > enrichment.MODEL_CHECK_MAX_RMS_DEX

    bare = Table()
    bare["x"] = [1]
    assert enrichment.check_model_against_frame(bare, conditions)["status"] == "unavailable"


def test_select_reference_table_picks_the_latest_by_time_not_position():
    """load_existing_tables() globs the directory, so list order is
    filesystem order -- the newest epoch is whichever has the latest
    mid-exposure time."""
    older, newer, stack, undated = Table(), Table(), Table(), Table()
    older.meta.update({"filename": "a.fits", "CTIME": 1000.0, "EXPTIME": 10.0})
    newer.meta.update({"filename": "b.fits", "CTIME": 2000.0, "EXPTIME": 10.0})
    stack.meta.update({"filename": "stack.fits", "CTIME": 9999.0, "EXPTIME": 200.0, "IS_STACK": True})
    undated.meta["filename"] = "c.fits"

    assert enrichment.select_reference_table([newer, stack, older]) is newer
    assert enrichment.select_reference_table([newer, older, undated]) is newer
    # Undated tables sort behind every dated one, and among themselves by
    # position -- a fixture of undated tables still yields its last entry.
    assert enrichment.select_reference_table([undated, older]) is older
    first, second = Table(), Table()
    assert enrichment.select_reference_table([first, second]) is second
    assert enrichment.select_reference_table([stack]) is None
    assert enrichment.select_reference_table([]) is None


def test_run_enrichment_never_raises(tmp_path, monkeypatch):
    """Enrichment runs before save_results in pipeline_magic.py; an
    unexpected exception there must not discard the detection results."""
    def explode(*args, **kwargs):
        raise RuntimeError("boom")
    monkeypatch.setattr(enrichment, "recommend_exposures", explode)

    report = enrichment.run_enrichment(_candidates_table(), {}, [_real_table()], tmp_path,
                                       config=PipelineConfig())
    assert report["status"] == "error" and "boom" in report["reason"]
    written = json.loads((tmp_path / enrichment.REPORT_FILENAME).read_text())
    assert written["status"] == "error"


# --- config -----------------------------------------------------------------

def test_followup_config_round_trips_through_yaml_dict():
    config = PipelineConfig.from_dict({"followup": {"target_snr": 20.0, "exposure_enabled": False}})
    assert config.followup.target_snr == 20.0
    assert config.followup.exposure_enabled is False


def test_config_round_trips_through_an_ini_file(tmp_path):
    """to_file() -> from_file() must survive. It used to raise
    InterpolationMissingOptionError: logging.format legitimately contains
    %(asctime)s, which ConfigParser's default interpolation tried to expand
    as a config reference -- so any written config failed to load back.
    """
    config = PipelineConfig()
    config.followup.target_snr = 7.5
    config.followup.max_planning_magerr = 0.3
    config.followup.exposure_enabled = False
    config.detection.min_n_detections = 4

    path = tmp_path / "config.ini"
    config.to_file(str(path))
    loaded = PipelineConfig.from_file(str(path))

    assert loaded.followup.target_snr == 7.5
    assert loaded.followup.max_planning_magerr == 0.3
    assert loaded.followup.exposure_enabled is False
    assert loaded.detection.min_n_detections == 4
    assert loaded.logging.format == PipelineConfig().logging.format
