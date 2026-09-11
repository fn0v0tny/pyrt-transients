"""Regression tests for the whole-project review fixes of 2026-09-03 that
don't belong to an existing test module (see FUTURE_IDEAS.md, "Whole-project
review")."""

import numpy as np
import pytest
from astropy.table import Table

from pyrt_transient.core import color_model
from pyrt_transient import transients
from pyrt_transient.config_trans import PipelineConfig
from pyrt_transient.core.timeutil import unix_to_mjd
from pyrt_transient.detection.blind_multicatalog import clustering, stdpipe_filters


def test_bundled_fotfit_imports_in_package_context():
    """`import fotfit` (bare) only resolved with the package dir as cwd, so
    every installed run had fotfit=None and colour terms silently off."""
    assert color_model.fotfit is not None
    assert transients.fotfit is color_model.fotfit  # shim re-export


def test_colour_terms_in_response_change_the_catalogue_magnitude():
    # PC is fotfit's linear g-r colour term: a star with g-r=1.0 and
    # PC=0.5 is predicted 0.5 mag fainter in this system than its r mag.
    assert transients.simple_color_model("Z=25.0,PC=0.5", (15.0, 1.0, 0.0, 0.0, 0.0)) == pytest.approx(15.5)
    # Only Z: nothing differential to apply.
    assert transients.simple_color_model("Z=25.0", (15.0, 1.0, 0.0, 0.0, 0.0)) == pytest.approx(15.0)
    # Zero colour: the term has nothing to act on.
    assert transients.simple_color_model("Z=25.0,PC=0.5", (15.0, 0.0, 0.0, 0.0, 0.0)) == pytest.approx(15.0)


def test_cached_fotfit_is_per_response_string():
    """The fotfit is memoised on the RESPONSE string (it was rebuilt for
    every detection of every epoch). Interleaving epochs with different
    responses must not leak one frame's colour terms into another's."""
    a, b = "Z=25.0,PC=0.5", "Z=25.0,PC=-0.25"
    star = (15.0, 1.0, 0.0, 0.0, 0.0)
    for _ in range(3):
        assert transients.simple_color_model(a, star) == pytest.approx(15.5)
        assert transients.simple_color_model(b, star) == pytest.approx(14.75)
        assert transients.simple_color_model("Z=25.0", star) == pytest.approx(15.0)


def test_unix_to_mjd_is_nan_not_the_input_on_failure():
    assert unix_to_mjd(1624147193.0) == pytest.approx(59384.99992, abs=1e-4)
    assert np.isnan(unix_to_mjd("not a time"))


def test_vsx_outage_does_not_discard_the_run(monkeypatch, tmp_path):
    """A VizieR failure at the last step used to propagate out of
    combine_with_lightcurves, exiting the pipeline before save_results."""
    def boom(*args, **kwargs):
        raise ConnectionError("VizieR unreachable")
    monkeypatch.setattr(clustering, "apply_vsx_filter", boom)

    # One epoch with three candidates at the same position, written the way
    # combine_with_lightcurves reads them back, so a final table exists for
    # the VSX step to run on.
    config = PipelineConfig()
    config.detection.min_n_detections = 1
    # A one-point lightcurve has mag_range=0, so the final quality gate
    # would drop it regardless of VSX -- not what this test is about.
    config.detection.min_quality = 0.0
    det = Table({
        "ALPHA_J2000": [10.0], "DELTA_J2000": [20.0], "X_IMAGE": [100.0], "Y_IMAGE": [100.0],
        "MAG_CALIB": [17.0], "MAGERR_CALIB": [0.05], "FWHM_IMAGE": [3.0],
        "quality_score": [1.0], "candidate_type": ["new"], "reference_catalog": ["gaia"],
    })
    det.meta.update({"filename": str(tmp_path / "e1.ecsv"), "CTIME": 1000.0, "EXPTIME": 10.0})
    det.write(tmp_path / "e1_transients.ecsv", format="ascii.ecsv")
    epochs = [det]
    from pyrt_transient.core.epochs import prepare_epoch_detections
    result, lightcurves = clustering.combine_with_lightcurves(
        tmp_path, epochs, prepare_epoch_detections(epochs),
        position_match_radius=2.0, min_n_detections=1, config=config,
    )
    assert len(result) == 1
    assert len(lightcurves) == 1


def test_core_pipeline_imports_without_matplotlib(monkeypatch):
    """matplotlib is the optional [frontend] extra; importing it at module
    level in plotting.py/extraction_manager.py made the core strategy
    unimportable without it."""
    import importlib, sys
    saved = {k: v for k, v in sys.modules.items() if k == "matplotlib" or k.startswith("matplotlib.")}
    for k in saved:
        monkeypatch.delitem(sys.modules, k)
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    for mod in ("pyrt_transient.detection.blind_multicatalog.plotting", "pyrt_transient.extraction_manager"):
        monkeypatch.delitem(sys.modules, mod, raising=False)
        importlib.import_module(mod)
    from pyrt_transient.detection.blind_multicatalog import plotting
    assert plotting.plt is None


def test_band_warning_only_when_non_r_band_lacks_colour_terms(caplog):
    import logging as _logging
    from astropy.table import Table as _Table
    from pyrt_transient.detection.blind_multicatalog import catalog_match

    log = _logging.getLogger("test.band")
    def run(meta):
        caplog.clear()
        t = _Table({"X_IMAGE": [1.0]}); t.meta.update(meta)
        with caplog.at_level(_logging.WARNING, logger="test.band"):
            catalog_match._warn_if_band_uncorrected(t, log)
        return any("compared against catalogue Sloan r" in r.message for r in caplog.records)

    assert not run({"PHFILTER": "Sloan_r", "RESPONSE": "Z=25.0"})
    assert run({"PHFILTER": "Sloan_i", "RESPONSE": "Z=25.0,PX=0.01"})
    assert not run({"PHFILTER": "Sloan_i", "RESPONSE": "Z=25.0,PC=0.3"})
    assert not run({"RESPONSE": "Z=25.0"})  # unknown band: nothing to say


def test_transients_shim_reexports_and_legacy_cli_is_gone():
    from pyrt_transient import transients as shim
    from pyrt_transient.io.ecsv import open_ecsv_file
    from pyrt_transient.core.color_model import simple_color_model
    assert shim.open_ecsv_file is open_ecsv_file
    assert shim.simple_color_model is simple_color_model
    assert not hasattr(shim, "process_single_image")


def test_open_ecsv_file_returns_none_and_sets_local_filename(tmp_path):
    from astropy.table import Table as _Table
    from pyrt_transient.io.ecsv import open_ecsv_file
    assert open_ecsv_file(tmp_path / "missing.ecsv") is None
    t = _Table({"X_IMAGE": [1.0]}); t.meta["filename"] = "/remote/host/path.fits"
    t.write(tmp_path / "e.ecsv", format="ascii.ecsv")
    loaded = open_ecsv_file(tmp_path / "e.fits")  # fits path -> its ecsv sibling
    assert loaded is not None and loaded.meta["filename"] == str(tmp_path / "e.ecsv")


def test_extraction_manager_raises_without_a_field_centre():
    from astropy.table import Table as _Table
    from pyrt_transient.extraction_manager import ImageExtractionManager
    t = _Table({"X_IMAGE": [1.0]}); t.meta.update({"CTRRA": 10.0, "CTRDEC": 20.0})
    assert ImageExtractionManager([t]).field_center == (10.0, 20.0)
    with pytest.raises(ValueError):
        ImageExtractionManager([_Table({"X_IMAGE": [1.0]})])
