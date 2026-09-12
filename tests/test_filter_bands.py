"""A lightcurve must keep the photometric band of every point.

D50 cycles g/r/i/z inside a single observation (obs_104223 on 2026-09-11:
17 i, 12 r, 10 g and 7 z frames), and the band lives in each epoch's meta.
Without it the epochs merge into one series and the page draws a lightcurve
that mixes filters."""
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pyrt_transient.core.epochs import prepare_epoch_detections  # noqa: E402
from pyrt_transient.detection.blind_multicatalog import plotting  # noqa: E402


def _epoch(filter_meta, n=2):
    t = Table({"ALPHA_J2000": np.linspace(10.0, 10.1, n), "DELTA_J2000": np.linspace(20.0, 20.1, n)})
    t.meta = {"CTIME": 1789164031, "EXPTIME": 5.0, "filename": "frame-df.ecsv", **filter_meta}
    return t


def test_the_band_of_each_epoch_reaches_its_detections():
    epochs = prepare_epoch_detections([
        _epoch({"PHFILTER": "Sloan_i", "FILTER": "Sloan_i"}),
        _epoch({"FILTER": "Sloan_g"}),                 # only FILTER
        _epoch({"PHFILTER": "Sloan_r", "FILTER": "R"}),  # the photometric band wins
        _epoch({}),                                     # unfiltered
    ])

    assert [str(e["filter"][0]) for e in epochs] == ["Sloan_i", "Sloan_g", "Sloan_r", ""]
    assert all(len(e["filter"]) == len(e) for e in epochs)


class _Ax:
    def __init__(self):
        self.series, self.legends = [], []
        self.transAxes = None

    def errorbar(self, x, y, yerr=None, label=None, **kw):
        self.series.append((label, [float(v) for v in x]))

    def legend(self, **kw):
        self.legends.append(kw)

    def __getattr__(self, _name):      # invert_yaxis, set_xlabel, grid, text, ...
        return lambda *a, **k: None


class _Plt:
    def __init__(self, ax):
        self.ax, self.saved = ax, None

    def subplots(self, *a, **k):
        return object(), self.ax

    def savefig(self, path, **kw):
        self.saved = path

    def __getattr__(self, _name):      # tight_layout, close, ...
        return lambda *a, **k: None


def _lightcurve(bands):
    n = len(bands)
    return Table({"obs_time": np.arange(n, dtype=float) * 3600.0,
                  "MAG_CALIB": np.linspace(15.0, 15.4, n),
                  "MAGERR_CALIB": np.full(n, 0.02),
                  "epoch_id": np.arange(n),
                  "ALPHA_J2000": np.full(n, 10.0), "DELTA_J2000": np.full(n, 20.0),
                  "filter": bands})


def test_the_plot_draws_one_series_per_band(tmp_path, monkeypatch):
    ax = _Ax()
    monkeypatch.setattr(plotting, "plt", _Plt(ax))

    plotting.plot_individual_lightcurve(
        "transient_1", _lightcurve(["Sloan_i", "Sloan_g", "Sloan_i", "Sloan_z"]), tmp_path)

    assert sorted(label for label, _ in ax.series) == ["Sloan_g", "Sloan_i", "Sloan_z"]
    by_band = dict(ax.series)
    assert by_band["Sloan_i"] == [0.0, 2.0]      # hours since the first point
    assert by_band["Sloan_g"] == [1.0]
    assert ax.legends, "a mixed-band lightcurve needs a legend"


def test_a_lightcurve_without_the_column_still_plots(tmp_path, monkeypatch):
    ax = _Ax()
    monkeypatch.setattr(plotting, "plt", _Plt(ax))
    lc = _lightcurve(["Sloan_r", "Sloan_r"])
    lc.remove_column("filter")                    # lightcurves from before this change

    plotting.plot_individual_lightcurve("transient_1", lc, tmp_path)

    assert [label for label, _ in ax.series] == ["unfiltered"]
    assert not ax.legends


def test_the_page_gets_the_band_with_every_point(tmp_path):
    pytest.importorskip("matplotlib")
    from pyrt_transient.config_trans import FrontendConfig
    from pyrt_transient.frontend_generator import FrontendGenerator

    data_dir = tmp_path / "obs"
    data_dir.mkdir()
    lc = _lightcurve(["Sloan_i", "Sloan_g", "Sloan_i"])
    lc.write(data_dir / "transient_1_lightcurve.ecsv", format="ascii.ecsv")
    gen = FrontendGenerator("1", data_dir, tmp_path / "public", FrontendConfig())
    candidate = Table({"transient_id": ["transient_1"]})[0]

    info = gen.process_lightcurve_data(candidate, "cand_1")

    assert [p.get("filter") for p in info["points"]] == ["Sloan_i", "Sloan_g", "Sloan_i"]
