"""Unit tests for detection/stacking.py -- image stacking/co-addition for
the GRB (blind_multicatalog) pipeline (see FUTURE_IDEAS.md's "Image
stacking/co-addition").

Real `pyrt-combine`/SEP/network-catalog calibration is not re-tested here
(same boundary test_detection_subtraction.py already draws for
extraction.py: that layer is validated only against real data, since it
needs live network catalog access). What's tested here is everything this
module adds on top: the subprocess wrapper's success/failure/missing-binary
paths (via a fake stub binary, not the real pyrt-combine), the pure
trigger/selection logic, delegation to build_diff_ecsv, and the
maybe_build_stack_table orchestration (with combine_epochs/build_stack_ecsv
monkeypatched so no real FITS/network access is needed).

No pytest dependency -- run directly with
`python3 tests/test_stacking.py`.
"""
import logging
import os
import stat
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from astropy.io import fits
from astropy.table import Table

from pyrt_transient.detection import stacking
from pyrt_transient.config_trans import PipelineConfig

_LOG = logging.getLogger("test_stacking")

_FAKE_COMBINE_SCRIPT = """#!/bin/sh
out=""
prev=""
for arg in "$@"; do
    if [ "$prev" = "-o" ]; then
        out="$arg"
    fi
    prev="$arg"
done
if [ -n "$out" ]; then
    printf 'FAKEFITS' > "$out"
fi
exit 0
"""

_FAKE_COMBINE_FAILING_SCRIPT = """#!/bin/sh
echo "simulated pyrt-combine failure" >&2
exit 3
"""


def _write_stub_binary(dir_path: Path, script: str) -> Path:
    path = dir_path / "pyrt-combine"
    path.write_text(script)
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _set_path(dirs):
    old_path = os.environ.get("PATH", "")
    os.environ["PATH"] = os.pathsep.join(str(d) for d in dirs)
    return lambda: os.environ.__setitem__("PATH", old_path)


def _make_config(**overrides):
    cfg = PipelineConfig()
    for k, v in overrides.items():
        setattr(cfg.detection, k, v)
    return cfg


def _make_detection_table(index: int, obs_dir: Path, phfilter: str = "Sloan_r", exptime: float = 10.0) -> Table:
    t = Table({"X_IMAGE": [1.0], "MAG_CALIB": [18.0]})
    ecsv_path = obs_dir / f"epoch{index}.ecsv"
    t.meta["filename"] = str(ecsv_path)
    t.meta["CTRRA"] = 180.0
    t.meta["CTRDEC"] = 30.0
    t.meta["PHFILTER"] = phfilter
    t.meta["EXPTIME"] = exptime
    return t


def _write_candidates_tbl(obs_dir: Path, quality_score) -> None:
    t = Table({"quality_score": [float(quality_score)]})
    t.write(str(obs_dir / "candidates.tbl"), format="ascii.ipac", overwrite=True)


def _assert_close(actual, expected, label, tol=1e-6):
    assert abs(actual - expected) < tol, f"{label}: expected {expected}, got {actual}"


# ---------------------------------------------------------------------------
# select_stack_inputs
# ---------------------------------------------------------------------------

def test_select_stack_inputs_returns_all_when_fewer_than_max():
    paths = [Path(f"e{i}.fits") for i in range(5)]
    result = stacking.select_stack_inputs(paths, max_epochs=20)
    assert result == paths
    print("test_select_stack_inputs_returns_all_when_fewer_than_max: PASS")


def test_select_stack_inputs_keeps_most_recent():
    paths = [Path(f"e{i}.fits") for i in range(30)]
    result = stacking.select_stack_inputs(paths, max_epochs=20)
    assert result == paths[-20:]
    assert len(result) == 20
    print("test_select_stack_inputs_keeps_most_recent: PASS")


# ---------------------------------------------------------------------------
# _band_exptime_key / _select_consistent_group
# ---------------------------------------------------------------------------

def test_band_exptime_key_rounds_exptime_and_stringifies_filter():
    t = Table()
    t.meta["PHFILTER"] = "Sloan_r"
    t.meta["EXPTIME"] = 10.04
    assert stacking._band_exptime_key(t) == ("Sloan_r", 10.0)
    print("test_band_exptime_key_rounds_exptime_and_stringifies_filter: PASS")


def test_band_exptime_key_none_when_metadata_missing():
    assert stacking._band_exptime_key(Table()) is None
    t = Table()
    t.meta["PHFILTER"] = "Sloan_r"  # EXPTIME missing
    assert stacking._band_exptime_key(t) is None
    print("test_band_exptime_key_none_when_metadata_missing: PASS")


def test_select_consistent_group_picks_the_largest_filter_exptime_group():
    # pyrt-combine does not normalize for exposure-time or filter
    # differences (verified from its own source) -- mixing them would
    # silently combine physically incompatible pixel values. Majority
    # group here: ("Sloan_r", 10.0) with 3 members.
    pairs = []
    for i, (phfilter, exptime) in enumerate([
        ("Sloan_r", 10.0), ("Sloan_r", 10.0), ("Sloan_r", 10.0),
        ("Sloan_i", 60.0), ("Sloan_r", 30.0),
    ]):
        t = Table()
        t.meta["PHFILTER"] = phfilter
        t.meta["EXPTIME"] = exptime
        pairs.append((t, Path(f"e{i}.fits")))

    group, key = stacking._select_consistent_group(pairs)
    assert key == ("Sloan_r", 10.0)
    assert len(group) == 3
    print("test_select_consistent_group_picks_the_largest_filter_exptime_group: PASS")


def test_select_consistent_group_excludes_epochs_missing_metadata():
    good = Table()
    good.meta["PHFILTER"] = "Sloan_r"
    good.meta["EXPTIME"] = 10.0
    missing_exptime = Table()
    missing_exptime.meta["PHFILTER"] = "Sloan_r"
    pairs = [(good, Path("a.fits")), (missing_exptime, Path("b.fits"))]

    group, key = stacking._select_consistent_group(pairs)
    assert key == ("Sloan_r", 10.0)
    assert len(group) == 1
    print("test_select_consistent_group_excludes_epochs_missing_metadata: PASS")


def test_select_consistent_group_empty_when_no_metadata_anywhere():
    pairs = [(Table(), Path("a.fits")), (Table(), Path("b.fits"))]
    group, key = stacking._select_consistent_group(pairs)
    assert group == []
    assert key is None
    print("test_select_consistent_group_empty_when_no_metadata_anywhere: PASS")


# ---------------------------------------------------------------------------
# should_rebuild_stack
# ---------------------------------------------------------------------------

def test_should_rebuild_stack_false_below_min_epochs():
    assert stacking.should_rebuild_stack(5, 0, min_epochs=10, rebuild_interval=5) is False
    print("test_should_rebuild_stack_false_below_min_epochs: PASS")


def test_should_rebuild_stack_true_on_first_crossing():
    assert stacking.should_rebuild_stack(10, 0, min_epochs=10, rebuild_interval=5) is True
    print("test_should_rebuild_stack_true_on_first_crossing: PASS")


def test_should_rebuild_stack_false_before_interval_elapsed():
    assert stacking.should_rebuild_stack(12, 10, min_epochs=10, rebuild_interval=5) is False
    print("test_should_rebuild_stack_false_before_interval_elapsed: PASS")


def test_should_rebuild_stack_true_after_interval_elapsed():
    assert stacking.should_rebuild_stack(15, 10, min_epochs=10, rebuild_interval=5) is True
    print("test_should_rebuild_stack_true_after_interval_elapsed: PASS")


# ---------------------------------------------------------------------------
# combine_epochs
# ---------------------------------------------------------------------------

def test_combine_epochs_returns_none_when_binary_missing():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        restore = _set_path([d])  # empty dir -- guaranteed no pyrt-combine
        try:
            result = stacking.combine_epochs([Path("a.fits"), Path("b.fits")], d / "out.fits")
        finally:
            restore()
        assert result is None
        print("test_combine_epochs_returns_none_when_binary_missing: PASS")


def test_combine_epochs_success_with_stub_binary():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        _write_stub_binary(d, _FAKE_COMBINE_SCRIPT)
        restore = _set_path([d])
        try:
            out_path = d / "stack.fits"
            result = stacking.combine_epochs([Path("a.fits"), Path("b.fits")], out_path)
        finally:
            restore()
        assert result == out_path
        assert out_path.exists()
        print("test_combine_epochs_success_with_stub_binary: PASS")


def test_combine_epochs_returns_none_on_nonzero_exit():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        _write_stub_binary(d, _FAKE_COMBINE_FAILING_SCRIPT)
        restore = _set_path([d])
        try:
            out_path = d / "stack.fits"
            result = stacking.combine_epochs([Path("a.fits"), Path("b.fits")], out_path)
        finally:
            restore()
        assert result is None
        assert not out_path.exists()
        print("test_combine_epochs_returns_none_on_nonzero_exit: PASS")


def test_combine_epochs_requires_at_least_two_inputs():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        _write_stub_binary(d, _FAKE_COMBINE_SCRIPT)
        restore = _set_path([d])
        try:
            result = stacking.combine_epochs([Path("a.fits")], d / "out.fits")
        finally:
            restore()
        assert result is None
        print("test_combine_epochs_requires_at_least_two_inputs: PASS")


# ---------------------------------------------------------------------------
# build_stack_ecsv
# ---------------------------------------------------------------------------

def test_build_stack_ecsv_delegates_with_stack_as_both_diff_and_science():
    calls = []

    def fake_build_diff_ecsv(diff_fits_path, science_fits_path, output_path=None,
                              photometric_catalog="ps1", detect_thresh=4.0, aper_px=None):
        calls.append((Path(diff_fits_path), Path(science_fits_path), photometric_catalog, detect_thresh))
        t = Table({"MAG_CALIB": [18.5]})
        t.meta["MAGZERO"] = 25.0
        t.write(str(output_path), format="ascii.ecsv", overwrite=True)
        return Path(output_path)

    original = stacking.sub_extraction.build_diff_ecsv
    stacking.sub_extraction.build_diff_ecsv = fake_build_diff_ecsv
    try:
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            stack_fits = d / "stack.fits"
            out_ecsv = d / "stack.ecsv"
            result = stacking.build_stack_ecsv(
                stack_fits, output_path=out_ecsv,
                photometric_catalog="ps1", detect_thresh=5.0, n_combined=12,
            )
            assert result == out_ecsv
            assert len(calls) == 1
            diff_p, sci_p, cat, thresh = calls[0]
            assert diff_p == stack_fits, "build_stack_ecsv must pass the stack as the diff image"
            assert sci_p == stack_fits, "build_stack_ecsv must pass the stack as the science image too"
            assert cat == "ps1"
            assert thresh == 5.0

            written = Table.read(str(out_ecsv), format="ascii.ecsv")
            assert written.meta["IS_STACK"] is True
            assert written.meta["NCOMBINE"] == 12
    finally:
        stacking.sub_extraction.build_diff_ecsv = original
    print("test_build_stack_ecsv_delegates_with_stack_as_both_diff_and_science: PASS")


def test_build_stack_ecsv_returns_none_when_calibration_fails():
    original = stacking.sub_extraction.build_diff_ecsv
    stacking.sub_extraction.build_diff_ecsv = lambda **kwargs: None
    try:
        result = stacking.build_stack_ecsv(Path("/nonexistent/stack.fits"))
        assert result is None
    finally:
        stacking.sub_extraction.build_diff_ecsv = original
    print("test_build_stack_ecsv_returns_none_when_calibration_fails: PASS")


def _write_synthetic_image(path, shape=(256, 256), noise_sigma=100.0, background=10000.0, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.normal(loc=background, scale=noise_sigma, size=shape).astype("float32")
    header = fits.Header({"NAXIS1": shape[1], "NAXIS2": shape[0], "FWHM": 3.0})
    fits.writeto(str(path), data, header, overwrite=True)


def test_empirical_maglim_keeps_only_sources_above_sigma_cut():
    try:
        import stdpipe.photometry  # noqa: F401
    except ImportError:
        print("test_empirical_maglim_keeps_only_sources_above_sigma_cut: SKIP (stdpipe unavailable)")
        return

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        stack_path = d / "stack.fits"
        _write_synthetic_image(stack_path, noise_sigma=100.0, seed=1)

        # aperture_noise for a 256x256, noise_sigma=100 image with FWHM=3
        # aperture: measured_backrms(~100) * sqrt(pi*(3**2)) ~= 100*5.32 ~= 532
        table = Table({
            "MAG_CALIB": [15.0, 18.0, 20.0],
            "FLUX": [50000.0, 3000.0, 100.0],  # SNR roughly ~94, ~5.6, ~0.19 at noise~532
        })
        maglim = stacking._empirical_maglim(table, stack_path, sigma_cut=5.0)
        assert maglim is not None
        # Only the first two clear SNR>=5 -- the faint 100-flux row must not
        # pull the percentile fainter than the second-brightest real source.
        assert maglim <= 18.0 + 1e-6, f"expected the empirical MAGLIM to reflect only real >=5-sigma sources, got {maglim}"
    print("test_empirical_maglim_keeps_only_sources_above_sigma_cut: PASS")


def test_empirical_maglim_returns_none_when_nothing_clears_sigma_cut():
    try:
        import stdpipe.photometry  # noqa: F401
    except ImportError:
        print("test_empirical_maglim_returns_none_when_nothing_clears_sigma_cut: SKIP (stdpipe unavailable)")
        return

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        stack_path = d / "stack.fits"
        _write_synthetic_image(stack_path, noise_sigma=100.0, seed=2)

        table = Table({"MAG_CALIB": [22.0], "FLUX": [10.0]})  # far below any reasonable sigma cut
        maglim = stacking._empirical_maglim(table, stack_path, sigma_cut=5.0)
        assert maglim is None
    print("test_empirical_maglim_returns_none_when_nothing_clears_sigma_cut: PASS")


def test_empirical_maglim_returns_none_without_required_columns():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        table = Table({"MAG_CALIB": [18.0]})  # no FLUX column
        assert stacking._empirical_maglim(table, d / "stack.fits") is None
        assert stacking._empirical_maglim(Table(), d / "stack.fits") is None
    print("test_empirical_maglim_returns_none_without_required_columns: PASS")


def test_build_stack_ecsv_uses_empirical_maglim_not_build_diff_ecsvs_fallback():
    # Regression: build_diff_ecsv's own MAGLIM fallback
    # (`zp - 2.5*log10(5.0)`) ignores the stack's actual noise level
    # entirely -- verified on a real 20-epoch GRB210410A stack it gave
    # MAGLIM=28.09 against a real per-epoch MAGLIM of ~17.5. build_stack_ecsv
    # must overwrite it with _empirical_maglim's result whenever that
    # succeeds.
    def fake_build_diff_ecsv(diff_fits_path, science_fits_path, output_path=None,
                              photometric_catalog="ps1", detect_thresh=4.0, aper_px=None):
        t = Table({"MAG_CALIB": [18.5], "FLUX": [1000.0]})
        t.meta["MAGZERO"] = 25.0
        t.meta["MAGLIM"] = 999.0  # deliberately implausible -- must not survive
        t.write(str(output_path), format="ascii.ecsv", overwrite=True)
        return Path(output_path)

    def fake_empirical_maglim(table, stack_fits_path, **kwargs):
        return 19.3  # arbitrary known value

    original_bde = stacking.sub_extraction.build_diff_ecsv
    original_emp = stacking._empirical_maglim
    stacking.sub_extraction.build_diff_ecsv = fake_build_diff_ecsv
    stacking._empirical_maglim = fake_empirical_maglim
    try:
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            out_ecsv = d / "stack.ecsv"
            stacking.build_stack_ecsv(d / "stack.fits", output_path=out_ecsv)
            written = Table.read(str(out_ecsv), format="ascii.ecsv")
            _assert_close(written.meta["MAGLIM"], 19.3, "MAGLIM must come from _empirical_maglim")
    finally:
        stacking.sub_extraction.build_diff_ecsv = original_bde
        stacking._empirical_maglim = original_emp
    print("test_build_stack_ecsv_uses_empirical_maglim_not_build_diff_ecsvs_fallback: PASS")


# ---------------------------------------------------------------------------
# maybe_build_stack_table
# ---------------------------------------------------------------------------

def _fake_combine(fits_paths, output_path, uniform=True):
    Path(output_path).write_bytes(b"FAKEFITS")
    return Path(output_path)


def _fake_build_stack_ecsv(stack_fits_path, output_path=None, photometric_catalog="ps1",
                            detect_thresh=5.0, n_combined=None):
    t = Table({"MAG_CALIB": [17.0]})
    t.meta["IS_STACK"] = True
    t.meta["NCOMBINE"] = n_combined
    t.write(str(output_path), format="ascii.ecsv", overwrite=True)
    return Path(output_path)


def _patch_build_fns(fake_combine=None, fake_build_ecsv=None):
    orig_combine = stacking.combine_epochs
    orig_build = stacking.build_stack_ecsv
    stacking.combine_epochs = fake_combine or orig_combine
    stacking.build_stack_ecsv = fake_build_ecsv or orig_build

    def restore():
        stacking.combine_epochs = orig_combine
        stacking.build_stack_ecsv = orig_build

    return restore


def test_maybe_build_stack_table_below_min_epochs_returns_none():
    calls = {"n": 0}

    def counting_combine(*a, **k):
        calls["n"] += 1
        return _fake_combine(*a, **k)

    restore = _patch_build_fns(fake_combine=counting_combine, fake_build_ecsv=_fake_build_stack_ecsv)
    try:
        with tempfile.TemporaryDirectory() as d:
            obs_dir = Path(d)
            cfg = _make_config(stacking_enabled=True, stacking_min_epochs=10)
            tables = [_make_detection_table(i, obs_dir) for i in range(5)]
            result = stacking.maybe_build_stack_table(obs_dir, tables, cfg, _LOG)
            assert result is None
            assert calls["n"] == 0
    finally:
        restore()
    print("test_maybe_build_stack_table_below_min_epochs_returns_none: PASS")


def test_maybe_build_stack_table_disabled_returns_none():
    with tempfile.TemporaryDirectory() as d:
        obs_dir = Path(d)
        cfg = _make_config(stacking_enabled=False, stacking_min_epochs=1)
        tables = [_make_detection_table(i, obs_dir) for i in range(20)]
        result = stacking.maybe_build_stack_table(obs_dir, tables, cfg, _LOG)
        assert result is None
    print("test_maybe_build_stack_table_disabled_returns_none: PASS")


def test_maybe_build_stack_table_builds_when_triggered():
    restore = _patch_build_fns(fake_combine=_fake_combine, fake_build_ecsv=_fake_build_stack_ecsv)
    try:
        with tempfile.TemporaryDirectory() as d:
            obs_dir = Path(d)
            cfg = _make_config(stacking_enabled=True, stacking_min_epochs=10,
                                stacking_rebuild_interval=5, stacking_score_threshold=1.0)
            tables = [_make_detection_table(i, obs_dir) for i in range(10)]
            result = stacking.maybe_build_stack_table(obs_dir, tables, cfg, _LOG)
            assert result is not None
            assert bool(result.meta.get("IS_STACK")) is True
            assert result.meta.get("NCOMBINE") == 10

            import json
            state = json.loads((obs_dir / "stack_state.json").read_text())
            assert state["last_build_n_epochs"] == 10
    finally:
        restore()
    print("test_maybe_build_stack_table_builds_when_triggered: PASS")


def test_maybe_build_stack_table_excludes_mismatched_filter_and_exptime():
    # End-to-end: a minority of epochs with a different filter/exposure
    # time than the majority must not reach combine_epochs at all.
    calls = []

    def capturing_combine(fits_paths, output_path, uniform=True):
        calls.append(list(fits_paths))
        return _fake_combine(fits_paths, output_path, uniform=uniform)

    restore = _patch_build_fns(fake_combine=capturing_combine, fake_build_ecsv=_fake_build_stack_ecsv)
    try:
        with tempfile.TemporaryDirectory() as d:
            obs_dir = Path(d)
            cfg = _make_config(stacking_enabled=True, stacking_min_epochs=8,
                                stacking_rebuild_interval=5, stacking_score_threshold=1.0)
            majority = [_make_detection_table(i, obs_dir, phfilter="Sloan_r", exptime=10.0) for i in range(8)]
            minority = [_make_detection_table(i, obs_dir, phfilter="Sloan_i", exptime=60.0) for i in range(8, 10)]
            tables = majority + minority

            result = stacking.maybe_build_stack_table(obs_dir, tables, cfg, _LOG)
            assert result is not None
            assert result.meta.get("NCOMBINE") == 8, "only the 8 same-filter/same-exptime epochs must be combined"
            assert len(calls) == 1
            assert len(calls[0]) == 8, "combine_epochs must only receive the majority-group FITS paths"
    finally:
        restore()
    print("test_maybe_build_stack_table_excludes_mismatched_filter_and_exptime: PASS")


def test_maybe_build_stack_table_skips_when_existing_candidate_scores_well():
    calls = {"n": 0}

    def counting_combine(*a, **k):
        calls["n"] += 1
        return _fake_combine(*a, **k)

    restore = _patch_build_fns(fake_combine=counting_combine, fake_build_ecsv=_fake_build_stack_ecsv)
    try:
        with tempfile.TemporaryDirectory() as d:
            obs_dir = Path(d)
            cfg = _make_config(stacking_enabled=True, stacking_min_epochs=10,
                                stacking_score_threshold=1.0)
            tables = [_make_detection_table(i, obs_dir) for i in range(10)]
            _write_candidates_tbl(obs_dir, quality_score=5.0)  # already convincing

            result = stacking.maybe_build_stack_table(obs_dir, tables, cfg, _LOG)
            assert result is None, "no stack on disk yet, and score already good -- nothing to build or return"
            assert calls["n"] == 0, "pyrt-combine should not have been invoked"
    finally:
        restore()
    print("test_maybe_build_stack_table_skips_when_existing_candidate_scores_well: PASS")


def test_maybe_build_stack_table_keeps_existing_stack_even_once_score_is_good():
    restore = _patch_build_fns(fake_combine=_fake_combine, fake_build_ecsv=_fake_build_stack_ecsv)
    try:
        with tempfile.TemporaryDirectory() as d:
            obs_dir = Path(d)
            cfg = _make_config(stacking_enabled=True, stacking_min_epochs=10,
                                stacking_score_threshold=1.0)
            tables = [_make_detection_table(i, obs_dir) for i in range(10)]

            # First call: no candidates yet -- triggers a real build.
            first = stacking.maybe_build_stack_table(obs_dir, tables, cfg, _LOG)
            assert first is not None

            # Now a good candidate exists (e.g. the stack itself helped find
            # it). A stack-anchored candidate must not vanish from future
            # runs just because scoring improved.
            _write_candidates_tbl(obs_dir, quality_score=5.0)
            second = stacking.maybe_build_stack_table(obs_dir, tables, cfg, _LOG)
            assert second is not None
            assert bool(second.meta.get("IS_STACK")) is True
    finally:
        restore()
    print("test_maybe_build_stack_table_keeps_existing_stack_even_once_score_is_good: PASS")


def test_maybe_build_stack_table_respects_rebuild_interval():
    calls = {"n": 0}

    def counting_combine(*a, **k):
        calls["n"] += 1
        return _fake_combine(*a, **k)

    restore = _patch_build_fns(fake_combine=counting_combine, fake_build_ecsv=_fake_build_stack_ecsv)
    try:
        with tempfile.TemporaryDirectory() as d:
            obs_dir = Path(d)
            cfg = _make_config(stacking_enabled=True, stacking_min_epochs=10,
                                stacking_rebuild_interval=5, stacking_score_threshold=1.0)
            tables = [_make_detection_table(i, obs_dir) for i in range(10)]

            first = stacking.maybe_build_stack_table(obs_dir, tables, cfg, _LOG)
            assert first is not None
            assert calls["n"] == 1

            # Same epoch count again -- not due for a rebuild yet, but the
            # already-built stack must still be returned.
            second = stacking.maybe_build_stack_table(obs_dir, tables, cfg, _LOG)
            assert second is not None
            assert calls["n"] == 1, "should not have re-run pyrt-combine"

            # Enough new epochs accumulated -- due again.
            more_tables = tables + [_make_detection_table(i, obs_dir) for i in range(10, 15)]
            third = stacking.maybe_build_stack_table(obs_dir, more_tables, cfg, _LOG)
            assert third is not None
            assert calls["n"] == 2
    finally:
        restore()
    print("test_maybe_build_stack_table_respects_rebuild_interval: PASS")


if __name__ == "__main__":
    test_select_stack_inputs_returns_all_when_fewer_than_max()
    test_select_stack_inputs_keeps_most_recent()
    test_band_exptime_key_rounds_exptime_and_stringifies_filter()
    test_band_exptime_key_none_when_metadata_missing()
    test_select_consistent_group_picks_the_largest_filter_exptime_group()
    test_select_consistent_group_excludes_epochs_missing_metadata()
    test_select_consistent_group_empty_when_no_metadata_anywhere()
    test_should_rebuild_stack_false_below_min_epochs()
    test_should_rebuild_stack_true_on_first_crossing()
    test_should_rebuild_stack_false_before_interval_elapsed()
    test_should_rebuild_stack_true_after_interval_elapsed()
    test_combine_epochs_returns_none_when_binary_missing()
    test_combine_epochs_success_with_stub_binary()
    test_combine_epochs_returns_none_on_nonzero_exit()
    test_combine_epochs_requires_at_least_two_inputs()
    test_build_stack_ecsv_delegates_with_stack_as_both_diff_and_science()
    test_build_stack_ecsv_returns_none_when_calibration_fails()
    test_empirical_maglim_keeps_only_sources_above_sigma_cut()
    test_empirical_maglim_returns_none_when_nothing_clears_sigma_cut()
    test_empirical_maglim_returns_none_without_required_columns()
    test_build_stack_ecsv_uses_empirical_maglim_not_build_diff_ecsvs_fallback()
    test_maybe_build_stack_table_below_min_epochs_returns_none()
    test_maybe_build_stack_table_disabled_returns_none()
    test_maybe_build_stack_table_builds_when_triggered()
    test_maybe_build_stack_table_excludes_mismatched_filter_and_exptime()
    test_maybe_build_stack_table_skips_when_existing_candidate_scores_well()
    test_maybe_build_stack_table_keeps_existing_stack_even_once_score_is_good()
    test_maybe_build_stack_table_respects_rebuild_interval()
    print("All detection/stacking.py tests passed.")
