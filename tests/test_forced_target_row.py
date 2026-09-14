"""pyrt writes one NUMBER==0 row per frame: the forced measurement at the
position the telescope was pointed at. It repeats at the same sky position in
every frame, so the detection side turned it into a persistent "new" source --
Q 7.1 in 20 of 20 epochs on the rebuilt GRB230818A replay page, Q 19.8 in 46
of 46 on GRB220403B, at a median SNR of 1.1 and 1.9. On D50's obs_104223 the
same row carries no magnitude at all and sits ~10 deg off the field.

It is kept only when it is a real measurement of the target."""
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pyrt_transient.io.ecsv import drop_insignificant_forced_row, open_ecsv_file  # noqa: E402


def _frame(mag, magerr, n_other=3):
    """A detection table whose first row is the forced NUMBER==0 one."""
    return Table({
        "NUMBER": np.arange(0, n_other + 1),
        "ALPHA_J2000": np.linspace(285.88, 285.90, n_other + 1),
        "DELTA_J2000": np.linspace(40.87, 40.90, n_other + 1),
        "MAG_CALIB": np.array([mag] + [17.0] * n_other),
        "MAGERR_CALIB": np.array([magerr] + [0.02] * n_other),
    })


def test_a_noise_measurement_at_the_target_is_not_a_detection():
    # GRB230818A's median forced row: mag ~20.3 +- 0.954 -> SNR 1.1.
    kept = drop_insignificant_forced_row(_frame(20.31, 0.954))

    assert list(kept["NUMBER"]) == [1, 2, 3]


def test_a_row_without_a_magnitude_goes():
    # obs_104223: no magnitude in any of its 46 frames.
    kept = drop_insignificant_forced_row(_frame(np.nan, np.nan))

    assert list(kept["NUMBER"]) == [1, 2, 3]


def test_a_real_measurement_of_the_target_stays():
    # Bright enough to be the target actually being seen: SNR 10.9.
    kept = drop_insignificant_forced_row(_frame(16.2, 0.1))

    assert list(kept["NUMBER"]) == [0, 1, 2, 3]
    assert float(kept["MAG_CALIB"][0]) == pytest.approx(16.2)


def test_a_row_without_a_calibrated_magnitude_goes_even_with_an_instrumental_one():
    # Candidates are built from MAG_CALIB, so a valid MAG_AUTO does not make
    # the forced row usable; falling back to it would bring the no-magnitude
    # rows of obs_104223 back.
    frame = _frame(np.nan, np.nan)
    frame["MAG_AUTO"] = np.array([-8.5] + [-10.0] * 3)
    frame["MAGERR_AUTO"] = np.array([0.05] + [0.02] * 3)

    assert list(drop_insignificant_forced_row(frame)["NUMBER"]) == [1, 2, 3]


@pytest.mark.parametrize("magerr, snr, kept", [(0.10, 10.9, True), (0.36, 3.0, True),
                                               (0.40, 2.7, True), (0.54, 2.0, True),
                                               (0.60, 1.8, False), (2.358, 0.5, False)])
def test_the_cut_is_on_significance(magerr, snr, kept):
    result = drop_insignificant_forced_row(_frame(19.0, magerr))

    assert (0 in list(result["NUMBER"])) is kept, f"SNR {snr}"


def test_tables_without_the_row_or_the_column_are_untouched():
    plain = Table({"NUMBER": [1, 2], "MAG_CALIB": [17.0, 18.0], "MAGERR_CALIB": [0.02, 0.03]})
    assert len(drop_insignificant_forced_row(plain)) == 2

    no_number = Table({"MAG_CALIB": [17.0], "MAGERR_CALIB": [0.02]})
    assert len(drop_insignificant_forced_row(no_number)) == 1
    assert drop_insignificant_forced_row(Table()) is not None


def test_every_frame_read_by_the_pipeline_is_filtered(tmp_path):
    path = tmp_path / "20230818232804-401-N-010-df.ecsv"
    _frame(20.31, 0.954).write(path, format="ascii.ecsv")   # SNR 1.1, noise

    det = open_ecsv_file(str(path))

    assert list(det["NUMBER"]) == [1, 2, 3]
    assert det.meta["filename"] == str(path)
