"""GRB 190919B seen by FRAM-Auger (Malargue), tests/190919B: 121 frames from
the burst night, 40 x 20 s unfiltered (23:47-00:05 UT, starting 36 s after
the INTEGRAL trigger) then 81 x 60 s R.

What makes this field useful: it is at Dec -45, south of Pan-STARRS, and the
afterglow (16.5-17.6 mag; reference light curve from the FRAM team in
tests/190919B/reference/lN.dat, GCN 25794) sits at the single-frame
detection limit, so the pipeline can only report it from a stack.

The ECSVs were generated from the raw frames with tools/make_pyrt_ecsv.sh.
The header GRB_RA/GRB_DEC is the INTEGRAL position, ~40" from the afterglow;
use AFTERGLOW below.

The stacking test queries VizieR and runs pyrt-combine; it only runs with
PYRT_NETWORK_TESTS=1.
"""
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

from pyrt_transient import PipelineConfig
from pyrt_transient.detection import stacking
from pyrt_transient.detection.subtraction import extraction

FIXTURE = Path(__file__).parent / "190919B"
# Optical afterglow, GCN 25789.
AFTERGLOW = SkyCoord(311.87765 * u.deg, -44.69533 * u.deg)

network = pytest.mark.skipif(
    not os.environ.get("PYRT_NETWORK_TESTS"),
    reason="queries VizieR; set PYRT_NETWORK_TESTS=1 to run",
)


def _unfiltered_ecsvs():
    paths = sorted(FIXTURE.glob("*-N-020-df.ecsv"))
    if not paths:
        pytest.skip("190919B ECSVs not present (see tools/make_pyrt_ecsv.sh)")
    return paths


def _tables(limit=None):
    tables = []
    for path in _unfiltered_ecsvs()[:limit]:
        table = Table.read(path, format="ascii.ecsv")
        # Absolute, so stacking finds the FITS whatever the cwd.
        table.meta["filename"] = str(path.with_suffix(".fits"))
        tables.append(table)
    return tables


def _nearest(table, coord):
    pos = SkyCoord(np.asarray(table["ALPHA_J2000"], dtype=float) * u.deg,
                   np.asarray(table["DELTA_J2000"], dtype=float) * u.deg)
    sep = coord.separation(pos).arcsec
    i = int(np.argmin(sep))
    return i, sep[i]


def test_single_frames_rarely_reach_the_afterglow():
    """Precondition for the stacking test: forced photometry puts the
    afterglow at SNR 3-7 in the 20 s frames, below the extraction threshold,
    so a stack that finds it is doing the work."""
    tables = _tables()
    assert len(tables) == 40
    hits = sum(_nearest(t, AFTERGLOW)[1] < 3.0 for t in tables)
    assert hits <= 3


def test_zeropoint_falls_back_to_an_allsky_catalogue_south_of_panstarrs(monkeypatch):
    """Pan-STARRS has no rows at Dec -45. The stack zeropoint used to come
    back NaN there, so every southern stack was discarded; it must retry
    against ATLAS-RefCat2. VizieR is replaced by the frame's own calibrated
    detections standing in for ATLAS."""
    from stdpipe import catalogs as stdpipe_catalogs

    path = _unfiltered_ecsvs()[10]
    ref = Table.read(path, format="ascii.ecsv")
    good = (np.asarray(ref["FLAGS"]) == 0) & (np.asarray(ref["MAGERR_CALIB"], dtype=float) < 0.05)
    atlas = Table({
        "RAJ2000": np.asarray(ref["ALPHA_J2000"], dtype=float)[good],
        "DEJ2000": np.asarray(ref["DELTA_J2000"], dtype=float)[good],
        "rmag": np.asarray(ref["MAG_CALIB"], dtype=float)[good],
        "e_rmag": np.full(good.sum(), 0.02),
    })
    queried = []

    def fake_get_cat_vizier(ra0, dec0, sr0, catalog="ps1", verbose=False):
        queried.append(catalog)
        return atlas if catalog == "atlas" else Table()

    monkeypatch.setattr(stdpipe_catalogs, "get_cat_vizier", fake_get_cat_vizier)

    zp, zp_err = extraction.calibrate_science_zeropoint(path.with_suffix(".fits"))

    assert queried == ["ps1", "atlas"]
    assert np.isfinite(zp) and zp_err < 0.05
    assert zp == pytest.approx(ref.meta["MAGZERO"], abs=0.3)


@network
@pytest.mark.skipif(shutil.which("pyrt-combine") is None, reason="pyrt-combine not installed")
def test_stack_of_the_first_twenty_frames_recovers_the_afterglow(tmp_path):
    """The pipeline's own stacking path, default config, on the first 20
    frames (23:47-23:56 UT). lN.dat has the afterglow at 16.5-17.0 over
    23:48-23:56 and not yet risen in the first two frames, so the 20-frame
    mean flux corresponds to ~16.9 mag."""
    tables = _tables(limit=20)
    single_maglim = np.median([t.meta["MAGLIM"] for t in tables])

    stack = stacking.maybe_build_stack_table(tmp_path, tables, PipelineConfig())

    assert stack is not None, "stack discarded -- no calibration catalogue for the field?"
    assert stack.meta["IS_STACK"] and stack.meta["NCOMBINE"] == 20
    assert stack.meta["MAGLIM"] > single_maglim + 1.0
    i, sep = _nearest(stack, AFTERGLOW)
    assert sep < 3.0
    assert 16.6 < stack["MAG_CALIB"][i] < 17.3
    assert stack["MAGERR_CALIB"][i] < 0.1
