"""Cutout geometry in frontend_generator, checked against real FRAM frames
from tests/190919B."""
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from pyrt_transient.frontend_generator import _lightcurve_pixel_positions, _read_padded_cutout

FIXTURE = Path(__file__).parent / "190919B"


def _frames(n):
    paths = sorted(FIXTURE.glob("*-N-020-df.ecsv"))
    if len(paths) < n:
        pytest.skip("190919B ECSVs not present (see tools/make_pyrt_ecsv.sh)")
    return paths[:n]


def _ecsv_wcs(meta):
    """The frame's own (refitted) WCS from ECSV meta, projection terms kept."""
    header = {k: v for k, v in meta.items()
              if not (isinstance(v, float) and not np.isfinite(v))
              and not (len(k) > 1 and k[-1] == "T" and k[:-1] in
                       ("CTYPE1", "CTYPE2", "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2",
                        "CDELT1", "CDELT2", "CROTA2"))}
    return WCS(header)


def test_lightcurve_positions_land_on_the_0_based_frame_pixel():
    """A lightcurve holds one detection row per epoch. Its X_IMAGE/Y_IMAGE
    are 1-based; the cutout reader is 0-based, so the position handed to it
    must match the frame WCS evaluated with origin 0 -- not be one pixel
    off, as it was."""
    rows, wcs_by_stem = [], {}
    for ecsv in _frames(5):
        det = Table.read(ecsv, format="ascii.ecsv")
        wcs_by_stem[ecsv.stem] = _ecsv_wcs(det.meta)
        j = int(np.argmin(np.asarray(det["MAGERR_CALIB"], dtype=float)))
        rows.append({
            "source_file": str(ecsv),
            "ALPHA_J2000": float(det["ALPHA_J2000"][j]),
            "DELTA_J2000": float(det["DELTA_J2000"][j]),
            "X_IMAGE": float(det["X_IMAGE"][j]),
            "Y_IMAGE": float(det["Y_IMAGE"][j]),
        })
    lightcurve = Table(rows=rows)

    positions = _lightcurve_pixel_positions(lightcurve)

    assert set(wcs_by_stem) <= set(positions)
    for row in lightcurve:
        path = Path(row["source_file"])
        x0, y0 = wcs_by_stem[path.stem].all_world2pix(row["ALPHA_J2000"], row["DELTA_J2000"], 0)
        x, y = positions[path.stem]
        # ALPHA/DELTA lag pyrt's final astrometric refit by ~0.1 px; an
        # off-by-one position is 1 px out.
        assert x == pytest.approx(float(x0), abs=0.3)
        assert y == pytest.approx(float(y0), abs=0.3)


def test_lightcurve_positions_also_key_the_astrometry_solved_dft_stem():
    # D50 production: the epoch row points at the "-df" catalog, the
    # cutout is read from the WCS-solved "-dft" image.
    lightcurve = Table(rows=[("/obs/20250813-a-df.ecsv", 11.0, 21.0),
                             ("/obs/20250813-b-df.ecsv", 31.0, 41.0)],
                       names=("source_file", "X_IMAGE", "Y_IMAGE"))

    positions = _lightcurve_pixel_positions(lightcurve)

    assert positions["20250813-a-dft"] == positions["20250813-a-df"] == (10.0, 20.0)
    assert positions["20250813-b-dft"] == (30.0, 40.0)


def test_corner_cutout_keeps_its_size_and_registers_to_the_frame():
    fits_path = _frames(1)[0].with_suffix(".fits")
    half = 20
    with fits.open(fits_path) as hdul:
        data = hdul[0].data
        ny, nx = data.shape
        x, y = 3, ny - 2      # top-left corner: 17 columns and 18 rows fall outside
        out, xmin, ymin = _read_padded_cutout(hdul[0].section, x, y, half, nx, ny)

    assert out.shape == (2 * half, 2 * half)
    assert (xmin, ymin) == (x - half, y - half)
    inside_cols, inside_rows = slice(-xmin, None), slice(None, ny - ymin)
    assert np.isnan(out[:, :-xmin]).all()
    assert np.isnan(out[ny - ymin:, :]).all()
    assert np.isfinite(out[inside_rows, inside_cols]).all()
    # Frame pixel (px, py) sits at out[py - ymin, px - xmin].
    for px, py in [(0, ny - 1), (7, ny - 5), (x + half - 1, y - half)]:
        assert out[py - ymin, px - xmin] == pytest.approx(float(data[py, px]))
