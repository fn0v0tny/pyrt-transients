"""Catalogue stars are matched to detections in pixel space, so the
projection must reproduce the frame's own WCS -- including its SIP
distortion or zenithal projection terms, which the detections'
X_IMAGE/Y_IMAGE are measured through."""
import glob
from pathlib import Path

import astropy.wcs
import numpy as np
import pytest
from astropy.table import Table

from pyrt_transient.catalog import CatTransients

TESTS = Path(__file__).parent


def _fixture_frame(pattern, **extra_meta):
    paths = [p for p in sorted(glob.glob(str(TESTS / pattern)))
             if not p.endswith("_transients.ecsv")]
    if not paths:
        pytest.skip(f"fixture {pattern} not present")
    det = Table.read(paths[0], format="ascii.ecsv")
    det.meta.update(extra_meta)
    return det


def _project(det):
    """Pipeline projection of a "catalogue" sitting exactly at the
    detections' sky positions."""
    cat = Table({"radeg": np.asarray(det["ALPHA_J2000"], dtype=float),
                 "decdeg": np.asarray(det["DELTA_J2000"], dtype=float)})
    return CatTransients._transform_catalog_to_pixel(cat, det)


def _full_wcs_pixels(det):
    """The same positions through the frame's complete WCS."""
    from pyrt_transient.core.wcs_meta import wcs_header_from_meta
    header = wcs_header_from_meta(det.meta)
    x, y = astropy.wcs.WCS(header).all_world2pix(
        np.asarray(det["ALPHA_J2000"], dtype=float),
        np.asarray(det["DELTA_J2000"], dtype=float), 1)
    return np.column_stack([x, y])


def test_projection_keeps_sip_on_d50_frames():
    """The D50 ECSVs' ALPHA/DELTA were computed from X/Y through their
    order-2 SIP WCS, so they must project straight back. Dropping the SIP
    terms was up to 0.27 px off here, and 2 px in the corners of the FRAM
    frames' order-3 header SIP -- beyond the 1 px adaptive match floor, so
    the same edge stars came back "new" in every epoch."""
    det = _fixture_frame("210619B/*-df.ecsv")
    xy = _project(det)
    offsets = np.hypot(xy[:, 0] - np.asarray(det["X_IMAGE"], dtype=float),
                       xy[:, 1] - np.asarray(det["Y_IMAGE"], dtype=float))
    assert offsets.max() < 0.05


def test_projection_keeps_the_refitted_zpn_solution_of_fram_frames():
    """The FRAM ECSVs carry pyrt's refitted ZPN solution (PV2_3 ~ 87).
    Their ALPHA/DELTA lag the final refit by up to ~1 px, so the reference
    here is the frame's own complete WCS rather than X_IMAGE/Y_IMAGE."""
    det = _fixture_frame("190919B/*-N-020-df.ecsv")
    assert str(det.meta["CTYPE1"]).endswith("ZPN")
    assert np.abs(_project(det) - _full_wcs_pixels(det)).max() < 1e-3


def test_projection_keeps_zpn_terms():
    """The same ZPN solution without the fixture: flattened to TAN, edge
    stars landed up to ~3 px off."""
    meta = {"CTYPE1": "RA---ZPN", "CTYPE2": "DEC--ZPN",
            "CRVAL1": 311.90975, "CRVAL2": -44.72125, "CRPIX1": 569.06, "CRPIX2": 562.02,
            "CD1_1": 5.0922e-4, "CD1_2": 1.3786e-5, "CD2_1": 1.3668e-5, "CD2_2": -5.0978e-4,
            "PV2_1": 1.0, "PV2_3": 87.1}
    x, y = (g.ravel() for g in np.meshgrid(np.linspace(1, 1024, 9), np.linspace(1, 1024, 9)))
    ra, dec = astropy.wcs.WCS(meta).all_pix2world(x, y, 1)
    det = Table({"X_IMAGE": x, "Y_IMAGE": y}, meta=meta)

    xy = CatTransients._transform_catalog_to_pixel(Table({"radeg": ra, "decdeg": dec}), det)

    assert np.hypot(xy[:, 0] - x, xy[:, 1] - y).max() < 0.05


def test_projection_survives_nan_meta():
    """pyrt writes ASTSIGMA=nan when its astrometric refit has nothing to
    fit; astropy refuses NaN header cards, which used to send the matcher
    to its legacy fallback."""
    det = _fixture_frame("190919B/*-N-020-df.ecsv", ASTSIGMA=float("nan"))
    assert np.abs(_project(det) - _full_wcs_pixels(det)).max() < 1e-3


def test_stack_ecsv_carries_the_wcs_the_matcher_projects_with():
    """Stack (and diff) ECSVs are written by extraction._adapt_to_ecsv_schema.
    It copied no WCS keys, so the matcher projected every catalogue star
    ~860 px away and flagged 560 of 571 stack detections "new"."""
    from astropy.io import fits
    from pyrt_transient.detection.subtraction import extraction

    paths = sorted((TESTS / "190919B").glob("*-N-020-df.fits"))
    if not paths:
        pytest.skip("190919B fixture not present")
    header = fits.getheader(paths[0])
    x0, y0 = np.linspace(20, 1000, 25), np.linspace(30, 990, 25)   # SEP: 0-based
    ra, dec = astropy.wcs.WCS(header).all_pix2world(x0, y0, 0)
    obj = Table({"x": x0, "y": y0, "ra": ra, "dec": dec,
                 "a": np.full(25, 2.0), "b": np.full(25, 1.8),
                 "flux": np.full(25, 1e4), "fluxerr": np.full(25, 1e2),
                 "flags": np.zeros(25, dtype=int)})

    det = extraction._adapt_to_ecsv_schema(obj, header, 23.7, 0.01, 0.00051, 4.0)
    xy = CatTransients._transform_catalog_to_pixel(Table({"radeg": ra, "decdeg": dec}), det)

    offsets = np.hypot(xy[:, 0] - np.asarray(det["X_IMAGE"]), xy[:, 1] - np.asarray(det["Y_IMAGE"]))
    assert offsets.max() < 0.05


def test_projection_survives_non_scalar_meta():
    """The stack ECSV once carried STACK_INPUTS as a list; astropy refuses a
    list header card, so every stack epoch fell back to the legacy matcher
    and produced no candidates at all."""
    det = _fixture_frame("190919B/*-N-020-df.ecsv", STACK_INPUTS=["a.fits", "b.fits"])
    assert np.abs(_project(det) - _full_wcs_pixels(det)).max() < 1e-3


def test_projection_survives_an_over_long_meta_string():
    """STACK_INPUTS as one comma-separated string (20 filenames) is longer
    than a FITS header card can hold; serialising the whole meta for the WCS
    failed on it, and the stack epoch fell back to the legacy matcher."""
    det = _fixture_frame("190919B/*-N-020-df.ecsv", STACK_INPUTS=",".join(["20190919234716-909-N-020-df.fits"] * 20))
    assert np.abs(_project(det) - _full_wcs_pixels(det)).max() < 1e-3
