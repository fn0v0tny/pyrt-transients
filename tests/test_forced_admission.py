"""Forced-photometry admission of stack-only candidates
(detection/blind_multicatalog/forced.py), on synthetic frames: a persistent
faint source must be admitted with a sensible lightcurve, a single-frame
flash averaged into the stack must not."""
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table
import astropy.wcs

from pyrt_transient import PipelineConfig
from pyrt_transient.detection.blind_multicatalog import forced
from pyrt_transient.io.naming import get_base_filename

N_FRAMES = 12
SIZE = 96
ZP = 25.0
SIGMA_PSF = 3.0 / 2.3548          # FWHM 3 px
PERSISTENT = (30.0, 30.0, 400.0)  # x0, y0 (0-based), total flux: SNR ~5 per frame
FLASH = (66.0, 66.0, 6000.0)      # in frame FLASH_FRAME only
FLASH_FRAME = 5
REF_STARS = [(15.0, 80.0), (80.0, 15.0), (48.0, 20.0), (20.0, 50.0), (75.0, 48.0), (50.0, 78.0)]
REF_FLUX = 30000.0


def _meta(i, path):
    return {
        "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN", "CRVAL1": 10.0, "CRVAL2": 20.0,
        "CRPIX1": SIZE / 2 + 0.5, "CRPIX2": SIZE / 2 + 0.5,
        "CD1_1": -1.0 / 3600, "CD1_2": 0.0, "CD2_1": 0.0, "CD2_2": 1.0 / 3600,
        "FWHM": 3.0, "GAIN": 1.0, "MAGZERO": ZP, "CTIME": 1_000_000 + 60 * i,
        "EXPTIME": 30.0, "filename": str(path),
    }


def _star(img, x0, y0, flux):
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    img += flux / (2 * np.pi * SIGMA_PSF**2) * np.exp(-((xx - x0)**2 + (yy - y0)**2) / (2 * SIGMA_PSF**2))


def _frames(tmp_path, with_fits=True):
    rng = np.random.default_rng(7)
    tables = []
    for i in range(N_FRAMES):
        path = tmp_path / f"frame{i:02d}.fits"
        img = rng.normal(100.0, 10.0, (SIZE, SIZE))
        _star(img, *PERSISTENT)
        if i == FLASH_FRAME:
            _star(img, *FLASH)
        for x0, y0 in REF_STARS:
            _star(img, x0, y0, REF_FLUX)
        meta = _meta(i, path)
        if with_fits:
            fits.PrimaryHDU(img.astype(np.float32)).writeto(path)
        det = Table({
            "X_IMAGE": [x0 + 1 for x0, _ in REF_STARS], "Y_IMAGE": [y0 + 1 for _, y0 in REF_STARS],
            "MAG_CALIB": np.full(len(REF_STARS), ZP - 2.5 * np.log10(REF_FLUX)),
            "MAGERR_CALIB": np.full(len(REF_STARS), 0.01), "FLAGS": np.zeros(len(REF_STARS), dtype=int),
        })
        w = astropy.wcs.WCS(meta)
        ra, dec = w.all_pix2world(det["X_IMAGE"], det["Y_IMAGE"], 1)
        det["ALPHA_J2000"], det["DELTA_J2000"] = ra, dec
        det.meta.update(meta)
        tables.append(det)
    return tables


def _sky(meta, x0, y0):
    ra, dec = astropy.wcs.WCS(meta).all_pix2world([x0], [y0], 0)
    return float(ra[0]), float(dec[0])


def _stack_epoch(tmp_path, tables):
    """A stack table plus its stack-epoch candidates on disk, with both the
    persistent source and the flash as candidates."""
    stack = Table({"X_IMAGE": [1.0]})
    stack.meta.update({"IS_STACK": True, "filename": "stack.fits",
                       "STACK_INPUTS": [Path(t.meta["filename"]).name for t in tables]})
    rows = []
    for (x0, y0, _), q, mag in ((PERSISTENT, 1.6, 18.8), (FLASH, 1.5, 17.9)):
        ra, dec = _sky(tables[0].meta, x0, y0)
        rows.append({"ALPHA_J2000": ra, "DELTA_J2000": dec, "X_IMAGE": x0 + 1, "Y_IMAGE": y0 + 1,
                     "MAG_CALIB": mag, "MAGERR_CALIB": 0.05, "FWHM_IMAGE": 3.0, "FLAGS": 0,
                     "candidate_type": "new", "magnitude_difference": 0.0, "quality_score": q})
    detection_tables = tables + [stack]
    path = tmp_path / f"{get_base_filename(stack, len(tables))}_transients.ecsv"
    Table(rows=rows).write(path, format="ascii.ecsv")
    return detection_tables


def test_persistent_source_is_admitted_and_the_flash_is_not(tmp_path):
    tables = _frames(tmp_path)
    detection_tables = _stack_epoch(tmp_path, tables)

    cands, lcs = forced.admit_stack_candidates(tmp_path, detection_tables, Table(), {},
                                               config=PipelineConfig())

    assert len(cands) == 1
    ra, dec = _sky(tables[0].meta, *PERSISTENT[:2])
    assert cands["ALPHA_J2000"][0] == pytest.approx(ra)
    assert cands["DELTA_J2000"][0] == pytest.approx(dec)
    assert cands["admission"][0] == "stack+forced"
    lc = lcs[str(cands["transient_id"][0])]
    assert lc["FORCED"].all() and len(lc) >= 0.3 * N_FRAMES
    assert cands["n_detections"][0] == len(lc)
    # Aperture-corrected onto the reference stars' MAG_CALIB scale.
    assert cands["mag_weighted_mean"][0] == pytest.approx(ZP - 2.5 * np.log10(PERSISTENT[2]), abs=0.25)


def test_admission_metrics_separate_a_flash_from_a_persistent_source():
    snr_persistent = np.full(12, 4.5)
    n, frac, share = forced.admission_metrics(snr_persistent, snr_persistent * 80.0, 3.0)
    assert (n, frac) == (12, 1.0) and share == pytest.approx(1 / 12)
    flash_flux = np.r_[np.zeros(5), 5000.0, np.zeros(6)] + np.linspace(-20, 20, 12)
    n, frac, share = forced.admission_metrics(flash_flux / 80.0, flash_flux, 3.0)
    assert frac < 0.3 and share > 0.5


def test_existing_final_candidate_is_not_duplicated(tmp_path):
    tables = _frames(tmp_path)
    detection_tables = _stack_epoch(tmp_path, tables)
    ra, dec = _sky(tables[0].meta, *PERSISTENT[:2])
    existing = Table({"ALPHA_J2000": [ra], "DELTA_J2000": [dec], "quality_score": [5.0],
                      "transient_id": ["transient_existing"]})

    cands, lcs = forced.admit_stack_candidates(tmp_path, detection_tables, existing, {"transient_existing": Table()},
                                               config=PipelineConfig())

    assert list(cands["transient_id"]) == ["transient_existing"]


def test_missing_fits_is_a_no_op(tmp_path):
    tables = _frames(tmp_path, with_fits=False)
    detection_tables = _stack_epoch(tmp_path, tables)
    cands, lcs = forced.admit_stack_candidates(tmp_path, detection_tables, Table(), {},
                                               config=PipelineConfig())
    assert len(cands) == 0 and lcs == {}


def test_disabled_by_config(tmp_path):
    tables = _frames(tmp_path)
    detection_tables = _stack_epoch(tmp_path, tables)
    cfg = PipelineConfig()
    cfg.detection.stack_forced_admission = False
    cands, _ = forced.admit_stack_candidates(tmp_path, detection_tables, Table(), {}, config=cfg)
    assert len(cands) == 0


# --- the real GRB 190919B stack (tests/190919B) ----------------------------

FIXTURE = Path(__file__).parent / "190919B"
# Stack-only candidates of the first 20 unfiltered frames (23:47-23:56 UT):
# the afterglow, and three single-frame flashes (14.0-14.6 mag in exactly one
# frame, 17.5-17.9 in the stack).
REAL_AFTERGLOW = (311.877758, -44.695044)
REAL_FLASHES = [(312.063814, -44.665734), (312.225157, -44.739347), (312.164245, -44.758100)]


def test_real_stack_admits_the_afterglow_and_rejects_the_flashes(tmp_path):
    paths = sorted(FIXTURE.glob("*-N-020-df.ecsv"))[:20]
    if len(paths) < 20:
        pytest.skip("190919B fixture not present")
    tables = []
    for p in paths:
        t = Table.read(p, format="ascii.ecsv")
        t.meta["filename"] = str(p.with_suffix(".fits"))
        tables.append(t)
    stack = Table({"X_IMAGE": [1.0]})
    stack.meta.update({"IS_STACK": True, "filename": "stack.fits",
                       "STACK_INPUTS": [p.with_suffix(".fits").name for p in paths]})
    rows = [{"ALPHA_J2000": ra, "DELTA_J2000": dec, "MAG_CALIB": 17.5, "MAGERR_CALIB": 0.06,
             "FWHM_IMAGE": 3.4, "FLAGS": 0, "candidate_type": "new",
             "magnitude_difference": 0.0, "quality_score": 1.0}
            for ra, dec in [REAL_AFTERGLOW] + REAL_FLASHES]
    Table(rows=rows).write(tmp_path / f"{get_base_filename(stack, 20)}_transients.ecsv",
                           format="ascii.ecsv")

    cands, lcs = forced.admit_stack_candidates(tmp_path, tables + [stack], Table(), {},
                                               config=PipelineConfig())

    assert len(cands) == 1
    assert cands["ALPHA_J2000"][0] == pytest.approx(REAL_AFTERGLOW[0])
    lc = lcs[str(cands["transient_id"][0])]
    assert len(lc) > 10
    # The FRAM team's own photometry (reference/lN.dat) has 16.5-17.3 over
    # these 9 minutes; the first two frames precede the rise.
    assert 16.6 < cands["mag_weighted_mean"][0] < 17.4


def test_stack_inputs_string_selects_exactly_those_frames():
    """build_stack_ecsv writes STACK_INPUTS as a comma-separated string."""
    frames = []
    for i in range(4):
        t = Table({"X_IMAGE": [1.0]})
        t.meta["filename"] = f"/data/obs/frame{i}.fits"
        frames.append(t)
    stack = Table({"X_IMAGE": [1.0]})
    stack.meta.update({"IS_STACK": True, "STACK_INPUTS": "frame1.fits,frame3.fits"})

    picked = forced.stack_input_tables(frames + [stack])

    assert [i for i, _ in picked] == [1, 3]
    stack.meta.pop("STACK_INPUTS")
    assert [i for i, _ in forced.stack_input_tables(frames + [stack])] == [0, 1, 2, 3]
