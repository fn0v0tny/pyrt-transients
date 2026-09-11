"""Catalogue-level injection (validation/injection.py) against the real
210619B fixture frames: the appended rows must be positionally consistent
with the frame WCS, photometrically consistent with the noise model, and
shaped like a real star of the same brightness."""
import glob
from pathlib import Path

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u

from pyrt_transient.validation import injection

FIXTURE = Path(__file__).parent / "210619B"


@pytest.fixture(scope="module")
def frames():
    files = sorted(f for f in glob.glob(str(FIXTURE / "*-df.ecsv")))[:3]
    if not files:
        pytest.skip("210619B fixture not present")
    tabs = []
    for f in files:
        t = Table.read(f, format="ascii.ecsv")
        t.meta["filename"] = f
        tabs.append(t)
    return tabs


def test_wcs_from_meta_reproduces_catalogue_pixels(frames):
    t = frames[0]
    wcs = injection.wcs_from_meta(t.meta)
    x, y = wcs.all_world2pix(np.asarray(t["ALPHA_J2000"][:50]), np.asarray(t["DELTA_J2000"][:50]), 1)
    assert np.max(np.abs(x - t["X_IMAGE"][:50])) < 0.05
    assert np.max(np.abs(y - t["Y_IMAGE"][:50])) < 0.05


def test_draw_sources_clear_of_real_detections_and_inside_frame(frames):
    rng = np.random.default_rng(1)
    srcs = injection.draw_sources(frames[0], 20, rng, min_sep_real_arcsec=10.0)
    assert len(srcs) == 20
    real = SkyCoord(np.asarray(frames[0]["ALPHA_J2000"]) * u.deg, np.asarray(frames[0]["DELTA_J2000"]) * u.deg)
    pos = SkyCoord([s.ra for s in srcs] * u.deg, [s.dec for s in srcs] * u.deg)
    idx, sep, _ = pos.match_to_catalog_sky(real)
    assert sep.arcsec.min() >= 10.0
    wcs = injection.wcs_from_meta(frames[0].meta)
    x, y = wcs.all_world2pix(pos.ra.deg, pos.dec.deg, 1)
    w, h = injection.frame_size(frames[0].meta)
    assert np.all((x > 30) & (x < w - 30) & (y > 30) & (y < h - 30))


def test_light_curve_model():
    s = injection.InjectedSource("a", 0, 0, m0=17.0, alpha=1.0, t0=0.0, t_ref=30.0)
    assert s.magnitude_at(30.0) == pytest.approx(17.0)
    assert s.magnitude_at(300.0) == pytest.approx(17.0 + 2.5)   # x10 in time, alpha=1 -> +2.5 mag
    c = injection.InjectedSource("b", 0, 0, m0=17.0, alpha=0.0, t0=0.0, t_ref=30.0)
    assert c.magnitude_at(1e5) == 17.0


def test_inject_epoch_rows_are_consistent(frames):
    rng = np.random.default_rng(2)
    t = frames[0]
    srcs = injection.draw_sources(t, 10, rng, mag_range=(15.0, 17.0))
    out, truth = injection.inject_epoch(t, srcs, rng, force_detect=True)
    assert len(out) == len(t) + 10
    assert out.colnames == t.colnames
    assert out.meta["N_INJECTED"] == 10
    new = out[len(t):]
    wcs = injection.wcs_from_meta(t.meta)
    x, y = wcs.all_world2pix(np.asarray(new["ALPHA_J2000"]), np.asarray(new["DELTA_J2000"]), 1)
    assert np.max(np.abs(x - new["X_IMAGE"])) < 0.05
    tr = Table(rows=truth)
    assert np.all(np.abs(tr["mag_obs"] - tr["mag_true"]) < 5 * tr["magerr"])
    assert np.all(tr["magerr"] < 0.1)
    assert np.all(new["FLAGS"] == 0)
    assert set(np.asarray(new["FWHM_IMAGE"])).issubset(set(np.asarray(t["FWHM_IMAGE"])))
    assert new["NUMBER"].min() > t["NUMBER"].max()


def test_faint_sources_drop_out_below_maglimit(frames):
    rng = np.random.default_rng(3)
    t = frames[0]
    maglimit = float(t.meta["MAGLIMIT"])
    srcs = injection.draw_sources(t, 30, rng, mag_range=(maglimit + 1.5, maglimit + 2.5))
    out, truth = injection.inject_epoch(t, srcs, rng)
    tr = Table(rows=truth)
    # The noise model gives ~1 mag errors this far below the floor, so an
    # occasional upward fluctuation past MAGLIMIT is expected (Eddington
    # bias), but the large majority must be absent from the catalogue.
    assert np.mean(tr["detected"]) < 0.2
    assert len(out) == len(t) + int(np.sum(tr["detected"]))
    bright = injection.draw_sources(t, 30, rng, mag_range=(14.0, 16.0))
    out2, truth2 = injection.inject_epoch(t, bright, rng)
    assert np.all(Table(rows=truth2)["detected"])


def test_inject_campaign_fades_between_epochs(frames):
    rng = np.random.default_rng(4)
    t_ref = injection.epoch_mid_time(frames[0].meta)
    srcs = injection.draw_sources(frames[0], 5, rng, mag_range=(15.0, 15.5), alphas=(1.0,), t0=t_ref - 30.0)
    tabs, truth = injection.inject_campaign(frames, srcs, rng)
    assert len(tabs) == len(frames)
    for s in srcs:
        rows = truth[truth["source_id"] == s.source_id]
        assert np.all(np.diff(rows["mag_true"]) > 0)


def test_bright_donor_may_be_saturated_but_never_blended():
    """A 12 mag injection on a frame whose unflagged stars start at 14 mag
    takes a saturated 12 mag star as its shape/flag template; a blended one
    is never used."""
    from astropy.table import Table
    from pyrt_transient.validation.injection import _donor_index
    t = Table({
        "MAG_CALIB": [12.0, 12.1, 14.0, 14.2, 16.0],
        "FLAGS": [4, 2, 0, 0, 0],
        "FWHM_IMAGE": [3.0, 3.0, 2.5, 2.5, 2.5],
        "ELLIPTICITY": [0.1, 0.1, 0.1, 0.1, 0.1],
    })
    rng = np.random.default_rng(0)
    picks = {_donor_index(t, 12.05, rng, n_pool=1) for _ in range(20)}
    assert picks == {0}                       # the saturated 12.0 mag star, not the blend
    picks = {_donor_index(t, 14.1, rng, n_pool=2) for _ in range(20)}
    assert picks <= {2, 3}                    # unflagged donors when they exist


def test_injected_bright_row_keeps_the_donors_saturation_flag(frames):
    """The saturated donor of the test above is only half the point: the row
    written from it used to have FLAGS zeroed, so an injected 12 mag source
    looked cleaner than the real 12 mag stars it stands in for and bright-end
    completeness came out too high."""
    table = frames[0]
    src = injection.InjectedSource(
        source_id="b000", ra=float(table["ALPHA_J2000"][0]), dec=float(table["DELTA_J2000"][0]),
        m0=12.0, alpha=0.0,
        t0=injection.epoch_mid_time(table.meta) - 30.0, t_ref=injection.epoch_mid_time(table.meta),
    )
    out, truth = injection.inject_epoch(table, [src], np.random.default_rng(3), force_detect=True)
    assert truth[0]["detected"]
    injected = out[len(table):]
    assert len(injected) == 1
    flags = int(injected["FLAGS"][0])
    assert flags & 2 == 0, "an injected point source is never blended"
    # Whatever the donor carried, only the saturation bit survives.
    assert flags in (0, 4)


def test_injected_bright_row_carries_the_saturated_donors_flag(frames):
    """Same frame, but arranged so the only donor near the injected
    magnitude is a saturated one: the injected row must then be saturated
    too, which is the whole reason _donor_index admits such donors."""
    table = frames[0].copy()
    mags = np.full(len(table), 16.0)
    mags[0] = 12.0
    table["MAG_CALIB"] = mags
    flags_col = np.zeros(len(table), dtype=int)
    flags_col[0] = 4
    table["FLAGS"] = flags_col
    t_ref = injection.epoch_mid_time(table.meta)
    src = injection.InjectedSource(
        source_id="b000", ra=float(table["ALPHA_J2000"][0]), dec=float(table["DELTA_J2000"][0]),
        m0=12.0, alpha=0.0, t0=t_ref - 30.0, t_ref=t_ref,
    )
    out, _ = injection.inject_epoch(table, [src], np.random.default_rng(3), force_detect=True)
    assert int(out["FLAGS"][len(table)]) == 4


def test_wcs_from_meta_keeps_zpn_terms_of_refitted_frames():
    """pyrt's astrometric refit writes ZPN (with PV terms) for FRAM's NF4
    camera; without the PV cards wcslib refuses the projection outright."""
    files = sorted(glob.glob(str(Path(__file__).parent / "190919B" / "*-N-020-df.ecsv")))
    if not files:
        pytest.skip("190919B fixture not present")
    t = Table.read(files[0], format="ascii.ecsv")

    wcs = injection.wcs_from_meta(t.meta)

    assert wcs.wcs.ctype[0].endswith("ZPN")
    x, y = wcs.all_world2pix(np.asarray(t["ALPHA_J2000"]), np.asarray(t["DELTA_J2000"]), 1)
    # ALPHA/DELTA lag pyrt's final refit by ~0.1 px (up to ~1 px at worst).
    assert np.median(np.hypot(x - t["X_IMAGE"], y - t["Y_IMAGE"])) < 0.2
