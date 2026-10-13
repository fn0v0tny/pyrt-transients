"""Unit tests for detection/host_search/ -- the host-galaxy search in a
trigger's error circle.

Network catalogues, survey cutouts and hotpants are replaced by fakes;
what is tested is the prior ranking, the frame selection and zeropoint
logic on synthetic frames, the photometry on a synthetic difference
image, the template cache lookup, the pipeline gate/throttle, and the
outputs. No pytest dependency -- run with `python3 tests/test_host_search.py`.
"""
import json
import logging
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from pyrt_transient.config_trans import PipelineConfig
from pyrt_transient.detection.host_search import galaxies, search, stack, subtract, templates

_LOG = logging.getLogger("test_host_search")


# ---------------------------------------------------------------- helpers
def _wcs(ra=21.8, dec=12.9, pixscale=1.19, n=200):
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [ra, dec]
    w.wcs.crpix = [n / 2 + 0.5, n / 2 + 0.5]
    w.wcs.cdelt = [-pixscale / 3600, pixscale / 3600]
    return w


def _gauss(n, x0, y0, flux, fwhm):
    yy, xx = np.mgrid[:n, :n]
    s = fwhm / 2.355
    return flux * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * s * s)) / (2 * np.pi * s * s)


def _frame(n=200, fwhm=3.5, sky=100.0, rms=3.0, exptime=300.0, zp=28.0, stars=None, seed=0):
    """A synthetic frame plus the ECSV-like table pyrt would write for it."""
    rng = np.random.default_rng(seed)
    img = sky + rng.normal(0, rms, (n, n)).astype(np.float32)
    stars = stars or [(40 + 25 * i, 30 + 20 * i, 15.0 + 0.5 * i) for i in range(6)]
    xs, ys, ms = [], [], []
    for x, y, m in stars:
        img += _gauss(n, x, y, 10 ** (0.4 * (zp - m)), fwhm)
        xs.append(x + 1); ys.append(y + 1); ms.append(m)
    tab = Table({"X_IMAGE": xs, "Y_IMAGE": ys, "MAG_CALIB": ms, "MAGERR_CALIB": [0.01] * len(ms)})
    tab.meta.update({"FWHM": fwhm, "EXPTIME": exptime, "PHFILTER": "Sloan_r", "CTIME": 1e9 + seed,
                     "GRB_RA": 21.8, "GRB_DEC": 12.9, "GRB_ERR": 0.05, "FIELD": 0.066})
    return img, tab


def _write_frame(d, name, img, tab, wcs):
    p = Path(d) / name
    fits.writeto(p, img, wcs.to_header(), overwrite=True)
    tab.meta["filename"] = str(p.with_suffix(".ecsv"))
    return p


# ---------------------------------------------------------------- tests
def test_prior_ranks_brighter_first_and_counts_scale_with_area():
    gals = [{"ra": 1, "dec": 1, "mag": 19.0}, {"ra": 1, "dec": 1, "mag": 17.7}, {"ra": 1, "dec": 1, "mag": 20.5}]
    out = galaxies.rank_by_prior(gals, 186.0)
    assert [g["mag"] for g in out] == [17.7, 19.0, 20.5]
    assert [g["prior_rank"] for g in out] == [1, 2, 3]
    assert 3.5 < out[0]["n_exp"] < 5.5, out[0]["n_exp"]        # EP261006a: ~4.4 for 17.72 in a 3.1' circle
    small = galaxies.rank_by_prior([{"ra": 1, "dec": 1, "mag": 17.7}], 10.0)
    assert small[0]["n_exp"] < 0.02 and small[0]["p_chance"] < 0.02, "a 10\" FXT circle makes the host unique"
    print("test_prior_ranks_brighter_first_and_counts_scale_with_area: PASS")


def test_query_falls_back_to_the_next_catalogue(monkeypatch=None):
    calls = []

    def fake_ls(ra, dec, r, timeout):
        calls.append("ls"); return []

    def fake_ps1(coords, r, timeout):
        calls.append("ps1")
        return [{"ra": 21.8, "dec": 12.9, "catalog": "panstarrs", "mags": {"g": None, "r": 18.0, "i": None, "z": None},
                 "radius_arcsec": None},
                {"ra": 21.81, "dec": 12.9, "catalog": "panstarrs", "mags": {"g": None, "r": 22.0, "i": None, "z": None},
                 "radius_arcsec": None}]
    saved = galaxies._ls_cone_query, galaxies._ps1_query
    galaxies._ls_cone_query, galaxies._ps1_query = fake_ls, fake_ps1
    try:
        out = galaxies.query_galaxies_in_circle(21.8, 12.9, 186.0, max_mag=21.0, log=_LOG)
    finally:
        galaxies._ls_cone_query, galaxies._ps1_query = saved
    assert calls == ["ls", "ps1"]
    assert len(out) == 1 and out[0]["mag"] == 18.0, "the 22 mag one is beyond max_mag"
    # every catalogue failing is None, not an empty list
    def boom(*a, **k):
        raise RuntimeError("down")
    galaxies._ls_cone_query, galaxies._ps1_query = boom, boom
    try:
        assert galaxies.query_galaxies_in_circle(21.8, 12.9, 186.0, log=_LOG) is None
    finally:
        galaxies._ls_cone_query, galaxies._ps1_query = saved
    print("test_query_falls_back_to_the_next_catalogue: PASS")


def test_frame_selection_and_zeropoint():
    img, tab = _frame(zp=28.0)
    zp = stack.frame_zeropoint(img, tab, 3.5)
    assert zp is not None and abs(zp - 28.0) < 0.05, zp
    with tempfile.TemporaryDirectory() as d:
        w = _wcs()
        long1 = _write_frame(d, "a-dft.fits", *_frame(exptime=300, seed=1), w)
        long2 = _write_frame(d, "b-dft.fits", *_frame(exptime=300, seed=2), w)
        short = _write_frame(d, "c-dft.fits", *_frame(exptime=0.1, seed=3), w)
        tabs = {}
        for p in (long1, long2, short):
            _, t = _frame(exptime=300 if p is not short else 0.1)
            t.meta["filename"] = str(p.with_suffix(".ecsv"))
            tabs[p] = t
        chosen = stack.select_frames([(p, tabs[p]) for p in (long1, long2, short)], 0.5, 40)
        assert [c[0] for c in chosen] == [long1, long2], "the 0.1 s pointing frame is left out"
        assert stack.select_frames([(Path(d) / "missing.fits", tabs[long1])], 0.5, 40) == []
    print("test_frame_selection_and_zeropoint: PASS")


def test_build_stack_median_of_calibrated_frames():
    try:
        import reproject  # noqa: F401
    except ImportError:
        print("test_build_stack_median_of_calibrated_frames: SKIP (reproject not installed)")
        return
    with tempfile.TemporaryDirectory() as d:
        w = _wcs()
        pairs = []
        for i, zp in enumerate((28.0, 27.5, 28.2)):      # same sky, different gains
            img, tab = _frame(zp=zp, seed=10 + i, rms=2.0)
            p = _write_frame(d, f"f{i}-dft.fits", img, tab, w)
            pairs.append((p, tab))
        res = stack.build_stack(pairs, _LOG)
    assert res is not None and res.n_frames == 3
    assert abs(res.zp - 28.0) < 0.1, res.zp                 # on the best-seeing (all equal) reference's scale
    assert 2.5 < res.fwhm_px < 4.5, res.fwhm_px
    x, y = 40, 30                                             # the 15.0 mag star
    flux = res.data[y - 6:y + 7, x - 6:x + 7].sum()
    assert abs(res.zp - 2.5 * np.log10(flux) - 15.0) < 0.1
    print("test_build_stack_median_of_calibrated_frames: PASS")


def test_measure_galaxies_and_population_sigma():
    n, fwhm, zp = 200, 4.0, 28.0
    w = _wcs(n=n)
    rng = np.random.default_rng(5)
    sci = np.zeros((n, n), dtype=np.float32)
    diff = rng.normal(0, 2.0, (n, n)).astype(np.float32)
    gals = []
    for i in range(8):
        x, y = 30 + 20 * i, 100
        gflux = 10 ** (0.4 * (zp - 18.0))
        sci += _gauss(n, x, y, gflux, 2 * fwhm)
        ra, dec = w.all_pix2world([[x, y]], 0)[0]
        gals.append({"ra": float(ra), "dec": float(dec), "mag": 18.0, "prior_rank": i + 1})
    # galaxy 3 carries a point-like 5 % nuclear excess; everybody else a 1 % blemish
    for i in range(8):
        x, y = 30 + 20 * i, 100
        frac = 0.05 if i == 3 else 0.01
        diff += _gauss(n, x, y, frac * 10 ** (0.4 * (zp - 18.0)) * 0.3, fwhm)
    mask = np.zeros((n, n), dtype=bool)
    rows = subtract.measure_galaxies(diff, sci, mask, w, gals, fwhm, zp, aperture_fwhm=0.5)
    assert len(rows) == 8
    med, mad = subtract.population_sigma(rows, bright_mag=19.5)
    best = max(rows, key=lambda r: r["excess_sigma"])
    assert best["prior_rank"] == 4, [round(r["excess_sigma"], 1) for r in rows]
    assert best["excess_sigma"] > 3, best["excess_sigma"]
    assert best["residual_snr"] > 3
    # a galaxy on masked pixels is not measured
    mask[90:110, 20:40] = True
    assert len(subtract.measure_galaxies(diff, sci, mask, w, gals, fwhm, zp)) == 7
    print("test_measure_galaxies_and_population_sigma: PASS")


def test_template_cache_lookup_tolerates_pointing_offset():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        hdr = _wcs(ra=21.806, dec=12.900, pixscale=0.5, n=2400).to_header()
        hdr["HSSRC"] = "legacysurvey"; hdr["HSBAND"] = "r"; hdr["HSRA"] = 21.806; hdr["HSDEC"] = 12.9; hdr["HSSIZE"] = 20.0
        fits.writeto(d / "t.fits", np.ones((10, 10), dtype=np.float32), hdr, overwrite=True)
        # the pointing 1.1' from the trigger position still uses it
        assert templates.find_cached_template(d, 21.787, 12.900, "r", 17.0) is not None
        assert templates.find_cached_template(d, 21.787, 12.900, "g", 17.0) is None, "other band"
        assert templates.find_cached_template(d, 21.9, 12.9, "r", 17.0) is None, "too far"
        assert templates.find_cached_template(d, 21.806, 12.9, "r", 25.0) is None, "too small"
        assert templates.prefetch_template(21.8, 12.9, 18.0, "r", ["legacysurvey"], d, log=_LOG) == "cached"
    print("test_template_cache_lookup_tolerates_pointing_offset: PASS")


def test_fetch_template_tries_sources_in_order():
    calls = []

    def fake_fetch(source, ra, dec, band, pixscale, size_px, timeout, log):
        calls.append(source)
        if source == "panstarrs":
            return np.ones((20, 20), dtype=np.float32), _wcs(pixscale=pixscale, n=20)
        return None
    saved = templates._fetch
    templates._fetch = fake_fetch
    try:
        with tempfile.TemporaryDirectory() as d:
            got = templates.fetch_template(21.8, 12.9, 10.0, "r", ["legacysurvey", "panstarrs"], d, log=_LOG)
            assert got is not None and got[2] == "panstarrs"
            assert calls == ["legacysurvey", "panstarrs"]
            assert len(list(Path(d).glob("*.fits"))) == 1, "cached"
            got2 = templates.fetch_template(21.8, 12.9, 10.0, "r", ["legacysurvey", "panstarrs"], d, log=_LOG)
            assert calls == ["legacysurvey", "panstarrs"], "second call served from the cache"
            assert got2[2] == "panstarrs"
            assert templates.fetch_template(21.8, 12.9, 10.0, "r", ["legacysurvey"], d, log=_LOG, use_cache=False) is None
    finally:
        templates._fetch = saved
    print("test_fetch_template_tries_sources_in_order: PASS")


def test_pipeline_gate_and_outputs():
    cfg = PipelineConfig()
    cfg.host_search.prefetch = False
    cfg.host_search.min_frames = 2
    cfg.host_search.rebuild_interval = 2
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        w = _wcs()
        tables = []
        for i in range(3):
            img, tab = _frame(seed=20 + i)
            _write_frame(d, f"f{i}-dft.fits", img, tab, w)
            tables.append(tab)
        # not a trigger observation: nothing happens
        plain = [Table(t) for t in tables]
        for t in plain:
            t.meta = {k: v for k, v in tables[0].meta.items() if not k.startswith("GRB")}
        assert search.maybe_run_host_search(d, plain, cfg, _LOG) is None
        # too wide a circle
        wide = [Table(t, meta=dict(tables[0].meta, GRB_ERR=1.0)) for t in tables]
        assert search.maybe_run_host_search(d, wide, cfg, _LOG) is None
        # the search itself is replaced: counts the calls, writes a table
        calls = []

        def fake_run(obs_dir, dets, c, log=None):
            calls.append(len(dets))
            rows = [{"ra": 21.8, "dec": 12.9, "mag": 17.7, "catalog": "legacysurvey", "type": "SER", "radius_arcsec": 2.0,
                     "zphot": 0.19, "n_exp": 4.4, "prior_rank": 1, "residual_flux": 500.0, "residual_err": 50.0,
                     "residual_snr": 10.0, "excess_frac": 0.05, "excess_sigma": 3.1, "residual_mag": 21.8,
                     "residual_lim5": 22.3, "galaxy_mag_meas": 18.2},
                    {"ra": 21.81, "dec": 12.91, "mag": 18.7, "catalog": "legacysurvey", "type": "EXP", "radius_arcsec": 1.0,
                     "zphot": 0.3, "n_exp": 9.3, "prior_rank": 2}]
            return search._write_outputs(obs_dir, rows, (21.8, 12.9, 180.0), "legacysurvey", len(dets), {}, c, _LOG)
        saved = search.run_host_search
        search.run_host_search = fake_run
        try:
            assert search.maybe_run_host_search(d, tables[:1], cfg, _LOG) is None, "below min_frames"
            t = search.maybe_run_host_search(d, tables[:2], cfg, _LOG)
            assert t is not None and len(t) == 2 and calls == [2]
            assert list(t["rank"]) == [1, 2] and t["residual_rank"][0] == 1 and t["residual_rank"][1] == -1
            assert (d / search.ECSV_FILENAME).exists() and (d / search.JSON_FILENAME).exists()
            payload = json.loads((d / search.JSON_FILENAME).read_text())
            assert payload["template_source"] == "legacysurvey" and payload["galaxies"][0]["excess_sigma"] == 3.1
            assert "residual_snr" not in payload["galaxies"][1], "prior-only galaxies carry no residual"
            assert search.maybe_run_host_search(d, tables[:3], cfg, _LOG) is None, "throttled until 4 frames"
            assert calls == [2]
            # a crash inside the search is a warning, not an exception
            def boom(*a, **k):
                raise RuntimeError("hotpants exploded")
            search.run_host_search = boom
            (d / search.STATE_FILENAME).unlink()
            assert search.maybe_run_host_search(d, tables[:3], cfg, _LOG) is None
        finally:
            search.run_host_search = saved
    print("test_pipeline_gate_and_outputs: PASS")


def test_trigger_circle_and_fits_lookup():
    t = Table({"a": [1]}); t.meta = {"GRB_RA": "21.8", "GRB_DEC": 12.9, "GRB_ERR": 0.05}
    assert search.trigger_circle([Table({"a": [1]}), t]) == (21.8, 12.9, 180.0)
    t.meta["GRB_ERR"] = "nan"
    assert search.trigger_circle([t]) is None
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x-dft.fits"
        p.write_bytes(b"")
        t.meta["filename"] = str(Path(d) / "x-df.ecsv")
        assert search._fits_path_for(t) == p, "D50's -dft.fits naming"
        t.meta["filename"] = str(Path(d) / "y-df.ecsv")
        assert search._fits_path_for(t) is None
    print("test_trigger_circle_and_fits_lookup: PASS")


if __name__ == "__main__":
    test_prior_ranks_brighter_first_and_counts_scale_with_area()
    test_query_falls_back_to_the_next_catalogue()
    test_frame_selection_and_zeropoint()
    test_build_stack_median_of_calibrated_frames()
    test_measure_galaxies_and_population_sigma()
    test_template_cache_lookup_tolerates_pointing_offset()
    test_fetch_template_tries_sources_in_order()
    test_pipeline_gate_and_outputs()
    test_trigger_circle_and_fits_lookup()
    print("All host_search tests passed.")
