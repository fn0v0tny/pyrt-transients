"""Catalogue positions moved to the frame's epoch, and saturated stars
left out of the candidate step (catalog.py)."""
import numpy as np
import pytest
from astropy.table import Table

from pyrt_transient.catalog import CatTransients


def _catalog(rows, epoch=2016.0):
    cat = CatTransients(Table(rows))
    cat.meta["astepoch"] = epoch
    return cat


def _tan_header(ra0=100.0, dec0=20.0, scale_arcsec=1.0, n=1024):
    s = scale_arcsec / 3600.0
    return {"NAXIS1": n, "NAXIS2": n, "CRPIX1": n / 2, "CRPIX2": n / 2,
            "CRVAL1": ra0, "CRVAL2": dec0, "CD1_1": -s, "CD1_2": 0.0, "CD2_1": 0.0, "CD2_2": s,
            "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN", "JD": 2461000.0}


def test_observation_epoch_from_header_keys():
    assert CatTransients.observation_epoch({"JD": 2451545.0}) == pytest.approx(2000.0)
    assert CatTransients.observation_epoch({"CTIME": 946728000}) == pytest.approx(2000.0)
    assert CatTransients.observation_epoch({"JD": "bad", "CTIME": 946728000 + 365.25 * 86400}) == pytest.approx(2001.0)
    assert CatTransients.observation_epoch({"DATE-OBS": "2026-09-14T00:00:00"}) == pytest.approx(2026.7, abs=0.01)
    assert CatTransients.observation_epoch({}) is None


def test_positions_move_with_proper_motion():
    # 150 mas/yr in RA*cos(dec) and -100 mas/yr in Dec, 10 years: 1.5" and -1.0".
    pm = 150 / 3.6e6
    cat = _catalog({"radeg": [100.0, 100.1], "decdeg": [60.0, 60.0],
                    "pmra": [pm, np.nan], "pmdec": [-100 / 3.6e6, np.nan]})
    ra, dec = cat.positions_at_epoch(2026.0)
    assert (ra[0] - 100.0) * 3600 * np.cos(np.radians(60.0)) == pytest.approx(1.5, abs=1e-6)
    assert (dec[0] - 60.0) * 3600 == pytest.approx(-1.0, abs=1e-6)
    assert ra[1] == 100.1 and dec[1] == 60.0  # no proper motion known: stays put
    ra, dec = cat.positions_at_epoch(None)
    assert ra[0] == 100.0 and dec[0] == 60.0
    cat.meta["astepoch"] = None
    cat.meta["catalog_props"] = {"epoch": None}
    assert cat.positions_at_epoch(2026.0)[0][0] == 100.0


def test_pixel_transform_uses_frame_epoch_and_can_be_switched_off():
    # 400 mas/yr in Dec over 10 years (2016 -> JD 2461000 is 2026.0) is 4" = 4 px here.
    cat = _catalog({"radeg": [100.0], "decdeg": [20.0], "pmra": [0.0], "pmdec": [400 / 3.6e6]})
    det = Table({"X_IMAGE": [512.0], "Y_IMAGE": [512.0]})
    det.meta.update(_tan_header())
    moved = cat._transform_catalog_to_pixel(det)
    assert moved[0, 1] - 512.0 == pytest.approx(4.0, abs=0.05)
    det.meta["propagate_proper_motion"] = False
    fixed = cat._transform_catalog_to_pixel(det)
    assert fixed[0, 1] == pytest.approx(512.0, abs=0.05)
    # The image id, which keys the cached pixel positions, changes with the epoch
    # and with the flag.
    det.meta["propagate_proper_motion"] = True
    det2 = Table(det)
    det2.meta.update(_tan_header())
    det2.meta["JD"] = 2461000.0 + 365.25
    assert cat._generate_image_id(det) != cat._generate_image_id(det2)
    det2.meta["JD"] = det.meta["JD"]
    assert cat._generate_image_id(det) == cat._generate_image_id(det2)
    det2.meta["propagate_proper_motion"] = False
    assert cat._generate_image_id(det) != cat._generate_image_id(det2)


def test_saturation_magnitude_and_mask():
    # 210619B fixture header: MAGZERO 25.612, FWHM 3.23 -> 10.98 mag.
    meta = {"MAGZERO": 25.6119707, "FWHM": 3.23}
    assert CatTransients.saturation_magnitude(meta, 60000.0) == pytest.approx(10.98, abs=0.01)
    assert CatTransients.saturation_magnitude({"FWHM": 3.0}, 60000.0) is None

    cat = _catalog({"radeg": [1.0], "decdeg": [1.0], "pmra": [0.0], "pmdec": [0.0]})
    det = Table({"NUMBER": [1, 2, 3, 0], "MAG_CALIB": [11.26, 11.43, 10.5, 9.0],
                 "FLAGS": [4, 0, 0, 0]})
    det.meta.update(meta)
    mask, limit = cat.saturated_mask(det)
    # flagged; the afterglow at 11.43 is kept; 10.5 is brighter than the estimate;
    # the forced target row is never rejected.
    assert mask.tolist() == [True, False, True, False] and limit == pytest.approx(10.98, abs=0.01)
    det.meta["saturation_margin_mag"] = 0.5
    assert cat.saturated_mask(det)[0].tolist() == [True, True, True, False]
    det.meta["saturation_margin_mag"] = 0.0
    det.meta["IS_STACK"] = True      # SEP's bit 4 is deblending overflow, not saturation
    assert cat.saturated_mask(det)[0].tolist() == [False, False, True, False]
    det.meta["reject_saturated"] = False
    assert not cat.saturated_mask(det)[0].any() and cat.saturated_mask(det)[1] is None
    assert "_saturation_mag" not in det.meta


def test_saturation_is_a_brightness_limit_not_a_rejection():
    # Two catalogue stars: a 14.5 mag one and a 9.0 mag one. The frame saturates at
    # about 10.98 mag (MAGZERO 25.6, FWHM 3.0 -> 10.96).
    cat = _catalog({"radeg": [1.0, 2.0], "decdeg": [1.0, 2.0], "pmra": [0.0, 0.0], "pmdec": [0.0, 0.0],
                    "Sloan_g": [15.0, 9.5], "Sloan_r": [14.5, 9.0]})
    cat.precompute_photometric_data()
    det = Table({"NUMBER": [1, 2, 3, 4], "X_IMAGE": [300.0, 400.0, 500.0, 600.0],
                 "Y_IMAGE": [300.0, 400.0, 500.0, 600.0],
                 "MAG_CALIB": [9.5, 15.0, 9.0, 7.5], "MAGERR_CALIB": [0.01, 0.05, 0.01, 0.01],
                 "FLAGS": [4, 0, 4, 4]})
    det.meta.update({"NAXIS1": 1024, "NAXIS2": 1024, "MAGZERO": 25.6, "FWHM": 3.0})
    none, faint, bright = np.array([], dtype=int), np.array([0]), np.array([1])
    # 1: saturated, no catalogue star -> new (a bright GRB saturates too)
    # 2: faint, no catalogue star -> new
    # 3: saturated on a 14.5 mag star: at least 10.96, so it brightened by 3.5 mag or more
    # 4: saturated on a 9.0 mag star: expected to saturate, its 7.5 says nothing
    out = cat._process_detections_for_candidates(det, [none, none, faint, bright], 1.0, 5.0, 10.0)
    assert out["NUMBER"].tolist() == [1, 2, 3]
    assert out["candidate_type"].tolist() == ["new", "new", "brightening"]
    assert out["mag_is_limit"].tolist() == [False, False, True]
    # The catalogue magnitude passes through the colour model first, hence the slack.
    assert out["magnitude_difference"][2] == pytest.approx(10.96 - 14.5, abs=0.25)
    # A saturated star can never be reported as fading, however faint it measures.
    det["MAG_CALIB"][3] = 12.0
    out = cat._process_detections_for_candidates(det, [none, none, faint, bright], 1.0, 5.0, 10.0)
    assert out["NUMBER"].tolist() == [1, 2, 3]
    # Without the rule the raw 7.5 vs 9.0 is a 1.5 mag "brightening".
    det["MAG_CALIB"][3] = 7.5
    det.meta["reject_saturated"] = False
    out = cat._process_detections_for_candidates(det, [none, none, faint, bright], 1.0, 5.0, 10.0)
    assert out["NUMBER"].tolist() == [1, 2, 3, 4] and out["candidate_type"][3] == "brightening"


def test_saturated_row_with_a_faint_neighbour_is_explained_by_the_bright_match():
    # A 9.0 mag star saturates; a 15 mag Gaia neighbour inside the identification
    # radius must not turn it into a "brightening" row (the largest |diff| used to win).
    cat = _catalog({"radeg": [1.0, 1.0005], "decdeg": [1.0, 1.0], "pmra": [0.0, 0.0], "pmdec": [0.0, 0.0],
                    "Sloan_g": [9.5, 15.5], "Sloan_r": [9.0, 15.0]})
    cat.precompute_photometric_data()
    det = Table({"NUMBER": [1], "X_IMAGE": [300.0], "Y_IMAGE": [300.0],
                 "MAG_CALIB": [9.0], "MAGERR_CALIB": [0.01], "FLAGS": [4]})
    det.meta.update({"NAXIS1": 1024, "NAXIS2": 1024, "MAGZERO": 25.6, "FWHM": 3.0})
    out = cat._process_detections_for_candidates(det, [np.array([1, 0])], 1.0, 5.0, 10.0)
    assert len(out) == 0


def test_flagged_faint_star_keeps_its_own_magnitude_as_the_limit():
    # A 17.0 mag detection carrying the flag (deblended child of a saturated
    # neighbour) on a 17.0 mag star: the limit is max(frame limit, measured), so
    # nothing is reported. A low-SNR flagged row is gated like any other.
    cat = _catalog({"radeg": [1.0], "decdeg": [1.0], "pmra": [0.0], "pmdec": [0.0],
                    "Sloan_g": [17.5], "Sloan_r": [17.0]})
    cat.precompute_photometric_data()
    det = Table({"NUMBER": [1, 2], "X_IMAGE": [300.0, 400.0], "Y_IMAGE": [300.0, 400.0],
                 "MAG_CALIB": [17.0, 17.0], "MAGERR_CALIB": [0.05, 0.5], "FLAGS": [4, 4]})
    det.meta.update({"NAXIS1": 1024, "NAXIS2": 1024, "MAGZERO": 25.6, "FWHM": 3.0})
    out = cat._process_detections_for_candidates(det, [np.array([0]), np.array([0])], 1.0, 5.0, 10.0)
    assert len(out) == 0


def test_usno_proper_motions_are_not_used_and_huge_ones_are_capped():
    pm = 150 / 3.6e6
    rows = {"radeg": [100.0, 100.1], "decdeg": [20.0, 20.0], "pmra": [pm, 30000 / 3.6e6], "pmdec": [0.0, 0.0]}
    gaia = _catalog(rows)
    gaia.meta["catalog"] = "gaia"
    ra, _ = gaia.positions_at_epoch(2026.0)
    assert (ra[0] - 100.0) * 3600 * np.cos(np.radians(20.0)) == pytest.approx(1.5, abs=1e-6)
    assert ra[1] == 100.1                       # 30 arcsec/yr is not a real star
    usno = _catalog(rows, epoch=2000.0)
    usno.meta["catalog"] = "usno"
    assert not usno.proper_motions_trusted()
    ra, _ = usno.positions_at_epoch(2026.0)
    assert ra.tolist() == [100.0, 100.1]


def test_cached_pixel_positions_honour_the_propagation_flag():
    cat = _catalog({"radeg": [100.0], "decdeg": [20.0], "pmra": [0.0], "pmdec": [400 / 3.6e6]})
    cat.precompute_photometric_data()
    det = Table({"X_IMAGE": [512.0], "Y_IMAGE": [512.0]})
    det.meta.update(_tan_header())
    moved = cat.get_pixel_coordinates_cached(det)
    assert moved[0, 1] - 512.0 == pytest.approx(4.0, abs=0.05)
    det.meta["propagate_proper_motion"] = False
    fixed = cat.get_pixel_coordinates_cached(det)   # a different cache key, not the stale entry
    assert fixed[0, 1] == pytest.approx(512.0, abs=0.05)
