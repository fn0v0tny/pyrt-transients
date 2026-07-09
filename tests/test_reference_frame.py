"""Fresh unit tests for detection/reference_frame.py (rewrite.md Phase 7).

No check_baseline.py coverage possible here -- ReferenceFrameSelector's
predecessor (ImageExtractionManager's reference-frame methods) was never
callable before (self.reference_idx was never set), so there's no existing
behavior to compare against. These tests validate the fix directly, against
synthetic WCS headers.

Run with: python3 -m pytest tests/test_reference_frame.py -v
"""
import sys
from pathlib import Path

import numpy as np
from astropy.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pyrt_transient.detection.reference_frame import ReferenceFrameSelector, ImageQuality


def _synthetic_detection_table(crval1, crval2, n_rows=5, fwhm=2.5, naxis=1024,
                                pixscale_deg=1.0 / 3600.0):
    """A minimal detection table with a simple tangent-projection WCS header
    and n_rows fake point sources, centered on (crval1, crval2).
    """
    meta = {
        'CRVAL1': crval1, 'CRVAL2': crval2,
        'CRPIX1': naxis / 2.0, 'CRPIX2': naxis / 2.0,
        'CDELT1': -pixscale_deg, 'CDELT2': pixscale_deg,
        'CTYPE1': 'RA---TAN', 'CTYPE2': 'DEC--TAN',
        'CTRRA': crval1, 'CTRDEC': crval2,
        'NAXIS1': naxis, 'NAXIS2': naxis,
        'FWHM': fwhm,
        'FITSFILE': f'synthetic_{crval1}_{crval2}.fits',
    }
    rng = np.random.default_rng(0)
    t = Table({
        'X_IMAGE': rng.uniform(100, naxis - 100, n_rows),
        'Y_IMAGE': rng.uniform(100, naxis - 100, n_rows),
        'MAG_AUTO': rng.uniform(14, 18, n_rows),
        'MAGERR_AUTO': np.full(n_rows, 0.05),
    })
    t.meta = meta
    return t


def test_compute_field_center_is_median():
    tables = [
        _synthetic_detection_table(180.0, 30.0),
        _synthetic_detection_table(180.002, 30.001),
        _synthetic_detection_table(179.998, 29.999),
    ]
    sel = ReferenceFrameSelector(tables)
    ra, dec = sel.field_center
    assert abs(ra - 180.0) < 1e-9
    assert abs(dec - 30.0) < 1e-9


def test_select_reference_image_picks_better_seeing():
    # Same n_sources/center_dist/limiting_mag contribution roughly equal;
    # image 1 has much better (smaller) FWHM -> higher (1/seeing)*0.4 term ->
    # should be selected.
    tables = [
        _synthetic_detection_table(180.0, 30.0, fwhm=5.0),
        _synthetic_detection_table(180.0, 30.0, fwhm=1.5),
    ]
    sel = ReferenceFrameSelector(tables)
    assert sel.reference_idx == 1


def test_reference_idx_actually_set_bug_fix():
    # The whole point of Phase 7: this must not raise/be None.
    tables = [_synthetic_detection_table(180.0, 30.0)]
    sel = ReferenceFrameSelector(tables)
    assert sel.reference_idx is not None
    assert isinstance(sel.reference_idx, int)


def test_transform_to_reference_uses_the_selected_images_wcs():
    # Two images at different pointings; force image 1 to be selected via
    # much better seeing, then confirm transform_to_reference uses image 1's
    # WCS specifically -- a candidate placed exactly at image 1's CRVAL
    # must land near image 1's CRPIX, not image 0's.
    tables = [
        _synthetic_detection_table(180.0, 30.0, fwhm=5.0),
        _synthetic_detection_table(181.0, 31.0, fwhm=1.5),
    ]
    sel = ReferenceFrameSelector(tables)
    assert sel.reference_idx == 1

    candidates = Table({'ALPHA_J2000': [181.0], 'DELTA_J2000': [31.0]})
    result = sel.transform_to_reference(candidates)

    assert 'X_REF' in result.colnames and 'Y_REF' in result.colnames
    # Should land at image 1's CRPIX (512, 512), not image 0's unrelated WCS.
    assert abs(result['X_REF'][0] - 512.0) < 0.5
    assert abs(result['Y_REF'][0] - 512.0) < 0.5
    assert result.meta['reference_idx'] == 1
    assert result.meta['reference_image'] == tables[1].meta['FITSFILE']


def test_validate_reference_coordinates_filters_out_of_bounds():
    tables = [_synthetic_detection_table(180.0, 30.0, naxis=1024)]
    sel = ReferenceFrameSelector(tables)

    candidates = Table({
        'X_REF': [500.0, -20.0, 1040.0, 1030.0],   # last one within margin=10+
        'Y_REF': [500.0, 500.0, 500.0, 500.0],
    })
    result = sel.validate_reference_coordinates(candidates, margin=10.0)
    # in-bounds (500,500): keep. (-20,*): reject (< -10 margin). (1040,*):
    # reject (> 1024+10=1034). (1030,*): keep (<= 1034).
    assert len(result) == 2
    assert list(result['X_REF']) == [500.0, 1030.0]


def test_get_detection_matches_respects_radius():
    tables = [_synthetic_detection_table(180.0, 30.0, n_rows=2)]
    sel = ReferenceFrameSelector(tables)

    # Place two known detections directly in the (only) image's detection table.
    tables[0]['X_IMAGE'] = [100.0, 100.0]
    tables[0]['Y_IMAGE'] = [100.0, 200.0]  # 100 px away from the first

    # A candidate at the same sky position as X_IMAGE=100,Y_IMAGE=100.
    wcs_ra, wcs_dec = _pix_to_sky(tables[0], 100.0, 100.0)
    candidates = Table({'ALPHA_J2000': [wcs_ra], 'DELTA_J2000': [wcs_dec]})

    matches_tight = sel.get_detection_matches(candidates, match_radius_px=5.0)
    assert matches_tight[0][0] == [0]  # only the close one matches

    matches_wide = sel.get_detection_matches(candidates, match_radius_px=150.0)
    assert sorted(matches_wide[0][0]) == [0, 1]  # both match now


def _pix_to_sky(det_table, x, y):
    import astropy.wcs
    wcs = astropy.wcs.WCS(det_table.meta)
    ra, dec = wcs.all_pix2world([x], [y], 1)
    return float(ra[0]), float(dec[0])


if __name__ == "__main__":
    test_compute_field_center_is_median()
    test_select_reference_image_picks_better_seeing()
    test_reference_idx_actually_set_bug_fix()
    test_transform_to_reference_uses_the_selected_images_wcs()
    test_validate_reference_coordinates_filters_out_of_bounds()
    test_get_detection_matches_respects_radius()
    print("All detection/reference_frame.py tests passed.")
