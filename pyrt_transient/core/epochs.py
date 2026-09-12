"""Epoch-detection preparation, generalized (no `self`) so a GRB replay
driver can reuse it for slicing epochs 1..k.
"""

import re
from typing import List

import numpy as np
from astropy.table import Table

from pyrt_transient.core.timeutil import unix_to_mjd

# A frame name like 20260911220031-484-i-005-df.ecsv: sequence, band, exposure.
_BAND_IN_NAME = re.compile(r"-(?:\d+-)?([ugrizUBVRIN])-\d+-", re.ASCII)
# Values that mean "no band", including what an empty column reads back as.
_NOT_A_BAND = {"", "0", "--", "nan", "none", "None", "N/A"}


def _clean(value) -> str:
    value = "" if value is None else str(value).strip()
    return "" if value in _NOT_A_BAND else value


def band_label(observed, calibrated) -> str:
    """How a frame's band is shown: the filter it was taken through, and the
    band its photometry was calibrated against when that differs.

    An unfiltered frame calibrated against Sloan r ("N" -> "Sloan_r") is not
    the same measurement as a real Sloan r frame -- the colour term differs --
    so it is labelled `N->Sloan_r` and never merged with `Sloan_r`.
    """
    observed, calibrated = _clean(observed), _clean(calibrated)
    if not observed:
        return calibrated
    if not calibrated or calibrated == observed or calibrated.split("_")[-1] == observed:
        return calibrated or observed
    return f"{observed}→{calibrated}"


# One colour per band, shared by the lightcurve PNG and the web page so the
# two views of the same data agree. Anything else falls back to grey-blue.
_BAND_COLOURS = {
    "Sloan_g": "#27ae60", "Sloan_r": "#e74c3c", "Sloan_i": "#8e44ad", "Sloan_z": "#d35400",
    "Sloan_u": "#2471a3", "Johnson_B": "#2980b9", "Johnson_V": "#16a085",
    "Johnson_R": "#c0392b", "Johnson_I": "#7f3f00", "N": "#566573",
    "N→Sloan_r": "#ec7063", "N→Sloan_g": "#52be80", "N→Sloan_i": "#af7ac5",
}


def band_colour(band) -> str:
    return _BAND_COLOURS.get(str(band), "#3498db")


def bands_of(lightcurve) -> np.ndarray:
    """The band label of every row, as strings.

    Uses the `filter` and `phot_filter` columns prepare_epoch_detections
    writes. Lightcurves built before those existed keep the band only in
    `source_file`, so the filter is read back from the frame name (D50's
    `...-i-005-df.ecsv`); the calibration band is unknown there. Rows whose
    band cannot be told are "".
    """
    if "filter" in lightcurve.colnames:
        observed = np.asarray(lightcurve["filter"], dtype=str)
        calibrated = (np.asarray(lightcurve["phot_filter"], dtype=str)
                      if "phot_filter" in lightcurve.colnames else np.full(len(observed), ""))
        return np.array([band_label(o, c) for o, c in zip(observed, calibrated)], dtype=str)
    if "source_file" not in lightcurve.colnames:
        return np.full(len(lightcurve), "")
    names = np.asarray(lightcurve["source_file"], dtype=str)
    found = [_BAND_IN_NAME.search(str(name).split("/")[-1]) for name in names]
    return np.array([m.group(1) if m else "" for m in found], dtype=str)


def prepare_epoch_detections(detection_tables: List[Table]) -> List[Table]:
    """Prepare epoch detection data with timing information."""
    all_epoch_detections = []

    for i, det_table in enumerate(detection_tables):
        # Extract timing information
        ctime = det_table.meta.get('CTIME', 0)
        exptime = det_table.meta.get('EXPTIME', 0)
        mid_time = ctime + exptime / 2.0

        # Add epoch information to detection table
        det_table_copy = det_table.copy()
        det_table_copy['epoch_id'] = i
        det_table_copy['obs_time'] = mid_time
        det_table_copy['mjd'] = unix_to_mjd(mid_time)
        det_table_copy['source_file'] = det_table.meta.get('filename', f'epoch_{i}')
        # The band of the frame, kept as two facts: the filter it was taken
        # through, and the band its photometry was calibrated against. They
        # live in the epoch's meta, so without these columns a lightcurve
        # built from several epochs loses them and mixes bands into one
        # series: D50 cycles g/r/i/z within a single observation (obs_104223:
        # 17 i, 12 r, 10 g, 7 z frames). Fixed-width strings, because an empty
        # one reads back from ECSV as "0".
        n = len(det_table_copy)
        det_table_copy['filter'] = np.full(n, _clean(det_table.meta.get('FILTER')), dtype='U32')
        det_table_copy['phot_filter'] = np.full(n, _clean(det_table.meta.get('PHFILTER')), dtype='U32')

        all_epoch_detections.append(det_table_copy)

    return all_epoch_detections
