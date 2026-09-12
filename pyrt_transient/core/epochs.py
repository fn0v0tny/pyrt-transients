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


def bands_of(lightcurve) -> np.ndarray:
    """The photometric band of every row, as strings.

    Uses the `filter` column that prepare_epoch_detections writes. Lightcurves
    built before that column existed keep the band only in `source_file`, so
    it is read back from the frame name (D50's `...-i-005-df.ecsv`) rather
    than reprocessing the observation. Rows whose band cannot be told are "".
    """
    if "filter" in lightcurve.colnames:
        return np.asarray(lightcurve["filter"], dtype=str)
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
        # The photometric band of the frame. It lives in the epoch's meta, so
        # without this column a lightcurve built from several epochs loses it
        # and mixes bands into one series: D50 cycles g/r/i/z within a single
        # observation (obs_104223: 17 i, 12 r, 10 g, 7 z frames).
        det_table_copy['filter'] = str(
            det_table.meta.get('PHFILTER') or det_table.meta.get('FILTER') or '')

        all_epoch_detections.append(det_table_copy)

    return all_epoch_detections
