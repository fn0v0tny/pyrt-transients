"""Epoch-detection preparation, generalized (no `self`) so a GRB replay
driver can reuse it for slicing epochs 1..k.
"""

from typing import List

from astropy.table import Table

from pyrt_transient.core.timeutil import unix_to_mjd


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
