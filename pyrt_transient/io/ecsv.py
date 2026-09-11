"""Reading pyrt's per-frame ECSV detection tables."""

import logging
import os
from typing import Optional

import astropy.table


def open_ecsv_file(arg, verbose=True) -> Optional[astropy.table.Table]:
    """Open the `.ecsv` detection table for `arg` (given either as the
    `.ecsv` path or as the matching `.fits`/`.cat` path), or None if it
    can't be read.

    `meta['filename']` is set to the path actually opened -- every consumer
    that needs the matching FITS derives it from this (e.g.
    detection/stacking.py), so it must be the local copy, not whatever
    `filename` the producing host wrote into the header.
    """
    fn = f"{os.path.splitext(str(arg))[0]}.ecsv"
    try:
        det = astropy.table.Table.read(fn, format="ascii.ecsv")
    except Exception as exc:
        if verbose:
            logging.warning(f"{fn} did not open as an ecsv table: {exc}")
        return None
    det.meta["filename"] = fn
    return det
