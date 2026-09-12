"""Reading pyrt's per-frame ECSV detection tables."""

import logging
import os
from typing import Optional

import astropy.table
import numpy as np


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
    return drop_insignificant_forced_row(det, fn if verbose else None)


# pyrt writes one NUMBER==0 row per frame: the forced measurement at the
# target position. It is a measurement of wherever the telescope pointed,
# not a detection, and it repeats at the same place in every frame, so the
# detection side turns it into a persistent "new" source. On the GRB replay
# it scored Q 7-20 in every epoch (GRB230818A, GRB220403B) at a median SNR
# of 1.1-1.9; on D50's obs_104223 the same row carries no magnitude at all.
#
# Kept when it is a real measurement (magnitude present and significant),
# because then it is the target actually being seen; dropped otherwise. The
# cut is SNR 2 (the user's choice): it errs towards keeping a target that was
# genuinely measured, at the cost of a weak candidate surviving in the
# minority of frames where the noise happened to be measured that well.
FORCED_ROW_MIN_SNR = float(os.environ.get("PYRT_FORCED_ROW_MIN_SNR", "2"))


def drop_insignificant_forced_row(det, fn=None):
    """Remove the forced NUMBER==0 row unless it is a significant detection."""
    if det is None or "NUMBER" not in det.colnames or not len(det):
        return det
    try:
        numbers = np.asarray(det["NUMBER"], dtype=float)
    except (TypeError, ValueError):
        return det
    forced = numbers == 0
    if not forced.any():
        return det

    magerr = None
    for column in ("MAGERR_CALIB", "MAGERR_AUTO", "MAGERR_SEX", "MAGERR_ISO"):
        if column in det.colnames:
            magerr = np.asarray(np.ma.filled(det[column], np.nan), dtype=float)
            break
    mag = None
    for column in ("MAG_CALIB", "MAG_AUTO", "MAG_SEX", "MAG_ISO"):
        if column in det.colnames:
            mag = np.asarray(np.ma.filled(det[column], np.nan), dtype=float)
            break

    significant = np.zeros(len(det), dtype=bool)
    if mag is not None and magerr is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            snr = 1.0857 / magerr
        significant = np.isfinite(mag) & np.isfinite(magerr) & (snr >= FORCED_ROW_MIN_SNR)

    remove = forced & ~significant
    if remove.any():
        if fn:
            logging.info(f"{os.path.basename(fn)}: dropped the forced NUMBER=0 row "
                         f"(no significant measurement at the target)")
        return det[~remove]
    return det
