"""Radius-based coordinate matching, dispatching to stdpipe.astrometry.

Thin adapter over stdpipe.astrometry.spherical_match / planar_match. Do not
reimplement matching math here; this module's job is only to present one
call signature to what were pyrt_transient's four independent hand-rolled
KDTree/chord-length implementations (catalog.py, transient_analyser.py x2,
extraction_manager.py) -- callers are adapted to this shape, not the other
way around.
"""

import numpy as np
import stdpipe.astrometry as stdpipe_astrometry


def match_radius(coords_a, coords_b, radius_arcsec, coord_system="sky"):
    """Find all pairs within radius_arcsec between coords_a and coords_b.

    Parameters
    ----------
    coords_a, coords_b : (N, 2) array_like
        (ra, dec) in degrees for coord_system="sky", or (x, y) in pixels for
        coord_system="pixel".
    radius_arcsec : float
        Match radius in arcseconds for coord_system="sky" (converted to
        degrees internally). For coord_system="pixel", used directly as a
        pixel radius -- there is no arcsec-to-pixel conversion here, since
        plate scale isn't known at this layer.
    coord_system : "sky" or "pixel"

    Returns
    -------
    idx_a, idx_b : ndarray of int
        Indices into coords_a / coords_b for matched pairs.
    dist : ndarray of float
        Pairwise distance for each match -- degrees for "sky", same units as
        the input coordinates for "pixel".
    """
    coords_a = np.asarray(coords_a)
    coords_b = np.asarray(coords_b)

    if coord_system == "sky":
        idx_a, idx_b, dist = stdpipe_astrometry.spherical_match(
            coords_a[:, 0], coords_a[:, 1],
            coords_b[:, 0], coords_b[:, 1],
            sr=radius_arcsec / 3600.0,
        )
    elif coord_system == "pixel":
        idx_a, idx_b, dist = stdpipe_astrometry.planar_match(
            coords_a[:, 0], coords_a[:, 1],
            coords_b[:, 0], coords_b[:, 1],
            sr=radius_arcsec,
        )
    else:
        raise ValueError(f"coord_system must be 'sky' or 'pixel', got {coord_system!r}")

    return idx_a, idx_b, dist
