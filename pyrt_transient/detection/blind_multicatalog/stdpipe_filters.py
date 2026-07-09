"""VSX and SkyBoT rejection filters, split into two functions because they
have opposite time-sensitivity:

- VSX is purely positional -- a variable star's position never changes, so
  it should run once on the final clustered candidate list (called from
  clustering.py's _combine_with_lightcurves), not once per catalog per
  epoch on raw candidates.
- SkyBoT is time-dependent (solar system objects move) and needs a real
  per-epoch timestamp, so it can't defer to the final-candidate stage the
  same way -- but it doesn't need per-catalog redundancy either. It runs
  once per epoch, on that epoch's catalogs already vstacked together
  (called from clustering.py's combine_results).

NED cross-match is deliberately not implemented here as a removal filter --
a galaxy-coincident candidate can be a real transient (e.g. a supernova in
its host); that's an enrichment/annotation concern, not a rejection filter.
"""

import numpy as np
from astropy.table import Table

import stdpipe.pipeline as stdpipe_pipeline


def _to_stdpipe_columns(candidates):
    """Build the lowercase ra/dec/flags/magerr/fwhm table stdpipe.pipeline
    expects, from our uppercase SExtractor-style column names."""
    obj = Table()
    obj['ra'] = np.array(candidates['ALPHA_J2000'], dtype=float)
    obj['dec'] = np.array(candidates['DELTA_J2000'], dtype=float)
    obj['flags'] = (
        np.array(candidates['FLAGS'], dtype=int) if 'FLAGS' in candidates.colnames
        else np.zeros(len(candidates), dtype=int)
    )
    obj['magerr'] = (
        np.array(candidates['MAGERR_CALIB'], dtype=float) if 'MAGERR_CALIB' in candidates.colnames
        else np.zeros(len(candidates))
    )
    obj['fwhm'] = (
        np.array(candidates['FWHM_IMAGE'], dtype=float) if 'FWHM_IMAGE' in candidates.colnames
        else np.full(len(candidates), 3.0)
    )
    return obj


def apply_vsx_filter(candidates, match_radius_arcsec=2.5):
    """Reject candidates matching known VSX variable stars. Purely
    positional -- call once on the final clustered candidate list.
    Returns (filtered_candidates, n_removed).
    """
    if len(candidates) == 0:
        return candidates, 0

    obj = _to_stdpipe_columns(candidates)
    mask = stdpipe_pipeline.filter_transient_candidates(
        obj,
        sr=match_radius_arcsec / 3600.0,
        vizier=['vsx'],
        skybot=False,
        ned=False,
        # Flag-based hard rejection is not part of current behavior (FLAGS
        # only soft-downweights quality_score today) -- don't silently add it.
        flagged=False,
        get_candidates=False,
    )
    n_removed = int(np.sum(~mask))
    return candidates[mask], n_removed


def apply_skybot_filter(candidates, time, match_radius_arcsec=2.5):
    """Reject candidates matching known SkyBoT solar-system objects at the
    given observation time. Time-dependent -- call once per epoch, on that
    epoch's catalogs already combined. Returns (filtered_candidates, n_removed).
    """
    if len(candidates) == 0 or time is None:
        return candidates, 0

    obj = _to_stdpipe_columns(candidates)
    mask = stdpipe_pipeline.filter_transient_candidates(
        obj,
        sr=match_radius_arcsec / 3600.0,
        vizier=[],
        skybot=True,
        ned=False,
        flagged=False,
        time=time,
        get_candidates=False,
    )
    n_removed = int(np.sum(~mask))
    return candidates[mask], n_removed
