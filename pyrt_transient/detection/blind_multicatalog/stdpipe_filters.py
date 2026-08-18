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

import logging
import numpy as np
from astropy.table import Table

import stdpipe.pipeline as stdpipe_pipeline

logger = logging.getLogger("detection.blind_multicatalog.stdpipe_filters")


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

    stdpipe.catalogs.xmatch_skybot (as installed here) crashes with
    `KeyError: 'RA'` whenever SkyBoT genuinely finds zero solar-system
    objects for a field/time -- verified directly against a real
    GRB210410A epoch, not hypothetical: `Skybot.cone_search` doesn't raise
    in that case (only warns `NoResultsWarning`), it returns a table with
    no 'RA'/'DEC' columns, and the very next line indexes into it
    unguarded. A field with no currently-known asteroids is an entirely
    ordinary result, not an error -- so a KeyError here is treated the same
    as "SkyBoT found nothing to reject" rather than propagated, matching
    how every other external-service failure in this codebase degrades
    (missing swarp/hotpants binaries, PS1 skycell bugs, etc.) rather than
    crashing the whole epoch.
    """
    if len(candidates) == 0 or time is None:
        return candidates, 0

    obj = _to_stdpipe_columns(candidates)
    try:
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
    except KeyError as e:
        logger.warning(f"SkyBoT cross-match failed ({e!r}), treating as no matches "
                        f"(a known stdpipe bug on genuinely empty SkyBoT results)")
        return candidates, 0
    n_removed = int(np.sum(~mask))
    return candidates[mask], n_removed
