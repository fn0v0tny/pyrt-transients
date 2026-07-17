#!/usr/bin/python3
"""
pipeline_magic_sn.py — Single-image supernova search pipeline.

Designed for deep/combined images where variability information is not available.
Key differences from pipeline_magic.py:
  - Single ecsv + fits pair (no multi-epoch accumulation, no locks, no metadata)
  - min_n_detections = 1 (single epoch, no lightcurves)
  - min_catalogs_fraction = 0.34 by default (any 1 of 3 catalogs is enough)
  - SN-specific post-processing pipeline:
      1. Morphology filter      — reject elongated sources (cosmic rays, streaks)
      2. Magnitude plausibility — reject saturated stars and noise spikes
      3. SkyBot asteroid rejection — IMCCE ephemeris service for known SSOs
      4. Galaxy proximity scoring — HyperLEDA via VizieR; flag nuclear AGN
      5. Composite SN score     — ranks candidates by SN likelihood

Usage:
    pipeline_magic_sn.py <ecsv_file> <fits_file> \\
        [--output-dir=<path>] [--config=<file>] \\
        [--min-catalogs=<fraction>] [--generate-frontend] [--debug]
"""

import os
import sys
import json
import logging
import warnings
import numpy as np
from pathlib import Path
from datetime import datetime
import re
import yaml
from shutil import copy as shcopy

from pyrt_transient.catalog import QueryParams, setup_catalog_cache
from pyrt_transient.transients import open_ecsv_file
from pyrt_transient.config_trans import PipelineConfig
from pyrt_transient.core.config_loader import load_config_with_yaml_support
from pyrt_transient.io.logging_setup import setup_pipeline_logging
from pyrt_transient.io.observation_store import ObservationStore, extract_observation_id
from pyrt_transient.web.orchestration import generate_frontend
from pyrt_transient.detection.blind_multicatalog import BlindMulticatalogStrategy
from pyrt_transient.detection.subtraction import SubtractionStrategy
from pyrt_transient.detection.subtraction.candidates import (
    load_diff_table, find_science_sibling, derive_observation_id,
)
from pyrt_transient.detection.subtraction.artifact_filters import (
    apply_morphology_filter,
    apply_magnitude_filter,
    reject_dipole_artifacts,
)

# Suppress noisy FITS header warnings
try:
    from astropy.io.fits.verify import VerifyWarning
    warnings.filterwarnings("ignore", category=VerifyWarning)
except Exception:
    pass
try:
    from astropy.utils.exceptions import AstropyWarning
    warnings.filterwarnings("ignore", category=AstropyWarning)
except Exception:
    pass


# ---------------------------------------------------------------------------
# SN-specific filter / enrichment functions
#
# apply_morphology_filter/apply_magnitude_filter moved to
# detection/subtraction/artifact_filters.py (imported above) -- they operate
# on generic ELLIPTICITY/fwhm_ratio/MAG_CALIB columns, equally valid for
# candidates from either detection strategy, so there's no reason for two
# copies. Everything below is strategy-agnostic post-processing that applies
# to whichever strategy produced `candidates`.
# ---------------------------------------------------------------------------


def reject_known_asteroids(candidates, obs_jd, field_ra, field_dec, field_deg,
                            logger, match_radius_arcsec=5.0):
    """Remove candidates that match known solar system objects via IMCCE SkyBot.

    Single-epoch images cannot distinguish slow-moving asteroids from static
    transients by position shift.  SkyBot provides predicted positions of all
    catalogued solar system objects at a given epoch so we can remove them
    before ranking.

    Returns (filtered_candidates, n_rejected).
    """
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy.time import Time

    if len(candidates) == 0:
        return candidates, 0

    if obs_jd is None:
        logger.warning("  SkyBot: no observation epoch — skipping asteroid rejection")
        return candidates, 0

    try:
        from astroquery.imcce import Skybot

        logger.info(f"  SkyBot: querying at JD={obs_jd:.5f} ...")
        center = SkyCoord(field_ra, field_dec, unit='deg')
        epoch = Time(obs_jd, format='jd')
        search_radius = (field_deg / 2.0) * u.deg

        sso_table = Skybot.cone_search(center, search_radius, epoch)

        if sso_table is None or len(sso_table) == 0:
            logger.info("  SkyBot: no known solar system objects in field")
            return candidates, 0

        logger.info(f"  SkyBot: {len(sso_table)} known SSOs found")

        # SkyBot returns RA/Dec as sexagesimal strings
        try:
            sso_coords = SkyCoord(sso_table['RA'], sso_table['DEC'],
                                  unit=('hourangle', 'deg'))
        except Exception:
            # Fallback: try decimal degrees
            sso_coords = SkyCoord(
                np.array(sso_table['RA'], dtype=float),
                np.array(sso_table['DEC'], dtype=float),
                unit='deg'
            )

        cand_coords = SkyCoord(
            np.array(candidates['ALPHA_J2000'], dtype=float),
            np.array(candidates['DELTA_J2000'], dtype=float),
            unit='deg'
        )

        keep_mask = np.ones(len(candidates), dtype=bool)
        for i, cc in enumerate(cand_coords):
            seps = cc.separation(sso_coords).arcsec
            if np.min(seps) <= match_radius_arcsec:
                keep_mask[i] = False

        n_rejected = int(np.sum(~keep_mask))
        if n_rejected:
            logger.info(f"  SkyBot: removed {n_rejected} SSO matches")

        return candidates[keep_mask], n_rejected

    except ImportError:
        logger.warning("  SkyBot: astroquery.imcce not available — skipping")
        return candidates, 0
    except Exception as e:
        logger.warning(f"  SkyBot: query failed ({e}) — skipping")
        return candidates, 0


def score_galaxy_proximity(candidates, ra_center, dec_center, search_radius_deg,
                            logger, host_radius_arcsec=30.0,
                            nuclear_radius_arcsec=2.0, cache_path=None):
    """Query HyperLEDA for nearby galaxies and annotate each candidate.

    Adds three new columns:
      galaxy_sep_arcsec  — angular separation to nearest known galaxy (arcsec)
      galaxy_name        — PGC name of nearest galaxy
      galaxy_flag        — 'in_galaxy' | 'nuclear' | 'isolated'

    'nuclear' candidates (< nuclear_radius_arcsec from galaxy centre) are more
    likely AGN/TDE than SN and are penalised in the SN score.

    The column information is used later by compute_sn_score but the candidates
    are NOT filtered here — the decision is left to the scorer.

    cache_path: if given, the HyperLEDA query result (a field-level property
    with no time dependence at all -- unlike reject_high_pm_stars, there's no
    per-call propagation step here either) is cached there and reused on
    later calls instead of re-querying VizieR. Matters the same way it does
    for reject_high_pm_stars: a daemon-style caller invoking this once per
    new epoch across a multi-night campaign would otherwise repeat the same
    HyperLEDA query, with the same result, on every single call.
    """
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    n = len(candidates)
    # Default values
    gal_sep = np.full(n, np.nan)
    gal_name = [''] * n
    gal_flag = ['isolated'] * n

    if n == 0:
        candidates['galaxy_sep_arcsec'] = gal_sep
        candidates['galaxy_name'] = gal_name
        candidates['galaxy_flag'] = gal_flag
        return candidates

    try:
        from astropy.table import Table as _Table

        gal_table = None
        cache_hit = cache_path is not None and Path(cache_path).exists()
        if cache_hit:
            gal_table = _Table.read(str(cache_path), format="ascii.ecsv")
            logger.info(f"  Galaxy: using cached HyperLEDA galaxies ({len(gal_table)}) "
                        f"from {cache_path}")
            if len(gal_table) == 0:
                gal_table = None  # cached "no galaxies" result

        if not cache_hit:
            from astroquery.vizier import Vizier

            logger.info(f"  Galaxy: querying HyperLEDA within {search_radius_deg:.2f} deg ...")
            v = Vizier(columns=['PGC', 'objname', 'RAJ2000', 'DEJ2000'], row_limit=-1)
            result = v.query_region(
                SkyCoord(ra_center, dec_center, unit='deg'),
                radius=search_radius_deg * u.deg,
                catalog='VII/237/pgc'
            )
            gal_table = result[0] if (result and len(result) > 0) else None
            if cache_path is not None:
                try:
                    # Cache even an empty result (as a zero-row table with the
                    # expected columns) so a genuinely galaxy-free field
                    # doesn't get re-queried every call either.
                    to_cache = gal_table if gal_table is not None else _Table(
                        names=['PGC', 'objname', 'RAJ2000', 'DEJ2000'],
                        dtype=['i8', 'U32', 'f8', 'f8'],
                    )
                    to_cache.write(str(cache_path), format="ascii.ecsv", overwrite=True)
                except Exception as e:
                    logger.debug(f"  Galaxy: could not write cache {cache_path}: {e}")

        if gal_table is None:
            logger.info("  Galaxy: no HyperLEDA galaxies found in this field")
        else:
            logger.info(f"  Galaxy: {len(gal_table)} HyperLEDA galaxies in field")

            # HyperLEDA can return RA as sexagesimal strings ("HH MM SS.s")
            # or as decimal degrees depending on the VizieR column format.
            try:
                gal_coords = SkyCoord(
                    np.array(gal_table['RAJ2000'], dtype=float),
                    np.array(gal_table['DEJ2000'], dtype=float),
                    unit='deg'
                )
            except (ValueError, TypeError):
                gal_coords = SkyCoord(
                    gal_table['RAJ2000'],
                    gal_table['DEJ2000'],
                    unit=('hourangle', 'deg')
                )

            for i in range(n):
                cc = SkyCoord(
                    float(candidates['ALPHA_J2000'][i]),
                    float(candidates['DELTA_J2000'][i]),
                    unit='deg'
                )
                seps = cc.separation(gal_coords).arcsec
                idx = int(np.argmin(seps))
                sep = float(seps[idx])
                gal_sep[i] = sep

                try:
                    name = str(gal_table['objname'][idx]).strip()
                    gal_name[i] = name if name else f"PGC{gal_table['PGC'][idx]}"
                except Exception:
                    gal_name[i] = ''

                if sep <= nuclear_radius_arcsec:
                    gal_flag[i] = 'nuclear'
                elif sep <= host_radius_arcsec:
                    gal_flag[i] = 'in_galaxy'
                # else remains 'isolated'

            n_in = sum(f == 'in_galaxy' for f in gal_flag)
            n_nuc = sum(f == 'nuclear' for f in gal_flag)
            logger.info(f"  Galaxy: {n_in} in-galaxy, {n_nuc} nuclear, "
                        f"{n - n_in - n_nuc} isolated candidates")

    except ImportError:
        logger.warning("  Galaxy: astroquery not available — skipping HyperLEDA query")
    except Exception as e:
        logger.warning(f"  Galaxy: query failed ({e}) — skipping")

    candidates['galaxy_sep_arcsec'] = gal_sep
    candidates['galaxy_name'] = gal_name
    candidates['galaxy_flag'] = gal_flag
    return candidates


def reject_high_pm_stars(candidates, ra_center, dec_center, field_deg, obs_jd,
                          logger, pm_threshold_masyr=20.0, match_radius_arcsec=5.0,
                          cache_path=None):
    """Remove candidates that are high proper-motion (PM) stars caught at their
    current (propagated) position rather than their Gaia J2016 catalog position.

    The standard transient detector compares each detection to the Gaia J2016
    catalog positions.  A fast-moving star (|PM| > pm_threshold_masyr mas/yr)
    may have drifted far enough that it no longer matches its catalog entry and
    is therefore flagged as a 'new' source.  This function:
      1. Queries Gaia DR3 via VizieR for all stars with measured PM in the field.
      2. Propagates their positions from J2016.0 to the observation epoch using
         astropy SkyCoord.apply_space_motion().
      3. Filters candidates whose position matches a propagated high-PM star.

    cache_path: if given, the raw high-PM Gaia star list (step 1's result, a
    field-level property that doesn't change from one call to the next) is
    cached there. Only the propagation (step 2, cheap and local) depends on
    obs_jd and is redone every call. Matters for a daemon-style caller that
    invokes this once per new epoch across a multi-night campaign (e.g.
    SubtractionStrategy's accumulation, see pipeline_magic_sn.py's
    run_sn_pipeline) -- without this, the same VizieR query for the same
    field's high-PM stars would otherwise repeat, unchanged, on every call.

    Returns (filtered_candidates, n_rejected).
    """
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy.time import Time

    if len(candidates) == 0:
        return candidates, 0

    if obs_jd is None:
        logger.warning("  PM check: no observation epoch — skipping high-PM star rejection")
        return candidates, 0

    try:
        gaia_hp = None
        if cache_path is not None and Path(cache_path).exists():
            from astropy.table import Table as _Table
            gaia_hp = _Table.read(str(cache_path), format="ascii.ecsv")
            logger.info(f"  PM check: using cached high-PM Gaia stars ({len(gaia_hp)}) "
                        f"from {cache_path}")

        if gaia_hp is None:
            from astroquery.vizier import Vizier

            logger.info(f"  PM check: querying Gaia DR3 for high-PM stars "
                        f"(|PM| > {pm_threshold_masyr} mas/yr) ...")

            v = Vizier(
                columns=['RA_ICRS', 'DE_ICRS', 'pmRA', 'pmDE', 'Gmag'],
                row_limit=-1,
            )
            center = SkyCoord(ra_center, dec_center, unit='deg')
            result = v.query_region(center, radius=field_deg * 0.75 * u.deg,
                                    catalog='I/355/gaiadr3')

            if not result or len(result) == 0:
                logger.info("  PM check: no Gaia DR3 stars returned")
                return candidates, 0

            gaia = result[0]

            # Filter to stars with measured PM above the threshold
            pmra = np.array(gaia['pmRA'], dtype=float)
            pmde = np.array(gaia['pmDE'], dtype=float)
            pm_mag = np.sqrt(np.where(np.isnan(pmra), 0.0, pmra)**2 +
                             np.where(np.isnan(pmde), 0.0, pmde)**2)
            high_pm_mask = pm_mag >= pm_threshold_masyr
            gaia_hp = gaia[high_pm_mask]

            if cache_path is not None:
                try:
                    gaia_hp.write(str(cache_path), format="ascii.ecsv", overwrite=True)
                except Exception as e:
                    logger.debug(f"  PM check: could not write cache {cache_path}: {e}")

        if len(gaia_hp) == 0:
            logger.info("  PM check: no high-PM stars in this field")
            return candidates, 0

        logger.info(f"  PM check: {len(gaia_hp)} high-PM Gaia stars, "
                    f"propagating to JD={obs_jd:.3f} ...")

        # Propagate to observation epoch
        gaia_epoch = Time('J2016.0')
        obs_epoch = Time(obs_jd, format='jd')

        pmra_hp = np.where(np.isnan(np.array(gaia_hp['pmRA'], dtype=float)),
                           0.0, np.array(gaia_hp['pmRA'], dtype=float))
        pmde_hp = np.where(np.isnan(np.array(gaia_hp['pmDE'], dtype=float)),
                           0.0, np.array(gaia_hp['pmDE'], dtype=float))

        gaia_coords = SkyCoord(
            ra=np.array(gaia_hp['RA_ICRS'], dtype=float) * u.deg,
            dec=np.array(gaia_hp['DE_ICRS'], dtype=float) * u.deg,
            pm_ra_cosdec=pmra_hp * u.mas / u.yr,
            pm_dec=pmde_hp * u.mas / u.yr,
            obstime=gaia_epoch,
            frame='icrs',
        )
        propagated = gaia_coords.apply_space_motion(new_obstime=obs_epoch)

        # Match candidates against propagated positions
        cand_coords = SkyCoord(
            np.array(candidates['ALPHA_J2000'], dtype=float) * u.deg,
            np.array(candidates['DELTA_J2000'], dtype=float) * u.deg,
        )

        keep_mask = np.ones(len(candidates), dtype=bool)
        for i, cc in enumerate(cand_coords):
            seps = cc.separation(propagated).arcsec
            if np.min(seps) <= match_radius_arcsec:
                keep_mask[i] = False

        n_rejected = int(np.sum(~keep_mask))
        if n_rejected:
            logger.info(f"  PM check: removed {n_rejected} high-PM star matches")
        else:
            logger.info("  PM check: no candidates match a high-PM star")

        return candidates[keep_mask], n_rejected

    except ImportError:
        logger.warning("  PM check: astroquery not available — skipping")
        return candidates, 0
    except Exception as e:
        logger.warning(f"  PM check: query failed ({e}) — skipping")
        return candidates, 0


def crossmatch_tns(candidates, logger, api_key=None, match_radius_arcsec=5.0,
                    cache_path=None, cache_max_age_hours=24.0):
    """Cross-match candidates against the IAU Transient Name Server (TNS).

    Adds three annotation columns to the candidate table (does NOT remove rows):
      tns_name   — IAU name of the matching transient (e.g. '2021abc'), or ''
      tns_type   — Spectroscopic classification (e.g. 'SN Ia'), or ''
      tns_z      — Redshift from TNS, or NaN

    Known TNS transients are kept in the output because they are confirmed real
    events, but the sn_score for them is not inflated — they simply appear with
    an annotation that lets the operator skip re-reporting them.

    API key:
      Pass via --tns-api-key=<key> (CLI) or TNS_API_KEY environment variable.
      If neither is set the step is skipped with a warning.

    cache_path: per-candidate result cache (keyed by transient_id, falling
    back to a rounded RA/Dec string), unlike reject_high_pm_stars/
    score_galaxy_proximity's per-field cache -- a daemon-style caller
    re-processing the same observation on every new epoch (see
    run_sn_pipeline) would otherwise re-query TNS, rate-limited to ~1/s, for
    the same candidates on every single call. Unlike those two, TNS
    registrations genuinely change over time (a source can get registered
    *after* our first check), so entries expire after cache_max_age_hours
    rather than being cached forever. The 1.1s rate-limit sleep only
    applies to candidates that actually hit the network, so a fully-cached
    re-run is fast.
    """
    import os
    import time

    n = len(candidates)
    tns_name = [''] * n
    tns_type = [''] * n
    tns_z = np.full(n, np.nan)

    key = api_key or os.environ.get('TNS_API_KEY', '')
    if not key:
        logger.warning("  TNS: no API key set (use --tns-api-key= or TNS_API_KEY env var) — skipping")
        candidates['tns_name'] = tns_name
        candidates['tns_type'] = tns_type
        candidates['tns_z'] = tns_z
        return candidates

    if n == 0:
        candidates['tns_name'] = tns_name
        candidates['tns_type'] = tns_type
        candidates['tns_z'] = tns_z
        return candidates

    try:
        import requests
    except ImportError:
        logger.warning("  TNS: 'requests' library not available — skipping")
        candidates['tns_name'] = tns_name
        candidates['tns_type'] = tns_type
        candidates['tns_z'] = tns_z
        return candidates

    def _candidate_key(i):
        if 'transient_id' in candidates.colnames:
            return str(candidates['transient_id'][i])
        return (f"{float(candidates['ALPHA_J2000'][i]):.4f}_"
                f"{float(candidates['DELTA_J2000'][i]):.4f}")

    cache = {}
    if cache_path is not None and Path(cache_path).exists():
        try:
            with open(cache_path) as f:
                cache = json.load(f)
        except Exception as e:
            logger.debug(f"  TNS: could not read cache {cache_path}: {e}")

    now = time.time()
    max_age_s = cache_max_age_hours * 3600.0

    TNS_URL = 'https://www.wis-tns.org/api/get/search'
    # TNS requires a User-Agent header that identifies the bot
    headers = {
        'User-Agent': 'tns_marker{"tns_id": 0, "type": "user", "name": "sn_pipeline"}',
        'Accept': 'application/json',
    }

    n_matched = 0
    n_from_cache = 0
    n_queried = 0
    cache_dirty = False

    for i in range(n):
        ra_i = float(candidates['ALPHA_J2000'][i])
        dec_i = float(candidates['DELTA_J2000'][i])
        cid = _candidate_key(i)

        cached = cache.get(cid)
        if cached is not None and (now - cached.get('checked_at', 0)) < max_age_s:
            tns_name[i] = cached.get('tns_name', '') or ''
            tns_type[i] = cached.get('tns_type', '') or ''
            cz = cached.get('tns_z')
            tns_z[i] = float(cz) if cz is not None else np.nan
            if tns_name[i]:
                n_matched += 1
            n_from_cache += 1
            continue

        search_data = json.dumps({
            'ra': f'{ra_i:.6f}',
            'declination': f'{dec_i:.6f}',
            'radius': str(match_radius_arcsec),
            'units': 'arcsec',
            'objtype[]': [],
            'groupid[]': [],
            'public': '1',
        })

        if n_queried == 0:
            logger.info(f"  TNS: querying (radius={match_radius_arcsec}\") ...")

        try:
            resp = requests.post(
                TNS_URL,
                data={'api_key': key, 'data': search_data},
                headers=headers,
                timeout=15,
            )
            resp.raise_for_status()
            payload = resp.json()

            reply = payload.get('data', {}).get('reply', [])
            if reply:
                obj = reply[0]  # closest / first match
                prefix = obj.get('name_prefix', obj.get('prefix', 'AT'))
                name = obj.get('objname', '')
                tns_name[i] = f"{prefix}{name}" if name else ''
                tns_type[i] = obj.get('type', '') or ''
                z = obj.get('redshift')
                try:
                    tns_z[i] = float(z) if z not in (None, '', 'None') else np.nan
                except (ValueError, TypeError):
                    tns_z[i] = np.nan
                if tns_name[i]:
                    n_matched += 1
                    logger.info(f"    TNS match: {tns_name[i]} ({tns_type[i]}) "
                                f"at RA={ra_i:.4f} Dec={dec_i:.4f}")

            cache[cid] = {
                'tns_name': tns_name[i], 'tns_type': tns_type[i],
                'tns_z': float(tns_z[i]) if np.isfinite(tns_z[i]) else None,
                'checked_at': now,
            }
            cache_dirty = True

        except requests.exceptions.Timeout:
            logger.debug(f"  TNS: timeout for candidate {i}, skipping")
        except Exception as e:
            logger.debug(f"  TNS: query failed for candidate {i}: {e}")

        n_queried += 1
        # Respect TNS rate limit: ~1 req/s for bot accounts. Only sleeps
        # after an actual network call, so a fully-cached re-run is fast.
        time.sleep(1.1)

    if cache_path is not None and cache_dirty:
        try:
            with open(cache_path, 'w') as f:
                json.dump(cache, f)
        except Exception as e:
            logger.debug(f"  TNS: could not write cache {cache_path}: {e}")

    logger.info(f"  TNS: {n_matched} of {n} candidates matched known transients "
                f"({n_from_cache} from cache, {n_queried} queried)")
    candidates['tns_name'] = tns_name
    candidates['tns_type'] = tns_type
    candidates['tns_z'] = tns_z
    return candidates


def compute_sn_score(candidates, logger, target_positions=None):
    """Compute a composite supernova-likelihood score for each candidate.

    Score components (all additive):
      base      = quality_score × type_weight
                    type_weight: new=1.0, brightening=0.8, fading=0.2,
                                 trail=0.0, unknown=0.3
      galaxy    = +0.4 if in_galaxy, +0.1 if nuclear (likely AGN, small bonus),
                  0.0 if isolated
      morphology= axis_ratio × 0.2  (round sources score up to 0.2 extra)

    Known TNS transients are NOT given a score boost — they are already
    confirmed and just need to be noted, not re-ranked.

    Typical range: 0 – ~3.4 (higher = more SN-like).
    """
    n = len(candidates)
    if n == 0:
        candidates['sn_score'] = np.array([], dtype=float)
        return candidates

    # Base: quality_score weighted by candidate type
    type_weight = {'new': 1.0, 'brightening': 0.8, 'fading': 0.2,
                   'trail': 0.0, 'unknown': 0.3}
    base = np.array(
        candidates['quality_score'] if 'quality_score' in candidates.colnames
        else np.ones(n) * 0.5,
        dtype=float
    )
    tw = np.array([
        type_weight.get(str(t), 0.3)
        for t in (candidates['candidate_type'] if 'candidate_type' in candidates.colnames
                  else ['unknown'] * n)
    ])

    # Galaxy proximity bonus
    gal_bonus = np.zeros(n)
    if 'galaxy_flag' in candidates.colnames:
        flags = np.array(candidates['galaxy_flag'])
        gal_bonus[flags == 'in_galaxy'] = 0.4
        gal_bonus[flags == 'nuclear'] = 0.1

    # Morphology bonus (round = point-like = more SN-like)
    morph_bonus = np.zeros(n)
    if 'axis_ratio' in candidates.colnames:
        ar = np.clip(np.array(candidates['axis_ratio'], dtype=float), 0.0, 1.0)
        morph_bonus = ar * 0.2

    # Proximity bonus: prefer candidates near known target positions (optional)
    PROX_RADIUS = 4.0   # degrees
    PROX_MAX    = 0.5
    prox_bonus = np.zeros(n)
    if (target_positions
            and 'ALPHA_J2000' in candidates.colnames
            and 'DELTA_J2000' in candidates.colnames):
        ras  = np.array(candidates['ALPHA_J2000'], dtype=float)
        decs = np.array(candidates['DELTA_J2000'], dtype=float)
        for t_ra, t_dec in target_positions:
            dra  = (ras - t_ra) * np.cos(np.radians(t_dec))
            ddec = decs - t_dec
            sep  = np.sqrt(dra**2 + ddec**2)
            bonus = PROX_MAX * np.clip(1.0 - sep / PROX_RADIUS, 0.0, 1.0)
            prox_bonus = np.maximum(prox_bonus, bonus)
        n_boosted = int(np.sum(prox_bonus > 0))
        logger.info(f"  Proximity bonus: {n_boosted} candidates within {PROX_RADIUS}° "
                    f"of target positions (max bonus {prox_bonus.max():.3f})")

    sn_score = base * tw + gal_bonus + morph_bonus + prox_bonus
    candidates['sn_score'] = np.round(sn_score, 4)

    logger.info(f"  SN score range: "
                f"{np.min(sn_score):.3f} – {np.max(sn_score):.3f} "
                f"(median {np.median(sn_score):.3f})")
    return candidates


# ---------------------------------------------------------------------------
# Core pipeline orchestration
# ---------------------------------------------------------------------------

# extract_observation_id/setup_logging removed -- both duplicated what
# pyrt_transient.io.observation_store.extract_observation_id and
# pyrt_transient.io.logging_setup.setup_pipeline_logging already do (same
# functions pipeline_magic.py uses), imported at the top of this file.


def _ensure_diff_epochs(sci_tables, sci_fits_paths, config, logger,
                        target_ra=None, target_dec=None):
    """Phase B: given accumulated raw science epochs, ensure every one has a
    corresponding diff epoch on disk (template acquisition + differencing +
    extraction/calibration -- detection/subtraction/templates.py,
    differencing.py, extraction.py), building any that don't exist yet.
    Idempotent: an epoch whose diff `.ecsv` already exists is loaded as-is,
    not rebuilt (matches Phase A's "already processed, skipping" pattern in
    SubtractionStrategy.run()'s Step 1).

    Returns the list of diff detection Tables to feed into
    SubtractionStrategy.run() -- the same role Phase A's pre-built diff
    tables play, just produced here instead of consumed directly.

    Lazy imports: templates.py/differencing.py/extraction.py pull in
    optional, heavier dependencies (stdpipe, PyZOGY, network access) that
    shouldn't make importing this whole module fail for a caller only using
    "prebuilt" diff_input_mode or the blind_multicatalog strategy.
    """
    from astropy.io import fits as _fits
    from pyrt_transient.detection.subtraction import templates as sub_templates
    from pyrt_transient.detection.subtraction import differencing as sub_differencing
    from pyrt_transient.detection.subtraction import extraction as sub_extraction

    template_source = config.detection.template_source
    engine = config.detection.subtraction_engine
    photometric_catalog = config.detection.photometric_catalog

    diff_tables = []
    for i, (sci_table, sci_fits_path) in enumerate(zip(sci_tables, sci_fits_paths)):
        sci_fits_path = Path(sci_fits_path)
        diff_fits_path = sci_fits_path.with_name(f"{sci_fits_path.stem}h.fits")
        diff_ecsv_path = diff_fits_path.with_suffix(".ecsv")

        if diff_ecsv_path.exists():
            existing = load_diff_table(diff_ecsv_path)
            if existing is not None:
                diff_tables.append(existing)
                continue

        if template_source == "own_epoch":
            other_tables = [t for j, t in enumerate(sci_tables) if j != i]
            other_paths = [p for j, p in enumerate(sci_fits_paths) if j != i]
            if not other_tables:
                logger.info(f"{sci_fits_path.name}: no other epochs yet for an "
                            f"own_epoch template, skipping this epoch's diff for now")
                continue
            template_result = sub_templates.get_template_own_epoch(
                other_tables, other_paths, target_ra=target_ra, target_dec=target_dec,
            )
        elif template_source in ("ps1", "legacysurvey"):
            try:
                with _fits.open(sci_fits_path) as hdul:
                    sci_header = hdul[0].header
            except Exception as e:
                logger.warning(f"{sci_fits_path}: could not read header: {e}")
                continue
            fetch_fn = (sub_templates.get_template_ps1 if template_source == "ps1"
                        else sub_templates.get_template_legacysurvey)
            template_result = fetch_fn(
                sci_header, cache_dir=config.detection.template_cache_dir,
                cache_max_size_gb=config.detection.template_cache_max_size_gb,
            )
        else:
            logger.error(f"Unknown template_source: {template_source!r}")
            continue

        if template_result is None:
            logger.warning(f"{sci_fits_path.name}: no template available yet, "
                            f"skipping this epoch's diff for now")
            continue

        template_array = template_result[0]
        template_mask = template_result[1] if len(template_result) == 3 else None
        provenance = template_result[-1]

        diff_out = sub_differencing.run_diff(
            engine, sci_fits_path, template_array, template_mask=template_mask,
            output_path=diff_fits_path, template_provenance=provenance,
            science_meta=dict(sci_table.meta),
        )
        if diff_out is None:
            logger.warning(f"{sci_fits_path.name}: differencing failed, skipping this epoch")
            continue

        ecsv_out = sub_extraction.build_diff_ecsv(
            diff_out, sci_fits_path, output_path=diff_ecsv_path,
            photometric_catalog=photometric_catalog,
        )
        if ecsv_out is None:
            logger.warning(f"{sci_fits_path.name}: diff extraction/calibration failed, skipping")
            continue

        diff_table = load_diff_table(ecsv_out)
        if diff_table is not None:
            diff_tables.append(diff_table)

    return diff_tables


def run_sn_pipeline(ecsv_file, fits_file, obs_dir, config, logger, tns_api_key=None,
                    target_positions=None):
    """Execute all pipeline steps; return final candidate table.

    Steps 1-2 (candidate sourcing) branch on config.detection.strategy:
    "blind_multicatalog" (default -- cross-match against reference catalogs,
    the original single/multi-image SN mode) or "subtraction" (differencing
    against a template, see detection/subtraction/). Both go through
    ObservationStore the same way pipeline_magic.py's GRB pipeline does, so
    repeated invocations (the daemon calling this once per new image)
    accumulate epochs correctly instead of only ever seeing one image at a
    time -- SubtractionStrategy in particular needs the full accumulated
    epoch list each call to cluster across nights and build lightcurves,
    exactly like BlindMulticatalogStrategy already does.

    Steps 3 onward (morphology/magnitude/SkyBot/PM-star/galaxy/TNS/scoring)
    are strategy-agnostic and apply to whichever strategy produced
    `candidates`.
    """
    from astropy.table import Table

    obs_dir = Path(obs_dir)
    obs_dir.mkdir(parents=True, exist_ok=True)
    observation_id = obs_dir.name.replace('obs_', '', 1)
    store = ObservationStore(obs_dir.parent, observation_id)

    strategy_name = config.detection.strategy if config else "blind_multicatalog"

    if strategy_name == "subtraction":
        diff_input_mode = config.detection.diff_input_mode if config else "prebuilt"

        if diff_input_mode == "raw":
            # ----- Phase B: ecsv_file/fits_file are a raw science epoch --
            # build the template/diff/extraction ourselves before handing
            # off to the same SubtractionStrategy (see _ensure_diff_epochs).
            sci_dest = obs_dir / Path(ecsv_file).name
            fits_dest = obs_dir / Path(fits_file).name
            if not sci_dest.exists():
                shcopy(ecsv_file, sci_dest)
            if not fits_dest.exists():
                shcopy(fits_file, fits_dest)

            if not store.already_processed(sci_dest.name):
                store.mark_processed(sci_dest.name)

            sci_tables, _ = store.load_existing_tables()
            if not sci_tables:
                logger.error("No usable science detection tables")
                return Table()
            sci_fits_paths = [obs_dir / f"{Path(t.meta.get('filename', '')).stem}.fits"
                               for t in sci_tables]
            logger.info(f"Science epochs accumulated: {len(sci_tables)}")

            target_ra = float(target_positions[0][0]) if target_positions else None
            target_dec = float(target_positions[0][1]) if target_positions else None

            with store.analysis_lock():
                detection_tables = _ensure_diff_epochs(
                    sci_tables, sci_fits_paths, config, logger,
                    target_ra=target_ra, target_dec=target_dec,
                )
                if not detection_tables:
                    logger.warning("No diff epochs could be built yet (template "
                                    "acquisition or differencing unavailable for "
                                    "every accumulated science epoch so far)")
                    return Table()
                logger.info(f"Detection tables (diff epochs built): {len(detection_tables)}")

                strategy = SubtractionStrategy(data_dir=str(obs_dir), config=config)
                candidates, lightcurves = strategy.run(
                    detection_tables, config=config,
                    template_provenance=config.detection.template_source,
                )

        else:
            # ----- Phase A: ecsv_file/fits_file are already a diff-image
            # pair (e.g. an externally-produced campaign like
            # tests/2026kid/). Copy into obs_dir, normalized to `.ecsv` so
            # ObservationStore.load_existing_tables' glob("*.ecsv") picks it
            # up regardless of the source's original extension (the real
            # fixture only has `.cat` for diff epochs). -----
            diff_table = load_diff_table(ecsv_file)
            if diff_table is None:
                logger.error(f"Failed to load diff detection table: {ecsv_file}")
                return Table()

            diff_dest = obs_dir / f"{Path(ecsv_file).stem}.ecsv"
            if not diff_dest.exists():
                diff_table.write(str(diff_dest), format="ascii.ecsv", overwrite=True)
            fits_dest = obs_dir / Path(fits_file).name
            if not fits_dest.exists():
                shcopy(fits_file, fits_dest)

            # Also copy the science sibling alongside it, same naming
            # convention, so candidates.py's find_science_sibling/
            # borrow_science_meta can still locate it after the copy. Never
            # mark_processed it: ObservationStore.load_existing_tables only
            # returns files it has marked processed, so the sibling stays
            # findable on disk without being treated as a diff epoch itself.
            sci_path = find_science_sibling(Path(ecsv_file))
            if sci_path is not None:
                sci_table = load_diff_table(sci_path)
                if sci_table is not None:
                    sci_dest = obs_dir / f"{sci_path.stem}.ecsv"
                    if not sci_dest.exists():
                        sci_table.write(str(sci_dest), format="ascii.ecsv", overwrite=True)
                sci_fits = sci_path.with_suffix(".fits")
                if sci_fits.exists():
                    sci_fits_dest = obs_dir / sci_fits.name
                    if not sci_fits_dest.exists():
                        shcopy(str(sci_fits), sci_fits_dest)
            else:
                logger.warning(f"No science sibling found for {ecsv_file} -- "
                                f"diff magnitudes may be uncalibrated")

            if not store.already_processed(diff_dest.name):
                store.mark_processed(diff_dest.name)

            detection_tables, _ = store.load_existing_tables()
            if not detection_tables:
                logger.error("No usable diff detection tables")
                return Table()
            logger.info(f"Detection tables (diff epochs): {len(detection_tables)}")

            strategy = SubtractionStrategy(data_dir=str(obs_dir), config=config)
            with store.analysis_lock():
                candidates, lightcurves = strategy.run(
                    detection_tables, config=config,
                    template_provenance=config.detection.template_source,
                )

    else:
        # ----- blind_multicatalog: original single/multi-image mode -----
        ecsv_dest = obs_dir / Path(ecsv_file).name
        fits_dest = obs_dir / Path(fits_file).name
        if not ecsv_dest.exists():
            shcopy(ecsv_file, ecsv_dest)
        if not fits_dest.exists():
            shcopy(fits_file, fits_dest)

        if not store.already_processed(ecsv_dest.name):
            store.mark_processed(ecsv_dest.name)

        detection_tables, _ = store.load_existing_tables()
        if not detection_tables:
            logger.error("No usable detection tables")
            return Table()
        logger.info(f"Detection tables: {len(detection_tables)}")

        first_det = detection_tables[0]
        query_params = QueryParams(
            ra=float(first_det.meta.get('CTRRA', first_det.meta.get('RA', 0.0))),
            dec=float(first_det.meta.get('CTRDEC', first_det.meta.get('DEC', 0.0))),
            width=1.2 * float(first_det.meta.get('FIELD', 0.5)),
            height=1.2 * float(first_det.meta.get('FIELD', 0.5)),
            mlim=float(first_det.meta.get('MAGLIM', first_det.meta.get('MAGLIMIT', 20.0))),
        )

        strategy = BlindMulticatalogStrategy(data_dir=str(obs_dir), config=config)
        with store.analysis_lock():
            candidates, lightcurves = strategy.run(
                detection_tables, config=config,
                catalogs=config.detection.catalogs,
                params=query_params,
                idlimit=config.detection.idlimit_px,
                radius_check=config.detection.radius_check,
                filter_pattern=config.detection.filter_pattern,
                mag_change_threshold=1.0,
            )

    logger.info(f"After detection strategy ({strategy_name}): {len(candidates)} candidates")
    if len(candidates) == 0:
        return candidates

    # ----- Field metadata for the strategy-agnostic filters below (SkyBot/
    # PM-star/galaxy proximity all need RA/Dec/field size/JD), taken from
    # whichever epoch was most recently added. -----
    ref_meta = detection_tables[-1].meta
    ra = float(ref_meta.get('CTRRA', ref_meta.get('RA', 0.0)))
    dec = float(ref_meta.get('CTRDEC', ref_meta.get('DEC', 0.0)))
    field_deg = float(ref_meta.get('FIELD', 0.5))
    obs_jd = None
    if 'JD' in ref_meta:
        obs_jd = float(ref_meta['JD'])
    elif 'MJD-OBS' in ref_meta:
        obs_jd = float(ref_meta['MJD-OBS']) + 2400000.5
    logger.info(f"Field: RA={ra:.5f}  Dec={dec:.5f}  size={field_deg:.3f} deg  JD={obs_jd}")

    # =========================================================================
    # Step 3 — Morphology filter (point-source consistency)
    # =========================================================================
    logger.info("--- Step 3: Morphology filter ---")
    candidates = apply_morphology_filter(
        candidates, logger,
        max_ellipticity=config.detection.morphology_max_ellipticity,
        fwhm_ratio_range=(config.detection.morphology_fwhm_ratio_min,
                          config.detection.morphology_fwhm_ratio_max),
    )
    logger.info(f"After morphology filter: {len(candidates)} candidates")

    if len(candidates) == 0:
        return candidates

    # =========================================================================
    # Step 4 — Magnitude plausibility
    # =========================================================================
    logger.info("--- Step 4: Magnitude plausibility filter ---")
    candidates = apply_magnitude_filter(candidates, logger, bright_limit=14.0)
    logger.info(f"After magnitude filter: {len(candidates)} candidates")

    if len(candidates) == 0:
        return candidates

    # =========================================================================
    # Step 5 — Asteroid rejection via IMCCE SkyBot
    # =========================================================================
    logger.info("--- Step 5: SkyBot asteroid rejection ---")
    candidates, n_sso = reject_known_asteroids(
        candidates, obs_jd, ra, dec, field_deg, logger
    )
    logger.info(f"After SkyBot: {len(candidates)} candidates ({n_sso} SSOs removed)")

    if len(candidates) == 0:
        return candidates

    # =========================================================================
    # Step 6 — High proper-motion star rejection (Gaia DR3 via VizieR)
    # =========================================================================
    # cache_path: this observation gets re-processed once per new epoch in a
    # daemon-style multi-night campaign (see run_sn_pipeline's docstring) --
    # without caching, the same field-level VizieR query would otherwise
    # repeat, with the same result, on every single call.
    logger.info("--- Step 6: High proper-motion star rejection ---")
    candidates, n_pm = reject_high_pm_stars(
        candidates, ra, dec, field_deg, obs_jd, logger,
        cache_path=obs_dir / "gaia_highpm_cache.ecsv",
    )
    logger.info(f"After PM check: {len(candidates)} candidates ({n_pm} PM stars removed)")

    if len(candidates) == 0:
        return candidates

    # =========================================================================
    # Step 7 — Galaxy proximity (HyperLEDA via VizieR)
    # =========================================================================
    logger.info("--- Step 7: Galaxy proximity scoring ---")
    gal_search_deg = field_deg * 0.65
    candidates = score_galaxy_proximity(
        candidates, ra, dec, gal_search_deg, logger,
        cache_path=obs_dir / "hyperleda_cache.ecsv",
    )

    # =========================================================================
    # Step 8 — TNS cross-match (annotates; does not remove candidates)
    # =========================================================================
    logger.info("--- Step 8: TNS cross-match ---")
    candidates = crossmatch_tns(
        candidates, logger, api_key=tns_api_key,
        cache_path=obs_dir / "tns_cache.json",
    )

    # =========================================================================
    # Step 8b — Spatial cut to target positions (optional)
    # =========================================================================
    if target_positions and 'ALPHA_J2000' in candidates.colnames:
        PROX_RADIUS = 4.0  # degrees — same radius as proximity bonus
        ras  = np.array(candidates['ALPHA_J2000'], dtype=float)
        decs = np.array(candidates['DELTA_J2000'], dtype=float)
        keep = np.zeros(len(candidates), dtype=bool)
        for t_ra, t_dec in target_positions:
            dra  = (ras - t_ra) * np.cos(np.radians(t_dec))
            ddec = decs - t_dec
            sep  = np.sqrt(dra**2 + ddec**2)
            keep |= (sep <= PROX_RADIUS)
        n_cut = int(np.sum(~keep))
        candidates = candidates[keep]
        logger.info(f"--- Step 8b: Spatial cut — {n_cut} candidates removed "
                    f"(>{PROX_RADIUS}° from all targets), {len(candidates)} remaining ---")

    # =========================================================================
    # Step 9 — Composite SN score & final sort
    # =========================================================================
    logger.info("--- Step 9: SN score ---")
    candidates = compute_sn_score(candidates, logger, target_positions=target_positions)

    order = np.argsort(np.array(candidates['sn_score'], dtype=float))[::-1]
    candidates = candidates[order]

    logger.info(f"Final: {len(candidates)} SN candidates")
    candidates.meta['obs_jd'] = obs_jd
    return candidates


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def save_results(candidates, obs_dir, observation_id, logger):
    """Write candidates.tbl (IPAC) and sn_report.json."""
    obs_dir = Path(obs_dir)

    candidates_file = obs_dir / 'candidates.tbl'
    candidates.write(str(candidates_file), format='ascii.ipac', overwrite=True)
    logger.info(f"Saved {len(candidates)} candidates → {candidates_file}")

    report_cols = [
        'ALPHA_J2000', 'DELTA_J2000', 'MAG_CALIB', 'MAGERR_CALIB',
        'candidate_type', 'quality_score', 'sn_score',
        'galaxy_sep_arcsec', 'galaxy_name', 'galaxy_flag',
        'tns_name', 'tns_type', 'tns_z',
        'reference_catalog', 'ELLIPTICITY', 'FLAGS',
    ]
    report = {
        'observation_id': observation_id,
        'generated': datetime.now().isoformat(),
        'n_candidates': len(candidates),
        'candidates': [],
    }
    for row in candidates:
        entry = {}
        for col in report_cols:
            if col not in candidates.colnames:
                continue
            val = row[col]
            try:
                fval = float(val)
                entry[col] = None if np.isnan(fval) else fval
            except (TypeError, ValueError):
                entry[col] = str(val)
        report['candidates'].append(entry)

    report_file = obs_dir / 'sn_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info(f"SN report → {report_file}")


def _save_empty_results(obs_dir, observation_id):
    from astropy.table import Table
    obs_dir = Path(obs_dir)
    Table().write(str(obs_dir / 'candidates.tbl'), format='ascii.ipac', overwrite=True)
    with open(obs_dir / 'sn_report.json', 'w') as f:
        json.dump({'observation_id': observation_id,
                   'generated': datetime.now().isoformat(),
                   'n_candidates': 0, 'candidates': []}, f, indent=2)


# Config loading: load_config_with_yaml_support (imported at the top) does
# exactly this (YAML via PipelineConfig.from_dict, else PipelineConfig.from_file)
# -- no reason for a second copy.


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 3:
        print(
            "Usage: pipeline_magic_sn.py <ecsv_file> <fits_file>\n"
            "       [--output-dir=<path>] [--config=<file>]\n"
            "       [--strategy=blind_multicatalog|subtraction]\n"
            "       [--min-catalogs=<fraction>] [--tns-api-key=<key>]\n"
            "       [--target-positions=<ra1,dec1;ra2,dec2;...>]\n"
            "       [--generate-frontend] [--debug]\n"
            "       For --strategy=subtraction, <ecsv_file> is a diff-image\n"
            "       detection file (.cat or .ecsv); its science sibling is\n"
            "       located automatically (see detection/subtraction/candidates.py).\n"
            "       TNS key can also be set via TNS_API_KEY env var.",
            file=sys.stderr,
        )
        sys.exit(1)

    ecsv_file = sys.argv[1]
    fits_file = sys.argv[2]

    output_dir = None
    config_file = None
    tns_api_key = None
    atlas_token = None
    target_positions = None
    debug = '--debug' in sys.argv
    generate_frontend_flag = '--generate-frontend' in sys.argv
    min_catalogs_fraction = None
    strategy_override = None

    for arg in sys.argv[3:]:
        if arg.startswith('--output-dir='):
            output_dir = arg.split('=', 1)[1]
        elif arg.startswith('--config='):
            config_file = arg.split('=', 1)[1]
        elif arg.startswith('--strategy='):
            strategy_override = arg.split('=', 1)[1]
        elif arg.startswith('--min-catalogs='):
            min_catalogs_fraction = float(arg.split('=', 1)[1])
        elif arg.startswith('--tns-api-key='):
            tns_api_key = arg.split('=', 1)[1]
        elif arg.startswith('--atlas-token='):
            atlas_token = arg.split('=', 1)[1]
        elif arg.startswith('--target-positions='):
            # Format: "ra1,dec1;ra2,dec2;..."
            target_positions = [
                tuple(float(x) for x in pair.split(','))
                for pair in arg.split('=', 1)[1].split(';')
                if pair.strip()
            ]
        elif not arg.startswith('--'):
            output_dir = arg  # positional output dir for backward compat

    # Load base config
    config = load_config_with_yaml_support(config_file) if config_file else PipelineConfig()
    if strategy_override:
        config.detection.strategy = strategy_override

    # Apply SN-mode overrides
    if config.detection.strategy != "subtraction":
        # blind_multicatalog SN mode: single/deep image, no cross-epoch
        # lightcurve requirement. Subtraction mode needs the opposite --
        # min_n_detections (default 3) is exactly what lets a real target
        # survive cross-epoch clustering across a multi-night campaign (see
        # SubtractionStrategy/FUTURE_IDEAS.md's Phase-A validation) -- do
        # not force it to 1 there.
        config.detection.min_n_detections = 1
    # Use SN-specific template unless the user has explicitly set one via config
    if not config.frontend.template_dir:
        config.frontend.template_dir = str(Path(__file__).parent / 'template_sn')
    # SN pipeline can produce many candidates; raise the cap
    if config.frontend.max_candidates <= 100:
        config.frontend.max_candidates = 500
    if min_catalogs_fraction is not None:
        config.detection.min_catalogs_fraction = min_catalogs_fraction
    elif config.detection.strategy != "subtraction" and config.detection.min_catalogs_fraction >= 1.0:
        # Default for blind-multicatalog SN mode: require majority of active
        # catalogs. With 4 catalogs (gaia, usno, atlas, legacysurvey):
        #   ceil(0.51 × 4) = 3  → 3 of 4 must flag the source
        # With 3 catalogs (no atlas):
        #   ceil(0.51 × 3) = 2  → 2 of 3 must flag the source
        # Not applicable to subtraction: SubtractionStrategy always treats
        # its diff-image detections as a single pseudo-catalog internally,
        # so min_catalogs_fraction=1.0 there is already trivially satisfied.
        config.detection.min_catalogs_fraction = 0.51
    if generate_frontend_flag:
        config.generate_frontend = True
    if debug:
        config.logging.level = 'DEBUG'
    if output_dir:
        config.base_data_dir = output_dir

    # Observation ID. For subtraction mode, prefer the campaign-stable
    # OBJECT/TARGET from the science sibling over extract_observation_id's
    # OBSID-based default -- real OBSID changes per night (verified on the
    # real tests/2026kid/ fixture: four different OBSID values across four
    # nights of the same AT2026kid campaign), which would otherwise put
    # every epoch in a different ObservationStore directory and defeat
    # cross-epoch clustering entirely (see candidates.derive_observation_id).
    observation_id = None
    if config.detection.strategy == "subtraction":
        observation_id = derive_observation_id(ecsv_file)
    if observation_id is None:
        observation_id = extract_observation_id(ecsv_file)
    obs_dir = Path(config.base_data_dir) / f'obs_{observation_id}'

    # Logging
    logger = setup_pipeline_logging(config, observation_id)
    log_file = Path(config.base_data_dir) / "logs" / f"pipeline_{observation_id}.log"
    logger.info(f"=== SN Pipeline ({config.detection.strategy} mode) ===")
    logger.info(f"Input ecsv : {ecsv_file}")
    logger.info(f"Input fits : {fits_file}")
    logger.info(f"Obs ID     : {observation_id}")
    logger.info(f"Output dir : {obs_dir}")
    logger.info(f"Log file   : {log_file}")
    logger.info(f"min_catalogs_fraction = {config.detection.min_catalogs_fraction}")
    logger.info(f"TNS key    : {'set' if tns_api_key else 'not set (TNS step will be skipped)'}")

    # Validate inputs
    for path, label in [(ecsv_file, 'ECSV'), (fits_file, 'FITS')]:
        if not Path(path).exists():
            logger.error(f"{label} file not found: {path}")
            sys.exit(1)

    setup_catalog_cache(str(Path.home() / 'catalog_cache'))

    # Run
    try:
        candidates = run_sn_pipeline(ecsv_file, fits_file, obs_dir, config, logger,
                                     tns_api_key=tns_api_key,
                                     target_positions=target_positions)
    except Exception as e:
        import traceback
        logger.error(f"Pipeline failed: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1)

    # Save
    if len(candidates) > 0:
        save_results(candidates, obs_dir, observation_id, logger)
    else:
        logger.info("No candidates found")
        _save_empty_results(obs_dir, observation_id)

    # Forced photometry lightcurves (PS1 always, ATLAS if token provided)
    #
    # NOTE: `forced_photometry` was removed from this package in e40a0ae
    # ("unwired") before pipeline_magic_sn.py was ported into the current
    # architecture -- it does not exist anywhere in pyrt_transient today, so
    # this whole block always raises ModuleNotFoundError and is caught by
    # the except below (logged as a warning, not fatal). Pre-existing gap,
    # unrelated to the subtraction strategy; not reimplemented here.
    if len(candidates) > 0:
        try:
            from forced_photometry import (
                query_panstarrs_lightcurves, query_atlas_lightcurves, merge_lightcurves
            )
            import json as _json
            lc_file = Path(obs_dir) / 'lightcurves.json'

            logger.info("--- Forced photometry: PS1 ---")
            ps1_lcs = query_panstarrs_lightcurves(candidates, logger=logger)

            # Filter out known persistent sources with stable PS1 history
            from forced_photometry import filter_ps1_known_sources
            logger.info("--- PS1 history filter ---")
            candidates, n_ps1_rejected = filter_ps1_known_sources(
                candidates, ps1_lcs, logger=logger
            )
            logger.info(f"After PS1 filter: {len(candidates)} candidates "
                        f"({n_ps1_rejected} known persistent sources removed)")
            if n_ps1_rejected > 0:
                save_results(candidates, obs_dir, observation_id, logger)

            atlas_lcs = {}
            _atlas_token = (atlas_token
                            or os.environ.get('ATLAS_TOKEN')
                            or os.environ.get('ATLASFORCED_SECRET_KEY'))
            # Load existing ATLAS data so we don't re-query what we already have
            if lc_file.exists():
                with open(lc_file) as f:
                    _existing = _json.load(f)
                atlas_lcs = {cid: entry['atlas']
                             for cid, entry in _existing.items() if entry.get('atlas')}
                logger.info(f"Loaded existing ATLAS data for {len(atlas_lcs)} candidates")
            if _atlas_token:
                from forced_photometry import _get_cid
                atlas_top_n = 25
                # Use post-filter candidates — these are the ones in candidates.tbl.
                # Querying pre-filter candidates would waste ATLAS jobs on rejected sources.
                sorted_cands = candidates[
                    np.argsort(np.array(candidates['sn_score'], dtype=float))[::-1]
                ][:atlas_top_n]
                new_cands = sorted_cands[[_get_cid(r) not in atlas_lcs for r in sorted_cands]]
                if len(new_cands) > 0:
                    logger.info(f"--- Forced photometry: ATLAS ({len(new_cands)} new candidates) ---")
                    new_atlas = query_atlas_lightcurves(
                        new_cands, obs_jd=candidates.meta.get('obs_jd', 0),
                        logger=logger, api_token=_atlas_token,
                        max_wait_per=600,
                    )
                    atlas_lcs.update(new_atlas)
                else:
                    logger.info("ATLAS: all top candidates already have data, skipping")

            lcs = merge_lightcurves(ps1_lcs, atlas_lcs, candidates)
            with open(lc_file, 'w') as f:
                _json.dump(lcs, f)
            logger.info(f"Lightcurves saved → {lc_file}")

            # Compute per-candidate brightening delta and update sn_score
            from forced_photometry import compute_lc_mag_delta, _get_cid
            lc_stats = compute_lc_mag_delta(lcs, candidates)

            deltas, lc_bonus = [], []
            for row in candidates:
                cid = _get_cid(row)
                stat = lc_stats.get(cid)
                d = stat['delta'] if (stat and np.isfinite(stat['delta'])) else float('nan')
                deltas.append(d)
                # Smooth bonus: 0 below 0.5 mag brightening, up to +1.5 at ≥2.5 mag
                bonus = min(max(d - 0.5, 0.0) * 0.75, 1.5) if np.isfinite(d) else 0.0
                lc_bonus.append(bonus)

            candidates['lc_mag_delta'] = np.round(deltas, 3)
            candidates['sn_score'] = np.round(
                np.array(candidates['sn_score'], dtype=float) + np.array(lc_bonus), 4
            )
            n_boosted = sum(1 for b in lc_bonus if b > 0)
            logger.info(f"LC score bonus applied: {n_boosted} candidates boosted "
                        f"(max bonus {max(lc_bonus):.2f})")
            # Re-sort and re-save with updated scores
            order = np.argsort(np.array(candidates['sn_score'], dtype=float))[::-1]
            candidates = candidates[order]
            save_results(candidates, obs_dir, observation_id, logger)
        except Exception as e:
            import traceback
            logger.warning(f"Forced photometry step failed: {e}")
            logger.debug(traceback.format_exc())

    # Frontend generation — always regenerate (bypass MD5 gating used in pipeline_magic.py)
    if config.generate_frontend and len(candidates) > 0:
        try:
            from pyrt_transient.frontend_generator import FrontendGenerator
            base_public_dir = config.base_public_dir or Path.home() / 'public_html'
            website_dir = Path(base_public_dir) / f'obs_{observation_id}'
            # Remove stale site state so template changes are always picked up
            stale_state = website_dir / '.site_state.json'
            if stale_state.exists():
                stale_state.unlink()
            frontend_gen = FrontendGenerator(
                observation_id=observation_id,
                data_dir=obs_dir,
                base_public_dir=base_public_dir,
                config=config.frontend,
            )
            success = frontend_gen.generate_complete_website()
            if success:
                logger.info(f"Website: file://{website_dir}/index.html")
            else:
                logger.warning("Frontend generation returned failure")
        except Exception as e:
            import traceback
            logger.warning(f"Frontend generation failed: {e}")
            logger.debug(traceback.format_exc())

    logger.info(f"=== Done: {len(candidates)} SN candidates ===")
    logger.info(f"Results : {obs_dir / 'candidates.tbl'}")
    logger.info(f"Report  : {obs_dir / 'sn_report.json'}")


if __name__ == '__main__':
    main()
