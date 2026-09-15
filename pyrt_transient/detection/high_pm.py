"""Final-candidate veto against high proper-motion stars.

The catalogue matcher already moves Gaia and ATLAS positions to the frame
epoch (catalog.py, positions_at_epoch), so a fast star normally matches
its own entry. This is the safety net for everything that path cannot
cover: a configuration without Gaia, a local ATLAS copy without the pm
columns, a star missing from pyrt's calibrator-style Gaia query (RUWE and
colour cuts drop ~9 % of stars), or a match radius tighter than the
residual. It queries Gaia DR3 through VizieR once per field (cached in the
observation directory), propagates the stars above the threshold with
astropy, and drops the final candidates that sit on one of them.

reject_high_pm_stars is the routine the subtraction pipeline has used as
its step 6; apply_high_pm_veto adapts it to the blind pipeline's final
clustered table.
"""
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from astropy.table import Table

HIGH_PM_CACHE_NAME = "gaia_highpm_cache.ecsv"


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

        # The cache holds whatever threshold wrote it; apply the current one
        # so a stricter setting is honoured without refetching (a looser one
        # needs the cache removed).
        if len(gaia_hp):
            pm_cached = np.hypot(np.nan_to_num(np.array(gaia_hp['pmRA'], dtype=float)),
                                 np.nan_to_num(np.array(gaia_hp['pmDE'], dtype=float)))
            gaia_hp = gaia_hp[pm_cached >= pm_threshold_masyr]
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
        if "NUMBER" in candidates.colnames:
            # The forced target row is the pointing; whoever asked for it
            # wants it reported even when it is a high-pm star.
            try:
                keep_mask |= np.asarray(candidates["NUMBER"], dtype=np.int64) == 0
            except (TypeError, ValueError):
                pass

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




def _field_geometry(detections: Table) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """(ra, dec, field_deg, obs_jd) of a detection table from its header,
    falling back to the detections themselves for the centre."""
    meta = detections.meta or {}

    def num(key):
        try:
            v = float(meta[key])
            return v if np.isfinite(v) else None
        except (KeyError, TypeError, ValueError):
            return None

    ra, dec = num("CRVAL1"), num("CRVAL2")
    if (ra is None or dec is None) and "ALPHA_J2000" in detections.colnames and len(detections):
        ra = float(np.nanmedian(np.asarray(detections["ALPHA_J2000"], float)))
        dec = float(np.nanmedian(np.asarray(detections["DELTA_J2000"], float)))
    field = num("FIELD")                     # pyrt: half-diagonal in degrees
    field_deg = 2.0 * field if field else None
    if field_deg is None:
        cd = num("CD1_1"); n = num("IMAGEW") or num("NAXIS1")
        field_deg = abs(cd) * n * 1.5 if cd and n else 0.5
    jd = num("JD")
    if jd is None:
        ctime = num("CTIME")
        jd = ctime / 86400.0 + 2440587.5 if ctime else None
    return ra, dec, field_deg, jd


def apply_high_pm_veto(candidates: Table, detection_tables, data_dir, config,
                       logger: Optional[logging.Logger] = None) -> Tuple[Table, int]:
    """Drop final candidates that sit on a propagated high-pm Gaia star.
    Returns (candidates, n_removed); on any failure the input is returned
    unchanged (this is a network step at the very end of a run)."""
    logger = logger or logging.getLogger("detection.blind_multicatalog")
    if len(candidates) == 0 or not detection_tables:
        return candidates, 0
    try:
        # The latest real epoch sets the time; stacks carry their inputs' times.
        tables = [t for t in detection_tables if not (t.meta or {}).get("IS_STACK")] or list(detection_tables)
        latest = max(tables, key=lambda t: float((t.meta or {}).get("JD", 0) or (t.meta or {}).get("CTIME", 0) or 0))
        ra, dec, field_deg, obs_jd = _field_geometry(latest)
        if ra is None or dec is None or obs_jd is None:
            logger.warning("High-pm veto: no field centre or epoch in the header, skipped")
            return candidates, 0
        det = config.detection
        return reject_high_pm_stars(
            candidates, ra, dec, field_deg, obs_jd, logger,
            pm_threshold_masyr=det.high_pm_threshold_masyr,
            match_radius_arcsec=det.high_pm_match_radius_arcsec,
            cache_path=Path(data_dir) / HIGH_PM_CACHE_NAME if data_dir else None,
        )
    except Exception as e:
        logger.warning(f"High-pm veto failed ({e!r}); keeping all {len(candidates)} candidates")
        return candidates, 0
