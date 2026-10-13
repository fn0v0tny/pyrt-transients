"""Catalogued galaxies inside a trigger's error circle, and a prior on
which of them hosts the transient.

Legacy Survey DR10 (Tractor model type, half-light radius, model fluxes,
photometric redshift) where it has coverage, Pan-STARRS DR1 (PSF minus
Kron > PS1_EXTENDED_MIN, Kron magnitudes, no size, no redshift) elsewhere
north of -30 deg. The query helpers for Pan-STARRS are galaxy_flag.py's.
"""
import logging
import math
from typing import Dict, Iterable, List, Optional

import numpy as np

from pyrt_transient.detection.blind_multicatalog.galaxy_flag import BANDS, LS_EXTENDED_TYPES, _ps1_query

logger = logging.getLogger("detection.host_search.galaxies")


def _ls_cone_query(ra: float, dec: float, radius_arcsec: float, timeout: float) -> List[dict]:
    import pyvo

    query = (
        "SELECT t.ra, t.dec, t.type, t.shape_r, t.flux_g, t.flux_r, t.flux_i, t.flux_z, "
        "p.z_phot_median, p.z_phot_std "
        "FROM ls_dr10.tractor AS t LEFT JOIN ls_dr10.photo_z AS p ON t.ls_id = p.ls_id "
        f"WHERE q3c_radial_query(t.ra, t.dec, {ra:.7f}, {dec:.7f}, {radius_arcsec / 3600.0:.7f}) = 't' "
        f"AND t.type IN ({', '.join(repr(t) for t in LS_EXTENDED_TYPES)}) AND t.flux_r > 0"
    )
    tap = pyvo.dal.TAPService("https://datalab.noirlab.edu/tap")
    tap.timeout = timeout
    tab = tap.search(query, maxrec=100000).to_table()

    def val(x):
        return None if np.ma.is_masked(x) or not np.isfinite(float(x)) else float(x)

    out = []
    for row in tab:
        mags = {}
        for b in BANDS:
            f = val(row[f"flux_{b}"])
            mags[b] = 22.5 - 2.5 * math.log10(f) if f is not None and f > 0 else None
        r = val(row["shape_r"])
        out.append({"ra": float(row["ra"]), "dec": float(row["dec"]), "catalog": "legacysurvey",
                    "type": str(row["type"]), "radius_arcsec": r if r is not None and r > 0 else None,
                    "mags": mags, "zphot": val(row["z_phot_median"]), "zphot_err": val(row["z_phot_std"])})
    return out


def query_galaxies_in_circle(ra: float, dec: float, radius_arcsec: float,
                             catalogs: Iterable[str] = ("legacysurvey", "panstarrs"),
                             max_mag: float = 21.0, band: str = "r", timeout: float = 120.0,
                             log=None) -> Optional[List[dict]]:
    """Extended sources within `radius_arcsec` of (ra, dec) brighter than
    `max_mag` in `band`, from the first catalogue in `catalogs` that has
    coverage there. Returns None if every catalogue asked failed (as
    opposed to an empty list from a catalogue that answered)."""
    log = log or logger
    n_failed = n_asked = 0
    for name in catalogs:
        if name == "panstarrs" and dec <= -30.0:
            continue
        n_asked += 1
        try:
            if name == "legacysurvey":
                found = _ls_cone_query(ra, dec, radius_arcsec, timeout)
            elif name == "panstarrs":
                found = _ps1_query([(ra, dec)], radius_arcsec, timeout)
                for g in found:
                    g.setdefault("type", "extended")
                    g.setdefault("zphot", None)
                    g.setdefault("zphot_err", None)
            else:
                log.warning(f"Host search: unknown galaxy catalogue '{name}', skipping")
                n_asked -= 1
                continue
        except Exception as e:
            n_failed += 1
            log.warning(f"Host search: {name} galaxy query failed ({e})")
            continue
        kept = []
        for g in found:
            m = g["mags"].get(band)
            if m is None:
                m = next((g["mags"].get(b) for b in BANDS if g["mags"].get(b) is not None), None)
            if m is None or m > max_mag:
                continue
            g["mag"] = float(m)
            kept.append(g)
        if kept or name == "legacysurvey" and found:
            log.info(f"Host search: {len(kept)} galaxies brighter than {max_mag:g} within "
                     f"{radius_arcsec / 60:.1f}' from {name}")
            return kept
        log.info(f"Host search: {name} has no galaxies here, trying the next catalogue")
    if n_asked and n_failed == n_asked:
        return None
    return []


def galaxy_surface_density(mag: float) -> float:
    """Cumulative number of galaxies brighter than `mag` (R band) per
    square arcsecond -- Bloom, Kulkarni & Djorgovski 2002, eq. 3."""
    return 10 ** (0.33 * (mag - 24.0) - 2.44) / (0.33 * math.log(10.0))


def rank_by_prior(galaxies: List[dict], radius_arcsec: float) -> List[dict]:
    """Order galaxies by how many galaxies at least this bright a random
    circle of this radius would contain (`n_exp`); fewer is a stronger
    host candidate. Adds `n_exp`, `p_chance` (1 - exp(-n_exp)) and
    `prior_rank` (1 = best) in place and returns the sorted list. With a
    3' circle n_exp is about 4 even for a 17.7 mag galaxy, so this orders
    the list rather than picking one: it is the prior the residual
    measurement is combined with."""
    area = math.pi * radius_arcsec ** 2
    for g in galaxies:
        g["n_exp"] = area * galaxy_surface_density(float(g["mag"]))
        g["p_chance"] = 1.0 - math.exp(-g["n_exp"])
    galaxies.sort(key=lambda g: g["n_exp"])
    for i, g in enumerate(galaxies, 1):
        g["prior_rank"] = i
    return galaxies


def sep_arcsec(ra1, dec1, ra2, dec2) -> float:
    d1, d2 = np.radians(dec1), np.radians(dec2)
    c = np.sin(d1) * np.sin(d2) + np.cos(d1) * np.cos(d2) * np.cos(np.radians(ra1 - ra2))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))) * 3600.0)
