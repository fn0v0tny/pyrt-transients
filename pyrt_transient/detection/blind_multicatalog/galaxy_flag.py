"""Known-galaxy flag for the final GRB (blind_multicatalog) candidates.

A catalogued galaxy is "new" in ATLAS, Gaia and USNO-B alike: ATLAS and
Gaia leave extended sources out, and USNO-B's galaxy entry sits right
under the detection. Nothing else in this pipeline catches it -- the only
thing that used to was a side effect of the quality cut running before the
catalogue count, removed on purpose in ce2d550 because the same mechanism
also threw out SN 2026zji on its host galaxy (see clustering.combine_results).

So this is a label, not a rejection filter (same reasoning as the NED note
in stdpipe_filters.py): each final candidate is matched against extended
sources in DESI Legacy Survey DR10 (Tractor model types REX/EXP/DEV/SER)
and Pan-STARRS DR1 (PSF minus Kron magnitude above PS1_EXTENDED_MIN), and
the detected magnitude is compared with the galaxy's *total* magnitude
(Tractor model flux / Kron). An aperture measurement of a galaxy can only
come out fainter than its total, so

  galaxy_flag = 'galaxy'  matched, and not brighter than the galaxy by more
                          than host_margin_mag -- the catalogued galaxy itself
  galaxy_flag = 'host'    matched, but brighter than the whole galaxy --
                          something has been added on top: a transient in it
  galaxy_flag = 'none'    no extended source within match_radius_arcsec

Comparing against PSF magnitudes instead would make every galaxy look
about a magnitude too bright, i.e. 'host'.

Queries are per candidate cone (not per field: the field-level extended
source list is large and PS1 on VizieR can't select extended sources
server-side) and cached per transient_id in the observation directory, so
the pipeline's once-per-epoch reruns only query candidates that are new.
A failed query leaves those candidates 'none' and is not cached, so the
next run retries.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
from astropy.table import Table

logger = logging.getLogger("detection.blind_multicatalog.galaxy_flag")

BANDS = ("g", "r", "i", "z")
# PS1 star/galaxy separation: PSF minus Kron magnitude, in the band with
# the best S/N (r, falling back to i). 0.05 mag is the usual threshold
# (Farrow et al. 2014); real galaxies at r < 21 are typically well above it.
PS1_EXTENDED_MIN = 0.05
LS_EXTENDED_TYPES = ("REX", "EXP", "DEV", "SER")

CACHE_FILENAME = "galaxy_flag_cache.json"
QUERY_CHUNK = 100


def band_for_filter(phfilter: Optional[str]) -> str:
    """Survey band closest to an observation's PHFILTER ("Sloan_r",
    "Johnson_R", ...). Unknown or clear filters fall back to r, which is
    what the clear-filter epochs are calibrated to anyway."""
    if not phfilter:
        return "r"
    last = str(phfilter).strip()[-1:]
    return {"g": "g", "B": "g", "V": "g", "r": "r", "R": "r",
            "i": "i", "I": "i", "z": "z"}.get(last, "r")


def _ls_query(coords: List[tuple], radius_arcsec: float, timeout: float) -> List[dict]:
    import pyvo

    cones = " OR ".join(
        f"q3c_radial_query(ra, dec, {ra:.7f}, {dec:.7f}, {radius_arcsec / 3600.0:.7f}) = 't'"
        for ra, dec in coords
    )
    query = (
        "SELECT ra, dec, type, flux_g, flux_r, flux_i, flux_z FROM ls_dr10.tractor "
        f"WHERE type IN ({', '.join(repr(t) for t in LS_EXTENDED_TYPES)}) AND ({cones})"
    )
    tap = pyvo.dal.TAPService("https://datalab.noirlab.edu/tap")
    tap.timeout = timeout
    tab = tap.search(query, maxrec=100000).to_table()
    out = []
    for row in tab:
        mags = {}
        for b in BANDS:
            x = row[f"flux_{b}"]
            flux = np.nan if np.ma.is_masked(x) else float(x)
            mags[b] = 22.5 - 2.5 * np.log10(flux) if np.isfinite(flux) and flux > 0 else None
        out.append({"ra": float(row["ra"]), "dec": float(row["dec"]),
                    "catalog": "legacysurvey", "mags": mags})
    return out


def _ps1_query(coords: List[tuple], radius_arcsec: float, timeout: float) -> List[dict]:
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astroquery.vizier import Vizier

    cols = ["RAJ2000", "DEJ2000"] + [f"{b}mag" for b in BANDS] + [f"{b}Kmag" for b in BANDS]
    v = Vizier(columns=cols, row_limit=-1, timeout=timeout)
    ra, dec = zip(*coords)
    res = v.query_region(SkyCoord(list(ra), list(dec), unit="deg"),
                         radius=radius_arcsec * u.arcsec, catalog="II/349/ps1")
    if not res:
        return []
    tab = res[0]

    def val(row, col):
        x = row[col]
        return None if np.ma.is_masked(x) or not np.isfinite(float(x)) else float(x)

    out = []
    for row in tab:
        diff = None
        for b in ("r", "i"):
            psf, kron = val(row, f"{b}mag"), val(row, f"{b}Kmag")
            if psf is not None and kron is not None:
                diff = psf - kron
                break
        if diff is None or diff <= PS1_EXTENDED_MIN:
            continue
        out.append({"ra": float(row["RAJ2000"]), "dec": float(row["DEJ2000"]),
                    "catalog": "panstarrs",
                    "mags": {b: val(row, f"{b}Kmag") for b in BANDS}})
    return out


QUERY_FUNCTIONS = {"legacysurvey": _ls_query, "panstarrs": _ps1_query}


def _query_extended(coords: List[tuple], catalogs: Iterable[str], radius_arcsec: float,
                    timeout: float, log) -> Optional[List[dict]]:
    """Extended sources around `coords` from every catalogue that answers.
    Returns None only if every catalogue that was asked failed -- an empty
    list from a working service is a real "no galaxies here". Pan-STARRS
    is not asked south of its -30 deg footprint."""
    found, n_ok, n_asked = [], 0, 0
    for name in catalogs:
        fn = QUERY_FUNCTIONS.get(name)
        if fn is None:
            log.warning(f"Galaxy flag: unknown catalogue '{name}', skipping")
            continue
        use = coords if name != "panstarrs" else [c for c in coords if c[1] > -30.0]
        if not use:
            continue
        n_asked += 1
        try:
            # Chunked: one OR-ed cone per candidate makes a long query.
            rows = []
            for k in range(0, len(use), QUERY_CHUNK):
                rows.extend(fn(use[k:k + QUERY_CHUNK], radius_arcsec, timeout))
            n_ok += 1
            found.extend(rows)
            log.info(f"Galaxy flag: {name} returned {len(rows)} extended source(s) "
                     f"around {len(use)} candidate(s)")
        except Exception as e:
            log.warning(f"Galaxy flag: {name} query failed ({e})")
    if n_asked and not n_ok:
        return None
    return found


def _read_cache(path: Path) -> Dict[str, list]:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _write_cache(path: Path, cache: Dict[str, list], log) -> None:
    try:
        with open(path, "w") as f:
            json.dump(cache, f)
    except OSError as e:
        log.debug(f"Galaxy flag: could not write {path}: {e}")


def _sep_arcsec(ra1, dec1, ra2, dec2) -> float:
    d1, d2 = np.radians(dec1), np.radians(dec2)
    c = (np.sin(d1) * np.sin(d2)
         + np.cos(d1) * np.cos(d2) * np.cos(np.radians(ra1 - ra2)))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))) * 3600.0)


def flag_known_galaxies(candidates: Table, band: str = "r", catalogs=("legacysurvey", "panstarrs"),
                        match_radius_arcsec: float = 2.0, host_margin_mag: float = 0.75,
                        cache_dir: Optional[Path] = None, timeout: float = 60.0,
                        log=None) -> Table:
    """Add galaxy_flag / galaxy_sep_arcsec / galaxy_mag / galaxy_catalog
    columns in place (see module docstring) and return the table. Never
    removes a row."""
    log = log or logger
    n = len(candidates)
    flags = ["none"] * n
    seps = np.full(n, np.nan)
    gmags = np.full(n, np.nan)
    gcats = ["none"] * n

    if n:
        ids = [str(t) for t in candidates["transient_id"]]
        ras = np.array(candidates["ALPHA_J2000"], dtype=float)
        decs = np.array(candidates["DELTA_J2000"], dtype=float)
        cache_path = Path(cache_dir) / CACHE_FILENAME if cache_dir is not None else None
        cache = _read_cache(cache_path) if cache_path else {}

        todo = [i for i in range(n) if ids[i] not in cache]
        if todo:
            found = _query_extended([(ras[i], decs[i]) for i in todo], catalogs,
                                    match_radius_arcsec, timeout, log)
            if found is not None:
                for i in todo:
                    cache[ids[i]] = [
                        s for s in found
                        if _sep_arcsec(ras[i], decs[i], s["ra"], s["dec"]) <= match_radius_arcsec
                    ]
                if cache_path:
                    _write_cache(cache_path, cache, log)

        mags = (np.array(candidates["MAG_CALIB"], dtype=float)
                if "MAG_CALIB" in candidates.colnames else np.full(n, np.nan))
        for i in range(n):
            matches = cache.get(ids[i]) or []
            if not matches:
                continue
            d = [_sep_arcsec(ras[i], decs[i], s["ra"], s["dec"]) for s in matches]
            k = int(np.argmin(d))
            src = matches[k]
            # Band as asked, else r, else whatever the catalogue has.
            gmag = next((src["mags"].get(b) for b in (band, "r", *BANDS)
                         if src["mags"].get(b) is not None), None)
            seps[i] = d[k]
            gcats[i] = src["catalog"]
            if gmag is None:
                flags[i] = "galaxy"
                continue
            gmags[i] = gmag
            brighter = np.isfinite(mags[i]) and mags[i] < gmag - host_margin_mag
            flags[i] = "host" if brighter else "galaxy"

        n_gal = flags.count("galaxy")
        n_host = flags.count("host")
        log.info(f"Galaxy flag: {n_gal} catalogued galaxies, {n_host} brighter than "
                 f"their galaxy (possible transient on host), {n - n_gal - n_host} unmatched")

    candidates["galaxy_flag"] = flags
    candidates["galaxy_sep_arcsec"] = seps
    candidates["galaxy_mag"] = gmags
    candidates["galaxy_catalog"] = gcats
    return candidates
