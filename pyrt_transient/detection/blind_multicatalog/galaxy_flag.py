"""Known-galaxy flag for the final GRB (blind_multicatalog) candidates.

A catalogued galaxy is "new" in ATLAS, Gaia and USNO-B alike: ATLAS and
Gaia leave extended sources out, and USNO-B's galaxy entry sits right
under the detection. Nothing else in this pipeline catches it -- the only
thing that used to was a side effect of the quality cut running before the
catalogue count, removed on purpose in ce2d550 because the same mechanism
also threw out SN 2026zji on its host galaxy (see clustering.combine_results).

Each final candidate is matched against extended sources in DESI Legacy
Survey DR10 (Tractor model types REX/EXP/DEV/SER, with their half-light
radius) and Pan-STARRS DR1 (PSF minus Kron magnitude above
PS1_EXTENDED_MIN), and the detected magnitude is compared with the galaxy's
*total* magnitude (Tractor model flux / Kron). MAG_CALIB is a Kron-like
(MAG_AUTO) measurement, so on a galaxy it comes out at the total or a little
fainter (Kron loss, colour term); it cannot come out brighter unless
something has been added on top. So

  galaxy_flag = 'host'         brighter than the whole galaxy by more than
                               host_margin_mag -- a transient on its host
  galaxy_flag = 'galaxy'       within [total - host_margin, total + faint_margin]:
                               the catalogued galaxy's own light
  galaxy_flag = 'near_galaxy'  matched, but fainter than the galaxy by more
                               than faint_margin_mag -- a separate source
                               projected on it (an unresolved galaxy's own
                               light would have been measured bright, not
                               faint). Only for galaxies no larger than the
                               seeing disc: a Kron aperture on a resolved
                               galaxy, with the background mesh eating its
                               wings, can come out fainter than the total
                               by any amount (EP261006 on D50: a 13.8 mag
                               galaxy of 8.9" half-light radius measured
                               at 15.1), so there the faint side is open.
  galaxy_flag = 'none'         no extended source within the match radius

The match radius is match_radius_arcsec, widened to radius_scale x the
galaxy's half-light radius where the catalogue gives one: SExtractor's
barycentre of a resolved, lumpy galaxy lands arcseconds from the Tractor
centre, and a 2" cone then misses exactly the galaxies that clutter a
field most. Comparing against PSF magnitudes instead would make every
galaxy look about a magnitude too bright, i.e. 'host'.

flag_known_galaxies only labels. remove_constant_galaxies, run right after
it in clustering, drops the 'galaxy' rows whose lightcurve is also
consistent with a constant source (mag_chi2_reduced at or below
max_chi2): a galaxy is constant, a transient on it brightens or fades.
The trade-off is explicit: a transient fainter than about faint_margin of
its host's light, sitting on the nucleus and not varying within the run,
is the galaxy as far as this test can tell; only image subtraction
separates those. 'host' and 'near_galaxy' rows are never removed.

Queries are per candidate cone (not per field: the field-level extended
source list is large and PS1 on VizieR can't select extended sources
server-side) and cached per transient_id in the observation directory, so
the pipeline's once-per-epoch reruns only query candidates that are new.
The cache records the cone radius it was built with and is discarded when
that changes. A failed query leaves those candidates 'none' and is not
cached, so the next run retries.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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
CACHE_META_KEY = "__meta__"
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
        "SELECT ra, dec, type, shape_r, flux_g, flux_r, flux_i, flux_z FROM ls_dr10.tractor "
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
        r = row["shape_r"]
        radius = None if np.ma.is_masked(r) or not np.isfinite(float(r)) or float(r) <= 0 else float(r)
        out.append({"ra": float(row["ra"]), "dec": float(row["dec"]),
                    "catalog": "legacysurvey", "mags": mags, "radius_arcsec": radius})
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
                    "mags": {b: val(row, f"{b}Kmag") for b in BANDS},
                    "radius_arcsec": None})
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


def _read_cache(path: Path, query_radius_arcsec: float, log) -> Dict[str, list]:
    """The cached per-candidate source lists, or {} if there is no cache or
    it was built with a different cone radius (its lists would be
    incomplete for a wider one)."""
    try:
        with open(path) as f:
            cache = json.load(f)
    except (OSError, ValueError):
        return {}
    meta = cache.pop(CACHE_META_KEY, None) or {}
    cached_radius = meta.get("query_radius_arcsec")
    if cached_radius is None or abs(float(cached_radius) - query_radius_arcsec) > 1e-6:
        log.info(f"Galaxy flag: cache at {path} was built with a {cached_radius}\" cone, "
                 f"now {query_radius_arcsec:g}\"; discarding it")
        return {}
    return cache


def _write_cache(path: Path, cache: Dict[str, list], query_radius_arcsec: float, log) -> None:
    try:
        with open(path, "w") as f:
            json.dump({CACHE_META_KEY: {"query_radius_arcsec": query_radius_arcsec}, **cache}, f)
    except OSError as e:
        log.debug(f"Galaxy flag: could not write {path}: {e}")


def _sep_arcsec(ra1, dec1, ra2, dec2) -> float:
    d1, d2 = np.radians(dec1), np.radians(dec2)
    c = (np.sin(d1) * np.sin(d2)
         + np.cos(d1) * np.cos(d2) * np.cos(np.radians(ra1 - ra2)))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))) * 3600.0)


def _match_radius(src: dict, match_radius_arcsec: float, radius_scale: float,
                  query_radius_arcsec: float) -> float:
    """How far from this galaxy's centre a detection still counts as it."""
    r = src.get("radius_arcsec")
    if r is None or radius_scale <= 0:
        return match_radius_arcsec
    return min(max(match_radius_arcsec, radius_scale * float(r)), query_radius_arcsec)


DEFAULT_SEEING_ARCSEC = 3.0


def flag_known_galaxies(candidates: Table, band: str = "r", catalogs=("legacysurvey", "panstarrs"),
                        match_radius_arcsec: float = 2.0, host_margin_mag: float = 0.75,
                        faint_margin_mag: float = 1.0, radius_scale: float = 1.0,
                        query_radius_arcsec: Optional[float] = None,
                        seeing_arcsec: Optional[float] = None,
                        cache_dir: Optional[Path] = None, timeout: float = 60.0,
                        log=None) -> Table:
    """Add galaxy_flag / galaxy_sep_arcsec / galaxy_mag / galaxy_dmag /
    galaxy_radius_arcsec / galaxy_catalog columns in place (see module
    docstring) and return the table. Never removes a row.

    query_radius_arcsec is the cone each candidate is queried with (the
    widest a galaxy can be matched at); it defaults to match_radius_arcsec
    when radius_scale is 0 and to 5 x match_radius_arcsec otherwise.
    seeing_arcsec (the frames' FWHM; DEFAULT_SEEING_ARCSEC if unknown)
    decides which galaxies count as resolved, i.e. get no faint-side test."""
    log = log or logger
    if seeing_arcsec is None or not np.isfinite(seeing_arcsec) or seeing_arcsec <= 0:
        seeing_arcsec = DEFAULT_SEEING_ARCSEC
    if query_radius_arcsec is None:
        query_radius_arcsec = match_radius_arcsec if radius_scale <= 0 else 5.0 * match_radius_arcsec
    query_radius_arcsec = max(float(query_radius_arcsec), float(match_radius_arcsec))
    n = len(candidates)
    flags = ["none"] * n
    seps = np.full(n, np.nan)
    gmags = np.full(n, np.nan)
    dmags = np.full(n, np.nan)
    radii = np.full(n, np.nan)
    gcats = ["none"] * n

    if n:
        ids = [str(t) for t in candidates["transient_id"]]
        ras = np.array(candidates["ALPHA_J2000"], dtype=float)
        decs = np.array(candidates["DELTA_J2000"], dtype=float)
        cache_path = Path(cache_dir) / CACHE_FILENAME if cache_dir is not None else None
        cache = _read_cache(cache_path, query_radius_arcsec, log) if cache_path else {}

        todo = [i for i in range(n) if ids[i] not in cache]
        if todo:
            found = _query_extended([(ras[i], decs[i]) for i in todo], catalogs,
                                    query_radius_arcsec, timeout, log)
            if found is not None:
                for i in todo:
                    cache[ids[i]] = [
                        s for s in found
                        if _sep_arcsec(ras[i], decs[i], s["ra"], s["dec"]) <= query_radius_arcsec
                    ]
                if cache_path:
                    _write_cache(cache_path, cache, query_radius_arcsec, log)

        mags = (np.ma.filled(np.ma.asarray(candidates["MAG_CALIB"], dtype=float), np.nan)
                if "MAG_CALIB" in candidates.colnames else np.full(n, np.nan))
        for i in range(n):
            matches = []
            for s in cache.get(ids[i]) or []:
                d = _sep_arcsec(ras[i], decs[i], s["ra"], s["dec"])
                if d <= _match_radius(s, match_radius_arcsec, radius_scale, query_radius_arcsec):
                    matches.append((d, s))
            if not matches:
                continue
            # Nearest in units of its own match radius: a candidate 3" from
            # a 10" galaxy is that galaxy, not the 2" one 2.5" away.
            d, src = min(matches, key=lambda m: m[0] / _match_radius(
                m[1], match_radius_arcsec, radius_scale, query_radius_arcsec))
            # Band as asked, else r, else whatever the catalogue has.
            gmag = next((src["mags"].get(b) for b in (band, "r", *BANDS)
                         if src["mags"].get(b) is not None), None)
            seps[i] = d
            gcats[i] = src["catalog"]
            if src.get("radius_arcsec") is not None:
                radii[i] = float(src["radius_arcsec"])
            if gmag is None:
                flags[i] = "galaxy"
                continue
            gmags[i] = gmag
            if not np.isfinite(mags[i]):
                flags[i] = "galaxy"
                continue
            dmags[i] = mags[i] - gmag
            resolved = np.isfinite(radii[i]) and radii[i] > seeing_arcsec
            if dmags[i] < -host_margin_mag:
                flags[i] = "host"
            elif dmags[i] > faint_margin_mag and not resolved:
                flags[i] = "near_galaxy"
            else:
                flags[i] = "galaxy"

        n_gal = flags.count("galaxy")
        n_host = flags.count("host")
        n_near = flags.count("near_galaxy")
        log.info(f"Galaxy flag: {n_gal} catalogued galaxies, {n_host} brighter than "
                 f"their galaxy (possible transient on host), {n_near} fainter sources "
                 f"projected on a galaxy, {n - n_gal - n_host - n_near} unmatched")

    candidates["galaxy_flag"] = flags
    candidates["galaxy_sep_arcsec"] = seps
    candidates["galaxy_mag"] = gmags
    candidates["galaxy_dmag"] = dmags
    candidates["galaxy_radius_arcsec"] = radii
    candidates["galaxy_catalog"] = gcats
    return candidates


def remove_constant_galaxies(candidates: Table, max_chi2: float = 2.0,
                             keep_within: Optional[Tuple[float, float, float]] = None,
                             log=None) -> Tuple[Table, int]:
    """Drop the 'galaxy' rows whose lightcurve is consistent with a
    constant source: mag_chi2_reduced (errors with the campaign's
    empirical floor, see lightcurve.py) at or below max_chi2. A row with
    no chi2 column counts as constant (a single detection has nothing to
    vary). 'host', 'near_galaxy' and 'none' rows, and the forced target
    row (NUMBER 0, a real measurement at the pointing), are kept.

    keep_within = (ra_deg, dec_deg, radius_arcsec) keeps every galaxy
    inside that circle -- the trigger's error box. A galaxy there is a
    host candidate, which is what a GRB/EP follow-up wants to see:
    EP261006a's counterpart (a nuclear TDE, GCN 45831) sat on a 17.6 mag
    galaxy that this pipeline listed as a constant 'galaxy' on every one
    of the five D50 nights; nothing fainter than its host survives the
    single-night test, so the label plus position is the only handle.
    Returns (table, n_removed)."""
    log = log or logger
    n = len(candidates)
    if n == 0 or "galaxy_flag" not in candidates.colnames:
        return candidates, 0
    is_galaxy = np.array([str(f) == "galaxy" for f in candidates["galaxy_flag"]], dtype=bool)
    if keep_within is not None:
        ra0, dec0, r0 = keep_within
        if all(np.isfinite(v) for v in (ra0, dec0, r0)) and r0 > 0:
            ras = np.ma.filled(np.ma.asarray(candidates["ALPHA_J2000"], dtype=float), np.nan)
            decs = np.ma.filled(np.ma.asarray(candidates["DELTA_J2000"], dtype=float), np.nan)
            inside = np.array([_sep_arcsec(ra0, dec0, ra, dec) <= r0 if np.isfinite(ra) and np.isfinite(dec) else False
                               for ra, dec in zip(ras, decs)], dtype=bool)
            n_in = int((is_galaxy & inside).sum())
            if n_in:
                log.info(f"Galaxy flag: {n_in} catalogued galaxies inside the {r0 / 60:.1f}' error circle kept as host candidates")
            is_galaxy &= ~inside
    if "mag_chi2_reduced" in candidates.colnames:
        chi2 = np.ma.filled(np.ma.asarray(candidates["mag_chi2_reduced"], dtype=float), np.nan)
        constant = ~np.isfinite(chi2) | (chi2 <= max_chi2)
    else:
        constant = np.ones(n, dtype=bool)
    forced = (np.ma.filled(np.ma.asarray(candidates["NUMBER"]), -1).astype(int) == 0
              if "NUMBER" in candidates.colnames else np.zeros(n, dtype=bool))
    drop = is_galaxy & constant & ~forced
    n_removed = int(drop.sum())
    if n_removed:
        n_var = int((is_galaxy & ~constant).sum())
        log.info(f"Galaxy flag: removed {n_removed} constant catalogued galaxies "
                 f"(chi2_reduced <= {max_chi2:g}); kept {n_var} variable one(s)")
        candidates = candidates[~drop]
    return candidates, n_removed
