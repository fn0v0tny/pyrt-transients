#!/usr/bin/python3
"""
Transient-detection extensions for the pyrt Catalog class.

Provides:
  CatalogCache             — disk-based cache for remote catalog queries
  CatalogOptimizationCache — per-instance precomputed photometric data
  filter_vsx_variables     — remove known VSX variables from candidate list
  CatTransients            — pyrt.catalog.Catalog subclass with all
                             transient-detection methods
  Catalog                  — backward-compatible alias for CatTransients
"""

import hashlib
import json
import logging
import os
import pickle
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import astropy.table
import astropy.units as u
import astropy.wcs
import numpy as np
from astropy.coordinates import SkyCoord
from sklearn.neighbors import KDTree

from pyrt_transient.core.color_model import has_colour_terms, simple_color_model
from pyrt_transient.core.radii import scaled_position_error

# ---------------------------------------------------------------------------
# Import base class from pyrt.  Re-export QueryParams / CatalogFilter so that
# callers can do `from pyrt_transient.catalog import QueryParams` without
# knowing about the upstream package layout.
# ---------------------------------------------------------------------------
try:
    from pyrt.catalog.catalog import (
        Catalog as _PyrtCatalog,
        CatalogFilter,
        CatalogFilters,
        QueryParams,
    )
except ImportError as _err:
    raise ImportError(
        "pyrt must be installed to use pyrt-transient.\n"
        "  Install it with:  pip install pyrt\n"
        f"  (original error: {_err})"
    ) from _err

# Type alias kept for internal use
CatalogConfig = Dict[str, Any]
FilterDict = Dict[str, CatalogFilter]


class CatalogNoCoverageError(ValueError):
    """The catalogue simply has no rows for this field.

    Distinct from a download that blew up: no coverage is a property of the
    field that will be just as true next run, so a caller can exclude the
    catalogue without marking the epoch degraded and re-analysing the whole
    campaign every time.  Kept a ValueError so the plain
    ``No data retrieved from ...`` handlers this replaced still catch it.
    """


# ---------------------------------------------------------------------------
# Per-instance optimisation cache (precomputed photometry + spatial indices)
# ---------------------------------------------------------------------------

@dataclass
class CatalogOptimizationCache:
    """Cache for precomputed catalog data to avoid repeated calculations."""
    coordinates: np.ndarray
    pixel_coordinates: Dict[str, np.ndarray]   # keyed by image identifier
    magnitudes: np.ndarray
    colors: np.ndarray
    valid_stars: np.ndarray
    kdtrees: Dict[str, KDTree]                 # cached KDTrees per image
    # Best available single magnitude per star from whatever the catalogue
    # carries (Sloan r, USNO-B R2/R1, Gaia G, ...), NaN if none: used only
    # to decide whether a positional match without usable Sloan photometry
    # can plausibly be the detected source (see
    # DetectionConfig.unphotometered_veto_max_brightening_mag).
    rough_mags: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Disk-based catalog cache (shared across instances)
# ---------------------------------------------------------------------------

class CatalogCache:
    """Disk cache for remote catalog queries with spatial grid binning."""

    # Pointings within CACHE_GRID_DEG share one cache entry.  The query box
    # is padded by this amount so the cached data always covers the footprint.
    CACHE_GRID_DEG = 1.0
    # Entries older than this are refreshed (see load_from_cache).
    MAX_AGE_DAYS = 30.0

    def __init__(self, cache_dir: str = "./catalog_cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        for name in ("panstarrs", "panstarrs@vizier", "gaia", "gaia_full", "atlas_vizier",
                     "atlas@vizier", "usno", "vsx", "legacysurvey"):
            (self.cache_dir / name).mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Key generation
    # ------------------------------------------------------------------

    def _generate_cache_key(self, catalog_name: str, params: QueryParams) -> str:
        """Snap RA/Dec to a coarse grid so nearby pointings share one key."""
        grid = self.CACHE_GRID_DEG
        if params.ra is not None and params.dec is not None:
            snapped_ra  = round(round(params.ra  / grid) * grid, 4)
            snapped_dec = round(round(params.dec / grid) * grid, 4)
        else:
            snapped_ra = snapped_dec = None

        key_data = {
            "catalog": catalog_name,
            "ra":      snapped_ra,
            "dec":     snapped_dec,
            "width":   round(params.width  + grid, 1),
            "height":  round(params.height + grid, 1),
            "mlim":    round(params.mlim,  1),
        }
        return hashlib.md5(str(sorted(key_data.items())).encode()).hexdigest()[:16]

    def get_cache_path(self, catalog_name: str, params: QueryParams) -> Path:
        key = self._generate_cache_key(catalog_name, params)
        (self.cache_dir / catalog_name).mkdir(parents=True, exist_ok=True)
        return self.cache_dir / catalog_name / f"{key}.pkl"

    def is_cached(self, catalog_name: str, params: QueryParams) -> bool:
        return self.get_cache_path(catalog_name, params).exists()

    # ------------------------------------------------------------------
    # Load / save
    # ------------------------------------------------------------------

    def load_from_cache(
        self, catalog_name: str, params: QueryParams, allow_stale: bool = False
    ) -> Optional[astropy.table.Table]:
        """The cached table, or None.

        An entry older than MAX_AGE_DAYS stays on disk: the refresh
        overwrites it, and if the refresh fails it is still returned with
        allow_stale. That fallback is used during a catalogue-server outage.
        Deleting the entry up front meant a field lost its catalogue to an
        outage (Gaia TAP returning 500 on lascaux50, 2026-09-11) even though
        month-old star positions would have done.
        """
        path = self.get_cache_path(catalog_name, params)
        if not path.exists():
            return None
        try:
            with open(path, "rb") as fh:
                cached = pickle.load(fh)
            if isinstance(cached, dict) and "data" in cached and "timestamp" in cached:
                age_days = (time.time() - cached["timestamp"]) / 86400
                if age_days < self.MAX_AGE_DAYS:
                    logging.info(
                        f"Loaded {catalog_name} from cache (age: {age_days:.1f} d)"
                    )
                    return cached["data"]
                if allow_stale:
                    logging.warning(
                        f"Using the expired {catalog_name} cache ({age_days:.1f} d): "
                        f"the refresh failed"
                    )
                    return cached["data"]
                logging.info(
                    f"Cache for {catalog_name} expired ({age_days:.1f} d), refreshing"
                )
                return None
        except Exception as exc:
            logging.info(f"Failed to load cache for {catalog_name}: {exc}")
            try:
                path.unlink()
            except Exception:
                pass
        return None

    def save_to_cache(
        self,
        catalog_name: str,
        params: QueryParams,
        data: astropy.table.Table,
    ) -> None:
        path = self.get_cache_path(catalog_name, params)
        try:
            with open(path, "wb") as fh:
                pickle.dump(
                    {"data": data, "timestamp": time.time(), "params": vars(params)},
                    fh,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            logging.info(f"Cached {catalog_name} → {path}")
        except Exception as exc:
            logging.info(f"Failed to save cache for {catalog_name}: {exc}")

    def clear_cache(
        self,
        catalog_name: Optional[str] = None,
        max_age_days: Optional[float] = None,
    ) -> None:
        dirs = (
            [self.cache_dir / catalog_name]
            if catalog_name
            else [d for d in self.cache_dir.iterdir() if d.is_dir()]
        )
        for d in dirs:
            if not d.exists():
                continue
            for f in d.glob("*.pkl"):
                remove = True
                if max_age_days is not None:
                    try:
                        age = (time.time() - f.stat().st_mtime) / 86400
                        remove = age > max_age_days
                    except Exception:
                        pass
                if remove:
                    try:
                        f.unlink()
                        logging.info(f"Removed cache: {f}")
                    except Exception as exc:
                        logging.info(f"Could not remove {f}: {exc}")

    def get_cache_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {}
        for d in self.cache_dir.iterdir():
            if not d.is_dir():
                continue
            files = list(d.glob("*.pkl"))
            info[d.name] = {
                "num_files":     len(files),
                "total_size_mb": sum(f.stat().st_size for f in files) / (1024 ** 2),
                "files": [],
            }
            for f in files:
                try:
                    info[d.name]["files"].append(
                        {
                            "name":     f.name,
                            "age_days": (time.time() - f.stat().st_mtime) / 86400,
                            "size_kb":  f.stat().st_size / 1024,
                        }
                    )
                except Exception:
                    pass
        return info

    # ------------------------------------------------------------------
    # VSX queries (cached)
    # ------------------------------------------------------------------

    def query_vsx(
        self,
        coords: SkyCoord,
        radius_arcsec: float = 2.5,
        catalog_id: str = "B/vsx/vsx",
    ) -> Optional[astropy.table.Table]:
        """Query VSX (Variable Star Index) with disk-level caching."""
        try:
            from astroquery.vizier import Vizier

            cache_params = QueryParams(
                ra=coords.ra.deg,
                dec=coords.dec.deg,
                width=radius_arcsec / 3600.0,
                height=radius_arcsec / 3600.0,
            )
            cached = self.load_from_cache("vsx", cache_params)
            if cached is not None:
                return cached

            vizier = Vizier(columns=["*"], row_limit=-1, timeout=60)
            logging.info(
                f"Querying VSX {catalog_id} at "
                f"{coords.ra.deg:.4f}, {coords.dec.deg:.4f}, r={radius_arcsec}\""
            )
            result = vizier.query_region(
                coords, radius=radius_arcsec * u.arcsec, catalog=[catalog_id]
            )
            if not result or len(result) == 0:
                empty: astropy.table.Table = astropy.table.Table()
                self.save_to_cache("vsx", cache_params, empty)
                return empty

            vsx_table = result[0]
            logging.info(f"Found {len(vsx_table)} VSX sources")
            self.save_to_cache("vsx", cache_params, vsx_table)
            return vsx_table

        except Exception as exc:
            logging.warning(f"VSX query failed: {exc}")
            return None

    def query_vsx_region(
        self,
        ra_deg: float,
        dec_deg: float,
        radius_arcsec: float = 2.5,
        catalog_id: str = "B/vsx/vsx",
    ) -> Optional[astropy.table.Table]:
        coords = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
        return self.query_vsx(coords, radius_arcsec, catalog_id)


# ---------------------------------------------------------------------------
# VSX filtering helper (standalone — no Catalog instance required)
# ---------------------------------------------------------------------------

def filter_vsx_variables(
    candidates: astropy.table.Table,
    cache: CatalogCache,
    match_radius_arcsec: float = 2.5,
    catalog_id: str = "B/vsx/vsx",
) -> Tuple[astropy.table.Table, List[Dict]]:
    """Remove candidates that match known VSX variable stars.

    Returns
    -------
    filtered_candidates : Table
        Candidates with VSX variables removed.
    vsx_matches : list of dict
        Details about every removed candidate.
    """
    if len(candidates) == 0:
        return candidates, []

    ra_col = dec_col = None
    for col in candidates.colnames:
        cl = col.lower()
        if cl in ("ra", "radeg", "ra_deg", "_ra", "alpha_j2000"):
            ra_col = col
        elif cl in ("dec", "decdeg", "dec_deg", "_dec", "delta_j2000"):
            dec_col = col

    if ra_col is None or dec_col is None:
        logging.warning("VSX filter: cannot find RA/Dec columns in candidates")
        return candidates, []

    logging.info(f"VSX filtering {len(candidates)} candidates")

    ra_vals  = u.Quantity(candidates[ra_col],  u.deg, copy=False).to_value(u.deg)
    dec_vals = u.Quantity(candidates[dec_col], u.deg, copy=False).to_value(u.deg)

    pad = match_radius_arcsec / 3600.0
    width  = (np.max(ra_vals)  - np.min(ra_vals))  + 2 * pad
    height = (np.max(dec_vals) - np.min(dec_vals)) + 2 * pad
    region_radius_arcsec = np.sqrt(width ** 2 + height ** 2) / 2 * 3600

    center = SkyCoord(
        ra=np.mean(ra_vals) * u.deg,
        dec=np.mean(dec_vals) * u.deg,
        frame="icrs",
    )
    vsx_table = cache.query_vsx(center, region_radius_arcsec, catalog_id)

    if vsx_table is None or len(vsx_table) == 0:
        logging.info("No VSX sources found in candidate region")
        return candidates, []

    logging.info(f"Matching against {len(vsx_table)} VSX sources")

    cand_coords = SkyCoord(
        ra=u.Quantity(candidates[ra_col],  u.deg, copy=False),
        dec=u.Quantity(candidates[dec_col], u.deg, copy=False),
        frame="icrs",
    )

    vsx_ra_col = vsx_dec_col = None
    for col in vsx_table.colnames:
        cl = col.lower()
        if cl in ("ra", "raj2000", "_raj2000", "ra_deg", "alpha_j2000"):
            vsx_ra_col = col
        elif cl in ("dec", "dej2000", "_dej2000", "dec_deg", "delta_j2000"):
            vsx_dec_col = col

    if vsx_ra_col is None or vsx_dec_col is None:
        logging.warning("VSX filter: cannot find RA/Dec columns in VSX table")
        return candidates, []

    vsx_coords = SkyCoord(
        ra=u.Quantity(vsx_table[vsx_ra_col],  u.deg, copy=False),
        dec=u.Quantity(vsx_table[vsx_dec_col], u.deg, copy=False),
        frame="icrs",
    )

    idx, d2d, _ = cand_coords.match_to_catalog_sky(vsx_coords)
    matches = d2d < (match_radius_arcsec * u.arcsec)

    def _to_deg(val: Any) -> float:
        try:
            if hasattr(val, "to"):
                return float(val.to_value(u.deg))
        except Exception:
            pass
        try:
            return float(val)
        except Exception:
            return np.nan

    vsx_matches: List[Dict] = []
    for cand_i, vsx_i, sep in zip(
        np.where(matches)[0], idx[matches], d2d[matches]
    ):
        src = vsx_table[vsx_i]
        info: Dict[str, Any] = {
            "candidate_index":   int(cand_i),
            "vsx_index":         int(vsx_i),
            "separation_arcsec": float(sep.arcsec),
            "vsx_name": str(src["Name"]) if "Name" in vsx_table.colnames else "Unknown",
            "vsx_type": str(src["Type"]) if "Type" in vsx_table.colnames else "Unknown",
            "vsx_ra":  _to_deg(src[vsx_ra_col]),
            "vsx_dec": _to_deg(src[vsx_dec_col]),
            "candidate_ra":  _to_deg(candidates[ra_col][cand_i]),
            "candidate_dec": _to_deg(candidates[dec_col][cand_i]),
        }
        for mag_col in ("Vmag", "V", "mag"):
            if mag_col in vsx_table.colnames:
                try:
                    info[f"vsx_{mag_col.lower()}"] = float(src[mag_col])
                    break
                except Exception:
                    pass
        vsx_matches.append(info)

    filtered = candidates[~matches]
    logging.info(
        f"VSX: removed {int(np.sum(matches))} known variables, "
        f"{len(filtered)} candidates remain"
    )
    for m in vsx_matches:
        logging.debug(
            f"  Filtered {m['vsx_name']} ({m['vsx_type']}, "
            f"sep {m['separation_arcsec']:.2f}\")"
        )
    return filtered, vsx_matches


# ---------------------------------------------------------------------------
# Main subclass
# ---------------------------------------------------------------------------

_LEGACYSURVEY_FILTERS: FilterDict = {
    "Sloan_g": CatalogFilter("Sloan_g", 4810, "AB", "Sloan_g_err"),
    "Sloan_r": CatalogFilter("Sloan_r", 6170, "AB", "Sloan_r_err"),
    "Sloan_z": CatalogFilter("Sloan_z", 9100, "AB", "Sloan_z_err"),
}


class CatTransients(_PyrtCatalog):
    """pyrt Catalog extended with transient-detection capabilities.

    Adds
    ----
    * Disk-level catalog caching via CatalogCache
    * DESI Legacy Survey DR10 as an additional reference catalog
    * Precomputed photometric data and spatial indexing
    * Per-detection adaptive identification radii
    * Optimised transient-candidate detection with magnitude-change analysis
    """

    # Extra catalog identifiers not present in the base class
    LEGACYSURVEY: str = "legacysurvey"
    # Gaia DR3 without pyrt's calibrator quality cuts (ruwe < 1.4,
    # visibility_periods_used >= 8, BP/RP present). Those cuts are right for
    # picking photometric calibrators and wrong for vetting transients: on
    # the 210619B field they drop ~9% of real 12-17 mag stars, every one of
    # which then becomes a persistent "new" candidate. Stars without BP/RP
    # come back with NaN Sloan magnitudes (positionally present,
    # photometrically invalid -- see DetectionConfig.unphotometered_match_is_new).
    GAIA_FULL: str = "gaia_full"
    # Pan-STARRS DR1 mean photometry from VizieR (II/349/ps1): fast (a
    # 0.3 deg box in ~1 s), includes galaxies, no calibrator cuts. The MAST
    # DR2 route ("panstarrs") is deeper but slower; both are limited to
    # Dec > -30 and return None (-> "catalogue unavailable", excluded from
    # the agreement requirement) south of that without querying.
    PANSTARRS_VIZIER: str = "panstarrs@vizier"
    PS1_SOUTHERN_LIMIT_DEG: float = -30.0

    # Substrings of the parent's Gaia ADQL WHERE clause that implement the
    # calibrator cuts; _get_gaia_full_data drops any "AND ..." line containing one.
    _GAIA_QUALITY_CUT_TOKENS = ("ruwe", "visibility_periods_used", "IS NOT NULL",
                                "flux_over_error")

    # Extend parent's KNOWN_CATALOGS: mark remote catalogs as cacheable and
    # add the Legacy Survey entry.
    KNOWN_CATALOGS = {
        k: dict(v, cacheable=(not v.get("local", False)))
        for k, v in _PyrtCatalog.KNOWN_CATALOGS.items()
    }
    KNOWN_CATALOGS["gaia_full"] = dict(
        KNOWN_CATALOGS["gaia"],
        description="Gaia DR3, no calibrator quality cuts (positionally complete)",
    )
    KNOWN_CATALOGS["panstarrs@vizier"] = {
        "description": "Pan-STARRS DR1 mean photometry (VizieR II/349/ps1)",
        "filters":     {k: v for k, v in _PyrtCatalog.KNOWN_CATALOGS["atlas@vizier"]["filters"].items()
                        if k in ("Sloan_g", "Sloan_r", "Sloan_i", "Sloan_z")},
        "epoch":       2012.0,
        "local":       False,
        "service":     "VizieR",
        "catalog_id":  "II/349/ps1",
        "column_mapping": {
            "RAJ2000": "radeg", "DEJ2000": "decdeg",
            "gmag": "Sloan_g", "e_gmag": "Sloan_g_err",
            "rmag": "Sloan_r", "e_rmag": "Sloan_r_err",
            "imag": "Sloan_i", "e_imag": "Sloan_i_err",
            "zmag": "Sloan_z", "e_zmag": "Sloan_z_err",
            "ymag": "y", "e_ymag": "y_err",
            "Nd": "n_detections", "Qual": "quality_flag",
        },
        "cacheable":   True,
    }
    KNOWN_CATALOGS["legacysurvey"] = {
        "description": "DESI Legacy Imaging Survey DR10",
        "filters":     _LEGACYSURVEY_FILTERS,
        "epoch":       2017.0,
        "local":       False,
        "service":     "NOIRLab TAP",
        "catalog_id":  "ls_dr10.tractor",
        "cacheable":   True,
    }

    # Class-level disk cache (shared across all instances)
    _cache: Optional[CatalogCache] = None

    # ------------------------------------------------------------------
    # Class-level cache management
    # ------------------------------------------------------------------

    @classmethod
    def set_cache_directory(cls, cache_dir: str) -> None:
        """Point all instances to a new disk-cache directory."""
        cls._cache = CatalogCache(cache_dir)

    @classmethod
    def get_cache(cls) -> CatalogCache:
        if cls._cache is None:
            cls._cache = CatalogCache()
        return cls._cache

    @classmethod
    def clear_all_cache(cls, max_age_days: Optional[float] = None) -> None:
        cls.get_cache().clear_cache(max_age_days=max_age_days)

    @classmethod
    def get_cache_info(cls) -> Dict[str, Any]:
        return cls.get_cache().get_cache_info()

    # ------------------------------------------------------------------
    # Initialisation — add per-instance optimisation caches
    # ------------------------------------------------------------------

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._photometric_cache: Optional[CatalogOptimizationCache] = None
        self._coordinate_cache: Dict[str, np.ndarray] = {}
        self._kdtree_cache: Dict[str, KDTree] = {}
        self._cache_enabled: bool = True
        self._original_query_params: Optional[QueryParams] = None
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # Gaia Sloan-band conversion (Jordi et al. 2010)
    # ------------------------------------------------------------------

    @staticmethod
    def _gaia_to_sloan(
        G: np.ndarray, BP: np.ndarray, RP: np.ndarray
    ) -> tuple:
        """Convert Gaia G/BP/RP to approximate Sloan g and r (Jordi+2010)."""
        BP_RP = BP - RP
        valid = (
            np.isfinite(G) & np.isfinite(BP_RP)
            & (BP_RP > -0.5) & (BP_RP < 3.5)
        )
        sloan_g = np.full(len(G), np.nan)
        sloan_r = np.full(len(G), np.nan)
        x = BP_RP[valid]
        sloan_g[valid] = G[valid] + 0.1942 + 1.0448 * x + 0.0635 * x ** 2
        sloan_r[valid] = G[valid] - 0.1313 - 0.2085 * x
        return sloan_g, sloan_r

    def _get_gaia_data(self) -> Optional[astropy.table.Table]:
        """Fetch Gaia data and append Sloan_g/Sloan_r synthetic columns."""
        result = super()._get_gaia_data()
        if result is not None and "G" in result.colnames:
            for col in ("G", "BP", "RP"):
                # Masked (missing BP/RP) -> NaN, never a fill value.
                if hasattr(result[col], "filled"):
                    result[col] = np.asarray(result[col].filled(np.nan), dtype=np.float64)
            G  = np.array(result["G"],  dtype=np.float64)
            BP = np.array(result["BP"], dtype=np.float64)
            RP = np.array(result["RP"], dtype=np.float64)
            sloan_g, sloan_r = self._gaia_to_sloan(G, BP, RP)
            result["Sloan_g"] = sloan_g
            result["Sloan_r"] = sloan_r
        return result

    # ------------------------------------------------------------------
    # Pan-STARRS (Dec > -30 only)
    # ------------------------------------------------------------------

    @classmethod
    def ps1_covers(cls, dec_deg: float, height_deg: float = 0.0) -> bool:
        """True if any part of the query box lies north of the PS1 footprint edge."""
        return (dec_deg + height_deg / 2.0) > cls.PS1_SOUTHERN_LIMIT_DEG

    @staticmethod
    def _to_float_nan(col) -> np.ndarray:
        """Column -> float64 array with masked / sentinel (-999) values as NaN."""
        if hasattr(col, "mask"):
            # Convert BEFORE filling: filled(nan) raises on integer columns
            # (VizieR's Nd/Qual are masked int16).
            arr = np.asarray(col.data, dtype=np.float64)
            arr[np.asarray(col.mask, dtype=bool)] = np.nan
        else:
            arr = np.asarray(col, dtype=np.float64)
        arr = np.where(arr < -100, np.nan, arr)
        return arr

    @classmethod
    def _ps1_vizier_to_catalog(cls, table: "astropy.table.Table") -> "astropy.table.Table":
        """Map a VizieR II/349/ps1 table onto our column names (pure)."""
        mapping = cls.KNOWN_CATALOGS[cls.PANSTARRS_VIZIER]["column_mapping"]
        out = astropy.table.Table()
        for src, dst in mapping.items():
            if src in table.colnames:
                out[dst] = cls._to_float_nan(table[src])
        n = len(out) if out.colnames else 0
        for col in ("pmra", "pmdec", "parallax"):
            out[col] = np.zeros(n, dtype=np.float64)
        return out

    def _get_panstarrs_vizier_data(self) -> Optional["astropy.table.Table"]:
        """Pan-STARRS DR1 from VizieR. None (unavailable) south of Dec -30."""
        from astroquery.vizier import Vizier

        qp = self._query_params
        if not self.ps1_covers(qp.dec, qp.height):
            logging.warning(f"Pan-STARRS has no coverage at Dec {qp.dec:.2f} (limit "
                            f"{self.PS1_SOUTHERN_LIMIT_DEG}); catalogue unavailable for this field")
            return None
        mapping = self.KNOWN_CATALOGS[self.PANSTARRS_VIZIER]["column_mapping"]
        vizier = Vizier(columns=list(mapping.keys()),
                        column_filters={"rmag": f"<{qp.mlim}"},
                        # The cache-padded box (~1.3 deg) is ~60k rows; VizieR
                        # normally answers in ~6 s but occasionally stalls, and a
                        # timeout here only degrades to "catalogue unavailable".
                        row_limit=-1, timeout=max(300, int(qp.timeout or 60)))
        coords = SkyCoord(ra=qp.ra * u.deg, dec=qp.dec * u.deg, frame="icrs")
        result = vizier.query_region(coords, width=qp.width * u.deg, height=qp.height * u.deg,
                                     catalog=self.KNOWN_CATALOGS[self.PANSTARRS_VIZIER]["catalog_id"])
        if not result or len(result) == 0 or len(result[0]) == 0:
            logging.warning("No Pan-STARRS (VizieR) data found for this field")
            return None
        cat = self._ps1_vizier_to_catalog(result[0])
        logging.info(f"Pan-STARRS (VizieR): {len(cat)} sources")
        return cat

    _PS1_MAST_COLUMNS = ("objName", "raMean", "decMean", "nDetections", "qualityFlag",
                         "gMeanPSFMag", "gMeanPSFMagErr", "rMeanPSFMag", "rMeanPSFMagErr",
                         "iMeanPSFMag", "iMeanPSFMagErr", "zMeanPSFMag", "zMeanPSFMagErr",
                         "yMeanPSFMag", "yMeanPSFMagErr")

    def _get_panstarrs_data(self) -> Optional["astropy.table.Table"]:
        """Pan-STARRS DR2 from MAST, replacing pyrt's implementation.

        pyrt's version passes criteria as ``nDetections.gt`` etc., which the
        current MAST API rejects (``Filter 'nDetections.gt' does not
        exist``); without an explicit column list the response also fails
        to parse (``could not convert string to float: 'None'``). It then
        drops every star lacking any of the five bands, which is right for
        calibrators and wrong for vetting. This override uses the
        ``column=[(op, value)]`` syntax, requests only numeric columns,
        keeps incomplete stars (NaN bands), and adds ``Sloan_*`` aliases so
        the photometric cache can use PS1 magnitudes. Unavailable (None)
        south of Dec -30.
        """
        from astroquery.mast import Catalogs

        qp = self._query_params
        if not self.ps1_covers(qp.dec, qp.height):
            logging.warning(f"Pan-STARRS has no coverage at Dec {qp.dec:.2f}; catalogue unavailable")
            return None
        config = self.KNOWN_CATALOGS[self.PANSTARRS]
        radius = np.sqrt(qp.width ** 2 + qp.height ** 2) / 2
        coords = SkyCoord(ra=qp.ra * u.deg, dec=qp.dec * u.deg, frame="icrs")
        ps1 = Catalogs.query_region(
            coords, catalog=config["catalog_id"], radius=radius * u.deg,
            data_release=config.get("release", "dr2"), table=config.get("table", "mean"),
            columns=list(self._PS1_MAST_COLUMNS),
            nDetections=[("gt", 4)], rMeanPSFMag=[("lt", qp.mlim)], qualityFlag=[("lt", 128)],
        )
        if ps1 is None or len(ps1) == 0:
            logging.warning("No Pan-STARRS (MAST) data found for this field")
            return None
        result = astropy.table.Table()
        for ps1_name, our_name in config["column_mapping"].items():
            if ps1_name in ps1.colnames:
                result[our_name] = self._to_float_nan(ps1[ps1_name])
        for band in ("g", "r", "i", "z"):
            if band in result.colnames:
                result[f"Sloan_{band}"] = result[band]
            if f"d{band}" in result.colnames:
                result[f"Sloan_{band}_err"] = result[f"d{band}"]
        for col in ("pmra", "pmdec", "parallax"):
            result[col] = np.zeros(len(result), dtype=np.float64)
        logging.info(f"Pan-STARRS (MAST): {len(result)} sources")
        return result

    @classmethod
    def _strip_gaia_quality_cuts(cls, query: str) -> str:
        """Remove the calibrator-quality conditions from pyrt's Gaia ADQL."""
        kept = []
        for line in query.splitlines():
            stripped = line.strip()
            if stripped.startswith("AND ") and any(tok in stripped for tok in cls._GAIA_QUALITY_CUT_TOKENS):
                continue
            if stripped.startswith("--"):
                continue
            kept.append(line)
        return "\n".join(kept)

    def _get_gaia_full_data(self) -> Optional[astropy.table.Table]:
        """Gaia DR3 query without the calibrator quality cuts (GAIA_FULL).

        Reuses the parent's query construction and column mapping by
        intercepting the ADQL string on its way to astroquery and stripping
        the quality conditions, so the two variants cannot drift apart in
        anything but the WHERE clause.
        """
        from astroquery.gaia import Gaia

        original = Gaia.launch_job_async

        def launch(query, *args, **kwargs):
            return original(self._strip_gaia_quality_cuts(query), *args, **kwargs)

        Gaia.launch_job_async = launch
        try:
            return self._get_gaia_data()
        finally:
            Gaia.launch_job_async = original

    # ------------------------------------------------------------------
    # Override _fetch_catalog_data to add disk caching + Legacy Survey
    # ------------------------------------------------------------------

    # Queries that failed in this process: (catalogue, cache key) -> error.
    # A run recomputes every epoch still marked degraded. Without this, a
    # catalogue server that is down (Gaia TAP answering 500 after ~3 min on
    # lascaux50, 2026-09-11) was asked again for each of those epochs, so
    # run n cost n timeouts until the daemon's 900 s limit killed it. The
    # next process tries the server again.
    _failed_queries: dict = {}

    def _tag_table(self, table, config, **flags):
        table.meta.update(
            {
                "catalog":  self._catalog_name,
                "astepoch": config["epoch"],
                "filters":  list(config["filters"].keys()),
                **flags,
            }
        )
        return table

    def _stale_cache(self, params, config):
        """The field's expired disk-cache entry, or None."""
        if not config.get("cacheable", False):
            return None
        stale = self.get_cache().load_from_cache(self._catalog_name, params, allow_stale=True)
        return None if stale is None else self._tag_table(stale, config, cached=True, stale=True)

    # How long a failed query is remembered on disk, for every process.
    FAILURE_TTL_S = float(os.environ.get("PYRT_CATALOG_FAILURE_TTL_S", "900"))

    def _failure_path(self, params):
        cache = self.get_cache()
        try:
            key = cache._generate_cache_key(self._catalog_name, params)
        except AttributeError:   # a cache without keys (a stub in the tests)
            return None
        return Path(cache.cache_dir) / ".failed" / self._catalog_name / f"{key}.json"

    def recent_failure(self, params):
        """Why this field's query failed within FAILURE_TTL_S, or None.

        The answer is a file, so every pipeline process sees it. While a
        catalogue server is down (Gaia TAP for more than 12 h on
        2026-09-11/12), each frame otherwise waits out the timeout again --
        3 min a frame, and twice over 2 h.
        """
        path = self._failure_path(params)
        if path is None or self.FAILURE_TTL_S <= 0:
            return None
        try:
            entry = json.loads(path.read_text())
            if time.time() - float(entry["time"]) < self.FAILURE_TTL_S:
                return entry.get("reason", "")
            path.unlink()
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return None

    def _remember_failure(self, params, reason):
        path = self._failure_path(params)
        if path is None or self.FAILURE_TTL_S <= 0:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"time": time.time(), "reason": str(reason)[:500]}))
        except OSError:
            pass

    def _forget_failure(self, params):
        path = self._failure_path(params)
        try:
            if path is not None:
                path.unlink()
        except OSError:
            pass

    def _fetch_catalog_data(self) -> Optional[astropy.table.Table]:  # type: ignore[override]
        """Fetch catalog data, adding disk caching and Legacy Survey support.

        A query that fails, as opposed to one that finds no coverage, falls
        back to the field's expired cache entry if there is one. A query
        that already failed in this process is not sent again (see
        _failed_queries).
        """
        if self._catalog_name not in self.KNOWN_CATALOGS:
            raise ValueError(f"Unknown catalog: {self._catalog_name}")

        config = self.KNOWN_CATALOGS[self._catalog_name]
        cacheable = config.get("cacheable", False)
        # A copy: _widen_query_for_cache widens self._query_params in place,
        # and the disk cache is keyed on the caller's box.
        params = QueryParams(**{k: getattr(self._query_params, k)
                                for k in QueryParams.__dataclass_fields__})
        failure_key = (self._catalog_name, params.ra, params.dec,
                       params.width, params.height, params.mlim)

        # Try disk cache first
        if cacheable:
            cached = self.get_cache().load_from_cache(self._catalog_name, params)
            if cached is not None:
                return self._tag_table(cached, config, cached=True)

        # A failure without a message is remembered as "": still a failure.
        recent = self._failed_queries.get(failure_key)
        if recent is None:
            recent = self.recent_failure(params)
        if recent is not None:
            stale = self._stale_cache(params, config)
            if stale is not None:
                return stale
            raise RuntimeError(f"{self._catalog_name} query failed recently: {recent}")

        if cacheable:
            # Widen the query so nearby pointings get a cache hit
            self._widen_query_for_cache()

        try:
            try:
                if self._catalog_name == self.LEGACYSURVEY:
                    result = self._get_legacysurvey_data()
                elif self._catalog_name == self.GAIA_FULL:
                    result = self._get_gaia_full_data()
                elif self._catalog_name == self.PANSTARRS_VIZIER:
                    result = self._get_panstarrs_vizier_data()
                else:
                    # Parent handles ATLAS, PANSTARRS, GAIA, USNOB, SDSS, MAKAK
                    result = super()._fetch_catalog_data()
            except CatalogNoCoverageError:
                raise
            except Exception as exc:
                self._failed_queries[failure_key] = str(exc)
                self._remember_failure(params, exc)   # for the other processes too
                stale = self._stale_cache(params, config)
                if stale is not None:
                    return stale
                raise
            else:
                self._forget_failure(params)          # the server answers again

            if result is None:
                # Every path that returns None here means "the query ran and
                # this field has no rows" (PS1 below its southern limit, an
                # empty box).  A download that actually failed raises out of
                # astroquery instead, and stays a failure.
                raise CatalogNoCoverageError(
                    f"No data retrieved from {self._catalog_name}")

            self._tag_table(result, config, cached=False)

            if cacheable and len(result) > 0:
                # Save under the original (pre-widening) params so that future
                # narrow queries at the same snapped position get a cache hit.
                save_params = (self._original_query_params
                               if self._original_query_params is not None
                               else self._query_params)
                self.get_cache().save_to_cache(
                    self._catalog_name, save_params, result
                )

            return result

        finally:
            # Restore original (un-widened) query params
            if self._original_query_params is not None:
                self._query_params = self._original_query_params
                self._original_query_params = None

    def _widen_query_for_cache(self) -> None:
        """Snap center to grid and pad box so the cache entry covers future queries."""
        grid = CatalogCache.CACHE_GRID_DEG
        if self._query_params.ra is not None:
            self._original_query_params = QueryParams(
                **{k: getattr(self._query_params, k)
                   for k in QueryParams.__dataclass_fields__}
            )
            self._query_params.ra     = round(self._query_params.ra  / grid) * grid
            self._query_params.dec    = round(self._query_params.dec / grid) * grid
            self._query_params.width  = self._query_params.width  + grid
            self._query_params.height = self._query_params.height + grid

    # ------------------------------------------------------------------
    # Legacy Survey fetcher (not in pyrt base)
    # ------------------------------------------------------------------

    def _get_legacysurvey_data(self) -> Optional[astropy.table.Table]:
        """Fetch DESI Legacy Imaging Survey DR10 via NOIRLab Astro Data Lab TAP."""
        try:
            import pyvo
        except ImportError:
            raise ValueError(
                "pyvo is required for Legacy Survey queries: pip install pyvo"
            )

        ra   = self._query_params.ra
        dec  = self._query_params.dec
        mlim = self._query_params.mlim
        radius = np.sqrt(self._query_params.width ** 2 +
                         self._query_params.height ** 2) / 2

        flux_r_min = 10 ** ((22.5 - mlim) / 2.5)
        cos_dec    = max(np.cos(np.radians(dec)), 0.01)
        ra_margin  = radius / cos_dec
        ra_lo, ra_hi   = ra - ra_margin,  ra + ra_margin
        dec_lo, dec_hi = dec - radius,    dec + radius

        if ra_lo < 0:
            ra_cond = f"(ra >= {ra_lo + 360} OR ra <= {ra_hi})"
        elif ra_hi > 360:
            ra_cond = f"(ra >= {ra_lo} OR ra <= {ra_hi - 360})"
        else:
            ra_cond = f"ra BETWEEN {ra_lo} AND {ra_hi}"

        query = f"""
            SELECT ra, dec,
                   flux_g, flux_r, flux_z,
                   flux_ivar_g, flux_ivar_r, flux_ivar_z,
                   nobs_r, type
            FROM ls_dr10.tractor
            WHERE {ra_cond}
              AND dec BETWEEN {dec_lo} AND {dec_hi}
              AND flux_r > {flux_r_min}
              AND nobs_r > 0
        """
        try:
            tap = pyvo.dal.TAPService("https://datalab.noirlab.edu/tap")
            tap.timeout = self._query_params.timeout
            ls_cat = tap.search(query, maxrec=500000).to_table()
        except Exception as exc:
            raise ValueError(f"Legacy Survey TAP query failed: {exc}") from exc

        if len(ls_cat) == 0:
            logging.info("Legacy Survey: no sources found")
            return None

        result = astropy.table.Table()
        result["radeg"]  = np.array(ls_cat["ra"],  dtype=np.float64)
        result["decdeg"] = np.array(ls_cat["dec"], dtype=np.float64)
        result["pmra"]   = np.zeros(len(ls_cat), dtype=np.float64)
        result["pmdec"]  = np.zeros(len(ls_cat), dtype=np.float64)

        for _, col_flux, col_ivar, col_mag, col_err in (
            ("g", "flux_g", "flux_ivar_g", "Sloan_g", "Sloan_g_err"),
            ("r", "flux_r", "flux_ivar_r", "Sloan_r", "Sloan_r_err"),
            ("z", "flux_z", "flux_ivar_z", "Sloan_z", "Sloan_z_err"),
        ):
            flux = np.array(ls_cat[col_flux], dtype=np.float64)
            ivar = np.array(ls_cat[col_ivar], dtype=np.float64)
            ok   = flux > 0
            mag  = np.full(len(ls_cat), np.nan)
            merr = np.full(len(ls_cat), np.nan)
            mag[ok] = 22.5 - 2.5 * np.log10(flux[ok])
            snr = np.where(ivar > 0, flux * np.sqrt(ivar), 0.0)
            det = ok & (snr > 0)
            merr[det] = 2.5 / (np.log(10) * snr[det])
            result[col_mag] = mag
            result[col_err] = merr

        logging.info(f"Legacy Survey: {len(result)} sources")
        return result

    # ------------------------------------------------------------------
    # Transient-detection methods
    # ------------------------------------------------------------------

    def precompute_photometric_data(
        self,
        bands: Optional[List[str]] = None,
        force_recompute: bool = False,
    ) -> CatalogOptimizationCache:
        """Precompute per-star photometry for fast transient detection."""
        if self._photometric_cache is not None and not force_recompute:
            return self._photometric_cache

        if bands is None:
            bands = ["Sloan_g", "Sloan_r", "Sloan_i", "Sloan_z", "J"]

        logging.info(f"Precomputing photometric data for {len(self)} stars ...")
        coordinates = np.column_stack([self["radeg"], self["decdeg"]])
        n = len(self)
        magnitudes = np.full((n, len(bands)), np.nan)
        colors     = np.full((n, len(bands) - 1), np.nan)

        for i, band in enumerate(bands):
            if band in self.columns:
                col = self[band]
                if hasattr(col, "mask"):
                    ok = ~col.mask & np.isfinite(col.data) & (col.data < 99)
                    magnitudes[ok, i] = col.data[ok]
                else:
                    ok = np.isfinite(col) & (col < 99)
                    magnitudes[ok, i] = col[ok]

        # A star is "valid" if it has at least 2 non-NaN Sloan-band magnitudes
        valid_stars = np.sum(~np.isnan(magnitudes), axis=1) >= 2
        logging.info(f"  {np.sum(valid_stars)} valid stars with >=2 Sloan bands")

        filled, filled_ok = self.fill_missing_photometry_batch(magnitudes)
        use = valid_stars & filled_ok
        magnitudes[use] = filled[use]
        if len(bands) >= 5:
            # g-r, r-i, i-z, z-J
            colors[use, :4] = -np.diff(magnitudes[use, :5], axis=1)

        self._photometric_cache = CatalogOptimizationCache(
            coordinates=coordinates,
            pixel_coordinates={},
            magnitudes=magnitudes,
            colors=colors,
            valid_stars=valid_stars,
            kdtrees={},
            rough_mags=self.rough_magnitudes(self),
        )
        return self._photometric_cache

    # Index into the photometric cache's magnitude columns
    # (Sloan_g, Sloan_r, Sloan_i, Sloan_z, J) for a frame's PHFILTER.
    _BAND_INDEX = {"sloan_g": 0, "g": 0, "sloan_r": 1, "r": 1, "sloan_i": 2, "i": 2,
                   "sloan_z": 3, "z": 3, "j": 4}

    @classmethod
    def catalog_band_index(cls, phfilter) -> int:
        """Column of the cached magnitudes matching this frame band; Sloan r
        (1) for anything unknown (Johnson, narrow-band, 'N', None)."""
        if phfilter is None:
            return 1
        return cls._BAND_INDEX.get(str(phfilter).strip().lower(), 1)

    # Magnitude columns usable as a rough brightness when a star has no
    # Sloan photometry, in order of preference (red/visual first: the
    # pipeline compares against Sloan r).
    ROUGH_MAG_COLUMNS = ("Sloan_r", "R2", "R1", "Johnson_R", "G", "Johnson_V",
                         "Sloan_i", "I", "Sloan_g", "B2", "B1", "J")

    @classmethod
    def rough_magnitudes(cls, table: astropy.table.Table) -> np.ndarray:
        """Per-row best available magnitude from ROUGH_MAG_COLUMNS (first
        finite value in preference order), NaN where the catalogue has none."""
        n = len(table)
        out = np.full(n, np.nan)
        for col in cls.ROUGH_MAG_COLUMNS:
            if col not in table.columns:
                continue
            vals = np.ma.filled(np.ma.asarray(table[col], dtype=float), np.nan)
            vals = np.where(np.isfinite(vals) & (vals < 99) & (vals > -5), vals, np.nan)
            fill = np.isnan(out) & np.isfinite(vals)
            out[fill] = vals[fill]
        return out

    @staticmethod
    def fill_missing_photometry(
        mags: np.ndarray,
        typical_colors: Optional[List[float]] = None,
    ) -> Optional[np.ndarray]:
        """Fill missing bands using typical stellar colors."""
        if typical_colors is None:
            typical_colors = [0.6, 0.3, 0.2, 0.8]  # g-r, r-i, i-z, z-J
        if np.sum(~np.isnan(mags)) < 2:
            return None
        filled = mags.copy()
        for i, c in enumerate(typical_colors):
            if i + 1 < len(filled):
                if not np.isnan(filled[i]) and np.isnan(filled[i + 1]):
                    filled[i + 1] = filled[i] + c
        for i in range(len(typical_colors) - 1, -1, -1):
            if i + 1 < len(filled):
                if not np.isnan(filled[i + 1]) and np.isnan(filled[i]):
                    filled[i] = filled[i + 1] - typical_colors[i]
        last = None
        for i in range(len(filled)):
            if not np.isnan(filled[i]):
                last = filled[i]
            elif last is not None:
                filled[i] = last
        return filled if not np.any(np.isnan(filled)) else None

    @classmethod
    def fill_missing_photometry_batch(
        cls,
        mags: np.ndarray,
        typical_colors: Optional[List[float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorised `fill_missing_photometry` over a whole catalogue.

        `mags` is an (n, n_bands) array.  Returns the filled array and a
        boolean mask of the rows that could be filled completely; rows where
        the scalar version returns None are masked False and returned
        unchanged.  Equivalent to calling `fill_missing_photometry` per row,
        but ~1000x faster on the million-star queries the pipeline runs
        (the per-star loop cost ~35 s of pure CPU in every subprocess).
        """
        if typical_colors is None:
            typical_colors = [0.6, 0.3, 0.2, 0.8]
        mags = np.asarray(mags, dtype=float)
        filled = mags.copy()
        n, n_bands = filled.shape
        ok = np.sum(~np.isnan(filled), axis=1) >= 2
        if n == 0 or n_bands == 0:
            return filled, ok

        rows  = np.arange(n)[:, None]
        # Bands 0..reach are chained by the typical colors, so subtracting the
        # cumulative colour offset turns the scalar version's forward and
        # backward passes into plain fills of a single quantity.
        reach = min(len(typical_colors), n_bands - 1)
        if reach > 0:
            offset = np.concatenate([[0.0], np.cumsum(typical_colors[:reach])])
            cols   = np.arange(reach + 1)
            chain  = filled[:, :reach + 1] - offset

            src = np.where(~np.isnan(chain), cols, 0)          # carry forward
            np.maximum.accumulate(src, axis=1, out=src)
            chain = chain[rows, src]

            src = np.where(~np.isnan(chain), cols, reach)      # then backward
            src = np.minimum.accumulate(src[:, ::-1], axis=1)[:, ::-1]
            chain = chain[rows, src]

            filled[:, :reach + 1] = chain + offset

        # Bands past the colour chain simply repeat the last known magnitude.
        if np.isnan(filled).any():
            cols = np.arange(n_bands)
            src  = np.where(~np.isnan(filled), cols, 0)
            np.maximum.accumulate(src, axis=1, out=src)
            filled = filled[rows, src]

        ok &= ~np.any(np.isnan(filled), axis=1)
        filled[~ok] = mags[~ok]
        return filled, ok

    # --- pixel-coordinate cache ---

    def _generate_image_id(self, detections: astropy.table.Table) -> str:
        meta = detections.meta
        wcs_keys: Dict[str, Any] = {}
        for key, val in meta.items():
            if any(p in key for p in ("CRVAL", "CRPIX", "CD", "CDELT", "CROTA", "NAXIS")):
                wcs_keys[key] = (
                    round(float(val), 8)
                    if isinstance(val, (float, np.floating))
                    else val
                )
        return "img_" + hashlib.md5(
            str(sorted(wcs_keys.items())).encode()
        ).hexdigest()[:12]

    def _transform_catalog_to_pixel(self, det: astropy.table.Table) -> np.ndarray:
        # X_IMAGE/Y_IMAGE are measured on the distorted image, so a frame's
        # SIP terms have to stay in the projection. Projecting through the
        # bare TAN core put stars near the edges of FRAM frames (order-3
        # SIP, tests/190919B) up to ~2 px off -- beyond the 1 px adaptive
        # match floor -- so the same edge stars came back "new" in every
        # epoch. Zenithal projections keep their PV terms too: pyrt's
        # astrometric refit writes ZPN for wide-field cameras (FRAM's NF4,
        # PV2_3 ~ 87), and flattening that to TAN cost up to ~3 px at the
        # edges. For TAN, stray PV cards would be read as TPV distortion, so
        # there they are still dropped.
        # Non-finite values are dropped: astropy refuses NaN header cards, and
        # a single one anywhere in the meta (pyrt writes ASTSIGMA=nan when its
        # astrometric refit has nothing to fit) sent the matcher to the
        # legacy fallback.
        # Only the WCS keywords themselves: astropy serialises the whole dict
        # into header cards, and a list or an over-long string (the stack
        # ECSV's STACK_INPUTS) failed it just like a NaN did.
        from pyrt_transient.core.wcs_meta import wcs_header_from_meta
        header = wcs_header_from_meta(det.meta)
        ctype1 = str(header.get("CTYPE1", ""))
        keep_pv = ctype1[5:8] in ("ZPN", "AZP", "ZEA")
        has_sip = not keep_pv and "SIP" in ctype1 and "A_ORDER" in header
        if not keep_pv:
            suffix = "-SIP" if has_sip else ""
            header["CTYPE1"] = "RA---TAN" + suffix
            header["CTYPE2"] = "DEC--TAN" + suffix
        for key in list(header):
            if key in ("CTYPE1T", "CTYPE2T", "CRVAL1T", "CRVAL2T",
                       "CDELT1T", "CDELT2T", "CROTA2T"):
                del header[key]
            elif "PV" in key and not keep_pv:
                del header[key]
            elif not has_sip and any(p in key for p in ("A_", "B_", "AP_", "BP_")):
                del header[key]
        wcs = astropy.wcs.WCS(header)
        cat_x, cat_y = wcs.all_world2pix(self["radeg"], self["decdeg"], 1)
        return np.column_stack([cat_x, cat_y])

    def get_pixel_coordinates_cached(
        self,
        detections: astropy.table.Table,
        image_id: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        if not self._cache_enabled:
            return self._transform_catalog_to_pixel(detections)
        if image_id is None:
            image_id = self._generate_image_id(detections)
        if (self._photometric_cache
                and image_id in self._photometric_cache.pixel_coordinates):
            return self._photometric_cache.pixel_coordinates[image_id]
        try:
            coords = self._transform_catalog_to_pixel(detections)
            if self._photometric_cache is not None:
                self._photometric_cache.pixel_coordinates[image_id] = coords
            else:
                self._coordinate_cache[image_id] = coords
            return coords
        except Exception as exc:
            logging.debug(f"WCS transform failed: {exc}")
            return None

    def build_spatial_index(
        self,
        coordinates: np.ndarray,
        index_id: Optional[str] = None,
    ) -> KDTree:
        if index_id is None and self._cache_enabled:
            index_id = (
                "kdtree_"
                + hashlib.md5(coordinates.tobytes()).hexdigest()[:12]
            )
        if not self._cache_enabled or index_id is None:
            return KDTree(coordinates)
        cache_dict = (
            self._photometric_cache.kdtrees
            if self._photometric_cache
            else self._kdtree_cache
        )
        if index_id not in cache_dict:
            cache_dict[index_id] = KDTree(coordinates)
        return cache_dict[index_id]

    # --- local statistics ---

    def compute_local_statistics(
        self,
        positions: np.ndarray,
        radius: float,
        filter_pattern: Optional[str] = None,
        image_id: Optional[str] = None,
        max_mag: Optional[float] = None,
    ) -> Dict[str, List]:
        """max_mag: when given, only catalogue entries brighter than this
        (by their rough magnitude, see rough_magnitudes; entries without any
        magnitude are kept) enter the isolation/density statistics -- see
        DetectionConfig.isolation_max_mag_margin."""
        n = len(positions)
        defaults: Dict[str, List] = {
            "nearby_sources":      [0]      * n,
            "source_density":      [0.0]    * n,
            "nearest_source_dist": [np.inf] * n,
        }
        if filter_pattern:
            defaults[f"mean_mag_{filter_pattern}"] = [np.nan] * n
            defaults[f"std_mag_{filter_pattern}"]  = [np.nan] * n
        if n == 0:
            return defaults

        cat_coords = None
        try:
            pc = self._photometric_cache
            if pc and pc.pixel_coordinates:
                key = (
                    image_id
                    if (image_id and image_id in pc.pixel_coordinates)
                    else next(iter(pc.pixel_coordinates))
                )
                cat_coords = pc.pixel_coordinates[key]
        except Exception:
            pass

        if cat_coords is None or len(cat_coords) == 0:
            return defaults

        index_suffix = ""
        neighbor_map = None
        if max_mag is not None and self._photometric_cache is not None:
            rough = getattr(self._photometric_cache, "rough_mags", None)
            if rough is not None and len(rough) == len(cat_coords):
                keep = ~(np.isfinite(rough) & (rough >= max_mag))
                if not np.all(keep):
                    cat_coords = cat_coords[keep]
                    neighbor_map = np.flatnonzero(keep)
                    index_suffix = f"_m{max_mag:.2f}"
                    if len(cat_coords) == 0:
                        return defaults

        try:
            tree      = self.build_spatial_index(
                cat_coords, f"stats_{image_id}{index_suffix}" if image_id else None
            )
            neighbors = tree.query_radius(positions, r=radius)
            if neighbor_map is not None:
                neighbors = [neighbor_map[nb] for nb in neighbors]
            nearby    = [len(nb) for nb in neighbors]
            density   = [cnt / (np.pi * radius ** 2) for cnt in nearby]
            dists, _  = tree.query(positions, k=1)
            nearest   = dists.flatten().tolist()
            results: Dict[str, List] = {
                "nearby_sources":      nearby,
                "source_density":      density,
                "nearest_source_dist": nearest,
            }
            if filter_pattern and self._photometric_cache is not None:
                try:
                    results.update(
                        self._compute_filter_statistics(neighbors, filter_pattern)
                    )
                except Exception:
                    results[f"mean_mag_{filter_pattern}"] = [np.nan] * n
                    results[f"std_mag_{filter_pattern}"]  = [np.nan] * n
            return results
        except Exception as exc:
            logging.debug(f"compute_local_statistics failed: {exc}")
            return defaults

    def _compute_filter_statistics(
        self,
        neighbors: List[np.ndarray],
        filter_pattern: str,
    ) -> Dict[str, List]:
        band_idx = {
            "g": 0, "r": 1, "i": 2, "z": 3, "j": 4,
            "sloan_g": 0, "sloan_r": 1, "sloan_i": 2, "sloan_z": 3,
        }.get(filter_pattern.lower(), 1)
        mean_mags: List[float] = []
        std_mags:  List[float] = []
        for nb in neighbors:
            if len(nb) > 0 and self._photometric_cache is not None:
                vals = self._photometric_cache.magnitudes[nb, band_idx]
                vals = vals[~np.isnan(vals)]
                mean_mags.append(float(np.mean(vals)) if len(vals) > 0 else np.nan)
                std_mags.append(
                    float(np.std(vals)) if len(vals) > 1
                    else 0.0 if len(vals) == 1
                    else np.nan
                )
            else:
                mean_mags.append(np.nan)
                std_mags.append(np.nan)
        return {
            f"mean_mag_{filter_pattern}": mean_mags,
            f"std_mag_{filter_pattern}":  std_mags,
        }

    # --- adaptive radii ---

    def _compute_adaptive_radii(
        self,
        detections: astropy.table.Table,
        nsigma: float = 3.0,
        idlimit_min_px: float = 1.0,
        idlimit_max_px: float = 8.0,
        use_astvar: bool = True,
    ) -> np.ndarray:
        if len(detections) == 0:
            return np.array([])
        try:
            if ("ERRX2_IMAGE" in detections.colnames
                    and "ERRY2_IMAGE" in detections.colnames):
                ex2 = detections["ERRX2_IMAGE"].data
                ey2 = detections["ERRY2_IMAGE"].data
                ok  = (np.isfinite(ex2) & np.isfinite(ey2)
                       & (ex2 >= 0) & (ey2 >= 0))
                if np.any(ok):
                    pos_err = np.sqrt(ex2 + ey2)
                    if use_astvar:
                        # ASTSIGMA floor + ASTVAR scale for pyrt >= 97101a7,
                        # plain sqrt(ASTVAR) for older ECSVs.
                        pos_err = scaled_position_error(pos_err, detections.meta)
                    radii = np.where(
                        ok & np.isfinite(nsigma * pos_err), nsigma * pos_err, np.nan
                    )
                    if np.sum(np.isfinite(radii)) > 0:
                        return np.clip(radii, idlimit_min_px, idlimit_max_px)

            # SNR fallback
            snr = None
            if "SNR" in detections.colnames:
                snr = detections["SNR"].data
            elif ("FLUX_ISO" in detections.colnames
                  and "FLUXERR_ISO" in detections.colnames):
                f  = detections["FLUX_ISO"].data
                fe = detections["FLUXERR_ISO"].data
                snr = np.where(
                    (fe > 0) & np.isfinite(f) & np.isfinite(fe), f / fe, np.nan
                )
            if snr is not None:
                ok = np.isfinite(snr) & (snr > 0)
                if np.any(ok):
                    fwhm = (
                        detections["FWHM_IMAGE"].data
                        if "FWHM_IMAGE" in detections.colnames
                        else np.full(
                            len(detections),
                            float(detections.meta.get("FWHM", 1.2))
                        )
                    )
                    fwhm = np.where(
                        np.isfinite(fwhm) & (fwhm > 0), fwhm, 1.2
                    )
                    pos_err = (fwhm / 2.35) / np.maximum(snr, 1e-6)
                    if use_astvar:
                        pos_err = scaled_position_error(pos_err, detections.meta)
                    radii = np.where(
                        ok & np.isfinite(nsigma * pos_err), nsigma * pos_err, np.nan
                    )
                    if np.sum(np.isfinite(radii)) > 0:
                        return np.clip(radii, idlimit_min_px, idlimit_max_px)
        except Exception as exc:
            logging.error(f"Adaptive radius computation error: {exc}")
        return np.array([])

    # --- main detection entry point ---

    def get_transient_candidates_optimized(
        self,
        detections: astropy.table.Table,
        idlimit: float = 5.0,
        mag_change_threshold: float = 1.0,
        siglim: float = 5.0,
        frame: float = 10.0,
        adaptive_radii: Optional[np.ndarray] = None,
        unphotometered_match_is_new: bool = True,
        new_source_siglim: Optional[float] = None,
        unphotometered_veto_max_brightening: Optional[float] = None,
    ) -> astropy.table.Table:
        if len(detections) == 0:
            return astropy.table.Table()

        if self._photometric_cache is None:
            self.precompute_photometric_data()

        image_id = self._generate_image_id(detections)
        cat_xy   = self.get_pixel_coordinates_cached(detections, image_id)

        if cat_xy is None:
            warnings.warn("WCS transform failed, falling back to legacy method")
            return self._legacy_get_transient_candidates(detections, idlimit)

        tree   = self.build_spatial_index(cat_xy, image_id)
        det_xy = np.column_stack([detections["X_IMAGE"], detections["Y_IMAGE"]])

        adaptive_enabled = False
        r_i = adaptive_radii

        if adaptive_radii is None:
            adaptive_enabled = bool(
                detections.meta.get("adaptive_idlimit_enabled", False)
            )
            if adaptive_enabled:
                r_i = self._compute_adaptive_radii(
                    detections,
                    nsigma=detections.meta.get("adaptive_nsigma", 3.0),
                    idlimit_min_px=detections.meta.get("idlimit_min_px", 1.0),
                    idlimit_max_px=detections.meta.get("idlimit_max_px", 8.0),
                    use_astvar=detections.meta.get("use_astvar", True),
                )
                if len(r_i) == 0 or np.all(~np.isfinite(r_i)):
                    adaptive_enabled = False
                    r_i = None
        else:
            adaptive_enabled = True

        if adaptive_enabled and r_i is not None:
            if len(r_i) != len(detections):
                raise ValueError(
                    f"adaptive_radii length ({len(r_i)}) != "
                    f"detections length ({len(detections)})"
                )
            valid = np.isfinite(r_i) & (r_i > 0)
            if not np.any(valid):
                adaptive_enabled = False
            else:
                r_i = np.where(valid, r_i, idlimit)
                r_query = np.percentile(
                    r_i, detections.meta.get("adaptive_percentile", 95.0)
                )
                all_m, all_d = tree.query_radius(
                    det_xy, r=r_query, return_distance=True
                )
                matches_list = [
                    m[d <= r_i[i]]
                    for i, (m, d) in enumerate(zip(all_m, all_d))
                ]

        if not adaptive_enabled:
            matches_list = tree.query_radius(det_xy, r=idlimit)

        return self._process_detections_for_candidates(
            detections, matches_list, mag_change_threshold, siglim, frame,
            new_source_siglim=new_source_siglim,
            unphotometered_match_is_new=unphotometered_match_is_new,
            unphotometered_veto_max_brightening=unphotometered_veto_max_brightening,
        )

    def get_transient_candidates(
        self, det: astropy.table.Table, idlimit: float = 5.0
    ) -> astropy.table.Table:
        warnings.warn(
            "get_transient_candidates is deprecated; "
            "use get_transient_candidates_optimized.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_transient_candidates_optimized(det, idlimit=idlimit)

    def _legacy_get_transient_candidates(
        self, det: astropy.table.Table, idlimit: float = 5.0
    ) -> astropy.table.Table:
        cat_xy = self._transform_catalog_to_pixel(det)
        if len(cat_xy) < 1:
            return det
        det_xy  = np.column_stack([det["X_IMAGE"], det["Y_IMAGE"]])
        tree    = KDTree(cat_xy)
        indices = tree.query_radius(det_xy, r=idlimit)
        return det[[len(idx) == 0 for idx in indices]].copy()

    def _process_detections_for_candidates(
        self,
        detections: astropy.table.Table,
        matches_list: List[np.ndarray],
        mag_change_threshold: float,
        siglim: float,
        frame: float,
        new_source_siglim: Optional[float] = None,
        unphotometered_match_is_new: bool = True,
        unphotometered_veto_max_brightening: Optional[float] = None,
    ) -> astropy.table.Table:
        """new_source_siglim, when lower than siglim, admits fainter/noisier
        detections as "new" candidates (no reference-catalog match at all)
        without loosening the bar for flagging a *matched* catalog source as
        changed. Defaults to siglim (no behavior change) -- see
        DetectionConfig.new_source_siglim's docstring for why these two are
        deliberately not the same knob.
        """
        if new_source_siglim is None:
            new_source_siglim = siglim

        det_x    = detections["X_IMAGE"].data
        det_y    = detections["Y_IMAGE"].data
        det_mags = detections["MAG_CALIB"].data
        det_errs = detections["MAGERR_CALIB"].data
        # Catalogue band to compare MAG_CALIB with: the frame's photometric
        # band (PHFILTER, e.g. "Sloan_i"), not always Sloan r. Falls back to
        # r per star when the catalogue lacks that band.
        band_idx = self.catalog_band_index(detections.meta.get("PHFILTER", detections.meta.get("FILTER")))
        # SExtractor FLAGS bit 2 = blended with another source. Threaded
        # through to _check_magnitude_changes_cached so a blend's combined
        # flux being brighter than the catalogued star alone would predict
        # can use the same more permissive significance bar new-source
        # detections already get, instead of silently reading as "same star,
        # unchanged" -- see DetectionConfig.new_source_siglim's docstring
        # and FUTURE_IDEAS.md's blending mechanism note for why this
        # dilution is real, not hypothetical.
        det_blended = (
            (detections["FLAGS"].data & 2) > 0
            if "FLAGS" in detections.colnames
            else np.zeros(len(detections), dtype=bool)
        )
        img_w = detections.meta.get(
            "NAXIS1", detections.meta.get(
                "IMGAXIS1", detections.meta.get("IMAGEW", np.max(det_x) + 100)
            )
        )
        img_h = detections.meta.get(
            "NAXIS2", detections.meta.get(
                "IMGAXIS2", detections.meta.get("IMAGEH", np.max(det_y) + 100)
            )
        )
        edge = ((det_x < frame) | (det_y < frame) |
                (det_x > img_w - frame) | (det_y > img_h - frame))
        # Blended detections use new_source_siglim here too (not just inside
        # _check_magnitude_changes_cached below) -- otherwise a blend's
        # inherently noisier combined-flux measurement gets filtered out by
        # this admission gate before it ever reaches the blend-aware check,
        # regardless of that check's own bar.
        # min(): nothing enforces new_source_siglim < siglim, and a config
        # with it higher must not make blended matched detections stricter
        # than unblended ones.
        blended_siglim = min(new_source_siglim, siglim)
        effective_matched_siglim = np.where(det_blended, blended_siglim, siglim)
        bad_snr_matched = det_errs >= (1.091 / effective_matched_siglim)
        bad_snr_new     = det_errs >= (1.091 / new_source_siglim)

        candidates: List[int] = []
        types:      List[str] = []
        diffs:      List[float] = []
        response = detections.meta.get("RESPONSE", "P0=25.0")

        for i, matches in enumerate(matches_list):
            if edge[i]:
                continue
            if len(matches) == 0:
                if bad_snr_new[i]:
                    continue
                candidates.append(i)
                types.append("new")
                diffs.append(0.0)
                continue
            if bad_snr_matched[i]:
                continue
            is_cand, ctype, mdiff = self._check_magnitude_changes_cached(
                matches, det_mags[i], det_errs[i], response,
                mag_change_threshold, siglim,
                is_blended=bool(det_blended[i]),
                new_source_siglim=new_source_siglim,
                unphotometered_match_is_new=unphotometered_match_is_new,
                unphotometered_veto_max_brightening=unphotometered_veto_max_brightening,
                band_idx=band_idx,
            )
            if is_cand:
                candidates.append(i)
                types.append(ctype)
                diffs.append(float(mdiff))

        if not candidates:
            return astropy.table.Table()
        result = detections[candidates].copy()
        result["candidate_type"]       = types
        result["magnitude_difference"] = diffs
        return result

    def _check_magnitude_changes_cached(
        self,
        matches: np.ndarray,
        det_mag: float,
        det_mag_err: float,
        response_model: str,
        mag_change_threshold: float,
        siglim: float,
        is_blended: bool = False,
        new_source_siglim: Optional[float] = None,
        unphotometered_match_is_new: bool = True,
        unphotometered_veto_max_brightening: Optional[float] = None,
        band_idx: int = 1,
    ) -> Tuple[bool, str, float]:
        """is_blended/new_source_siglim: a detection blended with a known
        catalog star (SExtractor FLAGS bit 2) measures the *combined* flux
        of both, which dilutes a real superimposed source's excess enough
        that it can fail the ordinary `siglim` significance bar even when
        genuinely brighter than the catalogued star alone -- see
        FUTURE_IDEAS.md's blending mechanism note (found via GRB200410A,
        a real GCN-confirmed afterglow this diluting effect hid). For a
        blended detection specifically, a real brightening excess is
        checked against the same more permissive bar new-source detections
        already get (`new_source_siglim`) instead of the strict one, rather
        than treating any blend as automatically "same star, unchanged".
        """
        if new_source_siglim is None:
            new_source_siglim = siglim
        name = self.catalog_name.lower()
        cat_sys = (
            0.01 if "gaia"         in name else
            0.02 if "panstarrs"    in name else
            0.02 if "legacysurvey" in name else
            0.03 if "atlas"        in name else
            0.10 if "usno"         in name else 0.02
        )
        det_sys = 0.01

        significant: List[Tuple[float, str]] = []
        any_valid = False
        # Catalogue magnitudes of every usable match, for the blend check
        # below: a blended detection measures the *summed* flux of all the
        # catalogued stars under it, so its excess has to be judged against
        # that sum. Comparing against each match individually flagged every
        # ordinary unresolved pair as "brightening" (the pair is always
        # brighter than its fainter member).
        matched_cat_mags: List[float] = []
        # Rough magnitudes of the matches without usable Sloan photometry
        # (USNO-B plate magnitudes, Gaia G without BP/RP, ...).
        rough_mags = getattr(self._photometric_cache, "rough_mags", None)
        unphotometered_rough: List[float] = []
        predicted: set = set()  # matches with a Sloan prediction in matched_cat_mags

        for idx in matches:
            if not self._photometric_cache.valid_stars[idx]:  # type: ignore[union-attr]
                if rough_mags is not None and np.isfinite(rough_mags[idx]):
                    unphotometered_rough.append(float(rough_mags[idx]))
                continue
            try:
                # The frame's own band where the catalogue has it (an i-band
                # frame compared against Sloan r read every red star as a
                # magnitude change -- FUTURE_IDEAS.md "Catalogue comparison
                # is always against Sloan r"), else Sloan r.
                r_mag  = self._photometric_cache.magnitudes[idx, band_idx]  # type: ignore[union-attr]
                if np.isnan(r_mag):
                    r_mag = self._photometric_cache.magnitudes[idx, 1]  # type: ignore[union-attr]
                colors = self._photometric_cache.colors[idx]          # type: ignore[union-attr]
                if np.isnan(r_mag) or np.any(np.isnan(colors)):
                    continue

                cat_mag = simple_color_model(
                    response_model,
                    (r_mag, colors[0], colors[1], colors[2], colors[3]),
                )

                sigma  = np.sqrt(det_mag_err ** 2 + det_sys ** 2 + cat_sys ** 2)
                diff   = det_mag - cat_mag
                nsigma = abs(diff) / sigma
                any_valid = True
                matched_cat_mags.append(float(cat_mag))
                predicted.add(int(idx))

                if is_blended:
                    # A blend measures every star under it at once, so a
                    # per-star comparison is meaningless (the pair is always
                    # "brighter" than its fainter member) -- judged once
                    # against the summed prediction below instead.
                    continue

                if abs(diff) >= mag_change_threshold and nsigma > siglim:
                    significant.append(
                        (diff, "brightening" if diff < 0 else "fading")
                    )
                # else (not significantly different, or intermediate): keep
                # checking the remaining matches before deciding -- with more
                # than one catalog match nearby, an early "not significant"
                # on the first one checked must not pre-empt a genuinely
                # significant match still to come.

            except Exception:
                continue

        if is_blended and matched_cat_mags:
            # The blend also holds the matches without a Sloan prediction.
            # Leaving them out of the sum read every such blend as
            # "brightening" by exactly their flux, so their rough magnitudes
            # go in too.
            blend_mags = list(matched_cat_mags)
            flux_unknown = False
            for idx in matches:
                if int(idx) in predicted:
                    continue
                if rough_mags is not None and np.isfinite(rough_mags[idx]):
                    blend_mags.append(float(rough_mags[idx]))
                else:
                    flux_unknown = True
            combined_cat_mag = -2.5 * np.log10(
                np.sum(10.0 ** (-0.4 * np.asarray(blend_mags)))
            )
            sigma = np.sqrt(det_mag_err ** 2 + det_sys ** 2 + cat_sys ** 2)
            diff = float(det_mag - combined_cat_mag)
            nsigma = abs(diff) / sigma
            # A member with no magnitude at all can hide an excess up to the
            # margin a positional-only match is allowed.
            explained = (diff < 0 and flux_unknown
                         and unphotometered_veto_max_brightening is not None
                         and diff > -abs(unphotometered_veto_max_brightening))
            if explained:
                pass
            elif abs(diff) >= mag_change_threshold and nsigma > siglim:
                significant.append((diff, "brightening" if diff < 0 else "fading"))
            elif (diff < 0 and abs(diff) >= mag_change_threshold
                    and nsigma > min(new_source_siglim, siglim)):
                # Combined flux measurably brighter than all the catalogued
                # stars under it together would predict, but not enough to
                # clear the strict bar -- the diluted-excess case
                # new_source_siglim exists for (FUTURE_IDEAS.md, GRB200410A).
                significant.append((diff, "brightening"))

        if significant:
            best = max(significant, key=lambda x: abs(x[0]))
            return True, best[1], float(best[0])
        if any_valid:
            return False, "none", 0.0
        # Matches exist but none has usable photometry. Historically this
        # was reported as "new", which for a catalogue without Sloan bands
        # (USNO-B) makes every detection "new" -- see
        # DetectionConfig.unphotometered_match_is_new.
        if unphotometered_match_is_new:
            return True, "new", np.nan
        # A positional-only veto must still be magnitude-aware: a 20 mag
        # USNO-B star 2" away cannot be a 15 mag detection (GRB 250813B's
        # afterglow was lost exactly this way). If the detection is brighter
        # than the brightest rough catalogue magnitude among the matches by
        # more than the allowed margin (colour terms, plate photometry
        # scatter and blending are all covered by a generous margin), report
        # it as brightening instead of vetoing it.
        if (unphotometered_veto_max_brightening is not None and unphotometered_rough
                and np.isfinite(det_mag)):
            diff = float(det_mag - min(unphotometered_rough))
            if diff <= -abs(unphotometered_veto_max_brightening):
                return True, "brightening", diff
        return False, "matched_unphotometered", 0.0

    # --- convenience helpers ---

    def compute_magnitude_difference(
        self, det: astropy.table.Table, filter_name: str
    ) -> astropy.table.Table:
        if filter_name not in self.filters:
            raise ValueError(f"Filter '{filter_name}' not in catalog")
        cat_xy = self._transform_catalog_to_pixel(det)
        det_xy = np.column_stack([det["X_IMAGE"], det["Y_IMAGE"]])
        tree   = KDTree(cat_xy)
        _, idx = tree.query(det_xy, k=1)
        cat_mag = self[filter_name][idx].flatten()
        det["mag_diff"] = np.array(det["MAG_CALIB"]) - cat_mag
        try:
            cat_err = self[self.filters[filter_name].error_name][idx]
            det["mag_diff_err"] = np.sqrt(
                cat_err ** 2 + np.array(det["MAGERR_CALIB"]) ** 2
            )
        except Exception:
            pass
        return det

    def match_with_external_catalog(
        self,
        other_cat: "CatTransients",
        max_separation: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        c1 = SkyCoord(ra=self["radeg"] * u.deg, dec=self["decdeg"] * u.deg)
        c2 = SkyCoord(ra=other_cat["radeg"] * u.deg, dec=other_cat["decdeg"] * u.deg)
        idx1, idx2, _, _ = c1.search_around_sky(c2, max_separation * u.arcsec)
        return idx1, idx2

    # --- runtime cache management ---

    def clear_cache(self) -> None:
        self._photometric_cache = None
        self._coordinate_cache.clear()
        self._kdtree_cache.clear()

    def get_runtime_cache_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "photometric_cache_exists": self._photometric_cache is not None,
            "coordinate_cache_size":    len(self._coordinate_cache),
            "kdtree_cache_size":        len(self._kdtree_cache),
            "cache_enabled":            self._cache_enabled,
        }
        if self._photometric_cache:
            info.update(
                {
                    "n_valid_stars":        int(np.sum(self._photometric_cache.valid_stars)),
                    "n_cached_coordinates": len(self._photometric_cache.pixel_coordinates),
                    "n_cached_kdtrees":     len(self._photometric_cache.kdtrees),
                }
            )
        return info

    def enable_cache(self, enabled: bool = True) -> None:
        self._cache_enabled = enabled
        if not enabled:
            self.clear_cache()


# ---------------------------------------------------------------------------
# Backward-compatible alias: existing code that does `from catalog import Catalog`
# continues to work without modification.
# ---------------------------------------------------------------------------
Catalog = CatTransients


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def setup_catalog_cache(cache_dir: str = "./catalog_cache") -> None:
    """Configure disk cache directory for all CatTransients instances."""
    CatTransients.set_cache_directory(cache_dir)
    logging.info(f"Catalog cache set to: {cache_dir}")


def print_cache_info() -> None:
    info = CatTransients.get_cache_info()
    total_files = 0
    total_mb    = 0.0
    for name, ci in info.items():
        logging.info(
            f"{name.upper()}: {ci['num_files']} files, {ci['total_size_mb']:.1f} MB"
        )
        total_files += ci["num_files"]
        total_mb    += ci["total_size_mb"]
    logging.info(f"Total: {total_files} files, {total_mb:.1f} MB")


def clear_old_cache(max_age_days: float = 30.0) -> None:
    CatTransients.clear_all_cache(max_age_days=max_age_days)


def clear_all_cache() -> None:
    CatTransients.clear_all_cache()


def add_catalog_argument(parser: Any) -> None:
    parser.add_argument(
        "--catalog",
        choices=list(CatTransients.KNOWN_CATALOGS.keys()),
        default="atlas@localhost",
        help="Reference catalog for transient detection",
    )
