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
import logging
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


# ---------------------------------------------------------------------------
# Disk-based catalog cache (shared across instances)
# ---------------------------------------------------------------------------

class CatalogCache:
    """Disk cache for remote catalog queries with spatial grid binning."""

    # Pointings within CACHE_GRID_DEG share one cache entry.  The query box
    # is padded by this amount so the cached data always covers the footprint.
    CACHE_GRID_DEG = 1.0

    def __init__(self, cache_dir: str = "./catalog_cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        for name in ("panstarrs", "gaia", "atlas_vizier", "usno", "vsx",
                     "legacysurvey"):
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
        return self.cache_dir / catalog_name / f"{key}.pkl"

    def is_cached(self, catalog_name: str, params: QueryParams) -> bool:
        return self.get_cache_path(catalog_name, params).exists()

    # ------------------------------------------------------------------
    # Load / save
    # ------------------------------------------------------------------

    def load_from_cache(
        self, catalog_name: str, params: QueryParams
    ) -> Optional[astropy.table.Table]:
        path = self.get_cache_path(catalog_name, params)
        if not path.exists():
            return None
        try:
            with open(path, "rb") as fh:
                cached = pickle.load(fh)
            if isinstance(cached, dict) and "data" in cached and "timestamp" in cached:
                age_days = (time.time() - cached["timestamp"]) / 86400
                if age_days < 30:
                    logging.info(
                        f"Loaded {catalog_name} from cache (age: {age_days:.1f} d)"
                    )
                    return cached["data"]
                logging.info(
                    f"Cache for {catalog_name} expired ({age_days:.1f} d), refreshing"
                )
                path.unlink()
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

    # Extra catalog identifier not present in the base class
    LEGACYSURVEY: str = "legacysurvey"

    # Extend parent's KNOWN_CATALOGS: mark remote catalogs as cacheable and
    # add the Legacy Survey entry.
    KNOWN_CATALOGS = {
        k: dict(v, cacheable=(not v.get("local", False)))
        for k, v in _PyrtCatalog.KNOWN_CATALOGS.items()
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
            G  = np.array(result["G"],  dtype=np.float64)
            BP = np.array(result["BP"], dtype=np.float64)
            RP = np.array(result["RP"], dtype=np.float64)
            sloan_g, sloan_r = self._gaia_to_sloan(G, BP, RP)
            result["Sloan_g"] = sloan_g
            result["Sloan_r"] = sloan_r
        return result

    # ------------------------------------------------------------------
    # Override _fetch_catalog_data to add disk caching + Legacy Survey
    # ------------------------------------------------------------------

    def _fetch_catalog_data(self) -> Optional[astropy.table.Table]:  # type: ignore[override]
        """Fetch catalog data, adding disk caching and Legacy Survey support."""
        if self._catalog_name not in self.KNOWN_CATALOGS:
            raise ValueError(f"Unknown catalog: {self._catalog_name}")

        config = self.KNOWN_CATALOGS[self._catalog_name]
        cacheable = config.get("cacheable", False)

        # Try disk cache first
        if cacheable:
            cached = self.get_cache().load_from_cache(
                self._catalog_name, self._query_params
            )
            if cached is not None:
                cached.meta.update(
                    {
                        "catalog":  self._catalog_name,
                        "astepoch": config["epoch"],
                        "filters":  list(config["filters"].keys()),
                        "cached":   True,
                    }
                )
                return cached

            # Widen the query so nearby pointings get a cache hit
            self._widen_query_for_cache()

        try:
            if self._catalog_name == self.LEGACYSURVEY:
                result = self._get_legacysurvey_data()
            else:
                # Parent handles ATLAS, PANSTARRS, GAIA, USNOB, SDSS, MAKAK
                result = super()._fetch_catalog_data()

            if result is None:
                raise ValueError(f"No data retrieved from {self._catalog_name}")

            result.meta.update(
                {
                    "catalog":  self._catalog_name,
                    "astepoch": config["epoch"],
                    "filters":  list(config["filters"].keys()),
                    "cached":   False,
                }
            )

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

        for i in np.where(valid_stars)[0]:
            filled = self.fill_missing_photometry(magnitudes[i].copy())
            if filled is not None:
                magnitudes[i] = filled
                if len(filled) >= 5:
                    colors[i] = [
                        filled[0] - filled[1],  # g-r
                        filled[1] - filled[2],  # r-i
                        filled[2] - filled[3],  # i-z
                        filled[3] - filled[4],  # z-J
                    ]

        self._photometric_cache = CatalogOptimizationCache(
            coordinates=coordinates,
            pixel_coordinates={},
            magnitudes=magnitudes,
            colors=colors,
            valid_stars=valid_stars,
            kdtrees={},
        )
        return self._photometric_cache

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
        header = dict(det.meta)
        header["CTYPE1"] = "RA---TAN"
        header["CTYPE2"] = "DEC--TAN"
        for key in list(header):
            if key in ("CTYPE1T", "CTYPE2T", "CRVAL1T", "CRVAL2T",
                       "CDELT1T", "CDELT2T", "CROTA2T"):
                del header[key]
            elif any(p in key for p in ("PV", "A_", "B_", "AP_", "BP_")):
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
    ) -> Dict[str, List]:
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

        try:
            tree      = self.build_spatial_index(
                cat_coords, f"stats_{image_id}" if image_id else None
            )
            neighbors = tree.query_radius(positions, r=radius)
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
                        av = float(detections.meta.get("ASTVAR", 1.0))
                        pos_err *= np.sqrt(av if (np.isfinite(av) and av > 0) else 1.0)
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
                        av = float(detections.meta.get("ASTVAR", 1.0))
                        pos_err *= np.sqrt(
                            av if (np.isfinite(av) and av > 0) else 1.0
                        )
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
        new_source_siglim: Optional[float] = None,
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
        bad_snr_matched = det_errs >= (1.091 / siglim)
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
    ) -> Tuple[bool, str, float]:
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

        for idx in matches:
            if not self._photometric_cache.valid_stars[idx]:  # type: ignore[union-attr]
                continue
            try:
                r_mag  = self._photometric_cache.magnitudes[idx, 1]  # type: ignore[union-attr]
                colors = self._photometric_cache.colors[idx]          # type: ignore[union-attr]
                if np.isnan(r_mag) or np.any(np.isnan(colors)):
                    continue

                try:
                    from pyrt_transient.transients import simple_color_model
                    cat_mag = simple_color_model(
                        response_model,
                        (r_mag, colors[0], colors[1], colors[2], colors[3]),
                    )
                except ImportError:
                    cat_mag = r_mag

                sigma  = np.sqrt(det_mag_err ** 2 + det_sys ** 2 + cat_sys ** 2)
                diff   = det_mag - cat_mag
                nsigma = abs(diff) / sigma
                any_valid = True

                if abs(diff) >= mag_change_threshold and nsigma > siglim:
                    significant.append(
                        (diff, "brightening" if diff < 0 else "fading")
                    )
                # else (not significantly different, or intermediate): keep
                # checking the remaining matches before deciding -- with more
                # than one catalog match nearby (e.g. a blended pair), an
                # early "not significant" on the first one checked must not
                # pre-empt a genuinely significant match still to come.

            except Exception:
                continue

        if significant:
            best = max(significant, key=lambda x: abs(x[0]))
            return True, best[1], float(best[0])
        if any_valid:
            return False, "none", 0.0
        return True, "new", np.nan

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
