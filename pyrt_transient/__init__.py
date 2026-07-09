"""
pyrt-transient — transient detection addon for the pyrt photometry pipeline.

Core (always available):
    BlindMulticatalogStrategy Production detection strategy: catalog_match ->
                              clustering/lightcurves -> plotting. This is
                              what pipeline_magic.py actually calls.
    DetectionStrategy         ABC every detection strategy implements

    CatTransients             Catalog subclass with all transient-detection methods
    CatalogCache              Disk-based cache for remote catalog queries
    CatalogOptimizationCache  Per-instance precomputed photometric data
    filter_vsx_variables      Remove known VSX variables from candidate list
    QueryParams               Catalog query parameters (re-exported from pyrt)
    setup_catalog_cache       Configure the shared disk cache directory

    ObservationStore          Per-observation on-disk state: processed-file
                              tracking, incremental epoch loading, results

    PipelineConfig            Full pipeline configuration (wraps DetectionConfig etc.)
    DetectionConfig

Optional — pipeline entry point (requires pyrt_transient.pipeline_magic):
    Run directly:  python -m pyrt_transient.pipeline_magic <ecsv> ...

Optional — frontend (requires Pillow + matplotlib):
    FrontendGenerator        Generate the HTML candidate browser
"""

from pyrt_transient.catalog import (
    CatTransients,
    CatalogCache,
    CatalogOptimizationCache,
    filter_vsx_variables,
    setup_catalog_cache,
    # re-exported from pyrt so callers don't need to know the upstream layout
    QueryParams,
    CatalogFilter,
    CatalogFilters,
    # backward-compatible alias
    Catalog,
)

from pyrt_transient.config_trans import PipelineConfig, DetectionConfig

from pyrt_transient.detection.base import DetectionStrategy
from pyrt_transient.detection.blind_multicatalog import BlindMulticatalogStrategy

from pyrt_transient.io.observation_store import ObservationStore

__version__ = "0.2.0"

__all__ = [
    # detection strategy (production path)
    "DetectionStrategy",
    "BlindMulticatalogStrategy",
    # catalog
    "CatTransients",
    "Catalog",
    "CatalogCache",
    "CatalogOptimizationCache",
    "filter_vsx_variables",
    "setup_catalog_cache",
    "QueryParams",
    "CatalogFilter",
    "CatalogFilters",
    # io
    "ObservationStore",
    # config
    "PipelineConfig",
    "DetectionConfig",
    # version
    "__version__",
]

# Optional components — imported lazily so missing dependencies don't break the core
try:
    from pyrt_transient.frontend_generator import FrontendGenerator
    __all__.append("FrontendGenerator")
except ImportError:
    pass
