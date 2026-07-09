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

Legacy (superseded by BlindMulticatalogStrategy, kept for backward
compatibility -- not called by the production pipeline anymore):
    OptimizedTransientAnalyzer        Single-image transient analyser
    OptimizedMultiDetectionAnalyzer   Multi-epoch detection analyser

Optional — pipeline entry point (requires pyrt_transient.pipeline_magic):
    Run directly:  python -m pyrt_transient.pipeline_magic <ecsv> ...

Optional — frontend (requires Pillow + matplotlib):
    FrontendGenerator        Generate the HTML candidate browser

Optional — forced photometry (requires pyvo):
    query_panstarrs_lightcurves
    query_atlas_lightcurves
    merge_lightcurves
    filter_ps1_known_sources
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

from pyrt_transient.transient_analyser import (
    OptimizedTransientAnalyzer,
    OptimizedMultiDetectionAnalyzer,
)

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
    # legacy analyser (superseded, kept for backward compatibility)
    "OptimizedTransientAnalyzer",
    "OptimizedMultiDetectionAnalyzer",
    # version
    "__version__",
]

# Optional components — imported lazily so missing dependencies don't break the core
try:
    from pyrt_transient.frontend_generator import FrontendGenerator
    __all__.append("FrontendGenerator")
except ImportError:
    pass

try:
    from pyrt_transient.forced_photometry import (
        query_panstarrs_lightcurves,
        query_atlas_lightcurves,
        merge_lightcurves,
        filter_ps1_known_sources,
    )
    __all__ += [
        "query_panstarrs_lightcurves",
        "query_atlas_lightcurves",
        "merge_lightcurves",
        "filter_ps1_known_sources",
    ]
except ImportError:
    pass
