"""Per-run catalog loading/caching."""

import logging

from pyrt_transient.catalog import Catalog, QueryParams


class CatalogLoader:
    """Loads and caches CatTransients instances per (catalog_name, params)
    for the lifetime of one analysis run, so repeated catalog lookups
    (e.g. once per epoch) don't re-fetch/re-precompute.
    """

    def __init__(self):
        self._loaded_catalogs = {}
        self.logger = logging.getLogger('transient_analyser.optimized')

    def get_optimized_catalog(self, cat_name, params):
        """Get catalog with optimization, including error handling."""
        cache_key = f"{cat_name}_{hash(str(params.__dict__) if params else 'default')}"

        if cache_key in self._loaded_catalogs:
            self.logger.debug(f"Reusing loaded catalog: {cat_name}")
            return self._loaded_catalogs[cache_key]

        if params is None:
            params = QueryParams()

        self.logger.info(f"Loading catalog: {cat_name}")
        catalog = Catalog(catalog=cat_name, **params.__dict__)

        # Try to enable optimizations
        try:
            self.logger.debug(f"Precomputing photometric data for {cat_name}...")
            catalog.precompute_photometric_data()
            self.logger.debug(f"✅ Optimization enabled for {cat_name}")
        except Exception as e:
            self.logger.warning(f"Optimization failed for {cat_name}: {e}")
            self.logger.debug(f"Will use standard methods")

        self._loaded_catalogs[cache_key] = catalog
        return catalog

    def clear_cache(self) -> None:
        """Clear loaded catalogs to free memory."""
        for catalog in self._loaded_catalogs.values():
            catalog.clear_cache()
        self._loaded_catalogs.clear()
        self.logger.info("Catalog cache cleared")
