"""BlindMulticatalogStrategy -- the production detection-strategy
orchestrator, wiring catalog_match.py (Step 1: per-catalog detection),
clustering.py (Step 2: cross-epoch/cross-catalog clustering + lightcurves),
and plotting.py (Step 3: lightcurve plots) together. Called directly by
pipeline_magic.py.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from astropy.table import Table

from pyrt_transient.core.epochs import prepare_epoch_detections
from pyrt_transient.io.naming import get_base_filename
from pyrt_transient.detection.base import DetectionStrategy
from pyrt_transient.detection.blind_multicatalog.catalog_query import CatalogLoader
from pyrt_transient.detection.blind_multicatalog import catalog_match
from pyrt_transient.detection.blind_multicatalog import clustering
from pyrt_transient.detection.blind_multicatalog import plotting


class BlindMulticatalogStrategy(DetectionStrategy):
    def __init__(self, data_dir, lightcurve_dir=None, config=None):
        self.data_dir = Path(data_dir)
        self.lightcurve_dir = Path(lightcurve_dir) if lightcurve_dir else self.data_dir
        self.config = config
        self.logger = logging.getLogger('detection.blind_multicatalog')
        self.catalog_loader = CatalogLoader()
        if not self.lightcurve_dir.exists():
            self.lightcurve_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        detection_tables: List[Table],
        config=None,
        catalogs: Optional[List[str]] = None,
        params=None,
        idlimit: float = 5.0,
        radius_check: float = 30.0,
        filter_pattern: Optional[str] = None,
        mag_change_threshold: float = 1.0,
        add_strategy_fields_fn=None,
    ) -> Tuple[Table, Dict]:
        """Returns (final_candidates_table, lightcurves_dict) -- see
        detection/base.py's module docstring for why this isn't
        List[Candidate] yet.

        add_strategy_fields_fn: not wired to anything by default -- no
        working implementation exists yet (see FUTURE_IDEAS.md, "Dead code
        to remove" -- the old attempt's `strategy_v2` import never
        resolved). Pass one explicitly if a caller has it; strategy fields
        are simply skipped otherwise (same as
        lightcurve.update_candidate_with_lightcurve_stats's default).
        """
        config = config or self.config

        if config:
            position_match_radius = config.detection.position_match_radius_arcsec
            min_n_detections = config.detection.min_n_detections
            min_catalogs = config.detection.min_catalogs_fraction
            min_quality = config.detection.min_quality
            catalogs = catalogs or config.detection.catalogs
        else:
            position_match_radius = 2.0
            min_n_detections = 3
            min_catalogs = 1.0
            min_quality = 0.1

        self.logger.info(f"Using position_match_radius: {position_match_radius} arcsec from config")
        self.logger.info(f"Starting processing of {len(detection_tables)} detection tables...")

        # Step 1: Process each detection table efficiently (incremental)
        self.logger.info("Step 1: Processing individual detection tables...")

        for i, det_table in enumerate(detection_tables):
            base_filename = get_base_filename(det_table, i)
            ecsv_path = self.data_dir / f"{base_filename}_transients.ecsv"

            if ecsv_path.exists():
                self.logger.info(f"Epoch {i+1}/{len(detection_tables)} already processed ({base_filename}), skipping")
                continue

            self.logger.info(f"Processing detection table {i+1}/{len(detection_tables)} ({base_filename})")

            transients = catalog_match.find_transients_multicatalog(
                self.catalog_loader,
                config,
                self.logger,
                det_table,
                catalogs,
                params=params,
                idlimit=idlimit,
                radius_check=radius_check,
                filter_pattern=filter_pattern,
                mag_change_threshold=mag_change_threshold,
            )

            clustering.save_epoch_results(
                transients, det_table, i, min_catalogs, min_quality,
                self.data_dir, config=config, logger=self.logger,
            )

        # Step 2: Enhanced cross-matching with lightcurve data collection
        self.logger.info("Step 2: Building lightcurves...")

        all_epoch_detections = prepare_epoch_detections(detection_tables)

        final_candidates, lightcurves = clustering.combine_with_lightcurves(
            self.data_dir,
            detection_tables,
            all_epoch_detections,
            position_match_radius=position_match_radius,
            min_n_detections=min_n_detections,
            config=config,
            add_strategy_fields_fn=add_strategy_fields_fn,
        )

        # Step 3: Generate lightcurve plots and analysis
        if lightcurves:
            self.logger.info("Step 3: Generating lightcurve analysis...")
            plotting.analyze_and_plot_lightcurves(
                lightcurves, self.lightcurve_dir, config=config,
                logger=self.logger, final_candidates=final_candidates,
            )
            plotting.create_lightcurve_summary(lightcurves, final_candidates, self.lightcurve_dir)

        self.logger.info(f"Final candidates: {len(final_candidates)}")
        self.logger.info(f"Lightcurves: {len(lightcurves)}")

        return final_candidates, lightcurves
