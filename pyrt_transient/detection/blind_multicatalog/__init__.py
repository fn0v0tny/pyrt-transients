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
from pyrt_transient.detection.blind_multicatalog import forced
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
        plot_lightcurves: bool = True,
    ) -> Tuple[Table, Dict]:
        """Returns (final_candidates_table, lightcurves_dict) -- see
        detection/base.py's module docstring for why this isn't
        List[Candidate] yet.

        plot_lightcurves: Step 3 (per-candidate lightcurve plots + summary
        file) is skipped when False. Detection output is identical either
        way; the validation replay (validation/replay.py) turns it off
        because it calls run() once per epoch count and only needs the
        candidate table.
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

            if clustering.epoch_is_cached(ecsv_path):
                self.logger.info(f"Epoch {i+1}/{len(detection_tables)} already processed ({base_filename}), skipping")
                continue
            if ecsv_path.exists():
                # Written by an earlier run that lost a catalogue or SkyBoT
                # to an outage: recompute rather than inherit the outage.
                self.logger.info(f"Epoch {i+1}/{len(detection_tables)} ({base_filename}) was cached "
                                 f"as degraded, recomputing")
                ecsv_path.unlink()

            self.logger.info(f"Processing detection table {i+1}/{len(detection_tables)} ({base_filename})")

            transients, failed_catalogs = catalog_match.find_transients_multicatalog(
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
                degraded_reason=(
                    f"reference catalogue(s) failed to load: {sorted(failed_catalogs)}"
                    if failed_catalogs else None),
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
        )

        # Step 2b: stack-only candidates -- the stack is one epoch, so they
        # cannot reach min_n_detections; admit them on a forced lightcurve
        # from the stack's input frames (forced.py). No-op without a stack.
        final_candidates, lightcurves = forced.admit_stack_candidates(
            self.data_dir, detection_tables, final_candidates, lightcurves,
            position_match_radius=position_match_radius, config=config, log=self.logger,
        )

        # Step 3: Generate lightcurve plots and analysis
        if lightcurves and plot_lightcurves:
            self.logger.info("Step 3: Generating lightcurve analysis...")
            plotting.analyze_and_plot_lightcurves(
                lightcurves, self.lightcurve_dir, config=config,
                logger=self.logger, final_candidates=final_candidates,
            )
            plotting.create_lightcurve_summary(lightcurves, final_candidates, self.lightcurve_dir)

        self.logger.info(f"Final candidates: {len(final_candidates)}")
        self.logger.info(f"Lightcurves: {len(lightcurves)}")

        return final_candidates, lightcurves
