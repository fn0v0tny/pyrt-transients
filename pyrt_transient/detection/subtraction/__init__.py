"""SubtractionStrategy -- image-differencing detection, implementing the
same DetectionStrategy contract as BlindMulticatalogStrategy (see
detection/base.py and the subtraction-branch plan's A1).

Structurally mirrors BlindMulticatalogStrategy
(detection/blind_multicatalog/__init__.py) exactly: Step 1 builds a
per-epoch candidates table and hands it to clustering.save_epoch_results
(caching it to disk as `<epoch>_transients.ecsv`); Step 2 reads all of those
back via clustering.combine_with_lightcurves for cross-epoch clustering,
lightcurve construction, and final scoring; Step 3 plots lightcurves. Only
Step 1's candidate *source* differs -- blind-multicatalog cross-matches
against reference catalogs (catalog_match.py), subtraction instead builds
"new" candidates directly from diff-image detections (candidates.py) and
runs them through subtraction-specific artifact filters
(artifact_filters.py) -- everything downstream of Step 1 is unchanged,
reused code.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from astropy.table import Table

from pyrt_transient.core.epochs import prepare_epoch_detections
from pyrt_transient.io.naming import get_base_filename
from pyrt_transient.detection.base import DetectionStrategy
from pyrt_transient.detection.blind_multicatalog import clustering
from pyrt_transient.detection.blind_multicatalog import plotting
from pyrt_transient.detection.subtraction import candidates as subtraction_candidates
from pyrt_transient.detection.subtraction import artifact_filters


class SubtractionStrategy(DetectionStrategy):
    def __init__(self, data_dir, lightcurve_dir=None, config=None):
        self.data_dir = Path(data_dir)
        self.lightcurve_dir = Path(lightcurve_dir) if lightcurve_dir else self.data_dir
        self.config = config
        self.logger = logging.getLogger('detection.subtraction')
        if not self.lightcurve_dir.exists():
            self.lightcurve_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        detection_tables: List[Table],
        config=None,
        template_provenance: str = "unknown",
        position_match_radius: Optional[float] = None,
        add_strategy_fields_fn=None,
    ) -> Tuple[Table, Dict]:
        """detection_tables: diff-image detection tables, one per epoch,
        each loaded via candidates.load_diff_table (so meta['filename'] is
        set -- needed to locate both the diff FITS for the dipole filter and
        the science sibling for calibration meta, see candidates.py).

        template_provenance: recorded per-candidate (repurposing
        reference_catalog -- see candidates.py) to say which template
        produced this batch of diff images, e.g. "ps1_template" or
        "own_epoch:20260427". A single string for the whole run, since all
        epochs handed to one SubtractionStrategy.run() call share a
        template_source config (own_epoch vs. external-survey) even though
        the actual template image/date may vary per epoch in Phase B.

        Returns (final_candidates_table, lightcurves_dict) -- same shape as
        BlindMulticatalogStrategy.run(), so pipeline_magic_sn.py and the
        frontend don't need to know which strategy produced a given run.
        """
        config = config or self.config

        if config:
            min_n_detections = config.detection.min_n_detections
            min_quality = config.detection.min_quality
            dipole_radius = config.detection.dipole_reject_radius_arcsec
            dipole_flux_ratio = config.detection.dipole_reject_flux_ratio
        else:
            min_n_detections = 3
            min_quality = 0.1
            dipole_radius = 3.0
            dipole_flux_ratio = 0.5
        position_match_radius = position_match_radius or (
            config.detection.position_match_radius_arcsec if config else 2.0
        )
        # Only one "catalog" of candidates exists for subtraction (there is
        # no cross-catalog voting concept), so min_catalogs_fraction always
        # resolves to "the subtraction detections agree with themselves" --
        # trivially satisfied, unlike the blind-multicatalog path where it
        # gates on unanimous agreement across several real catalogs.
        min_catalogs = 1.0

        self.logger.info(f"Starting subtraction-strategy processing of "
                          f"{len(detection_tables)} diff-image epochs...")

        # Step 1: build subtraction candidates for each epoch, cache to disk
        # in the exact same per-epoch shape catalog_match.py produces, via
        # the same clustering.save_epoch_results used by the
        # blind-multicatalog strategy.
        self.logger.info("Step 1: Building per-epoch subtraction candidates...")

        for i, diff_table in enumerate(detection_tables):
            base_filename = get_base_filename(diff_table, i)
            ecsv_path = self.data_dir / f"{base_filename}_transients.ecsv"

            epoch_candidates = subtraction_candidates.build_epoch_candidates(
                diff_table, config=config, template_provenance=template_provenance,
            )
            if epoch_candidates is None:
                self.logger.warning(f"Epoch {i+1}/{len(detection_tables)}: "
                                     f"could not build candidates, skipping")
                continue

            if ecsv_path.exists():
                self.logger.info(f"Epoch {i+1}/{len(detection_tables)} already "
                                  f"processed ({base_filename}), skipping")
                continue

            self.logger.info(f"Processing epoch {i+1}/{len(detection_tables)} "
                              f"({base_filename}): {len(epoch_candidates)} raw diff candidates")

            diff_fits_path = Path(diff_table.meta.get("filename", "")).with_suffix(".fits")
            if diff_fits_path.exists():
                epoch_candidates, n_dipole = artifact_filters.reject_dipole_artifacts(
                    epoch_candidates, diff_fits_path,
                    radius_arcsec=dipole_radius,
                    flux_ratio_thresh=dipole_flux_ratio,
                    logger=self.logger,
                )
            else:
                self.logger.warning(f"  Dipole filter: diff FITS {diff_fits_path} "
                                     f"not found, skipping")

            epoch_candidates = artifact_filters.apply_morphology_filter(
                epoch_candidates, self.logger,
                max_ellipticity=config.detection.morphology_max_ellipticity if config else 0.4,
                fwhm_ratio_range=(
                    (config.detection.morphology_fwhm_ratio_min, config.detection.morphology_fwhm_ratio_max)
                    if config else (0.5, 2.0)
                ),
            )
            epoch_candidates = artifact_filters.apply_magnitude_filter(
                epoch_candidates, self.logger,
            )
            self.logger.info(f"  After artifact filters: {len(epoch_candidates)} candidates")

            clustering.save_epoch_results(
                {"subtraction": epoch_candidates}, diff_table, i,
                min_catalogs, min_quality, self.data_dir,
                config=config, logger=self.logger,
            )

        # Step 2: cross-epoch clustering + lightcurves, exactly as
        # BlindMulticatalogStrategy does -- combine_with_lightcurves reads
        # back the `<epoch>_transients.ecsv` files Step 1 just wrote.
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

        # Step 3: lightcurve plots and analysis.
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
