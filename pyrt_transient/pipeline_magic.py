#!/usr/bin/python3

import sys
import shutil
import json
import time
import logging
import yaml
from pyrt_transient.catalog import QueryParams, setup_catalog_cache
from pyrt_transient.io.ecsv import open_ecsv_file
from pyrt_transient.extraction_manager import ImageExtractionManager
from pyrt_transient.config_trans import PipelineConfig
from pyrt_transient.core.config_loader import load_config_with_yaml_support
from pyrt_transient.io.logging_setup import setup_pipeline_logging
from pyrt_transient.io.observation_store import ObservationStore, resolve_observation_id
from pyrt_transient.web.orchestration import generate_frontend
from pyrt_transient.detection.blind_multicatalog import BlindMulticatalogStrategy
from pyrt_transient.detection import stacking
from pyrt_transient.followup import enrichment
import os
import warnings
from pathlib import Path
from datetime import datetime

# Reduce noisy FITS header warnings from astropy (HIERARCH cards etc.)
try:
    from astropy.io.fits.verify import VerifyWarning
    warnings.filterwarnings("ignore", category=VerifyWarning)
except Exception:
    pass
try:
    from astropy.utils.exceptions import AstropyWarning
    warnings.filterwarnings("ignore", category=AstropyWarning)
except Exception:
    pass

def main():
    if len(sys.argv) < 3:
        logging.error("Usage: pipeline_magic.py <ecsv_file> <fits_file> [base_data_dir] [--generate-frontend] [--config=<config_file>] [--output-dir=<path>] [--debug]")
        logging.error("  --generate-frontend: Optional flag to generate website after analysis")
        logging.error("  --config=<file>: Optional config file path (supports .yaml/.yml)")
        logging.error("  --output-dir=<path>: Override base data directory")
        logging.error("  --debug: Enable debug logging")
        sys.exit(1)

    # Parse command line arguments
    ecsv_file = sys.argv[1]
    fits_file = sys.argv[2]

    # Parse optional arguments
    generate_frontend_flag = "--generate-frontend" in sys.argv
    debug_flag = "--debug" in sys.argv
    config_file = None
    base_data_dir = None
    output_dir = None
    
    for arg in sys.argv[3:]:
        if arg.startswith("--config="):
            config_file = arg.split("=", 1)[1]
        elif arg.startswith("--output-dir="):
            output_dir = arg.split("=", 1)[1]
        elif not arg.startswith("--"):
            base_data_dir = arg
    
    # Load configuration
    if config_file:
        config = load_config_with_yaml_support(config_file)
    else:
        config = PipelineConfig()

    # Override with command line args
    if output_dir:
        config.base_data_dir = output_dir
    elif base_data_dir:
        config.base_data_dir = base_data_dir
    if generate_frontend_flag:
        config.generate_frontend = True
    if debug_flag:
        config.logging.level = "DEBUG"
    
    # Extract observation ID first (needed for logging setup); with
    # observation_grouping_radius_arcmin set, an epoch pointing at a field
    # that already has an observation directory joins it.
    observation_id, id_reason = resolve_observation_id(
        ecsv_file, config.base_data_dir,
        radius_arcmin=config.observation_grouping_radius_arcmin,
        max_gap_hours=config.observation_grouping_max_gap_hours,
    )

    # Setup comprehensive logging
    logger = setup_pipeline_logging(config, observation_id)
    logger.info(f"=== Transient Pipeline Started ===")
    logger.info(f"Processing files: {Path(ecsv_file).name}, {Path(fits_file).name}")
    logger.info(f"Observation ID: {observation_id} ({id_reason})")
    logger.info(f"Log files will be at: {Path(config.base_data_dir) / 'logs'}")
    
    # Setup catalog cache in the user's home directory so it persists across
    # different work directories and is shared by all pipeline invocations.
    setup_catalog_cache(str(Path.home() / "catalog_cache"))

    # Validate input files
    if not Path(ecsv_file).exists():
        logger.error(f"ECSV file not found: {ecsv_file}")
        sys.exit(1)
    
    if not Path(fits_file).exists():
        logger.error(f"FITS file not found: {fits_file}")
        sys.exit(1)
    
    # Setup observation directory
    store = ObservationStore(config.base_data_dir, observation_id)
    obs_dir = store.obs_dir
    logger.info(f"Working directory: {obs_dir}")

    # Check if this specific file was already processed
    ecsv_basename = Path(ecsv_file).name
    fits_basename = Path(fits_file).name

    # Keyed on the analysis having finished, not on the epoch merely being
    # registered: an epoch registered but not yet analysed must not take the
    # "results should already exist" shortcut.
    already_processed = store.already_analyzed(ecsv_basename)
    logger.info(f"File {ecsv_basename} already analyzed: {already_processed}")
    
    if already_processed:
        logger.info(f"File {ecsv_basename} already processed for observation {observation_id}")
        logger.info("Skipping analysis - results should already exist")
        
        # Check if results exist
        candidates_file = obs_dir / "candidates.tbl"
        if candidates_file.exists():
            logger.info(f"Existing results found at: {candidates_file}")
        else:
            logger.warning("File marked as processed but no results found")
            logger.info("Will reprocess...")
            # Force reprocessing by treating as new file
            already_processed = False

        if already_processed:
            if config.generate_frontend:
                # Check if website exists, only generate if missing
                base_public_dir = config.base_public_dir or Path.home() / "public_html"
                website_dir = Path(base_public_dir) / f"obs_{observation_id}"
                if not (website_dir / "index.html").exists():
                    logger.info("Website missing, generating...")
                    generate_frontend(obs_dir, observation_id, config)
                else:
                    logger.info(f"Website already exists at: {website_dir}")
            return

    # Copy files to observation directory
    ecsv_dest = obs_dir / ecsv_basename
    fits_dest = obs_dir / fits_basename

    # Copy new files if not already present
    if not ecsv_dest.exists():
        shutil.copy(ecsv_file, ecsv_dest)
        logger.info(f"Copied {ecsv_basename} to observation directory")

    if not fits_dest.exists():
        shutil.copy(fits_file, fits_dest)
        logger.info(f"Copied {fits_basename} to observation directory")

    # Load existing detection tables and metadata
    logger.info("Loading existing detection tables...")
    detection_tables, processed_files = store.load_existing_tables()

    # Process new ECSV file
    new_detection_added = False
    new_detection = None
    try:
        new_detection = open_ecsv_file(str(ecsv_dest), verbose=True)
        if new_detection is not None and new_detection.meta is not None:
            detection_tables.append(new_detection)
            # Registered now, analysed later. Registration is what makes the
            # file visible to load_existing_tables, and it has to happen
            # before the analysis: a run that died mid-analysis used to leave
            # its ECSV sitting in obs_dir unregistered, and since the daemon
            # deletes the job directory and never re-offers a failed file,
            # that epoch was silently lost from the observation forever.
            # store.mark_analyzed below is what gates the "skip, results
            # exist" shortcut, so a failed run still re-analyses.
            store.mark_processed(ecsv_basename)
            # Publish the pointing as soon as the epoch is registered, not
            # only after a successful analysis: resolve_observation_id reads
            # these files to group OBSIDs of one field, and a store that has
            # not written one yet is invisible to a concurrently arriving
            # sibling OBSID.
            store.record_pointing(new_detection.meta)
            new_detection_added = True
            logger.info(f"Added new detection table: {ecsv_basename}")
        else:
            logger.error(f"Could not process {ecsv_basename}")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to process {ecsv_basename}: {e}")
        sys.exit(1)
    
    if not detection_tables:
        logger.error("No valid detection tables found")
        sys.exit(1)
    
    logger.info(f"Total detection tables: {len(detection_tables)}")

    # Determine if we should run the analysis
    should_analyze, reason = store.should_run_analysis(new_detection_added)
    logger.info(f"Analysis decision: {reason}")

    if should_analyze:
        with store.analysis_lock():
            # Re-check after acquiring the lock: a previous holder may have
            # already processed this epoch (incremental ecsv files exist).
            # We still run the full pipeline — Step 1 will skip processed
            # epochs automatically and Step 2 rebuilds lightcurves.

            # Reload detection tables in case another process added data
            # while we were waiting for the lock.
            detection_tables, processed_files = store.load_existing_tables()
            if ecsv_basename not in processed_files:
                detection_tables.append(new_detection)
            logger.info(f"Detection tables after lock acquired: {len(detection_tables)}")

            try:
                # Drop any previously-loaded stack table *before* anything
                # reads detection_tables[0]: a stack ECSV carries FIELD=0.0
                # (pyrt-combine writes no FIELD), and if it sorted first the
                # catalog query box below collapsed to zero width.
                # maybe_build_stack_table's return value re-adds the
                # authoritative stack afterwards (see its docstring).
                detection_tables = [t for t in detection_tables if not t.meta.get("IS_STACK")]
                first_det = detection_tables[0]

                # Field center / query params
                image_manager = ImageExtractionManager(detection_tables)
                ra, dec = image_manager.field_center
                logger.info(f"Field center: RA={ra:.6f}, DEC={dec:.6f}")
                query_params = QueryParams(
                    ra=ra, dec=dec,
                    width=1.2 * first_det.meta["FIELD"],
                    height=1.2 * first_det.meta["FIELD"],
                    mlim=20,
                )

                # Image stacking/co-addition (FUTURE_IDEAS.md): a try-harder
                # fallback that runs in parallel with, not instead of,
                # per-epoch detection below -- adds at most one extra table.
                # maybe_build_stack_table's return value is authoritative
                # (see its docstring), so any previously-loaded stack table
                # is dropped and replaced rather than appended to.
                stack_table = stacking.maybe_build_stack_table(obs_dir, detection_tables, config, logger)
                if stack_table is not None:
                    detection_tables.append(stack_table)
                    store.mark_processed(Path(stack_table.meta["filename"]).name)

                # Run the analysis (Step 1 skips already-processed epochs)
                strategy = BlindMulticatalogStrategy(data_dir=obs_dir, lightcurve_dir=obs_dir, config=config)
                reliable_candidates, lightcurves = strategy.run(
                    detection_tables,
                    config=config,
                    params=query_params,
                    idlimit=config.detection.idlimit_px,
                    radius_check=config.detection.radius_check,
                    filter_pattern=config.detection.filter_pattern,
                )
                logger.info(f"Found {len(reliable_candidates)} reliable candidates")
                if len(reliable_candidates) > 0:
                    logger.info(reliable_candidates)

                # Follow-up exposure recommendation: enrichment on the
                # strategy's output (FUTURE_IDEAS.md, "Telescope strategy
                # suggester"), not part of detection. Runs before
                # save_results so followup_exptime_s lands in candidates.tbl;
                # run_enrichment never raises, so it can't take the
                # detection results down with it.
                if config.followup.exposure_enabled:
                    enrichment.run_enrichment(
                        reliable_candidates, lightcurves, detection_tables, obs_dir,
                        config=config, logger=logger,
                    )

                store.save_results(reliable_candidates, lightcurves)
                store.mark_analyzed(ecsv_basename)
                logger.info(f"Analysis completed successfully")

            except Exception as e:
                # The traceback goes to the error log too: a one-line message
                # such as "No table found" cannot be traced without it.
                logger.error(f"Analysis failed: {e}", exc_info=True)
                sys.exit(1)

        # Frontend generation runs after the lock is released so queued
        # pipelines can start their analysis immediately.
        if config.generate_frontend:
            logger.info("Generating frontend...")
            generate_frontend(obs_dir, observation_id, config)
    else:
        candidates_file = obs_dir / "candidates.tbl"
        logger.info("No new data to process and results already exist")
        logger.info(f"Existing results: {candidates_file}")

        if config.generate_frontend:
            base_public_dir = config.base_public_dir or Path.home() / "public_html"
            website_dir = Path(base_public_dir) / f"obs_{observation_id}"
            if not (website_dir / "index.html").exists():
                logger.info("Website missing, generating...")
                generate_frontend(obs_dir, observation_id, config)
            else:
                logger.info(f"Website already exists at: {website_dir}")
    
    logger.info(f"=== Pipeline Completed Successfully ===")
    logger.info(f"Observation: {observation_id}")
    logger.info(f"Results directory: {obs_dir}")
    logger.info(f"Log files: {Path(config.base_data_dir) / 'logs' / f'pipeline_{observation_id}.log'}")

if __name__ == "__main__":
    main()
