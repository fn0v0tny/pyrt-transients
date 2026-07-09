import math
import os
from pathlib import Path
import warnings
import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from itertools import chain
from astropy.table import Table, vstack
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import SkyCoord
from sklearn.neighbors import KDTree
from typing import List, Dict, Optional

from pyrt_transient.catalog import Catalog, QueryParams
from pyrt_transient.io.naming import get_base_filename
from pyrt_transient.detection.blind_multicatalog.catalog_query import CatalogLoader
from pyrt_transient.detection.blind_multicatalog import catalog_match
from pyrt_transient.detection.blind_multicatalog import clustering
from pyrt_transient.detection.blind_multicatalog import plotting
from collections import defaultdict


class UnionFind:
    """Union-Find (Disjoint Set) data structure for efficient connected components."""
    
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.n_components = n
    
    def find(self, x: int) -> int:
        """Find with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x: int, y: int) -> bool:
        """Union by rank. Returns True if components were merged."""
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False
        
        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
        
        self.n_components -= 1
        return True
    
    def get_components(self) -> Dict[int, List[int]]:
        """Get all connected components as dict of root -> [members]."""
        components = defaultdict(list)
        for i in range(len(self.parent)):
            root = self.find(i)
            components[root].append(i)
        return dict(components)





class OptimizedTransientAnalyzer:
    """
    Enhanced TransientAnalyzer that uses optimized catalog functions.
    Drop-in replacement for your existing TransientAnalyzer.
    """

    def __init__(self, data_dir="/home/fnovotny/transient_work/", config=None) -> None:
        self.data_dir = Path(data_dir)
        self.config = config
        self.logger = logging.getLogger('transient_analyser.optimized')

        # Track loaded catalogs to avoid reloading
        self._catalog_loader = CatalogLoader()


    def find_transients_multicatalog(
        self,
        detections,
        catalogs,
        params=None,
        idlimit=5.0,
        radius_check=30.0,
        filter_pattern=None,
        mag_change_threshold=1.0,
    ):
        """Enhanced version with better error handling."""
        return catalog_match.find_transients_multicatalog(
            self._catalog_loader,
            self.config,
            self.logger,
            detections,
            catalogs,
            params=params,
            idlimit=idlimit,
            radius_check=radius_check,
            filter_pattern=filter_pattern,
            mag_change_threshold=mag_change_threshold,
        )

    def reject_by_catalog(
        self,
        candidates: Table,
        catalog_name: str,
        params=None,
        match_radius_arcsec: float = 3.0,
    ):
        """Remove candidates that have a counterpart in *catalog_name*.

        Useful for deep rejection catalogs (e.g. Legacy Survey DR10) that are too
        large to include in the multi-catalog framework as a voting member but whose
        presence definitively marks a source as a known persistent object.

        Args:
            candidates: Table of transient candidates (must have ALPHA_J2000/DELTA_J2000).
            catalog_name: Catalog identifier understood by Catalog (e.g. 'legacysurvey').
            params: QueryParams for the catalog query.  If None the method tries to
                    derive ra/dec/width from the candidates table itself.
            match_radius_arcsec: Cross-match radius in arcseconds.

        Returns:
            Tuple (filtered_candidates, n_rejected).
        """
        from scipy.spatial import cKDTree

        if len(candidates) == 0:
            return candidates, 0

        # Derive query params from candidate positions if not supplied
        if params is None:
            from catalog import QueryParams
            ras = np.array(candidates["ALPHA_J2000"], dtype=float)
            decs = np.array(candidates["DELTA_J2000"], dtype=float)
            ra_c = float(np.mean(ras))
            dec_c = float(np.mean(decs))
            width = float((np.max(ras) - np.min(ras)) + 0.5)
            height = float((np.max(decs) - np.min(decs)) + 0.5)
            params = QueryParams(ra=ra_c, dec=dec_c, width=width, height=height)

        try:
            catalog = self._catalog_loader.get_optimized_catalog(catalog_name, params)
        except Exception as e:
            self.logger.warning(f"Could not load {catalog_name} for rejection: {e}")
            return candidates, 0

        if len(catalog) == 0:
            self.logger.info(f"{catalog_name}: empty, no candidates rejected")
            return candidates, 0

        # Build KDTree on catalog positions (degrees → approx arcsec via cos(dec))
        cos_dec = np.cos(np.radians(params.dec))
        cat_ra = np.array(catalog["radeg"], dtype=float)
        cat_dec = np.array(catalog["decdeg"], dtype=float)
        cat_xy = np.column_stack([cat_ra * cos_dec, cat_dec])
        tree = cKDTree(cat_xy)

        cand_ra = np.array(candidates["ALPHA_J2000"], dtype=float)
        cand_dec = np.array(candidates["DELTA_J2000"], dtype=float)
        cand_xy = np.column_stack([cand_ra * cos_dec, cand_dec])

        radius_deg = match_radius_arcsec / 3600.0
        matches = tree.query_ball_point(cand_xy, r=radius_deg)
        keep = np.array([len(m) == 0 for m in matches])

        n_rejected = int(np.sum(~keep))
        self.logger.info(
            f"{catalog_name} rejection: {n_rejected}/{len(candidates)} candidates "
            f"matched within {match_radius_arcsec}\" — removed"
        )
        return candidates[keep], n_rejected

    def clear_catalog_cache(self) -> None:
        """Clear loaded catalogs to free memory."""
        self._catalog_loader.clear_cache()


class OptimizedMultiDetectionAnalyzer:
    """
    Enhanced MultiDetectionAnalyzer using optimized catalog functions.
    Drop-in replacement for your existing MultiDetectionAnalyzer.
    """

    def __init__(self, transient_analyzer: OptimizedTransientAnalyzer, lightcurve_dir="lightcurves", config=None):
        """
        Initialize the enhanced analyzer.
        
        Args:
            transient_analyzer: Instance of OptimizedTransientAnalyzer
            lightcurve_dir: Directory to save lightcurve data and plots
            config: Optional configuration object
        """
        self.transient_analyzer = transient_analyzer
        self.lightcurve_dir = Path(lightcurve_dir)
        self.config = config
        self.logger = logging.getLogger('transient_analyser.multi_detection')
        if not self.lightcurve_dir.exists():
            self.lightcurve_dir.mkdir(parents=True, exist_ok=True)

    def process_detection_tables_with_lightcurves(
        self,
        detection_tables: List[Table],
        catalogs: List[str],
        params: Optional[QueryParams] = None,
        idlimit: float = 5.0,
        radius_check: float = 30.0,
        filter_pattern: Optional[str] = None,
        min_catalogs: float = 1.0,
        min_quality: float = 0.1,
        position_match_radius: float = 2.0,
        min_n_detections: int = 3,
        mag_change_threshold: float = 1.0,
    ) -> Tuple[Table, Dict]:
        """
        Enhanced processing with optimized catalog operations and config support.

        Args:
            detection_tables: List of detection tables to process
            catalogs: List of catalog names to use
            params: Query parameters for catalogs
            idlimit: Identification limit in pixels
            radius_check: Radius for context checking in pixels
            filter_pattern: Pattern to match filters
            min_catalogs: Fraction (0.0–1.0) of catalogs that must flag a source
            min_quality: Minimum quality score to include
            position_match_radius: Radius for position matching in arcsec (overridden by config)
            min_n_detections: Minimum number of detections for valid transient
            mag_change_threshold: Magnitude change threshold for variability
            
        Returns:
            Tuple of (final_candidates_table, lightcurves_dict)
        """
        
        # Use config values if available
        if self.config:
            position_match_radius = self.config.detection.position_match_radius_arcsec
            min_n_detections = self.config.detection.min_n_detections
            min_catalogs = self.config.detection.min_catalogs_fraction
            min_quality = self.config.detection.min_quality
            
        self.logger.info(f"Using position_match_radius: {position_match_radius} arcsec from config")
        self.logger.info(f"Starting processing of {len(detection_tables)} detection tables...")
        
        # Step 1: Process each detection table efficiently (incremental)
        self.logger.info("Step 1: Processing individual detection tables...")

        for i, det_table in enumerate(detection_tables):
            base_filename = get_base_filename(det_table, i)
            ecsv_path = self.transient_analyzer.data_dir / f"{base_filename}_transients.ecsv"

            if ecsv_path.exists():
                self.logger.info(f"Epoch {i+1}/{len(detection_tables)} already processed ({base_filename}), skipping")
                continue

            self.logger.info(f"Processing detection table {i+1}/{len(detection_tables)} ({base_filename})")

            # Use optimized transient analyzer
            transients = self.transient_analyzer.find_transients_multicatalog(
                det_table,
                catalogs,
                params,
                idlimit,
                radius_check,
                filter_pattern,
                mag_change_threshold=mag_change_threshold
            )

            # Save results for this epoch
            self._save_epoch_results(transients, det_table, i, min_catalogs, min_quality)
        
        # Step 2: Enhanced cross-matching with lightcurve data collection
        self.logger.info("Step 2: Building lightcurves...")
        
        all_epoch_detections = self._prepare_epoch_detections(detection_tables)
        
        final_candidates, lightcurves = self._combine_with_lightcurves(
            detection_tables=detection_tables,
            all_epoch_detections=all_epoch_detections,
            position_match_radius=position_match_radius,
            min_n_detections=min_n_detections
        )
        
        # Step 3: Generate lightcurve plots and analysis
        if lightcurves:
            self.logger.info("Step 3: Generating lightcurve analysis...")
            plotting.analyze_and_plot_lightcurves(
                lightcurves, self.lightcurve_dir, config=self.config,
                logger=self.logger, final_candidates=final_candidates,
            )
            plotting.create_lightcurve_summary(lightcurves, final_candidates, self.lightcurve_dir)
        
        self.logger.info(f"Final candidates: {len(final_candidates)}")
        self.logger.info(f"Lightcurves: {len(lightcurves)}")
        
        return final_candidates, lightcurves

    def _save_epoch_results(
        self,
        transients: Dict[str, Table],
        det_table: Table,
        epoch_index: int,
        min_catalogs: float,
        min_quality: float
    ) -> None:
        """Save results for this epoch using existing combine_results function."""
        config = getattr(self.transient_analyzer, 'config', None)
        clustering.save_epoch_results(
            transients, det_table, epoch_index, min_catalogs, min_quality,
            self.transient_analyzer.data_dir, config=config, logger=self.logger,
        )

    def _prepare_epoch_detections(self, detection_tables: List[Table]) -> List[Table]:
        """Prepare epoch detection data with timing information."""
        all_epoch_detections = []
        
        for i, det_table in enumerate(detection_tables):
            # Extract timing information
            ctime = det_table.meta.get('CTIME', 0)
            exptime = det_table.meta.get('EXPTIME', 0)
            mid_time = ctime + exptime / 2.0
            
            # Add epoch information to detection table
            det_table_copy = det_table.copy()
            det_table_copy['epoch_id'] = i
            det_table_copy['obs_time'] = mid_time
            det_table_copy['mjd'] = self._unix_to_mjd(mid_time)
            det_table_copy['source_file'] = det_table.meta.get('filename', f'epoch_{i}')
            
            all_epoch_detections.append(det_table_copy)
        
        return all_epoch_detections

    def _combine_with_lightcurves(
        self,
        detection_tables,
        all_epoch_detections,
        position_match_radius: float = 2.0,
        min_n_detections: int = 3,
    ):
        """Combine candidates and build lightcurves (same logic as before)."""
        return clustering.combine_with_lightcurves(
            self.transient_analyzer.data_dir,
            detection_tables,
            all_epoch_detections,
            position_match_radius=position_match_radius,
            min_n_detections=min_n_detections,
            config=self.config,
            add_strategy_fields_fn=self._add_strategy_fields,
        )

    def _add_strategy_fields(self, candidate: Table, lightcurve: Table):
        """
        Add strategy calculation fields to candidate based on previous observation.
        
        Args:
            candidate: Single-row candidate table to update
            lightcurve: Full lightcurve table for this candidate
        """
        # Initialize all strategy fields with null values
        strategy_fields = {
            'strategy_config': None,
            'strategy_exp_s': None,
            'strategy_snr': None,
            'strategy_filters': None,
            'strategy_emccd': None,
            'strategy_prev_frame': None,
            'strategy_ecsv': None,
            'strategy_time_since_trigger_s': None,
            'strategy_sky_1s': None,
            'strategy_fwhm_px': None,
            'strategy_magzero_1s': None
        }
        
        try:
            # Require at least 2 lightcurve points
            if len(lightcurve) < 2:
                for field, value in strategy_fields.items():
                    candidate[field] = [value]
                return
            
            # Sort lightcurve by observation time
            if 'obs_time' in lightcurve.colnames:
                sorted_lc = lightcurve[np.argsort(lightcurve['obs_time'])]
            elif 'mjd' in lightcurve.colnames:
                sorted_lc = lightcurve[np.argsort(lightcurve['mjd'])]
            else:
                # Use epoch_id as fallback
                sorted_lc = lightcurve[np.argsort(lightcurve['epoch_id'])]
            
            # Get indices for latest (L) and previous (P = L-1) observations
            latest_idx = len(sorted_lc) - 1
            prev_idx = latest_idx - 1
            
            prev_row = sorted_lc[prev_idx]
            
            # Get magnitude from previous observation
            if 'MAG_CALIB' not in prev_row.colnames:
                self.logger.debug("No MAG_CALIB in previous observation, skipping strategy calculation")
                for field, value in strategy_fields.items():
                    candidate[field] = [value]
                return
            
            magnitude = float(prev_row['MAG_CALIB'])
            
            # Calculate time since trigger
            time_since_trigger_s = None
            
            # Try to get GRB T0 from config or metadata
            grb_t0 = None
            if self.config and hasattr(self.config, 'grb_t0'):
                grb_t0 = self.config.grb_t0
            elif hasattr(sorted_lc, 'meta') and 'grb_t0' in sorted_lc.meta:
                grb_t0 = sorted_lc.meta['grb_t0']
            
            if grb_t0 is not None and 'obs_time' in prev_row.colnames:
                time_since_trigger_s = float(prev_row['obs_time']) - grb_t0
            else:
                # Fallback: relative time from first observation
                if 'obs_time' in sorted_lc.colnames:
                    first_time = float(sorted_lc['obs_time'][0])
                    prev_time = float(prev_row['obs_time'])
                    time_since_trigger_s = prev_time - first_time
                else:
                    time_since_trigger_s = 3600.0  # Default 1 hour
            
            # Get previous frame's source file and derive ECSV path
            prev_ecsv_path = None
            if 'source_file' in prev_row.colnames:
                source_file = str(prev_row['source_file'])
                strategy_fields['strategy_prev_frame'] = source_file
                
                # Try to find corresponding ECSV file
                data_dir = self.transient_analyzer.data_dir
                
                # Method 1: Direct match
                ecsv_candidate = data_dir / f"{source_file}.ecsv"
                if ecsv_candidate.exists():
                    prev_ecsv_path = str(ecsv_candidate)
                else:
                    # Method 2: Glob search
                    import glob
                    glob_pattern = str(data_dir / f"{source_file}*.ecsv")
                    matches = glob.glob(glob_pattern)
                    if matches:
                        prev_ecsv_path = matches[0]
                    else:
                        # Method 3: Fallback to image.ecsv
                        fallback_path = data_dir / "image.ecsv"
                        if fallback_path.exists():
                            prev_ecsv_path = str(fallback_path)
            
            if prev_ecsv_path is None:
                self.logger.debug("Could not find ECSV file for previous frame, skipping strategy calculation")
                for field, value in strategy_fields.items():
                    candidate[field] = [value]
                return
            
            strategy_fields['strategy_ecsv'] = prev_ecsv_path
            strategy_fields['strategy_time_since_trigger_s'] = time_since_trigger_s
            
            # Import and call strategy calculator
            try:
                from strategy_v2 import determine_grb_strategy
                
                strategy_result = determine_grb_strategy(
                    magnitude=magnitude,
                    time_since_trigger=time_since_trigger_s,
                    ecsv_file=prev_ecsv_path
                )
                
                # Extract strategy results
                strategy_fields['strategy_config'] = strategy_result.get('config_name')
                strategy_fields['strategy_exp_s'] = strategy_result.get('exp_time')
                strategy_fields['strategy_snr'] = strategy_result.get('snr')
                strategy_fields['strategy_filters'] = strategy_result.get('num_filters')
                strategy_fields['strategy_emccd'] = strategy_result.get('use_emccd')
                strategy_fields['strategy_magzero_1s'] = strategy_result.get('magzero_1s')
                
                # Extract background conditions if available
                if 'background_conditions' in strategy_result:
                    bg_conditions = strategy_result['background_conditions']
                    # Parse conditions string like 'sky_1s=11.5 ph/s/px, FWHM=2.1px'
                    try:
                        import re
                        sky_match = re.search(r'sky_1s=([0-9.]+)', bg_conditions)
                        fwhm_match = re.search(r'FWHM=([0-9.]+)', bg_conditions)
                        
                        if sky_match:
                            strategy_fields['strategy_sky_1s'] = float(sky_match.group(1))
                        if fwhm_match:
                            strategy_fields['strategy_fwhm_px'] = float(fwhm_match.group(1))
                    except Exception as parse_error:
                        self.logger.debug(f"Could not parse background conditions: {parse_error}")
                
                self.logger.debug(f"Strategy calculated: {strategy_result.get('config_name')} "
                                f"({strategy_result.get('exp_time'):.1f}s, SNR={strategy_result.get('snr'):.1f})")
                
            except ImportError as e:
                self.logger.warning(f"Could not import strategy_v2: {e}")
            except Exception as e:
                self.logger.warning(f"Strategy calculation failed: {e}")
        
        except Exception as e:
            self.logger.debug(f"Error in strategy field calculation: {e}")
        
        finally:
            # Always add all fields (with None values if calculation failed)
            for field, value in strategy_fields.items():
                candidate[field] = [value]

    def _unix_to_mjd(self, unix_time):
        """Convert Unix timestamp to Modified Julian Date."""
        try:
            t = Time(unix_time, format='unix')
            return t.mjd
        except:
            return unix_time
