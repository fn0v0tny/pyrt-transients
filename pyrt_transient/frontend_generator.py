import os
import shutil
import logging
import astropy.io.fits as fits
from astropy.visualization import ZScaleInterval, ImageNormalize
from astropy.table import Table
from astropy.wcs import WCS
import numpy as np
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


def _wcs_fallbacks(header):
    """Yield header variants to try for WCS creation, from most to least accurate.

    Some projection combinations (e.g. ZPN+SIP) are rejected by stricter wcslib
    builds. Fallbacks progressively strip SIP coefficients, PV distortion
    parameters, and finally replace the projection with plain TAN.
    """
    import re as _re

    yield header  # original — try first

    def _strip_sip(h):
        """Remove SIP distortion keywords and the -SIP CTYPE suffix."""
        h2 = h.copy()
        for k in ('CTYPE1', 'CTYPE2'):
            if k in h2:
                h2[k] = h2[k].replace('-SIP', '')
        # Remove SIP coefficient cards
        for key in list(h2.keys()):
            if _re.match(r'^[AB]P?_\d+_\d+$', key) or key in ('A_ORDER', 'B_ORDER', 'AP_ORDER', 'BP_ORDER'):
                del h2[key]
        return h2

    def _strip_pv(h):
        """Remove PV distortion parameters used by ZPN/ZEA/etc."""
        h2 = h.copy()
        for key in list(h2.keys()):
            if _re.match(r'^PV\d+_\d+$', key):
                del h2[key]
        return h2

    def _replace_proj_with_tan(h):
        """Replace the projection type in CTYPE with TAN, preserving padding."""
        h2 = h.copy()
        for k in ('CTYPE1', 'CTYPE2'):
            if k not in h2:
                continue
            ctype = h2[k]
            # Match: coord chars, one or more dashes, 3-letter projection, optional -SIP
            m = _re.match(r'^([A-Z]+)(-+)([A-Z]+)(-SIP)?$', ctype.strip())
            if m:
                coord, dashes, _proj, _sip = m.groups()
                h2[k] = f'{coord}{dashes}TAN'
        return h2

    has_sip = any('-SIP' in str(header.get(k, '')) for k in ('CTYPE1', 'CTYPE2'))
    proj = _re.search(r'-([A-Z]+)(?:-SIP)?$', str(header.get('CTYPE1', '')))
    proj_name = proj.group(1) if proj else ''

    # Fallback 2: strip SIP (keep original projection)
    if has_sip:
        yield _strip_sip(header)

    # Fallback 3: strip SIP + PV params, keep original projection
    if has_sip or proj_name not in ('TAN', ''):
        yield _strip_pv(_strip_sip(header))

    # Fallback 4: plain TAN (no SIP, no PV) — approximate but always works
    if proj_name not in ('TAN', ''):
        yield _replace_proj_with_tan(_strip_pv(_strip_sip(header)))


class FrontendGenerator:
    """Integrated frontend generator that uses templates and creates complete websites."""
    
    def __init__(self, observation_id, data_dir, base_public_dir=None, config=None):
        """
        Initialize the frontend generator.
        
        Args:
            observation_id: Unique observation identifier
            data_dir: Directory containing candidates.tbl and FITS files
            base_public_dir: Base directory for public websites (default: ~/public_html)
            config: FrontendConfig object with settings
        """
        self.observation_id = observation_id
        self.data_dir = Path(data_dir)
        self.config = config
        
        # Use config values if available, otherwise defaults
        if config:
            self.max_dir_size_bytes = int(config.max_dir_size_gb * 1024 * 1024 * 1024)
            self.cutout_size = config.cutout_size_px
            template_dir = config.template_dir
        else:
            self.max_dir_size_bytes = int(5.0 * 1024 * 1024 * 1024)
            self.cutout_size = 50
            template_dir = None
        
        # Set up template directory
        if template_dir:
            self.template_dir = Path(template_dir)
        else:
            self.template_dir = Path(__file__).parent / "template"
        
        # Set up public directory
        if base_public_dir is None:
            base_public_dir = Path.home() / "public_html"
        
        self.public_dir = Path(base_public_dir)
        self.output_dir = self.public_dir / f"obs_{observation_id}"
        
        # Create necessary directories
        self.setup_directories()
        
        # Initialize lightcurve requirements collection
        self.required_lightcurves = []
        
    def setup_directories(self):
        """Create all necessary directories for the website."""
        dirs_to_create = [
            self.public_dir,
            self.output_dir,
            self.output_dir / "cutouts",
            self.output_dir / "lightcurves"
        ]
        
        for directory in dirs_to_create:
            directory.mkdir(parents=True, exist_ok=True)
            
        logging.info(f"Created website directory structure at: {self.output_dir}")
    
    def sanitize_candidate_id(self, candidate_id):
        """Sanitize candidate ID to ensure filesystem-safe characters."""
        import re
        
        # Replace problematic characters with underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_\-.]', '_', str(candidate_id))
        
        # Remove multiple consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        
        # Remove leading/trailing underscores and dots
        sanitized = sanitized.strip('_.')
        
        # Ensure we have something valid
        if not sanitized:
            # Generate a hash-based fallback
            sanitized = f"cand_{abs(hash(str(candidate_id))) % 100000:05d}"
        
        # Limit length to prevent filesystem issues
        if len(sanitized) > 50:
            sanitized = sanitized[:47] + f"_{abs(hash(sanitized)) % 1000:03d}"
        
        return sanitized
    
    def get_directory_size(self, directory):
        """Calculate total size of directory in bytes."""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(directory):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
        except Exception as e:
            logging.info(f"Error calculating directory size: {e}")
        return total_size
    
    def cleanup_old_files(self, directory, keep_newest=10):
        """Remove old files to manage disk space, keeping only the newest files."""
        try:
            if not directory.exists():
                return
            
            # Get all files with their modification times and access times
            files_with_times = []
            for file_path in directory.rglob("*"):
                if file_path.is_file():
                    stat_info = file_path.stat()
                    # Use access time for LRU, modification time as secondary
                    files_with_times.append((file_path, stat_info.st_atime, stat_info.st_mtime, stat_info.st_size))
            
            # Sort by access time (LRU) then by modification time (oldest first for removal)
            files_with_times.sort(key=lambda x: (x[1], x[2]))
            
            # Remove old files beyond the keep limit
            files_removed = 0
            bytes_freed = 0
            for file_path, atime, mtime, size in files_with_times[:-keep_newest]:
                try:
                    file_path.unlink()
                    files_removed += 1
                    bytes_freed += size
                except Exception as e:
                    logging.info(f"Could not remove {file_path}: {e}")
            
            if files_removed > 0:
                logging.info(f"LRU cleanup: removed {files_removed} files, freed {bytes_freed / (1024*1024):.1f} MB from {directory}")
        
        except Exception as e:
            logging.info(f"Error during LRU cleanup: {e}")
    
    def enforce_disk_budget_strict(self, target_size_ratio=0.8):
        """Strictly enforce disk budget with aggressive LRU cleanup."""
        logger = logging.getLogger('frontend_generator')
        
        current_size = self.get_directory_size(self.output_dir)
        target_size = self.max_dir_size_bytes * target_size_ratio
        
        if current_size <= target_size:
            return True
        
        logger.warning(f"Disk budget exceeded: {current_size / (1024**3):.2f} GB > {target_size / (1024**3):.2f} GB")
        logger.info("Starting aggressive LRU cleanup...")
        
        # Priority order for cleanup: cutouts (largest), then lightcurves, then other files
        cleanup_dirs = [
            (self.output_dir / "cutouts", "cutouts"),
            (self.output_dir / "lightcurves", "lightcurves"),
            (self.output_dir, "root")
        ]
        
        for cleanup_dir, dir_name in cleanup_dirs:
            if not cleanup_dir.exists():
                continue
            
            current_size = self.get_directory_size(self.output_dir)
            if current_size <= target_size:
                break
            
            logger.info(f"Cleaning up {dir_name} directory...")
            
            # Get all files in this directory with their stats
            files_to_clean = []
            for file_path in cleanup_dir.rglob("*"):
                if file_path.is_file() and file_path.suffix not in ['.json', '.html', '.js']:  # Preserve essential files
                    stat_info = file_path.stat()
                    files_to_clean.append((file_path, stat_info.st_atime, stat_info.st_size))
            
            # Sort by access time (LRU - least recently used first)
            files_to_clean.sort(key=lambda x: x[1])
            
            # Remove files until we're under budget
            files_removed = 0
            bytes_freed = 0
            
            for file_path, atime, size in files_to_clean:
                try:
                    file_path.unlink()
                    files_removed += 1
                    bytes_freed += size
                    
                    # Check if we're now under budget
                    current_size = self.get_directory_size(self.output_dir)
                    if current_size <= target_size:
                        break
                        
                except Exception as e:
                    logger.debug(f"Could not remove {file_path}: {e}")
            
            if files_removed > 0:
                logger.info(f"Removed {files_removed} files from {dir_name}, freed {bytes_freed / (1024*1024):.1f} MB")
        
        # Final size check
        final_size = self.get_directory_size(self.output_dir)
        success = final_size <= target_size
        
        logger.info(f"Cleanup complete. Final size: {final_size / (1024**3):.2f} GB "
                   f"(target: {target_size / (1024**3):.2f} GB)")
        
        if not success:
            logger.warning("Could not reduce size to target even after aggressive cleanup")
        
        return success
    
    def check_and_manage_disk_space(self):
        """Check disk usage and clean up if necessary."""
        current_size = self.get_directory_size(self.output_dir)
        current_size_gb = current_size / (1024 * 1024 * 1024)
        
        logger = logging.getLogger('frontend_generator')
        logger.info(f"Current website size: {current_size_gb:.2f} GB")
        
        if current_size > self.max_dir_size_bytes:
            logger.warning(f"Website size exceeds limit ({self.max_dir_size_bytes / (1024**3):.1f} GB), cleaning up...")
            
            # Clean up cutouts first (usually the biggest space users)
            cutouts_dir = self.output_dir / "cutouts"
            if cutouts_dir.exists():
                self.cleanup_old_files(cutouts_dir, keep_newest=50)
            
            # Clean up lightcurves
            lc_dir = self.output_dir / "lightcurves"
            if lc_dir.exists():
                self.cleanup_old_files(lc_dir, keep_newest=20)
            
            # Check size again
            new_size = self.get_directory_size(self.output_dir)
            new_size_gb = new_size / (1024 * 1024 * 1024)
            logger.info(f"Size after cleanup: {new_size_gb:.2f} GB")
            
            return new_size <= self.max_dir_size_bytes
        
        return True
    
    def copy_template_files(self):
        """Copy and potentially customize template files."""
        template_files = ["index.html", "app.js", "chart.umd.min.js"]
        
        for template_file in template_files:
            source = self.template_dir / template_file
            dest = self.output_dir / template_file
            
            if source.exists():
                # For now, just copy directly. Later we could add template substitution
                shutil.copy2(source, dest)
                logging.info(f"Copied {template_file} to website")
            else:
                logging.info(f"WARNING: Template file {template_file} not found at {source}")
    
    def generate_candidates_data(self, max_candidates=None, max_cutouts_per_candidate=None):
        """Generate the candidates.json file with all necessary data using optimized processing."""
        logger = logging.getLogger('frontend_generator')
        
        # Use config values or defaults
        if max_candidates is None:
            max_candidates = self.config.max_candidates if self.config else 100
        if max_cutouts_per_candidate is None:
            max_cutouts_per_candidate = self.config.max_cutouts_per_candidate if self.config else 20
            
        candidates_file = self.data_dir / "candidates.tbl"
        
        if not candidates_file.exists():
            logger.error(f"Candidates file not found: {candidates_file}")
            return []
        
        try:
            for fmt in ("ascii.ecsv", "ascii.ipac"):
                try:
                    candidates = Table.read(candidates_file, format=fmt)
                    break
                except Exception:
                    continue
            else:
                raise ValueError("Neither ascii.ecsv nor ascii.ipac could parse the file")
            logger.info(f"Loading {len(candidates)} candidates from {candidates_file}")
        except Exception as e:
            logger.error(f"Could not read candidates file: {e}")
            return []
        
        # GATING: Sort candidates — prefer sn_score (SN pipeline), fall back to quality_score
        if 'sn_score' in candidates.colnames:
            sort_col = 'sn_score'
        elif 'quality_score' in candidates.colnames:
            sort_col = 'quality_score'
        else:
            sort_col = None
        if sort_col:
            sorted_indices = np.argsort(candidates[sort_col])[::-1]
            candidates = candidates[sorted_indices]
            logger.info(f"Sorted candidates by {sort_col} (best: {candidates[sort_col][0]:.3f})")
        
        # Limit number of candidates to prevent size explosion
        if len(candidates) > max_candidates:
            logger.warning(f"Too many candidates ({len(candidates)}), limiting to top {max_candidates}")
            candidates = candidates[:max_candidates]
        
        # Use optimized cutout generation
        candidates_json = self.generate_candidates_data_optimized(candidates, max_cutouts_per_candidate)
        
        # Save candidates JSON
        self.save_candidates(candidates_json)
        
        logging.info(f"Generated candidates.json with {len(candidates_json)} candidates")
        return candidates_json
    
    def save_candidates(self, candidates_json):
        """Save candidates data to candidates.json."""
        logger = logging.getLogger('frontend_generator')
        
        # Clean up any stale pagination files from previous runs
        self._cleanup_stale_pagination_files()
        
        # Apply top-N gating based on max_candidates config
        max_candidates = getattr(self.config, 'max_candidates', 200) if self.config else 200
        limited_candidates = candidates_json[:max_candidates]
        
        # Save single candidates.json file
        candidates_path = self.output_dir / "candidates.json"
        with open(candidates_path, "w") as f:
            json.dump(limited_candidates, f, indent=2)
        
        total_candidates = len(candidates_json)
        saved_candidates = len(limited_candidates)
        
        if total_candidates > saved_candidates:
            logger.info(f"Limited candidates from {total_candidates} to {saved_candidates} (max_candidates={max_candidates})")
        
        logger.info(f"Saved candidates.json with {saved_candidates} candidates")
    
    def _cleanup_stale_pagination_files(self):
        """Remove stale pagination files from previous runs."""
        logger = logging.getLogger('frontend_generator')
        
        # Remove pagination.json
        pagination_file = self.output_dir / "pagination.json"
        if pagination_file.exists():
            pagination_file.unlink()
            logger.debug("Removed stale pagination.json")
        
        # Remove all candidates_page_*.json files
        page_files = list(self.output_dir.glob("candidates_page_*.json"))
        for page_file in page_files:
            page_file.unlink()
            logger.debug(f"Removed stale {page_file.name}")
        
        if page_files:
            logger.info(f"Cleaned up {len(page_files)} stale pagination files")
    
    def generate_candidates_data_optimized(self, candidates, max_cutouts_per_candidate):
        """Optimized candidate processing with inverted loop structure."""
        logger = logging.getLogger('frontend_generator')
        
        # Find all FITS files and select subset for processing
        fits_files = sorted(self.data_dir.glob("*.fits"))
        logger.info(f"Found {len(fits_files)} FITS files")
        
        # Select frames per candidate using smart sampling
        selected_fits_files = self.select_frames_for_candidates(fits_files, max_cutouts_per_candidate)
        logger.info(f"Selected {len(selected_fits_files)} FITS files for processing")
        
        # Prepare candidate data structures
        candidates_data = {}
        for i, candidate in enumerate(candidates):
            # Use stable candidate ID: prefer transient_id, then candidate_id, then position
            if 'transient_id' in candidate.colnames and candidate['transient_id'] is not None:
                candidate_id = self.sanitize_candidate_id(str(candidate['transient_id']))
            elif 'candidate_id' in candidate.colnames and candidate['candidate_id'] is not None:
                candidate_id = self.sanitize_candidate_id(str(candidate['candidate_id']))
            else:
                # Fallback: use position coordinates with binning based on idlimit
                # Use config idlimit if available, otherwise default to 3 pixels (~2 arcsec typical)
                if self.config and hasattr(self.config, 'detection') and hasattr(self.config.detection, 'idlimit_px'):
                    # Convert pixels to arcsec (assume ~0.7 arcsec/pixel typical)
                    bin_size_arcsec = max(1, int(self.config.detection.idlimit_px * 0.7))
                else:
                    bin_size_arcsec = 2  # Default 2 arcsec bins
                
                # Bin coordinates to handle small position changes
                ra_arcsec = int(round(candidate['ALPHA_J2000'] * 3600 / bin_size_arcsec)) * bin_size_arcsec
                dec_arcsec = int(round(candidate['DELTA_J2000'] * 3600 / bin_size_arcsec)) * bin_size_arcsec
                candidate_id = f"tr_{ra_arcsec:06d}_{dec_arcsec:+07d}"
                candidate_id = self.sanitize_candidate_id(candidate_id)
            
            # Guard: if the same candidate_id was already registered (duplicate transient_id
            # in candidates.tbl), keep the first occurrence which has the higher quality_score
            # (candidates are sorted descending). Overwriting would cause cutout files
            # generated for the first candidate's pixel position to be reused under the
            # second candidate's (different) position — a cutout mismatch.
            if candidate_id in candidates_data:
                logger.warning(
                    f"Duplicate candidate_id '{candidate_id}' at index {i}, skipping "
                    f"(keeping higher quality_score entry)"
                )
                continue

            # Store direct SExtractor pixel position if present (avoids WCS for cutouts)
            colnames = candidate.colnames if hasattr(candidate, 'colnames') else []
            try:
                x_image = float(candidate["X_IMAGE"]) if "X_IMAGE" in colnames else None
                y_image = float(candidate["Y_IMAGE"]) if "Y_IMAGE" in colnames else None
            except (ValueError, TypeError):
                x_image = y_image = None

            candidates_data[candidate_id] = {
                'candidate': candidate,
                'ra': candidate["ALPHA_J2000"],
                'dec': candidate["DELTA_J2000"],
                'x_image': x_image,
                'y_image': y_image,
                'cutouts': [],
                'candidate_dict': self.candidate_to_dict(candidate, candidate_id)
            }

        # Pre-load per-epoch pixel positions from lightcurve ECSV files.
        # The SExtractor-measured pixel position in each frame is more accurate than
        # re-deriving it from a fixed sky position via WCS (especially for moving sources).
        # Map: candidate_id -> { fits_stem -> (x_px, y_px) }
        epoch_positions = {}
        for candidate_id, cand_data in candidates_data.items():
            candidate = cand_data['candidate']
            if 'transient_id' not in candidate.colnames:
                continue
            transient_id = str(candidate['transient_id'])
            lc_path = self.data_dir / f"{transient_id}_lightcurve.ecsv"
            if not lc_path.exists():
                continue
            try:
                from astropy.table import Table as _Table
                lc = _Table.read(lc_path, format='ascii.ecsv')
                positions = {}
                for row in lc:
                    # source_file stores the full path to the per-epoch ecsv; stem matches fits stem
                    src_stem = Path(str(row['source_file'])).stem
                    positions[src_stem] = (float(row['X_IMAGE']), float(row['Y_IMAGE']))
                if positions:
                    epoch_positions[candidate_id] = positions
            except Exception as e:
                logger.debug(f"Could not load epoch positions for {candidate_id}: {e}")

        logger.info(f"Pre-loaded per-epoch positions for {len(epoch_positions)}/{len(candidates_data)} candidates")

        # INVERTED LOOP: Process FITS files outermost
        for fits_file in selected_fits_files:
            logger.info(f"Processing FITS file: {fits_file.name}")
            frame_stem = fits_file.stem

            # Enforce disk budget before processing each file
            if not self.enforce_disk_budget_strict(target_size_ratio=0.7):
                logger.warning(f"Could not maintain disk budget, stopping FITS processing")
                break

            try:
                # Open FITS and read header only — defer per-pixel reads to hdul.section
                # so a truncated file doesn't prevent all cutouts (only affected rows fail).
                with fits.open(fits_file, memmap=True) as hdul:
                    header = hdul[0].header.copy()
                    naxis1 = header.get('NAXIS1', 0)
                    naxis2 = header.get('NAXIS2', 0)
                    if naxis1 == 0 or naxis2 == 0:
                        logger.warning(f"No image dimensions in FITS header {fits_file}")
                        continue

                    wcs = None
                    for _h in _wcs_fallbacks(header):
                        try:
                            wcs = WCS(_h, relax=True)
                            break
                        except Exception:
                            pass
                    if wcs is None:
                        logger.warning(
                            f"WCS failed for {fits_file.name} — will use X_IMAGE/Y_IMAGE positions"
                        )

                    # Build pixel positions for all candidates in this frame.
                    # Priority 1: per-epoch SExtractor position (from lightcurve ECSV).
                    # Priority 2: X_IMAGE/Y_IMAGE stored directly on the candidate row.
                    # Fallback:   WCS conversion from sky position (batch-vectorized).
                    candidate_ids = list(candidates_data.keys())
                    pixel_position_map = {}   # candidate_id -> (x, y)

                    needs_wcs = []
                    needs_wcs_coords = []
                    for candidate_id in candidate_ids:
                        ep = epoch_positions.get(candidate_id, {})
                        if frame_stem in ep:
                            pixel_position_map[candidate_id] = ep[frame_stem]
                        else:
                            cand_data = candidates_data[candidate_id]
                            xi, yi = cand_data.get('x_image'), cand_data.get('y_image')
                            if xi is not None and yi is not None:
                                # SExtractor pixel coords are 1-based; convert to 0-based
                                pixel_position_map[candidate_id] = (xi - 1, yi - 1)
                            else:
                                needs_wcs.append(candidate_id)
                                needs_wcs_coords.append([cand_data['ra'], cand_data['dec']])

                    if needs_wcs_coords and wcs is not None:
                        wcs_px = wcs.all_world2pix(np.array(needs_wcs_coords), 0)
                        for cid, (wx, wy) in zip(needs_wcs, wcs_px):
                            pixel_position_map[cid] = (wx, wy)

                    if not pixel_position_map:
                        continue

                    # Process cutouts for all candidates in this FITS file.
                    # Use hdul[0].section to read only the needed pixels — avoids loading
                    # the full array and gracefully handles truncated files (per-candidate).
                    for candidate_id in candidate_ids:
                        (x, y) = pixel_position_map[candidate_id]
                        try:
                            x, y = int(round(x)), int(round(y))

                            # Reject only if the position is completely outside the image
                            if x < 0 or y < 0 or x >= naxis1 or y >= naxis2:
                                continue

                            # Generate output filename
                            image_format = "webp"
                            if self.config:
                                config_format = getattr(self.config, 'image_format', 'WebP').lower()
                                if config_format in ['webp', 'jpeg', 'jpg']:
                                    image_format = config_format

                            output_filename = f"{candidate_id}_{fits_file.stem}.{image_format}"
                            output_path = self.output_dir / "cutouts" / output_filename

                            date_str = self.extract_date_from_filename(fits_file.name)

                            # Skip if already exists
                            if output_path.exists():
                                candidates_data[candidate_id]['cutouts'].append({
                                    "path": f"./cutouts/{output_filename}",
                                    "filename": fits_file.name,
                                    "date": date_str
                                })
                                continue

                            # Create cutout bounds
                            ymin = max(0, y - self.cutout_size)
                            ymax = min(naxis2, y + self.cutout_size)
                            xmin = max(0, x - self.cutout_size)
                            xmax = min(naxis1, x + self.cutout_size)

                            # Read only the cutout region (lazy — works on truncated files)
                            cutout = np.array(hdul[0].section[ymin:ymax, xmin:xmax])

                            # Save cutout with crosshairs
                            self.save_cutout_plot(cutout, x, y, xmin, ymin, output_path, candidate_id)

                            candidates_data[candidate_id]['cutouts'].append({
                                "path": f"./cutouts/{output_filename}",
                                "filename": fits_file.name,
                                "date": date_str
                            })

                        except Exception as e:
                            logger.debug(f"Error processing cutout for {candidate_id} in {fits_file.name}: {e}")
                            continue

            except Exception as e:
                logger.warning(f"Error processing FITS file {fits_file}: {e}")
                continue
        
        # Load forced photometry lightcurves if available
        forced_lcs = self._load_forced_lightcurves()

        # Assemble final candidate data
        candidates_json = []
        for candidate_id, cand_data in candidates_data.items():
            candidate_dict = cand_data['candidate_dict']
            candidate_dict["cutouts"] = cand_data['cutouts']

            # Add multi-epoch lightcurve data (from pipeline ECSV)
            lightcurve_info = self.process_lightcurve_data(cand_data['candidate'], candidate_id)
            if lightcurve_info:
                candidate_dict["lightcurve"] = lightcurve_info

            # Merge forced photometry lightcurves (PS1 + ATLAS)
            # lightcurves.json is keyed by _get_cid() — use it directly so the
            # position fallback format always matches, regardless of which columns exist.
            row = cand_data['candidate']
            try:
                from forced_photometry import _get_cid as _fp_get_cid
                lc_key = _fp_get_cid(row)
            except Exception:
                lc_key = candidate_id
            flc = forced_lcs.get(lc_key) or forced_lcs.get(candidate_id)
            if flc is not None:
                candidate_dict["forced_lc"] = flc

            candidates_json.append(candidate_dict)
        
        return candidates_json
    
    def _load_forced_lightcurves(self):
        """Load forced photometry lightcurves from lightcurves.json if present."""
        import json
        lc_file = self.data_dir / 'lightcurves.json'
        if not lc_file.exists():
            return {}
        try:
            with open(lc_file) as f:
                return json.load(f)
        except Exception as e:
            logging.getLogger('frontend_generator').warning(f"Could not load lightcurves.json: {e}")
            return {}

    def select_frames_for_candidates(self, fits_files, max_cutouts_per_candidate):
        """Select optimal subset of FITS files for candidate processing."""
        if len(fits_files) <= max_cutouts_per_candidate:
            return fits_files
        
        logger = logging.getLogger('frontend_generator')
        
        # Smart sampling: first, last, brightest (approximate), and uniform samples
        selected = []
        
        # Always include first and last
        selected.append(fits_files[0])
        if len(fits_files) > 1:
            selected.append(fits_files[-1])
        
        # Try to find brightest frame using heuristics
        brightest_file = self.find_brightest_frame_heuristic(fits_files)
        if brightest_file and brightest_file not in selected:
            selected.append(brightest_file)
            logger.debug(f"Selected brightest frame: {brightest_file.name}")
        
        # Add uniform samples from the remaining files
        remaining_slots = max_cutouts_per_candidate - len(selected)
        if remaining_slots > 0:
            # Get files not already selected
            remaining_files = [f for f in fits_files if f not in selected]
            if remaining_files:
                step = max(1, len(remaining_files) // remaining_slots)
                for i in range(0, len(remaining_files), step):
                    if len(selected) < max_cutouts_per_candidate:
                        selected.append(remaining_files[i])
        
        # Sort by filename to maintain temporal order
        selected.sort(key=lambda f: f.name)
        
        return selected[:max_cutouts_per_candidate]
    
    def find_brightest_frame_heuristic(self, fits_files):
        """Find likely brightest frame using filename heuristics or quick sampling."""
        logger = logging.getLogger('frontend_generator')
        
        # Method 1: Look for exposure time in filename
        # Longer exposures are often brighter (for same conditions)
        best_file = None
        best_score = -1
        
        for fits_file in fits_files:
            score = 0
            filename = fits_file.name.lower()
            
            # Look for exposure time indicators in filename
            # Common patterns: "30s", "exp30", "120sec", etc.
            import re
            exp_patterns = [
                r'(\d+)s[^ec]',  # "30s"
                r'exp(\d+)',     # "exp30"
                r'(\d+)sec',     # "30sec"
                r'_(\d+)_',      # "_30_"
            ]
            
            for pattern in exp_patterns:
                match = re.search(pattern, filename)
                if match:
                    try:
                        exp_time = float(match.group(1))
                        score += exp_time / 100.0  # Normalize exposure time contribution
                        break
                    except (ValueError, IndexError):
                        continue
            
            # Method 2: File size heuristic (larger files often indicate more signal)
            try:
                file_size = fits_file.stat().st_size
                score += file_size / (1024 * 1024 * 100)  # Normalize MB to 0-1 range
            except:
                pass
            
            # Method 3: Avoid bias/dark frames (often smaller or have specific naming)
            if any(keyword in filename for keyword in ['bias', 'dark', 'flat']):
                score -= 10  # Strong penalty for calibration frames
            
            # Method 4: Prefer frames from middle of sequence (often better conditions)
            total_files = len(fits_files)
            file_index = fits_files.index(fits_file)
            middle_bonus = 1.0 - abs(file_index - total_files/2) / (total_files/2)
            score += middle_bonus * 0.5
            
            if score > best_score:
                best_score = score
                best_file = fits_file
        
        if best_file:
            logger.debug(f"Brightest frame heuristic selected: {best_file.name} (score: {best_score:.2f})")
        
        return best_file
    
    def create_montage_for_candidate(self, candidate_id, cutout_data_list, grid_size=None):
        """Create a montage (grid) of cutouts for a candidate."""
        if not cutout_data_list:
            return None
        
        # Determine grid size
        num_cutouts = len(cutout_data_list)
        if grid_size is None:
            # Auto-calculate grid size
            cols = int(np.ceil(np.sqrt(num_cutouts)))
            rows = int(np.ceil(num_cutouts / cols))
        else:
            cols, rows = grid_size
        
        # Get thumbnail size from config
        thumbnail_size = getattr(self.config, 'thumbnail_size_px', 150) if self.config else 150
        
        # Create montage image
        montage_width = cols * thumbnail_size
        montage_height = rows * thumbnail_size
        montage = Image.new('L', (montage_width, montage_height), color=0)
        
        # Place cutouts in grid
        for i, (cutout_data, x, y, xmin, ymin, fits_file) in enumerate(cutout_data_list):
            if i >= cols * rows:
                break  # Don't exceed grid size
            
            row = i // cols
            col = i % cols
            
            # Normalize and convert cutout to PIL Image
            norm = ImageNormalize(cutout_data, interval=ZScaleInterval())
            normalized = norm(cutout_data)
            cutout_8bit = (normalized * 255).astype(np.uint8)
            cutout_img = Image.fromarray(cutout_8bit, mode='L')
            
            # Resize to thumbnail
            cutout_img = cutout_img.resize((thumbnail_size, thumbnail_size), Image.Resampling.LANCZOS)
            
            # Draw crosshairs
            draw = ImageDraw.Draw(cutout_img)
            scale_factor = thumbnail_size / cutout_data.shape[0]
            x_cutout = int((x - xmin) * scale_factor)
            y_cutout = int((y - ymin) * scale_factor)
            
            gap_size = max(2, thumbnail_size // 40)
            line_width = max(1, thumbnail_size // 150)
            crosshair_color = 200
            
            # Draw crosshairs (simplified for montage)
            if x_cutout - gap_size > 0:
                draw.rectangle([0, y_cutout - line_width//2, x_cutout - gap_size, y_cutout + line_width//2], 
                             fill=crosshair_color)
            if x_cutout + gap_size < thumbnail_size:
                draw.rectangle([x_cutout + gap_size, y_cutout - line_width//2, thumbnail_size, y_cutout + line_width//2], 
                             fill=crosshair_color)
            if y_cutout - gap_size > 0:
                draw.rectangle([x_cutout - line_width//2, 0, x_cutout + line_width//2, y_cutout - gap_size], 
                             fill=crosshair_color)
            if y_cutout + gap_size < thumbnail_size:
                draw.rectangle([x_cutout - line_width//2, y_cutout + gap_size, x_cutout + line_width//2, thumbnail_size], 
                             fill=crosshair_color)
            
            # Paste into montage
            paste_x = col * thumbnail_size
            paste_y = row * thumbnail_size
            montage.paste(cutout_img, (paste_x, paste_y))
        
        return montage
    
    def save_montage_for_candidate(self, candidate_id, cutout_data_list):
        """Save a montage image for a candidate."""
        if not cutout_data_list:
            return None
        
        montage = self.create_montage_for_candidate(candidate_id, cutout_data_list)
        if montage is None:
            return None
        
        # Determine output format
        image_format = "webp"
        image_quality = 85
        if self.config:
            image_format = getattr(self.config, 'image_format', 'WebP').lower()
            image_quality = getattr(self.config, 'image_quality', 85)
        
        # Save montage
        output_filename = f"{candidate_id}_montage.{image_format}"
        output_path = self.output_dir / "cutouts" / output_filename
        
        save_kwargs = {}
        if image_format.upper() in ['JPEG', 'JPG']:
            save_kwargs['quality'] = image_quality
            save_kwargs['optimize'] = True
        elif image_format.upper() == 'WEBP':
            save_kwargs['quality'] = image_quality
            save_kwargs['method'] = 6
        
        montage.save(output_path, format=image_format.upper(), **save_kwargs)
        
        logging.debug(f"Saved montage: {output_path}")
        return f"./cutouts/{output_filename}"
    
    def generate_cutouts_for_candidate(self, candidate, candidate_id, max_cutouts=20):
        """Generate cutout images for a candidate from all FITS files."""
        cutouts = []
        ra = candidate["ALPHA_J2000"]
        dec = candidate["DELTA_J2000"]
        
        # Find all FITS files and limit them
        fits_files = sorted(self.data_dir.glob("*.fits"))
        if len(fits_files) > max_cutouts:
            logging.info(f"  Limiting cutouts to {max_cutouts} most recent files (of {len(fits_files)} total)")
            fits_files = fits_files[-max_cutouts:]  # Take the most recent files
        
        for fits_file in fits_files:
            try:
                # Determine output format and filename
                image_format = "webp"
                if self.config:
                    config_format = getattr(self.config, 'image_format', 'WebP').lower()
                    if config_format in ['webp', 'jpeg', 'jpg']:
                        image_format = config_format
                
                output_filename = f"{candidate_id}_{fits_file.stem}.{image_format}"
                output_path = self.output_dir / "cutouts" / output_filename
                
                if output_path.exists():
                    # Cutout already exists, just add to list
                    date_str = self.extract_date_from_filename(fits_file.name)
                    cutouts.append({
                        "path": f"./cutouts/{output_filename}",
                        "filename": fits_file.name,
                        "date": date_str
                    })
                    logging.info(f"  Using existing cutout: {output_filename}")
                    continue
                
                cutout_data = self.create_cutout_image(fits_file, ra, dec)
                if cutout_data is None:
                    continue
                
                cutout, positions = cutout_data
                x, y, xmin, ymin = positions
                
                # Extract date from filename if possible
                date_str = self.extract_date_from_filename(fits_file.name)
                
                # Create and save the cutout plot
                self.save_cutout_plot(cutout, x, y, xmin, ymin, output_path, candidate_id)
                
                cutouts.append({
                    "path": f"./cutouts/{output_filename}",
                    "filename": fits_file.name,
                    "date": date_str
                })
                
            except Exception as e:
                logging.info(f"Error processing FITS {fits_file.name} for {candidate_id}: {e}")
        
        return cutouts
    
    def create_cutout_image(self, fits_path, ra, dec):
        """Create a cutout from a FITS file at given coordinates."""
        try:
            with fits.open(fits_path, memmap=True) as hdul:
                header = hdul[0].header.copy()
                naxis1 = header.get('NAXIS1', 0)
                naxis2 = header.get('NAXIS2', 0)
                if naxis1 == 0 or naxis2 == 0:
                    return None

                wcs = None
                for _h in _wcs_fallbacks(header):
                    try:
                        wcs = WCS(_h, relax=True)
                        break
                    except Exception:
                        pass
                if wcs is None:
                    logging.info(f"Error creating WCS from header in {fits_path}: all projections failed")
                    return None

                pixel_coords = wcs.all_world2pix(np.array([[ra, dec]]), 0)
                x, y = int(round(pixel_coords[0][0])), int(round(pixel_coords[0][1]))

                ymin = max(0, y - self.cutout_size)
                ymax = min(naxis2, y + self.cutout_size)
                xmin = max(0, x - self.cutout_size)
                xmax = min(naxis1, x + self.cutout_size)

                cutout = np.array(hdul[0].section[ymin:ymax, xmin:xmax])
                return cutout, (x, y, xmin, ymin)

        except Exception as e:
            logging.info(f"Error creating cutout from {fits_path}: {e}")
            return None
    
    def save_cutout_plot(self, cutout, x, y, xmin, ymin, output_path, candidate_id):
        """Save a cutout image with crosshairs using Pillow with enhanced quality."""
        # Enhanced normalization with multiple methods
        try:
            # Method 1: Try ZScale for optimal dynamic range
            zscale = ZScaleInterval()
            vmin, vmax = zscale.get_limits(cutout)
            
            # Fallback if ZScale fails or gives poor results
            if np.isnan(vmin) or np.isnan(vmax) or vmin >= vmax:
                # Method 2: Percentile-based normalization (more robust)
                vmin, vmax = np.percentile(cutout[np.isfinite(cutout)], [1, 99])
                
            # Further fallback for extreme cases
            if vmin >= vmax:
                # Method 3: Simple min-max with outlier rejection
                finite_data = cutout[np.isfinite(cutout)]
                if len(finite_data) > 0:
                    vmin, vmax = np.min(finite_data), np.max(finite_data)
                else:
                    vmin, vmax = 0, 1
            
            # Apply normalization with enhanced contrast
            normalized = np.clip((cutout - vmin) / (vmax - vmin), 0, 1)
            
            # Optional contrast enhancement for faint sources
            if hasattr(self.config, 'enhance_contrast') and self.config.enhance_contrast:
                # Apply gamma correction for better visibility
                gamma = getattr(self.config, 'gamma_correction', 0.8)
                normalized = np.power(normalized, gamma)
            
        except Exception as e:
            # Emergency fallback: simple normalization
            logging.warning(f"Advanced normalization failed for {candidate_id}: {e}")
            finite_data = cutout[np.isfinite(cutout)]
            if len(finite_data) > 0:
                vmin, vmax = np.percentile(finite_data, [5, 95])
                normalized = np.clip((cutout - vmin) / (vmax - vmin), 0, 1)
            else:
                normalized = np.zeros_like(cutout)
        
        # Convert to 8-bit grayscale with better precision
        cutout_8bit = (normalized * 255).astype(np.uint8)
        
        # Determine output format and quality with adaptive settings
        image_format = "WebP"
        image_quality = 85
        if self.config:
            image_format = getattr(self.config, 'image_format', 'WebP').upper()
            image_quality = getattr(self.config, 'image_quality', 85)
        
        # Set proper file extension based on format
        if image_format.upper() in ['WEBP', 'JPEG', 'JPG']:
            output_path = output_path.with_suffix(f'.{image_format.lower()}')
        else:
            output_path = output_path.with_suffix('.webp')  # Default fallback
            image_format = "WebP"
        
        # Adaptive sizing based on cutout content and config
        base_thumbnail_size = getattr(self.config, 'thumbnail_size_px', 150) if self.config else 150
        
        # Determine optimal output size based on source cutout size and content
        source_size = min(cutout.shape[0], cutout.shape[1])
        
        # If source is small, don't upscale too much
        if source_size < 50:
            thumbnail_size = min(base_thumbnail_size, 100)
        elif source_size < 100:
            thumbnail_size = min(base_thumbnail_size, 150)
        else:
            # For larger sources, allow full thumbnail size or even larger for quality
            thumbnail_size = max(base_thumbnail_size, 200)
        
        # Cap maximum size to prevent excessive file sizes
        max_size = getattr(self.config, 'max_cutout_size_px', 300) if self.config else 300
        thumbnail_size = min(thumbnail_size, max_size)
        
        # Create PIL Image
        pil_image = Image.fromarray(cutout_8bit, mode='L')
        
        # Use high-quality resampling with appropriate algorithm
        if thumbnail_size > source_size:
            # Upscaling: use LANCZOS or BICUBIC for better quality
            resample_method = Image.Resampling.LANCZOS
        else:
            # Downscaling: use LANCZOS with anti-aliasing
            resample_method = Image.Resampling.LANCZOS
        
        # Resize to thumbnail with high quality
        if cutout.shape[0] != cutout.shape[1]:
            # Handle non-square cutouts - maintain aspect ratio then crop/pad to square
            orig_height, orig_width = cutout.shape
            if orig_height > orig_width:
                new_height = thumbnail_size
                new_width = int(thumbnail_size * orig_width / orig_height)
            else:
                new_width = thumbnail_size
                new_height = int(thumbnail_size * orig_height / orig_width)
            
            pil_image = pil_image.resize((new_width, new_height), resample_method)
            
            # Create square image by centering
            square_image = Image.new('L', (thumbnail_size, thumbnail_size), color=0)
            paste_x = (thumbnail_size - new_width) // 2
            paste_y = (thumbnail_size - new_height) // 2
            square_image.paste(pil_image, (paste_x, paste_y))
            pil_image = square_image
        else:
            pil_image = pil_image.resize((thumbnail_size, thumbnail_size), resample_method)
        
        # Apply sharpening filter for better detail visibility (optional)
        if hasattr(self.config, 'apply_sharpening') and self.config.apply_sharpening:
            from PIL import ImageFilter
            # Gentle unsharp mask
            pil_image = pil_image.filter(ImageFilter.UnsharpMask(radius=1, percent=20, threshold=3))
        
        # Calculate crosshair position with better precision
        if cutout.shape[0] != cutout.shape[1]:
            # For non-square cutouts, need to account for centering
            orig_height, orig_width = cutout.shape
            if orig_height > orig_width:
                scale_factor = thumbnail_size / orig_height
                x_offset = (thumbnail_size - int(thumbnail_size * orig_width / orig_height)) // 2
                y_offset = 0
            else:
                scale_factor = thumbnail_size / orig_width
                x_offset = 0
                y_offset = (thumbnail_size - int(thumbnail_size * orig_height / orig_width)) // 2
            
            x_cutout = int((x - xmin) * scale_factor) + x_offset
            y_cutout = int((y - ymin) * scale_factor) + y_offset
        else:
            scale_factor = thumbnail_size / cutout.shape[0]
            x_cutout = int((x - xmin) * scale_factor)
            y_cutout = int((y - ymin) * scale_factor)
        
        # Ensure crosshair position is within bounds
        x_cutout = max(0, min(thumbnail_size - 1, x_cutout))
        y_cutout = max(0, min(thumbnail_size - 1, y_cutout))
        
        # Enhanced crosshair rendering
        draw = ImageDraw.Draw(pil_image)
        
        # Adaptive sizing based on thumbnail size
        gap_size = max(2, thumbnail_size // 25)  # Slightly larger gap
        line_width = max(1, thumbnail_size // 75)  # Thicker lines for visibility
        
        # Determine crosshair color based on local background
        # Sample area around crosshair to choose contrasting color
        sample_radius = min(5, gap_size)
        local_region = pil_image.crop((
            max(0, x_cutout - sample_radius),
            max(0, y_cutout - sample_radius), 
            min(thumbnail_size, x_cutout + sample_radius),
            min(thumbnail_size, y_cutout + sample_radius)
        ))
        
        # Calculate mean brightness in local region
        local_brightness = np.array(local_region).mean()
        
        # Choose contrasting color (white on dark background, dark on light background)
        if local_brightness < 128:
            crosshair_color = 255  # White on dark
            outline_color = 0      # Black outline
        else:
            crosshair_color = 0    # Black on light  
            outline_color = 255    # White outline
        
        # Draw crosshairs with outlines for better visibility
        def draw_outlined_line(x1, y1, x2, y2, width, fill_color, outline_color):
            # Draw outline (wider)
            if outline_color != fill_color:
                draw.rectangle([x1, y1 - width//2 - 1, x2, y2 + width//2 + 1], fill=outline_color)
            # Draw main line
            draw.rectangle([x1, y1 - width//2, x2, y2 + width//2], fill=fill_color)
        
        # Draw horizontal lines with better positioning
        if x_cutout - gap_size > 0:
            draw_outlined_line(0, y_cutout, x_cutout - gap_size, y_cutout, 
                             line_width, crosshair_color, outline_color)
        if x_cutout + gap_size < thumbnail_size:
            draw_outlined_line(x_cutout + gap_size, y_cutout, thumbnail_size, y_cutout,
                             line_width, crosshair_color, outline_color)
        
        # Draw vertical lines  
        if y_cutout - gap_size > 0:
            draw.rectangle([x_cutout - line_width//2 - (1 if outline_color != crosshair_color else 0), 0, 
                           x_cutout + line_width//2 + (1 if outline_color != crosshair_color else 0), y_cutout - gap_size], 
                         fill=outline_color if outline_color != crosshair_color else crosshair_color)
            draw.rectangle([x_cutout - line_width//2, 0, x_cutout + line_width//2, y_cutout - gap_size], 
                         fill=crosshair_color)
        if y_cutout + gap_size < thumbnail_size:
            draw.rectangle([x_cutout - line_width//2 - (1 if outline_color != crosshair_color else 0), y_cutout + gap_size, 
                           x_cutout + line_width//2 + (1 if outline_color != crosshair_color else 0), thumbnail_size], 
                         fill=outline_color if outline_color != crosshair_color else crosshair_color)
            draw.rectangle([x_cutout - line_width//2, y_cutout + gap_size, x_cutout + line_width//2, thumbnail_size], 
                         fill=crosshair_color)
        
        # Content-aware quality optimization
        save_kwargs = {}
        
        # Analyze image content to adjust quality settings
        image_array = np.array(pil_image)
        image_variance = np.var(image_array)
        image_mean = np.mean(image_array)
        
        # Adaptive quality based on content
        if image_variance < 100:  # Low detail image
            adjusted_quality = max(image_quality - 15, 60)  # Lower quality for uniform areas
        elif image_variance > 2000:  # High detail image  
            adjusted_quality = min(image_quality + 10, 95)  # Higher quality for detailed areas
        else:
            adjusted_quality = image_quality
        
        # Format-specific optimization
        if image_format.upper() in ['JPEG', 'JPG']:
            save_kwargs.update({
                'quality': adjusted_quality,
                'optimize': True,
                'progressive': True,  # Progressive JPEG for faster loading
                'subsampling': 0 if adjusted_quality > 85 else 2  # Better color for high quality
            })
        elif image_format.upper() == 'WEBP':
            save_kwargs.update({
                'quality': adjusted_quality,
                'method': 6,  # Best compression method
                'lossless': adjusted_quality >= 95,  # Lossless for very high quality
                'exact': False  # Allow color space conversion for smaller files
            })
            
            # Additional WebP optimizations
            if thumbnail_size <= 100:
                save_kwargs['method'] = 4  # Faster method for small images
        
        # Save the image
        pil_image.save(output_path, format=image_format, **save_kwargs)
        
        # Log file size for monitoring
        try:
            file_size = output_path.stat().st_size
            file_size_kb = file_size / 1024
            logging.debug(f"Saved {image_format} cutout: {output_path.name} "
                         f"({thumbnail_size}x{thumbnail_size}px, {file_size_kb:.1f}KB, "
                         f"quality={adjusted_quality})")
        except Exception as e:
            logging.debug(f"Saved {image_format} cutout: {output_path} ({thumbnail_size}x{thumbnail_size}px)")
    
    def extract_date_from_filename(self, filename):
        """Extract date string from filename."""
        # Try to extract date from filename (format like 20250207-359-R-020-dfsn.fits)
        try:
            parts = filename.split('-')
            if len(parts) > 0:
                date_part = parts[0]
                # Check if it looks like a date (8 digits)
                if date_part.isdigit() and len(date_part) == 8:
                    return date_part
        except:
            pass
        
        # Fallback to just the filename stem
        return Path(filename).stem
    
    def candidate_to_dict(self, candidate, candidate_id):
        """Convert an astropy table row to a dictionary with proper JSON types."""
        candidate_dict = {"id": candidate_id}
        
        for col in candidate.colnames:
            value = candidate[col]
            
            if isinstance(value, (float, np.float64, np.float32)):
                # Keep numeric JSON - use null for invalid values, not 0.0
                if np.isnan(value) or np.isinf(value):
                    candidate_dict[col] = None  # JSON null, not 0.0
                else:
                    candidate_dict[col] = float(value)
            elif isinstance(value, (int, np.int64, np.int32)):
                # Handle potential NaN in integer columns
                try:
                    float_val = float(value)
                    if np.isnan(float_val) or np.isinf(float_val):
                        candidate_dict[col] = None  # JSON null, not 0
                    else:
                        candidate_dict[col] = int(value)
                except (ValueError, OverflowError):
                    candidate_dict[col] = None
            elif isinstance(value, np.bool_):
                candidate_dict[col] = bool(value)
            else:
                # Handle string and other types
                str_value = str(value).strip()
                
                # Check for string representations of invalid values
                if str_value.lower() in ['nan', 'inf', '-inf', 'null', '', 'none']:
                    candidate_dict[col] = None  # JSON null, not "0.0" string
                else:
                    # Try to convert numeric strings to proper numbers
                    if str_value.replace('.', '').replace('-', '').replace('+', '').isdigit():
                        try:
                            # Try integer first
                            if '.' not in str_value:
                                candidate_dict[col] = int(str_value)
                            else:
                                candidate_dict[col] = float(str_value)
                        except (ValueError, OverflowError):
                            candidate_dict[col] = str_value
                    else:
                        candidate_dict[col] = str_value
        
        return candidate_dict
    
    def process_lightcurve_data(self, candidate, candidate_id):
        """Process lightcurve data if available."""
        lightcurve_info = {}
        
        # Check if transient_id is available
        if 'transient_id' not in candidate.colnames:
            return None
        
        transient_id = str(candidate['transient_id'])
        
        # Look for lightcurve files
        lightcurve_plot = self.data_dir / f"{transient_id}_lightcurve.png"
        lightcurve_data = self.data_dir / f"{transient_id}_lightcurve.ecsv"
        
        # Defer copying - collect requirements for final sync
        dest_plot = self.output_dir / "lightcurves" / f"{candidate_id}_lightcurve.png"
        
        if lightcurve_plot.exists():
            # Get source file stats for later comparison
            src_stat = lightcurve_plot.stat()
            src_stats = {
                'size': src_stat.st_size,
                'mtime': src_stat.st_mtime
            }
            
            # Record what needs to exist (defer actual copying)
            requirement = {
                'candidate_id': candidate_id,
                'transient_id': transient_id,
                'src': str(lightcurve_plot),
                'dest': str(dest_plot),
                'src_stats': src_stats
            }
            self.required_lightcurves.append(requirement)
            
            # Set reference in JSON (sync will ensure file exists)
            lightcurve_info['plot'] = f"./lightcurves/{candidate_id}_lightcurve.png"
            
            logging.debug(f"  Queued lightcurve plot for {candidate_id} -> {transient_id}")
        else:
            logging.debug(f"  No lightcurve plot found for {transient_id}")
        
        # Process lightcurve data if it exists
        if lightcurve_data.exists():
            try:
                lc_table = Table.read(lightcurve_data, format='ascii.ecsv')
                
                # Sort by obs_time to ensure proper time ordering (same as PNG plots)
                lc_table.sort('obs_time')
                
                # Determine magnitude columns to use
                if 'MAG_CALIB' in lc_table.colnames:
                    mag_col = 'MAG_CALIB'
                    mag_err_col = 'MAGERR_CALIB'
                else:
                    mag_col = 'MAG_ISO'
                    mag_err_col = 'MAGERR_ISO'
                
                # Extract data arrays
                times = lc_table['obs_time']
                mags = lc_table[mag_col]
                mag_errs = lc_table[mag_err_col]
                
                # Create validity mask - drop invalid points at source
                valid_mask = (
                    np.isfinite(times) &      # Valid obs_time
                    np.isfinite(mags) &       # Valid magnitude
                    np.isfinite(mag_errs) &   # Valid magnitude error
                    (mags > 0) &              # Positive magnitude
                    (mag_errs > 0)            # Positive magnitude error
                )
                
                # Apply mask to filter out invalid points
                valid_indices = np.where(valid_mask)[0]
                
                if len(valid_indices) == 0:
                    logging.warning(f"  No valid lightcurve points for {candidate_id}")
                    return lightcurve_info if lightcurve_info else None
                
                # Filter all arrays to valid points only
                valid_times = times[valid_mask]
                valid_mags = mags[valid_mask] 
                valid_mag_errs = mag_errs[valid_mask]
                valid_lc_table = lc_table[valid_mask]
                
                # Use same time base as PNG plots: time_hours = (times - times[0]) / 3600.0
                time_hours = (valid_times - valid_times[0]) / 3600.0
                
                # Extract key statistics from valid data only
                lightcurve_info['data'] = {
                    'n_points': len(valid_indices),
                    'n_total_points': len(lc_table),  # Include total for reference
                    'time_span_hours': float((np.max(valid_times) - np.min(valid_times)) / 3600.0),
                    'mag_range': float(np.max(valid_mags) - np.min(valid_mags)),
                    'mag_std': float(np.std(valid_mags))
                }
                
                # Build lightcurve points in sorted time order with clean JSON
                lightcurve_info['points'] = []
                
                for i, (t_hr, obs_t, mag, mag_err) in enumerate(zip(time_hours, valid_times, valid_mags, valid_mag_errs)):
                    point = {
                        'time': float(t_hr),           # Hours from first observation
                        'magnitude': float(mag),        # Magnitude value  
                        'error': float(mag_err)         # Magnitude error
                    }
                    
                    # Add epoch_id if available
                    if 'epoch_id' in valid_lc_table.colnames:
                        epoch_id = valid_lc_table['epoch_id'][i]
                        if np.isfinite(float(epoch_id)):
                            point['epoch_id'] = int(epoch_id)
                        # If epoch_id is invalid, omit it rather than setting to 0
                    
                    lightcurve_info['points'].append(point)
                
                logging.info(f"  Added {len(valid_indices)} valid lightcurve points for {candidate_id}"
                           f" (filtered out {len(lc_table) - len(valid_indices)} invalid points)")
                
            except Exception as e:
                logging.warning(f"  Error reading lightcurve data for {transient_id}: {e}")
        
        return lightcurve_info if lightcurve_info else None
    
    def copy_summary_files(self):
        """Copy essential summary files only if they exist."""
        # Only copy the enhanced summary which is most useful
        essential_files = [
            "lightcurves_summary_enhanced.png"
        ]
        
        for summary_file in essential_files:
            source_path = self.data_dir / summary_file
            dest_path = self.output_dir / "lightcurves" / summary_file
            
            if source_path.exists():
                if not dest_path.exists():
                    shutil.copy2(source_path, dest_path)
                    logging.info(f"Copied {summary_file} to website")
                else:
                    logging.info(f"Using existing {summary_file}")
    
    def sync_lightcurves(self):
        """
        Perform final sync of lightcurve plots using deferred copying system.
        
        This implements the deferred copying strategy:
        1. Load previous manifest (lightcurves_sync_manifest.json)
        2. Build current desired map from self.required_lightcurves
        3. Compare timestamps/sizes to decide what needs copying
        4. Perform parallel copy/link operations with atomicity
        5. Update manifest with new state
        """
        logger = logging.getLogger('frontend_generator')
        manifest_path = self.output_dir / "lightcurves_sync_manifest.json"
        
        # Step 1: Load previous manifest
        previous_manifest = {}
        if manifest_path.exists():
            try:
                with open(manifest_path, 'r') as f:
                    previous_manifest = json.load(f)
                logger.debug(f"Loaded previous manifest with {len(previous_manifest)} entries")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Could not load previous manifest: {e}")
        
        # Step 2: Build current desired map from requirements
        current_desired = {}
        for req in self.required_lightcurves:
            # Accept both key styles robustly
            source_path = req.get('source_path') or req.get('src')
            dest_path = req.get('dest_path') or req.get('dest')
            
            # Guard against missing keys
            if not source_path or not dest_path:
                logger.warning(f"Lightcurve requirement missing source_path/src or dest_path/dest: {req}")
                continue
            
            # Get current source file stats
            if os.path.exists(source_path):
                stat = os.stat(source_path)
                current_desired[dest_path] = {
                    'source_path': source_path,
                    'mtime': stat.st_mtime,
                    'size': stat.st_size,
                    'checksum': None  # Will compute if needed
                }
            else:
                logger.warning(f"Required lightcurve source missing: {source_path}")
        
        # Step 3: Determine what needs copying/linking
        copy_jobs = []
        for dest_path, current_info in current_desired.items():
            needs_copy = False
            
            # Check if destination exists and matches previous manifest
            if dest_path in previous_manifest:
                prev_info = previous_manifest[dest_path]
                if (os.path.exists(dest_path) and 
                    prev_info.get('mtime') == current_info['mtime'] and 
                    prev_info.get('size') == current_info['size']):
                    # File unchanged, skip
                    continue
            
            # Check if destination exists but not in manifest (external change)
            if os.path.exists(dest_path):
                dest_stat = os.stat(dest_path)
                if (dest_stat.st_mtime == current_info['mtime'] and 
                    dest_stat.st_size == current_info['size']):
                    # File exists and matches source, just update manifest
                    continue
            
            # Need to copy/link
            copy_jobs.append({
                'source_path': current_info['source_path'],
                'dest_path': dest_path,
                'mtime': current_info['mtime'],
                'size': current_info['size']
            })
        
        # Step 4: Perform parallel copy operations
        if copy_jobs:
            logger.info(f"Syncing {len(copy_jobs)} lightcurve files")
            
            def copy_file_atomically(job):
                """Copy a single file with atomic operation."""
                source_path = job['source_path']
                dest_path = job['dest_path']
                temp_path = dest_path + '.tmp'
                
                try:
                    # Get link mode from config
                    link_mode = getattr(self.config, 'lightcurve_link_mode', 'auto') if self.config else 'auto'
                    
                    # Ensure destination directory exists
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    
                    if link_mode == 'hardlink':
                        # Try hardlink first
                        try:
                            os.link(source_path, temp_path)
                        except OSError:
                            # Fallback to copy
                            shutil.copy2(source_path, temp_path)
                    elif link_mode == 'symlink':
                        # Try symlink first
                        try:
                            os.symlink(source_path, temp_path)
                        except OSError:
                            # Fallback to copy
                            shutil.copy2(source_path, temp_path)
                    else:
                        # Direct copy (auto mode or copy mode)
                        shutil.copy2(source_path, temp_path)
                    
                    # Atomic rename
                    os.rename(temp_path, dest_path)
                    
                    return {'status': 'success', 'dest_path': dest_path}
                    
                except Exception as e:
                    # Cleanup temp file if it exists
                    if os.path.exists(temp_path):
                        try:
                            os.remove(temp_path)
                        except:
                            pass
                    
                    return {'status': 'error', 'dest_path': dest_path, 'error': str(e)}
            
            # Use ThreadPoolExecutor for parallel copying
            max_workers = getattr(self.config, 'lightcurve_workers', 6) if self.config else 6
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all copy jobs
                future_to_job = {executor.submit(copy_file_atomically, job): job 
                                for job in copy_jobs}
                
                # Process results
                success_count = 0
                error_count = 0
                
                for future in as_completed(future_to_job):
                    result = future.result()
                    if result['status'] == 'success':
                        success_count += 1
                    else:
                        error_count += 1
                        logger.warning(f"Failed to copy {result['dest_path']}: {result['error']}")
                
                logger.info(f"Lightcurve sync completed: {success_count} success, {error_count} errors")
        
        # Step 5: Build and save updated manifest
        updated_manifest = {}
        for dest_path, info in current_desired.items():
            if os.path.exists(dest_path):
                # Update with actual destination file stats for accuracy
                dest_stat = os.stat(dest_path)
                updated_manifest[dest_path] = {
                    'source_path': info['source_path'],
                    'mtime': info['mtime'],  # Source mtime for comparison
                    'size': info['size'],    # Source size for comparison
                    'synced_at': time.time()
                }
        
        # Step 6: Optional cleanup of orphaned files
        if getattr(self.config, 'cleanup_orphaned_lightcurves', True) if self.config else True:
            self._cleanup_orphaned_lightcurves(updated_manifest)
        
        # Step 7: Save updated manifest
        try:
            with open(manifest_path, 'w') as f:
                json.dump(updated_manifest, f, indent=2)
            logger.debug(f"Saved updated manifest with {len(updated_manifest)} entries")
        except IOError as e:
            logger.warning(f"Could not save manifest: {e}")
    
    def _cleanup_orphaned_lightcurves(self, current_manifest):
        """Remove lightcurve files that are no longer needed."""
        logger = logging.getLogger('frontend_generator')
        lightcurves_dir = self.output_dir / "lightcurves"
        
        if not lightcurves_dir.exists():
            return
        
        # Find all .png files in lightcurves directory
        existing_files = set()
        for png_file in lightcurves_dir.glob("*.png"):
            existing_files.add(str(png_file))
        
        # Determine which files are still needed
        needed_files = set(current_manifest.keys())
        
        # Remove orphaned files
        orphaned_files = existing_files - needed_files
        
        for orphaned_file in orphaned_files:
            try:
                os.remove(orphaned_file)
                logger.debug(f"Removed orphaned lightcurve: {os.path.basename(orphaned_file)}")
            except OSError as e:
                logger.warning(f"Could not remove orphaned file {orphaned_file}: {e}")
        
        if orphaned_files:
            logger.info(f"Cleaned up {len(orphaned_files)} orphaned lightcurve files")
    
    def generate_complete_website(self):
        """Generate the complete website with all components."""
        logger = logging.getLogger('frontend_generator')
        logger.info(f"=== Generating website for observation {self.observation_id} ===")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Website directory: {self.output_dir}")
        
        try:
            # Step 0: Check initial disk space and clean up if needed
            if not self.check_and_manage_disk_space():
                logger.error("Cannot free enough disk space, skipping website update")
                return False
            
            # Step 1: Copy template files
            self.copy_template_files()
            
            # Step 2: Generate candidates data and images
            candidates = self.generate_candidates_data()
            
            # Step 3: Copy summary files
            self.copy_summary_files()
            
            # Step 4: Create a simple info file
            self.create_info_file(candidates)
            
            # Step 5: Final disk space check - skip if too large
            final_size_ok = self.check_and_manage_disk_space()
            if not final_size_ok:
                logger.error("Website size exceeds limit after generation, skipping further updates")
                return False
            
            # Step 6: Sync lightcurves using deferred copying
            self.sync_lightcurves()
            
            logger.info(f"Website generated successfully!")
            logger.info(f"Website location: {self.output_dir}")
            logger.info(f"Access via: file://{self.output_dir}/index.html")
            
            return True
            
        except Exception as e:
            logging.info(f"Error generating website: {e}")
            return False
    
    def create_info_file(self, candidates):
        """Create a simple info file with observation details."""
        info = {
            "observation_id": self.observation_id,
            "generated_at": str(Path().cwd()),
            "total_candidates": len(candidates),
            "data_directory": str(self.data_dir),
            "website_directory": str(self.output_dir)
        }
        
        with open(self.output_dir / "info.json", "w") as f:
            json.dump(info, f, indent=2)
    
    def get_quality_config_recommendations(self):
        """Get recommended configuration settings for optimal quality/size balance."""
        return {
            'image_format': 'webp',  # WebP for best compression
            'image_quality': 85,     # Good quality/size balance
            'thumbnail_size_px': 200, # Larger thumbnails for better detail
            'max_cutout_size_px': 300, # Cap to prevent huge files
            'enhance_contrast': True,   # Better visibility for faint sources
            'gamma_correction': 0.8,    # Enhance faint details
            'apply_sharpening': False,  # Usually not needed, can add noise
        }
    
    def optimize_for_mobile(self):
        """Adjust settings for mobile-optimized cutouts (smaller, faster loading)."""
        if self.config:
            self.config.thumbnail_size_px = min(getattr(self.config, 'thumbnail_size_px', 150), 150)
            self.config.image_quality = max(getattr(self.config, 'image_quality', 85) - 10, 70)
            self.config.max_cutout_size_px = 200