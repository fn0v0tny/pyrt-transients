"""ObservationStore -- moved (not copied) from pipeline_magic.py:
extract_observation_id, clean_observation_id, setup_observation_directory,
get_existing_detection_tables, update_metadata, check_if_already_processed,
AnalysisLock, should_run_analysis. These had no other callers to preserve,
so the free functions became bound methods directly rather than being kept
as wrappers.

ObservationStore(base_dir, observation_id)
  .already_processed(filename) -> bool
  .mark_processed(filename) -> None
  .load_existing_tables() -> (tables, processed_set)
  .should_run_analysis(new_detection_added) -> (bool, str)
  .analysis_lock() -> context manager
  .save_results(candidates, lightcurves) -> None

extract_observation_id/clean_observation_id stay as module-level functions
(not methods) since they're used to derive the observation_id *before* an
ObservationStore can be constructed.

save_results moves the candidates.tbl/lightcurve_summary.json writing logic
out of pipeline_magic.py's process_observation, which no longer exists as a
free function in that file -- its analysis logic is now
BlindMulticatalogStrategy.run().
"""

import fcntl
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Set, Tuple

import numpy as np
from astropy.table import Table

from pyrt_transient.transients import open_ecsv_file


def extract_observation_id(ecsv_file_path):
    """Extract observation ID from ECSV file metadata or filename."""
    try:
        # Try to get observation ID from file metadata first
        ecsv_data = open_ecsv_file(ecsv_file_path, verbose=False)
        if ecsv_data and ecsv_data.meta:
            # Check common observation ID fields
            for field in ['OBSID', 'OBS_ID', 'OBSERVATION_ID', 'FIELD_ID']:
                if field in ecsv_data.meta:
                    obs_id = str(ecsv_data.meta[field])
                    # Clean observation ID - remove decimal part
                    return clean_observation_id(obs_id)

        # Fallback: extract from filename (assuming pattern like obs_12345_...)
        filename = Path(ecsv_file_path).stem
        parts = filename.split('_')
        for i, part in enumerate(parts):
            if part.lower() in ['obs', 'obsid', 'field'] and i + 1 < len(parts):
                obs_id = parts[i + 1]
                # Clean observation ID - remove decimal part
                return clean_observation_id(obs_id)

        # Last resort: use filename without extension
        return clean_observation_id(filename)

    except Exception as e:
        logging.warning(f"Could not extract observation ID from {ecsv_file_path}: {e}")
        return clean_observation_id(Path(ecsv_file_path).stem)


def clean_observation_id(obs_id):
    """Clean observation ID by removing decimal parts and invalid characters."""
    obs_id = str(obs_id).strip()

    # Remove decimal part (e.g., 94249.01 -> 94249)
    if '.' in obs_id:
        obs_id = obs_id.split('.')[0]

    # Remove any other problematic characters and keep only alphanumeric and underscores
    obs_id = re.sub(r'[^a-zA-Z0-9_]', '_', obs_id)

    # Remove multiple consecutive underscores
    obs_id = re.sub(r'_+', '_', obs_id)

    # Remove leading/trailing underscores
    obs_id = obs_id.strip('_')

    # Ensure we have something valid
    if not obs_id:
        obs_id = "unknown"

    return obs_id


class AnalysisLock:
    """Exclusive analysis lock using fcntl.flock.

    Serialises concurrent pipeline invocations for the same observation
    directory.  When a second process tries to acquire the lock it blocks
    until the first one releases it, then proceeds (running only
    incremental work thanks to per-epoch ecsv caching).

    Usage:
        lock = AnalysisLock(obs_dir)
        with lock:
            run_analysis(...)
    """

    STALE_TIMEOUT = 900  # seconds (15 min)

    def __init__(self, obs_dir):
        self.lock_path = obs_dir / ".analysis.lock"
        self._fd = None

    def __enter__(self):
        self._fd = open(self.lock_path, "w")
        logging.info(f"Acquiring analysis lock: {self.lock_path}")
        try:
            # Try non-blocking first to log contention
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            logging.info("Analysis lock acquired (no contention)")
        except (OSError, BlockingIOError):
            logging.info("Another process holds the lock, waiting...")
            fcntl.flock(self._fd, fcntl.LOCK_EX)  # blocking wait
            logging.info("Analysis lock acquired after waiting")
        # Write owner info for debugging
        self._fd.seek(0)
        self._fd.truncate()
        self._fd.write(f"PID: {os.getpid()}\nStarted: {datetime.now().isoformat()}\n")
        self._fd.flush()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
                self._fd.close()
            except Exception as e:
                logging.warning(f"Could not release analysis lock: {e}")
            finally:
                self._fd = None
            try:
                self.lock_path.unlink(missing_ok=True)
            except Exception:
                pass
            logging.info("Analysis lock released")
        return False  # don't suppress exceptions


class ObservationStore:
    """Filesystem-backed store for one observation's detection tables,
    processed-file metadata, and analysis results.
    """

    def __init__(self, base_dir, observation_id):
        self.base_dir = Path(base_dir)
        self.observation_id = observation_id
        self.obs_dir = self._setup_observation_directory()

    def _setup_observation_directory(self) -> Path:
        """Create and return observation-specific directory."""
        obs_dir = self.base_dir / f"obs_{self.observation_id}"
        try:
            # Try to create directory (race condition safe)
            obs_dir.mkdir(exist_ok=True)
            return obs_dir
        except Exception as e:
            logging.warning(f"Could not create observation directory {obs_dir}: {e}")
            # Fallback to base directory if obs directory creation fails
            return self.base_dir

    def _metadata_path(self) -> Path:
        return self.obs_dir / "detection_metadata.json"

    def _read_processed_files(self) -> Set[str]:
        metadata_file = self._metadata_path()
        if not metadata_file.exists():
            return set()
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                return set(metadata.get('processed_files', []))
        except Exception as e:
            logging.warning(f"Could not load metadata: {e}")
            return set()

    def load_existing_tables(self) -> Tuple[List[Table], Set[str]]:
        """Load existing detection tables from observation directory."""
        detection_tables = []
        processed_files = self._read_processed_files()

        # Load existing detection tables
        for ecsv_file in self.obs_dir.glob("*.ecsv"):
            if ecsv_file.name in processed_files:
                try:
                    detection_data = open_ecsv_file(str(ecsv_file))
                    if detection_data is not None and detection_data.meta is not None:
                        detection_tables.append(detection_data)
                        logging.info(f"Loaded existing detection table: {ecsv_file.name}")
                except Exception as e:
                    logging.warning(f"Could not load {ecsv_file}: {e}")

        return detection_tables, processed_files

    def mark_processed(self, filename) -> None:
        """Update metadata file with newly processed file (thread-safe)."""
        processed_files = self._read_processed_files()
        processed_files.add(filename)
        metadata_file = self._metadata_path()

        metadata = {
            'processed_files': list(processed_files),
            'last_updated': datetime.now().isoformat(),
            'total_files': len(processed_files),
            'observation_id': self.obs_dir.name.replace('obs_', ''),
            'process_id': os.getpid()
        }

        temp_file = metadata_file.with_suffix('.tmp')
        try:
            # Use atomic write with temporary file to prevent corruption
            with open(temp_file, 'w') as f:
                # Try to get exclusive lock (non-blocking)
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    json.dump(metadata, f, indent=2)
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                except BlockingIOError:
                    # If we can't get the lock, another process is updating
                    logging.info(f"Another process is updating metadata, skipping...")
                    return

            # Atomic move
            temp_file.replace(metadata_file)

        except Exception as e:
            logging.warning(f"Could not save metadata: {e}")
            # Clean up temp file if it exists
            if temp_file.exists():
                temp_file.unlink()

    def already_processed(self, filename) -> bool:
        """Check if this specific file has already been processed (thread-safe)."""
        metadata_file = self._metadata_path()

        if not metadata_file.exists():
            return False

        try:
            with open(metadata_file, 'r') as f:
                # Try to get shared lock for reading
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
                    metadata = json.load(f)
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

                    processed_files = set(metadata.get('processed_files', []))
                    return filename in processed_files
                except BlockingIOError:
                    # If we can't get the lock, assume not processed to be safe
                    logging.info(f"INFO: Could not read metadata (locked), assuming not processed")
                    return False
        except Exception as e:
            logging.info(f"WARNING: Could not read metadata: {e}")
            return False

    def should_run_analysis(self, new_detection_added) -> Tuple[bool, str]:
        """Determine if analysis should be run."""
        candidates_file = self.obs_dir / "candidates.tbl"

        logging.info(f"Checking analysis conditions:")
        logging.info(f"  - New detection added: {new_detection_added}")
        logging.info(f"  - Candidates file exists: {candidates_file.exists()}")

        if new_detection_added:
            logging.info("  -> Decision: Need analysis (new detection data)")
            return True, "New detection data added"

        if not candidates_file.exists():
            logging.info("  -> Decision: Need analysis (no existing results)")
            return True, "No existing results found"

        logging.info("  -> Decision: Skip analysis (results exist, no new data)")
        return False, "Results exist and no new data"

    def analysis_lock(self) -> AnalysisLock:
        """Context manager serializing concurrent analysis runs for this observation."""
        return AnalysisLock(self.obs_dir)

    def save_results(self, candidates: Table, lightcurves: dict) -> None:
        """Write candidates.tbl and lightcurve_summary.json (if any) to obs_dir."""
        candidates_file = self.obs_dir / "candidates.tbl"
        candidates.write(str(candidates_file), format="ascii.ipac", overwrite=True)
        logging.info(f"Results saved to {candidates_file}")

        if lightcurves:
            logging.info(f"Generated {len(lightcurves)} lightcurves")
            lightcurve_summary_file = self.obs_dir / "lightcurve_summary.json"

            # Create a simple summary of lightcurves
            lightcurve_summary = {}
            for transient_id, lc_data in lightcurves.items():
                lightcurve_summary[transient_id] = {
                    'n_detections': len(lc_data),
                    'n_epochs': len(np.unique(lc_data['epoch_id'])) if 'epoch_id' in lc_data.colnames else 1,
                    'time_span_hours': float((np.max(lc_data['obs_time']) - np.min(lc_data['obs_time'])) / 3600.0) if 'obs_time' in lc_data.colnames else 0.0
                }

            try:
                with open(lightcurve_summary_file, 'w') as f:
                    json.dump(lightcurve_summary, f, indent=2)
                logging.info(f"Lightcurve summary saved to {lightcurve_summary_file}")
            except Exception as e:
                logging.warning(f"Could not save lightcurve summary: {e}")
