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
from typing import List, Optional, Set, Tuple

import numpy as np
from astropy.table import Table

from pyrt_transient.io.ecsv import open_ecsv_file


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


POINTING_FILE = "pointing.json"


def frame_pointing(meta) -> Optional[Tuple[float, float]]:
    """(CTRRA, CTRDEC) of a frame in degrees, or None."""
    meta = meta or {}
    try:
        return float(meta["CTRRA"]), float(meta["CTRDEC"])
    except (KeyError, TypeError, ValueError):
        return None


def _sep_arcmin(ra1, dec1, ra2, dec2) -> float:
    import math
    r1, d1, r2, d2 = (math.radians(v) for v in (ra1, dec1, ra2, dec2))
    c = math.sin(d1) * math.sin(d2) + math.cos(d1) * math.cos(d2) * math.cos(r1 - r2)
    return math.degrees(math.acos(max(-1.0, min(1.0, c)))) * 60.0


def resolve_observation_id(ecsv_file_path, base_dir, radius_arcmin=None, max_gap_hours=12.0):
    """Observation ID for this epoch: the raw OBSID-derived one, unless an
    existing observation under base_dir points at the same field (frame
    centre within radius_arcmin) and its last epoch is within
    max_gap_hours -- then that observation's ID, so a campaign split over
    several telescope OBSIDs accumulates in one store. Returns
    (observation_id, reason)."""
    raw_id = extract_observation_id(ecsv_file_path)
    if radius_arcmin is None:
        return raw_id, "obsid"
    try:
        table = open_ecsv_file(ecsv_file_path, verbose=False)
        meta = table.meta if table is not None else {}
    except Exception:
        meta = {}
    pointing = frame_pointing(meta)
    t_mid = epoch_mid_time(meta)
    if pointing is None or t_mid is None:
        return raw_id, "obsid (no pointing/time in header)"
    base = Path(base_dir)
    if not base.exists():
        return raw_id, "obsid"
    best = None
    for pfile in base.glob(f"obs_*/{POINTING_FILE}"):
        try:
            info = json.loads(pfile.read_text())
            sep = _sep_arcmin(pointing[0], pointing[1], float(info["ra"]), float(info["dec"]))
            gap_h = abs(t_mid - float(info["last_mid_time"])) / 3600.0
        except Exception:
            continue
        if sep <= radius_arcmin and gap_h <= max_gap_hours:
            obs_id = pfile.parent.name[len("obs_"):]
            # The epoch's own raw OBSID wins over any other store that also
            # points here. Without that tie-break, two OBSIDs of one field
            # that were created concurrently (neither had written its
            # pointing.json yet, so neither could join the other) stayed
            # split forever, and later epochs could even alternate between
            # them on sub-arcminute differences in the running mean.
            if best is None or (obs_id == raw_id and best[0] != raw_id) or (
                    best[0] != raw_id and sep < best[1]):
                best = (obs_id, sep, gap_h)
    if best is not None:
        if best[0] == raw_id:
            return raw_id, "obsid (pointing agrees)"
        return best[0], f"pointing: {best[1]:.1f}' from obs_{best[0]}, {best[2]:.1f} h after its last epoch (raw OBSID {raw_id})"
    # No observation points here. The raw ID may still belong to a
    # different field (the telescope reuses block IDs; GRB 250813B's set
    # carried one frame 7.5 deg away): never pour this epoch into a store
    # whose pointing disagrees -- take the first free suffix instead.
    candidate = raw_id
    for suffix in ("", "b", "c", "d", "e", "f"):
        candidate = f"{raw_id}{suffix}"
        pfile = base / f"obs_{candidate}" / POINTING_FILE
        if not pfile.exists():
            break
        try:
            info = json.loads(pfile.read_text())
            sep = _sep_arcmin(pointing[0], pointing[1], float(info["ra"]), float(info["dec"]))
        except Exception:
            break
        if sep <= radius_arcmin:
            break
    if candidate != raw_id:
        return candidate, f"obsid {raw_id} already used by another pointing; new observation {candidate}"
    return raw_id, "obsid (no matching pointing)"


def epoch_mid_time(meta) -> Optional[float]:
    """Mid-exposure unix time (CTIME + EXPTIME/2 -- the same convention
    core/epochs.py stamps on detections as `obs_time`), or None if the
    table carries no CTIME."""
    meta = meta or {}
    try:
        ctime = float(meta["CTIME"])
    except (KeyError, TypeError, ValueError):
        return None
    try:
        exptime = float(meta.get("EXPTIME", 0.0) or 0.0)
    except (TypeError, ValueError):
        exptime = 0.0
    return ctime + exptime / 2.0


def epoch_sort_key(table):
    """Sort key putting epochs in observation order: dated tables by
    mid-exposure time, undated ones first (by filename) so `[-1]` is always
    the newest dated epoch."""
    meta = getattr(table, "meta", None) or {}
    t = epoch_mid_time(meta)
    name = str(meta.get("filename", ""))
    return (0, 0.0, name) if t is None else (1, t, name)


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
        # The lock file is deliberately never unlinked. flock() locks an
        # inode, not a path: if A unlinks on release while B is blocked on
        # the old inode, C opens the path, gets a fresh inode, and acquires
        # immediately -- B and C then both run the analysis concurrently.
        # A stale lock file on disk costs nothing (flock state dies with the
        # holding process).
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
                self._fd.close()
            except Exception as e:
                logging.warning(f"Could not release analysis lock: {e}")
            finally:
                self._fd = None
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
        # No fallback to base_dir: that silently pooled every observation's
        # epochs, metadata and candidates.tbl into one shared directory.
        obs_dir.mkdir(parents=True, exist_ok=True)
        return obs_dir

    def _metadata_path(self) -> Path:
        return self.obs_dir / "detection_metadata.json"

    def record_pointing(self, meta) -> None:
        """Keep this observation's pointing and last epoch time in
        pointing.json (read by resolve_observation_id)."""
        pointing = frame_pointing(meta)
        t_mid = epoch_mid_time(meta)
        if pointing is None or t_mid is None:
            return
        path = self.obs_dir / POINTING_FILE
        info = {"ra": pointing[0], "dec": pointing[1], "first_mid_time": t_mid, "last_mid_time": t_mid, "n_epochs": 0}
        try:
            if path.exists():
                old = json.loads(path.read_text())
                info["first_mid_time"] = min(float(old.get("first_mid_time", t_mid)), t_mid)
                info["last_mid_time"] = max(float(old.get("last_mid_time", t_mid)), t_mid)
                info["n_epochs"] = int(old.get("n_epochs", 0))
                # Running mean pointing. RA is averaged through the wrapped
                # offset from the stored value, not linearly: a field
                # straddling RA=0 (359.9 and 0.1) averaged linearly to 180,
                # i.e. a stored centre half the sky away, after which every
                # later epoch failed resolve_observation_id's radius test.
                n = info["n_epochs"]
                ra_old = float(old.get("ra", pointing[0]))
                dra = (pointing[0] - ra_old + 180.0) % 360.0 - 180.0
                info["ra"] = (ra_old + dra / (n + 1)) % 360.0
                info["dec"] = (float(old.get("dec", pointing[1])) * n + pointing[1]) / (n + 1)
        except Exception as e:
            logging.warning(f"Could not read {path}: {e}")
        info["n_epochs"] += 1
        try:
            path.write_text(json.dumps(info, indent=1))
        except Exception as e:
            logging.warning(f"Could not write {path}: {e}")

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
        """Load existing detection tables from observation directory, in
        observation order (oldest first).

        Ordering matters: several consumers take `detection_tables[0]` (the
        field/query box) or `[-1]` ("the most recent epoch": follow-up
        conditions, the SN pipeline's SkyBoT/PM epoch, the stack's most-
        recent-N input selection). A bare glob() is filesystem order, which
        made all of those arbitrary on any run after the first.
        """
        detection_tables = []
        processed_files = self._read_processed_files()

        for ecsv_file in self.obs_dir.glob("*.ecsv"):
            if ecsv_file.name in processed_files:
                try:
                    detection_data = open_ecsv_file(str(ecsv_file))
                    if detection_data is not None and detection_data.meta is not None:
                        detection_tables.append(detection_data)
                        logging.info(f"Loaded existing detection table: {ecsv_file.name}")
                except Exception as e:
                    logging.warning(f"Could not load {ecsv_file}: {e}")

        detection_tables.sort(key=epoch_sort_key)
        return detection_tables, processed_files

    def _metadata_lock_path(self) -> Path:
        return self.obs_dir / ".metadata.lock"

    def _read_metadata(self) -> dict:
        metadata_file = self._metadata_path()
        if not metadata_file.exists():
            return {}
        try:
            with open(metadata_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Could not load metadata: {e}")
            return {}

    def _add_to_metadata_set(self, key: str, filename) -> None:
        """Add `filename` to the `key` set in detection_metadata.json, safe
        across concurrent runs.

        The read-modify-write happens under an exclusive, blocking flock on a
        dedicated lock file (never unlinked -- see AnalysisLock), and the JSON
        is written to a per-process temp file that is atomically renamed into
        place. The previous version truncated one shared `.tmp` before taking
        a non-blocking lock, so two runs could install a partial/empty file,
        or one could silently skip recording its epoch.
        """
        metadata_file = self._metadata_path()
        temp_file = metadata_file.with_name(f"{metadata_file.name}.{os.getpid()}.tmp")
        try:
            with open(self._metadata_lock_path(), "a+") as lock_fd:
                fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
                try:
                    metadata = self._read_metadata()
                    registered = set(metadata.get('processed_files', []))
                    # Stores written before registration and analysis were
                    # split have no 'analyzed_files': everything registered
                    # there had been analysed, so seed it that way once.
                    analyzed = set(metadata.get('analyzed_files', registered))
                    target = registered if key == 'processed_files' else analyzed
                    target.add(filename)
                    if key == 'analyzed_files':
                        registered.add(filename)   # analysed implies registered
                    metadata.update({
                        'processed_files': sorted(registered),
                        'analyzed_files': sorted(analyzed),
                        'last_updated': datetime.now().isoformat(),
                        'total_files': len(registered),
                        'observation_id': self.obs_dir.name.replace('obs_', ''),
                        'process_id': os.getpid()
                    })
                    with open(temp_file, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    temp_file.replace(metadata_file)
                finally:
                    fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
        except Exception as e:
            logging.warning(f"Could not save metadata: {e}")
            try:
                temp_file.unlink(missing_ok=True)
            except Exception:
                pass

    def mark_processed(self, filename) -> None:
        """Register `filename` as an epoch of this observation, i.e. make it
        visible to load_existing_tables. Says nothing about whether its
        analysis finished -- see mark_analyzed."""
        self._add_to_metadata_set('processed_files', filename)

    def mark_analyzed(self, filename) -> None:
        """Record that `filename`'s analysis completed and its results were
        saved. Kept apart from registration so that an epoch whose run died
        mid-analysis is still loaded by the next run (it is on disk and
        registered) instead of vanishing from the observation, while
        pipeline_magic's "already processed, skip" test stays keyed on the
        analysis actually having finished."""
        self._add_to_metadata_set('analyzed_files', filename)

    def already_analyzed(self, filename) -> bool:
        """Whether `filename`'s analysis has completed (thread-safe read)."""
        metadata = self._read_metadata()
        if not metadata:
            return False
        return filename in set(metadata.get('analyzed_files',
                                            metadata.get('processed_files', [])))

    def already_processed(self, filename) -> bool:
        """Whether `filename` is registered as an epoch of this observation."""
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
