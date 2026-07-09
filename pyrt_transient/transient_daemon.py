#!/usr/bin/env python3
# /home/fnovotny/bin/transient_daemon.py
"""
Asynchronous transient detection daemon.
Receives requests via Unix socket, copies files, responds immediately,
then processes transients in background with process limiting.

Debounce: when images arrive in rapid succession for the same observation,
the daemon waits DEBOUNCE_SECONDS after the last arrival before launching
the pipeline.  This collapses N rapid-fire invocations into a single run.
"""

import socket
import os
import json
import shutil
import subprocess
import logging
import threading
import time
import signal
import sys
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

# Configuration
SOCKET_PATH = "/home/fnovotny/transient_daemon.sock"
WORK_DIR = Path("/home/fnovotny/transient_work")
LOG_DIR = Path("/home/fnovotny/logs")
MAX_PARALLEL_PROCESSES = 4  # Limit concurrent transient detections
PIPELINE_SCRIPT = "/home/fnovotny/bin/pipeline_magic.py"
DEBOUNCE_SECONDS = 30  # Wait this long after last image before launching pipeline

# Setup logging
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'transient_daemon.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TransientDaemon:
    def __init__(self):
        self.work_dir = WORK_DIR
        self.work_dir.mkdir(exist_ok=True)

        # Thread pool for background processing
        self.executor = ThreadPoolExecutor(max_workers=MAX_PARALLEL_PROCESSES)
        self.active_jobs = 0
        self.jobs_lock = threading.Lock()

        # Job queue and statistics
        self.job_queue = Queue()
        self.processed_count = 0
        self.failed_count = 0

        # Debounce state: obs_dir -> {timer, files}
        self._debounce_lock = threading.Lock()
        self._debounce_timers = {}  # obs_dir_str -> threading.Timer
        self._debounce_files = {}   # obs_dir_str -> [(ecsv, fits, job_dir), ...]

        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        self.running = True

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
        # Fire all pending debounce timers immediately
        with self._debounce_lock:
            for timer in self._debounce_timers.values():
                timer.cancel()
        self.executor.shutdown(wait=True)
        try:
            os.unlink(SOCKET_PATH)
        except FileNotFoundError:
            pass
        sys.exit(0)

    def _generate_work_id(self) -> str:
        """Generate unique work ID for this job."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"job_{timestamp}_{os.getpid()}"

    def _get_obs_dir(self, ecsv_path: str) -> str:
        """Derive observation group key from ecsv path.

        The pipeline groups files by observation ID (from ECSV metadata)
        into obs_<id> dirs.  We read the same metadata here so that all
        images for the same observation share a single debounce key.

        Falls back to the full filename stem if reading metadata fails
        (each image gets its own key — the fcntl lock still ensures
        correctness, just without the debounce benefit).
        """
        try:
            from astropy.table import Table
            t = Table.read(ecsv_path, format='ascii.ecsv')
            if t.meta:
                for field in ('OBSID', 'OBS_ID', 'OBSERVATION_ID', 'FIELD_ID'):
                    if field in t.meta:
                        obs_id = str(t.meta[field]).split('.')[0].strip()
                        if obs_id:
                            logger.info(f"Debounce key from metadata {field}={obs_id}: {Path(ecsv_path).name}")
                            return obs_id
        except Exception as e:
            logger.warning(f"Could not read ECSV metadata for debounce key: {e}")

        # Fallback: each file gets its own key (no debounce grouping,
        # but fcntl lock still serialises correctly)
        stem = Path(ecsv_path).stem
        logger.warning(f"No observation ID in metadata, using filename as debounce key: {stem}")
        return stem

    def copy_files_for_processing(self, ecsv_path: str, fits_path: str) -> tuple:
        """Copy files to work directory and return new paths."""
        work_id = self._generate_work_id()
        job_dir = self.work_dir / work_id
        job_dir.mkdir(exist_ok=True)

        ecsv_copy = job_dir / Path(ecsv_path).name
        fits_copy = job_dir / Path(fits_path).name

        try:
            shutil.copy2(ecsv_path, ecsv_copy)
            shutil.copy2(fits_path, fits_copy)
            logger.info(f"Copied files for job {work_id}")
            return str(ecsv_copy), str(fits_copy), job_dir
        except Exception as e:
            logger.error(f"Failed to copy files for job {work_id}: {e}")
            # Clean up on failure
            shutil.rmtree(job_dir, ignore_errors=True)
            raise

    def _debounce_fire(self, obs_key: str):
        """Called when the debounce timer expires — launch pipeline for all batched files.

        Each file must be passed through the pipeline so it gets copied to
        the observation directory and registered in metadata.  The fcntl lock
        in pipeline_magic serialises these, and incremental epoch processing
        skips already-processed epochs, so this is fast.
        """
        with self._debounce_lock:
            self._debounce_timers.pop(obs_key, None)
            pending_files = self._debounce_files.pop(obs_key, [])

        if not pending_files:
            return

        logger.info(f"Debounce fired for {obs_key}: launching pipeline for "
                    f"{len(pending_files)} batched images")

        self.executor.submit(self._process_batch, obs_key, pending_files)

    def _process_batch(self, obs_key: str, file_list: list):
        """Process a batch of files sequentially in a single thread.

        Each file is passed through the pipeline (which copies it to the
        obs dir, registers it, and runs incremental analysis).  The
        fcntl lock ensures only one pipeline runs at a time per obs dir.
        """
        with self.jobs_lock:
            self.active_jobs += 1

        start_time = time.time()
        try:
            for i, (ecsv_path, fits_path, job_dir) in enumerate(file_list):
                job_id = job_dir.name
                logger.info(f"Batch {obs_key} [{i+1}/{len(file_list)}]: "
                            f"processing {Path(ecsv_path).name}")
                try:
                    result = subprocess.run(
                        [PIPELINE_SCRIPT, ecsv_path, fits_path],
                        cwd=str(job_dir),
                        capture_output=True,
                        text=True,
                        timeout=900,
                    )

                    if result.returncode == 0:
                        logger.info(f"Batch {obs_key} [{i+1}/{len(file_list)}]: "
                                    f"job {job_id} succeeded")
                        self.processed_count += 1
                    else:
                        logger.error(f"Batch {obs_key} [{i+1}/{len(file_list)}]: "
                                     f"job {job_id} failed (exit {result.returncode})")
                        if result.stderr:
                            logger.error(f"Stderr: {result.stderr}")
                        self.failed_count += 1

                except subprocess.TimeoutExpired:
                    logger.error(f"Batch {obs_key}: job {job_id} timed out")
                    self.failed_count += 1
                except Exception as e:
                    logger.error(f"Batch {obs_key}: job {job_id} failed: {e}")
                    self.failed_count += 1
                finally:
                    try:
                        shutil.rmtree(job_dir, ignore_errors=True)
                    except Exception:
                        pass

            elapsed = time.time() - start_time
            logger.info(f"Batch {obs_key} complete: {len(file_list)} files in {elapsed:.1f}s")

        finally:
            with self.jobs_lock:
                self.active_jobs -= 1
                logger.info(f"Batch {obs_key} finished. Active jobs: {self.active_jobs}")

    def handle_request(self, conn):
        """Handle incoming socket request with debounce."""
        try:
            # Receive request
            data = conn.recv(4096).decode()
            if not data:
                return

            request = json.loads(data)
            ecsv_path = request['ecsv_path']
            fits_path = request['fits_path']

            logger.info(f"Received request: {Path(ecsv_path).name}")

            # Copy files immediately (so source can be removed)
            try:
                ecsv_copy, fits_copy, job_dir = self.copy_files_for_processing(ecsv_path, fits_path)
                logger.info(f"Files copied successfully for job {job_dir.name}")
            except Exception as e:
                logger.error(f"File copy failed: {e}")
                response = {'success': False, 'error': f'File copy failed: {str(e)}'}
                conn.send(json.dumps(response).encode())
                return

            # Respond immediately that files are copied
            response = {
                'success': True,
                'message': f'Files copied, pipeline will launch after {DEBOUNCE_SECONDS}s debounce',
                'job_id': job_dir.name
            }
            conn.send(json.dumps(response).encode())

            # Debounce: group by observation and reset timer
            obs_key = self._get_obs_dir(ecsv_path)

            with self._debounce_lock:
                # Cancel existing timer for this observation
                existing_timer = self._debounce_timers.get(obs_key)
                if existing_timer is not None:
                    existing_timer.cancel()
                    logger.info(f"Debounce: reset timer for {obs_key} "
                                f"({len(self._debounce_files.get(obs_key, []))+1} images pending)")

                # Accumulate files
                if obs_key not in self._debounce_files:
                    self._debounce_files[obs_key] = []
                self._debounce_files[obs_key].append((ecsv_copy, fits_copy, job_dir))

                # Start new timer
                timer = threading.Timer(DEBOUNCE_SECONDS, self._debounce_fire, args=[obs_key])
                timer.daemon = True
                timer.start()
                self._debounce_timers[obs_key] = timer

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON request: {e}")
            error_response = {'success': False, 'error': 'Invalid JSON'}
            try:
                conn.send(json.dumps(error_response).encode())
            except Exception:
                pass  # Connection may be closed
        except Exception as e:
            logger.error(f"Error handling request: {e}")
            error_response = {'success': False, 'error': str(e)}
            try:
                conn.send(json.dumps(error_response).encode())
            except Exception:
                pass  # Connection may be closed

    def print_status(self):
        """Print periodic status information."""
        while self.running:
            time.sleep(60)  # Status every minute
            with self.jobs_lock:
                pending = sum(len(v) for v in self._debounce_files.values())
                logger.info(f"Status: {self.active_jobs} active jobs, "
                           f"{pending} pending (debounce), "
                           f"{self.processed_count} completed, {self.failed_count} failed")

    def run(self):
        """Main daemon loop."""
        # Remove old socket
        try:
            os.unlink(SOCKET_PATH)
        except FileNotFoundError:
            pass

        # Create socket
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(SOCKET_PATH)
        os.chmod(SOCKET_PATH, 0o666)  # Allow other users to connect
        sock.listen(5)

        logger.info(f"Transient daemon started")
        logger.info(f"Listening on {SOCKET_PATH}")
        logger.info(f"Max parallel processes: {MAX_PARALLEL_PROCESSES}")
        logger.info(f"Debounce window: {DEBOUNCE_SECONDS}s")
        logger.info(f"Work directory: {self.work_dir}")

        # Start status thread
        status_thread = threading.Thread(target=self.print_status, daemon=True)
        status_thread.start()

        try:
            while self.running:
                try:
                    sock.settimeout(1.0)  # Allow checking self.running
                    conn, addr = sock.accept()

                    # Handle request in main thread to avoid socket issues
                    self.handle_request(conn)

                    # Close connection after starting handler
                    conn.close()

                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:
                        logger.error(f"Socket error: {e}")
                        time.sleep(1)
        finally:
            sock.close()
            logger.info("Daemon stopped")


def main():
    daemon = TransientDaemon()
    daemon.run()


if __name__ == "__main__":
    main()
