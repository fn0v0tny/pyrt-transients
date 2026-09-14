#!/usr/bin/env python3
"""
Asynchronous transient detection daemon.
Receives requests via Unix socket, copies files, responds immediately,
then processes transients in background with process limiting.

Debounce: when images arrive in rapid succession for the same observation,
the daemon waits DEBOUNCE_SECONDS after the last arrival before launching
the pipeline.  This collapses N rapid-fire invocations into a single run.

Configuration is by environment variable (all optional):

  PYRT_TRANSIENT_SOCKET      Unix socket path      (default ~/transient_daemon.sock)
  PYRT_TRANSIENT_WORK_DIR    staging directory     (default ~/transient_work)
  PYRT_TRANSIENT_LOG_DIR     log directory         (default ~/logs)
  PYRT_TRANSIENT_PIPELINE    pipeline command      (default: the installed
                             `pyrt-transient-pipeline` entry point)
  PYRT_TRANSIENT_PIPELINE_ARGS  extra args appended to every pipeline call,
                             e.g. "--config=/etc/pyrt/transient.yaml"
  PYRT_TRANSIENT_MAX_PARALLEL   concurrent pipeline runs (default 4)
  PYRT_TRANSIENT_DEBOUNCE_S     debounce window, seconds (default 30)
  PYRT_STATUS_REFRESH_S         how often the daemon refreshes the status page
                                (tools/status_page.py; default 300, 0 = never)
"""

import socket
import os
import json
import shlex
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

_HOME = Path.home()
SOCKET_PATH = os.environ.get("PYRT_TRANSIENT_SOCKET", str(_HOME / "transient_daemon.sock"))
WORK_DIR = Path(os.environ.get("PYRT_TRANSIENT_WORK_DIR", _HOME / "transient_work"))
LOG_DIR = Path(os.environ.get("PYRT_TRANSIENT_LOG_DIR", _HOME / "logs"))
MAX_PARALLEL_PROCESSES = int(os.environ.get("PYRT_TRANSIENT_MAX_PARALLEL", "4"))
# How often the daemon itself refreshes the status page, so it does not stand
# still while the telescope is idle. 0 turns it off.
STATUS_REFRESH_S = float(os.environ.get("PYRT_STATUS_REFRESH_S", "300"))
DEBOUNCE_SECONDS = float(os.environ.get("PYRT_TRANSIENT_DEBOUNCE_S", "30"))
PIPELINE_TIMEOUT_S = 900


def resolve_pipeline_command():
    """The command that runs one epoch through the pipeline.

    The installed console script (`pyrt-transient-pipeline`, see
    pyproject.toml) by default; PYRT_TRANSIENT_PIPELINE overrides it with
    any executable or `python -m ...` string. Previously this was a
    hard-coded path into one user's home directory.
    """
    override = os.environ.get("PYRT_TRANSIENT_PIPELINE")
    if override:
        cmd = shlex.split(override)
    else:
        entry = shutil.which("pyrt-transient-pipeline")
        cmd = [entry] if entry else [sys.executable, "-m", "pyrt_transient.pipeline_magic"]
    cmd += shlex.split(os.environ.get("PYRT_TRANSIENT_PIPELINE_ARGS", ""))
    return cmd


# Setup logging
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'transient_daemon.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def _recv_json(conn, max_bytes=1 << 20):
    """Read one JSON request: the client may send it in several segments,
    so read until the socket is closed for writing or the buffer parses."""
    chunks = []
    total = 0
    while total < max_bytes:
        data = conn.recv(4096)
        if not data:
            break
        chunks.append(data)
        total += len(data)
        try:
            return json.loads(b"".join(chunks).decode())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue  # incomplete so far
    if not chunks:
        return None
    return json.loads(b"".join(chunks).decode())  # raises on genuinely bad JSON


class TransientDaemon:
    def __init__(self):
        self.work_dir = WORK_DIR
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline_cmd = resolve_pipeline_command()

        # Thread pool for background processing
        self.executor = ThreadPoolExecutor(max_workers=MAX_PARALLEL_PROCESSES)
        self.active_jobs = 0
        self.processed_count = 0
        self.failed_count = 0
        self.jobs_lock = threading.Lock()  # guards the three counters above

        # Debounce state: obs_key -> {timer, files}
        self._debounce_lock = threading.Lock()
        self._debounce_timers = {}  # obs_key -> threading.Timer
        self._debounce_files = {}   # obs_key -> [(ecsv, fits, job_dir), ...]

        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        self.running = True

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
        # Fire all pending debounce batches now rather than waiting out
        # their timers -- cancelling them (as this used to) silently dropped
        # every image received in the last DEBOUNCE_SECONDS and leaked its
        # job directory.
        with self._debounce_lock:
            for timer in self._debounce_timers.values():
                timer.cancel()
            pending_keys = list(self._debounce_timers)
        for obs_key in pending_keys:
            self._debounce_fire(obs_key)
        self.executor.shutdown(wait=True)
        try:
            os.unlink(SOCKET_PATH)
        except FileNotFoundError:
            pass
        sys.exit(0)

    def _generate_work_id(self) -> str:
        """Generate unique work ID for this job."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        return f"job_{timestamp}_{os.getpid()}"

    def _get_obs_key(self, ecsv_path: str) -> str:
        """Debounce key for an incoming file: the same observation ID the
        pipeline itself derives (io.observation_store.extract_observation_id),
        so every image of one observation shares one key. Falls back to the
        filename stem if the metadata can't be read -- each image then gets
        its own key (no debounce benefit; the pipeline's own lock still
        serialises correctly).
        """
        try:
            from pyrt_transient.io.observation_store import extract_observation_id
            obs_id = extract_observation_id(ecsv_path)
            logger.info(f"Debounce key {obs_id}: {Path(ecsv_path).name}")
            return obs_id
        except Exception as e:
            logger.warning(f"Could not derive observation ID for debounce key: {e}")
        stem = Path(ecsv_path).stem
        logger.warning(f"Using filename as debounce key: {stem}")
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
            shutil.rmtree(job_dir, ignore_errors=True)
            raise

    def _debounce_fire(self, obs_key: str):
        """Called when the debounce timer expires -- launch pipeline for all batched files.

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
        """Process a batch of files sequentially in a single thread."""
        with self.jobs_lock:
            self.active_jobs += 1

        start_time = time.time()
        try:
            for i, (ecsv_path, fits_path, job_dir) in enumerate(file_list):
                job_id = job_dir.name
                logger.info(f"Batch {obs_key} [{i+1}/{len(file_list)}]: "
                            f"processing {Path(ecsv_path).name}")
                ok = False
                try:
                    result = subprocess.run(
                        self.pipeline_cmd + [ecsv_path, fits_path],
                        cwd=str(job_dir),
                        capture_output=True,
                        text=True,
                        timeout=PIPELINE_TIMEOUT_S,
                    )
                    if result.returncode == 0:
                        logger.info(f"Batch {obs_key} [{i+1}/{len(file_list)}]: job {job_id} succeeded")
                        ok = True
                    else:
                        logger.error(f"Batch {obs_key} [{i+1}/{len(file_list)}]: "
                                     f"job {job_id} failed (exit {result.returncode})")
                        if result.stderr:
                            logger.error(f"Stderr: {result.stderr[-2000:]}")
                except subprocess.TimeoutExpired:
                    logger.error(f"Batch {obs_key}: job {job_id} timed out")
                except Exception as e:
                    logger.error(f"Batch {obs_key}: job {job_id} failed: {e}")
                finally:
                    with self.jobs_lock:
                        if ok:
                            self.processed_count += 1
                        else:
                            self.failed_count += 1
                    shutil.rmtree(job_dir, ignore_errors=True)

            elapsed = time.time() - start_time
            logger.info(f"Batch {obs_key} complete: {len(file_list)} files in {elapsed:.1f}s")

        finally:
            with self.jobs_lock:
                self.active_jobs -= 1
                logger.info(f"Batch {obs_key} finished. Active jobs: {self.active_jobs}")

    def handle_request(self, conn):
        """Handle incoming socket request with debounce."""
        try:
            request = _recv_json(conn)
            if not request:
                return
            ecsv_path = request['ecsv_path']
            fits_path = request['fits_path']

            logger.info(f"Received request: {Path(ecsv_path).name}")

            # Copy files immediately (so source can be removed)
            try:
                ecsv_copy, fits_copy, job_dir = self.copy_files_for_processing(ecsv_path, fits_path)
            except Exception as e:
                logger.error(f"File copy failed: {e}")
                conn.send(json.dumps({'success': False, 'error': f'File copy failed: {e}'}).encode())
                return

            conn.send(json.dumps({
                'success': True,
                'message': f'Files copied, pipeline will launch after {DEBOUNCE_SECONDS:g}s debounce',
                'job_id': job_dir.name
            }).encode())

            # Debounce: group by observation and reset timer. Keyed on the
            # local copy so the source can be removed right away.
            obs_key = self._get_obs_key(ecsv_copy)

            with self._debounce_lock:
                existing_timer = self._debounce_timers.get(obs_key)
                if existing_timer is not None:
                    existing_timer.cancel()
                    logger.info(f"Debounce: reset timer for {obs_key} "
                                f"({len(self._debounce_files.get(obs_key, []))+1} images pending)")

                self._debounce_files.setdefault(obs_key, []).append((ecsv_copy, fits_copy, job_dir))

                timer = threading.Timer(DEBOUNCE_SECONDS, self._debounce_fire, args=[obs_key])
                timer.daemon = True
                timer.start()
                self._debounce_timers[obs_key] = timer

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON request: {e}")
            try:
                conn.send(json.dumps({'success': False, 'error': 'Invalid JSON'}).encode())
            except Exception:
                pass
        except Exception as e:
            logger.error(f"Error handling request: {e}")
            try:
                conn.send(json.dumps({'success': False, 'error': str(e)}).encode())
            except Exception:
                pass

    def refresh_status_page(self):
        """Rewrite the status page's data (tools/status_page.py), in the background.

        Otherwise only a frame going through tools/pipeline_entry.py refreshes
        it, so the page stands still whenever the telescope is idle: on
        lascaux50 it showed the same data for eight hours after the last frame
        of the night, with a frame killed at its timeout still listed as
        running.
        """
        script = Path(__file__).resolve().parent.parent / "tools" / "status_page.py"
        if not script.exists():
            return
        try:
            subprocess.Popen([sys.executable, str(script)], stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             start_new_session=True)
        except OSError as exc:
            logger.debug(f"Status page refresh failed: {exc}")

    def print_status(self):
        """Print periodic status information."""
        last_page = None   # refresh on the first turn, then every interval
        while self.running:
            time.sleep(60)
            if STATUS_REFRESH_S > 0 and (last_page is None
                                         or time.time() - last_page >= STATUS_REFRESH_S):
                last_page = time.time()
                self.refresh_status_page()
            with self._debounce_lock:
                pending = sum(len(v) for v in self._debounce_files.values())
            with self.jobs_lock:
                logger.info(f"Status: {self.active_jobs} active jobs, "
                            f"{pending} pending (debounce), "
                            f"{self.processed_count} completed, {self.failed_count} failed")

    def run(self):
        """Main daemon loop."""
        try:
            os.unlink(SOCKET_PATH)
        except FileNotFoundError:
            pass

        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(SOCKET_PATH)
        os.chmod(SOCKET_PATH, 0o666)  # Allow other users to connect
        sock.listen(5)

        logger.info("Transient daemon started")
        logger.info(f"Listening on {SOCKET_PATH}")
        logger.info(f"Pipeline command: {' '.join(self.pipeline_cmd)}")
        logger.info(f"Max parallel processes: {MAX_PARALLEL_PROCESSES}")
        logger.info(f"Debounce window: {DEBOUNCE_SECONDS:g}s")
        logger.info(f"Work directory: {self.work_dir}")

        status_thread = threading.Thread(target=self.print_status, daemon=True)
        status_thread.start()

        try:
            while self.running:
                try:
                    sock.settimeout(1.0)  # Allow checking self.running
                    conn, addr = sock.accept()
                    try:
                        self.handle_request(conn)
                    finally:
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
