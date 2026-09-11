#!/usr/bin/env python3
"""Per-frame pipeline entry point that gives GRB frames priority.

The daemon runs one pipeline process per frame as `<this> <ecsv> <fits>`.
With pyrt-transient-daemon that is PYRT_TRANSIENT_PIPELINE; lascaux50's own
daemon calls ~/bin/pipeline_magic.py, which runs this file. Before the
package is imported, this sets the process priority from the frame's
target. Everything the pipeline starts inherits it:

  GRB-type target   nice 0,  ionice best-effort 4, 4 BLAS threads
  anything else     nice 10, ionice best-effort 7, 2 BLAS threads

Under load a GRB frame then gets about 10x the CPU share of routine
monitoring. A process can only lower its own priority, so the daemon itself
must run at nice 0: a unit-level Nice=10 would hold GRB frames at 10 too.

The target comes from the RTS2 database. The ECSV header carries the
target id (TARGET), and `targets` in the `stars` database gives its current
type and name. Names and numbers change and new targets appear, so this is
looked up for every frame, not kept. A target counts as GRB-type when:

- its type is 'G' (the automatic GRB, SVOM, EP, IceCube and GCN triggers);
- or its name is a GRB/EP/IceCube one (manually created follow-ups such as
  "GRB 260310A/AT2026fgk" are type 'O').

Without the database (PYRT_TARGETS_DB="" or unreachable), the header's
GRB_RA and OBJECT decide the same way.

For the status page (tools/status_page.py), the script leaves a marker in
<data dir>/.running/ while the frame runs. It also refreshes the page at the
start and the end, in the background, so the frame never waits for it.
PYRT_STATUS_PAGE=0 turns that off.
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

GRB_NAME = re.compile(
    r"\bGRB|SVOM|ECLAIRs|\bEP ?\d{6}|Einstein Probe|IceCube|\btrigger\b|\bGCN\b"
    r"|\bGW\d{6}|\bS\d{6}[a-z]{1,3}\b|Fermi|\bBAT\b",
    re.IGNORECASE)
GRB_TYPES = {"G"}  # RTS2 targets.type_id of automatic burst triggers
# GRB-type?  ->  (nice, ionice best-effort level, BLAS threads)
PRIORITY = {True: (0, 4, "4"), False: (10, 7, "2")}
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
HEADER_BYTES = 1 << 16
TARGETS_DB = os.environ.get("PYRT_TARGETS_DB", "stars")
DATA_DIR = Path(os.environ.get("PYRT_STATUS_DATA_DIR", Path.home() / "transient_work"))
_META_LINE = re.compile(r"#\s*-\s*\{([A-Za-z0-9_\-]+):\s*(.*)\}\s*$")


def read_header(ecsv_path):
    """{KEY: value} from the `# - {KEY: value}` lines of an ECSV header.

    Only the first HEADER_BYTES are read. Values are strings with YAML
    quotes stripped. The first occurrence of a key wins.
    """
    meta = {}
    try:
        with open(ecsv_path, "rb") as fh:
            head = fh.read(HEADER_BYTES).decode("utf-8", "replace")
    except OSError:
        return meta
    for line in head.splitlines():
        if not line.startswith("#"):
            break
        m = _META_LINE.match(line)
        if m:
            meta.setdefault(m.group(1), m.group(2).strip().strip("'\""))
    return meta


def target_id(meta):
    try:
        return int(float(meta.get("TARGET", "")))
    except ValueError:
        return None


def obs_id_from_meta(meta):
    """'104210.00' -> '104210', as the pipeline names obs_<id>."""
    try:
        return str(int(float(meta.get("OBSID", ""))))
    except ValueError:
        return ""


def lookup_targets(tar_ids):
    """{tar_id: {"type": type_id, "name": tar_name}} from the RTS2 database.

    Empty when there is no database, it cannot be reached within a few
    seconds, or the ids are unknown.
    """
    ids = sorted({int(t) for t in tar_ids if t is not None})
    if not ids or not TARGETS_DB:
        return {}
    query = ("select tar_id, type_id, tar_name from targets where tar_id in (%s)"
             % ",".join(map(str, ids)))
    # One SELECT in a read-only session: the database is RTS2's, never written.
    env = dict(os.environ, PGCONNECT_TIMEOUT="5",
               PGOPTIONS="-c default_transaction_read_only=on")
    try:
        out = subprocess.run(["psql", "-XAtq", "-F", "\t", "-d", TARGETS_DB, "-c", query],
                             capture_output=True, text=True, timeout=10, env=env)
    except (OSError, subprocess.SubprocessError):
        return {}
    rows = {}
    if out.returncode == 0:
        for line in out.stdout.splitlines():
            parts = line.split("\t", 2)
            if len(parts) == 3 and parts[0].strip().isdigit():
                rows[int(parts[0])] = {"type": parts[1].strip(), "name": parts[2].strip()}
    return rows


def is_grb(meta, target=None):
    """GRB-type from the database row when there is one, else the header."""
    if target:
        return target["type"] in GRB_TYPES or bool(GRB_NAME.search(target["name"]))
    return "GRB_RA" in meta or bool(GRB_NAME.search(meta.get("OBJECT", "")))


def apply_priority(grb):
    nice, io_level, threads = PRIORITY[grb]
    for var in THREAD_VARS:  # before numpy is imported
        os.environ.setdefault(var, threads)
    try:
        os.setpriority(os.PRIO_PROCESS, 0, max(os.getpriority(os.PRIO_PROCESS, 0), nice))
    except OSError:
        pass
    try:
        subprocess.run(["ionice", "-c2", f"-n{io_level}", "-p", str(os.getpid())],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
    except (OSError, subprocess.SubprocessError):
        pass


def write_marker(meta, name, grb, ecsv_path):
    try:
        running = DATA_DIR / ".running"
        running.mkdir(parents=True, exist_ok=True)
        path = running / f"{os.getpid()}.json"
        path.write_text(json.dumps({
            "pid": os.getpid(), "obs_id": obs_id_from_meta(meta), "object": name,
            "grb": grb, "frame": Path(ecsv_path).name, "started": time.time()}))
        return path
    except OSError:
        return None


def refresh_status_page():
    script = Path(__file__).resolve().with_name("status_page.py")
    if os.environ.get("PYRT_STATUS_PAGE") == "0" or not script.exists():
        return
    try:
        subprocess.Popen([sys.executable, str(script)], stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True, preexec_fn=lambda: os.nice(15))
    except OSError:
        pass


def main():
    ecsv_path = sys.argv[1] if len(sys.argv) > 1 else ""
    meta = read_header(ecsv_path) if ecsv_path else {}
    tid = target_id(meta)
    target = lookup_targets([tid]).get(tid) if tid is not None else None
    grb = is_grb(meta, target)
    apply_priority(grb)
    name = target["name"] if target else meta.get("OBJECT", "")
    marker = write_marker(meta, name, grb, ecsv_path) if ecsv_path else None
    refresh_status_page()
    try:
        from pyrt_transient.pipeline_magic import main as pipeline_main
        pipeline_main()
    finally:
        if marker is not None:
            try:
                marker.unlink()
            except OSError:
                pass
        refresh_status_page()


if __name__ == "__main__":
    main()
