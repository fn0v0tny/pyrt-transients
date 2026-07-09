"""Site-state checksum gating -- moved from pipeline_magic.py's
_load_site_state/_save_site_state, plus the inline checksum-gate logic that
used to live directly in generate_frontend() (current_hash/prev_state/
index_exists comparison), now named and testable on its own. Uses
core/fileutil.compute_file_md5.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from pyrt_transient.core.fileutil import compute_file_md5


def load_site_state(website_dir) -> Optional[dict]:
    state_file = Path(website_dir) / ".site_state.json"
    if not state_file.exists():
        return None
    try:
        with open(state_file, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def save_site_state(website_dir, state) -> None:
    state_file = Path(website_dir) / ".site_state.json"
    try:
        Path(website_dir).mkdir(parents=True, exist_ok=True)
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        logging.info(f"WARNING: Could not save site state: {e}")


def is_up_to_date(website_dir, candidates_file) -> Tuple[bool, Optional[str]]:
    """Whether the generated site at website_dir already reflects
    candidates_file's current content, and that file's current MD5 (for
    passing to record_generated() after a fresh generation).
    """
    current_hash = compute_file_md5(candidates_file)
    prev_state = load_site_state(website_dir)
    index_exists = (Path(website_dir) / "index.html").exists()
    up_to_date = bool(
        index_exists and prev_state
        and prev_state.get("candidates_md5") == current_hash
        and current_hash is not None
    )
    return up_to_date, current_hash


def record_generated(website_dir, current_hash) -> None:
    """Persist state after a successful website generation."""
    if current_hash is not None:
        save_site_state(website_dir, {
            "candidates_md5": current_hash,
            "updated": datetime.now().isoformat(),
        })
