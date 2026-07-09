"""Frontend generation orchestration -- moved from pipeline_magic.py's
generate_frontend (the real one, not _old), rewritten to call site_state.py
for the checksum-gating logic.

generate_frontend_old and the `except ImportError` fallback branch that
called it are deleted here, not ported -- confirmed dead (gen_frontend.py
lived only at /home/fnovotny/bin/gen_frontend.py on the remote production
host, hardcoded to obs_94523, never read sys.argv). That remote file is
outside this repo; deleting it there is a separate manual follow-up.
"""

import logging
from pathlib import Path

from pyrt_transient.web.site_state import is_up_to_date, record_generated


def generate_frontend(obs_dir, observation_id, config):
    """Generate frontend using the integrated FrontendGenerator, gated by candidates.tbl changes."""
    try:
        from pyrt_transient.frontend_generator import FrontendGenerator

        base_public_dir = config.base_public_dir or Path.home() / "public_html"
        website_dir = Path(base_public_dir) / f"obs_{observation_id}"
        candidates_file = Path(obs_dir) / "candidates.tbl"

        up_to_date, current_hash = is_up_to_date(website_dir, candidates_file)
        if up_to_date:
            logging.info(f"Website up-to-date for observation {observation_id}, skipping generation")
            return True

        logging.info(f"Generating website for observation {observation_id}...")
        frontend_gen = FrontendGenerator(
            observation_id=observation_id,
            data_dir=obs_dir,
            base_public_dir=base_public_dir,
            config=config.frontend
        )

        success = frontend_gen.generate_complete_website()

        if success:
            record_generated(website_dir, current_hash)

        return success

    except Exception as e:
        logging.info(f"ERROR: Frontend generation failed: {e}")
        return False
