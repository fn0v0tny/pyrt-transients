#!/usr/bin/env python3
"""Run the host-galaxy search on observations that already exist and put
the result on their pages.

The pipeline runs the search only when a new frame arrives, so an
observation finished before the search was deployed never gets one, and
its page still carries the app.js from before (the site is only rebuilt
when candidates.tbl changes). This does both for the observation
directories given:

  python tools/host_search_backfill.py /home/fnovotny/transient_work/obs_105068 ...
      --public-dir /home/fnovotny/public_html [--config site.yaml] [--force]
      [--regenerate-site]

Default: run the search (skipped when host_candidates.ecsv already exists,
unless --force) and refresh the existing site in place -- the new
index.html / app.js from the package template and host_candidates.json --
without rebuilding cutouts. --regenerate-site rebuilds the whole page
instead (slow: every cutout again). Observations without a trigger circle
in their frames are reported and skipped.
"""
import argparse
import logging
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pyrt_transient import PipelineConfig  # noqa: E402
from pyrt_transient.catalog import setup_catalog_cache  # noqa: E402
from pyrt_transient.detection.host_search import search  # noqa: E402
from pyrt_transient.io.observation_store import ObservationStore  # noqa: E402

log = logging.getLogger("host_search_backfill")


def refresh_site_in_place(obs_dir: Path, website_dir: Path) -> bool:
    """New page code and the host table into an existing site directory."""
    if not (website_dir / "index.html").exists():
        return False
    template_dir = REPO_ROOT / "pyrt_transient" / "template"
    for name in ("index.html", "app.js"):
        src = template_dir / name
        if src.exists():
            shutil.copy(src, website_dir / name)   # not copy2: the page should carry the refresh time
    src = obs_dir / search.JSON_FILENAME
    if src.exists():
        shutil.copy2(src, website_dir / search.JSON_FILENAME)
    return True


def backfill(obs_dir: Path, config, public_dir, force: bool, regenerate: bool) -> str:
    obs_dir = Path(obs_dir)
    if not obs_dir.is_dir() or not obs_dir.name.startswith("obs_"):
        return "not an obs_<id> directory"
    obs_id = obs_dir.name[len("obs_"):]
    store = ObservationStore(obs_dir.parent, obs_id)
    tables, _ = store.load_existing_tables()
    tables = [t for t in tables if not t.meta.get("IS_STACK")]
    if not tables:
        return "no registered frames"
    if search.trigger_circle(tables) is None:
        return "no trigger circle in the frames, nothing to search"
    if force:
        for name in (search.STATE_FILENAME, search.ECSV_FILENAME, search.JSON_FILENAME):
            try:
                (obs_dir / name).unlink()
            except FileNotFoundError:
                pass
    if (obs_dir / search.ECSV_FILENAME).exists():
        outcome = "host table already present"
    else:
        table = search.maybe_run_host_search(obs_dir, tables, config, log)
        if table is None:
            return "search did not run (see the log above)"
        top = table[0] if len(table) else None
        outcome = (f"{len(table)} galaxies; top {top['mag']:.2f} mag, S/N {top['residual_snr']:.1f}, "
                   f"{top['excess_sigma']:+.1f} sigma" if top is not None and top["residual_rank"] > 0
                   else f"{len(table)} galaxies, prior only")
    website_dir = Path(public_dir) / obs_dir.name
    if regenerate:
        from pyrt_transient.web.orchestration import generate_frontend
        try:
            (website_dir / ".site_state.json").unlink()
        except (FileNotFoundError, AttributeError):
            pass
        config.generate_frontend = True
        config.base_public_dir = str(public_dir)
        ok = generate_frontend(obs_dir, obs_id, config)
        outcome += "; site regenerated" if ok else "; site regeneration FAILED"
    elif refresh_site_in_place(obs_dir, website_dir):
        outcome += "; site refreshed in place"
    else:
        outcome += f"; no site at {website_dir}"
    return outcome


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("obs_dirs", nargs="+", type=Path)
    ap.add_argument("--public-dir", type=Path, default=Path.home() / "public_html")
    ap.add_argument("--config", default=None)
    ap.add_argument("--force", action="store_true", help="rerun the search even if a host table exists")
    ap.add_argument("--regenerate-site", action="store_true", help="rebuild the whole page (cutouts too)")
    ap.add_argument("--no-prefetch", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    setup_catalog_cache(str(Path.home() / "catalog_cache"))
    if args.config:
        from pyrt_transient.core.config_loader import load_config_with_yaml_support
        config = load_config_with_yaml_support(args.config)
    else:
        config = PipelineConfig()
    if args.no_prefetch:
        config.host_search.prefetch = False
    for d in args.obs_dirs:
        try:
            outcome = backfill(d, config, args.public_dir, args.force, args.regenerate_site)
        except Exception as e:
            outcome = f"FAILED: {e}"
            log.exception(f"{d}: {e}")
        print(f"{d.name}: {outcome}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
