"""The host search as the pipeline runs it: gate, stack, subtract, measure,
write; and the template prefetch it kicks off on the first frame.

maybe_run_host_search is called once per pipeline run, after detection,
for observations whose frames carry a trigger circle (GRB_RA, GRB_DEC,
GRB_ERR in the ECSV meta). It is throttled like stacking: not before
`min_frames` long exposures exist, then again every `rebuild_interval`
new ones. It never raises: a missing optional dependency, a template with
no coverage, a hotpants failure or a catalogue outage each end in a
warning and no table, and the next run tries again.

Outputs in the observation directory:
  host_candidates.ecsv   one row per galaxy in the circle
  host_candidates.json   the same plus the run's provenance, for the page
  host_search_state.json throttle state
"""
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from astropy.nddata import Cutout2D
from astropy.table import Table

from pyrt_transient.detection.host_search import galaxies as gal
from pyrt_transient.detection.host_search import templates as tpl
from pyrt_transient.detection.stacking import _fits_path_for_table as _fits_path_for

logger = logging.getLogger("detection.host_search")

STATE_FILENAME = "host_search_state.json"
ECSV_FILENAME = "host_candidates.ecsv"
JSON_FILENAME = "host_candidates.json"
TABLE_COLUMNS = ("rank", "ra", "dec", "mag", "catalog", "type", "radius_arcsec", "zphot", "n_exp", "prior_rank",
                 "residual_rank", "residual_flux", "residual_err", "residual_snr", "excess_frac", "excess_sigma",
                 "residual_mag", "residual_lim5", "galaxy_mag_meas", "sep_center_arcsec")


def trigger_circle(detection_tables) -> Optional[Tuple[float, float, float]]:
    """(ra, dec, radius_arcsec) of the trigger error circle from the first
    frame that carries one, or None for a non-trigger observation."""
    for t in detection_tables:
        meta = getattr(t, "meta", None) or {}
        try:
            ra, dec, err = float(meta["GRB_RA"]), float(meta["GRB_DEC"]), float(meta["GRB_ERR"])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(ra) and np.isfinite(dec) and np.isfinite(err) and err > 0:
            return ra, dec, err * 3600.0
    return None


def _read_state(obs_dir: Path) -> dict:
    try:
        return json.loads((obs_dir / STATE_FILENAME).read_text())
    except (OSError, ValueError):
        return {}


def _write_state(obs_dir: Path, state: dict) -> None:
    try:
        (obs_dir / STATE_FILENAME).write_text(json.dumps(state))
    except OSError:
        pass


def template_size_arcmin(detection_tables, cfg) -> float:
    """The template covers the whole frame (FIELD in the meta, degrees)
    when the config asks for it, else the configured size: one template
    per pointing serves the prefetch, every night and any error circle
    that fits the frame."""
    if cfg.template_size_arcmin and cfg.template_size_arcmin > 0:
        return float(cfg.template_size_arcmin)
    for t in detection_tables:
        try:
            f = float(t.meta.get("FIELD"))
            if np.isfinite(f) and f > 0:
                return f * 60.0
        except (TypeError, ValueError):
            continue
    return 20.0


def prefetch_async(ra: float, dec: float, size_arcmin: float, cfg, log=None) -> bool:
    """Start a detached process that fetches the field template into the
    cache, so the pipeline job that saw the first (pointing) frame does
    not wait for it and the first science frame finds it ready. Returns
    whether the process was started."""
    log = log or logger
    cache_dir = cfg.template_cache_dir or str(tpl.default_cache_dir())
    if tpl.find_cached_template(Path(cache_dir), ra, dec, cfg.band, size_arcmin) is not None:
        return False
    cmd = [sys.executable, "-m", "pyrt_transient.detection.host_search.prefetch",
           "--ra", f"{ra:.6f}", "--dec", f"{dec:.6f}", "--size-arcmin", f"{size_arcmin:.2f}",
           "--band", cfg.band, "--cache-dir", cache_dir, "--pixscale", f"{cfg.template_pixscale_arcsec:g}",
           "--timeout", f"{cfg.template_timeout_s:g}", "--sources", *cfg.template_sources]
    try:
        subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)
    except OSError as e:
        log.warning(f"Host search: could not start the template prefetch ({e})")
        return False
    log.info(f"Host search: template prefetch started for {ra:.4f} {dec:+.4f} ({size_arcmin:.0f}')")
    return True


def run_host_search(obs_dir: Path, detection_tables, cfg, log=None) -> Optional[Table]:
    """The search itself; raises on real errors (the caller catches)."""
    from pyrt_transient.detection.host_search import stack as stk
    from pyrt_transient.detection.host_search import subtract as sub

    log = log or logger
    obs_dir = Path(obs_dir)
    timing = {}
    circle = trigger_circle(detection_tables)
    if circle is None:
        return None
    ra0, dec0, radius = circle

    # galaxies in the circle (cached per observation: the sky does not change)
    t0 = time.monotonic()
    gal_cache = obs_dir / "host_galaxies.json"
    galaxies = None
    try:
        cached = json.loads(gal_cache.read_text())
        if abs(cached.get("radius_arcsec", -1) - radius) < 1e-3 and cached.get("max_mag") == cfg.max_galaxy_mag:
            galaxies = cached["galaxies"]
    except (OSError, ValueError, KeyError, TypeError):
        pass
    if galaxies is None:
        galaxies = gal.query_galaxies_in_circle(ra0, dec0, radius, cfg.galaxy_catalogs, cfg.max_galaxy_mag,
                                                cfg.band, cfg.query_timeout_s, log)
        if galaxies is None:
            log.warning("Host search: galaxy catalogues unavailable, no host table this run")
            return None
        try:
            gal_cache.write_text(json.dumps({"radius_arcsec": radius, "max_mag": cfg.max_galaxy_mag,
                                             "galaxies": galaxies}, default=float))
        except OSError:
            pass
    galaxies = gal.rank_by_prior(list(galaxies), radius)
    timing["galaxies_s"] = time.monotonic() - t0
    if not galaxies:
        log.info("Host search: no catalogued galaxy in the error circle")
        return _write_outputs(obs_dir, [], circle, None, 0, timing, cfg, log)

    # frames and stack
    pairs = [(_fits_path_for(t), t) for t in detection_tables if not t.meta.get("IS_STACK")]
    selected = stk.select_frames(pairs, cfg.min_exptime_fraction, cfg.max_frames)
    if len(selected) < cfg.min_frames:
        log.info(f"Host search: {len(selected)} usable frames, need {cfg.min_frames}")
        return None
    stack = stk.build_stack(selected, log)
    if stack is None:
        log.warning("Host search: no frame could be stacked")
        return None
    timing["stack_s"] = stack.seconds

    # template
    t0 = time.monotonic()
    size = template_size_arcmin(detection_tables, cfg)
    cen = stack.wcs.pixel_to_world(stack.data.shape[1] / 2, stack.data.shape[0] / 2)
    got = tpl.fetch_template(float(cen.ra.deg), float(cen.dec.deg), size, cfg.band, cfg.template_sources,
                             cfg.template_cache_dir, cfg.template_pixscale_arcsec, cfg.template_timeout_s, log)
    if got is None:
        log.warning("Host search: no survey template covers this field, ranking by prior only")
        rows = [dict(g) for g in galaxies]
        return _write_outputs(obs_dir, rows, circle, None, stack.n_frames, timing, cfg, log)
    template, template_wcs, source = got
    timing["template_s"] = time.monotonic() - t0

    # region around the circle, subtraction, photometry
    t0 = time.monotonic()
    pixscale = float(np.sqrt(np.abs(np.linalg.det(stack.wcs.pixel_scale_matrix))) * 3600.0)
    region_px = int(min(max(2 * radius * cfg.region_margin / pixscale + 8 * stack.fwhm_px, cfg.region_min_px),
                        max(stack.data.shape)))
    xy = stack.wcs.all_world2pix([[ra0, dec0]], 0)[0]
    cut = Cutout2D(stack.data, (xy[0], xy[1]), region_px, wcs=stack.wcs, mode="partial", fill_value=np.nan)
    sci = np.asarray(cut.data, dtype=np.float32)
    tmpl, tmask = sub.match_template(template, template_wcs, cut.wcs, sci.shape, cfg.template_pixscale_arcsec,
                                     cfg.template_fwhm_arcsec)
    mask = ~np.isfinite(sci) | tmask | sub.saturation_mask(sci)
    diff = sub.run_hotpants(np.where(mask, 0.0, sci), tmpl, mask, stack.fwhm_px, stack.n_frames,
                            exe=cfg.hotpants_executable, log=log)
    if diff is None:
        rows = [dict(g) for g in galaxies]
        return _write_outputs(obs_dir, rows, circle, source, stack.n_frames, timing, cfg, log)
    rows = sub.measure_galaxies(diff, sci, mask, cut.wcs, galaxies, stack.fwhm_px, stack.zp, cfg.aperture_fwhm)
    med, mad = sub.population_sigma(rows)
    timing["subtract_s"] = time.monotonic() - t0
    timing["population_median"] = med
    timing["population_mad"] = mad
    log.info(f"Host search: {len(rows)} galaxies measured on the difference image "
             f"(population excess {med:+.3f} +- {mad:.3f})")
    return _write_outputs(obs_dir, rows, circle, source, stack.n_frames, timing, cfg, log)


def _write_outputs(obs_dir: Path, rows: List[dict], circle, template_source, n_frames, timing, cfg, log) -> Table:
    ra0, dec0, radius = circle
    measured = [r for r in rows if "residual_snr" in r]
    measured.sort(key=lambda r: -(r["excess_sigma"] if np.isfinite(r.get("excess_sigma", np.nan)) else r["residual_snr"]))
    for i, r in enumerate(measured, 1):
        r["residual_rank"] = i
    # Final order: measured galaxies by residual, then the rest by prior.
    ordered = measured + sorted([r for r in rows if "residual_snr" not in r], key=lambda r: r.get("prior_rank", 1e9))
    for i, r in enumerate(ordered, 1):
        r["rank"] = i
        r["sep_center_arcsec"] = gal.sep_arcsec(r["ra"], r["dec"], ra0, dec0)
    data = {c: [] for c in TABLE_COLUMNS}
    for r in ordered:
        for c in TABLE_COLUMNS:
            v = r.get(c)
            if c in ("catalog", "type"):
                data[c].append(str(v) if v is not None else "")
            elif c in ("rank", "prior_rank", "residual_rank"):
                data[c].append(int(v) if v is not None else -1)
            else:
                data[c].append(float(v) if v is not None else np.nan)
    table = Table(data)
    table.meta.update({"trigger_ra": ra0, "trigger_dec": dec0, "trigger_radius_arcsec": radius,
                       "template_source": template_source or "none", "n_frames": int(n_frames),
                       "aperture_fwhm": cfg.aperture_fwhm, "generated": time.time()})
    try:
        table.write(obs_dir / ECSV_FILENAME, format="ascii.ecsv", overwrite=True)
        payload = {"trigger": {"ra": ra0, "dec": dec0, "radius_arcsec": radius},
                   "template_source": template_source, "n_frames": int(n_frames), "timing": timing,
                   "generated": time.time(),
                   "galaxies": [{k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                                 for k, v in r.items() if k != "mags"} for r in ordered]}
        (obs_dir / JSON_FILENAME).write_text(json.dumps(payload, default=float))
    except OSError as e:
        log.warning(f"Host search: could not write outputs ({e})")
    if measured:
        top = measured[0]
        log.info(f"Host search: top galaxy {top['mag']:.2f} mag at {top['sep_center_arcsec'] / 60:.1f}' from the "
                 f"trigger, residual S/N {top['residual_snr']:.1f}, excess {top['excess_sigma']:+.1f} sigma "
                 f"(prior rank {top['prior_rank']})")
    return table


def maybe_run_host_search(obs_dir, detection_tables, config, log=None) -> Optional[Table]:
    """Pipeline hook. Returns the host-candidate table when the search ran,
    else None. Never raises."""
    log = log or logger
    cfg = getattr(config, "host_search", None)
    if cfg is None or not cfg.enabled:
        return None
    obs_dir = Path(obs_dir)
    try:
        circle = trigger_circle(detection_tables)
        if circle is None:
            return None
        if circle[2] / 60.0 > cfg.max_error_radius_arcmin:
            log.info(f"Host search: error radius {circle[2] / 60:.1f}' above the {cfg.max_error_radius_arcmin:g}' limit")
            return None
        if cfg.prefetch:
            prefetch_async(circle[0], circle[1], template_size_arcmin(detection_tables, cfg), cfg, log)
        from pyrt_transient.detection.host_search import stack as stk
        n_long = len(stk.select_frames([(_fits_path_for(t), t) for t in detection_tables
                                        if not t.meta.get("IS_STACK")], cfg.min_exptime_fraction, cfg.max_frames))
        state = _read_state(obs_dir)
        last_n = int(state.get("n_frames", 0))
        if n_long < cfg.min_frames:
            return None
        if last_n and n_long < last_n + cfg.rebuild_interval and (obs_dir / ECSV_FILENAME).exists():
            log.info(f"Host search: {n_long} frames, last run at {last_n}; waiting for {last_n + cfg.rebuild_interval}")
            return None
        table = run_host_search(obs_dir, detection_tables, cfg, log)
        if table is not None:
            _write_state(obs_dir, {"n_frames": n_long, "time": time.time()})
        return table
    except ImportError as e:
        log.warning(f"Host search: skipped, missing dependency ({e}); pip install reproject")
    except Exception as e:
        log.warning(f"Host search: failed ({e})", exc_info=True)
    return None
