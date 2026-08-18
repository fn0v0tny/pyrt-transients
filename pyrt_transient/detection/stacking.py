"""Image stacking/co-addition for the GRB (blind_multicatalog) pipeline --
see FUTURE_IDEAS.md's "Image stacking/co-addition" (found while diagnosing
GRB151027B/211024B/210410A: consistently ~1.13-1.35x fainter than this
system's own single-exposure MAGLIM in every epoch, not fixable by any
per-epoch threshold tuning -- the only real fix is reaching deeper via a
co-added image).

Two real pieces of machinery already exist and are reused unchanged here:
- `pyrt-combine` (this host's installed `pyrt` package, confirmed via
  `pyrt-combine --help`) does the actual image combination -- invoked as a
  subprocess, the same boundary this package already keeps everywhere else
  (README: "this package never imports pyrt's astrometry/photometry code").
- `detection/subtraction/extraction.py::build_diff_ecsv` already does
  "SEP-detect + calibrate against a photometric survey catalog + adapt into
  the ECSV schema BlindMulticatalogStrategy expects". Its docstring explains
  a *diff* image can't be calibrated directly (real stars vanish in a diff),
  so it borrows a zeropoint from a separate science image -- but a *stacked*
  image still has all its real stars, so calibrating it directly is fine.
  Calling it with `diff_fits_path == science_fits_path == stack_path` reuses
  the exact same function for self-calibration, with zero new
  detection/calibration code.

The resulting stack ECSV is just one more table handed to
BlindMulticatalogStrategy.run() alongside the real per-epoch tables (see
pipeline_magic.py) -- it runs "in parallel with", not instead of, per-epoch
detection: no changes to catalog_match.py or clustering.py. In particular,
`min_n_detections` (the cross-epoch clustering admission gate) is left
untouched -- a stack-only candidate is gated exactly like a real epoch would
be. See FUTURE_IDEAS.md for the deferred idea of a depth-dependent
threshold.
"""

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from astropy.io import fits
from astropy.table import Table

from pyrt_transient.detection.subtraction import extraction as sub_extraction
from pyrt_transient.transients import open_ecsv_file

logger = logging.getLogger("detection.stacking")

_STACK_STATE_FILENAME = "stack_state.json"
_STACK_FITS_NAME = "stack.fits"
_STACK_ECSV_NAME = "stack.ecsv"
_STACK_TRANSIENTS_CACHE_NAME = "stack_transients.ecsv"


def select_stack_inputs(fits_paths: List[Path], max_epochs: int) -> List[Path]:
    """Most recent `max_epochs` paths (or all, if fewer) -- bounds
    pyrt-combine's runtime/memory rather than feeding it an ever-growing
    campaign. `fits_paths` is assumed ordered oldest-to-newest (the same
    order detection_tables accumulate in throughout this codebase).
    """
    if max_epochs <= 0 or len(fits_paths) <= max_epochs:
        return list(fits_paths)
    return list(fits_paths[-max_epochs:])


def should_rebuild_stack(
    n_real_epochs: int,
    last_build_n_epochs: int,
    min_epochs: int,
    rebuild_interval: int,
) -> bool:
    """Whether it's worth spending a fresh `pyrt-combine` run: not before
    `min_epochs` real epochs exist, and not again until at least
    `rebuild_interval` more have accumulated since the last build (bounds
    how often the expensive combine step actually runs).
    """
    if n_real_epochs < min_epochs:
        return False
    if last_build_n_epochs <= 0:
        return True
    return n_real_epochs >= last_build_n_epochs + rebuild_interval


def combine_epochs(
    fits_paths: List[Path],
    output_path: Path,
    uniform: bool = True,
) -> Optional[Path]:
    """Combine `fits_paths` into `output_path` via the `pyrt-combine` CLI.

    Returns `output_path` on success, or None -- never raises -- if the
    binary isn't installed, the subprocess exits non-zero, or it exits 0
    but didn't actually produce the output file. Same degrade-gracefully
    convention as templates.py/differencing.py's missing swarp/hotpants
    handling: callers should treat this exactly like a missing template
    (skip this feature this run, don't crash the pipeline).
    """
    binary = shutil.which("pyrt-combine")
    if binary is None:
        logger.warning("pyrt-combine: binary not found on PATH -- skipping stack build")
        return None

    if len(fits_paths) < 2:
        logger.warning(f"pyrt-combine: need at least 2 input images, got {len(fits_paths)}")
        return None

    output_path = Path(output_path)
    cmd = [binary, "-o", str(output_path)]
    if uniform:
        cmd.append("-u")
    cmd.extend(str(p) for p in fits_paths)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        logger.warning("pyrt-combine: timed out")
        return None
    except OSError as e:
        logger.warning(f"pyrt-combine: could not launch subprocess: {e}")
        return None

    if result.returncode != 0:
        logger.warning(f"pyrt-combine: exited {result.returncode}: {result.stderr.strip()[-500:]}")
        return None

    if not output_path.exists():
        logger.warning(f"pyrt-combine: exited 0 but {output_path} was not created")
        return None

    logger.info(f"pyrt-combine: combined {len(fits_paths)} epochs -> {output_path}")
    return output_path


def build_stack_ecsv(
    stack_fits_path: Path,
    output_path: Optional[Path] = None,
    photometric_catalog: str = "ps1",
    detect_thresh: float = 5.0,
    n_combined: Optional[int] = None,
) -> Optional[Path]:
    """Detect+calibrate sources directly on the stacked image and write an
    ECSV in the schema BlindMulticatalogStrategy expects.

    Delegates entirely to extraction.build_diff_ecsv, passing the stack as
    both the "diff" and the "science" image -- see this module's docstring
    for why that's correct here (a stack, unlike a subtraction diff, still
    has its own real stars to calibrate a zeropoint against). Adds
    IS_STACK/NCOMBINE meta on top so downstream code (and this module's own
    trigger logic) can recognize a stack-derived table.

    MAGLIM is then replaced with an empirical measurement from the stack's
    own actually-detected sources (_empirical_maglim) -- see that
    function's docstring for what's measured and why.

    Returns the ECSV path, or None if detection/calibration failed --
    callers should treat this the same as any other unusable epoch.
    """
    ecsv_path = sub_extraction.build_diff_ecsv(
        diff_fits_path=stack_fits_path,
        science_fits_path=stack_fits_path,
        output_path=output_path,
        photometric_catalog=photometric_catalog,
        detect_thresh=detect_thresh,
    )
    if ecsv_path is None:
        return None

    table = Table.read(str(ecsv_path), format="ascii.ecsv")
    table.meta["IS_STACK"] = True
    if n_combined is not None:
        table.meta["NCOMBINE"] = int(n_combined)

    empirical_maglim = _empirical_maglim(table, stack_fits_path)
    if empirical_maglim is not None:
        table.meta["MAGLIM"] = empirical_maglim

    table.write(str(ecsv_path), format="ascii.ecsv", overwrite=True)
    return ecsv_path


def _empirical_maglim(table: Table, stack_fits_path: Path,
                       percentile: float = 90.0, sigma_cut: float = 5.0) -> Optional[float]:
    """MAGLIM measured empirically from the stack's own actually-detected
    sources: the `percentile`-th faintest MAG_CALIB among sources whose
    real signal-to-noise (flux over this image's own directly-measured
    background noise) clears `sigma_cut` -- same "percentile of reliably
    measured detections" technique detection/reference_frame.py's
    ImageQuality already uses for limiting_mag, applied to real sources
    this image actually produced, not a formula.

    Deliberately *not* using MAGERR_CALIB (SEP's per-source flux-error
    model) for the reliability cut -- verified directly on a real 20-epoch
    GRB211024B stack that it's overconfident here: every one of 183
    detections had MAGERR_CALIB < 0.1 even at MAG_CALIB=23.1, because
    SEP's error model assumes single-exposure Poisson+read-noise
    statistics via the header's (unchanged, single-frame) GAIN, which
    doesn't hold once frames have been combined. The image's own
    background RMS, measured directly from its pixels, doesn't depend on
    trusting that per-source model -- whatever pyrt-combine's internal
    weighting/rejection actually did to the noise, this measures the
    result directly, then still reports a genuinely empirical faintest-
    reliable-detection magnitude, not a formula-derived one.

    Returns None if MAG_CALIB/FLUX aren't available, the image can't be
    read, or no detection clears sigma_cut.
    """
    if "MAG_CALIB" not in table.colnames or "FLUX" not in table.colnames or len(table) == 0:
        return None

    try:
        from stdpipe import photometry as stdpipe_photometry
    except ImportError:
        return None

    try:
        with fits.open(Path(stack_fits_path)) as hdul:
            image = np.array(hdul[0].data, dtype=np.float64)
            header = hdul[0].header.copy()
        mask = ~np.isfinite(image)
        _, backrms = stdpipe_photometry.get_background(image, mask=mask, size=128, get_rms=True)
        aper_px = sub_extraction._resolve_fwhm_px(header)
        aperture_noise = float(np.nanmedian(backrms)) * np.sqrt(np.pi * aper_px ** 2)
    except Exception as e:
        logger.debug(f"{stack_fits_path}: could not measure background noise: {e}")
        return None

    if not np.isfinite(aperture_noise) or aperture_noise <= 0:
        return None

    mag = np.asarray(table["MAG_CALIB"], dtype=float)
    flux = np.asarray(table["FLUX"], dtype=float)
    snr = flux / aperture_noise
    good = np.isfinite(mag) & np.isfinite(snr) & (snr >= sigma_cut)
    if not np.any(good):
        return None
    return float(np.percentile(mag[good], percentile))


def _fits_path_for_table(table: Table) -> Optional[Path]:
    filename = table.meta.get("filename")
    if not filename:
        return None
    return Path(filename).with_suffix(".fits")


def _band_exptime_key(table: Table) -> Optional[Tuple[str, float]]:
    """(filter, exposure_time) identity for grouping epochs before
    combining.

    pyrt-combine does not normalize flux for exposure-time differences in
    either weighting mode -- verified directly from its own source
    (pyrt.cli.combine.compute_weights): uniform mode assigns every image
    the same scalar weight regardless of EXPTIME, and weighted mode's
    per-image "counts" are for optimizing S/N on a hypothetical *variable*
    source (GRB decay tracking), not for normalizing background flux
    across frames of different exposure length. Combining frames with
    different exposure times (real per-epoch EXPTIME does vary within a
    single campaign -- verified directly: a real GRB151027B replay dataset
    mixed frames from 10s to several tens of seconds) or different filters
    would silently sum physically incompatible pixel values.

    Uses PHFILTER (the standardized photometric band, e.g. "Sloan_r") --
    more reliable than the raw hardware FILTER/FILTA code, which can be an
    instrument-specific label like "N" for "no filter/clear" that still
    maps to a real photometric band via PHFILTER (verified on real data:
    FILTER="N" epochs all carry PHFILTER="Sloan_r"). EXPTIME is rounded to
    the nearest 0.1s -- real per-epoch values can jitter by sub-second
    amounts for what is nominally "the same" exposure setting.

    Returns None if either piece of metadata is missing -- such an epoch
    can't be verified safe to combine with anything and is excluded by
    _select_consistent_group.
    """
    phfilter = table.meta.get("PHFILTER")
    exptime = table.meta.get("EXPTIME")
    if phfilter is None or exptime is None:
        return None
    try:
        return (str(phfilter), round(float(exptime), 1))
    except (TypeError, ValueError):
        return None


def _select_consistent_group(pairs: List[Tuple[Table, Path]]) -> Tuple[List[Tuple[Table, Path]], Optional[Tuple[str, float]]]:
    """Groups (table, fits_path) pairs by _band_exptime_key and returns the
    largest group, plus its (filter, exptime) key -- see that function's
    docstring for why mixing filters/exposure-times can't be safely
    combined. Pairs whose table lacks PHFILTER/EXPTIME are excluded
    entirely (can't be verified safe either way).
    """
    groups: Dict[Tuple[str, float], List[Tuple[Table, Path]]] = {}
    for t, p in pairs:
        key = _band_exptime_key(t)
        if key is not None:
            groups.setdefault(key, []).append((t, p))
    if not groups:
        return [], None
    best_key = max(groups, key=lambda k: len(groups[k]))
    return groups[best_key], best_key


def _read_state(obs_dir: Path) -> dict:
    state_path = obs_dir / _STACK_STATE_FILENAME
    if not state_path.exists():
        return {}
    try:
        with open(state_path) as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        logger.debug(f"stack_state.json unreadable ({e}), treating as empty")
        return {}


def _write_state(obs_dir: Path, state: dict) -> None:
    state_path = obs_dir / _STACK_STATE_FILENAME
    try:
        with open(state_path, "w") as f:
            json.dump(state, f)
    except OSError as e:
        logger.debug(f"Could not write {state_path}: {e}")


def _previous_best_score(obs_dir: Path) -> float:
    candidates_path = obs_dir / "candidates.tbl"
    if not candidates_path.exists():
        return 0.0
    try:
        candidates = Table.read(str(candidates_path), format="ascii.ipac")
    except Exception as e:
        logger.debug(f"Could not read {candidates_path}: {e}")
        return 0.0
    if len(candidates) == 0 or "quality_score" not in candidates.colnames:
        return 0.0
    return float(max(candidates["quality_score"]))


def maybe_build_stack_table(
    obs_dir: Path,
    detection_tables: List[Table],
    config,
    logger_=None,
) -> Optional[Table]:
    """Orchestration entry point pipeline_magic.py calls right before
    BlindMulticatalogStrategy.run(). Returns the Table that should
    represent the stack epoch for this run, or None if there shouldn't be
    one (disabled, or not enough real epochs yet).

    This is the *authoritative* state, not an incremental delta: the
    caller should drop any previous IS_STACK table it already had (e.g.
    loaded from disk by ObservationStore.load_existing_tables()) and
    replace it with whatever this function returns, rather than appending
    -- both to avoid double-counting an already-loaded stack table, and so
    a fresh rebuild's improved depth is reflected immediately in the same
    run rather than lagging a full invocation behind.

    `detection_tables` may already include a previously-loaded stack table
    (IS_STACK meta) -- it's filtered out of the "how many real epochs do we
    have" count and ignored otherwise; disk state (stack.ecsv,
    stack_state.json) is authoritative, not whatever happens to already be
    in memory.

    Rebuilding (an actual fresh pyrt-combine run) is throttled by
    should_rebuild_stack's cadence, and skipped entirely once an existing
    candidate already scores at or above stacking_score_threshold -- but
    once a stack exists on disk, it keeps being returned (and thus stays
    included) on every subsequent call regardless of score, so a
    previously stack-anchored candidate doesn't disappear from future runs
    just because it did its job.
    """
    log = logger_ or logger
    obs_dir = Path(obs_dir)
    det_cfg = config.detection if config else None

    if det_cfg is None or not det_cfg.stacking_enabled:
        return None

    real_tables = [t for t in detection_tables if not t.meta.get("IS_STACK")]
    n_real_epochs = len(real_tables)

    stack_ecsv_path = obs_dir / _STACK_ECSV_NAME
    state = _read_state(obs_dir)
    last_build_n_epochs = int(state.get("last_build_n_epochs", 0))

    if n_real_epochs >= det_cfg.stacking_min_epochs:
        best_score = _previous_best_score(obs_dir)
        due_for_rebuild = should_rebuild_stack(
            n_real_epochs, last_build_n_epochs,
            det_cfg.stacking_min_epochs, det_cfg.stacking_rebuild_interval,
        )
        if best_score >= det_cfg.stacking_score_threshold:
            log.info(
                f"Stacking: not (re)building this run (existing best quality_score "
                f"{best_score:.2f} >= stacking_score_threshold {det_cfg.stacking_score_threshold})"
            )
        elif due_for_rebuild:
            _attempt_rebuild(obs_dir, real_tables, det_cfg, n_real_epochs, log)

    if not stack_ecsv_path.exists():
        return None

    return open_ecsv_file(str(stack_ecsv_path), verbose=False)


def _attempt_rebuild(obs_dir: Path, real_tables: List[Table], det_cfg, n_real_epochs: int, log) -> None:
    """Runs pyrt-combine + build_stack_ecsv and updates state on success.
    Failures at any step just leave the previous stack (if any) in place --
    logged by combine_epochs/build_stack_ecsv themselves.
    """
    pairs = [(t, _fits_path_for_table(t)) for t in real_tables]
    pairs = [(t, p) for t, p in pairs if p is not None]

    n_before = len(pairs)
    pairs, group_key = _select_consistent_group(pairs)
    if n_before and len(pairs) < n_before:
        excluded = n_before - len(pairs)
        band, exptime = group_key if group_key else ("?", "?")
        log.info(f"Stacking: excluded {excluded}/{n_before} epoch(s) with a different "
                 f"filter/exposure time than the majority group ({band}, {exptime}s) -- "
                 f"pyrt-combine does not normalize for either")

    pairs = select_stack_inputs(pairs, det_cfg.stacking_max_epochs)
    if len(pairs) < 2:
        log.warning("Stacking: fewer than 2 resolvable, same-filter/same-exposure FITS paths, skipping")
        return
    fits_paths = [p for _, p in pairs]

    stack_fits_path = obs_dir / _STACK_FITS_NAME
    combined = combine_epochs(
        fits_paths, stack_fits_path, uniform=det_cfg.stacking_uniform_weighting,
    )
    if combined is None:
        return

    stack_ecsv_path = obs_dir / _STACK_ECSV_NAME
    built = build_stack_ecsv(
        combined, output_path=stack_ecsv_path,
        photometric_catalog=det_cfg.photometric_catalog,
        detect_thresh=det_cfg.stacking_detect_thresh,
        n_combined=len(fits_paths),
    )
    if built is None:
        return

    # Invalidate BlindMulticatalogStrategy's Step-1 per-epoch cache so the
    # refreshed (deeper) stack actually gets reprocessed instead of being
    # skipped as "already processed".
    (obs_dir / _STACK_TRANSIENTS_CACHE_NAME).unlink(missing_ok=True)

    _write_state(obs_dir, {"last_build_n_epochs": n_real_epochs})
    log.info(f"Stacking: rebuilt from {len(fits_paths)} epochs")
