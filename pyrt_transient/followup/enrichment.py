"""Pipeline glue for the exposure calculator: annotate a finished candidate
list with the exposure the next follow-up frame needs, using the conditions
of the epoch most recently observed.

This is the first half of FUTURE_IDEAS.md's "Telescope strategy suggester"
(exposure time; filter and EMCCD on/off are still deferred, as is the
GRB-t0/time-since-trigger term). Per that entry it runs *on* a strategy's
output rather than inside the strategy: it takes the `(candidates,
lightcurves)` pair `BlindMulticatalogStrategy.run()` already returns plus
the detection tables it ran on, and touches no detection logic.

Every candidate row gets `followup_exptime_s`, but the JSON report is about
the highest-scoring one -- that's the candidate a follow-up decision is
actually made on.

Three things are deliberately *not* trusted:

- The magnitude a follow-up is planned against. A lightcurve's latest
  point is the right epoch to plan from, but with `new_source_siglim=1.5`
  it can carry a 0.7 mag error, and in the background-limited regime the
  required time scales roughly as 10^(0.8*dm) -- a x3-4 uncertainty. So
  the latest point is used only when it is well measured
  (`max_planning_magerr`); otherwise the last few points are averaged by
  inverse variance. The magnitude error is propagated into an exposure
  *range* in the report either way.
- The filter. Each epoch's `MAG_CALIB` is calibrated to that epoch's
  filter, and a campaign can switch bands mid-way (FUTURE_IDEAS' OBSID
  note: z-band epochs, then i-band). Only points from epochs in the
  reference frame's filter are used when any exist; otherwise the
  mismatch is flagged in the report rather than silently mixed.
- The model itself. Its constants are a fit to one instrument. The
  reference frame carries every calibrated source in it, so the model is
  checked against that frame's own measured errors on every run
  (`model_check` in the report) -- the same test that found the two bugs
  in the standalone script, now run automatically.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from astropy.table import Table

from pyrt_transient.followup import exposure as exposure_model
from pyrt_transient.followup.exposure import ExposureModelError, ReferenceConditions

REPORT_FILENAME = "followup_exposure.json"

_MAG_COLUMN_FALLBACKS = ("mag_weighted_mean", "MAG_CALIB")

# Above this rms (dex, log10 of predicted/measured error over the reference
# frame's own sources) the model is not describing this frame -- roughly
# twice the calibration's own quoted 0.047.
MODEL_CHECK_MAX_RMS_DEX = 0.1
MODEL_CHECK_MIN_SOURCES = 20


def _meta(table) -> Dict[str, Any]:
    return getattr(table, "meta", None) or {}


def _column(table: Table, name: str) -> np.ndarray:
    """A column as a plain float array, masked entries -> NaN."""
    return np.asarray(np.ma.filled(np.ma.asarray(table[name], dtype=float), np.nan), dtype=float)


def _epoch_mid_time(table) -> Optional[float]:
    """Mid-exposure unix time, the same quantity core/epochs.py stamps on
    every detection as `obs_time`; None if the frame carries no CTIME."""
    meta = _meta(table)
    try:
        ctime = float(meta["CTIME"])
    except (KeyError, TypeError, ValueError):
        return None
    try:
        exptime = float(meta.get("EXPTIME", 0.0) or 0.0)
    except (TypeError, ValueError):
        exptime = 0.0
    return ctime + exptime / 2.0


def select_reference_table(detection_tables: List[Table]) -> Optional[Table]:
    """The most recently observed real epoch -- "the previous image" a
    follow-up is planned against.

    Chosen by mid-exposure time, not list position: `ObservationStore.
    load_existing_tables()` globs the observation directory, so the list is
    in filesystem order once a run reloads it. Tables without a CTIME sort
    behind every dated one, in list order, so a fixture of undated tables
    still yields its last entry.

    Stack tables (`IS_STACK`, see detection/stacking.py) are skipped: a
    co-add's EXPTIME/BGSIGMA describe an image nobody can re-take in one
    exposure, and its background typically sits below the readout floor the
    model needs (`ReferenceConditions.from_meta` rejects it outright).
    """
    best, best_key = None, None
    for position, table in enumerate(detection_tables or []):
        if _meta(table).get("IS_STACK"):
            continue
        mid_time = _epoch_mid_time(table)
        key = (1, mid_time, position) if mid_time is not None else (0, 0.0, position)
        if best_key is None or key > best_key:
            best, best_key = table, key
    return best


def _reference_name(table: Table) -> Optional[str]:
    meta = _meta(table)
    for key in ("filename", "FITSFILE", "detf"):
        value = meta.get(key)
        if value:
            return Path(str(value)).name
    return None


def _epoch_filters(detection_tables: List[Table]) -> Dict[int, Optional[str]]:
    """epoch_id (position in detection_tables, as core/epochs.py assigns it)
    -> that epoch's FILTER."""
    filters = {}
    for i, table in enumerate(detection_tables or []):
        value = _meta(table).get("FILTER")
        filters[i] = str(value) if value is not None else None
    return filters


def planning_magnitude_from_lightcurve(
    lightcurve: Table,
    reference_filter: Optional[str] = None,
    epoch_filters: Optional[Dict[int, Optional[str]]] = None,
    max_planning_magerr: float = 0.2,
    n_average: int = 3,
) -> Optional[Dict[str, Any]]:
    """Magnitude to plan a follow-up against, from a candidate's lightcurve.

    Returns None when the lightcurve has no usable calibrated point,
    otherwise a dict with `mag`, `mag_err`, `source` (how it was derived),
    `filter` (the band the points were measured in, when known) and
    `filter_mismatch` (True if that band differs from `reference_filter`).
    """
    if lightcurve is None or len(lightcurve) == 0 or "MAG_CALIB" not in lightcurve.colnames:
        return None

    mags = _column(lightcurve, "MAG_CALIB")
    errs = (_column(lightcurve, "MAGERR_CALIB") if "MAGERR_CALIB" in lightcurve.colnames
            else np.full(len(mags), np.nan))
    times = (_column(lightcurve, "obs_time") if "obs_time" in lightcurve.colnames
             else np.arange(len(mags), dtype=float))
    usable = np.isfinite(mags) & np.isfinite(times)
    if not np.any(usable):
        return None

    point_filter = np.array([None] * len(mags), dtype=object)
    if epoch_filters and "epoch_id" in lightcurve.colnames:
        epoch_ids = _column(lightcurve, "epoch_id")
        point_filter = np.array(
            [epoch_filters.get(int(e)) if np.isfinite(e) else None for e in epoch_ids],
            dtype=object,
        )

    # Prefer points measured in the reference frame's own band.
    mismatch = False
    chosen = usable
    if reference_filter is not None:
        same_band = usable & np.array([f == reference_filter for f in point_filter], dtype=bool)
        if np.any(same_band):
            chosen = same_band
        else:
            mismatch = any(f is not None for f in point_filter[usable])

    order = np.flatnonzero(chosen)[np.argsort(times[chosen])][::-1]  # newest first
    latest = order[0]
    used_filter = point_filter[latest]

    # A well-measured latest point stands on its own.
    if np.isfinite(errs[latest]) and errs[latest] <= max_planning_magerr:
        return {"mag": float(mags[latest]), "mag_err": float(errs[latest]),
                "source": "latest lightcurve point", "filter": used_filter,
                "filter_mismatch": mismatch}

    # Otherwise average the most recent few by inverse variance; points
    # without an error get a nominal one so they still count.
    recent = order[:n_average]
    weights = 1.0 / np.where(np.isfinite(errs[recent]) & (errs[recent] > 0),
                             errs[recent], max_planning_magerr) ** 2
    mag = float(np.sum(weights * mags[recent]) / np.sum(weights))
    mag_err = float(1.0 / np.sqrt(np.sum(weights)))
    return {"mag": mag, "mag_err": mag_err,
            "source": f"inverse-variance mean of last {len(recent)} point(s)",
            "filter": used_filter, "filter_mismatch": mismatch}


def candidate_magnitude(row, lightcurves: Optional[Dict[str, Table]], **kwargs) -> Optional[Dict[str, Any]]:
    """Planning magnitude for one candidate row: from its lightcurve when it
    has one (see `planning_magnitude_from_lightcurve`), else from the row's
    own summary columns.

    The row's own `MAG_CALIB` is the *best-quality* epoch's photometry
    (clustering.py picks the highest-scoring detection), not the latest --
    hence the lightcurve is preferred.
    """
    transient_id = str(row["transient_id"]) if "transient_id" in row.colnames else None
    if lightcurves and transient_id and transient_id in lightcurves:
        result = planning_magnitude_from_lightcurve(lightcurves[transient_id], **kwargs)
        if result is not None:
            return result

    for column in _MAG_COLUMN_FALLBACKS:
        if column not in row.colnames:
            continue
        try:
            value = float(np.ma.filled(row[column], np.nan))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            return {"mag": value, "mag_err": float("nan"),
                    "source": f"candidate column {column}", "filter": None,
                    "filter_mismatch": False}
    return None


def check_model_against_frame(reference_table: Table, conditions: ReferenceConditions) -> Dict[str, Any]:
    """Compare the model's predicted error with the measured MAGERR_CALIB of
    every calibrated source in the reference frame. Returns a dict with
    `status` ("ok" / "poor" / "unavailable") and the residual statistics."""
    if not {"MAG_CALIB", "MAGERR_CALIB"} <= set(reference_table.colnames):
        return {"status": "unavailable", "reason": "reference frame has no calibrated photometry"}
    mag = _column(reference_table, "MAG_CALIB")
    err = _column(reference_table, "MAGERR_CALIB")
    good = np.isfinite(mag) & np.isfinite(err) & (err > 0)
    if good.sum() < MODEL_CHECK_MIN_SOURCES:
        return {"status": "unavailable",
                "reason": f"only {int(good.sum())} measured sources (need {MODEL_CHECK_MIN_SOURCES})"}
    predicted = 10 ** exposure_model.log_magerror(
        exposure_model.instrumental_magnitude(mag[good], conditions.magzero),
        conditions.bgsigma_adu, conditions.fwhm_px, conditions.gain,
    )
    residual = np.log10(predicted / err[good])
    rms = float(np.std(residual))
    return {
        "status": "ok" if rms <= MODEL_CHECK_MAX_RMS_DEX else "poor",
        "n_sources": int(good.sum()),
        "median_dex": float(np.median(residual)),
        "rms_dex": rms,
        "max_rms_dex": MODEL_CHECK_MAX_RMS_DEX,
    }


def _solve(mag: float, target_magerror: float, conditions: ReferenceConditions,
           min_exptime_s: float, max_exptime_s: float) -> float:
    exptime = exposure_model.required_exptime(
        mag, target_magerror, conditions,
        min_exptime_s=min_exptime_s, max_exptime_s=max_exptime_s,
    )
    return float("nan") if exptime is None else float(exptime)


def recommend_exposures(candidates: Table, lightcurves: Optional[Dict[str, Table]],
                        detection_tables: Optional[List[Table]], config=None,
                        logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """Add `followup_mag` / `followup_exptime_s` columns to `candidates` and
    return a report on the highest-scoring one.

    `candidates` is modified in place (it arrives sorted by `quality_score`
    from clustering.py, so row 0 is the top candidate). The report always
    has a `status`: "ok", "skipped" (nothing to do) or "error" (the model
    could not be driven) with a `reason` -- never a bare empty result that
    reads the same as a real answer.
    """
    log = logger or logging.getLogger("followup.exposure")
    followup_cfg = getattr(config, "followup", None)
    target_snr = float(getattr(followup_cfg, "target_snr", 10.0))
    readout_noise_e = float(getattr(followup_cfg, "readout_noise_e",
                                    exposure_model.DEFAULT_READOUT_NOISE_E))
    min_exptime_s = float(getattr(followup_cfg, "min_exptime_s", 1.0))
    max_exptime_s = float(getattr(followup_cfg, "max_exptime_s", 3600.0))
    max_planning_magerr = float(getattr(followup_cfg, "max_planning_magerr", 0.2))

    report: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target_snr": target_snr,
        "n_candidates": int(len(candidates)) if candidates is not None else 0,
    }

    if not (target_snr > 0) or not (readout_noise_e >= 0) or not (0 < min_exptime_s < max_exptime_s):
        reason = (f"invalid followup config: target_snr={target_snr}, readout_noise_e="
                  f"{readout_noise_e}, exposure bracket [{min_exptime_s}, {max_exptime_s}]")
        log.warning(f"Exposure recommendation unavailable: {reason}")
        report.update(status="error", reason=reason)
        return report
    target_magerror = float(exposure_model.snr_to_magerror(target_snr))
    report["target_magerror"] = target_magerror

    if candidates is None or len(candidates) == 0:
        report.update(status="skipped", reason="no candidates")
        return report
    reference_table = select_reference_table(detection_tables)
    if reference_table is None:
        report.update(status="skipped", reason="no reference epoch available")
        return report

    reference_name = _reference_name(reference_table)
    try:
        conditions = ReferenceConditions.from_meta(
            _meta(reference_table), readout_noise_e=readout_noise_e, source=reference_name,
        )
    except ExposureModelError as exc:
        log.warning(f"Exposure recommendation unavailable: {exc}")
        report.update(status="error", reason=str(exc), reference={"source": reference_name})
        return report

    reference_filter = _meta(reference_table).get("FILTER")
    reference_filter = str(reference_filter) if reference_filter is not None else None
    report["reference"] = dict(conditions.as_dict(), filter=reference_filter)

    model_check = check_model_against_frame(reference_table, conditions)
    report["model_check"] = model_check
    if model_check["status"] == "poor":
        log.warning(
            f"Exposure model does not describe {reference_name or 'the reference frame'} well: "
            f"rms {model_check['rms_dex']:.3f} dex over {model_check['n_sources']} sources "
            f"(median {model_check['median_dex']:+.3f}); recommendations are unreliable"
        )

    epoch_filters = _epoch_filters(detection_tables)
    mag_kwargs = dict(reference_filter=reference_filter, epoch_filters=epoch_filters,
                      max_planning_magerr=max_planning_magerr)

    planned: List[Optional[Dict[str, Any]]] = []
    mags: List[float] = []
    exptimes: List[float] = []
    for row in candidates:
        info = candidate_magnitude(row, lightcurves, **mag_kwargs)
        planned.append(info)
        if info is None:
            mags.append(float("nan"))
            exptimes.append(float("nan"))
            continue
        mags.append(info["mag"])
        exptimes.append(_solve(info["mag"], target_magerror, conditions, min_exptime_s, max_exptime_s))

    candidates["followup_mag"] = np.array(mags, dtype=float)
    candidates["followup_exptime_s"] = np.array(exptimes, dtype=float)

    top_info = planned[0]
    top: Dict[str, Any] = {
        "transient_id": str(candidates["transient_id"][0]) if "transient_id" in candidates.colnames else None,
        "quality_score": (float(np.ma.filled(candidates["quality_score"][0], np.nan))
                          if "quality_score" in candidates.colnames else None),
    }
    for col, key in (("ALPHA_J2000", "ra"), ("DELTA_J2000", "dec")):
        if col in candidates.colnames:
            top[key] = float(np.ma.filled(candidates[col][0], np.nan))

    if top_info is None:
        report.update(status="error", reason="top candidate has no usable magnitude", top_candidate=top)
        log.warning("Exposure recommendation unavailable: top candidate has no usable magnitude")
        return report

    top_mag, top_err = top_info["mag"], top_info["mag_err"]
    top.update(
        magnitude=top_mag,
        magnitude_err=(top_err if np.isfinite(top_err) else None),
        magnitude_source=top_info["source"],
        magnitude_filter=top_info["filter"],
        filter_mismatch=bool(top_info["filter_mismatch"]),
    )
    if top_info["filter_mismatch"]:
        log.warning(
            f"Follow-up magnitude for {top['transient_id']} was measured in "
            f"{top_info['filter']!r}, reference frame is {reference_filter!r} -- "
            f"cross-band estimate"
        )

    exptime = exptimes[0]
    if not np.isfinite(exptime):
        top.update(exptime_s=None, unreachable_within_max_exptime_s=max_exptime_s)
        report.update(status="ok", reason=(
            f"SNR {target_snr:g} on mag {top_mag:.2f} is not reachable within "
            f"{max_exptime_s:g}s under these conditions"), top_candidate=top)
        log.info(
            f"Follow-up exposure for {top['transient_id']} (mag {top_mag:.2f}): "
            f"SNR {target_snr:g} unreachable within {max_exptime_s:g}s"
        )
        return report

    predicted_magerror = exposure_model.predict_magerror(top_mag, exptime, conditions)
    top.update(
        exptime_s=exptime,
        at_min_exptime=bool(exptime <= min_exptime_s),
        predicted_magerror=float(predicted_magerror),
        predicted_snr=float(exposure_model.magerror_to_snr(predicted_magerror)),
        magzero_at_exptime=conditions.magzero_at(exptime),
        bgsigma_at_exptime=conditions.bgsigma_at(exptime),
        limiting_mag_at_exptime=exposure_model.limiting_magnitude(exptime, conditions, target_magerror),
    )
    # The planning magnitude's own uncertainty, as an exposure range: what
    # the same SNR needs if the source is really 1-sigma brighter/fainter.
    if np.isfinite(top_err) and top_err > 0:
        lo = _solve(top_mag - top_err, target_magerror, conditions, min_exptime_s, max_exptime_s)
        hi = _solve(top_mag + top_err, target_magerror, conditions, min_exptime_s, max_exptime_s)
        top["exptime_range_s"] = [lo, None if not np.isfinite(hi) else hi]
    if top["at_min_exptime"]:
        top["note"] = (f"already met in {min_exptime_s:g}s; SNR is not the binding constraint "
                       f"for a source this bright (saturation is not modelled)")

    report.update(status="ok", top_candidate=top)
    range_text = ""
    if "exptime_range_s" in top:
        lo, hi = top["exptime_range_s"]
        range_text = f" (±1σ mag: {lo:.1f}-{hi:.1f}s)" if hi is not None else f" (±1σ mag: {lo:.1f}s-unreachable)"
    log.info(
        f"Follow-up exposure for {top['transient_id']} (mag {top_mag:.2f}, "
        f"{top_info['source']}): {exptime:.1f}s for SNR {target_snr:g}{range_text}, from "
        f"{conditions.source or 'reference epoch'} "
        f"({conditions.exptime_s:g}s, FWHM {conditions.fwhm_px:.2f}px)"
    )
    return report


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def write_exposure_report(obs_dir, report: Dict[str, Any],
                          logger: Optional[logging.Logger] = None) -> Optional[Path]:
    """Write the report next to the other per-observation outputs."""
    log = logger or logging.getLogger("followup.exposure")
    path = Path(obs_dir) / REPORT_FILENAME
    try:
        with open(path, "w") as handle:
            json.dump(report, handle, indent=2, default=_json_default)
        return path
    except Exception as exc:
        log.warning(f"Could not write {REPORT_FILENAME}: {exc}")
        return None


def run_enrichment(candidates: Table, lightcurves: Optional[Dict[str, Table]],
                   detection_tables: Optional[List[Table]], obs_dir, config=None,
                   logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """`recommend_exposures` + `write_exposure_report`, guaranteed not to
    raise. Enrichment decorates detection output; it must never be able to
    abort the run that produced it (in pipeline_magic.py that would happen
    before `save_results`, losing the candidates). Any unexpected failure
    becomes an `error` report on disk and a warning in the log.
    """
    log = logger or logging.getLogger("followup.exposure")
    try:
        report = recommend_exposures(candidates, lightcurves, detection_tables,
                                     config=config, logger=log)
    except Exception as exc:  # deliberately broad -- see docstring
        log.warning(f"Exposure recommendation failed unexpectedly: {exc!r}", exc_info=True)
        report = {
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "status": "error",
            "reason": f"unexpected failure: {exc!r}",
        }
    write_exposure_report(obs_dir, report, logger=log)
    return report
