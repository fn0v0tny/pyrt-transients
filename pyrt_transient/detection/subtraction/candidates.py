"""Candidate emission for subtraction-based detection (Phase A: consuming
already-differenced epochs -- see the subtraction-branch plan and
FUTURE_IDEAS.md's "New detection strategy: image subtraction").

Diff-image detections need no cross-catalog matching: subtraction already
removed constant sources, so (after the artifact filters in
artifact_filters.py) essentially any significant residual is a genuine
candidate. This module turns a raw diff-image detection table into the same
per-epoch "candidates" table shape catalog_match.py's
find_transients_multicatalog produces (quality_score, fwhm_ratio, axis_ratio,
snr_auto, saturated/blended/near_bright, candidate_type='new'), so
clustering.py's save_epoch_results/combine_with_lightcurves can cluster
across epochs and build lightcurves completely unchanged -- the genuinely
new work in Phase A is the candidate *source*, not the cross-epoch machinery.
"""

import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
from astropy.table import Table

from pyrt_transient.config_trans import DetectionConfig
from pyrt_transient.core.scoring import add_base_quality_scores

logger = logging.getLogger("detection.subtraction")

# Science-epoch meta a diff-image table needs but never carries itself: a
# difference image has no calibration/field context of its own. Verified
# against the real tests/2026kid/ fixture -- the diff `.cat` files have none
# of these, while the matching science `.ecsv` (same night/filter) has all
# of them. See the subtraction-branch plan's A1 note for the full reasoning.
_SCIENCE_META_KEYS = (
    "MAGZERO", "DMAGZERO", "RESPONSE", "MAGLIM", "MAGLIMIT",
    "CTRRA", "CTRDEC", "FIELD", "CTIME", "EXPTIME", "OBSID",
    "JD", "MJD-OBS", "OBJECT", "TARGET",
)


def load_diff_table(diff_path) -> Optional[Table]:
    """Load a diff-image detection table.

    Tries `.ecsv` first (Phase B's own extraction output), falls back to
    `.cat` (the real tests/2026kid/ fixture only has `.cat` for diff
    epochs) -- both are ascii.ecsv format on disk, just named differently
    upstream. Mirrors transients.open_ecsv_file's failure contract (returns
    None rather than raising) but doesn't force a `.ecsv` extension the way
    that function does.
    """
    diff_path = Path(diff_path)
    for suffix in (".ecsv", ".cat"):
        candidate = diff_path.with_suffix(suffix)
        if candidate.exists():
            try:
                table = Table.read(str(candidate), format="ascii.ecsv")
                table.meta["filename"] = str(candidate)
                return table
            except Exception as e:
                logger.debug(f"{candidate}: failed to read as ecsv ({e})")
    return None


def find_science_sibling(diff_path) -> Optional[Path]:
    """Science-image detection-file path for a diff-image path, using the
    real fixture's own naming convention: the diff stem ends in `h`
    immediately before the extension (`0429rh.fits` <-> `0429r.fits`)."""
    diff_path = Path(diff_path)
    stem = diff_path.stem
    if not stem.endswith("h"):
        return None
    sci_stem = stem[:-1]
    for suffix in (".ecsv", ".cat"):
        candidate = diff_path.with_name(sci_stem + suffix)
        if candidate.exists():
            return candidate
    return None


def _object_id_from_table(table: Table) -> Optional[str]:
    for key in ("OBJECT", "TARGET"):
        if key in table.meta and table.meta[key]:
            return re.sub(r"[^a-zA-Z0-9_]", "_", str(table.meta[key])).strip("_") or None
    return None


def derive_observation_id(path) -> Optional[str]:
    """Stable per-campaign observation ID for either a diff-image epoch
    (Phase A) or a raw science epoch (Phase B's diff_input_mode="raw").

    io.observation_store.extract_observation_id (used for the
    blind-multicatalog path) keys on OBSID/OBS_ID/OBSERVATION_ID/FIELD_ID --
    but real telescope OBSID changes per observing-block/night, not per
    campaign (verified against the real tests/2026kid/ fixture: OBSID is
    58636.00, 58659.00, 58688.00, 58733.00 across four different nights of
    the *same* AT2026kid campaign -- the exact same per-block-not-per-target
    fragmentation already found and fixed for the GRB replay work). Using it
    here would put every epoch of one campaign in a different
    ObservationStore directory, defeating cross-epoch clustering entirely.

    OBJECT/TARGET, by contrast, are stable across the whole campaign in the
    same fixture. Tries `path` itself first (covers Phase B's raw-mode
    input, where `path` already *is* the science file), then falls back to
    a science sibling via the `<stem>h` naming convention (covers Phase A's
    diff-file input, whose own `.cat`/`.ecsv` carries neither key -- see
    _SCIENCE_META_KEYS) -- or None if neither has OBJECT/TARGET, so the
    caller can fall back to the generic extract_observation_id.
    """
    path = Path(path)

    direct_table = load_diff_table(path)
    if direct_table is not None:
        object_id = _object_id_from_table(direct_table)
        if object_id is not None:
            return object_id

    sci_path = find_science_sibling(path)
    if sci_path is None:
        return None
    sci_table = load_diff_table(sci_path)
    if sci_table is None:
        return None
    return _object_id_from_table(sci_table)


def borrow_science_meta(diff_table: Table, sci_table: Table) -> None:
    """Copy calibration/field meta the diff table never carries itself from
    its matching science epoch (mutates diff_table.meta in place)."""
    for key in _SCIENCE_META_KEYS:
        if key not in diff_table.meta and key in sci_table.meta:
            diff_table.meta[key] = sci_table.meta[key]


def calibrate_diff_magnitudes(diff_table: Table) -> None:
    """Fill in MAG_CALIB/MAGERR_CALIB from MAG_AUTO/MAGERR_AUTO using the
    science epoch's zeropoint (already merged into diff_table.meta by
    borrow_science_meta), if not already present.

    HOTPANTS's `-n i` normalization (stdpipe.subtraction.run_hotpants, see
    Phase B) keeps the diff image in the same ADU units as the science
    image, so the science epoch's own zeropoint applies directly -- no
    separate diff-image calibration needed. Deliberately ignores the
    science RESPONSE string's color term (needs each source's own color);
    good enough to validate candidate emission/scoring/frontend against
    real data, not meant to be publication-accurate photometry.
    """
    if "MAG_CALIB" in diff_table.colnames:
        return
    if "MAG_AUTO" not in diff_table.colnames or "MAGZERO" not in diff_table.meta:
        return
    zp = float(diff_table.meta["MAGZERO"])
    diff_table["MAG_CALIB"] = diff_table["MAG_AUTO"] + zp
    diff_table["MAGERR_CALIB"] = diff_table["MAGERR_AUTO"]


def add_detection_features(candidates: Table) -> None:
    """Populate fwhm_ratio/axis_ratio/snr_auto/saturated/blended/near_bright
    from the diff-catalog's own columns.

    A subtraction-specific analogue of catalog_match.py's
    _add_detection_features: diff catalogs carry ELLIPTICITY (not
    A_IMAGE/B_IMAGE) and MAGERR_CALIB (not FLUX_AUTO/FLUXERR_AUTO), so the
    blind-multicatalog version's column assumptions don't apply here.
    FLAGS bit meanings (4=saturated, 2=blended, 8=near_bright) match
    catalog_match.py's exactly -- same upstream SExtractor convention.
    """
    if len(candidates) == 0:
        return
    if "ELLIPTICITY" in candidates.colnames:
        candidates["axis_ratio"] = 1.0 - np.asarray(candidates["ELLIPTICITY"], dtype=float)
    if "FWHM_IMAGE" in candidates.colnames:
        fwhm = np.asarray(candidates["FWHM_IMAGE"], dtype=float)
        median_fwhm = float(np.median(fwhm)) if len(fwhm) else 0.0
        candidates["fwhm_ratio"] = fwhm / median_fwhm if median_fwhm > 0 else np.ones_like(fwhm)
    if "MAGERR_CALIB" in candidates.colnames:
        # Standard mag-error <-> S/N relation used throughout this codebase
        # (see DetectionConfig.siglim's docstring): sigma_mag ~= 1.0857/SNR.
        err = np.asarray(candidates["MAGERR_CALIB"], dtype=float)
        candidates["snr_auto"] = 1.0857 / np.maximum(err, 1e-6)
    if "FLAGS" in candidates.colnames:
        flags = np.asarray(candidates["FLAGS"], dtype=int)
        candidates["saturated"] = (flags & 4) > 0
        candidates["blended"] = (flags & 2) > 0
        candidates["near_bright"] = (flags & 8) > 0


def build_epoch_candidates(
    diff_path_or_table,
    config=None,
    template_provenance: str = "unknown",
) -> Optional[Table]:
    """Build a per-epoch subtraction-candidates table from a diff-image
    detection file or an already-loaded table, in the same column shape
    catalog_match.py's per-catalog tables use, so clustering.py's
    save_epoch_results/combine_with_lightcurves can consume it completely
    unchanged (see subtraction-branch plan, A1).

    diff_path_or_table: a path to a diff-image detection file, or an
    already-loaded Table (e.g. one SubtractionStrategy.run() received
    directly as part of its `detection_tables` list, matching
    DetectionStrategy's contract) -- its `meta['filename']` is used to find
    the science sibling either way.

    template_provenance: recorded in the reference_catalog column (repurposed
    here -- see that column's assignment below for why).

    Returns None if the diff table can't be loaded, or if no MAG_CALIB can be
    established (missing science sibling or its MAGZERO) -- mirrors
    open_ecsv_file's "unusable epoch" contract of returning None rather than
    raising, so callers can skip a bad epoch the same way they already do.
    """
    if isinstance(diff_path_or_table, Table):
        diff_table = diff_path_or_table
        diff_path = diff_table.meta.get("filename")
        if diff_path is None:
            logger.warning("diff table has no meta['filename'] -- cannot find science sibling")
            return None
        diff_path = Path(diff_path)
    else:
        diff_path = Path(diff_path_or_table)
        diff_table = load_diff_table(diff_path)
        if diff_table is None:
            logger.warning(f"{diff_path}: could not load diff detection table")
            return None

    sci_path = find_science_sibling(diff_path)
    if sci_path is not None:
        sci_table = load_diff_table(sci_path)
        if sci_table is not None:
            borrow_science_meta(diff_table, sci_table)
        else:
            logger.warning(f"{diff_path}: science sibling {sci_path} found but unreadable")
    else:
        logger.warning(f"{diff_path}: no science sibling found (expected '<stem>h' naming)")

    calibrate_diff_magnitudes(diff_table)

    if "MAG_CALIB" not in diff_table.colnames or "MAGERR_CALIB" not in diff_table.colnames:
        logger.warning(
            f"{diff_path}: no MAG_CALIB available -- skipping this epoch's diff candidates"
        )
        return None

    candidates = diff_table.copy()

    # Significance gate: subtraction candidates are always "new" (nothing to
    # cross-match against), so new_source_siglim applies -- same reasoning
    # as DetectionConfig.new_source_siglim's docstring, ported here since
    # catalog.py's split significance gate is specific to the
    # blind-multicatalog path (matched vs. new detections).
    weights = config.detection if config else DetectionConfig()
    siglim = weights.new_source_siglim or weights.siglim
    mag_err = np.asarray(candidates["MAGERR_CALIB"], dtype=float)
    bad_snr = mag_err >= (1.091 / siglim)
    candidates = candidates[~bad_snr]
    if len(candidates) == 0:
        return candidates

    add_detection_features(candidates)

    candidates["candidate_type"] = "new"
    # reference_catalog is a required column elsewhere in the pipeline
    # (matched-catalog name for the blind-multicatalog path) -- repurposed
    # here to record template provenance instead (e.g. "ps1_template" or
    # "own_epoch:20260427"), since subtraction candidates were never matched
    # against a reference catalog at all.
    candidates["reference_catalog"] = template_provenance

    add_base_quality_scores(candidates, weights)

    return candidates
