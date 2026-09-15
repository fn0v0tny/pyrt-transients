"""Cross-catalog and cross-epoch clustering -- combine_results,
split_component_by_epoch, save_epoch_results, combine_with_lightcurves.

Both independent KDTree/chord-length blocks (one in combine_results, one --
actually two -- in combine_with_lightcurves) are replaced with
core/matching.match_radius. compute_per_detection_radius is NOT switched to
core/radii.py's unified compute_adaptive_radius here -- that would be an
additional deliberate behavior change beyond "replace the match mechanism",
and is a separate, not-yet-scheduled follow-up (see FUTURE_IDEAS.md).

VSX/SkyBoT filtering (see stdpipe_filters.py for the full reasoning on why
they're split): SkyBoT runs once per epoch here in combine_results, right
after that epoch's catalogs are vstacked together. VSX runs once at the very
end of _combine_with_lightcurves, on the final clustered candidates.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
import astropy.units as u
from astropy.time import Time
from sklearn.neighbors import KDTree

from pyrt_transient.config_trans import DetectionConfig
from pyrt_transient.core.union_find import UnionFind
from pyrt_transient.core.matching import match_radius
from pyrt_transient.core.scoring import add_score_probability, apply_lightcurve_score_factor
from pyrt_transient.detection.blind_multicatalog.stdpipe_filters import (
    apply_skybot_filter,
    apply_vsx_filter,
)
from pyrt_transient.detection.high_pm import apply_high_pm_veto
from pyrt_transient.detection.blind_multicatalog.lightcurve import (
    build_lightcurve_for_group,
    estimate_magnitude_error_floor,
    update_candidate_with_lightcurve_stats,
)


def compute_per_detection_radius(
    detections: Table,
    nsigma: float = 3.0,
    idlimit_min_px: float = 1.0,
    idlimit_max_px: float = 8.0,
    default_plate_scale_arcsec_per_px: float = 0.33,
    config: Optional = None
) -> np.ndarray:
    """
    Compute per-detection radius in arcseconds using PSF and SNR scaling.

    Args:
        detections: Table with detection data
        nsigma: Multiplier for PSF sigma (overridable by config.detection.adaptive_nsigma)
        idlimit_min_px: Minimum radius in pixels (overridable by config.detection.idlimit_min_px)
        idlimit_max_px: Maximum radius in pixels (overridable by config.detection.idlimit_max_px)
        default_plate_scale_arcsec_per_px: Default plate scale if WCS unavailable
        config: Optional config for parameter overrides

    Returns:
        Array of radii in arcseconds for each detection
    """
    # Override parameters from config if available
    if config:
        nsigma = config.detection.adaptive_nsigma
        idlimit_min_px = config.detection.idlimit_min_px
        idlimit_max_px = config.detection.idlimit_max_px
        default_plate_scale_arcsec_per_px = config.detection.default_plate_scale_arcsec_per_px

    n_det = len(detections)
    radii_arcsec = np.full(n_det, 2.0)  # Default 2 arcsec

    try:
        # Get PSF proxy from FWHM_IMAGE
        if 'FWHM_IMAGE' in detections.colnames:
            fwhm_px = detections['FWHM_IMAGE']
            psf_sigma_px = fwhm_px / 2.35
        else:
            # Fallback: estimate from source size
            if 'A_IMAGE' in detections.colnames:
                psf_sigma_px = detections['A_IMAGE'] / 2.0
            else:
                psf_sigma_px = np.full(n_det, 2.0)  # Default 2 pixels

        # Get SNR proxy
        snr = np.full(n_det, 5.0)  # Default SNR
        if 'SNR' in detections.colnames:
            snr = np.maximum(detections['SNR'], 3.0)
        elif 'FLUX_AUTO' in detections.colnames and 'FLUXERR_AUTO' in detections.colnames:
            flux = detections['FLUX_AUTO']
            flux_err = detections['FLUXERR_AUTO']
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                snr = np.maximum(flux / np.maximum(flux_err, 1e-10), 3.0)

        # Scale radius by SNR (lower SNR = larger radius)
        snr_scale = np.sqrt(10.0 / snr)  # Scale factor

        # Calculate radius in pixels
        radii_px = nsigma * psf_sigma_px * snr_scale
        radii_px = np.clip(radii_px, idlimit_min_px, idlimit_max_px)

        # Convert to arcseconds
        plate_scale = default_plate_scale_arcsec_per_px

        # Try to get plate scale from WCS/header if available
        if hasattr(detections, 'meta') and detections.meta:
            # Check for CD matrix elements or CDELT
            cd11 = detections.meta.get('CD1_1', 0)
            cd22 = detections.meta.get('CD2_2', 0)
            if cd11 != 0 and cd22 != 0:
                plate_scale = 3600 * np.sqrt(abs(cd11 * cd22))
            else:
                cdelt1 = detections.meta.get('CDELT1', 0)
                if cdelt1 != 0:
                    plate_scale = 3600 * abs(cdelt1)

        radii_arcsec = radii_px * plate_scale

    except Exception as e:
        logging.warning(f"Error computing per-detection radii: {e}, using defaults")
        radii_arcsec = np.full(n_det, 2.0)

    return radii_arcsec


def combine_results(
    transients: Dict[str, Table], min_catalogs_fraction: float = 1.0, min_quality: float = 0.5,
    use_sky_coords: bool = True, position_match_radius_arcsec: float = 1.0,
    config: Optional = None, time: Optional[Time] = None,
) -> Table:
    """Combine and filter transient candidates using match_radius + union-find clustering.

    Args:
        transients: Dictionary of transient tables from different catalogs
        min_catalogs_fraction: Fraction of catalogs that must flag the source as a
            transient (0.0–1.0).  Converted to an integer threshold via ceiling so
            e.g. 1.0 with 3 catalogs → 3, 0.67 with 3 catalogs → 2.
        min_quality: Minimum quality score to include
        use_sky_coords: If True, use sky coordinates; if False, use pixel coordinates
        position_match_radius_arcsec: Matching radius in arcseconds (for sky coords)
        config: Optional configuration object
        time: Observation time for this epoch (for SkyBoT cross-matching); None skips it

    Returns:
        Combined table of reliable transient candidates. `meta['DEGRADED']`
        is set when an external filter could not run (SkyBoT outage), so the
        caller can decline to cache this epoch as a finished result.
    """
    degraded = None

    def _out(table: Table) -> Table:
        if degraded:
            table.meta['DEGRADED'] = degraded
        return table

    if not transients:
        return Table()

    # Stack all candidates
    all_candidates = vstack(list(transients.values()))

    # Filter by quality
    quality_mask = all_candidates["quality_score"] >= min_quality
    all_candidates = all_candidates[quality_mask]

    if len(all_candidates) == 0:
        return Table()

    # SkyBoT: once per epoch, right after this epoch's catalogs are combined
    # (time-dependent, so it can't defer to the final-candidate stage like
    # VSX does -- see stdpipe_filters.py).
    # `time is None` means this epoch's header carries no observation time at
    # all -- a permanent property of the data, not a transient failure, so it
    # is not marked degraded (that would make the epoch recompute forever).
    if config and config.detection.vsx_filter_enabled and time is not None:
        all_candidates, n_removed, skybot_ok = apply_skybot_filter(
            all_candidates, time=time,
            match_radius_arcsec=config.detection.vsx_match_radius_arcsec,
        )
        if n_removed > 0:
            logging.info(f"SkyBoT filter removed {n_removed} candidates for this epoch")
        if not skybot_ok:
            degraded = ("SkyBoT cross-match could not run for this epoch "
                        "(service unavailable, or the epoch carries no observation time)")
            logging.warning(f"Epoch marked degraded: {degraded}")
        if len(all_candidates) == 0:
            return _out(Table())

    n_candidates = len(all_candidates)

    if use_sky_coords and 'ALPHA_J2000' in all_candidates.colnames and 'DELTA_J2000' in all_candidates.colnames:
        # Use sky coordinates with per-detection radii
        try:
            # Compute per-detection radii with configurable parameters
            nsigma = 3.0
            idlimit_min_px = 1.0
            idlimit_max_px = 8.0

            if config:
                nsigma = config.detection.adaptive_nsigma
                idlimit_min_px = config.detection.idlimit_min_px
                idlimit_max_px = config.detection.idlimit_max_px

            per_det_radii = compute_per_detection_radius(
                all_candidates,
                nsigma=nsigma,
                idlimit_min_px=idlimit_min_px,
                idlimit_max_px=idlimit_max_px,
                config=config
            )

            # Build adjacency graph using per-detection radii
            uf = UnionFind(n_candidates)

            # Use maximum possible radius for initial query, then filter
            max_radius_arcsec = max(position_match_radius_arcsec, np.max(per_det_radii))
            radec = np.column_stack((
                np.array(all_candidates["ALPHA_J2000"], dtype=float),
                np.array(all_candidates["DELTA_J2000"], dtype=float),
            ))
            idx_a, idx_b, dist_deg = match_radius(
                radec, radec, max_radius_arcsec, coord_system="sky"
            )
            dist_arcsec = dist_deg * 3600.0
            pair_mask = idx_a < idx_b  # avoid duplicate/self edges

            for i, j, angular_sep_arcsec in zip(
                idx_a[pair_mask], idx_b[pair_mask], dist_arcsec[pair_mask]
            ):
                # Check if within min of both detection radii
                max_allowed_radius = min(per_det_radii[i], per_det_radii[j])

                if angular_sep_arcsec <= max_allowed_radius:
                    uf.union(int(i), int(j))

        except Exception as e:
            logging.warning(f"Sky coordinate clustering failed: {e}, falling back to pixel coordinates")
            use_sky_coords = False

    if not use_sky_coords:
        # Fallback to pixel coordinates
        coords = np.column_stack((all_candidates["X_IMAGE"], all_candidates["Y_IMAGE"]))
        tree = KDTree(coords)

        # Build adjacency graph with fixed pixel radius
        uf = UnionFind(n_candidates)
        pixel_radius = 2.0  # pixels

        neighbors_list = tree.query_radius(coords, r=pixel_radius)

        for i, neighbors in enumerate(neighbors_list):
            for j in neighbors:
                if i < j:  # Avoid duplicate edges
                    uf.union(i, j)

    # Get connected components
    components = uf.get_components()

    # Derive integer threshold from fraction and total catalog count
    n_catalogs = max(1, len(transients))
    min_catalogs = max(1, math.ceil(min_catalogs_fraction * n_catalogs))

    # Filter candidates appearing in enough catalogs and process components.
    #
    # Vectorized: the original version built one brand-new single-row
    # Table() per surviving subcluster, setting every column individually
    # (Table.__setitem__/add_column has real per-call overhead -- profiled
    # at ~14,000 calls / 6.5s for a single 436-candidate subtraction epoch,
    # the dominant cost in this whole function). Collecting indices/override
    # values in plain Python lists and building the result with one fancy-index
    # select + at most two whole-column overwrites is behaviorally identical
    # (same selection and override logic below) but avoids that per-row
    # Table-construction cost entirely.
    has_candidate_type = 'candidate_type' in all_candidates.colnames
    has_magdiff = 'magnitude_difference' in all_candidates.colnames

    best_indices: List[int] = []
    type_overrides: List[str] = []
    magdiff_overrides: List[float] = []

    for root, component_indices in components.items():
        # Count unique catalogs in this component
        component_data = all_candidates[component_indices]
        cat_count = len(set(component_data["reference_catalog"]))

        if cat_count >= min_catalogs:
            # Split component by epoch constraint if epoch information available
            if 'epoch_id' in all_candidates.colnames:
                subclusters = split_component_by_epoch(
                    component_indices,
                    all_candidates,
                    position_match_radius_arcsec
                )
            else:
                # No epoch info, treat as single cluster
                subclusters = [component_indices]

            # Process each subcluster
            for subcluster_indices in subclusters:
                subcluster_data = all_candidates[subcluster_indices]
                subcluster_cat_count = len(set(subcluster_data["reference_catalog"]))

                if subcluster_cat_count >= min_catalogs:
                    # Take the one with highest quality score from the subcluster
                    subcluster_qualities = subcluster_data["quality_score"]
                    best_local_idx = np.argmax(subcluster_qualities)
                    best_global_idx = subcluster_indices[best_local_idx]
                    best_indices.append(best_global_idx)

                    # Use most informative candidate_type from the entire subcluster,
                    # not just the highest-quality-score detection. This prevents a USNO
                    # detection (always 'new' due to no Sloan photometry) from overriding
                    # a 'brightening'/'fading' classification from Gaia or ATLAS.
                    if has_candidate_type:
                        type_priority = {'brightening': 3, 'fading': 3, 'trail': 2, 'new': 1, 'unknown': 0}
                        all_types = [str(t) for t in subcluster_data['candidate_type']]
                        best_type = max(all_types, key=lambda t: type_priority.get(t, 0))
                        type_overrides.append(best_type)
                        # Copy magnitude_difference from the detection that provided
                        # best_type so it is consistent with the classification.
                        if has_magdiff:
                            type_mask = np.array([str(t) == best_type for t in subcluster_data['candidate_type']])
                            type_rows = subcluster_data[type_mask]
                            best_md_idx = np.argmax(np.abs(type_rows['magnitude_difference']))
                            magdiff_overrides.append(type_rows['magnitude_difference'][best_md_idx])

    if not best_indices:
        return _out(Table())

    result = all_candidates[best_indices]
    if has_candidate_type:
        result['candidate_type'] = type_overrides
        if has_magdiff:
            result['magnitude_difference'] = magdiff_overrides
    return _out(result)


def split_component_by_epoch(
    component_indices: List[int],
    detections: Table,
    base_radius_arcsec: float = 2.0
) -> List[List[int]]:
    """
    Split a connected component to enforce one-per-epoch constraint.
    Uses greedy splitting with quality-based ordering.

    Args:
        component_indices: Indices of detections in this component
        detections: Full detection table
        base_radius_arcsec: Base radius for subcluster validation

    Returns:
        List of subclusters (each is a list of indices)
    """
    if len(component_indices) <= 1:
        return [component_indices]

    # Nothing to enforce when every member is from a different epoch: the
    # component already satisfies the constraint, and the greedy split below
    # would only re-validate distances against a centroid grown from the
    # highest-scoring row -- which, when that row is a junk blob 2" off the
    # source, strands the real rows in a second subcluster below
    # min_n_detections (GRB 220403B at epoch 42, 2026-09-04).
    if 'epoch_id' in detections.colnames:
        epochs = [detections['epoch_id'][i] for i in component_indices]
        if len(set(epochs)) == len(epochs):
            return [list(component_indices)]

    # Get component data
    component_data = detections[component_indices]

    # Sort by quality score descending (best first)
    if 'quality_score' in component_data.colnames:
        sorted_order = np.argsort(component_data['quality_score'])[::-1]
    else:
        # Fallback to magnitude (brighter = better)
        if 'MAG_AUTO' in component_data.colnames:
            sorted_order = np.argsort(component_data['MAG_AUTO'])
        else:
            sorted_order = np.arange(len(component_indices))

    sorted_indices = [component_indices[i] for i in sorted_order]

    # Initialize subclusters
    subclusters = []

    for i, idx in enumerate(sorted_indices):
        # Get epoch_id safely from table
        epoch_id = detections['epoch_id'][idx] if 'epoch_id' in detections.colnames else 0

        # Find first compatible subcluster
        placed = False
        for subcluster in subclusters:
            # Check epoch constraint
            subcluster_data = detections[subcluster]
            if 'epoch_id' in subcluster_data.colnames:
                subcluster_epochs = set(subcluster_data['epoch_id'])
                if epoch_id in subcluster_epochs:
                    continue  # Epoch conflict

            # Check distance constraint to subcluster centroid
            if len(subcluster) > 0:
                subcluster_positions = detections[subcluster]
                centroid_ra = np.mean(subcluster_positions['ALPHA_J2000'])
                centroid_dec = np.mean(subcluster_positions['DELTA_J2000'])

                # Calculate angular separation using dot product (avoids SkyCoord overhead in tight loop)
                d_ra = np.radians(float(detections['ALPHA_J2000'][idx]))
                d_dec = np.radians(float(detections['DELTA_J2000'][idx]))
                c_ra = np.radians(centroid_ra)
                c_dec = np.radians(centroid_dec)
                dot = np.clip(
                    np.cos(d_dec)*np.cos(d_ra)*np.cos(c_dec)*np.cos(c_ra) +
                    np.cos(d_dec)*np.sin(d_ra)*np.cos(c_dec)*np.sin(c_ra) +
                    np.sin(d_dec)*np.sin(c_dec),
                    -1.0, 1.0
                )
                separation = np.degrees(np.arccos(dot)) * 3600

                if separation <= base_radius_arcsec:
                    subcluster.append(idx)
                    placed = True
                    break

        # If not placed, start new subcluster
        if not placed:
            subclusters.append([idx])

    return subclusters


def save_epoch_results(
    transients: Dict[str, Table],
    det_table: Table,
    epoch_index: int,
    min_catalogs: float,
    min_quality: float,
    data_dir,
    config: Optional = None,
    logger=None,
    degraded_reason: Optional[str] = None,
) -> None:
    """Save results for this epoch using combine_results.

    `degraded_reason` records that this epoch was produced with less than
    its full input (a reference catalogue that failed to download, say). It
    is stamped into the cached table's `meta['DEGRADED']`, which `epoch_is
    _cached` treats as "recompute next run": the result is still written, so
    this run's lightcurves are complete, but a transient outage does not
    become a permanent property of the observation.
    """
    from pyrt_transient.io.naming import get_base_filename

    base_filename = get_base_filename(det_table, epoch_index)
    ecsv_table = f"{base_filename}_transients.ecsv"
    ecsv_path = data_dir / ecsv_table

    if not ecsv_path.exists():
        obs_time = detections_time(det_table)
        reliable = combine_results(
            transients,
            min_catalogs_fraction=min_catalogs,
            min_quality=min_quality,
            use_sky_coords=True,
            position_match_radius_arcsec=2.0,
            config=config,
            time=obs_time,
        )
        reasons = [r for r in (degraded_reason, reliable.meta.get('DEGRADED')) if r]
        if reasons:
            reliable.meta['DEGRADED'] = "; ".join(reasons)
        reliable.write(str(ecsv_path), overwrite=True)
        if logger:
            logger.debug(f"Saved {len(reliable)} candidates to {ecsv_table}")
            if reasons:
                logger.warning(f"{ecsv_table} written but marked degraded ({reliable.meta['DEGRADED']}); "
                               f"it will be recomputed on the next run")


def epoch_is_cached(ecsv_path) -> bool:
    """Whether a per-epoch `<base>_transients.ecsv` counts as a finished
    result. A file stamped `DEGRADED` by save_epoch_results does not: it was
    produced while a catalogue or SkyBoT was unreachable, and re-running must
    get another chance at the full input rather than inheriting the outage
    forever. Only the ECSV's YAML header is read, not the table."""
    from pathlib import Path
    path = Path(ecsv_path)
    if not path.exists():
        return False
    try:
        with open(path) as fh:
            for line in fh:
                if not line.startswith("#"):
                    break
                if "DEGRADED" in line:
                    return False
    except OSError:
        return False
    return True


def detections_time(detections) -> Optional[Time]:
    """Observation time (astropy Time) for SkyBoT cross-matching, derived the
    same way as core/epochs.py's mid-exposure time (CTIME + EXPTIME/2), or
    from JD / MJD-OBS for tables that carry those instead (the subtraction
    pipeline's difference epochs do).

    None when the epoch carries no time at all. It used to default a missing
    CTIME to 0, i.e. query SkyBoT at 1970-01-01 -- a valid-looking Time that
    silently rejected nothing while the caller believed the epoch had been
    cross-matched.
    """
    meta = getattr(detections, "meta", None) or {}
    try:
        if meta.get('CTIME') is not None:
            exptime = float(meta.get('EXPTIME', 0.0) or 0.0)
            return Time(float(meta['CTIME']) + exptime / 2.0, format='unix')
        if meta.get('JD') is not None:
            return Time(float(meta['JD']), format='jd')
        if meta.get('MJD-OBS') is not None:
            return Time(float(meta['MJD-OBS']), format='mjd')
    except Exception:
        return None
    return None


def combine_with_lightcurves(
    data_dir,
    detection_tables: List[Table],
    all_epoch_detections: List[Table],
    position_match_radius: float = 2.0,
    min_n_detections: int = 3,
    config: Optional = None,
) -> Tuple[Table, Dict]:
    """Combine candidates and build lightcurves (same logic as before)."""
    from pyrt_transient.io.naming import get_base_filename

    logging.info(f"Starting lightcurve combination with {len(detection_tables)} epochs...")

    # Load all transient candidates
    all_candidates = []
    candidate_sources = []

    for i, det_table in enumerate(detection_tables):
        base_filename = get_base_filename(det_table, i)
        file_path = data_dir / f"{base_filename}_transients.ecsv"

        if file_path.exists():
            try:
                candidates = Table.read(str(file_path))
                if len(candidates) > 0:
                    candidates['epoch_id'] = i
                    candidates['source_file'] = base_filename
                    all_candidates.append(candidates)
                    candidate_sources.extend([i] * len(candidates))
            except Exception as e:
                logging.info(f"Error reading {file_path}: {e}")

    if not all_candidates:
        return Table(), {}

    # Stack all candidates
    stacked_candidates = vstack(all_candidates, metadata_conflicts="silent")

    # Remove problematic columns
    if "quality_flag" in stacked_candidates.colnames:
        stacked_candidates.remove_column("quality_flag")

    # Filter out invalid coordinates early
    ra_values = stacked_candidates["ALPHA_J2000"]
    dec_values = stacked_candidates["DELTA_J2000"]

    # Check for NaN, infinite, or unrealistic coordinate values
    valid_coords = (
        np.isfinite(ra_values) &
        np.isfinite(dec_values) &
        (ra_values >= 0) & (ra_values <= 360) &
        (dec_values >= -90) & (dec_values <= 90)
    )

    if not np.all(valid_coords):
        n_invalid = np.sum(~valid_coords)
        logging.info(f"Filtering out {n_invalid} candidates with invalid coordinates")
        stacked_candidates = stacked_candidates[valid_coords]
        candidate_sources = [candidate_sources[i] for i in range(len(candidate_sources)) if valid_coords[i]]

    if len(stacked_candidates) == 0:
        logging.info("No candidates with valid coordinates remaining")
        return Table(), {}

    # Fast match_radius + union-find clustering with per-detection radii
    n_candidates = len(stacked_candidates)

    # Compute per-detection radii with configurable parameters
    nsigma = 3.0
    idlimit_min_px = 1.0
    idlimit_max_px = 8.0

    if config:
        nsigma = config.detection.adaptive_nsigma
        idlimit_min_px = config.detection.idlimit_min_px
        idlimit_max_px = config.detection.idlimit_max_px

    per_det_radii = compute_per_detection_radius(
        stacked_candidates,
        nsigma=nsigma,
        idlimit_min_px=idlimit_min_px,
        idlimit_max_px=idlimit_max_px,
        config=config
    )

    # Two-pass approach: tight radius first, then optional merging
    # Pass 1: Build components with tight radius (0.6x base radius)
    tight_radius_arcsec = 0.6 * position_match_radius

    logging.debug(f"Pass 1: Tight clustering with radius {tight_radius_arcsec:.2f} arcsec")

    # Build initial adjacency graph using per-detection radii
    uf = UnionFind(n_candidates)

    # Use maximum possible radius for initial query, then filter
    max_radius_arcsec = max(position_match_radius, np.max(per_det_radii))
    radec = np.column_stack((
        np.array(stacked_candidates["ALPHA_J2000"], dtype=float),
        np.array(stacked_candidates["DELTA_J2000"], dtype=float),
    ))
    idx_a, idx_b, dist_deg = match_radius(radec, radec, max_radius_arcsec, coord_system="sky")
    dist_arcsec = dist_deg * 3600.0
    pair_mask = idx_a < idx_b

    for i, j, angular_sep_arcsec in zip(idx_a[pair_mask], idx_b[pair_mask], dist_arcsec[pair_mask]):
        # Check if within min of both detection radii AND tight radius
        max_allowed_radius = min(
            per_det_radii[i],
            per_det_radii[j],
            tight_radius_arcsec
        )

        if angular_sep_arcsec <= max_allowed_radius:
            uf.union(int(i), int(j))

    # Get initial connected components
    components = uf.get_components()
    logging.debug(f"Pass 1: Found {len(components)} initial components")

    # Pass 2: merge components whose *members* come within the base radius
    # (single linkage), subject to the one-detection-per-epoch constraint
    # evaluated against the components as they are at that moment.
    #
    # This used to compare component CENTROIDS at the base radius. A single
    # outlying member -- e.g. a 14-px blob from a bad frame with zero
    # centroid errors, hence a wide adaptive radius -- linked in pass 1 to
    # one epoch's row drags that pair's centroid ~2" off the source, the
    # centroid test then fails, and a source with 30 real detections ends
    # as two 2-member components, both below min_n_detections (GRB 220403B
    # under the vetting catalogues, lost at epoch 42 of 57; 2026-09-04).
    # Pair separations from the pass-1 query already cover the base radius
    # (max_radius_arcsec >= position_match_radius), so no new matching.
    comp_epochs = {root: set(candidate_sources[i] for i in idx)
                   for root, idx in components.items()}
    order = np.argsort(dist_arcsec)  # closest pairs first
    for p in order:
        i, j = int(idx_a[p]), int(idx_b[p])
        if j <= i or dist_arcsec[p] > position_match_radius:
            continue
        ri, rj = uf.find(i), uf.find(j)
        if ri == rj:
            continue
        if comp_epochs[ri] & comp_epochs[rj]:
            continue  # two detections in one epoch cannot be one source
        uf.union(i, j)
        merged = comp_epochs[ri] | comp_epochs[rj]
        comp_epochs[ri] = merged
        comp_epochs[rj] = merged
        comp_epochs[uf.find(i)] = merged

    # Pass 2b: the original centroid test, on centroids weighted by the
    # inverse square of each member's positional radius, so a member with
    # a wide radius (the blob case above) barely moves the centroid.
    components = uf.get_components()
    comp_epochs = {root: set(candidate_sources[i] for i in idx) for root, idx in components.items()}
    roots = list(components.keys())
    if len(roots) > 1:
        cents = []
        for root in roots:
            idx = np.asarray(components[root])
            w = 1.0 / np.maximum(per_det_radii[idx], 1e-3) ** 2
            cents.append((float(np.sum(radec[idx, 0] * w) / np.sum(w)),
                          float(np.sum(radec[idx, 1] * w) / np.sum(w))))
        cents = np.asarray(cents)
        c_a, c_b, _ = match_radius(cents, cents, position_match_radius, coord_system="sky")
        for i, j in zip(c_a, c_b):
            if j <= i:
                continue
            ri, rj = uf.find(roots[i]), uf.find(roots[j])
            if ri == rj or (comp_epochs[ri] & comp_epochs[rj]):
                continue
            uf.union(roots[i], roots[j])
            merged = comp_epochs[ri] | comp_epochs[rj]
            comp_epochs[ri] = comp_epochs[rj] = comp_epochs[uf.find(roots[i])] = merged

    # Get final components after merging
    final_components = uf.get_components()
    logging.info(f"Fast clustering: {len(final_components)} components from {n_candidates} candidates")

    # Pre-build KDTrees for all epoch detections once.
    # build_lightcurve_for_group is called per group, so doing this here avoids
    # re-creating SkyCoord arrays over thousands of detections for every group.
    epoch_kdtrees = []
    for _edet in all_epoch_detections:
        if len(_edet) == 0:
            epoch_kdtrees.append(None)
            continue
        try:
            _ra_v = np.array(_edet['ALPHA_J2000'], dtype=float)
            _dec_v = np.array(_edet['DELTA_J2000'], dtype=float)
            _valid = np.isfinite(_ra_v) & np.isfinite(_dec_v)
            if not np.any(_valid):
                epoch_kdtrees.append(None)
                continue
            _ra_r = np.radians(_ra_v[_valid])
            _dec_r = np.radians(_dec_v[_valid])
            _x = np.cos(_dec_r) * np.cos(_ra_r)
            _y = np.cos(_dec_r) * np.sin(_ra_r)
            _z = np.sin(_dec_r)
            _xyz = np.column_stack((_x, _y, _z))
            epoch_kdtrees.append((KDTree(_xyz), _xyz, _edet[_valid]))
        except Exception:
            epoch_kdtrees.append(None)
    logging.debug(f"Pre-built {len(epoch_kdtrees)} epoch KDTrees for lightcurve lookup")

    # Campaign-level magnitude-error floor for the variability statistics
    # (lightcurve.estimate_magnitude_error_floor): measured from the bright
    # constant stars of these very epochs when enabled, else the configured
    # value; 0 disables it.
    magerr_floor = 0.0
    if config is not None:
        magerr_floor = float(getattr(config.detection, "magerr_floor_mag", 0.0) or 0.0)
        if getattr(config.detection, "magerr_floor_auto", False):
            est, n_stars = estimate_magnitude_error_floor(all_epoch_detections, epoch_kdtrees)
            if np.isfinite(est):
                logging.info(f"Magnitude-error floor {est:.3f} mag from {n_stars} bright constant stars")
                magerr_floor = est
            else:
                logging.info(f"Magnitude-error floor: too few constant stars ({n_stars}); "
                             f"using configured {magerr_floor:.3f} mag")

    # Process each component with epoch constraint enforcement
    final_candidates = []
    lightcurve_store = []  # parallel list to final_candidates; indexed via _lc_idx column

    for root, component_indices in final_components.items():
        if len(component_indices) < min_n_detections:
            continue

        # Split component by epoch constraint
        subclusters = split_component_by_epoch(
            component_indices,
            stacked_candidates,
            position_match_radius
        )

        # Process each subcluster
        for subcluster_indices in subclusters:
            if len(subcluster_indices) < min_n_detections:
                continue

            # Build lightcurve for this subcluster
            group_candidates = stacked_candidates[subcluster_indices]
            group_epochs = [candidate_sources[idx] for idx in subcluster_indices]

            try:
                lightcurve_data = build_lightcurve_for_group(
                    group_candidates, group_epochs, all_epoch_detections, position_match_radius,
                    epoch_kdtrees=epoch_kdtrees
                )

                if len(lightcurve_data) >= min_n_detections:
                    # Create final candidate entry
                    best_local_idx = np.argmax(group_candidates["quality_score"])
                    best_global_idx = subcluster_indices[best_local_idx]
                    best_row = stacked_candidates[best_global_idx]

                    # Convert to single-row table
                    best_candidate = Table()
                    for col_name in best_row.colnames:
                        best_candidate[col_name] = [best_row[col_name]]

                    # Use most informative candidate_type from all detections in this
                    # multi-epoch cluster before update_candidate_with_lightcurve_stats
                    # can override it (trail detection still takes precedence via that method).
                    if 'candidate_type' in group_candidates.colnames:
                        type_priority = {'brightening': 3, 'fading': 3, 'trail': 2, 'new': 1, 'unknown': 0}
                        all_types = [str(t) for t in group_candidates['candidate_type']]
                        best_type = max(all_types, key=lambda t: type_priority.get(t, 0))
                        best_candidate['candidate_type'] = [best_type]
                        # Copy magnitude_difference from the detection that provided
                        # best_type so it is consistent with the classification.
                        if 'magnitude_difference' in group_candidates.colnames:
                            type_mask = np.array([str(t) == best_type for t in group_candidates['candidate_type']])
                            type_rows = group_candidates[type_mask]
                            if len(type_rows) > 0:
                                best_md_idx = np.argmax(np.abs(type_rows['magnitude_difference']))
                                best_candidate['magnitude_difference'] = [type_rows['magnitude_difference'][best_md_idx]]

                    # Update with lightcurve statistics
                    update_candidate_with_lightcurve_stats(
                        best_candidate, lightcurve_data, config=config,
                        magerr_floor=magerr_floor,
                    )

                    # Fold in the lightcurve-stage quality_score factor (stages
                    # 2+3: brightness/variability boost * mag_range) -- replaces
                    # both the deleted `quality_score *=` line inside
                    # update_candidate_with_lightcurve_stats and the old
                    # `result_table["quality_score"] *= result_table["mag_range"]`
                    # that used to run after vstack. Multiplies into the
                    # already-computed base_score rather than recomputing it,
                    # since this fresh single-row Table doesn't carry the
                    # original per-epoch table's MAGLIM meta.
                    weights = config.detection if config else DetectionConfig()
                    apply_lightcurve_score_factor(best_candidate, weights)

                    # Generate unique ID
                    transient_id = f"transient_{best_candidate['ALPHA_J2000'][0]:.3f}_{best_candidate['DELTA_J2000'][0]:.3f}"
                    best_candidate['transient_id'] = transient_id

                    # Track which lightcurve belongs to this candidate row.
                    # Storing directly into a dict keyed by transient_id would be
                    # overwritten if two subclusters share the same quantized ID;
                    # instead, use an index column resolved after sort+dedup.
                    best_candidate['_lc_idx'] = [len(lightcurve_store)]
                    final_candidates.append(best_candidate)
                    lightcurve_store.append(lightcurve_data)

            except Exception as e:
                logging.warning(f"Error processing subcluster: {e}")
                continue

    # Convert to table
    if final_candidates:
        result_table = vstack(final_candidates)
        result_table.sort("quality_score", reverse=True)

        # Deduplicate by transient_id: two clusters at the same quantized position
        # (within 3 decimal places of RA/Dec) would produce the same transient_id,
        # causing candidate_id collisions in the frontend that mismatch cutouts.
        # After sort (highest quality_score first), first-seen wins.
        seen_ids: set = set()
        keep = []
        for i, row_id in enumerate(result_table['transient_id']):
            row_id_str = str(row_id)
            if row_id_str not in seen_ids:
                seen_ids.add(row_id_str)
                keep.append(i)
        if len(keep) < len(result_table):
            n_dupes = len(result_table) - len(keep)
            logging.warning(f"Removed {n_dupes} duplicate transient_id(s) from result table")
            result_table = result_table[keep]

        # Build lightcurves dict using _lc_idx so each kept row gets the
        # lightcurve from its own subcluster, not from a duplicate's subcluster
        # that may have overwritten it if we had used a dict during the loop.
        lightcurves = {
            str(result_table['transient_id'][i]): lightcurve_store[int(result_table['_lc_idx'][i])]
            for i in range(len(result_table))
        }
        result_table.remove_column('_lc_idx')

        # VSX: once, on the final clustered candidates (purely positional --
        # see stdpipe_filters.py for why this runs here rather than per-epoch).
        if config and config.detection.vsx_filter_enabled:
            try:
                result_table, n_removed = apply_vsx_filter(
                    result_table, match_radius_arcsec=config.detection.vsx_match_radius_arcsec
                )
            except Exception as e:
                # VizieR is a network dependency at the very last step of a
                # run; an outage here must not discard everything computed
                # above (the caller exits non-zero before save_results).
                logging.warning(f"VSX filter failed ({e!r}); keeping all "
                                f"{len(result_table)} candidates unfiltered")
                n_removed = 0
            if n_removed > 0:
                logging.info(f"VSX filter removed {n_removed} final candidates")
                surviving_ids = set(str(t) for t in result_table['transient_id'])
                lightcurves = {k: v for k, v in lightcurves.items() if k in surviving_ids}

        # High proper-motion stars: once, on the final candidates, against
        # Gaia DR3 propagated to this run's latest epoch (see high_pm.py for
        # why the matcher's own propagation is not enough on its own).
        if config and getattr(config.detection, "high_pm_veto_enabled", False):
            result_table, n_removed = apply_high_pm_veto(result_table, detection_tables, data_dir, config)
            if n_removed > 0:
                logging.info(f"High-pm veto removed {n_removed} final candidates")
                surviving_ids = set(str(t) for t in result_table['transient_id'])
                lightcurves = {k: v for k, v in lightcurves.items() if k in surviving_ids}

        if config:
            add_score_probability(result_table, config.detection)

        # Final quality gate, on the fully-computed score (base_score *
        # lightcurve_score_factor, including the n_detections consistency
        # term) -- not the early, base-score-only min_quality check in
        # combine_results. That earlier check runs before any lightcurve
        # exists and can't know how consistently a source repeats; this one
        # can. Relaxing admission (new_source_siglim, min_n_detections) only
        # works safely alongside this: it lets marginal-but-real candidates
        # reach full scoring, then filters on the real, accumulated
        # confidence rather than a hard per-epoch or per-count cutoff.
        if config:
            min_quality_final = config.detection.min_quality
            # A masked/NaN score never passes (and must not crash the int()
            # below -- seen on GRB 210312B with the vetting catalogues).
            scores = np.ma.filled(np.ma.asarray(result_table['quality_score'], dtype=float), np.nan)
            quality_keep = np.isfinite(scores) & (scores >= min_quality_final)
            n_dropped = int(np.sum(~quality_keep))
            if n_dropped > 0:
                logging.info(
                    f"Final quality gate removed {n_dropped} candidate(s) below "
                    f"min_quality={min_quality_final}"
                )
                result_table = result_table[quality_keep]
                surviving_ids = set(str(t) for t in result_table['transient_id'])
                lightcurves = {k: v for k, v in lightcurves.items() if k in surviving_ids}

        # Trail detection summary logging
        if 'candidate_type' in result_table.colnames:
            trail_mask = result_table['candidate_type'] == 'trail'
            n_trails = np.sum(trail_mask)
            n_total = len(result_table)

            if n_trails > 0:
                logging.info(f"Trail detection summary: {n_trails} trails out of {n_total} candidates ({100*n_trails/n_total:.1f}%)")

                # Log trail motion statistics
                if 'motion_rate_as_per_hr' in result_table.colnames and 'motion_significance' in result_table.colnames:
                    trail_rates = result_table['motion_rate_as_per_hr'][trail_mask]
                    trail_sigs = result_table['motion_significance'][trail_mask]

                    logging.info(f"Trail motion rates: {np.min(trail_rates):.2f}-{np.max(trail_rates):.2f}\"/hr "
                               f"(median: {np.median(trail_rates):.2f}\"/hr)")
                    logging.info(f"Trail motion significance: {np.min(trail_sigs):.1f}-{np.max(trail_sigs):.1f} "
                               f"(median: {np.median(trail_sigs):.1f})")
            else:
                logging.info(f"Trail detection summary: no trails detected among {n_total} candidates")
    else:
        result_table = Table()
        lightcurves = {}

    return result_table, lightcurves
