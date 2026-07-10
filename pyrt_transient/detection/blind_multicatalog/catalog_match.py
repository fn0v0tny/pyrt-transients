"""Per-catalog transient candidate detection --
find_transients_multicatalog (dropping the unused `gen_images` parameter,
confirmed dead: the only call site never passes it) and
_add_catalog_context_safe.

VSX/SkyBoT filtering: replaced two dead hand-rolled filter blocks (wrong
import, always silently caught by try/except and never actually run) with
stdpipe.pipeline.filter_transient_candidates(vizier=['vsx'], skybot=True,
ned=False) -- validated against a fixture (one epoch, gaia catalog: 159 raw
candidates -> 151 survive, all 8 removed were real VSX matches, 0 SkyBoT, no
accidental removal near the known GRB afterglow position). NED cross-match
is deliberately NOT used for removal (a galaxy-coincident candidate can be a
real transient, e.g. a supernova in its host) -- that's an
enrichment/annotation concern, not a removal filter.

Where the filtering actually runs is NOT here, though -- see
stdpipe_filters.py and clustering.py. Running it per-catalog-per-epoch (the
first cut) meant checking up to ~1600 raw candidate positions x 2 catalogs
x 9 epochs against VSX, when only the ~5-6 *final* clustered candidates
matter for a purely positional catalog like VSX (a variable star's position
never changes). VSX now runs once, on final clustered candidates
(clustering.py). SkyBoT is time-dependent and can't defer to final
candidates the same way, but doesn't need per-catalog redundancy either --
it now runs once per epoch, on that epoch's catalogs already vstacked
together (clustering.py's combine_results).

_add_detection_features (pure, no self/config dependency) moves here too,
since it's only ever called from this function.

_add_quality_metrics's body is now core/scoring.add_base_quality_scores --
computing base_score only (stage 1), since no lightcurve features exist yet
at this point. combine_results' min_quality filtering needs *some*
quality_score before any lightcurve exists; the final quality_score
(base_score * lightcurve_boost * mag_range_factor) is completed later in
clustering.py's combine_with_lightcurves via apply_lightcurve_score_factor,
which multiplies in the remaining stages rather than recomputing from
scratch (the final candidate row lives in a fresh Table() that doesn't
carry this table's MAGLIM meta -- see core/scoring.py's module docstring
for why recomputing stage 1 there would silently use the wrong MAGLIM
context). The old `quality_flag` (HIGH/MEDIUM/LOW) column
_add_quality_metrics also set is not reproduced -- confirmed dead: never
present in final candidates.tbl, always stripped before the final table is
written.
"""

import numpy as np
from astropy.table import Table

from pyrt_transient.config_trans import DetectionConfig
from pyrt_transient.core.scoring import add_base_quality_scores


def _add_detection_features(candidates):
    """Add detection features (pure -- no config/self dependency)."""
    if "A_IMAGE" in candidates.columns and "B_IMAGE" in candidates.columns:
        candidates["axis_ratio"] = candidates["B_IMAGE"] / candidates["A_IMAGE"]

    if "FWHM_IMAGE" in candidates.columns:
        median_fwhm = np.median(candidates["FWHM_IMAGE"])
        if median_fwhm > 0:
            candidates["fwhm_ratio"] = candidates["FWHM_IMAGE"] / median_fwhm
        else:
            candidates["fwhm_ratio"] = [1.0] * len(candidates)

    if "FLUX_AUTO" in candidates.columns and "FLUXERR_AUTO" in candidates.columns:
        candidates["snr_auto"] = candidates["FLUX_AUTO"] / np.maximum(candidates["FLUXERR_AUTO"], 1e-10)

    if "FLAGS" in candidates.columns:
        candidates["saturated"] = (candidates["FLAGS"] & 4) > 0
        candidates["blended"] = (candidates["FLAGS"] & 2) > 0
        candidates["near_bright"] = (candidates["FLAGS"] & 8) > 0


def find_transients_multicatalog(
    catalog_loader,
    config,
    logger,
    detections,
    catalogs,
    params=None,
    idlimit=5.0,
    radius_check=30.0,
    filter_pattern=None,
    mag_change_threshold=1.0,
):
    """Enhanced version with better error handling."""
    results = {}

    for cat_name in catalogs:
        try:
            logger.info(f"Processing catalog: {cat_name}")

            # Load catalog once — let download failures propagate immediately
            # to the outer except so a timed-out catalog is not retried.
            catalog = catalog_loader.get_optimized_catalog(cat_name, params)

            # Try optimized detection path, fall back to standard on failure.
            try:
                # Prepare detections with magnitude fallback if needed
                det_for_analysis = detections.copy()
                if 'MAG_CALIB' not in det_for_analysis.colnames:
                    raise ValueError("No suitable magnitude/error columns available")

                # Pass adaptive configuration through detections.meta if enabled
                if config and config.detection.enable_adaptive_idlimit:
                    det_for_analysis.meta['adaptive_idlimit_enabled'] = True
                    det_for_analysis.meta['adaptive_nsigma'] = config.detection.adaptive_nsigma
                    det_for_analysis.meta['adaptive_percentile'] = config.detection.adaptive_percentile
                    det_for_analysis.meta['idlimit_min_px'] = config.detection.idlimit_min_px
                    det_for_analysis.meta['idlimit_max_px'] = config.detection.idlimit_max_px
                    det_for_analysis.meta['use_astvar'] = config.detection.use_astvar

                    logger.debug(f"Enabled adaptive identification for {cat_name}: "
                                 f"nsigma={config.detection.adaptive_nsigma}, "
                                 f"percentile={config.detection.adaptive_percentile}%")

                candidates = catalog.get_transient_candidates_optimized(
                    detections=det_for_analysis,
                    idlimit=idlimit,
                    mag_change_threshold=mag_change_threshold,
                    siglim=config.detection.siglim if config else 5.0,
                    new_source_siglim=config.detection.new_source_siglim if config else None,
                    frame=10.0
                )
                logger.info(f"✅ Used optimized detection for {cat_name}")

            except Exception as opt_error:
                logger.warning(f"Optimized detection failed for {cat_name}: {opt_error}")
                logger.info(f"Falling back to standard detection...")

                # Reuse the already-downloaded catalog; do not re-fetch.
                candidates = catalog.get_transient_candidates(detections, idlimit)
                logger.info(f"✅ Used standard detection for {cat_name}")

            if len(candidates) > 0:
                logger.info(f"Found {len(candidates)} candidates from {cat_name}")

                # Apply MAGLIM-based filtering: drop rows fainter than this
                # exposure's own single-image depth (not physically
                # recoverable without stacking -- see DetectionConfig.
                # maglim_filter_multiplier's docstring).
                try:
                    # Respect user's request: do not use MAG_ISO-substituted MAG_CALIB for this rule
                    if candidates.meta.get('mag_calib_is_fallback', False):
                        logger.debug("Skipping MAGLIM-based filtering (MAG_CALIB fallback was used)")
                    else:
                        maglim = None
                        for key in ('MAGLIM', 'MAGLIMIT', 'maglim', 'maglimit'):
                            if key in candidates.meta:
                                maglim = float(candidates.meta[key])
                                break
                        maglim_mult = config.detection.maglim_filter_multiplier if config else 1.1
                        if maglim is not None and 'MAG_CALIB' in candidates.colnames:
                            keep_mask = np.array(candidates['MAG_CALIB'], dtype=float) <= (maglim_mult * maglim)
                            removed = int(np.sum(~keep_mask))
                            if removed > 0:
                                candidates = candidates[keep_mask]
                                logger.info(f"MAGLIM filter removed {removed} candidates (>{maglim_mult}x MAGLIM), {len(candidates)} remain")
                except Exception as e:
                    logger.debug(f"MAGLIM-based filtering skipped due to error: {e}")

                # Add features with error handling
                try:
                    # Compute image_id for proper catalog context caching
                    image_id = catalog._generate_image_id(detections)

                    _add_detection_features(candidates)
                    _add_catalog_context_safe(candidates, catalog, radius_check, config, logger,
                                               filter_pattern, image_id=image_id)
                    weights = config.detection if config else DetectionConfig()
                    add_base_quality_scores(candidates, weights)
                except Exception as feature_error:
                    logger.warning(f"Feature addition failed for {cat_name}: {feature_error}")
                    # Ensure we have minimum required columns
                    if 'quality_score' not in candidates.columns:
                        candidates['quality_score'] = [0.5] * len(candidates)
                    if 'candidate_type' not in candidates.columns:
                        candidates['candidate_type'] = ['new'] * len(candidates)

                candidates["reference_catalog"] = cat_name
            else:
                logger.info(f"No candidates found from {cat_name}")
                # Create empty table with required columns
                candidates = Table()
                candidates['quality_score'] = []
                candidates['candidate_type'] = []
                candidates["reference_catalog"] = []

            results[cat_name] = candidates

        except Exception as e:
            logger.error(f"Failed to process catalog {cat_name}: {str(e)}")
            # Create empty table with required columns for failed catalog
            empty_table = Table()
            empty_table['quality_score'] = []
            empty_table['candidate_type'] = []
            empty_table["reference_catalog"] = []
            results[cat_name] = empty_table
            continue

    # VSX/SkyBoT filtering does NOT happen here -- see stdpipe_filters.py.
    # VSX is purely positional (a variable star's position never changes), so
    # filtering it per-catalog-per-epoch on raw candidates (up to ~1600 rows
    # from usno alone, x2 catalogs x9 epochs) was ~300x more position checks
    # than necessary; it runs once on the final clustered candidates instead
    # (clustering.py). SkyBoT is time-dependent (needs a real per-epoch
    # timestamp) so it can't move to the final-candidate stage the same way,
    # but it was also running once per catalog per epoch when the catalogs
    # share the same epoch and timestamp -- it now runs once per epoch, in
    # clustering.py's combine_results, right after catalogs are vstacked
    # together for that epoch.

    return results


def _add_catalog_context_safe(candidates, catalog, radius, config, logger, filter_pattern=None, image_id=None):
    """Safe version of catalog context addition with fallbacks and config support."""
    if len(candidates) == 0:
        return

    # Use config values if available
    if config and hasattr(config.detection, 'radius_check'):
        radius = config.detection.radius_check

    try:
        # Try optimized method first
        positions = np.column_stack((candidates["X_IMAGE"], candidates["Y_IMAGE"]))
        stats = catalog.compute_local_statistics(
            positions=positions,
            radius=radius,
            filter_pattern=filter_pattern,
            image_id=image_id
        )

        for stat_name, values in stats.items():
            candidates[stat_name] = values
        logger.debug(f"✅ Used optimized context statistics")

    except Exception as e:
        logger.warning(f"Optimized context failed: {e}")
        # Add default values
        candidates["nearby_sources"] = [0] * len(candidates)
        candidates["source_density"] = [0.0] * len(candidates)
        candidates["nearest_source_dist"] = [np.inf] * len(candidates)
        logger.debug(f"✅ Added default context values")
