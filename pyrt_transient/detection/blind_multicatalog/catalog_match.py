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

from pyrt_transient.catalog import CatalogNoCoverageError
from pyrt_transient.config_trans import DetectionConfig
from pyrt_transient.core.color_model import has_colour_terms
from pyrt_transient.core.radii import _plate_scale_arcsec_per_px
from pyrt_transient.core.scoring import add_base_quality_scores


def _match_floor_px(config, cat_name, detections):
    """config.detection.catalog_match_floor_arcsec entry for this catalogue
    (substring match on the name), in pixels; None if there is none."""
    floors = getattr(config.detection, "catalog_match_floor_arcsec", None) or {}
    name = str(cat_name).lower()
    arcsec = max((v for k, v in floors.items() if k.lower() in name), default=None)
    if arcsec is None:
        return None
    scale = _plate_scale_arcsec_per_px(detections, config.detection.default_plate_scale_arcsec_per_px)
    return float(arcsec) / float(scale)


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


def _empty_candidates_table() -> Table:
    """Empty placeholder for a catalog that failed or found nothing.

    Must give quality_score/candidate_type/reference_catalog the same
    dtypes a real, non-empty result would -- `Table()['col'] = []` infers
    float64 for an empty Python list, but a successful catalog's
    candidate_type/reference_catalog are real strings. vstack-ing a
    float64 empty column against a string column raises
    `TableMergeError: incompatible types` in combine_results -- verified
    directly against a real field (GRB151027B) where USNO-B genuinely has
    zero coverage while Gaia succeeds, an entirely ordinary per-field
    catalog-coverage gap, not a rare corner case.
    """
    return Table({
        "quality_score": np.array([], dtype=float),
        "candidate_type": np.array([], dtype=str),
        "reference_catalog": np.array([], dtype=str),
    })


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
    """Enhanced version with better error handling.

    Returns (results, failed): `results` maps catalogue name -> candidate
    table for every catalogue that could be searched, `failed` maps
    catalogue name -> error string for those that raised (a download
    failure, a timeout). A catalogue that was searched and legitimately has
    no coverage or no rows appears in neither -- that is a permanent
    property of the field, while `failed` is this run's bad luck and the
    caller must not cache the epoch as finished because of it.
    """
    results = {}
    _warn_if_band_uncorrected(detections, logger)
    # Catalogues that could not be used for this field: no coverage, no
    # rows, download failure. They are NOT put into `results` as empty
    # tables -- combine_results counts len(results) as the agreement
    # denominator, so an empty entry would veto every source under
    # min_catalogs_fraction=1.0 and the field would silently yield zero
    # candidates (e.g. any field south of Dec -30 with Pan-STARRS in the
    # list). A catalogue that WAS searched and found nothing is different
    # and does stay in (see _empty_candidates_table below).
    #
    # The two are kept apart: `no_coverage` is a fact about the field that
    # will be just as true next run, `failed` is a download that blew up and
    # may well work next time.
    no_coverage = {}
    failed = {}

    for cat_name in catalogs:
        try:
            logger.info(f"Processing catalog: {cat_name}")

            # Load catalog once — let download failures propagate immediately
            # to the outer except so a timed-out catalog is not retried.
            # Note the load is eager: pyrt's Catalog.__init__ runs the query,
            # so an empty field surfaces as CatalogNoCoverageError below
            # rather than as a None/empty return here.
            catalog = catalog_loader.get_optimized_catalog(cat_name, params)
            if catalog is None or len(catalog) == 0:
                no_coverage[cat_name] = "no coverage / no rows for this field"
                logger.warning(f"Catalog {cat_name} unavailable for this field (no rows); "
                               f"excluded from the agreement requirement")
                continue

            # The detection table the matcher sees: a copy, so the options
            # below never reach the caller's epoch table, used by both the
            # optimized path and the fallback so both run with the same
            # configuration.
            det_for_analysis = detections.copy()
            if config:
                det_for_analysis.meta['propagate_proper_motion'] = config.detection.propagate_proper_motion
                det_for_analysis.meta['reject_saturated'] = config.detection.reject_saturated
                det_for_analysis.meta['saturation_adu'] = config.detection.saturation_adu
                det_for_analysis.meta['saturation_margin_mag'] = config.detection.saturation_margin_mag

            # Try optimized detection path, fall back to standard on failure.
            try:
                if 'MAG_CALIB' not in det_for_analysis.colnames:
                    raise ValueError("No suitable magnitude/error columns available")

                # Pass adaptive configuration through detections.meta if enabled
                if config and config.detection.enable_adaptive_idlimit:
                    det_for_analysis.meta['adaptive_idlimit_enabled'] = True
                    det_for_analysis.meta['adaptive_nsigma'] = config.detection.adaptive_nsigma
                    det_for_analysis.meta['adaptive_percentile'] = config.detection.adaptive_percentile
                    min_px = config.detection.idlimit_min_px
                    floor_px = _match_floor_px(config, cat_name, det_for_analysis)
                    if floor_px is not None:
                        min_px = max(min_px, floor_px)
                    det_for_analysis.meta['idlimit_min_px'] = min_px
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
                    unphotometered_match_is_new=(
                        config.detection.unphotometered_match_is_new if config else True),
                    unphotometered_veto_max_brightening=(
                        getattr(config.detection, "unphotometered_veto_max_brightening_mag", None)
                        if config else None),
                    frame=10.0
                )
                logger.info(f"✅ Used optimized detection for {cat_name}")

            except Exception as opt_error:
                logger.warning(f"Optimized detection failed for {cat_name}: {opt_error}")
                logger.info(f"Falling back to standard detection...")

                # Reuse the already-downloaded catalog; do not re-fetch.
                candidates = catalog.get_transient_candidates(det_for_analysis, idlimit)
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
                candidates = _empty_candidates_table()

            results[cat_name] = candidates

        except CatalogNoCoverageError as e:
            no_coverage[cat_name] = str(e)
            logger.warning(f"Catalog {cat_name} unavailable for this field ({e}); "
                           f"excluded from the agreement requirement")
            continue

        except Exception as e:
            logger.error(f"Failed to process catalog {cat_name}: {str(e)}")
            failed[cat_name] = str(e)
            logger.warning(f"Catalog {cat_name} failed for this field ({e}); "
                           f"excluded from the agreement requirement")
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

    unavailable = {**no_coverage, **failed}
    if unavailable:
        logger.warning(f"{len(unavailable)}/{len(catalogs)} catalogs unavailable for this field: "
                       f"{sorted(unavailable)}; agreement required among the {len(results)} available")
    if not results:
        logger.error("No reference catalog available for this field -- no candidates can be produced")
    return results, failed


def _warn_if_band_uncorrected(detections, logger) -> None:
    """catalog.py compares MAG_CALIB against the catalogue's Sloan r
    (`magnitudes[idx, 1]`), corrected to the frame's system only through
    the colour terms in RESPONSE. A non-r frame whose RESPONSE has no colour
    term is therefore compared against plain r -- every red/blue star then
    carries a colour-dependent offset that can read as a magnitude change.
    """
    meta = getattr(detections, "meta", None) or {}
    band = str(meta.get("PHFILTER", meta.get("FILTER", "")) or "")
    if not band or "r" in band.lower():
        return
    if has_colour_terms(meta.get("RESPONSE")):
        return
    logger.warning(
        f"{meta.get('filename', 'this epoch')}: band {band!r} is compared against catalogue "
        f"Sloan r with no colour term in RESPONSE={meta.get('RESPONSE')!r} -- colour-dependent "
        f"offsets can be mistaken for magnitude changes"
    )


def _add_catalog_context_safe(candidates, catalog, radius, config, logger, filter_pattern=None, image_id=None):
    """Safe version of catalog context addition with fallbacks and config support."""
    if len(candidates) == 0:
        return

    # Use config values if available
    if config and hasattr(config.detection, 'radius_check'):
        radius = config.detection.radius_check

    max_mag = None
    margin = getattr(config.detection, "isolation_max_mag_margin", None) if config else None
    if margin is not None:
        meta = getattr(candidates, "meta", None) or {}
        limit = meta.get("MAGLIMIT", meta.get("MAGLIM"))
        try:
            max_mag = float(limit) + float(margin)
        except (TypeError, ValueError):
            max_mag = None

    try:
        # Try optimized method first
        positions = np.column_stack((candidates["X_IMAGE"], candidates["Y_IMAGE"]))
        stats = catalog.compute_local_statistics(
            positions=positions,
            radius=radius,
            filter_pattern=filter_pattern,
            image_id=image_id,
            max_mag=max_mag,
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
