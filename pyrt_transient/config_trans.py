#!/usr/bin/python3

import argparse
import configparser
import os
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from pathlib import Path

DEFAULT_CONFIG_FILE = "~/.config/dophot3/config"


@dataclass
class DetectionConfig:
    """Configuration for transient detection parameters."""
    idlimit_px: float = 3.0
    position_match_radius_arcsec: float = 2.0
    # Lowered from 5: paired with new_source_siglim (below) and
    # compute_lightcurve_score_factor's n_detections confidence term, a real
    # source now earns its quality_score through repeated, consistent
    # detection rather than needing to individually pass a stricter
    # per-epoch bar just to be *considered*. 3 still requires genuine
    # repeat confirmation, not a single/double fluke.
    min_n_detections: int = 3
    min_catalogs_fraction: float = 1.0
    min_quality: float = 0.2
    radius_check: float = 20.0
    filter_pattern: str = "r"

    # Significance threshold (in units of magnitude S/N) below which a
    # detection is excluded before candidate consideration. siglim=5.0 means
    # MAGERR_CALIB >= 1.091/5.0 =~ 0.218 mag is excluded outright.
    #
    # new_source_siglim applies only to detections with no reference-catalog
    # match at all ("new" candidates -- the case a real, previously
    # uncatalogued GRB afterglow falls into). It defaults to siglim (i.e. no
    # behavior change) but can be set lower: cross-checking a real 18-GRB
    # replay against GCN circulars found several genuine, positionally- and
    # photometrically-confirmed afterglows excluded here because a single
    # epoch's S/N fell just under 5, despite appearing consistently across
    # many epochs. min_n_detections (the cross-epoch clustering requirement,
    # not this gate) is what actually protects against admitting noise --
    # a real source repeats at the same position; noise doesn't. Matched
    # (known-catalog) detections keep the stricter `siglim` bar regardless,
    # since flagging a catalogued star as "changed" on marginal significance
    # risks false positives from ordinary photometric scatter.
    # catalog_match.py drops any candidate fainter than
    # maglim_filter_multiplier * this epoch's own MAGLIM (the image's own
    # characterized single-exposure limiting magnitude) -- a second,
    # independent admission gate from siglim/new_source_siglim below, based
    # on absolute brightness relative to this specific exposure's depth
    # rather than measurement S/N. Left at the original 1.1 deliberately:
    # unlike the significance gate, this one is not just conservative --
    # a source meaningfully fainter than a single exposure's own MAGLIM is
    # not physically recoverable from that exposure alone, no matter how
    # the significance/consistency thresholds are tuned. Reaching fainter
    # than MAGLIM needs actual image stacking/co-addition before detection
    # (not built yet -- see FUTURE_IDEAS.md), not a looser cutoff on noise.
    # (The in-code log message says "1.5x MAGLIM" while this has always
    # been 1.1 -- a stale message, not evidence 1.1 was a regression.)
    maglim_filter_multiplier: float = 1.1

    siglim: float = 5.0
    # Aggressively low by traditional standards (S/N > 1.5) -- deliberate.
    # Admission this permissive is only safe because of two things downstream:
    # compute_lightcurve_score_factor's n_detections term (a source repeating
    # consistently across many epochs earns real confidence; isolated noise
    # doesn't repeat at the same position) and the final min_quality gate in
    # combine_with_lightcurves (applied to the fully-accumulated score, not
    # this early per-epoch one). Verified against GCN-confirmed real GRBs:
    # several genuine afterglows only clear ~7/18-20 epochs at this bar, not
    # enough at siglim=3.0 (1 epoch) or siglim=2.0 (3 epochs, no margin)
    # against min_n_detections=3.
    new_source_siglim: Optional[float] = 1.5

    # Reference catalogs to query per detection table. "atlas@localhost" is a
    # local-only service unavailable outside the production host; local dev
    # and the baseline harness override this to ["gaia", "usno"].
    catalogs: List[str] = field(default_factory=lambda: ["atlas@localhost", "gaia", "usno"])
    
    # Adaptive identification parameters
    enable_adaptive_idlimit: bool = True
    adaptive_nsigma: float = 3.0
    adaptive_percentile: float = 95.0
    idlimit_min_px: float = 1.0
    idlimit_max_px: float = 8.0
    use_astvar: bool = True
    default_plate_scale_arcsec_per_px: float = 0.33
    
    # Trail detection parameters
    trail_min_epochs: int = 3
    trail_motion_sigma_min: float = 0.5  # arcsec
    trail_motion_sig_tau: float = 3.0
    trail_score_threshold: float = 0.7
    trail_downweight_factor: float = 3.0
    # Minimum displacement in units of per-frame WCS error (ASTSIGMA) to flag as trail.
    # Total displacement = motion_rate * time_span must exceed this × ASTSIGMA.
    trail_astsigma_displacement_threshold: float = 3.0

    # Time-adaptive linking for moving objects
    moving_if_sigma_gt: float = 3.0
    position_match_radius_arcsec_moving: float = 8.0
    position_match_radius_arcsec_moving_max: float = 15.0
    
    # Score weights for candidate ranking
    magnitude_weight: float = 1.0
    significance_weight: float = 2.0
    consistency_weight: float = 1.5
    isolation_weight: float = 1.0
    lc_shape_weight: float = 1.0
    
    # VSX variable star filter parameters
    vsx_filter_enabled: bool = True
    vsx_match_radius_arcsec: float = 2.5
    vsx_catalog_id: str = "B/vsx/vsx"

    # Detection strategy: "blind_multicatalog" (cross-match against reference
    # catalogs, the only strategy today) or "subtraction" (differencing
    # against a template -- see detection/subtraction/). Kept as a plain str
    # rather than an enum so config files don't need an import to set it.
    strategy: str = "blind_multicatalog"

    # subtraction strategy only, below. template_source picks how the
    # template image is obtained: "own_epoch" reuses
    # detection/reference_frame.py's ReferenceFrameSelector to pick the best
    # prior epoch of the same field (no external dependency, needs enough
    # prior epochs); "ps1"/"legacysurvey" fetch an external-survey template
    # via stdpipe.templates, reprojected onto the science WCS -- works on a
    # field's very first observation but depends on survey coverage/network.
    template_source: str = "ps1"
    # subtraction_engine picks the differencing algorithm: "hotpants" (via
    # stdpipe.subtraction.run_hotpants -- matches how tests/2026kid/'s real
    # fixture was produced) or "zogy" (via PyZOGY, matching the one-off
    # subtract_supernova.py reference script). Both write a diff FITS with a
    # `TEMPLATE` header keyword so downstream cutout/frontend code doesn't
    # need to know which engine produced a given diff image.
    subtraction_engine: str = "hotpants"

    # Dipole/artifact rejection (detection/subtraction/artifact_filters.py):
    # a positive+negative flux pair close together is the standard signature
    # of imperfect subtraction (bad registration, saturated-star wings,
    # cosmic rays) rather than a real transient. Reject any diff detection
    # whose nearest opposite-sign counterpart is closer than
    # dipole_reject_radius_arcsec and whose flux ratio to that counterpart
    # exceeds dipole_reject_flux_ratio (i.e. comparable brightness, not a
    # coincidental faint neighbor).
    dipole_reject_radius_arcsec: float = 3.0
    dipole_reject_flux_ratio: float = 0.5

    # Template cache (subtraction strategy, template_source != "own_epoch"):
    # reprojected survey templates are expensive to build and reusable
    # across many nights of the same field, but are large FITS files that
    # will fill the disk if kept forever. template_cache_dir stores them
    # keyed by field/radius/band; template_cache_max_size_gb bounds the
    # cache the same way FrontendConfig.max_dir_size_gb bounds the website
    # directory (LRU eviction, see frontend_generator.py's
    # cleanup_old_files/enforce_disk_budget_strict).
    template_cache_dir: str = "./template_cache"
    template_cache_max_size_gb: float = 20.0

    # diff_input_mode picks what pipeline_magic_sn.py's subtraction branch
    # expects as input: "prebuilt" (Phase A -- ecsv_file/fits_file are
    # already a diff-image pair, e.g. an externally-produced campaign like
    # tests/2026kid/) or "raw" (Phase B -- ecsv_file/fits_file are a raw
    # science epoch, and the pipeline builds the template/diff/extraction
    # itself via detection/subtraction/templates.py, differencing.py,
    # extraction.py before handing off to the same SubtractionStrategy).
    # Defaults to "prebuilt" so existing Phase A behavior is unchanged
    # unless a caller opts in.
    diff_input_mode: str = "prebuilt"
    # Reference photometric catalog used to derive each science epoch's own
    # zeropoint (detection/subtraction/extraction.py) -- calibrating
    # against the *science* image, never the diff image itself (see
    # extraction.py's module docstring for why that doesn't work).
    photometric_catalog: str = "ps1"

    # apply_morphology_filter's thresholds (pipeline_magic_sn.py), exposed
    # here rather than left hardcoded in the function signature: verified
    # directly that the 0.4/({0.5,2.0}) defaults, implicitly tuned against
    # the real tests/2026kid/ fixture's external SExtractor-based diff
    # catalogs, are measurably too strict for stdpipe's SEP-based
    # extraction (detection/subtraction/extraction.py, Phase B's own
    # diff-image source measurement) -- on one real diff image, 57% of all
    # genuine SEP detections (43/75) had ELLIPTICITY >= 0.4, including the
    # real AT2026kid target itself (0.594), which was silently dropped by
    # the filter as a result. Needs real-data tuning per extraction method
    # rather than one blind guess at a replacement number -- exposed as
    # config so that tuning can happen without a code change once there's
    # enough real SEP-extracted data to calibrate against.
    morphology_max_ellipticity: float = 0.4
    morphology_fwhm_ratio_min: float = 0.5
    morphology_fwhm_ratio_max: float = 2.0

    # Image stacking/co-addition (detection/stacking.py) -- GRB
    # (blind_multicatalog) pipeline only, see FUTURE_IDEAS.md's "Image
    # stacking/co-addition". Runs automatically (no separate strategy to
    # opt into) once enough same-field epochs exist, but only as a
    # try-harder fallback: skipped entirely once an existing candidate
    # already scores above stacking_score_threshold, so it doesn't spend
    # pyrt-combine's runtime on every single run.
    stacking_enabled: bool = True
    # FUTURE_IDEAS' proposed range (10-20 epochs) for a worthwhile depth
    # gain without needing the full ~250 stacked frames a 3-mag gain would
    # need.
    stacking_min_epochs: int = 10
    stacking_max_epochs: int = 20
    # Don't re-run pyrt-combine on every single new epoch once triggered --
    # only once this many more real epochs have accumulated since the last
    # build.
    stacking_rebuild_interval: int = 5
    # Skip stacking once an existing candidate already scores at or above
    # this -- min_quality=0.2 is the bare admission floor; real confirmed
    # afterglows found via check_baseline.py range roughly 2.7-56.5, so a
    # candidate already at 1.0 is meaningfully above noise and probably not
    # worth the extra compute to try to beat. Tunable.
    stacking_score_threshold: float = 1.0
    # SEP detection threshold on the stack -- matches
    # extraction.py's calibrate_science_zeropoint default.
    stacking_detect_thresh: float = 5.0
    # pyrt-combine -u: no per-frame photometry file available in this
    # pipeline's context, so uniform (equal-weight) combination is used
    # rather than photometric weighting.
    stacking_uniform_weighting: bool = True


@dataclass
class FrontendConfig:
    """Configuration for frontend website generation."""
    max_candidates: int = 100
    max_cutouts_per_candidate: int = 20
    image_format: str = "png"
    image_quality: int = 85  # For JPEG, ignored for PNG
    thumbnail_size_px: int = 100
    cutout_size_px: int = 50
    max_dir_size_gb: float = 5.0
    
    # Template and styling
    template_dir: Optional[str] = None
    css_theme: str = "default"
    
    # Deferred lightcurve copying system
    fast_lightcurve_copy: bool = True  # Defer to final sync
    lightcurve_link_mode: str = "auto"  # "auto" | "hardlink" | "symlink" | "copy"
    lightcurve_workers: int = 6  # Parallel copy workers
    verify_by_hash: bool = False  # Optional hash verification
    cleanup_orphaned_lightcurves: bool = True  # Remove stale files


@dataclass
class CachingConfig:
    """Configuration for caching behavior."""
    cache_dir: str = "./catalog_cache"
    max_age_days: float = 30.0
    enable_catalog_cache: bool = True
    enable_coord_cache: bool = True  
    enable_kdtree_cache: bool = True
    enable_photometric_cache: bool = True
    
    # Cache size limits
    max_cache_size_mb: float = 1000.0
    cleanup_on_startup: bool = False


@dataclass
class LoggingConfig:
    """Configuration for logging behavior."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Optional[str] = None
    console: bool = True
    
    # Module-specific levels
    module_levels: Dict[str, str] = field(default_factory=lambda: {
        "catalog": "INFO",
        "transient_analyser": "INFO", 
        "frontend_generator": "INFO",
        "pipeline_magic": "INFO"
    })


@dataclass
class PipelineConfig:
    """Main pipeline configuration combining all sub-configs."""
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    frontend: FrontendConfig = field(default_factory=FrontendConfig)
    caching: CachingConfig = field(default_factory=CachingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Global pipeline settings
    base_data_dir: str = "/home/fnovotny/transient_work/"
    base_public_dir: Optional[str] = None  # Default to ~/public_html
    generate_frontend: bool = True
    parallel_processing: bool = False
    max_workers: int = 4
    
    @classmethod
    def from_file(cls, config_file: str) -> "PipelineConfig":
        """Load pipeline configuration from file."""
        config_path = Path(config_file).expanduser()
        if not config_path.exists():
            logging.warning(f"Config file {config_file} not found, using defaults")
            return cls()
        
        config = configparser.ConfigParser()
        config.read(config_path)
        
        # Create config object with defaults
        pipeline_config = cls()
        
        # Update detection config
        if "detection" in config:
            det_section = config["detection"]
            pipeline_config.detection.idlimit_px = det_section.getfloat("idlimit_px", pipeline_config.detection.idlimit_px)
            pipeline_config.detection.position_match_radius_arcsec = det_section.getfloat("position_match_radius_arcsec", pipeline_config.detection.position_match_radius_arcsec)
            pipeline_config.detection.min_n_detections = det_section.getint("min_n_detections", pipeline_config.detection.min_n_detections)
            pipeline_config.detection.min_catalogs_fraction = det_section.getfloat("min_catalogs_fraction", pipeline_config.detection.min_catalogs_fraction)
            pipeline_config.detection.min_quality = det_section.getfloat("min_quality", pipeline_config.detection.min_quality)
            pipeline_config.detection.radius_check = det_section.getfloat("radius_check", pipeline_config.detection.radius_check)
            pipeline_config.detection.filter_pattern = det_section.get("filter_pattern", pipeline_config.detection.filter_pattern)
            
            # Adaptive identification parameters
            pipeline_config.detection.enable_adaptive_idlimit = det_section.getboolean("enable_adaptive_idlimit", pipeline_config.detection.enable_adaptive_idlimit)
            pipeline_config.detection.adaptive_nsigma = det_section.getfloat("adaptive_nsigma", pipeline_config.detection.adaptive_nsigma)
            pipeline_config.detection.adaptive_percentile = det_section.getfloat("adaptive_percentile", pipeline_config.detection.adaptive_percentile)
            pipeline_config.detection.idlimit_min_px = det_section.getfloat("idlimit_min_px", pipeline_config.detection.idlimit_min_px)
            pipeline_config.detection.idlimit_max_px = det_section.getfloat("idlimit_max_px", pipeline_config.detection.idlimit_max_px)
            pipeline_config.detection.use_astvar = det_section.getboolean("use_astvar", pipeline_config.detection.use_astvar)
            
            # Trail detection parameters
            pipeline_config.detection.trail_min_epochs = det_section.getint("trail_min_epochs", pipeline_config.detection.trail_min_epochs)
            pipeline_config.detection.trail_motion_sigma_min = det_section.getfloat("trail_motion_sigma_min", pipeline_config.detection.trail_motion_sigma_min)
            pipeline_config.detection.trail_motion_sig_tau = det_section.getfloat("trail_motion_sig_tau", pipeline_config.detection.trail_motion_sig_tau)
            pipeline_config.detection.trail_score_threshold = det_section.getfloat("trail_score_threshold", pipeline_config.detection.trail_score_threshold)
            pipeline_config.detection.trail_downweight_factor = det_section.getfloat("trail_downweight_factor", pipeline_config.detection.trail_downweight_factor)
            pipeline_config.detection.trail_astsigma_displacement_threshold = det_section.getfloat("trail_astsigma_displacement_threshold", pipeline_config.detection.trail_astsigma_displacement_threshold)
            pipeline_config.detection.moving_if_sigma_gt = det_section.getfloat("moving_if_sigma_gt", pipeline_config.detection.moving_if_sigma_gt)
            pipeline_config.detection.position_match_radius_arcsec_moving = det_section.getfloat("position_match_radius_arcsec_moving", pipeline_config.detection.position_match_radius_arcsec_moving)
            pipeline_config.detection.position_match_radius_arcsec_moving_max = det_section.getfloat("position_match_radius_arcsec_moving_max", pipeline_config.detection.position_match_radius_arcsec_moving_max)
            
            # Score weights
            pipeline_config.detection.magnitude_weight = det_section.getfloat("magnitude_weight", pipeline_config.detection.magnitude_weight)
            pipeline_config.detection.significance_weight = det_section.getfloat("significance_weight", pipeline_config.detection.significance_weight)
            pipeline_config.detection.consistency_weight = det_section.getfloat("consistency_weight", pipeline_config.detection.consistency_weight)
            pipeline_config.detection.isolation_weight = det_section.getfloat("isolation_weight", pipeline_config.detection.isolation_weight)
            # lc_shape_weight was incorrectly assigned to isolation_weight; fix to proper target
            pipeline_config.detection.lc_shape_weight = det_section.getfloat("lc_shape_weight", pipeline_config.detection.lc_shape_weight)
            # VSX variable star filter parameters
            pipeline_config.detection.vsx_filter_enabled = det_section.getboolean("vsx_filter_enabled", pipeline_config.detection.vsx_filter_enabled)
            pipeline_config.detection.vsx_match_radius_arcsec = det_section.getfloat("vsx_match_radius_arcsec", pipeline_config.detection.vsx_match_radius_arcsec)
            pipeline_config.detection.vsx_catalog_id = det_section.get("vsx_catalog_id", pipeline_config.detection.vsx_catalog_id)

            # Detection strategy / subtraction parameters
            pipeline_config.detection.strategy = det_section.get("strategy", pipeline_config.detection.strategy)
            pipeline_config.detection.template_source = det_section.get("template_source", pipeline_config.detection.template_source)
            pipeline_config.detection.subtraction_engine = det_section.get("subtraction_engine", pipeline_config.detection.subtraction_engine)
            pipeline_config.detection.dipole_reject_radius_arcsec = det_section.getfloat("dipole_reject_radius_arcsec", pipeline_config.detection.dipole_reject_radius_arcsec)
            pipeline_config.detection.dipole_reject_flux_ratio = det_section.getfloat("dipole_reject_flux_ratio", pipeline_config.detection.dipole_reject_flux_ratio)
            pipeline_config.detection.template_cache_dir = det_section.get("template_cache_dir", pipeline_config.detection.template_cache_dir)
            pipeline_config.detection.template_cache_max_size_gb = det_section.getfloat("template_cache_max_size_gb", pipeline_config.detection.template_cache_max_size_gb)
            pipeline_config.detection.diff_input_mode = det_section.get("diff_input_mode", pipeline_config.detection.diff_input_mode)
            pipeline_config.detection.photometric_catalog = det_section.get("photometric_catalog", pipeline_config.detection.photometric_catalog)
            pipeline_config.detection.morphology_max_ellipticity = det_section.getfloat("morphology_max_ellipticity", pipeline_config.detection.morphology_max_ellipticity)
            pipeline_config.detection.morphology_fwhm_ratio_min = det_section.getfloat("morphology_fwhm_ratio_min", pipeline_config.detection.morphology_fwhm_ratio_min)
            pipeline_config.detection.morphology_fwhm_ratio_max = det_section.getfloat("morphology_fwhm_ratio_max", pipeline_config.detection.morphology_fwhm_ratio_max)

            # Image stacking/co-addition
            pipeline_config.detection.stacking_enabled = det_section.getboolean("stacking_enabled", pipeline_config.detection.stacking_enabled)
            pipeline_config.detection.stacking_min_epochs = det_section.getint("stacking_min_epochs", pipeline_config.detection.stacking_min_epochs)
            pipeline_config.detection.stacking_max_epochs = det_section.getint("stacking_max_epochs", pipeline_config.detection.stacking_max_epochs)
            pipeline_config.detection.stacking_rebuild_interval = det_section.getint("stacking_rebuild_interval", pipeline_config.detection.stacking_rebuild_interval)
            pipeline_config.detection.stacking_score_threshold = det_section.getfloat("stacking_score_threshold", pipeline_config.detection.stacking_score_threshold)
            pipeline_config.detection.stacking_detect_thresh = det_section.getfloat("stacking_detect_thresh", pipeline_config.detection.stacking_detect_thresh)
            pipeline_config.detection.stacking_uniform_weighting = det_section.getboolean("stacking_uniform_weighting", pipeline_config.detection.stacking_uniform_weighting)

        # Update frontend config
        if "frontend" in config:
            fe_section = config["frontend"]
            pipeline_config.frontend.max_candidates = fe_section.getint("max_candidates", pipeline_config.frontend.max_candidates)
            pipeline_config.frontend.max_cutouts_per_candidate = fe_section.getint("max_cutouts_per_candidate", pipeline_config.frontend.max_cutouts_per_candidate)
            pipeline_config.frontend.image_format = fe_section.get("image_format", pipeline_config.frontend.image_format)
            pipeline_config.frontend.image_quality = fe_section.getint("image_quality", pipeline_config.frontend.image_quality)
            pipeline_config.frontend.thumbnail_size_px = fe_section.getint("thumbnail_size_px", pipeline_config.frontend.thumbnail_size_px)
            pipeline_config.frontend.cutout_size_px = fe_section.getint("cutout_size_px", pipeline_config.frontend.cutout_size_px)
            pipeline_config.frontend.max_dir_size_gb = fe_section.getfloat("max_dir_size_gb", pipeline_config.frontend.max_dir_size_gb)
            pipeline_config.frontend.template_dir = fe_section.get("template_dir", pipeline_config.frontend.template_dir)
            pipeline_config.frontend.css_theme = fe_section.get("css_theme", pipeline_config.frontend.css_theme)
        
        # Update caching config
        if "caching" in config:
            cache_section = config["caching"]
            pipeline_config.caching.cache_dir = cache_section.get("cache_dir", pipeline_config.caching.cache_dir)
            pipeline_config.caching.max_age_days = cache_section.getfloat("max_age_days", pipeline_config.caching.max_age_days)
            pipeline_config.caching.enable_catalog_cache = cache_section.getboolean("enable_catalog_cache", pipeline_config.caching.enable_catalog_cache)
            pipeline_config.caching.enable_coord_cache = cache_section.getboolean("enable_coord_cache", pipeline_config.caching.enable_coord_cache)
            pipeline_config.caching.enable_kdtree_cache = cache_section.getboolean("enable_kdtree_cache", pipeline_config.caching.enable_kdtree_cache)
            pipeline_config.caching.enable_photometric_cache = cache_section.getboolean("enable_photometric_cache", pipeline_config.caching.enable_photometric_cache)
            pipeline_config.caching.max_cache_size_mb = cache_section.getfloat("max_cache_size_mb", pipeline_config.caching.max_cache_size_mb)
            pipeline_config.caching.cleanup_on_startup = cache_section.getboolean("cleanup_on_startup", pipeline_config.caching.cleanup_on_startup)
        
        # Update logging config
        if "logging" in config:
            log_section = config["logging"]
            pipeline_config.logging.level = log_section.get("level", pipeline_config.logging.level)
            pipeline_config.logging.format = log_section.get("format", pipeline_config.logging.format)
            pipeline_config.logging.file = log_section.get("file", pipeline_config.logging.file)
            pipeline_config.logging.console = log_section.getboolean("console", pipeline_config.logging.console)
        
        # Update global settings
        if "global" in config:
            global_section = config["global"]
            pipeline_config.base_data_dir = global_section.get("base_data_dir", pipeline_config.base_data_dir)
            pipeline_config.base_public_dir = global_section.get("base_public_dir", pipeline_config.base_public_dir)
            pipeline_config.generate_frontend = global_section.getboolean("generate_frontend", pipeline_config.generate_frontend)
            pipeline_config.parallel_processing = global_section.getboolean("parallel_processing", pipeline_config.parallel_processing)
            pipeline_config.max_workers = global_section.getint("max_workers", pipeline_config.max_workers)
        
        return pipeline_config
    
    @classmethod
    def from_dict(cls, config_data: Dict[str, Any]) -> "PipelineConfig":
        """Create PipelineConfig from a dictionary (for YAML support)."""
        pipeline_config = cls()

        # Update detection config
        if "detection" in config_data:
            det_data = config_data["detection"]
            for key, value in det_data.items():
                if hasattr(pipeline_config.detection, key):
                    setattr(pipeline_config.detection, key, value)
        else:
            # Support flat YAML without sections: map known detection keys at top level
            for key, value in config_data.items():
                if hasattr(pipeline_config.detection, key):
                    setattr(pipeline_config.detection, key, value)
        
        # Update frontend config
        if "frontend" in config_data:
            fe_data = config_data["frontend"]
            for key, value in fe_data.items():
                if hasattr(pipeline_config.frontend, key):
                    setattr(pipeline_config.frontend, key, value)
        
        # Update caching config
        if "caching" in config_data:
            cache_data = config_data["caching"]
            for key, value in cache_data.items():
                if hasattr(pipeline_config.caching, key):
                    setattr(pipeline_config.caching, key, value)
        
        # Update logging config
        if "logging" in config_data:
            log_data = config_data["logging"]
            for key, value in log_data.items():
                if key == "module_levels" and isinstance(value, dict):
                    pipeline_config.logging.module_levels.update(value)
                elif hasattr(pipeline_config.logging, key):
                    setattr(pipeline_config.logging, key, value)
        
        # Update global settings
        if "global" in config_data:
            global_data = config_data["global"]
            for key, value in global_data.items():
                if hasattr(pipeline_config, key):
                    setattr(pipeline_config, key, value)
        
        # Support flat structure (no sections)
        for key, value in config_data.items():
            if key not in ["detection", "frontend", "caching", "logging", "global"] and hasattr(pipeline_config, key):
                setattr(pipeline_config, key, value)
        
        return pipeline_config
    
    def setup_logging(self) -> None:
        """Setup logging based on configuration."""
        # Convert string level to logging constant
        level = getattr(logging, self.logging.level.upper(), logging.INFO)
        
        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        
        # Clear any existing handlers
        root_logger.handlers.clear()
        
        formatter = logging.Formatter(self.logging.format)
        
        # Console handler
        if self.logging.console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
        
        # File handler
        if self.logging.file:
            file_handler = logging.FileHandler(self.logging.file)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        
        # Module-specific levels
        for module_name, module_level in self.logging.module_levels.items():
            module_logger = logging.getLogger(module_name)
            module_logger.setLevel(getattr(logging, module_level.upper(), logging.INFO))
    
    def to_file(self, config_file: str) -> None:
        """Save pipeline configuration to file."""
        config = configparser.ConfigParser()
        
        # Detection section
        config["detection"] = {
            "idlimit_px": str(self.detection.idlimit_px),
            "position_match_radius_arcsec": str(self.detection.position_match_radius_arcsec),
            "min_n_detections": str(self.detection.min_n_detections),
            "min_catalogs_fraction": str(self.detection.min_catalogs_fraction),
            "min_quality": str(self.detection.min_quality),
            "radius_check": str(self.detection.radius_check),
            "filter_pattern": self.detection.filter_pattern,
            "enable_adaptive_idlimit": str(self.detection.enable_adaptive_idlimit),
            "adaptive_nsigma": str(self.detection.adaptive_nsigma),
            "adaptive_percentile": str(self.detection.adaptive_percentile),
            "idlimit_min_px": str(self.detection.idlimit_min_px),
            "idlimit_max_px": str(self.detection.idlimit_max_px),
            "use_astvar": str(self.detection.use_astvar),
            "trail_min_epochs": str(self.detection.trail_min_epochs),
            "trail_motion_sigma_min": str(self.detection.trail_motion_sigma_min),
            "trail_motion_sig_tau": str(self.detection.trail_motion_sig_tau),
            "trail_score_threshold": str(self.detection.trail_score_threshold),
            "trail_downweight_factor": str(self.detection.trail_downweight_factor),
            "trail_astsigma_displacement_threshold": str(self.detection.trail_astsigma_displacement_threshold),
            "moving_if_sigma_gt": str(self.detection.moving_if_sigma_gt),
            "position_match_radius_arcsec_moving": str(self.detection.position_match_radius_arcsec_moving),
            "position_match_radius_arcsec_moving_max": str(self.detection.position_match_radius_arcsec_moving_max),
            "magnitude_weight": str(self.detection.magnitude_weight),
            "significance_weight": str(self.detection.significance_weight),
            "consistency_weight": str(self.detection.consistency_weight),
            "isolation_weight": str(self.detection.isolation_weight),
            "lc_shape_weight": str(self.detection.lc_shape_weight),
            "vsx_filter_enabled": str(self.detection.vsx_filter_enabled),
            "vsx_match_radius_arcsec": str(self.detection.vsx_match_radius_arcsec),
            "vsx_catalog_id": self.detection.vsx_catalog_id,
            "strategy": self.detection.strategy,
            "template_source": self.detection.template_source,
            "subtraction_engine": self.detection.subtraction_engine,
            "dipole_reject_radius_arcsec": str(self.detection.dipole_reject_radius_arcsec),
            "dipole_reject_flux_ratio": str(self.detection.dipole_reject_flux_ratio),
            "template_cache_dir": self.detection.template_cache_dir,
            "template_cache_max_size_gb": str(self.detection.template_cache_max_size_gb),
            "diff_input_mode": self.detection.diff_input_mode,
            "photometric_catalog": self.detection.photometric_catalog,
            "morphology_max_ellipticity": str(self.detection.morphology_max_ellipticity),
            "morphology_fwhm_ratio_min": str(self.detection.morphology_fwhm_ratio_min),
            "morphology_fwhm_ratio_max": str(self.detection.morphology_fwhm_ratio_max),
            "stacking_enabled": str(self.detection.stacking_enabled),
            "stacking_min_epochs": str(self.detection.stacking_min_epochs),
            "stacking_max_epochs": str(self.detection.stacking_max_epochs),
            "stacking_rebuild_interval": str(self.detection.stacking_rebuild_interval),
            "stacking_score_threshold": str(self.detection.stacking_score_threshold),
            "stacking_detect_thresh": str(self.detection.stacking_detect_thresh),
            "stacking_uniform_weighting": str(self.detection.stacking_uniform_weighting),
        }
        
        # Frontend section  
        config["frontend"] = {
            "max_candidates": str(self.frontend.max_candidates),
            "max_cutouts_per_candidate": str(self.frontend.max_cutouts_per_candidate),
            "image_format": self.frontend.image_format,
            "image_quality": str(self.frontend.image_quality),
            "thumbnail_size_px": str(self.frontend.thumbnail_size_px),
            "cutout_size_px": str(self.frontend.cutout_size_px),
            "max_dir_size_gb": str(self.frontend.max_dir_size_gb),
            "css_theme": self.frontend.css_theme,
        }
        if self.frontend.template_dir:
            config["frontend"]["template_dir"] = self.frontend.template_dir
        
        # Caching section
        config["caching"] = {
            "cache_dir": self.caching.cache_dir,
            "max_age_days": str(self.caching.max_age_days),
            "enable_catalog_cache": str(self.caching.enable_catalog_cache),
            "enable_coord_cache": str(self.caching.enable_coord_cache),
            "enable_kdtree_cache": str(self.caching.enable_kdtree_cache),
            "enable_photometric_cache": str(self.caching.enable_photometric_cache),
            "max_cache_size_mb": str(self.caching.max_cache_size_mb),
            "cleanup_on_startup": str(self.caching.cleanup_on_startup),
        }
        
        # Logging section
        config["logging"] = {
            "level": self.logging.level,
            "format": self.logging.format,
            "console": str(self.logging.console),
        }
        if self.logging.file:
            config["logging"]["file"] = self.logging.file
        
        # Global section
        config["global"] = {
            "base_data_dir": self.base_data_dir,
            "generate_frontend": str(self.generate_frontend),
            "parallel_processing": str(self.parallel_processing),
            "max_workers": str(self.max_workers),
        }
        if self.base_public_dir:
            config["global"]["base_public_dir"] = self.base_public_dir
        
        # Write to file
        with open(Path(config_file).expanduser(), 'w') as f:
            config.write(f)


def load_config(config_file: str):  # -> Dict[str, Any]:
    """
    Load configuration from the specified file.
    
    :param config_file: Path to the configuration file
    :return: Dictionary containing configuration options
    """
    config = configparser.ConfigParser()
    config.read(os.path.expanduser(config_file))
    return dict(config["DEFAULT"])


def parse_arguments(args=None):
    """
    Parse command-line arguments, integrating with config file options.
    
    :param args: Command line arguments (if None, sys.argv is used)
    :return: Namespace object containing all configuration options
    """
    # First, we'll create a parser just for the config file argument
    conf_parser = argparse.ArgumentParser(add_help=False)
    conf_parser.add_argument(
        "-c",
        "--config",
        default=DEFAULT_CONFIG_FILE,
        help="Specify config file",
        metavar="FILE",
    )
    conf_args, remaining_argv = conf_parser.parse_known_args(args)

    # Now we can load the config file
    config = load_config(conf_args.config)

    # Create the main parser
    parser = argparse.ArgumentParser(
        description="Compute photometric calibration for FITS images",
        # Inherit options from config_parser
        parents=[conf_parser],
    )

    # Add arguments, using config values as defaults
    parser.add_argument(
        "-a",
        "--astrometry",
        action="store_true",
        default=config.get("astrometry", "False"),
        help="Refit astrometric solution using photometry-selected stars",
    )
    parser.add_argument(
        "-A",
        "--aterms",
        default=config.get("aterms"),
        help="Terms to fit for astrometry",
    )
    parser.add_argument(
        "--usewcs",
        default=config.get("usewcs"),
        help="Use this astrometric solution (file with header)",
    )
    parser.add_argument(
        "-b",
        "--basemag",
        default=config.get("basemag", None),
        help='ID of the base filter to be used while fitting (def="Sloan_r"/"Johnson_V")',
    )
    parser.add_argument(
        "-C",
        "--catalog",
        default=config.get("catalog"),
        help="Use this catalog as a reference",
    )
    parser.add_argument(
        "-d",
        "--date",
        action="store",
        help="what to put into the third column (char,mid,bjd), default=mid",
    )
    parser.add_argument(
        "-e",
        "--enlarge",
        type=float,
        default=config.get("enlarge"),
        help="Enlarge catalog search region",
    )
    parser.add_argument(
        "-f",
        "--filter",
        default=config.get("filter"),
        help="Override filter info from fits",
    )
    parser.add_argument(
        "--fsr",
        help="Use forward stepwise regression",
        default=config.get("fsr", "False"),
    )
    parser.add_argument(
        "--fsr-terms",
        help="Terms to be used to do forward stepwise regression",
        default=config.get("fsr_terms", None),
    )
    parser.add_argument("-F", "--flat", help="Produce flats", action="store_true")
    parser.add_argument(
        "-g",
        "--guessbase",
        action="store_true",
        default=config.get("guessbase", "False"),
        help="Try and set base filter from fits header (implies -j if Bessel filter is found)",
    )
    parser.add_argument(
        "-j",
        "--johnson",
        action="store_true",
        default=config.get("johnson", "False"),
        help="Use Stetson Johnson/Cousins filters and not SDSS",
    )
    parser.add_argument(
        "-X", "--tryflt", action="store_true", help="Try different filters (broken)"
    )
    parser.add_argument(
        "-G",
        "--gain",
        action="store",
        help="Provide camera gain",
        type=float,
        default=config.get("gain", 2.3),
    )
    parser.add_argument(
        "-i",
        "--idlimit",
        help="Set a custom idlimit",
        type=float,
        default=config.get("idlimit"),
    )
    parser.add_argument(
        "-k",
        "--makak",
        help="Makak tweaks",
        action="store_true",
        default=config.get("makak", "False"),
    )
    parser.add_argument(
        "-R",
        "--redlim",
        help="Do not get stars redder than this g-r",
        type=float,
        default=config.get("redlim"),
    )
    parser.add_argument(
        "-B",
        "--bluelim",
        help="Do not get stars bler than this g-r",
        type=float,
        default=config.get("bluelim"),
    )
    parser.add_argument(
        "-l",
        "--maglim",
        help="Do not get stars fainter than this limit",
        type=float,
        default=config.get("maglim"),
    )
    parser.add_argument(
        "-L",
        "--brightlim",
        help="Do not get any less than this mag from the catalog to compare",
        type=float,
    )
    parser.add_argument(
        "-m",
        "--median",
        help="Give me just the median of zeropoints, no fitting",
        action="store_true",
    )
    parser.add_argument("-M", "--model", help="Read model from a file", type=str)
    parser.add_argument(
        "-n",
        "--nonlin",
        help="CCD is not linear, apply linear correction on mag",
        action="store_true",
    )
    parser.add_argument("-p", "--plot", help="Produce plots", action="store_true")
    parser.add_argument(
        "-r", "--reject", help="No outputs for Reduced Chi^2 > value", type=float
    )
    parser.add_argument(
        "--select-best",
        action="store_true",
        default=config.get("select_best", None),
        help="Try to select the best filter for photometric fitting",
    )
    parser.add_argument(
        "-s",
        "--stars",
        action="store_true",
        default=config.get("stars", "False"),
        help="Output fitted numbers to a file",
    )
    parser.add_argument(
        "-S",
        "--sip",
        help="Order of SIP refinement for the astrometric solution (0=disable)",
        type=int,
    )
    parser.add_argument(
        "-t", "--fit-terms", help="Comma separated list of terms to fit", type=str
    )
    parser.add_argument(
        "-T",
        "--trypar",
        type=str,
        help="Terms to examine to see if necessary (and include in the fit if they are)",
    )
    parser.add_argument(
        "-u",
        "--autoupdate",
        action="store_true",
        help="Update .det if .fits is newer",
        default=config.get("autoupdate", "False"),
    )
    parser.add_argument("-U", "--terms", help="Terms to fit", type=str)
    parser.add_argument(
        "-w", "--weight", action="store_true", help="Produce weight image"
    )
    parser.add_argument("-W", "--save-model", help="Write model into a file", type=str)
    parser.add_argument(
        "-x",
        "--fix-terms",
        help="Comma separated list of terms to keep fixed",
        type=str,
    )
    parser.add_argument(
        "-y",
        "--fit-xy",
        action="store_true",
        help="Fit xy tilt for each image separately (i.e. terms PX/PY)",
    )
    parser.add_argument(
        "-z", "--refit-zpn", action="store_true", help="Refit the ZPN radial terms"
    )
    parser.add_argument(
        "-Z", "--szp", action="store_true", help="use SZP while fitting astrometry"
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=config.get("verbose", "False"),
        help="Print debugging info",
    )
    #   parser.add_argument("files", help="Frames to process", nargs='+', action='extend', type=str)
    parser.add_argument("files", nargs="+", help="Frames to process")

    # Parse remaining arguments
    args = parser.parse_args(remaining_argv)

    # Convert string 'True'/'False' to boolean for action="store_true" arguments
    for arg in [
        "astrometry",
        "guessbase",
        "johnson",
        "verbose",
        "makak",
        "fsr",
        "select_best",
    ]:
        setattr(args, arg, str(getattr(args, arg)).lower() == "true")

    return args


# Example usage
if __name__ == "__main__":
    options = parse_arguments()
    print(options)
