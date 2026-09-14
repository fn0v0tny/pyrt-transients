#!/usr/bin/python3

import configparser
import dataclasses
import json
import logging
import typing
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from pathlib import Path

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

    # --- Persistent-source handling (all default to the historical
    # behaviour; the recommended vetting configuration is documented in
    # README.md "Constant sources" and validated in FUTURE_IDEAS.md).
    #
    # unphotometered_match_is_new: what to do with a detection that HAS a
    # catalogue counterpart within the match radius but none of the matched
    # stars has usable photometry (e.g. USNO-B, which carries no Sloan
    # bands, or Gaia stars without BP/RP). True (historical) reports it as a
    # "new" source -- which, for USNO-B, means *every* detection in the
    # frame is "new" and the catalogue contributes nothing to the unanimity
    # rule. False reports it as matched-with-unknown-photometry, i.e. not a
    # candidate: the star exists, we just cannot say whether it changed.
    unphotometered_match_is_new: bool = True
    # With unphotometered_match_is_new=False, a match whose only photometry
    # is a plate/broad-band magnitude (USNO-B B/R/I, Gaia G without BP/RP)
    # still vetoes the detection -- unless the detection is brighter than
    # the brightest such magnitude among the matches by more than this many
    # magnitudes, in which case the catalogue entry cannot be the detected
    # source and it is reported as "brightening". 2.0 mag leaves room for
    # colour terms and photographic scatter; None restores the purely
    # positional veto (which lost the 14.9 mag GRB 250813B afterglow to a
    # 19.7 mag USNO-B star 2" away, FUTURE_IDEAS.md 2026-09-03).
    unphotometered_veto_max_brightening_mag: Optional[float] = 2.0
    # Per-catalogue floor on the adaptive match radius, in arcsec, to absorb
    # the catalogue's own astrometric error (USNO-B positions scatter by
    # ~0.5-1" against Gaia; the 1-px default floor is 1.2" on D50). Keys are
    # matched as substrings of the catalogue name, e.g. {"usno": 2.0}.
    catalog_match_floor_arcsec: Dict[str, float] = field(default_factory=dict)
    # For candidates typed "new", never let the light-curve variability
    # factors of compute_lightcurve_score_factor fall below 1, so a constant
    # uncatalogued source keeps its base score instead of being multiplied
    # by ~1e-3 (the injection result in FUTURE_IDEAS.md). Only safe once the
    # reference catalogues are positionally complete (gaia_full) and
    # unphotometered_match_is_new=False; otherwise every star missing from
    # a catalogue's quality-cut query becomes a high-scoring candidate.
    new_source_variability_floor: bool = False

    # --- Photometric error model for the light-curve statistics.
    # MAGERR_CALIB (SExtractor error + zeropoint error) is right for faint
    # stars and several times too small for bright ones: on the 15-field GRB
    # replay the epoch-to-epoch rms of unflagged 12-15 mag stars is
    # 0.02-0.04 mag (0.1 on a poor night) against quoted 0.003-0.01, while
    # faint stars scatter as quoted (FUTURE_IDEAS.md, 2026-09-04). The
    # light-curve chi^2 (mag_chi2_reduced, is_variable) therefore uses
    # err_eff^2 = MAGERR_CALIB^2 + floor^2. With magerr_floor_auto the floor
    # is measured from the bright constant stars of the campaign itself
    # (lightcurve.estimate_magnitude_error_floor); magerr_floor_mag is the
    # value used when that is off or cannot be measured (sparse field, < 3
    # epochs). 0 with auto off restores the raw chi^2. The weighted mean
    # magnitude and the quality_score are not affected.
    magerr_floor_auto: bool = True
    magerr_floor_mag: float = 0.05

    # --- Calibrated probability for the frontend:
    # p_real = sigmoid(intercept + slope * ln quality_score), a logistic
    # map of the score fitted on the 15-field GRB replay (real afterglows +
    # injected sources against everything else; Brier 0.10, reliable to
    # ~0.05 per bin -- FUTURE_IDEAS.md 2026-09-04). The constants belong to
    # a catalogue set: vetting catalogues (gaia_full + ATLAS + USNO-B with
    # the README options) 0.840 / 0.843; historical gaia + usno
    # -2.539 / 1.101 (there q=1 means p=0.07 because ~45 uncatalogued
    # stars per field share that score). None -> no p_real column.
    score_probability_intercept: Optional[float] = None
    score_probability_slope: Optional[float] = None

    # --- Isolation statistic: nearest_source_dist / nearby_sources are
    # computed against every catalogue entry, including stars far below the
    # frame limit. A detection 1-2" from a 20 mag Gaia star then gets the
    # isolation factor clip(d/10, 0, 1) ~ 0.1-0.2 and can fall under
    # min_quality in one catalogue (2% of injected sources lost this way,
    # FUTURE_IDEAS.md "Isolation penalty"). With a margin set, only entries
    # brighter than the frame's MAGLIMIT + margin count (entries with no
    # magnitude at all are kept). None = historical behaviour.
    isolation_max_mag_margin: Optional[float] = None
    
    # Move catalogue positions to the frame's epoch with the catalogue's
    # proper motions before matching. Gaia and ATLAS positions are for
    # 2015.5-2016, USNO-B for 2000: a star at 150 mas/yr has moved 2-4
    # arcsec since, beyond the identification radius, and turned up as a
    # persistent "new" source in every frame of every night.
    propagate_proper_motion: bool = True

    # Detections that SExtractor flags as saturated (FLAGS & 4) or that are
    # brighter than the frame's estimated saturation magnitude (peak pixel
    # at saturation_adu for a Gaussian star of the header FWHM and MAGZERO)
    # plus saturation_margin_mag count as "at least as bright as that
    # limit" when compared with a matched catalogue star, instead of as a
    # measurement: a catalogue star much fainter than the limit has
    # brightened (reported with the limit as magnitude difference), one
    # already brighter is consistent, and fading is never claimed. They are
    # still reported as new sources when no catalogue star matches, since a
    # bright GRB or nova saturates as well. The margin stays at 0: on the
    # 210619B fixture
    # the Gaussian estimate is 10.98 mag, the first flagged star is at
    # 11.26 and the afterglow at 11.43. The forced target row (NUMBER 0) is
    # never masked.
    reject_saturated: bool = True
    saturation_adu: float = 60000.0
    saturation_margin_mag: float = 0.0

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
    # Stack-only candidates and min_n_detections: the stack is one epoch, so
    # a source the stack reaches but single frames mostly do not can never
    # collect min_n_detections detections (GRB 190919B, tests/190919B: the
    # afterglow is the stack's top candidate and was still dropped). Such
    # candidates instead get forced aperture photometry in the frames the
    # stack was built from (detection/blind_multicatalog/forced.py) and are
    # admitted when that forced lightcurve shows a persistent source: SNR >=
    # stack_forced_snr in at least stack_forced_min_fraction of those
    # frames, with no single frame carrying more than
    # stack_forced_max_flux_fraction of the positive flux. The last test is
    # what rejects cosmic rays, hot pixels and glints averaged into the
    # stack (pyrt-combine has no per-pixel rejection): all their flux is in
    # one frame. On tests/190919B's 20-frame stack the afterglow reaches SNR
    # > 3 in 24/40 frames; the three single-frame flashes in 1/40 each.
    stack_forced_admission: bool = True
    stack_forced_snr: float = 3.0
    stack_forced_min_fraction: float = 0.3
    stack_forced_max_flux_fraction: float = 0.5
    stack_forced_min_frames: int = 5
    # At most this many stack-only candidates (by quality) are measured per
    # run -- bounds the cost when a stack is full of artefacts.
    stack_forced_max_candidates: int = 50
    # Forced points at or above this SNR make up the admitted candidate's
    # lightcurve (and so its n_detections and scores).
    stack_forced_lc_snr: float = 2.0
    # Aperture radius in units of the frame's FWHM.
    stack_forced_aperture_fwhm: float = 1.0


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
class FollowupConfig:
    """Configuration for candidate enrichment run on a strategy's output --
    today, the exposure-time recommendation (followup/exposure.py). Kept out
    of DetectionConfig deliberately: nothing here influences what is
    detected, only what is suggested afterwards."""
    exposure_enabled: bool = True
    # Follow-up aims for a solid measurement, not a marginal detection --
    # SNR 10 is magerr ~0.109.
    target_snr: float = 10.0
    # A lightcurve's latest point is used as the planning magnitude on its
    # own only if its error is at most this; a noisier one (admission goes
    # down to new_source_siglim=1.5, i.e. ~0.7 mag) is averaged with the
    # preceding points instead, since exposure time scales as ~10^(0.8*dm).
    max_planning_magerr: float = 0.2
    # Effective readout noise (electrons). Not a follow-up policy but a
    # camera constant, kept here because the ECSV header carries no RN
    # keyword; pairs with the fitted constants in followup/exposure.py
    # (APE, ZERO_OFFSET), which are likewise instrument-specific -- the
    # per-run `model_check` in the report is what tells you whether they
    # describe the camera actually in use.
    readout_noise_e: float = 8.0
    # Bracket the solver searches in. The upper bound doubles as the
    # "can this even be reached?" test -- beyond it the report says so
    # explicitly rather than returning an absurd number.
    min_exptime_s: float = 1.0
    max_exptime_s: float = 3600.0


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
    followup: FollowupConfig = field(default_factory=FollowupConfig)
    caching: CachingConfig = field(default_factory=CachingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Global pipeline settings
    base_data_dir: str = "/home/fnovotny/transient_work/"
    base_public_dir: Optional[str] = None  # Default to ~/public_html
    generate_frontend: bool = True
    parallel_processing: bool = False
    max_workers: int = 4
    # Observation grouping by pointing. The telescope's OBSID changes per
    # filter/observing block within one night, so one GRB campaign can be
    # split over several IDs (and an ID can be shared by two GRBs observed
    # the same night) -- FUTURE_IDEAS.md "Observation-ID fragmentation".
    # With a radius set, a new epoch whose frame centre (CTRRA/CTRDEC) lies
    # within that many arcmin of an existing observation directory's
    # pointing, and whose mid-time is within observation_grouping_max_gap_hours
    # of that observation's last epoch, joins it under the existing ID
    # instead of opening a new one. None keeps the raw OBSID (historical).
    observation_grouping_radius_arcmin: Optional[float] = None
    observation_grouping_max_gap_hours: float = 12.0
    
    # INI (configparser) support. Both directions are driven by the
    # dataclass fields themselves -- the hand-maintained key lists this
    # replaced had drifted to omit seven DetectionConfig fields (siglim,
    # new_source_siglim, catalogs, maglim_filter_multiplier, ...), so an INI
    # file silently could not set them. YAML goes through from_dict.
    _SECTIONS = ("detection", "frontend", "followup", "caching", "logging")

    @classmethod
    def from_file(cls, config_file: str) -> "PipelineConfig":
        """Load pipeline configuration from an INI file."""
        config_path = Path(config_file).expanduser()
        if not config_path.exists():
            logging.warning(f"Config file {config_file} not found, using defaults")
            return cls()

        # interpolation=None: logging.format legitimately contains %(asctime)s
        # etc., which ConfigParser's default BasicInterpolation tries to
        # expand as config references.
        config = configparser.ConfigParser(interpolation=None)
        config.read(config_path)

        pipeline_config = cls()
        for section_name in cls._SECTIONS:
            if section_name in config:
                _apply_ini_section(getattr(pipeline_config, section_name), config[section_name])
        if "global" in config:
            _apply_ini_section(pipeline_config, config["global"],
                               skip=set(cls._SECTIONS))
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

        # Update followup config
        if "followup" in config_data:
            fu_data = config_data["followup"]
            for key, value in fu_data.items():
                if hasattr(pipeline_config.followup, key):
                    setattr(pipeline_config.followup, key, value)

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
            if key not in ["detection", "frontend", "followup", "caching", "logging", "global"] and hasattr(pipeline_config, key):
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
        """Save pipeline configuration to an INI file (every field of every
        section, so from_file() reads back exactly what was written)."""
        config = configparser.ConfigParser(interpolation=None)
        for section_name in self._SECTIONS:
            config[section_name] = _ini_section_dict(getattr(self, section_name))
        config["global"] = _ini_section_dict(self, skip=set(self._SECTIONS))
        with open(Path(config_file).expanduser(), 'w') as f:
            config.write(f)


def _ini_section_dict(obj, skip=()) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for f in dataclasses.fields(obj):
        if f.name in skip:
            continue
        value = getattr(obj, f.name)
        if value is None:
            # Written explicitly, not omitted: an omitted key restores the
            # field's *default*, which is only equivalent to None for fields
            # that default to None. new_source_siglim (1.5) and
            # unphotometered_veto_max_brightening_mag (2.0) are Optional with
            # non-None defaults, so an explicit None came back as the default
            # and the round trip to_file/from_file promised above was a lie.
            # _coerce_ini_value reads "none" back as None for any Optional.
            out[f.name] = "none"
            continue
        if isinstance(value, (list, dict)):
            out[f.name] = json.dumps(value)
        else:
            out[f.name] = str(value)
    return out


def _apply_ini_section(obj, section, skip=()) -> None:
    """Set each dataclass field of `obj` present in the configparser
    `section`, converting from string by the field's declared type."""
    for f in dataclasses.fields(obj):
        if f.name in skip or f.name not in section:
            continue
        raw = section.get(f.name)
        try:
            setattr(obj, f.name, _coerce_ini_value(raw, f.type, getattr(obj, f.name)))
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            logging.warning(f"config: ignoring {type(obj).__name__}.{f.name}={raw!r}: {exc}")


def _coerce_ini_value(raw: str, field_type, current):
    origin = typing.get_origin(field_type)
    args = typing.get_args(field_type)
    if origin is typing.Union:  # Optional[X]
        inner = [a for a in args if a is not type(None)]
        if raw.strip().lower() in ("", "none", "null"):
            return None
        return _coerce_ini_value(raw, inner[0], current)
    if origin in (list, dict) or field_type in (list, dict):
        text = raw.strip()
        if origin is list and not text.startswith("["):
            return [item.strip() for item in text.split(",") if item.strip()]
        return json.loads(text)
    if field_type is bool:
        text = raw.strip().lower()
        if text in ("1", "true", "yes", "on"):
            return True
        if text in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"not a boolean: {raw!r}")
    if field_type is int:
        return int(raw)
    if field_type is float:
        return float(raw)
    if field_type is str:
        return raw
    return raw

