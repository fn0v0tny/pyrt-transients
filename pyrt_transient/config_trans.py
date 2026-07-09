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
    min_n_detections: int = 5
    min_catalogs_fraction: float = 1.0
    min_quality: float = 0.2
    radius_check: float = 20.0
    filter_pattern: str = "r"

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
