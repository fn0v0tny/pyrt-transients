"""Config-file loading with YAML support."""

import logging
from pathlib import Path

import yaml

from pyrt_transient.config_trans import PipelineConfig


def load_config_with_yaml_support(config_file):
    """Load configuration with YAML support."""
    config_path = Path(config_file)

    # Check if it's a YAML file
    if config_path.suffix.lower() in ['.yaml', '.yml']:
        try:
            with open(config_path, 'r') as f:
                yaml_data = yaml.safe_load(f)

            # Create PipelineConfig from YAML data
            return PipelineConfig.from_dict(yaml_data)
        except Exception as e:
            logging.error(f"Failed to load YAML config from {config_file}: {e}")
            raise
    else:
        # Use existing from_file method for non-YAML files
        return PipelineConfig.from_file(config_file)
