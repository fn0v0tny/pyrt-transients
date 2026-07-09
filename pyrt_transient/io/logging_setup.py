"""Pipeline logging setup."""

import logging
from pathlib import Path


def setup_pipeline_logging(config, observation_id):
    """Setup comprehensive logging for the pipeline."""
    # Create logs directory
    log_dir = Path(config.base_data_dir) / "logs"
    log_dir.mkdir(exist_ok=True)

    # Setup file handlers
    log_file = log_dir / f"pipeline_{observation_id}.log"
    debug_log_file = log_dir / f"pipeline_{observation_id}_debug.log"

    # Get root logger and clear existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers = []

    # Set logging level
    if config.logging.level == "DEBUG":
        root_logger.setLevel(logging.DEBUG)
    else:
        root_logger.setLevel(logging.INFO)

    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s')

    # File handler for INFO+ messages
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # File handler for DEBUG+ messages
    debug_handler = logging.FileHandler(debug_log_file)
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(formatter)
    root_logger.addHandler(debug_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    return logging.getLogger('pipeline_magic')
