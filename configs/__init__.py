"""
AFN-DeSeg Configuration.

This module provides configuration management.
Default configuration is in default_config.yaml.
"""

import yaml
from pathlib import Path


def load_config(config_path: str = None) -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, loads default config.

    Returns:
        Configuration dictionary.
    """
    if config_path is None:
        config_path = Path(__file__).parent / 'default_config.yaml'
    else:
        config_path = Path(config_path)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


__all__ = ['load_config']
