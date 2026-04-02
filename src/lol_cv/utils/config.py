"""Configuration loading from YAML files with environment variable overrides."""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CONFIG = _PROJECT_ROOT / "configs" / "default.yaml"


def load_config(config_path: str = None) -> dict:
    """Load pipeline configuration from a YAML file.

    API keys are NOT stored in the returned dict — each module reads
    them directly from environment variables to avoid accidental logging.

    Args:
        config_path: Path to a YAML config file. Defaults to configs/default.yaml.

    Returns:
        Configuration dictionary.
    """
    load_dotenv(_PROJECT_ROOT / ".env")

    path = Path(config_path) if config_path else _DEFAULT_CONFIG
    with open(path) as f:
        config = yaml.safe_load(f)

    return config
