"""Configuration loading from YAML files with environment variable overrides."""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CONFIG = _PROJECT_ROOT / "configs" / "default.yaml"


def load_config(config_path: str = None) -> dict:
    """Load pipeline configuration from a YAML file.

    Args:
        config_path: Path to a YAML config file. Defaults to configs/default.yaml.

    Returns:
        Configuration dictionary.
    """
    load_dotenv(_PROJECT_ROOT / ".env")

    path = Path(config_path) if config_path else _DEFAULT_CONFIG
    with open(path) as f:
        config = yaml.safe_load(f)

    # Inject API keys from environment
    config.setdefault("keys", {})
    config["keys"]["riot_api_key"] = os.getenv("RIOT_API_KEY")
    config["keys"]["gemini_api_key"] = os.getenv("GEMINI_API_KEY")

    return config
