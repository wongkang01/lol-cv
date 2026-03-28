"""Path helpers for data directories and model files."""

from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def get_data_dir(subdir: str = None) -> Path:
    """Return a path under the project data/ directory.

    Args:
        subdir: Optional subdirectory (e.g. 'raw', 'processed', 'models').

    Returns:
        Absolute path to the data directory.
    """
    base = _PROJECT_ROOT / "data"
    return base / subdir if subdir else base


def get_model_path(filename: str) -> Path:
    """Return the path to a model weights file under data/models/."""
    return get_data_dir("models") / filename


def ensure_dir(path: Path) -> Path:
    """Create directory (and parents) if it doesn't exist. Returns the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path
