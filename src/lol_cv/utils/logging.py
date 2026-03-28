"""Logging setup for the pipeline."""

import logging
import sys


def setup_logger(name: str = "lol_cv", level: int = logging.INFO) -> logging.Logger:
    """Create a configured logger.

    Args:
        name: Logger name (dot-separated for hierarchy).
        level: Logging level.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)
    return logger
