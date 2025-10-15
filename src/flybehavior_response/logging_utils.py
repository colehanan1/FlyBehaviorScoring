"""Logging utilities for flybehavior_response."""

from __future__ import annotations

import logging
from typing import Optional

_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def get_logger(name: str, verbose: bool = False) -> logging.Logger:
    """Return a configured logger.

    Args:
        name: Logger name.
        verbose: If True, set level to DEBUG; otherwise INFO.
    """
    logger = logging.getLogger(name)
    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)
    logger.propagate = False
    return logger


def set_global_logging(verbose: bool = False, level: Optional[int] = None) -> None:
    """Configure root logging level."""
    logging.basicConfig(level=(level or (logging.DEBUG if verbose else logging.INFO)), format=_LOG_FORMAT)
