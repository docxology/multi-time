"""
Structured logging utilities for multi-time.

Provides consistent, configurable logging across all modules with
both console and optional file output.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_CONFIGURED = False

LOG_FORMAT = "%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: str = "INFO",
    log_file: str | Path | None = None,
    force: bool = False,
) -> None:
    """Configure the root logger for multi-time.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional path to write logs to file.
        force: Re-configure even if already configured.
    """
    global _CONFIGURED  # noqa: PLW0603
    if _CONFIGURED and not force:
        return

    root_logger = logging.getLogger("multi_time")
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    root_logger.addHandler(console_handler)

    # File handler
    if log_file is not None:
        file_path = Path(log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(file_path), encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
        root_logger.addHandler(file_handler)

    _CONFIGURED = True
    root_logger.info("Logging configured: level=%s, file=%s", level, log_file)


def get_logger(name: str) -> logging.Logger:
    """Get a child logger under the multi_time namespace.

    Args:
        name: Logger name (typically __name__ of calling module).

    Returns:
        Configured Logger instance.
    """
    if name.startswith("multi_time"):
        return logging.getLogger(name)
    return logging.getLogger(f"multi_time.{name}")
