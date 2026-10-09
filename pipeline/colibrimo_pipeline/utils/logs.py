"""Logging setup: console + combined log file + one log file per step.

Layout inside a run folder::

    logs/pipeline.log      everything (all steps, all tool output)
    logs/<step>.log        one step only
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

LOGGER_NAME = "colibrimo_pipeline"
_FILE_FORMAT = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s")


def setup_logging(log_dir: Path, verbose: bool = False) -> logging.Logger:
    """Configure the pipeline logger. Safe to call several times (e.g. in a notebook)."""
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(LOGGER_NAME)
    close_logging(logger)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(
        logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(console)

    combined = logging.FileHandler(log_dir / "pipeline.log", mode="a", encoding="utf-8")
    combined.setLevel(logging.DEBUG)
    combined.setFormatter(_FILE_FORMAT)
    logger.addHandler(combined)
    return logger


def close_logging(logger: logging.Logger) -> None:
    """Remove and close all handlers (releases log files, important on Windows)."""
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


@contextmanager
def step_log_file(logger: logging.Logger, path: Path) -> Iterator[None]:
    """Also write everything logged inside this block to ``path``."""
    handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(_FILE_FORMAT)
    logger.addHandler(handler)
    try:
        yield
    finally:
        logger.removeHandler(handler)
        handler.close()
