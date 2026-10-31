"""Logging and run tracking helpers."""

from __future__ import annotations

import logging
import subprocess

from dengue.utils.config import ROOT


def get_logger(name: str) -> logging.Logger:
    """Return a logger with a consistent format."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    return logging.getLogger(name)


def git_hash() -> str:
    """Return the current git commit hash, or 'unknown' outside a repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
