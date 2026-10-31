"""Config loading and project paths."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


def load_dotenv(env_file: Path = ROOT / ".env") -> None:
    """Set variables from a local, gitignored ``.env`` file without overriding the environment."""
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML config. Relative paths in ``paths`` are resolved against the repo root."""
    load_dotenv()
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(cfg_path)
    cfg["paths"] = {k: str((ROOT / v).resolve()) for k, v in cfg["paths"].items()}
    return cfg


def path(cfg: dict[str, Any], key: str, *parts: str) -> Path:
    """Return ``cfg.paths[key] / parts`` and create the parent directory."""
    p = Path(cfg["paths"][key]).joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
