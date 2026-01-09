"""Path helpers for flypca."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

ENV_REPO_ROOT = "FLYBEHAVIOR_RESPONSE_REPO_ROOT"
ENV_CONFIG_PATH = "FLYPCA_CONFIG_PATH"
ENV_OUTPUTS_DIR = "FLYBEHAVIOR_RESPONSE_OUTPUTS_DIR"


def find_repo_root(start: Path | None = None) -> Path:
    """Discover the repository root by walking up for pyproject.toml or .git."""

    start = (start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() or (candidate / ".git").exists():
            return candidate
    module_path = Path(__file__).resolve()
    for candidate in (module_path, *module_path.parents):
        if (candidate / "pyproject.toml").is_file() or (candidate / ".git").exists():
            return candidate
    return start


def repo_root() -> Path:
    env = os.environ.get(ENV_REPO_ROOT)
    if env:
        return Path(env).expanduser().resolve()
    return find_repo_root()


def outputs_dir() -> Path:
    env = os.environ.get(ENV_OUTPUTS_DIR)
    if env:
        return Path(env).expanduser().resolve()
    return repo_root() / "outputs"


def first_existing(candidates: Iterable[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def default_config_path() -> Path | None:
    env = os.environ.get(ENV_CONFIG_PATH)
    if env:
        return Path(env).expanduser().resolve()

    root = repo_root()
    return first_existing(
        [
            root / "config" / "default.yaml",
            root / "config" / "example.yaml",
            root / "configs" / "default.yaml",
        ]
    )
