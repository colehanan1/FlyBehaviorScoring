"""Shared path helpers for repo scripts."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable

ENV_REPO_ROOT = "FLYBEHAVIOR_RESPONSE_REPO_ROOT"
ENV_OUTPUTS_DIR = "FLYBEHAVIOR_RESPONSE_OUTPUTS_DIR"
ENV_CONFIG_DIR = "FLYBEHAVIOR_RESPONSE_CONFIG_DIR"
ENV_DATA_DIR = "FLYBEHAVIOR_RESPONSE_DATA_DIR"
ENV_FLYPCA_CONFIG_PATH = "FLYPCA_CONFIG_PATH"


def find_repo_root(start: Path | None = None) -> Path:
    start = (start or Path(__file__).resolve()).resolve()
    for candidate in (start, *start.parents):
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


def artifacts_dir() -> Path:
    base = outputs_dir() / "artifacts"
    legacy = repo_root() / "artifacts"
    if legacy.exists() and not base.exists():
        return legacy
    return base


def config_dir() -> Path:
    env = os.environ.get(ENV_CONFIG_DIR)
    if env:
        return Path(env).expanduser().resolve()
    return repo_root() / "config"


def data_dir() -> Path:
    env = os.environ.get(ENV_DATA_DIR)
    if env:
        return Path(env).expanduser().resolve()
    return repo_root() / "data"


def first_existing(candidates: Iterable[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return repo_root() / path


def resolve_flypca_config() -> Path | None:
    env = os.environ.get(ENV_FLYPCA_CONFIG_PATH)
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


def ensure_src_on_path() -> Path:
    src_dir = repo_root() / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    return src_dir
