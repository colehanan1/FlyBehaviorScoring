"""Tests for CLI utility helpers."""

from pathlib import Path

import pytest

pytest.importorskip("typer")

from flypca.cli import _load_config_or_default


def test_load_config_or_default_prefers_explicit(tmp_path: Path) -> None:
    explicit = tmp_path / "explicit.yaml"
    explicit.write_text("foo: 1\n", encoding="utf-8")
    default = tmp_path / "default.yaml"
    default.write_text("foo: 2\n", encoding="utf-8")

    cfg = _load_config_or_default(explicit, default_path=default)
    assert cfg["foo"] == 1


def test_load_config_or_default_falls_back(tmp_path: Path) -> None:
    default = tmp_path / "default.yaml"
    default.write_text("bar: 3\n", encoding="utf-8")

    cfg = _load_config_or_default(None, default_path=default)
    assert cfg["bar"] == 3


def test_load_config_or_default_handles_missing(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"

    cfg = _load_config_or_default(None, default_path=missing)
    assert cfg == {}
