#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(Path(__file__).resolve().parent / "scripts/analysis/analyze_model_errors.py", run_name="__main__")
