#!/usr/bin/env python3
"""Repository entry point for the deterministic scientific-review pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from modules.research.core.scientific_pipeline_cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
