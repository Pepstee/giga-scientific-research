#!/usr/bin/env python3
"""Compatibility launcher for modules/research/scripts/elicit_evidence.py."""

from pathlib import Path

_TARGET = Path(__file__).resolve().parents[1] / "modules/research/scripts/elicit_evidence.py"
__file__ = str(_TARGET)
exec(compile(_TARGET.read_bytes(), str(_TARGET), "exec"), globals(), globals())
