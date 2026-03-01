"""Pytest setup for archive optimizer tests.

Ensures local optimizer modules (e.g. gradient_refiner.py) are importable when
pytest runs from the repository root.
"""
from __future__ import annotations

import sys
from pathlib import Path


OPTIMIZER_DIR = Path(__file__).resolve().parents[1]
if str(OPTIMIZER_DIR) not in sys.path:
    sys.path.insert(0, str(OPTIMIZER_DIR))
