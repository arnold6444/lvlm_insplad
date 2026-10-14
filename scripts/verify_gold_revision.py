"""Verify the exact clean Git revision before importing or executing gold code."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path


def verify(root: Path, expected: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", expected):
        raise ValueError("gold execution requires a full immutable commit SHA")
    actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if actual != expected:
        raise RuntimeError("gold code revision differs from the pinned commit; use a fresh runtime")
    dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True)
    if dirty:
        raise RuntimeError("tracked gold code was modified after checkout")
    return actual
