"""Numeric input checks shared by training and experiment entry points."""
from __future__ import annotations

import math


def is_finite_number(value) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        # Python integers can exceed the range of floating-point optimizers.
        return False


def is_torch_seed(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and -(2**63) <= value < 2**64
