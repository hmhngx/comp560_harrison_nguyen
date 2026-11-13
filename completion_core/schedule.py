"""Learning-rate schedules for the trainer: a pure, testable description of how the rate changes per epoch.

One canonical string is used identically on the command line, in specs and in protocols:

    step:<m1>[,<m2>...]:<gamma>        for example  step:2000:0.1

Epochs are 1-based. Epoch e trains at lr * gamma ** k, where k is the number of milestones strictly below e. With
step:2000:0.1, epochs 1 to 2000 use lr exactly and epochs 2001 onward use lr * 0.1; the milestone epoch itself is not decayed.
"""
from __future__ import annotations

import math
import re

_MILESTONES = re.compile(r"[1-9][0-9]*(,[1-9][0-9]*)*")


def format_lr_schedule(schedule: dict) -> str:
    return f"step:{','.join(str(milestone) for milestone in schedule['milestones'])}:{schedule['gamma']!r}"


def parse_lr_schedule(text: str) -> dict:
    """Parse the canonical string; ValueError for anything else, including a non-canonical spelling of a valid schedule."""
    if not isinstance(text, str):
        raise ValueError("lr schedule must be a string")
    parts = text.split(":")
    if len(parts) != 3 or parts[0] != "step":
        raise ValueError(f"lr schedule must look like step:2000:0.1, got {text!r}")
    if not _MILESTONES.fullmatch(parts[1]):
        raise ValueError(f"lr schedule milestones must be positive integers separated by commas, got {parts[1]!r}")
    milestones = [int(item) for item in parts[1].split(",")]
    if any(later <= earlier for earlier, later in zip(milestones, milestones[1:])):
        raise ValueError("lr schedule milestones must be strictly increasing")
    try:
        gamma = float(parts[2])
    except ValueError:
        raise ValueError(f"lr schedule gamma must be a number, got {parts[2]!r}") from None
    if not math.isfinite(gamma) or gamma <= 0 or gamma > 1:
        raise ValueError("lr schedule gamma must be a finite number in (0, 1]")
    schedule = {"type": "step", "milestones": milestones, "gamma": gamma}
    if format_lr_schedule(schedule) != text:
        raise ValueError(f"lr schedule must be written canonically as {format_lr_schedule(schedule)!r}, got {text!r}")
    return schedule


def lr_factor(schedule: dict, epoch: int) -> float:
    """Multiplier on the base learning rate at a 1-based epoch."""
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
        raise ValueError("epoch must be an integer >= 1")
    decays = sum(1 for milestone in schedule["milestones"] if milestone < epoch)
    return schedule["gamma"] ** decays
