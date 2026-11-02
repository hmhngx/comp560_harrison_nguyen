"""Load the predictions a researcher wrote down before any result existed, to embed them in a locked spec.

The file is a strict JSON object (no NaN or Infinity); its contents are copied into the spec's
decision_rules unchanged, so what was committed is exactly what was written.
"""
from __future__ import annotations

import json
from pathlib import Path


def load_predictions(path) -> dict:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"cannot read predictions file {path}: {error}") from error

    def refuse(constant):
        raise ValueError(f"predictions must be strict JSON, found {constant}")

    try:
        data = json.loads(text, parse_constant=refuse)
    except json.JSONDecodeError as error:
        raise ValueError(f"predictions file {path} is not valid JSON: {error}") from error
    if not isinstance(data, dict) or not data:
        raise ValueError(f"predictions file {path} must hold a non-empty JSON object")
    return data
