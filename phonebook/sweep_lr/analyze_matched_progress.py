"""Matched-progress analysis for the L2-SP lambda=0 vs lambda=0.1 comparison
in this directory's README (Part 4).

Comparing two runs at the SAME epoch number is confounded: L2-SP's penalty
competes with the task loss, so a higher lambda could simply be learning
Phase B more slowly, in which case "retains more of phonebook A at epoch N"
would just mean "less far along," not a property of L2-SP itself. This
script controls for that by matching each lambda=0.1 epoch to the lambda=0
epoch with the closest Phase-B training progress (Train Loss, cross-checked
against Train Token Acc) instead of the same epoch number, and reports the
Val Seq Acc delta at that matched point.

Usage:
    python phonebook/sweep_lr/analyze_matched_progress.py

Reads the two --accuracy-interval 1 (full per-epoch resolution) runs already
committed in seeded/lr0.0001_ep100_seed42_l2sp{0,0.1}_finegrained/.
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE_CSV = ROOT / "phonebook/sweep_lr/seeded/lr0.0001_ep100_seed42_l2sp0_finegrained/metrics.csv"
L2SP_CSV = ROOT / "phonebook/sweep_lr/seeded/lr0.0001_ep100_seed42_l2sp0.1_finegrained/metrics.csv"


def load(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in r:
            if k != "Epoch":
                r[k] = float(r[k]) if r[k] != "N/A" else None
        r["Epoch"] = int(r["Epoch"])
    return rows


def matched_deltas(base: list[dict], l2sp: list[dict], match_key: str) -> list[tuple]:
    """For each l2sp-run row, find the base-run row with the closest
    match_key value, and return (l2sp_epoch, base_epoch, l2sp_val, base_val, delta)."""
    out = []
    for s in l2sp:
        target = s[match_key]
        closest = min(base, key=lambda r: abs(r[match_key] - target))
        delta = s["Val Seq Acc"] - closest["Val Seq Acc"]
        out.append((s["Epoch"], closest["Epoch"], s[match_key], closest[match_key], delta))
    return out


def main() -> None:
    base = load(BASE_CSV)
    l2sp = load(L2SP_CSV)
    assert len(base) == 100 and len(l2sp) == 100, (
        f"expected 100 per-epoch rows in each file, got {len(base)}, {len(l2sp)} "
        "-- were these regenerated with --accuracy-interval 1?"
    )

    for match_key in ["Train Loss", "Train Token Acc"]:
        print(f"\n=== Matched on {match_key} ===")
        deltas = matched_deltas(base, l2sp, match_key)
        nonzero = [d for d in deltas if abs(d[4]) > 1e-9]
        print(f"Nonzero-delta epochs: {len(nonzero)} / 100")
        for l2sp_ep, base_ep, l2sp_v, base_v, delta in nonzero:
            sign = "+" if delta > 0 else ""
            print(
                f"  l2sp ep {l2sp_ep:3d} ({match_key}={l2sp_v:.4f}) <-> "
                f"base ep {base_ep:3d} ({match_key}={base_v:.4f})  delta={sign}{delta * 100:.0f}pp"
            )


if __name__ == "__main__":
    main()
