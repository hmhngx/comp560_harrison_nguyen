"""Report on the E6 calibration (scratch, NOT evidence): 4000-epoch runs of the first rung that failed in E4, one per width,
on the throwaway book (seed 100) with training seeds 100 and 101. The locked E6 predictions were drafted from these eight runs.

    python phonebook/calibration/e6_calibration_report.py [--results FILE]

Prints one line per run (first epoch with 100% exact recall, logged epochs at 100% out of 400, best and final entries
recalled) and, per width, the window means pooled over the two seeds and the trend under the E6 rule (rising if the gain
from epochs 1001-2000 to 3001-4000 is at least +0.005, falling if at most -0.005, else flat).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from phonebook.analyze_extension import pooled_window_mean, trend

RESULTS = Path(__file__).with_name("e6_calibration_results.jsonl")
EARLY, LATE = (1000, 2000), (3000, 4000)
ROUNDING = 5.1e-5  # the curves store each accuracy to four decimals


def load_rows(path: Path = RESULTS) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def entries(row: dict, accuracy: float) -> int:
    """Entries recalled exactly: the curve stores the fraction to four decimals, the entry count is its product with n."""
    count = round(accuracy * row["n"])
    if abs(count / row["n"] - accuracy) > ROUNDING:
        raise ValueError(f"accuracy {accuracy} is not a whole number of the {row['n']} entries")
    return count


def run_points(row: dict) -> list[dict]:
    return [{"epoch": epoch, "A": entries(row, accuracy)} for epoch, accuracy in row["curve"]]


def perfect_epochs(row: dict) -> int:
    """Logged epochs at which every entry was recalled."""
    return sum(1 for _, accuracy in row["curve"] if entries(row, accuracy) == row["n"])


def width_summary(rows: list[dict]) -> dict:
    runs = [{"points": run_points(row), "n": row["n"]} for row in rows]
    early, late = pooled_window_mean(runs, *EARLY), pooled_window_mean(runs, *LATE)
    return {"early": early, "late": late, "gain": late - early, "trend": trend(late - early)}


def report(rows: list[dict]) -> str:
    lines = ["run: first epoch at 100% | logged epochs at 100% of 400 | best / final entries recalled"]
    for row in sorted(rows, key=lambda r: (r["d"], r["seed"])):
        best = max(entries(row, accuracy) for _, accuracy in row["curve"])
        final = entries(row, row["curve"][-1][1])
        first = row["first_perfect"] if row["first_perfect"] is not None else "never"
        lines.append(f"d={row['d']:<2} n={row['n']:<3} seed={row['seed']}: {first} | {perfect_epochs(row)} | {best}/{row['n']} / {final}/{row['n']}")
    lines.append("pooled over seeds: mean accuracy in epochs 1001-2000 -> 3001-4000, gain, trend")
    for d in sorted({row["d"] for row in rows}):
        s = width_summary([row for row in rows if row["d"] == d])
        lines.append(f"d={d:<2}: {float(s['early']):.4f} -> {float(s['late']):.4f}, gain {float(s['gain']):+.4f}, {s['trend']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args(argv)
    print(report(load_rows(args.results)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
