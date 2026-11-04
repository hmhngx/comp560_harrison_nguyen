"""A Monte Carlo prior for E4 built from the throwaway calibration (book seed 100, training seeds 100 and 101). Not evidence.

It produced the ranges in `phonebook/preregistration/e4_predictions.json`. It is a prior, not a test: the model is a
guess fitted to 57 calibration cells.

Model: for each width d, the first epoch with 100% exact recall of n entries is lognormal around a curve m_d(n) that is
piecewise linear in (log n, log epochs) through the calibration medians (a run that stalled below 100% at epoch 1000 is
treated as completing at 1500, a guess, and one that was nowhere close at 3000; the ranges are wide because of it). Each
seed's run gets independent lognormal noise of width sigma. A rung passes at budget E if at least 2 of 3 seeds finish by E;
the capacity at E is the largest rung with every smaller rung passing (the rule in analyze_capacity.py).

    python phonebook/calibration/e4_montecarlo.py [--sigma 0.4] [--trials 20000] [--seed 7]
"""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path
import random
import statistics

LADDERS = {8: [10, 14, 20, 28, 40], 16: [10, 20, 40, 57, 80, 113, 160, 226],
           32: [10, 20, 40, 80, 113, 160, 226, 320, 453], 64: [10, 20, 40, 80, 160, 226, 320, 453, 640, 905]}
BUDGETS = [100, 300, 500, 1000]
STALL = 1500.0           # assumed completion epoch of a run that sat at 0.5 to 1.0 of its entries at epoch 1000
DEFAULT_RESULTS = Path(__file__).with_name("e4_calibration_results.jsonl")


def load_results(path: Path) -> dict:
    """The longest run per (width, entries, training seed)."""
    best: dict = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (row["d"], row["n"], row["seed"])
        if key not in best or row["epochs"] > best[key]["epochs"]:
            best[key] = row
    return best


def observed(best: dict, d: int) -> dict:
    """{n: median completion epoch}; a run without a perfect epoch counts as STALL (close) or 2 x STALL (far); a short run says nothing."""
    out = collections.defaultdict(list)
    for (width, n, _), row in best.items():
        if width != d:
            continue
        if row["first_perfect"] is not None:
            out[n].append(float(row["first_perfect"]))
        elif row["epochs"] >= 1000 and row["final_train_seq"] > 0.5:
            out[n].append(STALL)
        elif row["epochs"] >= 1000:
            out[n].append(STALL * 2)
    return {n: statistics.median(v) for n, v in out.items()}


def curve(best: dict, d: int):
    """Median completion epoch as a function of n: piecewise linear in (log n, log epochs), never decreasing."""
    points = sorted(observed(best, d).items())
    xs, ys = [math.log(n) for n, _ in points], [math.log(t) for _, t in points]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1])

    def m(n):
        x = math.log(n)
        if x <= xs[0]:
            return math.exp(ys[0])
        for i in range(1, len(xs)):
            if x <= xs[i]:
                w = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
                return math.exp(ys[i - 1] + w * (ys[i] - ys[i - 1]))
        slope = max((ys[-1] - ys[-2]) / (xs[-1] - xs[-2]), 1.0)       # beyond the last rung: at least linear in n
        return math.exp(ys[-1] + slope * (x - xs[-1]))
    return m


def capacity_from_times(times: dict, ladder: list[int], budget: int) -> int:
    """times: {rung: sorted completion epochs of the 3 seeds}. The second-fastest seed finishing by the budget means at least 2 of 3 passed."""
    cap = 0
    for n in ladder:
        if times[n][1] <= budget:
            cap = n
        else:
            break
    return cap


def simulate(best: dict, d: int, sigma: float, trials: int = 20000, seed: int = 7):
    rng = random.Random(seed)
    m = curve(best, d)
    caps = {b: collections.Counter() for b in BUDGETS}
    boundary = collections.Counter()
    for _ in range(trials):
        times = {n: sorted(m(n) * math.exp(sigma * rng.gauss(0, 1)) for _ in range(3)) for n in LADDERS[d]}
        result = {b: capacity_from_times(times, LADDERS[d], b) for b in BUDGETS}
        for b in BUDGETS:
            caps[b][result[b]] += 1
        top_rung = LADDERS[d][-1]
        boundary["ceiling" if result[1000] == top_rung else "none" if result[1000] == 0 else "stable" if result[500] == result[1000] else "moving"] += 1
    return caps, boundary, m


def report(best: dict, sigma: float, trials: int, seed: int) -> str:
    lines = [f"sigma {sigma}"]
    for d in (8, 16, 32, 64):
        caps, boundary, m = simulate(best, d, sigma, trials, seed)
        lines.append(f"\n=== d={d}  curve (median first-perfect epoch): " + ", ".join(f"{n}:{m(n):.0f}" for n in LADDERS[d]))
        for b in BUDGETS:
            total = sum(caps[b].values())
            items = sorted(caps[b].items())
            mode = max(caps[b], key=caps[b].get)
            cum, lo, hi = 0, None, None
            for n, c in items:
                cum += c
                if lo is None and cum / total >= 0.1:
                    lo = n
                if hi is None and cum / total >= 0.9:
                    hi = n
            lines.append(f"  budget {b:4d}: mode {mode:4d}  80% range [{lo}, {hi}]  " + "  ".join(f"{n}:{c / total:.2f}" for n, c in items if c / total >= 0.04))
        lines.append("  boundary status: " + str({k: round(v / sum(boundary.values()), 2) for k, v in boundary.items()}))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--sigma", type=float, default=0.25, help="lognormal noise per seed (0.25 and 0.4 were used)")
    parser.add_argument("--trials", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    print(report(load_results(args.results), args.sigma, args.trials, args.seed))


if __name__ == "__main__":
    main()
