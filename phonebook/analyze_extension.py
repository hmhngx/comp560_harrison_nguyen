"""E6 analysis: does a longer run let the first failing rung of each width reach 100 percent?

E6 retrains, unchanged except for the length (4000 epochs instead of 1000), the first rung of each width that failed
in E4 on the same three book seeds. This reads the trainer's own logs and reports, per width:

- how many of the 3 seeds are perfect by each budget, whether the rung passes, and the first budget at which it does;
- how many seeds ever reached 100 percent and how many are still there at the last epoch (100 percent at the wide
  widths is often a one-epoch hit that falls back);
- the pooled window means of the training accuracy and their gain, with the trend rule below;
- whether the first 1000 epochs of every run equal the E4 run point for point (they must: training is deterministic
  and a longer run only continues a shorter one).

    python phonebook/analyze_extension.py --root ROOT [--root ROOT ...] --earlier-root E4ROOT [--earlier-root ...] \\
        --budgets 1000 2000 3000 4000 --out phonebook/results

Rules (fixed before any E6 run, see phonebook/preregistration/e6_predictions.json):
- A run is *perfect by budget E* if every training entry is recalled exactly at some logged epoch <= E; a rung passes
  at E if at least 2 of its 3 book seeds are perfect by E (the same code as the E4 capacity rule, one rung per width).
- Accuracy at a logged epoch is entries recalled / entries. A window mean covers the logged epochs after the lower
  edge up to and including the upper edge, pooled over the seeds with equal weight. The gain is the mean over
  epochs 3001-4000 minus the mean over epochs 1001-2000; it is computed in exact fractions.
- Trend: rising if the gain is at least +0.005, falling if it is at most -0.005, flat otherwise.

Options added for E7 (without them the output is byte for byte what it was for E6):
- --prefix-epochs N compares the first N epochs (default 1000) with the earlier runs and names the two JSON keys
  first_N_match_by_seed and first_N_matches_earlier; --earlier-label names those runs in the printed summary.
- --epochs-at-100 adds "epochs_at_100" to every width: per seed, how many logged epochs have every entry recalled, over the
  whole run and inside the late window (3001-4000), with the number of logged epochs they are counted among.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from phonebook.analyze_capacity import SCENARIO, capacity, cell_summary
from phonebook.report_experiments import load_runs

THRESHOLD = Fraction(5, 1000)
EARLY = (1000, 2000)
LATE = (3000, 4000)
PREFIX = 1000


# ------------------------------------------------------------------ windows and trend


def window_points(points: list[dict], lo: int, hi: int) -> list[dict]:
    return [point for point in points if lo < point["epoch"] <= hi]


def window_mean(points: list[dict], n: int, lo: int, hi: int) -> Fraction:
    """Mean fraction of the n entries recalled exactly over the logged epochs in (lo, hi]."""
    inside = window_points(points, lo, hi)
    if not inside:
        raise ValueError(f"no logged epoch in ({lo}, {hi}]")
    if max(point["epoch"] for point in inside) != hi:
        raise ValueError(f"the run never logged epoch {hi}, so the window ({lo}, {hi}] is incomplete")
    if any(point["A"] is None for point in inside):
        raise ValueError(f"a logged epoch in ({lo}, {hi}] has no recall count")
    return Fraction(sum(point["A"] for point in inside), n * len(inside))


def pooled_window_mean(runs: list[dict], lo: int, hi: int) -> Fraction:
    """Mean over the seeds of each seed's window mean (equal weight per seed)."""
    epochs = {tuple(point["epoch"] for point in window_points(run["points"], lo, hi)) for run in runs}
    if len(epochs) != 1:
        raise ValueError(f"the seeds logged different epochs in ({lo}, {hi}], so their window means cannot be pooled")
    return sum((window_mean(run["points"], run["n"], lo, hi) for run in runs), Fraction(0)) / len(runs)


def trend(gain: Fraction, threshold: Fraction = THRESHOLD) -> str:
    if gain >= threshold:
        return "rising"
    if gain <= -threshold:
        return "falling"
    return "flat"


# ------------------------------------------------------------------ the first epochs against the earlier run


def prefix_equal(new: list[dict], old: list[dict], up_to: int) -> bool:
    """The logged (epoch, A, B) points up to and including `up_to` are identical, and both runs reached `up_to`."""
    def cut(points):
        return [(point["epoch"], point["A"], point["B"]) for point in points if point["epoch"] <= up_to]

    a, b = cut(new), cut(old)
    return bool(a) and a[-1][0] == up_to and a == b


# ------------------------------------------------------------------ how long a run holds 100 percent


def epochs_at_100(points: list[dict], n: int, window: tuple[int, int] | None = None) -> int:
    """How many logged epochs have every one of the n entries recalled; `window` = (lo, hi] limits the count to those epochs."""
    inside = points if window is None else window_points(points, *window)
    if any(point["A"] is None for point in inside):
        raise ValueError("a logged epoch has no recall count")
    return sum(point["A"] == n for point in inside)


def held_report(runs: list[dict], late: tuple[int, int]) -> dict:
    """Per seed, the logged epochs at 100 percent over the whole run and inside the late window, and what they are counted among."""
    label = f"{late[0] + 1}_{late[1]}"
    runs = sorted(runs, key=lambda run: run["seed"])
    logged = {(len(run["points"]), len(window_points(run["points"], *late))) for run in runs}
    if len(logged) != 1:
        raise ValueError("the seeds have different numbers of logged epochs, so no single denominator describes them")
    (everything, inside), = logged
    return {"logged": {"all": everything, label: inside},
            "by_seed": {str(run["seed"]): {"all": epochs_at_100(run["points"], run["n"]),
                                           label: epochs_at_100(run["points"], run["n"], late)} for run in runs}}


# ------------------------------------------------------------------ one width


def width_report(runs: list[dict], budgets: tuple[int, ...], *, min_seeds: int = 2, earlier: dict | None = None,
                 early: tuple[int, int] = EARLY, late: tuple[int, int] = LATE, prefix: int = PREFIX,
                 held: bool = False) -> dict:
    """runs: dicts with seed, n and the logged points of one width; earlier: {(seed, n): points} of the E4 runs."""
    rungs = {run["n"] for run in runs}
    if len(rungs) != 1:
        raise ValueError(f"a width runs one rung only, found {sorted(rungs)}")
    n = rungs.pop()
    runs = sorted(runs, key=lambda run: run["seed"])
    cells = [{"n": n, "seed": run["seed"], **cell_summary(run["points"], n, budgets)} for run in runs]
    per_budget = {budget: capacity(cells, budget=budget, min_seeds=min_seeds)["rungs"][n] for budget in budgets}
    passes = {str(budget): per_budget[budget]["ok"] for budget in budgets}
    gain_early, gain_late = pooled_window_mean(runs, *early), pooled_window_mean(runs, *late)
    gain = gain_late - gain_early

    by_seed = None
    if earlier is not None:
        by_seed = {}
        for run in runs:
            if (run["seed"], n) not in earlier:
                raise ValueError(f"no earlier run for seed {run['seed']} rung {n}")
            by_seed[str(run["seed"])] = prefix_equal(run["points"], earlier[(run["seed"], n)], prefix)
    report = {
        "n": n,
        "seeds": len(runs),
        "first_perfect": {str(cell["seed"]): cell["first_perfect"] for cell in cells},
        "perfect_by": {str(budget): per_budget[budget]["passed"] for budget in budgets},
        "passes": passes,
        "first_pass_budget": next((budget for budget in sorted(budgets) if passes[str(budget)]), None),
        "reached_perfect": sum(cell["first_perfect"] is not None for cell in cells),
        "perfect_at_end": sum(cell["perfect_at_end"] for cell in cells),
        "mean_accuracy": {f"{early[0] + 1}_{early[1]}": float(gain_early), f"{late[0] + 1}_{late[1]}": float(gain_late)},
        "gain": float(gain),
        "trend": trend(gain),
        "control_violations": sum(not cell["control_ok"] for cell in cells),
        f"first_{prefix}_match_by_seed": by_seed,
        f"first_{prefix}_matches_earlier": None if by_seed is None else all(by_seed.values()),
    }
    if held:
        report["epochs_at_100"] = held_report(runs, late)
    return report


def summarize_extension(runs_by_width: dict[int, list[dict]], budgets: tuple[int, ...], *, min_seeds: int = 2,
                        earlier: dict | None = None, prefix: int = PREFIX, held: bool = False) -> dict:
    """earlier: {(width, seed, n): points} of the E4 runs, or None."""
    report = {}
    for d in sorted(runs_by_width):
        mine = None if earlier is None else {(seed, n): points for (width, seed, n), points in earlier.items() if width == d}
        report[str(d)] = width_report(runs_by_width[d], budgets, min_seeds=min_seeds, earlier=mine, prefix=prefix, held=held)
    return report


# ------------------------------------------------------------------ evidence


def load_extension_runs(root: Path) -> list[dict]:
    runs = []
    for run in load_runs(Path(root), budgets=()):
        match = SCENARIO.fullmatch(run["scenario"])
        if match is None:
            raise ValueError(f"scenario id {run['scenario']!r} does not follow s<seed>_n<size>")
        seed, n = int(match.group(1)), int(match.group(2))
        if run["n_a"] != n:
            raise ValueError(f"{run['id']}: scenario id says {n} entries but the book size is {run['n_a']}")
        if "model" not in run:
            raise ValueError(f"{run['id']}: a capacity run must declare its model")
        runs.append({"id": run["id"], "d": run["model"]["embedding_dim"], "seed": seed, "n": n,
                     "epochs": run["epochs"], "points": run["points"]})
    return runs


def index_runs(runs: list[dict]) -> dict:
    index = {}
    for run in runs:
        key = (run["d"], run["seed"], run["n"])
        if key in index:
            raise ValueError(f"width {key[0]}, seed {key[1]}, rung {key[2]} appears twice in the earlier evidence")
        index[key] = run["points"]
    return index


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, action="append", required=True, help="E6 evidence folder from run_experiment.py")
    parser.add_argument("--earlier-root", type=Path, action="append", default=[],
                        help="E4 evidence folder holding the 1000-epoch runs of the same cells")
    parser.add_argument("--budgets", type=int, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prefix", default="e6")
    parser.add_argument("--min-seeds", type=int, default=2, help="book seeds that must be perfect for a rung to pass")
    parser.add_argument("--prefix-epochs", type=int, default=PREFIX,
                        help="epochs of every run that must equal the earlier run point for point (default: 1000, the length of E4)")
    parser.add_argument("--earlier-label", default="E4", help="what the earlier runs are called in the printed summary")
    parser.add_argument("--epochs-at-100", action="store_true",
                        help="also count, per seed, the logged epochs with every entry recalled (whole run and late window)")
    args = parser.parse_args(argv)
    budgets = tuple(sorted(set(args.budgets)))
    if args.prefix_epochs < 1:
        raise ValueError("--prefix-epochs must be an integer >= 1")
    runs = [run for root in args.root for run in load_extension_runs(root)]
    if not runs:
        raise ValueError("no extension runs found in the evidence folders")
    for run in runs:
        too_long = [budget for budget in budgets if budget > run["epochs"]]
        if too_long:
            raise ValueError(f"{run['id']}: budget {too_long[0]} exceeds the {run['epochs']} epochs trained")
        if args.prefix_epochs > run["epochs"]:
            raise ValueError(f"{run['id']}: --prefix-epochs {args.prefix_epochs} exceeds the {run['epochs']} epochs trained")
        if args.earlier_root and args.prefix_epochs not in {point["epoch"] for point in run["points"]}:
            # prefix_equal needs a logged point at the cutoff; without one every seed would silently report "no match"
            raise ValueError(f"{run['id']}: --prefix-epochs {args.prefix_epochs} is not a logged epoch of this run")
    earlier = index_runs([run for root in args.earlier_root for run in load_extension_runs(root)]) if args.earlier_root else None
    runs_by_width: dict[int, list[dict]] = {}
    for run in runs:
        runs_by_width.setdefault(run["d"], []).append(run)
    report = summarize_extension(runs_by_width, budgets, min_seeds=args.min_seeds, earlier=earlier,
                                 prefix=args.prefix_epochs, held=args.epochs_at_100)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.prefix}_extension.json").write_bytes((json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    for d, entry in report.items():
        print(f"d={d}: rung {entry['n']}, perfect by budget {entry['perfect_by']}, first pass {entry['first_pass_budget']}, "
              f"gain {entry['gain']:+.4f} ({entry['trend']}), first {args.prefix_epochs} epochs equal {args.earlier_label}: "
              f"{entry[f'first_{args.prefix_epochs}_matches_earlier']}")


if __name__ == "__main__":
    main()
