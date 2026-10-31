"""Write the runner spec for one model size of the capacity experiment (E4).

One spec per model size: scenarios are (book seed, rung) pairs, so every model gets its own ladder of
rungs; the single condition trains that model from scratch on each book and validates on the control.

    python phonebook/make_capacity_spec.py --d 16 --rungs 10,20,40,80 --seeds 1,2,3 --epochs 1000 \\
        --budgets 100,300,1000 --timeout 7200 --out phonebook/specs/e4_capacity_d16.json
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from phonebook.make_capacity_books import capacity_path, control_path
from phonebook.run_experiment import expected_logged_epochs, write_json


def capacity_spec(d: int, rungs: list[int], seeds: list[int], *, epochs: int, accuracy_interval: int,
                  budgets: list[int], timeout_seconds: int, n_heads: int = 2, n_layers: int = 2,
                  lr: float = 0.001, batch_size: int = 64, inputs_dir: str = "phonebook/inputs",
                  predictions: dict | None = None, min_seeds: int = 2, experiment_seeds: int | None = None,
                  train_seed: int = 42, name: str | None = None) -> dict:
    if not rungs or rungs != sorted(set(rungs)) or rungs[0] < 1:
        raise ValueError("rungs must be a strictly increasing list of positive sizes")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a non-empty list of distinct integers")
    experiment_seeds = len(seeds) if experiment_seeds is None else experiment_seeds
    if experiment_seeds < len(seeds):
        raise ValueError("experiment_seeds must be at least the number of seeds in this spec")
    if not 1 <= min_seeds <= experiment_seeds:
        raise ValueError("min_seeds must be between 1 and experiment_seeds")
    if epochs % accuracy_interval != 0:
        raise ValueError("epochs must be a multiple of accuracy_interval")
    if not budgets:
        raise ValueError("budgets must not be empty")
    logged = expected_logged_epochs(epochs, accuracy_interval)
    for budget in budgets:
        if budget not in logged:
            raise ValueError(f"budget {budget} is not a logged epoch, so it cannot be read back from the log")
    if max(budgets) != epochs:
        raise ValueError("the largest budget must equal the run length")

    inputs = Path(inputs_dir)
    rules = {
        "question": "How many random name-to-number entries does a transformer of this size memorize exactly "
                    "within a given number of epochs, and does that boundary stop moving with more training?",
        "success": "A cell is perfect by budget E if the trainer's logged training sequence accuracy is 100% "
                   "(every entry recalled exactly, which equals strict greedy generation) at some logged epoch <= E.",
        "capacity": f"A rung passes at budget E if at least {min_seeds} of its {experiment_seeds} book seeds are perfect by E. "
                    "The capacity at E is the largest rung such that it and every smaller rung pass; passes above the "
                    "first failing rung are reported but never counted.",
        "replicates": f"Replicates are book seeds. Within this spec the training seed is {train_seed}; the full experiment "
                      "varies the initialisation together with the book only when each book seed is its own spec with "
                      "that book seed as its training seed.",
        "boundary": "The boundary is stable if the capacity at the largest budget equals the capacity at the previous "
                    "budget; otherwise it is still moving and is reported as a budget limit, not a capacity. If no rung "
                    "fails at the largest budget the ladder ended before the boundary and the capacity is only a lower bound.",
        "controls": "The control book holds entries sharing no name or number with training; its exact recall must "
                    "stay 0. The logged parameter count must equal the closed form for the declared model.",
        "budgets": f"Capacities are read at epochs {sorted(budgets)} from one run of {epochs} epochs per cell "
                   "(training is deterministic with an isolated loader stream, so a shorter budget is a prefix).",
    }
    if predictions is not None:
        rules["predictions"] = predictions
    return {
        "name": name or f"e4_capacity_d{d}",
        "description": f"Capacity of a {n_layers}-layer, {n_heads}-head transformer with embedding width {d}: train from "
                       f"scratch on nested books of {rungs} random entries for {epochs} epochs, "
                       f"{len(seeds)} book seed{'s' if len(seeds) != 1 else ''}.",
        "seed": train_seed,
        "model_config": "phonebook/config/phonebook.py",
        "evaluate_books": ["A", "B"],
        "require_one_update_per_epoch": False,
        "timeout_seconds": timeout_seconds,
        "decision_rules": rules,
        "scenarios": [{"id": f"s{seed}_n{n}", "book_a": capacity_path(inputs, seed, n).as_posix(),
                       "book_b": control_path(inputs, seed).as_posix()} for seed in seeds for n in rungs],
        "conditions": [{"id": f"d{d}", "train": {"a_copies": 1, "b_copies": 0}, "eval_book": "B", "lr": lr,
                        "epochs": epochs, "batch_size": batch_size, "accuracy_interval": accuracy_interval,
                        "adapt": False, "model": {"embedding_dim": d, "n_heads": n_heads, "n_layers": n_layers}}],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--d", type=int, required=True, help="embedding width")
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--rungs", required=True, help="comma-separated strictly increasing book sizes")
    parser.add_argument("--seeds", required=True, help="comma-separated book seeds")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--accuracy-interval", type=int, default=10)
    parser.add_argument("--budgets", required=True, help="comma-separated epochs at which capacity is read")
    parser.add_argument("--timeout", type=int, required=True, help="seconds allowed per run")
    parser.add_argument("--min-seeds", type=int, default=2, help="book seeds that must be perfect for a rung to pass")
    parser.add_argument("--experiment-seeds", type=int, help="book seeds in the whole experiment (default: this spec's)")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--name", help="spec name (default e4_capacity_d<d>)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.out.exists():
            raise ValueError(f"{args.out} already exists; refusing to overwrite a spec")
        spec = capacity_spec(args.d, [int(n) for n in args.rungs.split(",")], [int(s) for s in args.seeds.split(",")],
                             epochs=args.epochs, accuracy_interval=args.accuracy_interval,
                             budgets=[int(b) for b in args.budgets.split(",")], timeout_seconds=args.timeout,
                             n_heads=args.heads, n_layers=args.layers, min_seeds=args.min_seeds,
                             experiment_seeds=args.experiment_seeds, train_seed=args.train_seed, name=args.name)
        write_json(args.out, spec)
        print(f"wrote {args.out}: {len(spec['scenarios'])} runs")
    except ValueError as error:
        sys.exit(f"error: {error}")


if __name__ == "__main__":
    main()
