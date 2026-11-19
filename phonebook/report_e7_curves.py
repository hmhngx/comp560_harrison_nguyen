"""The table behind the report's E7 figure: the recalled count at every logged epoch of the E6 runs and the two E7 arms at widths 16, 32 and 64.

    python phonebook/report_e7_curves.py --out phonebook/results [--force]

Reads only committed evidence: the 9 E6 runs at widths 16, 32 and 64 (week_2026-10-31, the 12 E6 runs minus width 8) and the 18 E7 runs
(week_2026-11-13: arm W, weight decay 0, and arm S, learning rate x0.1 after epoch 2000, three widths x three seeds each). Writes
fig_e7_curves.csv (LF line endings), one row per logged epoch of each of the 27 runs:

    arm      E6 (the E6 control: weight decay 0.1, constant learning rate), W or S
    model_d  embedding width
    entries  phonebook entries trained on (160, 320 or 640 for the E7 rungs)
    seed     1, 2 or 3
    epoch    logged epoch (every 10th, 10 to 4000)
    recalled entries recalled exactly at that epoch (teacher-forced exact sequence match, "Train Seq Acc" times entries)

The E6 rows are exactly the rows of fig_e6_curves.csv at those widths. An existing file is never overwritten unless --force is given.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from phonebook.report_capacity_replay import curve_data, write_table
from phonebook.report_experiments import load_runs

WEEK_E6 = ROOT_DIR / "phonebook" / "experiments" / "week_2026-10-31"
WEEK_E7 = ROOT_DIR / "phonebook" / "experiments" / "week_2026-11-13"

TABLE_NAME = "fig_e7_curves.csv"
COLUMNS = ["arm", "model_d", "entries", "seed", "epoch", "recalled"]
ARMS = ("E6", "W", "S")
WIDTHS = (16, 32, 64)
TRAIN_EPOCHS = 4000
LOGGED_EPOCHS = list(range(10, TRAIN_EPOCHS + 1, 10))           # logged every 10th epoch
S_SCHEDULE = "step:2000:0.1"                                    # arm S: the learning rate x0.1 after epoch 2000


def check_arm(name: str, arm: str, run: dict) -> None:
    """The protocol of an E7 run must be the arm its folder name claims: W has weight decay 0.0 and a constant learning rate, S has the
    step schedule and the default weight decay (no weight_decay key)."""
    if arm == "W" and not (run.get("weight_decay") == 0.0 and "lr_schedule" not in run):
        raise ValueError(f"{name}: arm W needs weight_decay 0.0 and no lr_schedule, the protocol has weight_decay="
                         f"{run.get('weight_decay')!r} and lr_schedule={run.get('lr_schedule')!r}")
    if arm == "S" and not (run.get("lr_schedule") == S_SCHEDULE and "weight_decay" not in run):
        raise ValueError(f"{name}: arm S needs lr_schedule {S_SCHEDULE!r} and the default weight decay, the protocol has lr_schedule="
                         f"{run.get('lr_schedule')!r} and weight_decay={run.get('weight_decay')!r}")


def load_e7_runs(week: Path) -> dict:
    """{(arm, width, seed): curve} for the E7 runs named e7_w_d<width>_seed<seed> and e7_s_d<width>_seed<seed>."""
    pattern = re.compile(r"e7_([ws])_d(\d+)_seed(\d+)")
    curves = {}
    for root in sorted(Path(week).glob("e7_*_d*_seed*")):
        match = pattern.fullmatch(root.name)
        if match is None or not root.is_dir():
            continue
        runs = load_runs(root, budgets=())
        if len(runs) != 1:
            raise ValueError(f"{root.name}: expected 1 run, found {len(runs)}")
        (run,) = runs
        arm, width, seed = match.group(1).upper(), int(match.group(2)), int(match.group(3))
        if run["model"]["embedding_dim"] != width:
            raise ValueError(f"{root.name}: the run declares width {run['model']['embedding_dim']}")
        if run["epochs"] != TRAIN_EPOCHS:
            raise ValueError(f"{root.name}: the run trains for {run['epochs']} epochs, not {TRAIN_EPOCHS} epochs")
        check_arm(root.name, arm, run)
        if any(point["A"] is None for point in run["points"]):
            raise ValueError(f"{root.name}: a logged epoch has no recall count")
        curves[(arm, width, seed)] = {"entries": run["n_a"], "epochs": [p["epoch"] for p in run["points"]],
                                      "recalled": [p["A"] for p in run["points"]]}
    return curves


def e7_curve_data(week_e6: Path, week_e7: Path) -> dict:
    """{(arm, width, seed): {entries, epochs, recalled}} for the 27 runs, ordered E6, W, S, then by width and seed."""
    e7 = load_e7_runs(week_e7)
    if len(e7) != 18:
        raise ValueError(f"expected 18 E7 runs, found {len(e7)} under {week_e7}")
    e6 = {("E6", width, seed): curve for (width, seed), curve in curve_data(week_e6).items() if width in WIDTHS}
    if len(e6) != 9:
        raise ValueError(f"expected 9 E6 runs at widths 16, 32 and 64, found {len(e6)} under {week_e6}")
    curves = {**e6, **e7}
    order = {arm: i for i, arm in enumerate(ARMS)}
    curves = dict(sorted(curves.items(), key=lambda item: (order[item[0][0]], item[0][1], item[0][2])))
    for (arm, width, seed), curve in curves.items():
        if curve["epochs"] != LOGGED_EPOCHS:
            raise ValueError(f"{arm} width {width} seed {seed}: the logged epochs are not 10, 20, ..., {TRAIN_EPOCHS}")
        if any(not 0 <= value <= curve["entries"] for value in curve["recalled"]):
            raise ValueError(f"{arm} width {width} seed {seed}: a recalled count is outside 0..{curve['entries']}")
    return curves


def e7_curve_rows(curves: dict) -> list[dict]:
    return [{"arm": arm, "model_d": width, "entries": c["entries"], "seed": seed, "epoch": epoch, "recalled": recalled}
            for (arm, width, seed), c in curves.items() for epoch, recalled in zip(c["epochs"], c["recalled"])]


def write_e7_table(out: Path, *, week_e6: Path = WEEK_E6, week_e7: Path = WEEK_E7, force: bool = False) -> Path:
    out = Path(out)
    target = out / TABLE_NAME
    if target.exists() and not force:
        raise ValueError(f"{TABLE_NAME} already exists in {out}; refusing to overwrite (use --force)")
    rows = e7_curve_rows(e7_curve_data(week_e6, week_e7))
    out.mkdir(parents=True, exist_ok=True)
    write_table(target, rows, COLUMNS)
    return target


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--week-e6", type=Path, default=WEEK_E6)
    parser.add_argument("--week-e7", type=Path, default=WEEK_E7)
    parser.add_argument("--force", action="store_true", help="replace an existing table")
    args = parser.parse_args(argv)
    try:
        path = write_e7_table(args.out, week_e6=args.week_e6, week_e7=args.week_e7, force=args.force)
    except ValueError as error:
        sys.exit(f"error: {error}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
