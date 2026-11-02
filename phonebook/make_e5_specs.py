"""Write the runner specs of the replay-ratio confirmation (E5).

    phase-a   fresh Phase-A models for a list of phonebooks, one spec per init seed
    main      the k grid on one phonebook's three Phase-A inits (pins read from the Phase-A evidence)
    noise     two configurations of three units under another shuffle seed (the noise arm)
    anchor    E3's k=1 and k=3 at lr 0.001 under the historical loader, to reproduce E3 exactly

    python phonebook/make_e5_specs.py phase-a --init 42 --phonebooks 2,4,7,8,9,12 --out phonebook/specs/e5_phase_a_s42.json
    python phonebook/make_e5_specs.py main --phonebook 2 --phase-a-root 42=EVIDENCE_S42 --phase-a-root 123=... --out ...
    python phonebook/make_e5_specs.py noise --shuffle-seed 1 --phonebooks 2,4,7 --phase-a-root 42=EVIDENCE_S42 --out ...
    python phonebook/make_e5_specs.py anchor --e3-spec phonebook/specs/e3_replay_ratio.json --out ...

A Phase-A model that did not recall all of its own book A is refused when its pin is read (the gate).
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from phonebook import audit_experiment as audit_mod
from phonebook.evaluate_checkpoint import sha256
from phonebook.make_heldout_dataset import book_paths
from phonebook.predictions import load_predictions
from phonebook.run_experiment import write_json

MODEL_CONFIG = "phonebook/config/phonebook.py"
INPUTS_DIR = Path("phonebook/inputs")
INITS = (42, 123, 999)
MAIN_SEED = 42
PRIMARY_LR, PRIMARY_KS = 0.001, (1, 2, 3, 4, 5, 9)
ROBUST_LRS, ROBUST_KS = (0.003, 0.0003), (1, 2, 3, 5)
NOISE_KS = (1, 3)
ANCHOR_IDS = ("k1_lr0.001", "k3_lr0.001")
EPOCHS, BATCH_SIZE = 500, 128


def _books(phonebook: int) -> dict:
    a_path, b_path = book_paths(INPUTS_DIR, phonebook)
    return {"book_a": a_path.as_posix(), "book_b": b_path.as_posix()}


def replay_condition(k: int, lr: float) -> dict:
    return {"id": f"k{k}_lr{lr}", "train": {"a_copies": 1, "b_copies": k}, "eval_book": "A", "lr": lr,
            "epochs": EPOCHS, "batch_size": BATCH_SIZE, "accuracy_interval": 1, "adapt": True}


def plain_condition(lr: float) -> dict:
    return {"id": f"plain_lr{lr}", "train": {"a_copies": 0, "b_copies": 1}, "eval_book": "A", "lr": lr,
            "epochs": EPOCHS, "batch_size": BATCH_SIZE, "accuracy_interval": 1, "adapt": True}


# ------------------------------------------------------------------ Phase A and pins


def phase_a_spec(init_seed: int, phonebooks: list[int]) -> dict:
    return {
        "name": f"e5_phase_a_s{init_seed}",
        "description": f"Fresh Phase-A models (init seed {init_seed}), one per phonebook: train on book A, evaluate on book B, "
                       "the same recipe as E2's Phase A.",
        "seed": init_seed,
        "model_config": MODEL_CONFIG,
        "evaluate_books": ["A", "B"],
        "require_one_update_per_epoch": True,
        "decision_rules": {
            "gate": "Each Phase-A checkpoint must recall all ten entries of its own book A exactly. If one does not after 100 "
                    "updates, that phonebook is retrained from scratch with 200 and then 400 updates (a new spec each time) "
                    "and every attempt is kept as evidence. The main-grid generator refuses a checkpoint that fails the gate."},
        "scenarios": [{"id": f"pb{p}", **_books(p)} for p in phonebooks],
        "conditions": [{"id": "phase_a", "train": {"a_copies": 1, "b_copies": 0}, "eval_book": "B", "lr": 0.001,
                        "epochs": 100, "batch_size": 64, "accuracy_interval": 10, "adapt": False}],
    }


def read_pins(phase_a_root: Path, phonebook: int, *, repo_root: Path) -> dict:
    folder = Path(phase_a_root) / f"pb{phonebook}"
    model = folder / "phase_a" / "model.pth"
    meta = folder / "data_a1b0_evalB" / "meta.pkl"
    evaluation = folder / "phase_a.evaluation.json"
    for path in (model, meta, evaluation):
        if not path.is_file():
            raise ValueError(f"missing Phase-A evidence: {path}")
    own_book = json.loads(evaluation.read_text(encoding="utf-8"))["books"]["A"]["generation"]
    if own_book["correct"] != own_book["total"]:
        raise ValueError(f"Phase-A gate failed for phonebook {phonebook}: book A recalled "
                         f"{own_book['correct']}/{own_book['total']}")

    def relative(path: Path) -> str:
        try:
            return path.resolve().relative_to(Path(repo_root).resolve()).as_posix()
        except ValueError:
            raise ValueError(f"Phase-A evidence must live inside the repository to be pinned: {path}") from None

    return {"checkpoint": relative(model), "checkpoint_sha256": sha256(model),
            "meta": relative(meta), "meta_sha256": sha256(meta)}


def audit_phase_a(phase_a_root: Path, *, repo_root: Path) -> None:
    """Pin only evidence that passes the full audit, including re-generating every answer from the saved weights."""
    try:
        audit_mod.audit(Path(phase_a_root), repo_root=Path(repo_root), require_local_artifacts=True)
    except (ValueError, OSError) as error:
        raise ValueError(f"Phase-A evidence failed its audit ({phase_a_root}): {error}") from error


# ------------------------------------------------------------------ main grid, noise arm, anchor


def main_spec(phonebook: int, pins_by_init: dict, *, predictions: dict | None = None) -> dict:
    missing = [init for init in INITS if init not in pins_by_init]
    if missing:
        raise ValueError(f"missing pins for init seeds {missing}")
    rules = {
        "question": "With one copy of A and k copies of B, which k reaches exact recall of both books soonest, and is "
                    "the difference larger than run-to-run noise?",
        "endpoint": "stable_joint: the first update from which every logged update has both books recalled exactly; "
                    "a run that never gets there is censored and counts as infinitely slow. Secondary: work = updates x "
                    "(10 + 10k) sequences, and the lowest A count at any update (the dip).",
        "unit": "The phonebook. The three Phase-A inits of a phonebook are reduced to their median first, and every "
                "comparison is paired within a phonebook against k = 1.",
        "noise": "The noise band is the largest spread of stable_joint among runs that differ only in the shuffle seed "
                 "(the e5_noise specs plus the seed-42 run of the main grid), measured at lr 0.001.",
        "verdict": "At lr 0.001 only: k beats k = 1 if it is faster in at least 5 of 6 phonebooks and its median paired "
                   "gain exceeds the noise band. Other learning rates are descriptive.",
        "plateau": "Which k is fastest is answered by the plateau: every k whose median over phonebooks is within the "
                   "noise band of the best median.",
        "work": "Counted in sequences processed, the cheapest k is reported separately from the fastest in updates.",
        "controls": "Plain fine-tuning (no copy of A) must lose A. The anchor re-run (e5_anchor_e3) must reproduce "
                    "E3's k=1 and k=3 curves exactly under the historical loader stream.",
        "scope": "Six independent phonebooks of one task type, one model size, full-batch AdamW; not a claim about "
                 "other data, sizes or optimizers.",
    }
    if predictions is not None:
        rules["predictions"] = predictions
    conditions = [replay_condition(k, PRIMARY_LR) for k in PRIMARY_KS] + [plain_condition(PRIMARY_LR)]
    conditions += [replay_condition(k, lr) for lr in ROBUST_LRS for k in ROBUST_KS]
    return {
        "name": f"e5_main_pb{phonebook}",
        "description": f"Replay ratio on phonebook {phonebook}: A once plus B k times, three Phase-A inits, shuffle seed {MAIN_SEED}.",
        "seed": MAIN_SEED,
        "model_config": MODEL_CONFIG,
        "evaluate_books": ["A", "B"],
        "require_one_update_per_epoch": True,
        "timeout_seconds": 1800,
        "decision_rules": rules,
        "scenarios": [{"id": f"pb{phonebook}_s{init}", **_books(phonebook), "init": dict(pins_by_init[init])} for init in INITS],
        "conditions": conditions,
    }


def main_spec_from_evidence(phonebook: int, roots_by_init: dict, *, repo_root: Path, predictions: dict | None = None) -> dict:
    pins = {init: read_pins(root, phonebook, repo_root=repo_root) for init, root in roots_by_init.items()}
    return main_spec(phonebook, pins, predictions=predictions)


def noise_spec(shuffle_seed: int, pins_by_phonebook: dict) -> dict:
    if shuffle_seed == MAIN_SEED:
        raise ValueError(f"the noise arm needs a shuffle seed other than the main grid's seed {MAIN_SEED}")
    return {
        "name": f"e5_noise_seed{shuffle_seed}",
        "description": f"Noise arm: k=1 and k=3 at lr 0.001 on init {MAIN_SEED} of each phonebook, shuffle seed {shuffle_seed}.",
        "seed": shuffle_seed,
        "model_config": MODEL_CONFIG,
        "evaluate_books": ["A", "B"],
        "require_one_update_per_epoch": True,
        "timeout_seconds": 1800,
        "decision_rules": {"purpose": "Runs identical to the main grid except for the shuffle seed; their spread is the noise band."},
        "scenarios": [{"id": f"pb{p}_s{MAIN_SEED}", **_books(p), "init": dict(pins)} for p, pins in sorted(pins_by_phonebook.items())],
        "conditions": [replay_condition(k, PRIMARY_LR) for k in NOISE_KS],
    }


def anchor_spec(e3_spec: dict) -> dict:
    wanted = {condition["id"]: condition for condition in e3_spec["conditions"]}
    for condition_id in ANCHOR_IDS:
        if condition_id not in wanted:
            raise ValueError(f"the E3 spec has no condition {condition_id}")
    return {
        "name": "e5_anchor_e3",
        "description": "E3's k=1 and k=3 at lr 0.001 on E3's three checkpoints, rerun with the historical loader stream. "
                       "Its logged rows must equal E3's.",
        "seed": e3_spec["seed"],
        "model_config": e3_spec["model_config"],
        "evaluate_books": e3_spec.get("evaluate_books", ["A", "B"]),
        "require_one_update_per_epoch": e3_spec.get("require_one_update_per_epoch", True),
        "timeout_seconds": e3_spec.get("timeout_seconds", 900),
        "loader_rng": "legacy",
        "decision_rules": {"check": "Every logged cell of each run equals the corresponding E3 run (report_experiments.py compare)."},
        "scenarios": copy.deepcopy(e3_spec["scenarios"]),
        "conditions": [copy.deepcopy(wanted[condition_id]) for condition_id in ANCHOR_IDS],
    }


# ------------------------------------------------------------------ command line


def _roots(values: list[str]) -> dict:
    roots = {}
    for value in values:
        init, _, path = value.partition("=")
        roots[int(init)] = Path(path)
    return roots


def _write(out: Path, spec: dict) -> None:
    if out.exists():
        raise ValueError(f"{out} already exists; refusing to overwrite a spec")
    write_json(out, spec)
    print(f"wrote {out}: {len(spec['scenarios'])} scenarios x {len(spec['conditions'])} conditions")


def build_spec(args, *, repo_root: Path) -> dict:
    if args.command == "phase-a":
        return phase_a_spec(args.init, [int(p) for p in args.phonebooks.split(",")])
    if args.command == "main":
        roots = _roots(args.phase_a_root)
        for root in roots.values():
            audit_phase_a(root, repo_root=repo_root)
        predictions = getattr(args, "predictions", None)
        return main_spec_from_evidence(args.phonebook, roots, repo_root=repo_root,
                                       predictions=load_predictions(predictions) if predictions else None)
    if args.command == "noise":
        roots = _roots([args.phase_a_root])
        if MAIN_SEED not in roots:
            raise ValueError(f"the noise arm pins init {MAIN_SEED}: give the Phase-A root as {MAIN_SEED}=PATH")
        audit_phase_a(roots[MAIN_SEED], repo_root=repo_root)
        return noise_spec(args.shuffle_seed, {int(p): read_pins(roots[MAIN_SEED], int(p), repo_root=repo_root)
                                              for p in args.phonebooks.split(",")})
    return anchor_spec(json.loads(args.e3_spec.read_text(encoding="utf-8")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    phase_a = sub.add_parser("phase-a")
    phase_a.add_argument("--init", type=int, required=True)
    phase_a.add_argument("--phonebooks", required=True)
    main_cmd = sub.add_parser("main")
    main_cmd.add_argument("--phonebook", type=int, required=True)
    main_cmd.add_argument("--phase-a-root", action="append", required=True, metavar="INIT=PATH")
    main_cmd.add_argument("--predictions", type=Path, required=True,
                          help="JSON object of the predictions written before any E5 result existed; embedded verbatim")
    noise = sub.add_parser("noise")
    noise.add_argument("--shuffle-seed", type=int, required=True)
    noise.add_argument("--phonebooks", required=True)
    noise.add_argument("--phase-a-root", required=True, metavar="42=PATH")
    anchor = sub.add_parser("anchor")
    anchor.add_argument("--e3-spec", type=Path, required=True)
    for command in (phase_a, main_cmd, noise, anchor):
        command.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        _write(args.out, build_spec(args, repo_root=ROOT_DIR))
    except (ValueError, KeyError, OSError) as error:
        sys.exit(f"error: {error}")


if __name__ == "__main__":
    main()
