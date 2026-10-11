"""Independently audit one experiment evidence folder written by run_experiment.py.

Recomputes every score from the saved raw strings, reconciles summaries, logs,
manifests, hashes and the protocol, and checks that the trainer's own logged
recall agrees with the strict generation evaluation. It writes nothing unless
--output is given. Run from the repository root:

    python phonebook/audit_experiment.py --root phonebook/experiments/week_2026-10-07/NAME
    python phonebook/audit_experiment.py --root ... --require-local-artifacts
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from phonebook.analyze_experiment import counts_from_row, load_metrics
from phonebook.evaluate_checkpoint import evaluate_checkpoint, sha256
from phonebook.make_replay_input import build_mixture_lines
from phonebook.prepare_phonebook import load_phonebook_lines
from phonebook.run_experiment import canonical_sha256, expand_runs, expected_logged_epochs, resolve, validate_spec

TOKEN_TOLERANCE = 5.1e-5  # the trainer logs four decimals


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def text_hashes(data: bytes) -> set[str]:
    """Hashes of a text file under any checkout newline convention."""
    lf = data.replace(b"\r\n", b"\n")
    return {hashlib.sha256(variant).hexdigest() for variant in (data, lf, lf.replace(b"\n", b"\r\n"))}


def _flag(command: list[str], name: str, where: str) -> str:
    require(command.count(name) == 1, f"command for {where} must contain {name} exactly once")
    return command[command.index(name) + 1]


def _audit_command(run: dict, record: dict, protocol: dict, repo_root: Path) -> None:
    spec, command, where = protocol["spec"], record["command"], run["id"]
    require(command[1] == "train.py" and command[2] == spec["model_config"], f"command for {where} does not train with the spec's config")
    expectations = {"--device": "cpu", "--seed": str(spec["seed"])}
    for flag, expected in expectations.items():
        require(_flag(command, flag, where) == expected, f"command for {where} differs from the protocol at {flag}")
    numeric = {"--lr": run["lr"], "--l2sp-lambda": run["l2sp_lambda"], "--epochs": run["epochs"],
               "--batch-size": run["batch_size"], "--accuracy-interval": run["accuracy_interval"]}
    for flag, expected in numeric.items():
        require(float(_flag(command, flag, where)) == float(expected), f"command for {where} differs from the protocol at {flag}")
    require(("--adapt-from" in command) == run["adapt"], f"command for {where} differs from the protocol in its initialization mode")
    if run["adapt"]:
        init = protocol["initializations"][run["scenario"]]
        used = Path(_flag(command, "--adapt-from", where)).resolve()
        require(used == resolve(repo_root, init["checkpoint"]).resolve(),
                f"command for {where} did not adapt from the declared Phase-A checkpoint")
    data_dir, out_dir = Path(_flag(command, "--data-dir", where)), Path(_flag(command, "--out-dir", where))
    require(data_dir.name == f"data_{run['mode']}" and data_dir.parent.name == run["scenario"], f"command for {where} reads the wrong data folder")
    require(out_dir.name == run["condition"] and out_dir.parent.name == run["scenario"], f"command for {where} writes the wrong folder")


def _audit_book(run_id: str, name: str, scores: dict, source: Path) -> None:
    lines = load_phonebook_lines(source)
    expected = dict(line.rstrip("\n").split("=", 1) for line in lines)
    require(len(expected) == len(lines), f"book {name} has duplicate prompts")
    require(sha256(source) == scores["source_sha256"], f"Input hash differs: {source} (evaluated {run_id}/{name})")
    generation, forced = scores["generation"], scores["teacher_forced"]
    predictions = generation["predictions"]
    require(len(predictions) == len(lines) == generation["total"], f"Unexpected generation denominator: {run_id}/{name}")
    require({p["prompt"] for p in predictions} == set(expected), f"Prompt coverage differs: {run_id}/{name}")
    correct, stops = 0, {"newline": 0, "budget": 0, "context": 0}
    for prediction in predictions:
        require(prediction["expected"] == expected[prediction["prompt"]], f"Ground truth differs: {run_id}/{name}")
        valid = prediction["prediction"] == prediction["expected"] and prediction["stop_reason"] == "newline"
        require(valid == prediction["correct"], f"Raw generation disagrees with its score: {run_id}/{name}")
        correct += valid
        stops[prediction["stop_reason"]] += 1
    require(correct == generation["correct"], f"Generation total differs from the raw strings: {run_id}/{name}")
    require(generation["accuracy"] == correct / len(lines) and generation["stop_counts"] == stops, f"Generation rate or stop counts differ: {run_id}/{name}")
    require(forced["token_total"] == sum(len(answer) + 1 for answer in expected.values()), f"Token denominator differs: {run_id}/{name}")
    require(forced["sequence_total"] == len(lines), f"Sequence denominator differs: {run_id}/{name}")
    require(forced["token_correct"] / forced["token_total"] == forced["token_accuracy"], f"Token accuracy differs: {run_id}/{name}")
    require(forced["sequence_correct"] / forced["sequence_total"] == forced["sequence_accuracy"], f"Sequence accuracy differs: {run_id}/{name}")
    require(forced["sequence_correct"] == generation["correct"],
            f"Teacher-forced and strict generation exact recall do not agree: {run_id}/{name}")
    require(math.isfinite(forced["loss"]), f"Non-finite cross-entropy: {run_id}/{name}")


def _audit_logged_metrics(run: dict, metrics: list[dict], evaluation: dict) -> None:
    require([row["epoch"] for row in metrics] == expected_logged_epochs(run["epochs"], run["accuracy_interval"]),
            f"Logged epochs differ from the spec: {run['id']}")
    for row in metrics:
        for column, value in row.items():
            require(value is None or math.isfinite(value), f"Non-finite {column} at epoch {row['epoch']}: {run['id']}")
    books, final = evaluation["books"], metrics[-1]
    if {"A", "B"} <= set(books):
        n_a, n_b = books["A"]["generation"]["total"], books["B"]["generation"]["total"]
        counts = counts_from_row(final, a_copies=run["train"]["a_copies"], b_copies=run["train"]["b_copies"],
                                 eval_book=run["eval_book"], n_a=n_a, n_b=n_b)
        for name, value in zip("AB", counts):
            require(value is None or value == books[name]["generation"]["correct"],
                    f"The trainer's logged recall and the strict evaluation do not agree for book {name}: {run['id']}")
        copies = {"A": run["train"]["a_copies"], "B": run["train"]["b_copies"]}
        correct = sum(copies[n] * books[n]["teacher_forced"]["token_correct"] for n in "AB")
        total = sum(copies[n] * books[n]["teacher_forced"]["token_total"] for n in "AB")
        require(abs(final["Train Token Acc"] - correct / total) <= TOKEN_TOLERANCE, f"Train token accuracy differs from the evaluation: {run['id']}")
    if run["eval_book"] in books:
        require(abs(final["Val Token Acc"] - books[run["eval_book"]]["teacher_forced"]["token_accuracy"]) <= TOKEN_TOLERANCE,
                f"Validation token accuracy differs from the evaluation: {run['id']}")


def _same_evaluation(fresh: dict, saved: dict) -> bool:
    for name, scores in saved["books"].items():
        new = fresh["books"][name]
        if new["generation"] != scores["generation"]:
            return False
        for key, value in scores["teacher_forced"].items():
            other = new["teacher_forced"][key]
            if isinstance(value, float):
                if not math.isclose(value, other, rel_tol=1e-6, abs_tol=1e-9):
                    return False
            elif value != other:
                return False
    return True


def audit(root: Path, *, repo_root: Path, require_local_artifacts: bool = False, check_git: bool = True) -> dict:
    root, repo_root = Path(root), Path(repo_root)
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    spec = protocol["spec"]
    require(not protocol["git"]["dirty"], "Training began on a dirty git tree")
    require(canonical_sha256(spec) == protocol["spec_sha256"], "The embedded spec does not match its recorded hash")
    validate_spec(spec)
    if check_git and protocol.get("spec_path"):
        shown = subprocess.run(["git", "show", f"{protocol['git']['commit']}:{protocol['spec_path']}"],
                               cwd=repo_root, capture_output=True, text=True)
        require(shown.returncode == 0, f"The recorded commit does not contain the spec {protocol['spec_path']}")
        require(json.loads(shown.stdout) == spec, "The embedded spec differs from the spec committed at the recorded commit")

    for path, expected in protocol["inputs"].items():
        require(sha256(resolve(repo_root, path)) == expected, f"Input hash differs: {path}")
    config_path = resolve(repo_root, protocol["config"]["path"])
    require(protocol["config"]["sha256"] in text_hashes(config_path.read_bytes()), "Model config differs from its recorded hash")
    if check_git and protocol.get("spec_path"):
        committed = subprocess.run(["git", "show", f"{protocol['git']['commit']}:{protocol['config']['path']}"],
                                   cwd=repo_root, capture_output=True)
        require(committed.returncode == 0 and protocol["config"]["sha256"] in text_hashes(committed.stdout),
                "The model config at the recorded commit differs from its recorded hash")

    expected_runs = expand_runs(spec)
    runs = protocol["runs"]
    require([r["id"] for r in runs] == [r["id"] for r in expected_runs], "Protocol runs differ from the spec expansion")
    for run, expected in zip(runs, expected_runs):
        require({key: run[key] for key in expected} == expected, f"Protocol run differs from the spec: {run['id']}")
        require(run["updates_per_epoch"] == 1 or not spec.get("require_one_update_per_epoch", True),
                f"Run does not have one update per epoch: {run['id']}")

    records = [json.loads(line) for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    require([record["run"]["id"] for record in records] == [run["id"] for run in runs],
            "The manifest does not record every protocol run exactly once, in order")
    evaluate_books = spec.get("evaluate_books", ["A", "B"])
    scenarios = {scenario["id"]: scenario for scenario in spec["scenarios"]}
    completions = 0
    evaluations = {}
    for run, record in zip(runs, records):
        require(record["run"] == run and record["git"] == protocol["git"], f"manifest record differs from the protocol: {run['id']}")
        require(record["returncode"] == 0 and not record["timed_out"], f"Training failed or timed out: {run['id']}")
        _audit_command(run, record, protocol, repo_root)
        evaluation = json.loads((root / run["scenario"] / f"{run['condition']}.evaluation.json").read_text(encoding="utf-8"))
        evaluations[run["id"]] = evaluation
        require(Path(evaluation["checkpoint"]).parts[-3:] == (run["scenario"], run["condition"], "model.pth"),
                f"The evaluation was made on a different run's checkpoint: {run['id']}")
        require(Path(evaluation["metadata"]).parts[-3:] == (run["scenario"], f"data_{run['mode']}", "meta.pkl"),
                f"The evaluation used a different run's vocabulary file: {run['id']}")
        require(set(evaluation["books"]) == set(evaluate_books), f"Evaluated books differ from the spec: {run['id']}")
        for name in evaluate_books:
            _audit_book(run["id"], name, evaluation["books"][name], resolve(repo_root, scenarios[run["scenario"]]["book_" + name.lower()]))
            completions += len(evaluation["books"][name]["generation"]["predictions"])
        _audit_logged_metrics(run, load_metrics(root / run["scenario"] / run["condition"] / "metrics.csv"), evaluation)
        log = root / run["scenario"] / f"{run['condition']}.log"
        if log.exists():
            require(sha256(log) == record["log_sha256"], f"Log differs from its recorded hash: {run['id']}")

    prepared = json.loads((root / "prepared_data.json").read_text(encoding="utf-8"))
    modes = {}
    for run in runs:
        modes.setdefault(f"{run['scenario']}/{run['mode']}", run)
    require(set(prepared) == set(modes), "Prepared data does not cover exactly the scenario/mode pairs of the protocol")
    for key, run in modes.items():
        scenario = scenarios[run["scenario"]]
        a_lines = load_phonebook_lines(resolve(repo_root, scenario["book_a"]))
        b_lines = load_phonebook_lines(resolve(repo_root, scenario["book_b"])) if "book_b" in scenario else []
        train = "".join(build_mixture_lines(a_lines, b_lines, **run["train"])).encode("utf-8")
        evaluation_lines = "".join(a_lines if run["eval_book"] == "A" else b_lines).encode("utf-8")
        require(hashlib.sha256(train).hexdigest() == prepared[key]["train_sha256"],
                f"Prepared training mixture differs from the declared composition: {key}")
        require(hashlib.sha256(evaluation_lines).hexdigest() == prepared[key]["eval_sha256"], f"Prepared evaluation book differs: {key}")

    with (root / "summary.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    require([row["run_id"] for row in rows] == [run["id"] for run in runs], "The summary rows differ from the protocol runs")
    for row, run in zip(rows, runs):
        books = evaluations[run["id"]]["books"]
        fixed = {"scenario": run["scenario"], "condition": run["condition"], "lr": run["lr"], "epochs": run["epochs"],
                 "batch_size": run["batch_size"], "l2sp_lambda": run["l2sp_lambda"], "a_copies": run["train"]["a_copies"],
                 "b_copies": run["train"]["b_copies"]}
        for key, expected in fixed.items():
            require(float(row[key]) == float(expected) if key not in ("scenario", "condition") else row[key] == expected,
                    f"The summary row differs from the protocol at {key}: {run['id']}")
        for name in "AB":
            if name in books:
                require(int(row[f"{name}_correct"]) == books[name]["generation"]["correct"]
                        and int(row[f"{name}_total"]) == books[name]["generation"]["total"],
                        f"The summary count differs from the evaluation for book {name}: {run['id']}")
                require(float(row[f"{name}_token_accuracy"]) == books[name]["teacher_forced"]["token_accuracy"]
                        and float(row[f"{name}_cross_entropy"]) == books[name]["teacher_forced"]["loss"],
                        f"The summary accuracy differs from the evaluation for book {name}: {run['id']}")

    if require_local_artifacts:
        for scenario_id, init in protocol["initializations"].items():
            require(sha256(resolve(repo_root, init["checkpoint"])) == init["checkpoint_sha256"], f"Checkpoint hash differs: initialization of {scenario_id}")
            require(sha256(resolve(repo_root, init["meta"])) == init["meta_sha256"], f"Metadata hash differs: initialization of {scenario_id}")
        for run in runs:
            evaluation = evaluations[run["id"]]
            require(sha256(Path(evaluation["checkpoint"])) == evaluation["checkpoint_sha256"], f"Checkpoint hash differs: {run['id']}")
            require(sha256(Path(evaluation["metadata"])) == evaluation["metadata_sha256"], f"Metadata hash differs: {run['id']}")
            scenario = scenarios[run["scenario"]]
            books = {name: resolve(repo_root, scenario["book_" + name.lower()]) for name in evaluate_books}
            fresh = evaluate_checkpoint(Path(evaluation["checkpoint"]), Path(evaluation["metadata"]), books)
            require(_same_evaluation(fresh, evaluation), f"The saved evaluation differs from a fresh evaluation of its checkpoint: {run['id']}")
        for key, record in prepared.items():
            scenario_id, mode = key.split("/")
            for name, digest in record["sha256"].items():
                require(sha256(root / scenario_id / f"data_{mode}" / name) == digest, f"Prepared data hash differs: {key}/{name}")
            require(sha256(root / scenario_id / f"inputs_{mode}" / "train.txt") == record["train_sha256"], f"Prepared training text differs: {key}")

    return {"root": str(root), "status": "passed", "spec": spec["name"], "git_commit": protocol["git"]["commit"],
            "runs": len(runs), "completions": completions, "local_artifacts_checked": require_local_artifacts,
            "git_checked": check_git}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--require-local-artifacts", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.root, repo_root=ROOT_DIR, require_local_artifacts=args.require_local_artifacts)
    payload = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
