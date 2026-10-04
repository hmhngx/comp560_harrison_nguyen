"""Independently reconcile this week's raw predictions, summaries, CSVs and hashes.

Run from the repository root. --require-local-artifacts also checks ignored
checkpoints and prepared binaries; omit it to audit the committed readable evidence.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = Path(__file__).resolve().parent


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_run(root: Path, local_artifacts: bool) -> dict:
    protocol = json.loads((root / "protocol.json").read_text())
    require(not protocol["git"]["dirty"], f"Training began dirty: {root}")
    for path, expected in protocol["inputs"].items():
        require(digest(ROOT / path.replace("\\", "/")) == expected, f"Input hash differs: {path}")
    config_path = "phonebook/config/phonebook.py"
    committed_config = subprocess.check_output(["git", "show", f"{protocol['git']['commit']}:{config_path}"], cwd=ROOT)
    canonical = committed_config.replace(b"\r\n", b"\n")
    # The original raw hash may use Windows checkout newlines. Check both
    # byte layouts, while requiring the current logical config to match its commit.
    config_hashes = {hashlib.sha256(value).hexdigest() for value in [canonical, canonical.replace(b"\n", b"\r\n")]}
    require(protocol["config_sha256"] in config_hashes and
            (ROOT / config_path).read_bytes().replace(b"\r\n", b"\n") == canonical, "Config/commit hash differs")
    if local_artifacts:
        prepared = json.loads((root / "prepared_data.json").read_text())
        for name, expected in prepared["sha256"].items():
            require(digest(root / "data" / name) == expected, f"Prepared data hash differs: {root / name}")
    rows = list(csv.DictReader((root / "summary.csv").open(newline="")))
    require(len({row["condition"] for row in rows}) == len(rows), "Duplicate summary condition")
    require({p.stem.removesuffix(".evaluation") for p in root.glob("*.evaluation.json")} ==
            {row["condition"] for row in rows}, "Summary/evaluation set differs")
    results = {}
    completions = 0
    for row in rows:
        name = row["condition"]
        result = json.loads((root / f"{name}.evaluation.json").read_text())
        results[name] = result
        if local_artifacts:
            require(digest(Path(result["checkpoint"])) == result["checkpoint_sha256"], f"Checkpoint hash differs: {name}")
            require(digest(Path(result["metadata"])) == result["metadata_sha256"], f"Metadata hash differs: {name}")
        require(set(result["books"]) == {"A", "B"}, f"Missing independent book: {name}")
        for book in ["A", "B"]:
            scores = result["books"][book]
            source = ROOT / "phonebook/inputs" / f"phonebook{book}.txt"
            require(digest(source) == scores["source_sha256"], f"Evaluation source differs: {name}/{book}")
            expected = dict(line.split("=", 1) for line in source.read_text().splitlines())
            generation = scores["generation"]
            predictions = generation["predictions"]
            require(len(predictions) == len(expected) == 10, f"Unexpected generation denominator: {name}/{book}")
            require({p["prompt"] for p in predictions} == set(expected), f"Prompt coverage differs: {name}/{book}")
            correct = 0
            stops = {stop: 0 for stop in ["newline", "budget", "context"]}
            for prediction in predictions:
                require(prediction["expected"] == expected[prediction["prompt"]], "Ground truth differs")
                valid = prediction["prediction"] == prediction["expected"] and prediction["stop_reason"] == "newline"
                require(valid == prediction["correct"], f"Raw generation disagrees with score: {name}/{book}")
                correct += valid
                stops[prediction["stop_reason"]] += 1
            require(correct == generation["correct"] == int(row[f"{book}_generation_correct"]), "Generation totals differ")
            require(generation["total"] == int(row[f"{book}_total"]) == 10, "Summary denominator differs")
            require(generation["accuracy"] == correct / 10 and generation["stop_counts"] == stops, "Generation rates/stops differ")
            teacher = scores["teacher_forced"]
            require(teacher["token_total"] == sum(len(value) + 1 for value in expected.values()) == 130, "Token denominator differs")
            require(teacher["sequence_total"] == 10, "Sequence denominator differs")
            require(teacher["token_correct"] / 130 == teacher["token_accuracy"] == float(row[f"{book}_token_accuracy"]), "Token accuracy differs")
            require(teacher["sequence_correct"] / 10 == teacher["sequence_accuracy"], "Sequence accuracy differs")
            require(math.isfinite(teacher["loss"]) and teacher["loss"] == float(row[f"{book}_cross_entropy"]), "Cross entropy differs")
            completions += len(predictions)
    records = [json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines()]
    expected_runs = {c["name"] for c in protocol["primary_conditions"]}
    for extension in protocol["conditional_extensions"]:
        trigger = results[extension["trigger"]]
        if any(v["generation"]["correct"] < v["generation"]["total"] for v in trigger["books"].values()):
            expected_runs.add(extension["name"])
    require(len(records) == len(expected_runs) and {r["condition"]["name"] for r in records} == expected_runs, "Protocol/manifest run set differs")
    for record in records:
        require(record["git"] == protocol["git"] and record["returncode"] == 0 and not record["timed_out"], "Unclean or failed run")
        condition = record["condition"]
        command = record["command"]
        for flag, expected in {"--device": "cpu", "--seed": "42", "--lr": str(condition["lr"]),
                               "--epochs": str(condition["epochs"]), "--l2sp-lambda": "0"}.items():
            require(command.count(flag) == 1 and command[command.index(flag) + 1] == expected,
                    f"Command/protocol differs: {condition['name']}/{flag}")
        require(("--adapt-from" in command) == condition["adapt"], "Initialization mode differs")
        if condition["adapt"]:
            phase_a_name = f"phaseA_seed{protocol.get('phase_a_seed', 42)}"
            require(command[command.index("--adapt-from") + 1] == protocol["historical_checkpoints"][phase_a_name]["path"],
                    "Adaptation did not start from the original Phase-A checkpoint")
        metrics = list(csv.DictReader((root / condition["name"] / "metrics.csv").open(newline="")))
        require([int(row["Epoch"]) for row in metrics] == list(range(10, condition["epochs"] + 1, 10)), "Logged epochs incomplete")
        require(all(math.isfinite(float(value)) for row in metrics for value in row.values()), "Nonfinite CSV metric")
        books = results[condition["name"]]["books"]
        final = metrics[-1]
        for column, expected in {
            "Train Token Acc": sum(b["teacher_forced"]["token_correct"] for b in books.values()) / 260,
            "Train Seq Acc": sum(b["teacher_forced"]["sequence_correct"] for b in books.values()) / 20,
            "Val Token Acc": books["A"]["teacher_forced"]["token_accuracy"],
            "Val Seq Acc": books["A"]["teacher_forced"]["sequence_accuracy"],
        }.items():
            require(math.isclose(float(final[column]), expected, abs_tol=0.000051), f"Final CSV/evaluation differs: {condition['name']}/{column}")
    return {"directory": root.name, "code_commit": protocol["git"]["commit"],
            "evaluations": len(rows), "completions": completions, "successful_training_runs": len(records)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-local-artifacts", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    runs = [verify_run(EVIDENCE / name, args.require_local_artifacts) for name in
            ["run_2026-10-04", "replication_phaseA_seed123", "replication_phaseA_seed999"]]
    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "status": "passed",
              "local_artifact_hashes_checked": args.require_local_artifacts, "runs": runs,
              "total_evaluations": sum(r["evaluations"] for r in runs),
              "total_completions": sum(r["completions"] for r in runs),
              "total_successful_training_runs": sum(r["successful_training_runs"] for r in runs)}
    payload = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
