"""Run the bounded, CPU-only October 4 replay pilot with immutable output directories.

The protocol includes four primary runs, and two conditional 500-update
extensions declared before any results are inspected. Historical runs are never overwritten.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from completion_core.vocabulary import Vocabulary
from phonebook.evaluate_checkpoint import evaluate_checkpoint, sha256
from phonebook.prepare_phonebook import load_phonebook_lines
from train import TrainConfig


def primary_conditions() -> list[dict]:
    return [
        {"name": "replay_lr0.0001_ep100", "lr": 0.0001, "epochs": 100, "adapt": True},
        {"name": "replay_lr0.0001_ep500", "lr": 0.0001, "epochs": 500, "adapt": True},
        {"name": "replay_lr0.001_ep100", "lr": 0.001, "epochs": 100, "adapt": True},
        {"name": "joint_lr0.001_ep100", "lr": 0.001, "epochs": 100, "adapt": False},
    ]


def environment() -> dict:
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "packages": {name: importlib.metadata.version(name) for name in ["torch", "numpy", "matplotlib", "pytest"]},
            "torch_version": torch.__version__, "torch_num_threads": torch.get_num_threads(), "device": "cpu"}


def git_state() -> dict:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT_DIR, check=True,
                            capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT_DIR, check=True,
                           capture_output=True, text=True).stdout.strip()
    return {"commit": commit, "dirty": bool(dirty)}


def validate_inputs(books: dict[str, Path], replay: Path) -> None:
    lines = {name: load_phonebook_lines(path) for name, path in books.items()}
    names = {book: [line.split("=", 1)[0] for line in content] for book, content in lines.items()}
    if any(len(content) != 10 for content in lines.values()):
        raise ValueError("Pilot requires exactly ten records per book.")
    if any(len(set(content)) != 10 for content in names.values()) or set(names["A"]) & set(names["B"]):
        raise ValueError("Pilot names must be unique within and disjoint across books.")
    if not all(re.fullmatch(r"[^=]+=[2-9][0-9]{2}-[2-9][0-9]{2}-[0-9]{4}\n", line)
               for content in lines.values() for line in content):
        raise ValueError("Pilot records must use the expected phone-number format.")
    if load_phonebook_lines(replay) != lines["A"] + lines["B"]:
        raise ValueError("Balanced replay must contain A once and B once, in that order.")


def training_command(condition: dict, data: Path, output: Path, checkpoint: Path) -> list[str]:
    command = [sys.executable, "train.py", "phonebook/config/phonebook.py", "--device", "cpu",
               "--data-dir", str(data), "--out-dir", str(output), "--seed", "42",
               "--lr", str(condition["lr"]), "--epochs", str(condition["epochs"]), "--l2sp-lambda", "0"]
    if condition["adapt"]:
        command += ["--adapt-from", str(checkpoint)]
    return command


def needs_extension(result: dict) -> bool:
    return any(book["generation"]["correct"] != book["generation"]["total"]
               for book in result["books"].values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=Path("phonebook/experiments/week_2026-09-30/run_2026-10-04"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output = args.out_root.resolve()
    books = {name: ROOT_DIR / f"phonebook/inputs/phonebook{name}.txt" for name in ["A", "B"]}
    replay = ROOT_DIR / "phonebook/inputs/replay_k1.txt"
    checkpoint = ROOT_DIR / "phonebook/out_phaseA/model.pth"
    source_meta = ROOT_DIR / "phonebook/data_phaseA/meta.pkl"
    baseline_paths = {
        "phaseA_seed42": checkpoint,
        "historical_l2sp_lambda2": ROOT_DIR / "phonebook/sweep_lr/seeded/lr0.0001_ep100_seed42_l2sp2.0/model.pth",
        "historical_replay_k1": ROOT_DIR / "phonebook/out_phaseB_replay_k1/model.pth",
        "historical_replay_k3": ROOT_DIR / "phonebook/out_phaseB_replay_k3/model.pth",
        "historical_replay_k5": ROOT_DIR / "phonebook/out_phaseB_replay_k5/model.pth",
        "historical_combined": ROOT_DIR / "phonebook/out_phaseB_replay_k1_l2sp0.1/model.pth",
    }
    for seed in [123, 999]:
        baseline_paths[f"phaseA_seed{seed}"] = ROOT_DIR / f"phonebook/out_phaseA_seed{seed}/model.pth"
        baseline_paths[f"historical_l2sp_lambda5_phaseA_seed{seed}"] = (
            ROOT_DIR / f"phonebook/sweep_lr/seeded/lr0.0001_ep100_seed42_l2sp5.0_pA-out_phaseA_seed{seed}/model.pth"
        )
    validate_inputs(books, replay)
    provenance = git_state()
    conditions = primary_conditions()
    extensions = [
        {"name": "replay_lr0.001_ep500", "lr": 0.001, "epochs": 500, "adapt": True,
         "trigger": "replay_lr0.001_ep100"},
        {"name": "joint_lr0.001_ep500", "lr": 0.001, "epochs": 500, "adapt": False,
         "trigger": "joint_lr0.001_ep100"},
    ]
    protocol = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "git": provenance,
                "environment": environment(), "primary_conditions": conditions,
                "conditional_extensions": extensions,
                "extension_rule": "Run the declared 500-update extension if either book has less than perfect strict generation at 100 updates.",
                "seed": 42, "batch_size": 64, "train_sequences": 20, "updates_per_epoch": 1,
                "optimizer": {"name": "AdamW", "fresh_state_per_run": True,
                              "weight_decay": TrainConfig().weight_decay,
                              "grad_clip": TrainConfig().grad_clip},
                "inputs": {str(path.relative_to(ROOT_DIR)): sha256(path) for path in [*books.values(), replay]},
                "config_sha256": sha256(ROOT_DIR / "phonebook/config/phonebook.py"),
                "historical_checkpoints": {name: {"path": str(path), "sha256": sha256(path)}
                                           for name, path in baseline_paths.items()},
                "phase_a_checkpoint_sha256": sha256(checkpoint), "phase_a_meta_sha256": sha256(source_meta)}
    if args.dry_run:
        print(json.dumps(protocol, indent=2))
        return
    if provenance["dirty"]:
        parser.error("Commit the implementation and inputs before training; the pilot requires a clean tree.")
    if output.exists():
        parser.error(f"Output root already exists; choose a new root to preserve evidence: {output}")
    output.mkdir(parents=True)
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    data = output / "data"
    prep = [sys.executable, "phonebook/prepare_phonebook.py", "--train-input", str(replay),
            "--eval-input", str(books["A"]), "--out-dir", str(data)]
    subprocess.run(prep, cwd=ROOT_DIR, check=True)
    if Vocabulary.from_pickle(data / "meta.pkl").stoi != Vocabulary.from_pickle(source_meta).stoi:
        raise ValueError("Prepared vocabulary differs from the Phase-A vocabulary.")
    if any(sha256(ROOT_DIR / path) != digest for path, digest in protocol["inputs"].items()):
        raise ValueError("Inputs changed during preparation.")
    (output / "prepared_data.json").write_text(json.dumps({
        "command": prep, "sha256": {name: sha256(data / name) for name in ["train.bin", "val.bin", "meta.pkl"]},
    }, indent=2) + "\n", encoding="utf-8")

    results = {}

    def save_evaluation(name: str, path: Path) -> dict:
        result = evaluate_checkpoint(path, data / "meta.pkl", books)
        results[name] = result
        (output / f"{name}.evaluation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(name, {book: metrics["generation"]["correct"] for book, metrics in result["books"].items()}, flush=True)
        return result

    for name, path in baseline_paths.items():
        save_evaluation(name, path)

    def run_condition(condition: dict) -> None:
        if git_state() != provenance:
            raise ValueError("Git state changed during the pilot; refusing to train.")
        if sha256(checkpoint) != protocol["phase_a_checkpoint_sha256"]:
            raise ValueError("Phase-A checkpoint changed during the pilot.")
        if any(sha256(ROOT_DIR / path) != digest for path, digest in protocol["inputs"].items()):
            raise ValueError("Inputs changed during the pilot.")
        for name, digest in json.loads((output / "prepared_data.json").read_text())["sha256"].items():
            if sha256(data / name) != digest:
                raise ValueError("Prepared training data changed during the pilot.")
        out_dir = output / condition["name"]
        command = training_command(condition, data, out_dir, checkpoint)
        print("Training", condition["name"], flush=True)
        start = time.perf_counter()
        timed_out = False
        try:
            completed = subprocess.run(command, cwd=ROOT_DIR, capture_output=True, text=True, timeout=300)
            returncode = completed.returncode
            log = completed.stdout + completed.stderr
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            returncode = None
            def decode(value):
                return value.decode(errors="replace") if isinstance(value, bytes) else (value or "")
            log = decode(exc.stdout) + decode(exc.stderr) + "\nTIMEOUT after 300 seconds\n"
        record = {"condition": condition, "command": command, "cwd": str(ROOT_DIR), "git": provenance,
                  "timestamp_utc": datetime.now(timezone.utc).isoformat(), "returncode": returncode, "timed_out": timed_out,
                  "elapsed_seconds": time.perf_counter() - start}
        (output / f"{condition['name']}.log").write_text(log, encoding="utf-8")
        with (output / "manifest.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")
        if timed_out:
            raise RuntimeError(f"Training timed out: {condition['name']}; failure recorded in manifest.")
        completed.check_returncode()
        with (out_dir / "metrics.csv").open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        if int(rows[-1]["Epoch"]) != condition["epochs"]:
            raise ValueError(f"Incomplete training run: {condition['name']}")
        save_evaluation(condition["name"], out_dir / "model.pth")

    for condition in conditions:
        run_condition(condition)
    for condition in extensions:
        if needs_extension(results[condition["trigger"]]):
            run_condition(condition)

    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["condition", "A_generation_correct", "A_total", "B_generation_correct", "B_total",
                         "A_token_accuracy", "B_token_accuracy", "A_cross_entropy", "B_cross_entropy"])
        for name, result in results.items():
            a, b = result["books"]["A"], result["books"]["B"]
            writer.writerow([name, a["generation"]["correct"], a["generation"]["total"],
                             b["generation"]["correct"], b["generation"]["total"],
                             a["teacher_forced"]["token_accuracy"], b["teacher_forced"]["token_accuracy"],
                             a["teacher_forced"]["loss"], b["teacher_forced"]["loss"]])
    print(f"Results saved to {output}", flush=True)


if __name__ == "__main__":
    main()
