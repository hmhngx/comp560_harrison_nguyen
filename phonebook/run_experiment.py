"""Run a declarative phonebook experiment spec and leave immutable, auditable evidence.

A spec lists scenarios (books plus an optional Phase-A initialization) and
conditions (training mixture, learning rate, updates, L2-SP strength). Each
scenario x condition pair is one training run. The protocol (spec, code commit,
input hashes, environment) is written before any training. The runner refuses a
dirty git tree and an existing output folder, so evidence is never overwritten.

Usage:
    python phonebook/run_experiment.py --spec phonebook/specs/NAME.json --dry-run
    python phonebook/run_experiment.py --spec phonebook/specs/NAME.json --out-root phonebook/experiments/week_X/NAME
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch

from completion_core.vocabulary import Vocabulary
from phonebook.evaluate_checkpoint import evaluate_checkpoint, sha256
from phonebook.make_replay_input import build_mixture_lines
from phonebook.prepare_phonebook import load_phonebook_lines
from phonebook.run_replay_pilot import environment, git_state
from train import TrainConfig

ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
HEX64 = re.compile(r"[0-9a-f]{64}")
SUMMARY_COLUMNS = [
    "run_id", "scenario", "condition", "lr", "epochs", "batch_size", "l2sp_lambda", "a_copies", "b_copies",
    "A_correct", "A_total", "B_correct", "B_total",
    "A_token_accuracy", "B_token_accuracy", "A_cross_entropy", "B_cross_entropy",
]


# --------------------------------------------------------------------------- spec


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_keys(obj, required: set, optional: set, where: str) -> None:
    if not isinstance(obj, dict):
        raise ValueError(f"{where} must be an object")
    if required - obj.keys():
        raise ValueError(f"{where} is missing {sorted(required - obj.keys())}")
    if obj.keys() - required - optional:
        raise ValueError(f"{where} has unknown keys {sorted(obj.keys() - required - optional)}")


def _check_id(value, where: str) -> None:
    if not isinstance(value, str) or not ID_PATTERN.fullmatch(value):
        raise ValueError(f"{where} id {value!r} must match {ID_PATTERN.pattern}")


def validate_spec(spec: dict) -> None:
    _check_keys(spec, {"name", "seed", "model_config", "scenarios", "conditions"},
                {"description", "decision_rules", "evaluate_books", "require_one_update_per_epoch",
                 "timeout_seconds"}, "spec")
    _check_id(spec["name"], "spec")
    if not _is_int(spec["seed"]):
        raise ValueError("spec seed must be an integer")
    if not isinstance(spec["model_config"], str):
        raise ValueError("spec model_config must be a path string")
    books = spec.get("evaluate_books", ["A", "B"])
    if not books or set(books) - {"A", "B"} or len(set(books)) != len(books):
        raise ValueError("evaluate_books must be a non-empty list of distinct values from A and B")
    if not isinstance(spec.get("require_one_update_per_epoch", True), bool):
        raise ValueError("require_one_update_per_epoch must be a boolean")
    if not _is_int(spec.get("timeout_seconds", 900)) or spec.get("timeout_seconds", 900) < 1:
        raise ValueError("timeout_seconds must be a positive integer")
    if not spec["scenarios"] or not spec["conditions"]:
        raise ValueError("spec needs at least one scenario and one condition")

    seen = set()
    for scenario in spec["scenarios"]:
        _check_keys(scenario, {"id", "book_a"}, {"book_b", "init"}, "scenario")
        _check_id(scenario["id"], "scenario")
        if scenario["id"] in seen:
            raise ValueError(f"duplicate scenario id {scenario['id']!r}")
        seen.add(scenario["id"])
        if "init" in scenario:
            where = f"scenario {scenario['id']} init"
            _check_keys(scenario["init"], {"checkpoint", "meta"}, {"checkpoint_sha256", "meta_sha256"}, where)
            for key in ("checkpoint_sha256", "meta_sha256"):
                if key in scenario["init"] and not (isinstance(scenario["init"][key], str)
                                                    and HEX64.fullmatch(scenario["init"][key])):
                    raise ValueError(f"{where} {key} must be 64 lowercase hex characters")

    seen = set()
    for condition in spec["conditions"]:
        _check_keys(condition, {"id", "train", "eval_book", "lr", "epochs", "batch_size",
                                "accuracy_interval", "adapt"}, {"l2sp_lambda"}, "condition")
        _check_id(condition["id"], "condition")
        where = f"condition {condition['id']}"
        if condition["id"] in seen:
            raise ValueError(f"duplicate condition id {condition['id']!r}")
        seen.add(condition["id"])
        _check_keys(condition["train"], {"a_copies", "b_copies"}, set(), f"{where} train")
        copies = condition["train"]
        if not (_is_int(copies["a_copies"]) and _is_int(copies["b_copies"])) \
                or copies["a_copies"] < 0 or copies["b_copies"] < 0 or copies["a_copies"] + copies["b_copies"] < 1:
            raise ValueError(f"{where} train copies must be integers >= 0 with at least one copy in total")
        if condition["eval_book"] not in ("A", "B"):
            raise ValueError(f"{where} eval_book must be A or B")
        if not _is_number(condition["lr"]) or condition["lr"] <= 0:
            raise ValueError(f"{where} lr must be a positive number")
        for field in ("epochs", "batch_size", "accuracy_interval"):
            if not _is_int(condition[field]) or condition[field] < 1:
                raise ValueError(f"{where} {field} must be an integer >= 1")
        if not isinstance(condition["adapt"], bool):
            raise ValueError(f"{where} adapt must be a boolean")
        l2sp = condition.get("l2sp_lambda", 0.0)
        if not _is_number(l2sp) or l2sp < 0:
            raise ValueError(f"{where} l2sp_lambda must be a number >= 0")
        if l2sp > 0 and not condition["adapt"]:
            raise ValueError(f"{where}: l2sp_lambda > 0 requires adapt=true (there is no anchor otherwise)")
        if condition["adapt"] and any("init" not in scenario for scenario in spec["scenarios"]):
            raise ValueError(f"{where}: adapt=true needs an init (checkpoint and meta) in every scenario")

    uses_b = "B" in books or any(c["train"]["b_copies"] > 0 or c["eval_book"] == "B" for c in spec["conditions"])
    if uses_b and any("book_b" not in scenario for scenario in spec["scenarios"]):
        raise ValueError("book_b is required in every scenario because a condition or evaluate_books uses book B")


def load_spec(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical_sha256(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def mode_name(train: dict, eval_book: str) -> str:
    return f"a{train['a_copies']}b{train['b_copies']}_eval{eval_book}"


def expand_runs(spec: dict) -> list[dict]:
    runs = []
    for scenario in spec["scenarios"]:
        for condition in spec["conditions"]:
            train = {"a_copies": condition["train"]["a_copies"], "b_copies": condition["train"]["b_copies"]}
            runs.append({
                "id": f"{scenario['id']}/{condition['id']}", "scenario": scenario["id"],
                "condition": condition["id"], "mode": mode_name(train, condition["eval_book"]),
                "train": train, "eval_book": condition["eval_book"], "lr": float(condition["lr"]),
                "epochs": condition["epochs"], "batch_size": condition["batch_size"],
                "accuracy_interval": condition["accuracy_interval"],
                "l2sp_lambda": float(condition.get("l2sp_lambda", 0.0)), "adapt": condition["adapt"],
            })
    return runs


# ------------------------------------------------------------------- run helpers


def training_command(run: dict, *, config: str, data_dir: Path, out_dir: Path, seed: int,
                     init_checkpoint: Path | None) -> list[str]:
    command = [sys.executable, "train.py", config, "--device", "cpu", "--data-dir", str(data_dir),
               "--out-dir", str(out_dir), "--seed", str(seed), "--lr", str(run["lr"]),
               "--epochs", str(run["epochs"]), "--batch-size", str(run["batch_size"]),
               "--accuracy-interval", str(run["accuracy_interval"]), "--l2sp-lambda", str(run["l2sp_lambda"])]
    if run["adapt"]:
        command += ["--adapt-from", str(init_checkpoint)]
    return command


def updates_per_epoch(n_train: int, batch_size: int) -> int:
    return -(-n_train // batch_size)


def expected_logged_epochs(epochs: int, accuracy_interval: int) -> list[int]:
    """The epochs train.py writes to metrics.csv: every 10th, every accuracy_interval-th, and the last."""
    return [e for e in range(1, epochs + 1) if e % 10 == 0 or e % accuracy_interval == 0 or e == epochs]


def check_sequence_lengths(run_id: str, train_lines: list[str], eval_lines: list[str],
                           checkpoint_seq_len: int | None) -> int:
    """train.py sizes the position table from the training book alone; guard both ways that breaks."""
    seq_len = max(len(line) for line in train_lines)
    if max(len(line) for line in eval_lines) > seq_len:
        raise ValueError(f"{run_id}: the evaluation book has a line longer than any training line "
                         f"({seq_len}); the trainer sizes the context from the training book only")
    if checkpoint_seq_len is not None and checkpoint_seq_len != seq_len:
        raise ValueError(f"{run_id}: training context is {seq_len} but the initial checkpoint was trained "
                         f"with context {checkpoint_seq_len}")
    return seq_len


def checkpoint_seq_len(path: Path) -> int:
    raw = torch.load(path, map_location="cpu")
    state = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
    return int(state["position_embedding.weight"].shape[0])


def resolve(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def write_text_lf(path: Path, text: str) -> None:
    path.write_bytes(text.encode("utf-8"))


def write_json(path: Path, obj) -> None:
    write_text_lf(path, json.dumps(obj, indent=2) + "\n")


def read_metrics(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def check_metrics(run: dict, rows: list[dict]) -> None:
    logged = [int(row["Epoch"]) for row in rows]
    if logged != expected_logged_epochs(run["epochs"], run["accuracy_interval"]):
        raise ValueError(f"{run['id']}: logged epochs differ from what the spec requires")
    for row in rows:
        for column, value in row.items():
            if value != "N/A" and not math.isfinite(float(value)):
                raise ValueError(f"{run['id']}: non-finite {column} at epoch {row['Epoch']}")


def summary_row(run: dict, result: dict) -> list:
    books = result["books"]
    row = [run["id"], run["scenario"], run["condition"], run["lr"], run["epochs"], run["batch_size"],
           run["l2sp_lambda"], run["train"]["a_copies"], run["train"]["b_copies"]]
    for name in ("A", "B"):
        row += [books[name]["generation"]["correct"], books[name]["generation"]["total"]] if name in books else ["", ""]
    for key in ("token_accuracy", "loss"):
        for name in ("A", "B"):
            row.append(books[name]["teacher_forced"][key] if name in books else "")
    return row


# ---------------------------------------------------------------------- planning


def _load_scenarios(spec: dict, repo_root: Path) -> dict:
    loaded = {}
    for scenario in spec["scenarios"]:
        a_path = resolve(repo_root, scenario["book_a"])
        b_path = resolve(repo_root, scenario["book_b"]) if "book_b" in scenario else None
        init = None
        if "init" in scenario:
            init = {"checkpoint": resolve(repo_root, scenario["init"]["checkpoint"]),
                    "meta": resolve(repo_root, scenario["init"]["meta"])}
        loaded[scenario["id"]] = {
            "spec": scenario, "a_path": a_path, "b_path": b_path, "init": init,
            "a_lines": load_phonebook_lines(a_path),
            "b_lines": load_phonebook_lines(b_path) if b_path else [],
        }
    return loaded


def plan(spec: dict, *, repo_root: Path, provenance: dict, spec_path: str | None = None) -> dict:
    validate_spec(spec)
    require_one = spec.get("require_one_update_per_epoch", True)
    config_path = resolve(repo_root, spec["model_config"])
    scenarios = _load_scenarios(spec, repo_root)
    inputs, initializations, init_lengths = {}, {}, {}
    for scenario_id, info in scenarios.items():
        for key in ("book_a", "book_b"):
            if key in info["spec"]:
                inputs[info["spec"][key]] = sha256(resolve(repo_root, info["spec"][key]))
        if info["init"]:
            init_lengths[scenario_id] = checkpoint_seq_len(info["init"]["checkpoint"])
            initializations[scenario_id] = {
                "checkpoint": info["spec"]["init"]["checkpoint"], "checkpoint_sha256": sha256(info["init"]["checkpoint"]),
                "meta": info["spec"]["init"]["meta"], "meta_sha256": sha256(info["init"]["meta"]),
                "context_length": init_lengths[scenario_id],
            }
            for key in ("checkpoint", "meta"):
                pinned = info["spec"]["init"].get(f"{key}_sha256")
                actual = initializations[scenario_id][f"{key}_sha256"]
                if pinned is not None and pinned != actual:
                    raise ValueError(f"{scenario_id}: {key} hash {actual} differs from the pinned {pinned}")

    runs = expand_runs(spec)
    for run in runs:
        info = scenarios[run["scenario"]]
        train_lines = build_mixture_lines(info["a_lines"], info["b_lines"], **run["train"])
        eval_lines = info["a_lines"] if run["eval_book"] == "A" else info["b_lines"]
        run["n_train"] = len(train_lines)
        run["updates_per_epoch"] = updates_per_epoch(len(train_lines), run["batch_size"])
        if require_one and run["updates_per_epoch"] != 1:
            raise ValueError(f"{run['id']}: {run['n_train']} training sequences with batch_size {run['batch_size']} "
                             f"give {run['updates_per_epoch']} updates per epoch, but the spec requires exactly one")
        run["seq_len"] = check_sequence_lengths(run["id"], train_lines, eval_lines,
                                                init_lengths.get(run["scenario"]) if run["adapt"] else None)
        for name in spec.get("evaluate_books", ["A", "B"]):
            lines = info["a_lines"] if name == "A" else info["b_lines"]
            if max(len(line) for line in lines) > run["seq_len"]:
                raise ValueError(f"{run['id']}: book {name} has a line longer than the model context {run['seq_len']}")

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "git": provenance,
        "spec_path": spec_path, "spec_sha256": canonical_sha256(spec), "spec": spec,
        "environment": environment(),
        "optimizer": {"name": "AdamW", "fresh_state_per_run": True,
                      "weight_decay": TrainConfig().weight_decay, "grad_clip": TrainConfig().grad_clip},
        "config": {"path": spec["model_config"], "sha256": sha256(config_path)},
        "inputs": inputs, "initializations": initializations, "runs": runs,
    }


# --------------------------------------------------------------------- execution


def _verify_unchanged(protocol: dict, repo_root: Path, prepared: dict, out_root: Path) -> None:
    for path, digest in protocol["inputs"].items():
        if sha256(resolve(repo_root, path)) != digest:
            raise ValueError(f"Input changed during the experiment: {path}")
    if sha256(resolve(repo_root, protocol["config"]["path"])) != protocol["config"]["sha256"]:
        raise ValueError("Model config changed during the experiment")
    for scenario, init in protocol["initializations"].items():
        if sha256(resolve(repo_root, init["checkpoint"])) != init["checkpoint_sha256"] \
                or sha256(resolve(repo_root, init["meta"])) != init["meta_sha256"]:
            raise ValueError(f"Initialization changed during the experiment: {scenario}")
    for key, record in prepared.items():
        data_dir = out_root / key.split("/")[0] / f"data_{key.split('/')[1]}"
        for name, digest in record["sha256"].items():
            if sha256(data_dir / name) != digest:
                raise ValueError(f"Prepared data changed during the experiment: {key}/{name}")


def run_spec(spec: dict, out_root: Path, *, repo_root: Path, provenance_fn=git_state,
             spec_path: str | None = None) -> dict:
    provenance = provenance_fn()
    if provenance["dirty"]:
        raise ValueError("A clean git tree is required: commit the code, inputs and spec before training.")
    out_root = Path(out_root).resolve()
    if out_root.exists():
        raise ValueError(f"Output root already exists; choose a new root to preserve evidence: {out_root}")
    protocol = plan(spec, repo_root=repo_root, provenance=provenance, spec_path=spec_path)
    scenarios = _load_scenarios(spec, repo_root)
    out_root.mkdir(parents=True)
    write_json(out_root / "protocol.json", protocol)

    prepared = {}
    for run in protocol["runs"]:
        key = f"{run['scenario']}/{run['mode']}"
        if key in prepared:
            continue
        info = scenarios[run["scenario"]]
        train_lines = build_mixture_lines(info["a_lines"], info["b_lines"], **run["train"])
        eval_lines = info["a_lines"] if run["eval_book"] == "A" else info["b_lines"]
        inputs_dir = out_root / run["scenario"] / f"inputs_{run['mode']}"
        inputs_dir.mkdir(parents=True)
        train_txt, eval_txt = inputs_dir / "train.txt", inputs_dir / "eval.txt"
        write_text_lf(train_txt, "".join(train_lines))
        write_text_lf(eval_txt, "".join(eval_lines))
        data_dir = out_root / run["scenario"] / f"data_{run['mode']}"
        command = [sys.executable, "phonebook/prepare_phonebook.py", "--train-input", str(train_txt),
                   "--eval-input", str(eval_txt), "--out-dir", str(data_dir)]
        subprocess.run(command, cwd=repo_root, check=True, capture_output=True, text=True)
        if info["init"] and Vocabulary.from_pickle(data_dir / "meta.pkl").stoi != \
                Vocabulary.from_pickle(info["init"]["meta"]).stoi:
            raise ValueError(f"{key}: prepared vocabulary differs from the initialization's vocabulary")
        prepared[key] = {"command": command, "train_sha256": sha256(train_txt), "eval_sha256": sha256(eval_txt),
                         "sha256": {name: sha256(data_dir / name) for name in ("train.bin", "val.bin", "meta.pkl")}}
    write_json(out_root / "prepared_data.json", prepared)

    books_by_scenario = {sid: {"A": info["a_path"], "B": info["b_path"]} for sid, info in scenarios.items()}
    evaluate_books = spec.get("evaluate_books", ["A", "B"])
    timeout = spec.get("timeout_seconds", 900)
    results, summary = {}, []
    total = len(protocol["runs"])
    for index, run in enumerate(protocol["runs"], start=1):
        if provenance_fn() != provenance:
            raise ValueError("Git state changed during the experiment; refusing to train.")
        _verify_unchanged(protocol, repo_root, prepared, out_root)
        info = scenarios[run["scenario"]]
        data_dir = out_root / run["scenario"] / f"data_{run['mode']}"
        out_dir = out_root / run["scenario"] / run["condition"]
        command = training_command(run, config=spec["model_config"], data_dir=data_dir, out_dir=out_dir,
                                   seed=spec["seed"], init_checkpoint=info["init"]["checkpoint"] if info["init"] else None)
        start = time.perf_counter()
        timed_out = False
        try:
            completed = subprocess.run(command, cwd=repo_root, capture_output=True, text=True, timeout=timeout)
            returncode, log = completed.returncode, completed.stdout + completed.stderr
        except subprocess.TimeoutExpired as exc:
            timed_out, returncode = True, None
            decode = lambda value: value.decode(errors="replace") if isinstance(value, bytes) else (value or "")
            log = decode(exc.stdout) + decode(exc.stderr) + f"\nTIMEOUT after {timeout} seconds\n"
        elapsed = time.perf_counter() - start
        log_path = out_root / run["scenario"] / f"{run['condition']}.log"
        write_text_lf(log_path, log)
        record = {"run": run, "command": command, "cwd": str(repo_root), "git": provenance,
                  "timestamp_utc": datetime.now(timezone.utc).isoformat(), "returncode": returncode,
                  "timed_out": timed_out, "elapsed_seconds": elapsed, "log_sha256": sha256(log_path)}
        with (out_root / "manifest.jsonl").open("a", encoding="utf-8", newline="\n") as file:
            file.write(json.dumps(record) + "\n")
        if timed_out:
            raise RuntimeError(f"Training timed out: {run['id']}; the failure is recorded in the manifest.")
        if returncode != 0:
            raise RuntimeError(f"Training failed: {run['id']} (see {log_path})")
        check_metrics(run, read_metrics(out_dir / "metrics.csv"))
        books = {name: books_by_scenario[run["scenario"]][name] for name in evaluate_books}
        result = evaluate_checkpoint(out_dir / "model.pth", data_dir / "meta.pkl", books)
        results[run["id"]] = result
        write_json(out_root / run["scenario"] / f"{run['condition']}.evaluation.json", result)
        summary.append(summary_row(run, result))
        counts = " ".join(f"{name} {result['books'][name]['generation']['correct']}/"
                          f"{result['books'][name]['generation']['total']}" for name in evaluate_books)
        print(f"[{index}/{total}] {run['id']}: {counts} ({elapsed:.0f}s)", flush=True)

    with (out_root / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(SUMMARY_COLUMNS)
        writer.writerows(summary)
    print(f"Evidence saved to {out_root}", flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out-root", type=Path)
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate the spec and print the protocol without training or requiring a clean tree.")
    args = parser.parse_args()
    spec = load_spec(args.spec)
    spec_path = str(args.spec.resolve().relative_to(ROOT_DIR)).replace("\\", "/") \
        if args.spec.resolve().is_relative_to(ROOT_DIR) else str(args.spec)
    if args.dry_run:
        print(json.dumps(plan(spec, repo_root=ROOT_DIR, provenance=git_state(), spec_path=spec_path), indent=2))
        return
    if args.out_root is None:
        parser.error("--out-root is required unless --dry-run is given")
    run_spec(spec, args.out_root, repo_root=ROOT_DIR, spec_path=spec_path)


if __name__ == "__main__":
    main()
