"""Bounded CPU development pilot. This CLI cannot launch a confirmation grid.

Uses one ordered full batch of distinct records per AdamW update. A source-byte
snapshot identifies dirty development code; it is not a clean committed study.
"""
from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.nn import functional as F

from completion_core.modeling import CompletionTransformer, ModelConfig, make_checkpoint_payload
from completion_core.objectives import weighted_token_cross_entropy
from completion_core.prep import encode_chars
from completion_core.training import SequenceDataset
from completion_core.validation import is_finite_number, is_torch_seed
from completion_core.vocabulary import Vocabulary
from phonebook.evaluate_replay import evaluate_partitions, audit_partition_report, _check_vocab, _vocab
from phonebook.replay_protocol import (canonical_digest, digest_bytes, read_json, membership,
                                      validate_training_input, write_protocol)

SCHEMA = "phonebook-development-pilot-v1"
DEFAULT_CONFIG = {"d_model": 64, "n_heads": 2, "n_layers": 2}
DEFAULT_SETTINGS = {"seed": 42, "lr": 0.001, "weight_decay": 0.1, "grad_clip": 1.0,
                    "primary_updates": 300, "secondary_updates": 100,
                    "sequence_budget": 6000, "evaluation_interval": 10,
                    "diagnostic_updates": [1, 100, 300]}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return digest_bytes(Path(path).read_bytes())


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def state_digest(state):
    # A serialization-independent fingerprint, including every parameter and its dtype/shape.
    return canonical_digest({name: {"dtype": str(value.dtype), "shape": list(value.shape),
                                    "bytes": digest_bytes(value.detach().cpu().contiguous().numpy().tobytes())}
                             for name, value in sorted(state.items())})


def validate_settings(settings, model_config, context):
    require(set(settings) == set(DEFAULT_SETTINGS), "Unexpected effective training settings")
    for key in ("primary_updates", "secondary_updates", "sequence_budget", "evaluation_interval"):
        require(type(settings[key]) is int and settings[key] > 0, f"Invalid {key}")
    require(is_torch_seed(settings["seed"]), "Invalid seed")
    require(settings["secondary_updates"] <= settings["primary_updates"], "Secondary endpoint exceeds primary")
    for key in ("lr", "weight_decay", "grad_clip"):
        require(is_finite_number(settings[key]) and settings[key] >= 0
                and (key != "lr" or settings[key] > 0), f"Invalid {key}")
    require(isinstance(settings["diagnostic_updates"], list)
            and all(type(i) is int and 0 < i <= settings["primary_updates"] for i in settings["diagnostic_updates"])
            and sorted(set(settings["diagnostic_updates"])) == settings["diagnostic_updates"], "Invalid diagnostic updates")
    require(set(model_config) == {"seq_len", "d_model", "n_heads", "n_layers"}
            and all(type(v) is int and v > 0 for v in model_config.values())
            and model_config["seq_len"] == context
            and model_config["d_model"] % model_config["n_heads"] == 0, "Invalid model configuration")


def tensor_batch(records, vocab, context):
    # Explicit encoding independently checked against the production binary dataset below.
    x = torch.full((len(records), context - 1), vocab.pad_id, dtype=torch.long)
    y = torch.full_like(x, -100)
    for index, record in enumerate(records):
        ids = torch.tensor([vocab.stoi[c] for c in record.line], dtype=torch.long)
        require(len(ids) <= context, "Record exceeds common context")
        x[index, :len(ids)-1] = ids[:-1]
        y[index, len(record.prompt):len(ids)-1] = ids[len(record.prompt)+1:]
    return x, y


def training_batch(selected, path, vocab, binary=None):
    validate_training_input(selected, path)
    lookup = {r.id: r for r in selected.a + selected.b}
    rows = [lookup[i] for i in selected.training_ids]
    x, y = tensor_batch(rows, vocab, selected.context_length)
    if binary is not None:
        require(Path(binary).read_bytes() == encode_chars(selected.training_text(), vocab.stoi).tobytes(),
                "Prepared binary differs from protocol training IDs")
        dataset = SequenceDataset(binary, vocab)
        require(dataset.max_len == selected.context_length and len(dataset) == len(rows), "Dataset shape differs")
        for i in range(len(rows)):
            xx, yy = dataset[i]
            require(torch.equal(xx, x[i]) and torch.equal(yy, y[i]), "Binary masks/order differ from explicit encoding")
    n_a = len(selected.replay_ids)
    counts = (y != -100).sum(dim=1)
    weights = torch.empty(len(rows), dtype=torch.float32)
    if n_a:
        weights[:n_a] = 0.5 / counts[:n_a].sum()
    weights[n_a:] = (0.5 if n_a else 1.0) / counts[n_a:].sum()
    return x, y, weights


def new_model(vocab, cfg):
    return CompletionTransformer(vocab.vocab_size, **cfg)


def preflight(protocol, protocol_pin, checkpoint, checkpoint_pin, meta, meta_pin,
              settings, model_config, *, repo_root=ROOT):
    require(sha(checkpoint) == checkpoint_pin and sha(meta) == meta_pin, "Initialization artifact pin differs")
    require(len(protocol["conditions"]) <= 6, "Development pilot allows at most six conditions")
    selected = [membership(protocol, c["id"], protocol_pin, repo_root=repo_root) for c in protocol["conditions"]]
    validate_settings(settings, model_config, protocol["context_length"])
    for item in selected:
        require(settings["sequence_budget"] // len(item.training_ids) >= settings["primary_updates"],
                "Sequence budget cannot include the mandatory primary endpoint")
    vocab = Vocabulary.from_pickle(Path(meta))
    _check_vocab(vocab, selected[0])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    require(isinstance(payload, dict) and canonical_digest(payload.get("model_config")) == canonical_digest(model_config),
            "Initialization must explicitly declare the exact context/dimensions/heads/layers")
    with torch.random.fork_rng(devices=[]):
        model = new_model(vocab, model_config)
        model.load_state_dict(payload["state_dict"], strict=True)
    require(all(t.is_floating_point() and t.dtype == torch.float32 and torch.isfinite(t).all()
                for t in payload["state_dict"].values()), "Expected finite float32 initialization")
    return selected, vocab, copy.deepcopy(payload["state_dict"])


def optimizer_for(model, settings):
    return torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"],
                             betas=(0.9, 0.999), eps=1e-8, amsgrad=False, foreach=False, fused=False)


def step(model, optimizer, x, y, weights, grad_clip):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = weighted_token_cross_entropy(model(x), y, weights)
    require(bool(torch.isfinite(loss)), "Nonfinite task loss")
    loss.backward()
    # Measure/check the norm even when clipping is disabled.
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip if grad_clip else float("inf"),
                                         error_if_nonfinite=True, foreach=False)
    optimizer.step()
    require(all(torch.isfinite(p).all() for p in model.parameters()), "Nonfinite parameter after update")
    return {"task_loss_before_update": loss.item(), "gradient_norm_before_clipping": norm.item()}


def local_diagnostic(before, after, vocab, selected):
    """Taylor terms on the fixed panel using an isolated model and actual AdamW delta.

    Analyst access to the panel is diagnostic only; no values enter the optimizer.
    Uses one Hessian-vector product for the symmetric Hessian quadratic.
    """
    model = copy.deepcopy(before).train()
    ids = set(selected.probe_ids)
    x, y = tensor_batch([r for r in selected.a if r.id in ids], vocab, selected.context_length)
    parameters = tuple(model.parameters())
    delta = tuple((b.detach() - a.detach()) for a, b in zip(before.parameters(), after.parameters(), strict=True))
    loss = F.cross_entropy(model(x).reshape(-1, vocab.vocab_size), y.reshape(-1))
    gradient = torch.autograd.grad(loss, parameters, create_graph=True)
    linear = sum((g * d).sum() for g, d in zip(gradient, delta, strict=True))
    hvp = torch.autograd.grad(linear, parameters)
    quadratic = 0.5 * sum((h * d).sum() for h, d in zip(hvp, delta, strict=True))
    with torch.no_grad():
        updated = copy.deepcopy(after).train()
        new_loss = F.cross_entropy(updated(x).reshape(-1, vocab.vocab_size), y.reshape(-1))
    actual = (new_loss - loss).item()
    return {"partition": "A_probe", "analyst_only": True, "loss_before": loss.item(),
            "loss_after": new_loss.item(), "actual_loss_change": actual,
            "linear_term": linear.item(), "quadratic_term": quadratic.item(),
            "second_order_residual": actual - linear.item() - quadratic.item(),
            "delta_l2": sum(d.square().sum() for d in delta).sqrt().item()}


def work(selected, update):
    rows = len(selected.training_ids)
    lookup = {r.id: r for r in selected.a + selected.b}
    targets = sum(len(lookup[i].target) + 1 for i in selected.training_ids)
    return {"updates": update, "training_sequences": update * rows,
            "supervised_tokens": update * targets,
            "padded_input_tokens": update * rows * (selected.context_length - 1)}


def endpoints(selected, settings):
    last = settings["sequence_budget"] // len(selected.training_ids)
    return sorted({0, settings["secondary_updates"], settings["primary_updates"], last})


def evaluation_updates(selected, settings):
    last = endpoints(selected, settings)[-1]
    return sorted(set(range(0, last + 1, settings["evaluation_interval"])) | set(endpoints(selected, settings)))


def sources(root):
    names = [p.relative_to(root).as_posix() for p in (root / "completion_core").glob("*.py")]
    names += ["phonebook/" + name for name in ("run_partial_replay.py", "evaluate_replay.py", "replay_protocol.py",
                                               "prepare_phonebook.py")]
    return {name: sha(root / name) for name in sorted(names)}


def run_condition(folder, selected, vocab, initial_state, model_config, settings):
    path = folder / "train.txt"
    binary = folder / "train.bin"
    binary.write_bytes(encode_chars(selected.training_text(), vocab.stoi).tobytes())
    x, y, weights = training_batch(selected, path, vocab, binary)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(settings["seed"])
        model = new_model(vocab, model_config)
        if selected.kind != "fresh_joint":
            model.load_state_dict(initial_state, strict=True)
        initial_digest = state_digest(model.state_dict())
        optimizer = optimizer_for(model, settings)  # No inherited momentum/state.
        require(not optimizer.state, "Optimizer must start empty")
        records, evaluations, diagnostics = [], [], []
        training_seconds = evaluation_seconds = diagnostic_seconds = 0.0
        last = endpoints(selected, settings)[-1]
        for update in range(last + 1):
            if update:
                before = copy.deepcopy(model) if update in settings["diagnostic_updates"] else None
                started = time.perf_counter()
                measurement = step(model, optimizer, x, y, weights, settings["grad_clip"])
                elapsed = time.perf_counter() - started
                training_seconds += elapsed
                records.append({**work(selected, update), **measurement, "elapsed_training_seconds": elapsed})
                if before is not None:
                    started = time.perf_counter()
                    # Disabling fused attention permits its second derivative in this PyTorch backend.
                    with torch.random.fork_rng(devices=[]), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                        diagnostics.append({"update": update, **local_diagnostic(before, model, vocab, selected)})
                    diagnostic_seconds += time.perf_counter() - started
            if update in evaluation_updates(selected, settings):
                started = time.perf_counter()
                evaluation = evaluate_partitions(model, vocab, selected, path)
                evaluation_seconds += time.perf_counter() - started
                row = {"work": work(selected, update), "evaluation": evaluation}
                evaluations.append(row)
                save_json(folder / f"evaluation-{update:04d}.json", row)
                if update == 0 and selected.kind != "fresh_joint":
                    require(evaluation["partitions"]["A_all"]["generation"]["correct"] == len(selected.a),
                            "Phase A must pass strict whole-book recall before adaptation")
            if update in endpoints(selected, settings):
                payload = make_checkpoint_payload(model, ModelConfig(**model_config))
                payload["replay_training"] = {"protocol_sha256": selected.protocol_sha256,
                                               "condition": selected.condition_id, "work": work(selected, update),
                                               "initial_state_sha256": initial_digest, "settings_sha256": canonical_digest(settings)}
                if update == last:
                    payload["optimizer_state_dict"] = optimizer.state_dict()
                torch.save(payload, folder / f"checkpoint-{update:04d}.pth")
        with (folder / "updates.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)
        report = {"condition": selected.condition_id, "initialization": "fresh" if selected.kind == "fresh_joint" else "phase_a",
                  "initial_state_sha256": initial_digest, "optimizer_started_empty": True,
                  "row_weights": weights.tolist(), "binary_sha256": sha(binary), "training_text_sha256": sha(path),
                  "work": work(selected, last), "sequence_budget_unused": settings["sequence_budget"] - last * len(x),
                  "training_seconds": training_seconds, "evaluation_seconds": evaluation_seconds,
                  "diagnostic_seconds": diagnostic_seconds, "diagnostics": diagnostics,
                  "parameter_count": sum(p.numel() for p in model.parameters()),
                  "parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
                  "optimizer_tensor_bytes": sum(v.numel() * v.element_size() for state in optimizer.state.values()
                                                for v in state.values() if isinstance(v, torch.Tensor)),
                  "retained_replay_records": len(selected.replay_ids), "anchor_tensor_bytes": 0,
                  "retained_replay_utf8_bytes": sum(len(r.line.encode()) for r in selected.a if r.id in selected.replay_ids)}
        save_json(folder / "run.json", report)
        return report


def run_pilot(protocol, protocol_pin, checkpoint, checkpoint_pin, meta, meta_pin, out_root,
              *, settings=None, model_config=None, repo_root=ROOT):
    root, out_root = Path(repo_root), Path(out_root)
    require(not out_root.exists(), "Output root must be unused")
    require(torch.get_num_threads() == 1, "Use exactly one CPU thread for deterministic development")
    settings = copy.deepcopy(DEFAULT_SETTINGS if settings is None else settings)
    model_config = dict(seq_len=protocol["context_length"], **DEFAULT_CONFIG) if model_config is None else dict(model_config)
    selected, vocab, initial = preflight(protocol, protocol_pin, checkpoint, checkpoint_pin, meta, meta_pin,
                                         settings, model_config, repo_root=root)
    source_hashes = sources(root)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True).stdout
    specification = {"schema": SCHEMA, "stage": "development_only", "protocol_sha256": protocol_pin,
                     "initial_checkpoint_sha256": checkpoint_pin, "initial_meta_sha256": meta_pin,
                     "model_config": model_config, "settings": settings, "source_sha256": source_hashes,
                     "base_commit": revision, "working_tree_dirty": bool(dirty.strip()),
                     "objective": "B-only: L_B; replay/joint: 0.5 token-mean L_A_subset + 0.5 token-mean L_B",
                     "order": "protocol training IDs, unique ordered full batch, no shuffle",
                     "optimizer": {"name": "AdamW", "betas": [0.9, 0.999], "eps": 1e-8, "amsgrad": False,
                                   "foreach": False, "fused": False, "l2sp_lambda": 0.0},
                     "sampling_limit": "Recall, minima, first/sustained recall and integral measured only at declared evaluation updates",
                     "confirmation_ready": False}
    spec_pin = canonical_digest(specification)
    out_root.mkdir(parents=True)
    save_json(out_root / "specification.json", specification)
    save_json(out_root / "execution.json", {"started_at_utc": datetime.now(timezone.utc).isoformat(),
              "argv": sys.argv, "python_executable": sys.executable, "python": platform.python_version(),
              "torch": torch.__version__, "numpy": np.__version__, "platform": platform.platform(),
              "threads": torch.get_num_threads(), "specification_sha256": spec_pin,
              "git_status_before": dirty, "device": "cpu", "dtype": "float32"})
    for name in source_hashes:
        target = out_root / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root / name).read_bytes())
    (out_root / "initial.pth").write_bytes(Path(checkpoint).read_bytes())
    (out_root / "meta.pkl").write_bytes(Path(meta).read_bytes())
    write_protocol(protocol, out_root / "prepared", repo_root=root)
    reports = []
    with torch.random.fork_rng(devices=[]):
        for item in selected:
            print(f"Training {item.condition_id}: {endpoints(item, settings)[-1]} updates", flush=True)
            reports.append(run_condition(out_root / "prepared" / item.condition_id, item, vocab, initial, model_config, settings))
    require(source_hashes == sources(root), "Source bytes changed during pilot")
    require(sha(checkpoint) == checkpoint_pin and sha(meta) == meta_pin, "Initialization changed during pilot")
    for item in selected:
        membership(protocol, item.condition_id, protocol_pin, repo_root=root)  # Sources unchanged.
    save_json(out_root / "complete.json", {"status": "training_completed_pending_audit", "specification_sha256": spec_pin,
              "completed_at_utc": datetime.now(timezone.utc).isoformat(), "conditions": [r["condition"] for r in reports]})
    return spec_pin


def compare_nested(actual, expected):
    if isinstance(expected, torch.Tensor):
        require(isinstance(actual, torch.Tensor) and actual.dtype == expected.dtype
                and actual.shape == expected.shape and torch.equal(actual, expected), "Recomputed tensor differs")
    elif isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), "Recomputed fields differ")
        for key in expected:
            compare_nested(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        require(type(actual) is type(expected) and len(actual) == len(expected), "Recomputed sequence differs")
        for a, b in zip(actual, expected, strict=True):
            compare_nested(a, b)
    else:
        require(type(actual) is type(expected) and actual == expected, "Recomputed value differs")


def audit_pilot(out_root, spec_pin, *, replay_training=False, repo_root=ROOT):
    """Readable reconciliation plus optional full deterministic training replay.

    A local re-execution verifies the saved recipe/weights/optimizer/answers. It
    does not authenticate who ran a job or an untrusted co-edited external pin.
    """
    out_root, root = Path(out_root), Path(repo_root)
    spec = read_json(out_root / "specification.json")
    require(canonical_digest(spec) == spec_pin and spec["schema"] == SCHEMA
            and spec["stage"] == "development_only" and spec["confirmation_ready"] is False, "Specification pin/stage differs")
    require(read_json(out_root / "complete.json")["specification_sha256"] == spec_pin, "Completion pin differs")
    for name, pin in spec["source_sha256"].items():
        require(sha(out_root / "source" / name) == pin, "Source snapshot differs")
        if replay_training:
            require(sha(root / name) == pin, "Local source differs from recorded execution")
    protocol = read_json(out_root / "prepared" / "protocol.json")
    selected, vocab, initial = preflight(protocol, spec["protocol_sha256"], out_root / "initial.pth",
                spec["initial_checkpoint_sha256"], out_root / "meta.pkl", spec["initial_meta_sha256"],
                spec["settings"], spec["model_config"], repo_root=root)
    settings = spec["settings"]
    audited = []
    require(torch.get_num_threads() == 1, "Audit needs one CPU thread")
    with torch.random.fork_rng(devices=[]):
        for item in selected:
            folder = out_root / "prepared" / item.condition_id
            report = read_json(folder / "run.json")
            require(report["condition"] == item.condition_id and report["optimizer_started_empty"] is True,
                    "Run identity/optimizer differs")
            x, y, weights = training_batch(item, folder / "train.txt", vocab, folder / "train.bin")
            require(report["binary_sha256"] == sha(folder / "train.bin")
                    and report["training_text_sha256"] == sha(folder / "train.txt")
                    and canonical_digest(report["row_weights"]) == canonical_digest(weights.tolist()), "Prepared input/weights differ")
            last = endpoints(item, settings)[-1]
            require(canonical_digest(report["work"]) == canonical_digest(work(item, last))
                    and type(report["sequence_budget_unused"]) is int
                    and report["sequence_budget_unused"] == settings["sequence_budget"] - last * len(x), "Work budget differs")
            for field in ("training_seconds", "evaluation_seconds", "diagnostic_seconds"):
                require(is_finite_number(report[field]) and report[field] >= 0, "Invalid duration")
            with (folder / "updates.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            require(len(rows) == last, "Missing optimizer updates")
            for update, row in enumerate(rows, 1):
                for key, value in work(item, update).items():
                    require(row[key] == str(value), "Update/work accounting differs")
                for field in ("task_loss_before_update", "gradient_norm_before_clipping", "elapsed_training_seconds"):
                    require(is_finite_number(float(row[field])) and float(row[field]) >= 0, "Nonfinite/negative update metric")
            torch.manual_seed(settings["seed"])
            model = new_model(vocab, spec["model_config"])
            if item.kind != "fresh_joint":
                model.load_state_dict(initial)
            require(report["initial_state_sha256"] == state_digest(model.state_dict()), "Initialization identity differs")
            optimizer = optimizer_for(model, settings)
            compare_nested(report["initialization"], "fresh" if item.kind == "fresh_joint" else "phase_a")
            compare_nested(report["parameter_count"], sum(p.numel() for p in model.parameters()))
            compare_nested(report["parameter_bytes"], sum(p.numel() * p.element_size() for p in model.parameters()))
            compare_nested(report["retained_replay_records"], len(item.replay_ids))
            compare_nested(report["retained_replay_utf8_bytes"], sum(len(r.line.encode()) for r in item.a if r.id in item.replay_ids))
            compare_nested(report["anchor_tensor_bytes"], 0)
            require(abs(sum(float(row["elapsed_training_seconds"]) for row in rows) - report["training_seconds"]) < 1e-9,
                    "Training duration sum differs")
            require([d["update"] for d in report["diagnostics"]] == settings["diagnostic_updates"], "Diagnostic coverage differs")
            for update in range(last + 1):
                if update and replay_training:
                    before = copy.deepcopy(model) if update in settings["diagnostic_updates"] else None
                    measurement = step(model, optimizer, x, y, weights, settings["grad_clip"])
                    for field, value in measurement.items():
                        require(float(rows[update-1][field]) == value, "Fresh optimizer metric differs")
                    if before is not None:
                        with torch.random.fork_rng(devices=[]), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                            diagnostic = {"update": update, **local_diagnostic(before, model, vocab, item)}
                        compare_nested(next(d for d in report["diagnostics"] if d["update"] == update), diagnostic)
                if update in evaluation_updates(item, settings):
                    saved = read_json(folder / f"evaluation-{update:04d}.json")
                    require(canonical_digest(saved["work"]) == canonical_digest(work(item, update)), "Evaluation update differs")
                    audit_partition_report(saved["evaluation"], item, folder / "train.txt")
                    if replay_training:
                        compare_nested(saved["evaluation"], evaluate_partitions(model, vocab, item, folder / "train.txt"))
                if update in endpoints(item, settings):
                    payload = torch.load(folder / f"checkpoint-{update:04d}.pth", map_location="cpu", weights_only=True)
                    compare_nested(payload["model_config"], spec["model_config"])
                    compare_nested(payload["replay_training"], {"protocol_sha256": item.protocol_sha256,
                        "condition": item.condition_id, "work": work(item, update),
                        "initial_state_sha256": report["initial_state_sha256"], "settings_sha256": canonical_digest(settings)})
                    if replay_training:
                        compare_nested(payload["state_dict"], model.state_dict())
                        if update == last:
                            compare_nested(payload["optimizer_state_dict"], optimizer.state_dict())
                            compare_nested(report["optimizer_tensor_bytes"], sum(v.numel() * v.element_size()
                                for state in optimizer.state.values() for v in state.values() if isinstance(v, torch.Tensor)))
                    else:
                        # Authenticate endpoint metrics with fresh inference even in the non-replay mode.
                        endpoint_model = new_model(vocab, spec["model_config"])
                        endpoint_model.load_state_dict(payload["state_dict"])
                        compare_nested(read_json(folder / f"evaluation-{update:04d}.json")["evaluation"],
                                       evaluate_partitions(endpoint_model, vocab, item, folder / "train.txt"))
            audited.append({"condition": item.condition_id, "updates": last,
                            "evaluation_points": len(evaluation_updates(item, settings)), "training_replayed": replay_training})
            print(f"Audited {item.condition_id} (training replay: {replay_training})", flush=True)
    return {"status": "passed", "specification_sha256": spec_pin,
            "training_recipe_reproduced": replay_training, "conditions": audited,
            "limitation": "Local deterministic recipe verification; external provenance and scientific validity remain separate"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    run = sub.add_parser("run")
    run.add_argument("--protocol", type=Path, required=True)
    run.add_argument("--protocol-sha256", required=True)
    run.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument("--checkpoint-sha256", required=True)
    run.add_argument("--meta", type=Path, required=True)
    run.add_argument("--meta-sha256", required=True)
    run.add_argument("--out-root", type=Path, required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("--root", type=Path, required=True)
    audit.add_argument("--specification-sha256", required=True)
    audit.add_argument("--replay-training", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.action == "run":
        pin = run_pilot(read_json(args.protocol), args.protocol_sha256, args.checkpoint, args.checkpoint_sha256,
                        args.meta, args.meta_sha256, args.out_root)
        print(f"Training complete. Specification pin: {pin}", flush=True)
    else:
        report = audit_pilot(args.root, args.specification_sha256, replay_training=args.replay_training)
        print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
