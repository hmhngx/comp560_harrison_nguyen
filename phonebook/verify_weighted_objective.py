"""Verify replay objectives without training or changing historical artifacts.

This analysis tool is deliberately separate from the trainer. It compares every
parameter gradient on the configured phonebook transformer in CPU float64.
Run: python phonebook/verify_weighted_objective.py --out-root NEW_DIRECTORY
"""
from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from torch.nn import functional as F

from completion_core.modeling import CompletionTransformer, load_model_and_config
from completion_core.training import l2sp_penalty
from completion_core.objectives import weighted_token_cross_entropy
from completion_core.vocabulary import Vocabulary
from phonebook.prepare_phonebook import load_phonebook_lines

# Declared before running comparisons. These are coordinatewise tolerances;
# relative error alone is misleading for mathematically zero gradients.
ATOL = 1e-10
RTOL = 1e-9
REPETITIONS = (0, 1, 2, 3, 5, 9)
BOOK_FRACTIONS = (0.25, 0.5, 0.75)
L2SP_LAMBDA = 0.3


def completion_batch(lines, vocab, context_length):
    """Encode fixed-context fixtures; parity with SequenceDataset is tested."""
    x = torch.full((len(lines), context_length - 1), vocab.pad_id, dtype=torch.long)
    y = torch.full_like(x, -100)
    for row, line in enumerate(lines):
        if not line.endswith("\n") or line.count("=") != 1 or len(line) > context_length:
            raise ValueError("Fixture must be a newline-terminated completion within the common context")
        prompt, answer = line[:-1].split("=")
        if not prompt or not answer:
            raise ValueError("Fixture must have a nonempty prompt and answer")
        ids = torch.tensor([vocab.stoi[char] for char in line], dtype=torch.long)
        x[row, :len(ids) - 1] = ids[:-1]
        eq = len(prompt)
        y[row, eq:len(ids) - 1] = ids[eq + 1:]
    return x, y


def comparison(actual, reference):
    if actual.shape != reference.shape:
        raise ValueError("Comparison requires identical shapes; broadcasting would hide missing coordinates")
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(reference).all())
    difference = (actual - reference).abs()
    return {
        "passed": finite and bool(torch.all(difference <= ATOL + RTOL * reference.abs())),
        "max_abs_error": difference.max().item() if finite else None,
        "max_tolerance_fraction": (difference / (ATOL + RTOL * reference.abs())).max().item() if finite else None,
    }


def compare_objective(model, x, y, n_a, *, repetitions=None, fraction=None):
    """Return task/regularized comparisons, without mutating the supplied model."""
    if (repetitions is None) == (fraction is None):
        raise ValueError("Choose exactly one objective: repetitions or book fraction")
    if not 0 < n_a < len(x):
        raise ValueError("Both books must contain records")
    if repetitions is not None and (isinstance(repetitions, bool)
                                    or not isinstance(repetitions, int) or repetitions < 0):
        raise ValueError("repetitions must be an integer >= 0")
    if fraction is not None and (isinstance(fraction, bool)
                                 or not isinstance(fraction, (int, float)) or not 0 <= fraction <= 1):
        raise ValueError("fraction must be finite and in [0, 1]")
    reference, weighted = copy.deepcopy(model).train(), copy.deepcopy(model).train()
    if any((isinstance(module, nn.Dropout) and module.p != 0)
           or (isinstance(module, nn.MultiheadAttention) and module.dropout != 0)
           for module in model.modules()):
        raise ValueError("This check requires zero dropout and batch-independent model outputs")
    count_a = int((y[:n_a] != -100).sum())
    count_b = int((y[n_a:] != -100).sum())
    if min(count_a, count_b) <= 0:
        raise ValueError("Both books must have supervised tokens")
    weights = torch.ones(len(x), dtype=torch.float64)
    if repetitions is not None:
        indices = torch.cat([torch.arange(n_a), torch.arange(n_a, len(x)).repeat(repetitions)])
        logits = reference(x[indices])
        task_reference = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y[indices].reshape(-1))
        weights[n_a:] = repetitions
        objective_name = f"duplicate_B_{repetitions}"
    else:
        a_logits, b_logits = reference(x[:n_a]), reference(x[n_a:])
        a_mean = F.cross_entropy(a_logits.reshape(-1, a_logits.shape[-1]), y[:n_a].reshape(-1))
        b_mean = F.cross_entropy(b_logits.reshape(-1, b_logits.shape[-1]), y[n_a:].reshape(-1))
        task_reference = fraction * a_mean + (1 - fraction) * b_mean
        weights[:n_a] = fraction / count_a
        weights[n_a:] = (1 - fraction) / count_b
        objective_name = f"book_fraction_A_{fraction}"
    task_weighted = weighted_token_cross_entropy(weighted(x), y, weights)
    names = [name for name, _ in reference.named_parameters()]
    ref_params, weighted_params = tuple(reference.parameters()), tuple(weighted.parameters())
    # Deliberately nonzero penalty. Compare task gradients separately so a
    # shared large regularizer cannot conceal an incorrect task objective.
    anchor = {name: p.detach().clone() + 0.001 for name, p in model.named_parameters()}
    total_reference = task_reference + L2SP_LAMBDA * l2sp_penalty(reference, anchor)
    total_weighted = task_weighted + L2SP_LAMBDA * l2sp_penalty(weighted, anchor)
    scalars = {}
    parameters = []
    for kind, actual, expected in (("task", task_weighted, task_reference),
                                   ("with_l2sp", total_weighted, total_reference)):
        scalars[kind] = {"reference": expected.item() if torch.isfinite(expected) else None,
                         "weighted": actual.item() if torch.isfinite(actual) else None,
                         **comparison(actual, expected)}
        ref_grad = torch.autograd.grad(expected, ref_params, retain_graph=(kind == "task"))
        weighted_grad = torch.autograd.grad(actual, weighted_params, retain_graph=(kind == "task"))
        for name, actual_grad, expected_grad in zip(names, weighted_grad, ref_grad, strict=True):
            parameters.append({"objective": kind, "parameter": name, "elements": expected_grad.numel(),
                               **comparison(actual_grad, expected_grad)})
        if kind == "task":
            weighted_task_grad = weighted_grad
        else:
            for name, p, total_grad, task_grad in zip(names, weighted_params, weighted_grad,
                                                     weighted_task_grad, strict=True):
                derivative = 2 * L2SP_LAMBDA * (p.detach() - anchor[name])
                parameters.append({"objective": "l2sp_derivative", "parameter": name,
                                   "elements": p.numel(), **comparison(total_grad - task_grad, derivative)})
    return {
        "objective": objective_name, "a_rows": n_a, "b_rows": len(x) - n_a,
        "a_tokens": count_a, "b_tokens": count_b,
        "unique_rows": len(x), "reference_rows": len(x) if fraction is not None else n_a + repetitions * (len(x) - n_a),
        "row_weights": weights.tolist(), "scalars": scalars, "parameters": parameters,
        "passed": all(v["passed"] for v in scalars.values()) and all(v["passed"] for v in parameters),
    }


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_verification(repo_root=ROOT):
    repo_root = Path(repo_root)
    started = datetime.now(timezone.utc).isoformat()
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True,
                              text=True, check=True).stdout.strip()
    sources = ["phonebook/verify_weighted_objective.py", "completion_core/modeling.py",
               "completion_core/training.py", "completion_core/objectives.py", "completion_core/vocabulary.py", "completion_core/prep.py",
               "phonebook/prepare_phonebook.py",
               "phonebook/config/phonebook.py", "phonebook/specs/e3_replay_ratio.json"]
    artifacts = ["phonebook/out_phaseA/model.pth", "phonebook/data_phaseA/meta.pkl",
                 "phonebook/inputs/phonebookA.txt", "phonebook/inputs/phonebookB.txt"]
    source_hashes = {p: _sha(repo_root / p) for p in sources}
    artifact_hashes = {p: _sha(repo_root / p) for p in artifacts}
    checkpoint = repo_root / "phonebook/out_phaseA/model.pth"
    meta = repo_root / "phonebook/data_phaseA/meta.pkl"
    vocab = Vocabulary.from_pickle(meta)
    a = load_phonebook_lines(repo_root / "phonebook/inputs/phonebookA.txt")
    b = load_phonebook_lines(repo_root / "phonebook/inputs/phonebookB.txt")
    spec = json.loads((repo_root / "phonebook/specs/e3_replay_ratio.json").read_text(encoding="utf-8"))
    committed = subprocess.run(["git", "show", f"{revision}:phonebook/specs/e3_replay_ratio.json"], cwd=repo_root,
                               capture_output=True, text=True, check=True).stdout
    if spec != json.loads(committed):
        raise ValueError("Initialization declaration must match the spec in HEAD")
    init = next(s["init"] for s in spec["scenarios"] if s["id"] == "d0_seed42")
    if _sha(checkpoint) != init["checkpoint_sha256"] or _sha(meta) != init["meta_sha256"]:
        raise ValueError("Historical initialization does not match the committed experiment declaration")
    cases = []
    with torch.random.fork_rng(devices=[]):
        trained, config = load_model_and_config(checkpoint, vocab, "cpu", default_n_heads=2)
        if (config.d_model, config.n_heads, config.n_layers) != (64, 2, 2):
            raise ValueError("Expected the configured 64d/2head/2layer phonebook architecture")
        models = {"phase_a_float32_promoted_to_float64": trained.double()}
        for seed in (42, 123):
            torch.manual_seed(seed)
            models[f"random_seed{seed}"] = CompletionTransformer(vocab.vocab_size, config.seq_len,
                                                                 config.d_model, config.n_heads, config.n_layers).double()
        def shorten(lines, lengths):
            return [line.split("=")[0] + "=" + line.split("=")[1][:length] + "\n"
                    for line, length in zip(lines, lengths)]
        fixtures = {
            "original_phonebooks": (a, b),
            "unequal_target_lengths": (shorten(a, (1, 4, 8, 12)), shorten(b, (2, 6, 12))),
            "unequal_book_sizes": (a[:2], b),
        }
        for state, model in models.items():
            for fixture, (a_lines, b_lines) in fixtures.items():
                x, y = completion_batch(a_lines + b_lines, vocab, config.seq_len)
                choices = [{"repetitions": k} for k in REPETITIONS] + [{"fraction": f} for f in BOOK_FRACTIONS]
                for choice in choices:
                    result = compare_objective(model, x, y, len(a_lines), **choice)
                    result.update({"state": state, "fixture": fixture})
                    cases.append(result)
                print(f"{state}: {fixture}, {len(choices)} comparisons completed", flush=True)
    if source_hashes != {p: _sha(repo_root / p) for p in sources} \
            or artifact_hashes != {p: _sha(repo_root / p) for p in artifacts}:
        raise ValueError("Source or input bytes changed during verification")
    current_revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True,
                                      text=True, check=True).stdout.strip()
    if current_revision != revision:
        raise ValueError("Git HEAD changed during verification")
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=repo_root, capture_output=True,
                           text=True, check=True).stdout
    return {
        "status": "passed" if all(c["passed"] for c in cases) else "failed",
        "purpose": "Numerical objective/gradient verification; no optimizer steps or research training",
        "started_at_utc": started, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "atol": ATOL, "rtol": RTOL, "dtype": "float64", "device": "cpu", "l2sp_lambda": L2SP_LAMBDA,
        "model_config": asdict(config), "vocab_size": vocab.vocab_size,
        "random_initialization": f"Default {torch.get_default_dtype()} initialization, then promoted to float64",
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "threads": torch.get_num_threads()},
        "base_commit": revision, "working_tree_dirty": bool(dirty.strip()),
        "source_sha256": source_hashes, "input_sha256": artifact_hashes,
        "fixtures": {name: {"a_lines": aa, "b_lines": bb} for name, (aa, bb) in fixtures.items()},
        "cases": cases,
    }


def write_verification(report, out_root):
    out_root = Path(out_root)
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    out_root.mkdir(parents=True, exist_ok=False)
    (out_root / "verification.json").write_text(encoded, encoding="utf-8", newline="\n")
    columns = ["state", "fixture", "comparison", "objective", "parameter", "elements", "passed",
               "max_abs_error", "max_tolerance_fraction"]
    with (out_root / "parameter-comparisons.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for case in report["cases"]:
            for parameter in case["parameters"]:
                writer.writerow({"state": case["state"], "fixture": case["fixture"],
                                 "comparison": case["objective"], **parameter})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    if args.out_root.exists():
        parser.error("--out-root must be a new directory; existing evidence is never overwritten")
    torch.set_num_threads(1)
    report = run_verification()
    write_verification(report, args.out_root)
    print(f"{report['status']}: {len(report['cases'])} objective comparisons; {args.out_root}")
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
