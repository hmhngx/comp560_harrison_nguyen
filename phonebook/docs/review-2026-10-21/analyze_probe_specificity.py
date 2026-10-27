"""Post-hoc, closed-candidate diagnostic on saved checkpoints; zero training.

This does not change the declared exact-recall endpoint. Candidate likelihood,
rank and wrong-name controls measure different behavior from free generation.
"""
from datetime import datetime, timezone
import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F
from completion_core.modeling import load_model_and_config
from completion_core.vocabulary import Vocabulary
from phonebook.evaluate_replay import _check_vocab
from phonebook.replay_protocol import Record, membership, read_json, canonical_digest
from phonebook.report_partial_replay import verify_artifacts
from phonebook.run_partial_replay import tensor_batch, sha, require, save_json

SPEC_PIN = "d54a941e5cd794e590eadfc197395e55d4954988fdf5b030d8787f2af911d2f3"
MANIFEST_PIN = "7552d5fd34ca9e52497b52f310d19c683720b1a8fb7a78a8f6ddc0f424256720"
ATOL = RTOL = 1e-5  # Fixed before the new inference; logits use CPU float32.


def candidate_statistics(scores, target_indices, *, tie_atol=1e-5):
    """Larger log probability is better; ties produce an explicit rank interval."""
    require(scores.ndim == 2 and scores.shape[0] == len(target_indices)
            and len(target_indices) > 1 and scores.shape[1] > 1
            and bool(torch.isfinite(scores).all()), "Invalid candidate score matrix")
    rows = []
    for index, target in enumerate(target_indices):
        require(type(target) is int and 0 <= target < scores.shape[1], "Invalid target index")
        correct = scores[index, target].item()
        wrong = torch.cat((scores[index, :target], scores[index, target+1:]))
        best_wrong = wrong.max().item()
        ties = int((wrong - correct).abs().le(tie_atol).sum())
        rank_min = 1 + int((wrong > correct + tie_atol).sum())
        # Wrong-name baseline: same target number conditioned on the other
        # fixed probe names. Mean of probabilities, implemented in log space.
        other_queries = torch.cat((scores[:index, target], scores[index+1:, target]))
        wrong_name_logmean = (torch.logsumexp(other_queries, dim=0) - math.log(len(other_queries))).item()
        rows.append({"target_log_probability": correct, "rank_min": rank_min, "rank_max": rank_min+ties,
                     "margin_over_best_wrong_number": correct-best_wrong,
                     "target_log_probability_minus_wrong_name_logmean": correct-wrong_name_logmean})
    return rows


def score_candidates(model, queries, candidates, vocab, context):
    records = [Record(f"counterfactual-{i}-{j}", query.prompt, candidate.target)
               for i, query in enumerate(queries) for j, candidate in enumerate(candidates)]
    x, y = tensor_batch(records, vocab, context)
    model.eval()
    with torch.no_grad():
        logits = model(x)
        require(bool(torch.isfinite(logits).all()), "Nonfinite diagnostic logits")
        token_ce = F.cross_entropy(logits.reshape(-1, vocab.vocab_size), y.reshape(-1),
                                  ignore_index=-100, reduction="none").reshape_as(y)
    require(bool(((y != -100).sum(1) == 13).all()), "This declared phonebook diagnostic requires 13 targets per candidate")
    return -token_ce.double().sum(1).reshape(len(queries), len(candidates))


def main():
    root = ROOT / "phonebook/experiments/partial_replay_development_v1"
    out = Path(__file__).parent / "probe-specificity-exploration"
    require(not out.exists(), "Choose an unused diagnostic output directory")
    torch.set_num_threads(1)
    started = datetime.now(timezone.utc).isoformat()
    manifest = read_json(ROOT / "phonebook/docs/review-2026-10-20/partial-replay-results-v2/artifact-manifest.json")
    verify_artifacts(root, manifest, MANIFEST_PIN)
    spec = read_json(root / "specification.json")
    require(canonical_digest(spec) == SPEC_PIN, "Pilot specification differs")
    for name, pin in spec["source_sha256"].items():
        require(sha(ROOT / name) == pin, "Live inference/training dependency changed")
    protocol = read_json(root / "prepared/protocol.json")
    selected = membership(protocol, "plain_b", spec["protocol_sha256"])
    lookup = {r.id: r for r in selected.a + selected.b}
    queries = [lookup[i] for i in selected.probe_ids]
    candidates = list(selected.a + selected.b)
    require(len(candidates) == 20 and len(queries) == 3
            and len({r.target for r in candidates}) == 20, "Expected three fixed probes and 20 unique candidate numbers")
    target_indices = [next(i for i, r in enumerate(candidates) if r.id == q.id) for q in queries]
    vocab = Vocabulary.from_pickle(root / "meta.pkl")
    _check_vocab(vocab, selected)
    jobs = [("phase_a_reference", "plain_b", 0)] + [(c["id"], c["id"], 300) for c in protocol["conditions"]]
    rows, matrices = [], []
    with torch.random.fork_rng(devices=[]):
        for label, condition, update in jobs:
            folder = root / "prepared" / condition
            checkpoint = folder / f"checkpoint-{update:04d}.pth"
            saved = read_json(folder / f"evaluation-{update:04d}.json")["evaluation"]
            saved_rows = {r["id"]: r for r in saved["records"]}
            model, config = load_model_and_config(checkpoint, vocab, "cpu")
            require(config.seq_len == selected.context_length, "Diagnostic context changed")
            scores = score_candidates(model, queries, candidates, vocab, config.seq_len)
            statistics = candidate_statistics(scores, target_indices)
            for query_index, (query, target, stats) in enumerate(zip(queries, target_indices, statistics, strict=True)):
                source = saved_rows[query.id]
                ce = -scores[query_index, target].item()/13
                # Compare the same correct-target likelihood against the
                # independently saved full-book teacher evaluation. Different
                # batch shapes can introduce tiny float32 discrepancies.
                expected_ce = source["teacher"]["loss_sum"] / source["teacher"]["token_total"]
                require(math.isclose(ce, expected_ce, abs_tol=ATOL, rel_tol=RTOL), "Correct-candidate CE differs from saved teacher evaluation")
                if source["teacher"]["loss_sum"] < math.log(2):
                    require(stats["rank_min"] == stats["rank_max"] == 1, "A target with probability >1/2 must rank first")
                rows.append({"condition": label, "update": update, "probe_id": query.id,
                             "prompt": query.prompt, "expected": query.target, "free_prediction": source["generation"]["prediction"],
                             "free_exact": source["generation"]["correct"], "teacher_CE": expected_ce,
                             "teacher_token_correct": source["teacher"]["token_correct"], "teacher_token_total": 13,
                             **stats})
            matrices.append({"condition": label, "checkpoint_sha256": sha(checkpoint), "query_ids": list(selected.probe_ids),
                             "candidate_ids": [r.id for r in candidates], "log_probabilities": scores.tolist()})
    verify_artifacts(root, manifest, MANIFEST_PIN)
    report = {"stage": "post_hoc_development_diagnostic", "training_updates": 0,
              "started_at_utc": started, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
              "specification_sha256": SPEC_PIN, "artifact_manifest_sha256": MANIFEST_PIN,
              "script_sha256": sha(Path(__file__)), "correct_target_CE_tolerance": {"atol": ATOL, "rtol": RTOL},
              "tie_absolute_tolerance_log_probability": 1e-5, "rows": rows, "score_matrices": matrices,
              "limitations": ["Selected after inspecting existing teacher scores; exploratory, no revised primary outcome",
                  "One dataset, seed42, one memory draw and three probe entries",
                  "Closed 20-number candidate ranking differs from free generation",
                  "Wrong-name control uses only the other two fixed probe names",
                  "Teacher token accuracy includes punctuation/newline and uses true preceding digits",
                  "No relearning, memory-erasure proof, causal mechanism or novelty claim"]}
    out.mkdir()
    save_json(out / "diagnostic.json", report)
    with (out / "per-probe.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    for label, _, _ in jobs:
        chosen = [r for r in rows if r["condition"] == label]
        print(label, "free recall", sum(r["free_exact"] for r in chosen), "/3; candidate rank intervals",
              [(r["rank_min"], r["rank_max"]) for r in chosen], "; name advantages",
              [round(r["target_log_probability_minus_wrong_name_logmean"], 3) for r in chosen], flush=True)


if __name__ == "__main__":
    main()
