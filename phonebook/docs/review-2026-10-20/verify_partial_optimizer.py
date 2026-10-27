"""Float32 production-model check against separate book means and first-step AdamW."""
import copy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F
from phonebook import run_partial_replay as pilot


def main():
    torch.set_num_threads(1)
    root = ROOT / "phonebook/experiments/partial_replay_development_v1"
    spec = pilot.read_json(root / "specification.json")
    protocol = pilot.read_json(root / "prepared/protocol.json")
    selected, vocab, initial = pilot.preflight(protocol, spec["protocol_sha256"], root / "initial.pth",
        spec["initial_checkpoint_sha256"], root / "meta.pkl", spec["initial_meta_sha256"], spec["settings"], spec["model_config"])
    cases = []
    for item in selected:
        x, y, weights = pilot.training_batch(item, root / "prepared" / item.condition_id / "train.txt", vocab)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(spec["settings"]["seed"])
            actual = pilot.new_model(vocab, spec["model_config"])
            if item.kind != "fresh_joint":
                actual.load_state_dict(initial)
            reference = copy.deepcopy(actual)
            before = copy.deepcopy(actual.state_dict())
            n_a = len(item.replay_ids)
            task = pilot.weighted_token_cross_entropy(actual(x), y, weights)
            logits = reference(x[n_a:])
            expected = F.cross_entropy(logits.reshape(-1, vocab.vocab_size), y[n_a:].reshape(-1))
            if n_a:
                logits = reference(x[:n_a])
                expected = 0.5 * expected + 0.5 * F.cross_entropy(logits.reshape(-1, vocab.vocab_size), y[:n_a].reshape(-1))
            torch.testing.assert_close(task, expected, atol=1e-6, rtol=1e-5)
            task.backward()
            expected.backward()
            gradients = []
            for (name, p), q in zip(actual.named_parameters(), reference.parameters(), strict=True):
                torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-4)
                gradients.append({"parameter": name, "elements": p.numel(), "max_abs_gradient_difference": (p.grad-q.grad).abs().max().item()})
            for model in (actual, reference):
                torch.nn.utils.clip_grad_norm_(model.parameters(), spec["settings"]["grad_clip"], foreach=False)
                saved_grad = {name: p.grad.clone() for name, p in model.named_parameters()}
                pilot.optimizer_for(model, spec["settings"]).step()
                for name, p in model.named_parameters():
                    g = saved_grad[name]
                    formula = before[name] * (1-spec["settings"]["lr"]*spec["settings"]["weight_decay"])
                    formula -= spec["settings"]["lr"] * g / (g.abs()+1e-8)
                    torch.testing.assert_close(p, formula, atol=2e-7, rtol=1e-5)
            for row, p, q in zip(gradients, actual.parameters(), reference.parameters(), strict=True):
                row["max_abs_parameter_difference_after_step"] = (p-q).abs().max().item()
            with torch.no_grad():
                logit_gap = (actual(x)-reference(x)).abs().max().item()
            cases.append({"condition": item.condition_id, "scalar_difference": abs(task.item()-expected.item()),
                          "loss_and_all_gradients_pass": True, "both_analytic_AdamW_checks_pass": True,
                          "maximum_logit_difference_after_step": logit_gap, "parameters": gradients})
    report = {"status": "passed", "dtype": "float32", "device": "cpu", "model_config": spec["model_config"],
              "loss_tolerance": {"atol": 1e-6, "rtol": 1e-5}, "gradient_tolerance": {"atol": 1e-6, "rtol": 1e-4},
              "analytic_first_step_tolerance": {"atol": 2e-7, "rtol": 1e-5},
              "cross_implementation_step_identity_asserted": False,
              "cases": cases, "script_sha256": pilot.sha(Path(__file__)),
              "limitation": "One step, six development conditions; parameter differences do not establish functional trajectory differences"}
    pilot.save_json(Path(__file__).with_name("partial-optimizer-verification.json"), report)
    print("Passed", len(cases), "conditions;", sum(len(c["parameters"]) for c in cases), "parameter-gradient comparisons")
    print("Maximum after-step parameter gap", max(r["max_abs_parameter_difference_after_step"] for c in cases for r in c["parameters"]))
    print("Maximum after-step logit gap", max(c["maximum_logit_difference_after_step"] for c in cases))


if __name__ == "__main__":
    main()
