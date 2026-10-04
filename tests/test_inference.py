import sys

import pytest
import torch

from completion_core.inference import complete_sequence, evaluate_file, generate_completion, is_exact_completion
from completion_core.vocabulary import Vocabulary
from generate import parse_args


def vocabulary(tokens=None):
    tokens = tokens or sorted(set("abcA=\n_0123456789-"))
    stoi = {token: i for i, token in enumerate(tokens)}
    return Vocabulary(len(stoi), stoi, {i: token for token, i in stoi.items()})


class ScriptedModel:
    def __init__(self, vocab, output_tokens, prompt="abc"):
        self.vocab = vocab
        self.output = [vocab.stoi[token] for token in output_tokens]
        self.prefix_len = len(vocab.tokenize_string(prompt)) + 1
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        index = x.shape[1] - self.prefix_len
        next_id = self.output[index] if index < len(self.output) else self.vocab.eot_id
        logits = torch.full((1, x.shape[1], self.vocab.vocab_size), -1000.0)
        logits[0, -1, next_id] = 1
        return logits


NUMBER = "333-512-9823"


def test_default_budget_completes_phone_number_and_newline(tmp_path):
    vocab = vocabulary()
    model = ScriptedModel(vocab, list(NUMBER + "\n"))
    source = tmp_path / "eval.txt"
    source.write_text(f"abc={NUMBER}\n", encoding="utf-8")
    assert evaluate_file(source, model, vocab, None, 50, "cpu") == (1, 1)


@pytest.mark.parametrize("budget,reason", [(10, "budget"), (12, "budget"), (13, "newline")])
def test_explicit_budget_and_required_termination(budget, reason):
    vocab = vocabulary()
    model = ScriptedModel(vocab, list(NUMBER + "\n"))
    result = generate_completion("abc", model, vocab, budget, 50, "cpu")
    assert result.stop_reason == reason
    assert is_exact_completion(result, NUMBER, vocab) == (budget == 13)


@pytest.mark.parametrize("suffix", ["A\n", "_\n", "_A\n"])
def test_extra_characters_and_generated_padding_are_rejected(suffix):
    vocab = vocabulary()
    model = ScriptedModel(vocab, list(NUMBER + suffix))
    result = generate_completion("abc", model, vocab, None, 50, "cpu")
    assert result.stop_reason == "newline"
    assert not is_exact_completion(result, NUMBER, vocab)


def test_context_exhaustion_does_not_accept_unterminated_answer():
    vocab = vocabulary()
    model = ScriptedModel(vocab, list(NUMBER + "\n"))
    result = generate_completion("abc", model, vocab, None, 4 + len(NUMBER), "cpu")
    assert result.text == NUMBER
    assert result.stop_reason == "context"
    assert not is_exact_completion(result, NUMBER, vocab)


def test_prompt_at_context_limit_never_calls_model():
    vocab = vocabulary()
    model = ScriptedModel(vocab, list(NUMBER + "\n"))
    result = generate_completion("abc", model, vocab, None, 4, "cpu")
    assert result.stop_reason == "context"
    assert result.token_ids == ()
    assert model.calls == 0


def test_two_char_output_padding_is_matched_as_tokens():
    vocab = vocabulary(["ab", "c_", "ac", "_", "=", "\n"])
    model = ScriptedModel(vocab, ["ab", "c_", "\n"], prompt="ac")
    result = generate_completion("ac", model, vocab, None, 20, "cpu")
    assert is_exact_completion(result, "abc", vocab)
    assert complete_sequence("ac", model, vocab, 20, 20, "cpu") == "abc"


def test_two_char_evaluation_respects_explicit_budget(tmp_path):
    vocab = vocabulary(["ab", "c_", "ac", "_", "=", "\n"])
    model = ScriptedModel(vocab, ["ab", "c_", "\n"], prompt="ac")
    source = tmp_path / "eval.txt"
    source.write_text("ac=abc\n", encoding="utf-8")
    assert evaluate_file(source, model, vocab, 1, 20, "cpu") == (0, 1)
    assert evaluate_file(source, model, vocab, None, 20, "cpu") == (1, 1)


@pytest.mark.parametrize("budget", [0, -1])
def test_nonpositive_budget_rejected_in_api_and_cli(monkeypatch, budget):
    vocab = vocabulary()
    with pytest.raises(ValueError, match="positive"):
        generate_completion("abc", ScriptedModel(vocab, []), vocab, budget, 20, "cpu")
    monkeypatch.setattr(sys, "argv", ["generate.py", "--max-new-tokens", str(budget)])
    with pytest.raises(SystemExit) as exc:
        parse_args()
    assert exc.value.code == 2


def test_cli_default_uses_automatic_budget(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["generate.py"])
    assert parse_args().max_new_tokens is None
