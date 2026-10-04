import pytest
import torch

from completion_core.vocabulary import Vocabulary
from phonebook.evaluate_checkpoint import evaluate_book


class AlwaysOne(torch.nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab

    def forward(self, x):
        logits = torch.full((*x.shape, self.vocab.vocab_size), -1000.0)
        logits[:, :, self.vocab.eot_id] = 1
        logits[:, 1, self.vocab.eot_id] = -1000
        logits[:, 1, self.vocab.stoi["1"]] = 1
        return logits


def test_books_are_evaluated_separately_despite_repeated_a_entries(tmp_path):
    stoi = {c: i for i, c in enumerate("ab12_=\n")}
    vocab = Vocabulary(len(stoi), stoi, {i: c for c, i in stoi.items()})
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("a=1\na=1\na=1\n", encoding="utf-8")
    b.write_text("b=2\n", encoding="utf-8")
    model = AlwaysOne(vocab)
    a_result = evaluate_book(model, vocab, a, 5)
    b_result = evaluate_book(model, vocab, b, 5)
    assert a_result["generation"]["correct"] == 3
    assert b_result["generation"]["correct"] == 0
    assert b_result["generation"]["total"] == 1
    assert b_result["teacher_forced"]["token_correct"] == 1
    assert b_result["teacher_forced"]["token_total"] == 2
    assert b_result["teacher_forced"]["token_accuracy"] == 0.5
    assert b_result["teacher_forced"]["sequence_accuracy"] == 0.0


def test_empty_evaluation_book_is_an_error(tmp_path):
    stoi = {c: i for i, c in enumerate("_=\n")}
    vocab = Vocabulary(len(stoi), stoi, {i: c for c, i in stoi.items()})
    source = tmp_path / "empty.txt"
    source.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Evaluation book is empty"):
        evaluate_book(AlwaysOne(vocab), vocab, source, 5)
