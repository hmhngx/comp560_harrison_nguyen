import importlib.util
import math
from pathlib import Path

import pytest
import torch

SOURCE = Path(__file__).resolve().parents[1] / "phonebook/docs/review-2026-10-21/analyze_probe_specificity.py"
spec = importlib.util.spec_from_file_location("probe_specificity", SOURCE)
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


def test_ranks_margins_and_wrong_name_control_match_hand_calculation():
    scores = torch.tensor([[.6, .3, .1], [.1, .7, .2], [.2, .2, .6]], dtype=torch.float64).log()
    rows = diagnostic.candidate_statistics(scores, [0, 1, 2])
    assert all(r["rank_min"] == r["rank_max"] == 1 for r in rows)
    assert rows[0]["margin_over_best_wrong_number"] == pytest.approx(math.log(2))
    assert rows[0]["target_log_probability_minus_wrong_name_logmean"] == pytest.approx(math.log(4))
    assert rows[1]["target_log_probability_minus_wrong_name_logmean"] == pytest.approx(math.log(.7/.25))


def test_near_ties_return_a_rank_interval_instead_of_a_winner():
    rows = diagnostic.candidate_statistics(torch.tensor([[-1., -1.000005, -2.], [-2., -1., -.5]], dtype=torch.float64), [0, 1])
    assert (rows[0]["rank_min"], rows[0]["rank_max"]) == (1, 2)
    assert (rows[1]["rank_min"], rows[1]["rank_max"]) == (2, 2)


def test_candidate_permutation_preserves_statistics_with_aligned_targets():
    scores = torch.tensor([[.6, .3, .1], [.1, .7, .2], [.2, .2, .6]], dtype=torch.float64).log()
    expected = diagnostic.candidate_statistics(scores, [0, 1, 2])
    actual = diagnostic.candidate_statistics(scores[:, [2, 0, 1]], [1, 2, 0])
    for a, b in zip(actual, expected, strict=True):
        assert a == pytest.approx(b)


@pytest.mark.parametrize("scores,targets", [
    (torch.zeros(3), [0, 1, 2]), (torch.zeros(3, 2), [0, 1]),
    (torch.zeros(1, 2), [0]), (torch.zeros(2, 1), [0, 0]),
    (torch.tensor([[0., float('nan')], [0., 0.]]), [0, 1]),
    (torch.zeros(2, 2), [True, 1]), (torch.zeros(2, 2), [2, 1]),
])
def test_invalid_matrices_and_target_indices_fail(scores, targets):
    with pytest.raises(ValueError):
        diagnostic.candidate_statistics(scores, targets)
