from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    sys.path.insert(0, str(ROOT / "phonebook" / "sweep_lr"))
    import analyze_matched_progress as module

    return module


def test_matched_progress_counts_match_the_readme_part_4_claims() -> None:
    """Locks in the exact nonzero-delta counts the README's Part 4 cites,
    so a future change to the committed _finegrained CSVs or the matching
    logic can't silently drift without this test catching it."""
    module = _load_module()
    base = module.load(module.BASE_CSV)
    l2sp = module.load(module.L2SP_CSV)
    assert len(base) == 100
    assert len(l2sp) == 100

    loss_deltas = module.matched_deltas(base, l2sp, "Train Loss")
    loss_nonzero = [d for d in loss_deltas if abs(d[4]) > 1e-9]
    assert len(loss_nonzero) == 23

    tok_deltas = module.matched_deltas(base, l2sp, "Train Token Acc")
    tok_nonzero = [d for d in tok_deltas if abs(d[4]) > 1e-9]
    assert len(tok_nonzero) == 19

    main_window = [d for d in loss_nonzero if 48 <= d[0] <= 65]
    second_window = [d for d in loss_nonzero if 88 <= d[0] <= 93]
    assert len(main_window) == 18
    assert len(second_window) == 5
    assert len(main_window) + len(second_window) == len(loss_nonzero), (
        "every loss-matched nonzero epoch should fall in one of the two "
        "documented clusters -- a new one outside both would mean the "
        "README's description of 'two clusters' is now incomplete"
    )

    plus10 = sum(1 for d in main_window if abs(d[4] * 100 - 10) < 0.01)
    plus20 = sum(1 for d in main_window if abs(d[4] * 100 - 20) < 0.01)
    assert (plus10, plus20) == (12, 6)

    tok_negative_epochs = sorted(d[0] for d in tok_nonzero if d[4] < 0)
    assert tok_negative_epochs == [26, 27, 31, 90]
