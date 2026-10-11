# Week of October 7–13: matched-optimization comparison and replication

Protocol written Saturday October 10, 2026, before any run below was executed.
Results are appended in later commits, so the history shows what was decided in advance.

## Why

Last week's [Part 10](../../sweep_lr/README.md#part-10--replay-learns-both-books-after-adjusting-optimization-settings)
showed that balanced replay reaches exact recall of both books at lr=0.001 and
100 updates, and that the earlier "replay cannot learn B" result came from
lr=0.0001. Every L2-SP result so far also used lr=0.0001, where no method learns
phonebook B within 100 updates. L2-SP has therefore never been tested where B is
learnable, and the L2-SP-versus-replay comparison is still unfair. This week closes that
gap, then checks whether the conclusions survive new phonebooks.

## Evidence standard

- Every experiment is a committed JSON spec in `phonebook/specs/`. The runner
  (`phonebook/run_experiment.py`) writes `protocol.json` (spec, code commit,
  input and checkpoint hashes, environment) before any training.
- The runner refuses a dirty git tree and an existing output folder.
- Phase-A starting checkpoints are pinned by SHA-256 in the spec. E1 uses the
  same three checkpoints as last week; the pins are copied from last week's protocol.
- Each run records its command, return code, runtime and log hash. Every endpoint
  is evaluated by strict generation (exact answer plus newline, no length hint) on
  both books separately.
- Per-update curves come from the trainer's own `metrics.csv` with
  `--accuracy-interval 1`. Its teacher-forced sequence accuracy equals strict
  generation recall (identical on all 40 book evaluations last week); the audit
  re-checks this on every endpoint here.
- Training is deterministic with constant learning rate, so a 100-update run is
  the first 100 rows of a 300-update run. `e1_prefix_check` verifies this directly.
- Small evidence files are committed with `git add -f` (the folder is ignored).
  Model weights, prepared binaries and the per-epoch logs stay local; their hashes
  are in the manifest.

## E1: L2-SP versus plain fine-tuning versus balanced replay at lr=0.001

**Question.** Can L2-SP, which needs no old data, retain phonebook A and also learn
phonebook B once the optimizer can learn B? How does it compare with plain
fine-tuning and with balanced replay?

**Design.** Three Phase-A checkpoints (seeds 42, 123, 999), books A and B unchanged
(ten entries each, [phonebookA.txt](../../inputs/phonebookA.txt),
[phonebookB.txt](../../inputs/phonebookB.txt)). Every run starts from the Phase-A
checkpoint, uses lr=0.001, 300 updates (one per epoch), AdamW defaults of the
trainer, Phase-B seed 42, batch size 64, and evaluates book A as the validation set.

| Condition | Training data | L2-SP λ |
|---|---|---|
| plain | B | 0 |
| l2sp_lam… | B | 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100 |
| replay_a1b1 | A once + B once | 0 |

That is 11 conditions × 3 checkpoints = 33 runs. [`e1_prefix_check`](../../specs/e1_prefix_check.json)
adds three 100-update runs on one checkpoint to confirm the prefix property.

**Decision rules (fixed now).**

- Primary endpoint: exact recall counts (of 10) for A and for B after 100 updates.
  Secondary: the same after 300 updates, and the full curves.
- L2-SP *matches replay* at a budget if one λ gives A=10/10 and B=10/10 on at least
  2 of the 3 checkpoints. It *nearly matches* if one λ gives A≥9/10 and B=10/10 on at
  least 2 of 3.
- The λ carried into the held-out replication (E2) is the one with the largest sum
  over the 3 checkpoints of (A + B) correct at 100 updates. Ties go to the larger
  summed A, then to the smaller λ.
- Every grid point is reported, including failures. Best-epoch-in-hindsight numbers
  are exploratory and never the headline.

**Reproduce.**

```bash
python phonebook/run_experiment.py --spec phonebook/specs/e1_matched_lr.json --dry-run
python phonebook/run_experiment.py --spec phonebook/specs/e1_matched_lr.json --out-root phonebook/experiments/week_2026-10-07/e1_matched_lr
python phonebook/run_experiment.py --spec phonebook/specs/e1_prefix_check.json --out-root phonebook/experiments/week_2026-10-07/e1_prefix_check
```

E2 (new phonebooks) and E3 (the README's repeated-B replay question) are specified
in later commits, each before its own runs.
