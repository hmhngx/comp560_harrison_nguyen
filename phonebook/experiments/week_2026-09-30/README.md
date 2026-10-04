# September 30–October 6: evaluation corrections and bounded replay pilot

Protocol written October 4 before running the new training conditions.

## Question

Does balanced replay's zero exact B recall at lr=0.0001, 100 updates persist
with 500 updates or lr=0.001? Can the same architecture memorize A+B from
fresh initialization under the tested budget?

The previous log compared L2-SP B token accuracy to replay B exact recall.
That comparison cannot establish a recall advantage. Existing seed42
checkpoints have 10/10 A and 0/10 B generation recall for both methods at
the cited full-retention settings. B token accuracy is 26.92% for L2-SP λ=2,
61.54% for replay k=1, and 53.08% for their tested combination.

## Protocol

- Fixed A and B: ten unique, disjoint names per book. Balanced replay is
  exactly A once followed by B once. Sources are version-controlled.
- Model: 64 dimensions, 2 heads, 2 layers. Batch size 64, CPU, seed42,
  default AdamW settings from the committed trainer, no L2-SP.
- Twenty training records give exactly one optimizer update per epoch.
- Four primary runs: replay lr=0.0001 at 100 and 500 updates; replay
  lr=0.001 at 100 updates; fresh joint A+B training lr=0.001 at 100 updates.
- Predeclared follow-ups: extend lr=0.001 replay and fresh joint training
  to 500 updates if either book lacks perfect generation at 100 updates.
  Every adaptation starts anew from the original Phase-A checkpoint.
- Evaluation: A and B separately, strict target-token match including
  newline; generation uses the context limit without target-length hints.
  Teacher-forced loss, token accuracy, sequence accuracy, counts, and raw
  generation predictions are retained independently for each book.
- Code must be committed and the tree clean before training. The runner
  refuses an existing output root, preserving all historical evidence.
- `protocol.json` records input/checkpoint hashes, code commit, environment,
  primary conditions, and extension triggers before training. Each run has
  an exact command, exit status, runtime, log, and hashed final evaluation.
- Only endpoint checkpoints are saved. The pilot cannot measure the first
  update reaching joint perfection, or validate generation at intermediate epochs.

## Reproduce

Run from the repository root in the recorded environment. Rebuild the
original Phase-A artifacts using `phonebook/README.md` if missing, and
verify their hashes against `protocol.json` before expecting identical results.

```powershell
python phonebook/run_replay_pilot.py --dry-run
python phonebook/run_replay_pilot.py --out-root phonebook/experiments/week_2026-09-30-rerun
```

The runner prepares new binary data directly from the tracked sources.
It additionally evaluates the original seed42 Phase-A, L2-SP λ=2, replay k=1/3/5,
and combination checkpoints, plus Phase-A seeds123/999 and their λ=5 L2-SP
checkpoints. These historical local checkpoint files must
exist for that audit. Their hashes identify the exact evaluated artifacts.

Checkpoints and prepared binaries are intentionally ignored. Small result
files are explicitly committed after the run, including the raw predictions,
summary, protocol, preparation record, logs, and run manifest.

## Results

Pending execution after regression tests and the full suite pass.

## Scope

One Phase-A checkpoint, one fixed twenty-entry task, two learning rates,
and two update budgets. This establishes only the observed behavior at
those settings. No claim about optimal replay weighting, general method
superiority, independent-dataset replication, or maximum model capacity follows.

Code, audit, and writing used Codex assistance; the measurements must be
checked against the committed machine-readable evidence.
