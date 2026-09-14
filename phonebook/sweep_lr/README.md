# Learning-rate sweep (phonebook Phase B adaptation)

Twelve original local runs exploring how the Phase-B adaptation learning
rate affects forgetting of phonebook A, plus a second, properly-logged
9-run follow-up (`seeded/`) that independently reproduces them and settles
an open question about seed variance. Read this file before citing any
number below in a report.

## Part 1 — the original 12 runs

**The exact commands that produced these were not logged anywhere and could
not be recovered from shell history** (checked both PowerShell and bash
history on 2026-09-09 — history stops at the initial `create_phonebook.py` /
train-eval split step, before any training was run). The protocol below was
reconstructed by matching metrics, not read from a log — see "Independent
reproduction" below for how this was subsequently confirmed.

### How the protocol was reconstructed

`out_lr_0.001/metrics.csv` is decimal-identical, across all 8 metric columns,
to [`phonebook/out_phaseB/metrics.csv`](../out_phaseB/metrics.csv). `out_phaseB`
is the Phase-B adaptation command documented in
[`phonebook/README.md`](../README.md):

```bash
py -u train.py --data-dir phonebook/data_phaseB --out-dir phonebook/out_phaseB \
    --adapt-from phonebook/out_phaseA/model.pth phonebook/config/phonebook.py
```

which uses `phonebook/config/phonebook.py`'s epochs=100 and `train.py`'s
default `--lr` (1e-3). The sweep runs are almost certainly this same command
with `--lr` (and, for the 500-epoch runs below, `--epochs 500`) overridden.

### Runs

| Folder | Learning rate | Epochs | Val token acc (retention of A) | Val seq acc (exact retention) |
|---|---|---|---|---|
| `out_lr_0.0001` | 0.0001 | 100 | 93.85% | 50.00% |
| `out_lr_0.0003` | 0.0003 | 100 | 64.62% | 0.00% |
| `out_lr_0.001`  | 0.001  | 100 | 41.54% | 0.00% |
| `out_lr_0.003`  | 0.003  | 100 | 30.77% | 0.00% |
| `out_lr_0.01`   | 0.01   | 100 | 22.31% | 0.00% |
| `diag_lr_0.0001_e500` | 0.0001  | 500 | 65.38% | 0.00% |
| `fine_lr_0.00012`     | 0.00012 | 500 | 65.38% | 0.00% |
| `fine_lr_0.00015`     | 0.00015 | 500 | 63.85% | 0.00% |
| `fine_lr_0.00018`     | 0.00018 | 500 | 60.00% | 0.00% |
| `fine_lr_0.00021`     | 0.00021 | 500 | 58.46% | 0.00% |
| `fine_lr_0.00025`     | 0.00025 | 500 | 57.69% | 0.00% |
| `fine_lr_0.0003`      | 0.0003  | 500 | 56.92% | 0.00% |

Finding: lower learning rate and fewer adaptation epochs both preserve more
of phonebook A (compare `out_lr_0.0001` at 100 epochs, 93.85% retained,
against `diag_lr_0.0001_e500` at the same learning rate but 500 epochs,
65.38% retained).

## Part 2 — `seeded/`: independent reproduction + a seed-variance dead end (2026-09-14)

Added [`phonebook/run_lr_sweep.py`](../run_lr_sweep.py), a driver script that
runs `train.py` per (learning rate, seed) combination and appends the exact
command, working directory, and timestamp for every run to
`seeded/manifest.jsonl`. This is the fix for Part 1's "commands not logged"
problem, going forward — nothing here is reconstructed.

Ran it for the three headline learning rates (0.0001, 0.001, 0.01) at 100
epochs, three seeds each (42, 123, 999) — 9 runs total, ~2 minutes wall
clock. Two results:

### Independent reproduction — confirmed

All three learning rates reproduced Part 1's numbers **exactly** (every
column, every epoch, not just the final row):

```
old out_lr_0.0001 (single-seed, unlogged):
100,1.2527,0.3872,0.6538,0.9385,0.0000,0.5000
new lr0.0001_ep100_seed{42,123,999} (logged, 3x):
100,1.2527,0.3872,0.6538,0.9385,0.0000,0.5000   (all three, identical)
```

This doesn't recover the *original* commands, but it's strong evidence the
reconstruction in Part 1 was correct — an independent run of the
reconstructed command lands on the same numbers.

### Seed has no effect on Phase-B adaptation here — verified, not assumed

**All three seeds at every learning rate produced byte-identical results —
every metric, every epoch, differing only in wall-clock training time.**
This was supposed to give error bars; instead it revealed the experiment as
currently designed is fully deterministic. Traced the cause:

1. Phonebook B has 10 training sequences; `phonebook/config/phonebook.py`
   sets `batch_size=64`. 10 < 64, so there is exactly **one batch per
   epoch**, every epoch — `shuffle=True` in the `DataLoader` has nothing to
   shuffle *between* batches, and order within a single all-inclusive batch
   doesn't change the loss/gradient computed from it.
2. `--adapt-from` loads a saved checkpoint's weights immediately after model
   construction, **overwriting** whatever the seed's random initialization
   produced — so the seed's other normal job is also moot here.
3. No CUDA (`torch.cuda.is_available()` → `False`, verified) and no dropout
   (`dropout=0.0` in `CompletionTransformer`) — CPU execution of this
   architecture has no remaining source of nondeterminism.

**Consequence:** "re-run with 3+ seeds for error bars" (the original Phase-0
plan item) is not achievable this way — it was based on an assumption that
didn't hold for this specific setup, not a mistake in execution. Reporting
that honestly rather than quietly re-running the same deterministic
computation three times and calling it a confidence interval.

**The correct fix, not yet done:** vary the seed used to train **Phase A**
itself (`train.py --data-dir phonebook/data_phaseA --out-dir
phonebook/out_phaseA_seed<N> --seed <N> phonebook/config/phonebook.py`, no
`--adapt-from`, so weight init genuinely differs by seed), producing several
different Phase-A checkpoints, then adapt each to Phase B. That asks a
different, arguably more interesting question — does *which* solution
Phase-A's training happened to converge to affect how much it forgets — and
is real future work, not something to fold into this week silently.

## Primary metric (locked in)

**Val sequence accuracy on phonebook A** is the primary forgetting metric
going forward. Val token accuracy is reported alongside it for context, but
token accuracy can look deceptively high from shared alphabet/character
structure even after genuine forgetting (e.g. `out_phaseB` retains 41.54%
token accuracy on A at the point where exact-entry recall has already
dropped to 0%) — sequence accuracy is the one that actually answers "does
the model still know the fact."

## Not committed

`model.pth` checkpoints are intentionally excluded, for both Part 1 and
`seeded/` (matches the repo's existing `*.pth` gitignore rule). Each run
retrains in well under a minute on CPU (see the `Training Time (sec)` column
in each `metrics.csv`, or `elapsed_sec` in `seeded/manifest.jsonl`), so
re-training is cheaper than storing the checkpoints.
