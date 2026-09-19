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

## Part 3 — L2-SP regularization sweep (2026-09-18)

Extended [`phonebook/run_lr_sweep.py`](../run_lr_sweep.py) with `--l2sp-lambdas`
(crossed with `--lrs` and `--seeds`) to test whether L2-SP regularization
(Li et al., 2018 — see [`phonebook/docs/mitigation-options.md`](../docs/mitigation-options.md)
and [`phonebook/docs/L2-SP-regularization-chat.md`](../docs/L2-SP-regularization-chat.md))
beats the LR-only ceiling from Part 1/2 (93.85% val token acc / 50.00% val seq
acc, `lr=0.0001`, 100 epochs). L2-SP adds λ·Σᵢ‖θᵢ−θᵢ_old‖² to the training
loss, where θ_old is a frozen snapshot of the Phase-A checkpoint taken right
after `--adapt-from` loads it.

Swept λ ∈ {0, 1e-4, 1e-3, 1e-2, 1e-1} at the same `lr=0.0001`, `epochs=100`,
`seed=42` as the established best LR-only run (λ=0 reproduces that run
exactly, byte-for-byte on every metric column):

```bash
python phonebook/run_lr_sweep.py --lrs 0.0001 --seeds 42 --l2sp-lambdas 0 0.0001 0.001 0.01 0.1
```

### Final-epoch (100) results

| λ | Val token acc | Val seq acc | Val loss | Train loss |
|---|---|---|---|---|
| 0 (baseline) | 93.85% | 50.00% | 0.3872 | 1.2527 |
| 1e-4 | 93.85% | 50.00% | 0.3871 | 1.2527 |
| 1e-3 | 93.85% | 50.00% | 0.3861 | 1.2528 |
| 1e-2 | 93.85% | 50.00% | 0.3765 | 1.2549 |
| 1e-1 | 93.85% | 50.00% | 0.2920 | 1.3239 |

### Finding: no λ in this range beats the LR-only ceiling at epoch 100 — but λ=0.1 visibly delays forgetting mid-training

**Val sequence accuracy at epoch 100 is 50.00% for every λ tested, identical
to the λ=0 baseline.** Val loss improves monotonically and substantially with
λ (0.3872 → 0.2920 at λ=0.1) and train loss degrades slightly (the penalty
trading off task fit, as expected), so the regularization is measurably doing
*something* — but none of it converts into retaining a 6th phonebook-A entry
beyond the 5/10 the baseline already retains at epoch 100.

The full per-epoch trajectory is more informative than the final row alone.
Comparing λ=0 to λ=0.1 val seq acc across epochs:

| Epoch | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|
| λ=0   | 100% | 100% | 90% | 80% | 70% | 70% | 70% | 70% | 50% | 50% |
| λ=0.1 | 100% | 100% | 90% | 80% | 80% | 90% | 70% | 70% | 60% | 50% |

λ=0.1 retains *more* of phonebook A at epochs 50-60 (80-90% vs. 70%) before
converging to the same 50% by epoch 100. This matches the "known limitation"
flagged in `mitigation-options.md` *before* this sweep was run: L2-SP
"penalizes parameter drift generically... may slow forgetting without fully
preventing it." Confirmed empirically here, not just a theoretical caveat —
at this λ range it slows forgetting but doesn't change the 100-epoch
endpoint.

**This is a single Phase-A checkpoint, one seed, one lr — not yet confirmed
across independent starting points.** Per Part 2, seed has no effect on
*this* Phase-B adaptation (it's fully deterministic), so re-running more
seeds here would not add real replication. Genuine replication requires
independently-trained Phase-A checkpoints (different seeds, no
`--adapt-from`) — deferred, tracked as an open item, not folded into this
result.

**Possible next question (not run this week):** does L2-SP combined with
early stopping (around epoch 50-60 instead of 100) beat the LR-only ceiling,
given λ=0.1's mid-training bump to 80-90%? The trajectory above suggests it
might, but that's a different experiment than the one scoped this week.

## Part 4 — early stopping + L2-SP: is the mid-training bump real, or just slower training? (2026-09-18)

Part 3 compared λ=0 and λ=0.1 **at the same epoch number** and found λ=0.1
retains more of phonebook A at epochs 50-60 (80-90% vs. 70%). That
comparison has a confound: λ=0.1's penalty competes with the task loss, so
it could simply be learning phonebook B *more slowly* -- in which case
"more of A retained at epoch 50" would just mean "less far along," not a
genuine property of L2-SP. It's also the wrong framing of the question:
early stopping alone, with **no** L2-SP, already retains 70-90% of A at
epochs 30-80 (Part 1's own finding that fewer adaptation epochs preserve
more of A), so "beats the 50% epoch-100 ceiling" is trivially true for
*any* early-stopped run, λ=0 included.

Re-ran λ=0 and λ=0.1 with `--accuracy-interval 1` for full per-epoch
resolution (Part 3 only logged every 10 epochs), then compared them at
**matched Phase-B learning progress** -- closest Train Loss, cross-checked
against closest Train Token Acc -- instead of matched epoch number. That's
the actual control for "is this just slower training."

```bash
python train.py --data-dir phonebook/data_phaseB --out-dir phonebook/sweep_lr/seeded/lr0.0001_ep100_seed42_l2sp0_finegrained --adapt-from phonebook/out_phaseA/model.pth phonebook/config/phonebook.py --lr 0.0001 --epochs 100 --seed 42 --l2sp-lambda 0 --accuracy-interval 1
python train.py --data-dir phonebook/data_phaseB --out-dir phonebook/sweep_lr/seeded/lr0.0001_ep100_seed42_l2sp0.1_finegrained --adapt-from phonebook/out_phaseA/model.pth phonebook/config/phonebook.py --lr 0.0001 --epochs 100 --seed 42 --l2sp-lambda 0.1 --accuracy-interval 1
```

**Result: mostly confound, but with a real, narrow exception.** For most of
training, matching on progress instead of epoch erases the gap entirely
(Δ = 0). But in one specific window -- train token acc ≈ 0.42-0.48,
λ=0.1's epochs ~55-65 -- there's a reproducible advantage that survives the
matched-progress control under *both* matching variables:

| λ=0.1 epoch | λ=0.1 train tok acc | λ=0.1 val seq acc | matched λ=0 epoch | λ=0 train tok acc | λ=0 val seq acc | Δ val seq acc |
|---|---|---|---|---|---|---|
| 55 | 0.4231 | 90% | 50 | 0.4231 | 70% | **+20pp** |
| 60 | 0.4538 | 90% | 59 | 0.4538 | 70% | **+20pp** |
| 65 | 0.4769 | 80% | 60 | 0.4846 | 70% | +10pp |

So L2-SP at λ=0.1 does do something beyond just slowing training down --
at matched learning progress, not just matched epoch -- but only in this
narrow window; everywhere else the matched-progress gap is zero, and both
runs converge to the same 50% floor by epoch 100 regardless of λ.

**Caveats, stated now, not after the fact:**
- `Val Seq Acc` only takes 10 discrete values (10 phonebook-A entries), so
  small deltas are coarse. A lone -10pp blip at epoch 90 under the
  token-acc matching (absent under the loss matching) is within that
  discretization noise, not a second effect worth chasing.
- This is one λ (0.1), one window, one Phase-A checkpoint, one seed, one
  lr -- the same single-checkpoint caveat as Part 3, plus a new one: the
  window's location/width for the *other* untested lambdas (1e-4, 1e-3,
  1e-2) is unknown without equally fine-grained runs for each.
- This is a real, cross-validated observation, not yet a claim: worth
  finer lambda resolution around 0.1 before treating "stop early with
  L2-SP" as a strategy rather than a two-point pattern.

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

## manifest.jsonl schema note

`manifest.jsonl`'s schema grew twice: `l2sp_lambda`/`timed_out`/`git_commit`/
`git_dirty` were added by the sweep-driver hardening and L2-SP work later
than the first 10 rows. Those rows were backfilled (2026-09-18) with
`l2sp_lambda: 0.0` and `timed_out: false` -- both definitionally correct,
since the rows predate the L2-SP flag existing and each already has a
recorded `returncode: 0`, which a timed-out run can't have. `git_commit` and
`git_dirty` were backfilled as `null`, not guessed: this file's whole point
is exact, non-reconstructed provenance, and a plausible-looking commit hash
inferred from a timestamp would be exactly the kind of fabrication it exists
to avoid. `null` means "not recorded when this run happened," nothing more.
