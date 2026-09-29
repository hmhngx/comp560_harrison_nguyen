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
to `phonebook/out_phaseB/metrics.csv` produced by running the Phase-B
adaptation command documented in [`phonebook/README.md`](../README.md) (not
linked directly: `out_phaseB` is a gitignored build artifact like every
`out_*` directory in this repo, reproducible via the command below, not
committed):

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

## Part 4 — early stopping + L2-SP: is the mid-training bump real, or just slower training? (2026-09-18, corrected 2026-09-18)

**This section replaces an earlier version of itself.** The first version of
this analysis had two real errors, found by an independent full-scan audit
and reproduced independently before rewriting anything below: its 3-row
table was actually built from Train-Token-Acc matching while the prose
called Train-Loss matching primary (the two disagree at epoch 55: Loss-
matching gives +10pp there, not the +20pp the old table stated), and the
underlying script only sampled every 5th epoch, which structurally could
not see two real clusters the full 100-epoch scan below finds. The
qualitative conclusion survives the correction; the specific numbers
below are the corrected ones. The old commit is `6244e5b` if the prior
version is needed for reference.

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
**matched Phase-B learning progress** instead of matched epoch number --
the actual control for "is this just slower training." Two independent
matching variables (Train Loss, Train Token Acc), reported separately
rather than conflated into one table, since they don't agree pointwise:

```bash
python train.py --data-dir phonebook/data_phaseB --out-dir phonebook/sweep_lr/seeded/lr0.0001_ep100_seed42_l2sp0_finegrained --adapt-from phonebook/out_phaseA/model.pth phonebook/config/phonebook.py --lr 0.0001 --epochs 100 --seed 42 --l2sp-lambda 0 --accuracy-interval 1
python train.py --data-dir phonebook/data_phaseB --out-dir phonebook/sweep_lr/seeded/lr0.0001_ep100_seed42_l2sp0.1_finegrained --adapt-from phonebook/out_phaseA/model.pth phonebook/config/phonebook.py --lr 0.0001 --epochs 100 --seed 42 --l2sp-lambda 0.1 --accuracy-interval 1
python phonebook/sweep_lr/analyze_matched_progress.py
```

**Result: mostly confound, with two real exceptions under Train-Loss
matching (the stated primary method).** Of 100 epochs, 23 show a nonzero
matched-progress delta, in two clusters -- everywhere else (77/100 epochs)
the gap is exactly zero, meaning the naive matched-*epoch* comparison in
Part 3 was mostly (not entirely) a training-speed artifact:

| Cluster | λ=0.1 epochs | Δ val seq acc |
|---|---|---|
| Main window | 48-65 (18 epochs) | +10pp (12 epochs), +20pp (6 epochs) |
| Second window | 88, 89, 91, 92, 93 | +10pp each |

Train-Token-Acc matching shows a broadly similar positive region (19/100
nonzero, concentrated at epochs 52-65) but does **not** agree pointwise with
Loss-matching -- it additionally shows small **negative** blips at epochs
26-27, 31, and 90 (-10pp each) that Loss-matching shows as exactly zero at
those same points. Given `Val Seq Acc` only takes 10 discrete values (10
phonebook-A entries), a handful of isolated ±10pp blips under only one of
two matching methods is within that discretization noise, not a second
effect worth chasing -- but it means "cross-checked against Token-Acc
matching" should be read as "broadly corroborates the main window exists,"
not "the two methods agree point for point."

So L2-SP at λ=0.1 does do something beyond just slowing training down --
at matched learning progress, not just matched epoch -- across a real
(if scattered) 23% of the training trajectory, most concentrated in a
main window around epochs 48-65. Both runs still converge to the same 50%
floor by epoch 100 regardless of λ.

**Caveats, stated now, not after the fact:**
- Full output, including every nonzero epoch and both matching variables,
  is reproducible with `python phonebook/sweep_lr/analyze_matched_progress.py`
  against the two committed `*_finegrained/metrics.csv` files -- nothing
  above is a summary of something that can't be checked directly.
- This is one λ (0.1), two windows, one Phase-A checkpoint, one seed, one
  lr -- the same single-checkpoint caveat as Part 3, plus a new one: the
  windows' location/width for the *other* untested lambdas (1e-4, 1e-3,
  1e-2) is unknown without equally fine-grained runs for each.
- This is a real, reproducible observation, not yet a claim: worth finer
  lambda resolution around 0.1 before treating "stop early with L2-SP" as
  a strategy rather than a pattern seen at one lambda value.

## Part 5 — multi-checkpoint replication: does L2-SP's benefit generalize? (2026-09-28)

Everything in Parts 3-4 used a single Phase-A checkpoint (seed=42). Trained
2 more, fully independent Phase-A checkpoints (`--seed 123`, `--seed 999`,
no `--adapt-from`, so weight initialization genuinely differs -- the
"correct fix, not yet done" flagged in Part 2) and re-ran the same λ sweep
against each, using `run_lr_sweep.py`'s new `--adapt-from` override:

```bash
python train.py --data-dir phonebook/data_phaseA --out-dir phonebook/out_phaseA_seed123 --seed 123 phonebook/config/phonebook.py
python train.py --data-dir phonebook/data_phaseA --out-dir phonebook/out_phaseA_seed999 --seed 999 phonebook/config/phonebook.py
python phonebook/run_lr_sweep.py --lrs 0.0001 --seeds 42 --l2sp-lambdas 0 0.0001 0.001 0.01 0.1 --adapt-from phonebook/out_phaseA_seed123/model.pth
python phonebook/run_lr_sweep.py --lrs 0.0001 --seeds 42 --l2sp-lambdas 0 0.0001 0.001 0.01 0.1 --adapt-from phonebook/out_phaseA_seed999/model.pth
```

Both new checkpoints converged normally (100%/100% train token/seq accuracy
on phonebook A, matching the original) and produced genuinely different Val
Token Acc during Phase-A training itself (16.15% / 18.46% / 20.00% across
seed 42/123/999) -- confirming these are independent solutions, not
accidentally identical weights.

### Result: neither the baseline forgetting severity nor L2-SP's benefit generalizes across checkpoints

Val Seq Acc at epoch 100, all 3 checkpoints x all 5 λ values:

| Checkpoint | λ=0 | λ=1e-4 | λ=1e-3 | λ=1e-2 | λ=1e-1 |
|---|---|---|---|---|---|
| seed42 (original) | 50% | 50% | 50% | 50% | 50% |
| seed123 (new) | 20% | 20% | 20% | 20% | 20% |
| seed999 (new) | 40% | 40% | 40% | 40% | **70%** |

Two findings, both real:

**1. The baseline "50% ceiling" (Parts 1-4) was specific to one checkpoint.**
With no L2-SP at all (λ=0), the three independent checkpoints retain 50%,
20%, and 40% of phonebook A after identical Phase-B adaptation. How badly a
given Phase-A solution forgets depends on *which* solution the optimizer
happened to converge to -- the "ceiling" documented throughout this file up
to now is this one checkpoint's ceiling, not a general property of the
setup.

**2. L2-SP's benefit at epoch 100 is inconsistent, not absent.** For 2 of 3
checkpoints (seed42, seed123), every tested λ produces *exactly* the same
Val Seq Acc as no regularization -- flat across the entire range. For the
third (seed999), nothing happens until λ=0.1, where retention jumps from
40% to 70%, a genuine +30pp improvement. Checked this isn't a fluke by
reading the full trajectory: the baseline keeps decaying through epoch 100
(90%→70%→70%→60%→40% from epoch 60 on) while λ=0.1 plateaus at 70% from
epoch 70 onward and holds there for the rest of training -- L2-SP visibly
resisting *further* drift once the model has already partly forgotten,
exactly the mechanism it's supposed to provide. This is a real, checkpoint-
specific effect, not the same transient mid-training bump seen with seed42
in Part 4 (that one fully reconverged to baseline by epoch 100; this one
does not).

### What this means for research direction

L2-SP is not a reliable fix in this setup: no effect for 2 of 3 independent
starting points across the whole tested λ range, and a large, real, but
*non-monotonic-looking* effect for the third that only appears at the single
largest λ tested. Two honest readings, not yet distinguished by this data:
- λ=0.1 is near some threshold, and untested larger values (0.2, 0.5, 1.0)
  might reveal effects for seed42/seed123 too -- this range just never
  went far enough.
- The effect is checkpoint-specific: something about *which* solution
  Phase-A converges to determines whether L2-SP can help at all,
  independent of λ.

Distinguishing these needs a wider λ range on all 3 checkpoints, not more
checkpoints at the existing range -- flagged as the next step, not run here,
to keep this replication scoped to what was actually asked: 2+ additional
checkpoints at the grid already established.

### Caveats

- Still one seed per checkpoint for Phase-B adaptation itself (proven
  inconsequential in Part 2 -- deterministic given a fixed Phase-A
  checkpoint -- so this isn't a new gap, just restated for completeness).
- Still one learning rate (0.0001, the established best from Part 1).
- 3 checkpoints is enough to show the effect *isn't* consistent; it is not
  enough to characterize the checkpoint-dependence itself (a 1-in-3 rate
  here could be anywhere from rare to common with a larger sample).

## Part 6 — answering Part 5's own open question: is λ=0.1 a threshold? (2026-09-29)

Part 5 left one question explicitly open: is λ=0.1 near some threshold that
larger untested values might cross for seed42 and seed123 too, or is
seed999's effect checkpoint-specific regardless of λ? Extended the grid to
λ ∈ {0.2, 0.5, 1.0} across all 3 checkpoints to find out, reusing the same
`--adapt-from` override:

```bash
python phonebook/run_lr_sweep.py --lrs 0.0001 --seeds 42 --l2sp-lambdas 0.2 0.5 1.0
python phonebook/run_lr_sweep.py --lrs 0.0001 --seeds 42 --l2sp-lambdas 0.2 0.5 1.0 --adapt-from phonebook/out_phaseA_seed123/model.pth
python phonebook/run_lr_sweep.py --lrs 0.0001 --seeds 42 --l2sp-lambdas 0.2 0.5 1.0 --adapt-from phonebook/out_phaseA_seed999/model.pth
```

No NaN/Inf anywhere in Train Loss across any of the 9 new runs, checked
directly before trusting anything else here -- λ=1.0 is 10x anything tested
before, worth confirming numerical stability rather than assuming it.

### Answer: λ=0.1 was not special. Every checkpoint has its own threshold, and all three cross it.

Val Seq Acc at epoch 100, extended:

| Checkpoint | λ=0 | λ=1e-4 | λ=1e-3 | λ=1e-2 | λ=0.1 | λ=0.2 | λ=0.5 | λ=1.0 |
|---|---|---|---|---|---|---|---|---|
| seed42 (original) | 50% | 50% | 50% | 50% | 50% | **80%** | **90%** | 90% |
| seed123 (new) | 20% | 20% | 20% | 20% | 20% | 20% | **50%** | **60%** |
| seed999 (new) | 40% | 40% | 40% | 40% | 70% | 70% | 80% | 80% |

**This revises Part 5's tentative conclusion.** Part 5's data alone couldn't
distinguish "λ=0.1 is a threshold" from "the effect is checkpoint-specific" --
this data does: every checkpoint eventually shows a large, genuine jump, just
at a different λ (seed999 crosses somewhere around 0.1, seed42 around 0.2,
seed123 around 0.5). Read as a per-checkpoint threshold effect, not absence
of an effect: seed42 and seed123 were never "immune" to L2-SP in Part 5, the
tested range (up to 0.1) simply hadn't reached their threshold yet.

Verified two of these jumps aren't single-epoch flukes by reading full
trajectories, the same check applied to seed999 in Part 5: seed42 at λ=0.5
reaches 90% by epoch 30 and holds exactly there for the remaining 8 logged
epochs (30 through 100, all 90%); seed123 at λ=1.0 settles at 60% from epoch
70 onward and holds for the rest of training. Same mechanistic signature in
both: partial early forgetting, then L2-SP arresting further drift once its
penalty dominates -- consistent with what Part 5 found for seed999, not a
different phenomenon.

### The honest other half: this is a trade-off, not a free improvement

Retention (Val Seq Acc on A) and how much of phonebook B actually gets
learned (Train Token Acc on B) move in opposite directions, consistently,
for all 3 checkpoints:

| Checkpoint | Train Tok Acc (B) at λ=0 | at λ=0.1 | at λ=0.5 | at λ=1.0 |
|---|---|---|---|---|
| seed42 | 65.38% | 64.62% | 48.46% | 36.15% |
| seed123 | 83.85% | 80.77% | 60.77% | 49.23% |
| seed999 | 86.92% | 84.62% | 69.23% | 50.00% |

Protecting phonebook A comes at a real, consistent cost to learning
phonebook B -- exactly the stability/plasticity trade-off L2-SP is
theoretically supposed to navigate, not a side effect to explain away. None
of the three checkpoints reach full (100%) retention even at λ=1.0
(they plateau at 90% / 60% / 80% respectively), so there's still some
forgetting L2-SP doesn't reach in this range.

### What this means for research direction (supersedes Part 5's verdict)

L2-SP does generalize as a mechanism across all 3 independent checkpoints --
Part 5's "not a reliable fix" verdict was accurate for the λ range tested at
the time, but incomplete. The revised, more useful framing: L2-SP needs a
larger λ than this project had been treating as the realistic range (0 to
0.1), and the effective value is checkpoint-dependent, not one fixed number
to tune once. A future user of this technique should sweep further than
0.1 by default, expect to tune per starting point, and weigh the retention
gain against the real cost to new-task learning rather than treating higher
λ as strictly better.

### Caveats

- Same single-seed/single-lr scope as Part 5 -- not restated in full, see
  Part 5's own caveats section.
- Only 3 λ points above 0.1 (0.2, 0.5, 1.0), log-spaced-ish rather than
  dense -- the three checkpoints' exact threshold values are bracketed, not
  pinpointed (e.g. seed42's crossing is known to be between 0.1 and 0.2,
  not narrowed further).
- Whether retention approaches 100% at even larger λ, or plateaus below it
  as these three plateaus suggest, is untested past 1.0 -- flagged as a
  further question, not run here.

## Part 7 — pinpointing the thresholds and testing past λ=1.0 (2026-09-29)

Part 6 left both of its own open questions unresolved: the three checkpoints'
crossing points were only bracketed, and behavior past λ=1.0 was untested.
Closed both: a finer grid inside each checkpoint's known bracket
(seed42: 0.125/0.15/0.175 inside 0.1-0.2; seed123: 0.25/0.3/0.4 inside
0.2-0.5; seed999: 0.02/0.05/0.07 inside 0.01-0.1), plus λ=2.0 and 5.0 for
all three. Checked for NaN/Inf in every one of the 15 new runs before
trusting anything else -- λ=5.0 is 5x the largest value tested before --
and confirmed all 10 logged epochs are present in each file. None found.

### Answer: all three checkpoints reach full retention -- as a smooth, monotonic climb, not a sudden jump

Val Seq Acc at epoch 100, full resolution:

| Checkpoint | ↑ climb | 100% reached at |
|---|---|---|
| seed42 | 50%(.1)→50%(.125)→60%(.15)→70%(.175)→80%(.2)→90%(.5)→90%(1.0)→**100%(2.0)** | λ=2.0 |
| seed123 | 20%(.2)→20%(.25)→40%(.3)→50%(.4)→50%(.5)→60%(1.0)→80%(2.0)→**100%(5.0)** | λ=5.0 |
| seed999 | 40%(.01)→40%(.02)→50%(.05)→70%(.07)→70%(.1)→70%(.2)→80%(.5)→80%(1.0)→80%(2.0)→**100%(5.0)** | λ=5.0 |

Every checkpoint climbs smoothly and monotonically once past its own starting
point -- not the step-function the coarser Part 6 grid made it look like.
Verified the seed42 λ=2.0 case isn't a last-epoch fluke by reading the full
trajectory: Val Seq Acc is already 100% by epoch 10 and holds there,
unmoving, through epoch 100 -- the strongest, cleanest plateau found in this
whole file. **Whether retention plateaus below 100% (Part 6's open question)
is answered: it does not. Given enough λ, all three checkpoints reach it.**

### But full retention has a real, steep cost -- not a free win

At the λ where each checkpoint first reaches 100% Val Seq Acc, Train Token
Acc on phonebook B (how much of the new task is actually being learned)
drops to roughly a quarter of its unregularized value, consistently:

| Checkpoint | Train Tok Acc (B) at λ=0 | at 100%-retention λ |
|---|---|---|
| seed42 | 65.38% | 26.92% (at λ=2.0) |
| seed123 | 83.85% | 23.85% (at λ=5.0) |
| seed999 | 86.92% | 27.69% (at λ=5.0) |

Not a degenerate "ignore B entirely" solution (that would look closer to
random-guess accuracy, well under 10% for this vocabulary) -- the model is
still learning some of B, just a small fraction of what it would without
the penalty. Full retention is achievable, but the honest framing is a
dial with a steep, real trade-off at its far end, not a setting to reach
for by default.

### Caveats

- Same single-seed/single-lr scope as Parts 5-6.
- The dose-response curve is now well-characterized in shape (smooth,
  monotonic) but still coarse in resolution near each crossing point --
  "60% climb between 0.15 and 0.2" is known; the exact half-integer-percent
  crossing is not, and wasn't the point of this pass.
- All three checkpoints happen to reach exactly 100% somewhere in {2.0, 5.0}
  -- untested whether this is a hard ceiling effect of this specific tiny
  10-entry phonebook-A task, or would hold at a different task scale.

## Part 8 — replay/rehearsal: the comparison mitigation-options.md called for a month ago (2026-09-29)

`phonebook/docs/mitigation-options.md` named replay as L2-SP's natural
second candidate ("pairs naturally with replay as a cheap second follow-up
-- giving a regularization-vs-rehearsal comparison") when L2-SP was first
chosen. Never built until now. Implemented via
[`phonebook/make_replay_input.py`](../make_replay_input.py): *k* copies of
phonebook A's lines mixed into Phase-B training, phonebook B once,
eval set unchanged (still phonebook A, for a direct comparison against
every L2-SP number above).

```bash
python phonebook/make_replay_input.py --phonebook-a phonebook/inputs/phonebookA.txt --phonebook-b phonebook/inputs/phonebookB.txt --k 1 --out phonebook/inputs/replay_k1.txt
python phonebook/prepare_phonebook.py --train-input phonebook/inputs/replay_k1.txt --eval-input phonebook/inputs/phonebookA.txt --out-dir phonebook/data_phaseB_replay_k1
python train.py --data-dir phonebook/data_phaseB_replay_k1 --out-dir phonebook/out_phaseB_replay_k1 --adapt-from phonebook/out_phaseA/model.pth phonebook/config/phonebook.py --lr 0.0001 --epochs 100 --seed 42
# (k=3, k=5 identical, --k/--out/--out-dir changed accordingly)
```

Tested k ∈ {1, 3, 5} against the seed42 checkpoint. All three stay within
the existing single-batch-per-epoch regime (10k+10 sequences ≤ batch_size
64), so -- unlike the batch-*count* confound `mitigation-options.md`
flagged in advance -- the actual variable changing between k values here is
batch *composition* (what fraction of the one batch is A vs. B), not batch
count. Worth being precise about which confound is actually in play rather
than citing the doc's advance concern without checking it against what was
actually run.

### Result: replay gives perfect, near-immediate retention -- and, independently verified, complete failure to learn B's specific entries

Val Seq Acc (phonebook A) hit 100% for all three k, including the smallest
(k=1, a 50/50 mix). Cross-checked this against `train.py`'s own logged
metric using a completely different evaluation method --
[`generate.py`](../../generate.py)'s true autoregressive generation
(feeding the model's own predictions back in, not teacher-forced against
the real prefix) -- and it agreed exactly: 100% on phonebook A for all
three k, via actual generation, not just the training loop's internal
metric.

The same independent check on phonebook B told a story the aggregate
training metrics couldn't: **0% exact-match accuracy on phonebook B, for
all three k**, via true generation. The model isn't producing truncated or
malformed output (initially got this via a too-short `--max-new-tokens`
budget on this tool's default, caught and corrected before trusting it --
phone numbers need 12+ generated characters, not the tool's default of 10)
-- it produces full-length, phone-number-*shaped* strings, just with the
wrong digits (e.g. expected `400-377-3079`, got `483-3-3722-2185`).

This is fully consistent with, not contradicted by, the training-set
metrics: Train Seq Acc on the mixed set was 50.00% / 75.00% / 83.33% for
k=1/3/5 -- which is *exactly* k/(k+1), the proportion of A-lines in each
mix (1/2, 3/4, 5/6). That equality isn't a coincidence; it's the same
"100% of A, 0% of B" result decomposed two ways, independently confirming
each other.

### What this means: replay and L2-SP are not interchangeable, and neither is simply "better"

Replay reaches 100% retention essentially for free in effort (no new
mechanism, no threshold to find) and at minimal k, where L2-SP needed a
checkpoint-specific λ between 2.0 and 5.0 to get there. But replay's cost
to B-learning here is total exact-match failure, not the partial-but-real
learning L2-SP left at its own 100%-retention point (24-28% Train Token
Acc, Part 7). Two different failure/success shapes for two different
mechanisms -- data access (replay) vs. a loss penalty (L2-SP) -- not two
points on the same trade-off curve.

### Caveats

- Only the seed42 checkpoint tested for replay -- unlike L2-SP, not yet
  replicated across seed123/seed999.
- Only k ∈ {1, 3, 5}, all within the single-batch regime; k large enough to
  cross into 2 batches/epoch (k≥6 here) is untested, and per
  `mitigation-options.md`'s original concern, that regime changes gradient
  step count too, not just batch composition -- a genuinely different
  confound than the one actually in play for k≤5.
- 100 epochs / lr=0.0001 is this project's established default, not
  re-tuned for the replay condition specifically -- whether more epochs
  (more gradient exposure to B, still only 1 step each) would eventually
  teach exact B entries under replay, the same way L2-SP needed larger λ
  rather than more epochs, is untested.
- No combination of replay + L2-SP together was tried.

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
