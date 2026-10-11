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

## E1 results

Run Saturday October 10 from clean code commit `e877957`: 33 runs, all return
code 0, about 45 seconds each. Evidence is in [`e1_matched_lr/`](e1_matched_lr/)
and [`e1_prefix_check/`](e1_prefix_check/); derived tables and figures are in
[`../../results/`](../../results/): [`e1_endpoints.csv`](../../results/e1_endpoints.csv),
[`e1_decision.json`](../../results/e1_decision.json),
[`e1_curves.png`](../../results/e1_curves.png),
[`e1_dose_response.png`](../../results/e1_dose_response.png).
Every cell below is entries recalled exactly (whole answer plus newline), out of ten
per book: A is the old book, B the new one.


**After 100 updates** (entries of book A and of book B, out of 10 each, recalled exactly)

| Condition | seed 42 | seed 123 | seed 999 |
|---|---|---|---|
| plain fine-tuning | A 0, B 10 | A 0, B 10 | A 0, B 10 |
| L2-SP λ=0.01 | A 0, B 10 | A 2, B 10 | A 2, B 10 |
| L2-SP λ=0.03 | A 1, B 10 | A 2, B 10 | A 3, B 10 |
| L2-SP λ=0.1 | A 1, B 10 | A 1, B 10 | A 2, B 10 |
| L2-SP λ=0.3 | A 3, B 4 | A 3, B 7 | A 2, B 8 |
| L2-SP λ=1 | A 9, B 0 | A 7, B 0 | A 8, B 0 |
| L2-SP λ=3 | A 10, B 0 | A 9, B 0 | A 10, B 0 |
| L2-SP λ=10 | A 10, B 0 | A 10, B 0 | A 10, B 0 |
| L2-SP λ=30 | A 10, B 0 | A 10, B 0 | A 10, B 0 |
| L2-SP λ=100 | A 10, B 0 | A 10, B 0 | A 10, B 0 |
| balanced replay (A once + B once) | A 10, B 10 | A 10, B 10 | A 10, B 10 |

**After 300 updates** (entries of book A and of book B, out of 10 each, recalled exactly)

| Condition | seed 42 | seed 123 | seed 999 |
|---|---|---|---|
| plain fine-tuning | A 0, B 10 | A 0, B 10 | A 0, B 10 |
| L2-SP λ=0.01 | A 1, B 10 | A 0, B 10 | A 4, B 10 |
| L2-SP λ=0.03 | A 0, B 10 | A 1, B 10 | A 1, B 10 |
| L2-SP λ=0.1 | A 1, B 10 | A 1, B 10 | A 1, B 10 |
| L2-SP λ=0.3 | A 4, B 8 | A 3, B 9 | A 3, B 8 |
| L2-SP λ=1 | A 9, B 0 | A 6, B 0 | A 8, B 0 |
| L2-SP λ=3 | A 10, B 0 | A 8, B 0 | A 10, B 0 |
| L2-SP λ=10 | A 10, B 0 | A 10, B 0 | A 10, B 0 |
| L2-SP λ=30 | A 10, B 0 | A 10, B 0 | A 10, B 0 |
| L2-SP λ=100 | A 10, B 0 | A 10, B 0 | A 10, B 0 |
| balanced replay (A once + B once) | A 10, B 10 | A 10, B 10 | A 10, B 10 |

**Timing** (update numbers)

| Condition | seed 42 | seed 123 | seed 999 |
|---|---|---|---|
| plain fine-tuning: B first 10/10 | 44 | 34 | 31 |
| plain fine-tuning: lowest A | 0 (update 17) | 0 (update 12) | 0 (update 12) |
| balanced replay (A once + B once): B first 10/10 | 51 | 41 | 33 |
| balanced replay (A once + B once): lowest A | 8 (update 4) | 8 (update 3) | 8 (update 2) |
| balanced replay: both books 10/10 | 51 | 41 | 33 |

**Best L2-SP compromise (largest A + B)**

- after 100 updates: seed 42: 11/20 (λ=0.03, 0.1); seed 123: 12/20 (λ=0.01, 0.03); seed 999: 13/20 (λ=0.03)
- after 300 updates: seed 42: 12/20 (λ=0.3); seed 123: 12/20 (λ=0.3); seed 999: 14/20 (λ=0.01)

![Exact recall of A and B after every update, three checkpoints](../../results/e1_curves.png)

![Recall after 100 and 300 updates as lambda grows](../../results/e1_dose_response.png)

**Reading the table.**

1. *Plain fine-tuning forgets A before it learns B.* A is down to 0 of 10 by update
   17, 12 and 12 on the three checkpoints; B first reaches 10 of 10 at update 44, 34
   and 31. At 100 and at 300 updates the result is A 0, B 10 on every checkpoint.
2. *L2-SP trades one book for the other along a single axis of λ.*
   - λ ≤ 0.1 behaves like plain fine-tuning: B is complete by update 31 to 65, and A
     ends at 0 to 4 of 10 after 300 updates.
   - λ = 0.3 is the only intermediate point: A 3 to 4 and B 8 to 9 after 300 updates
     (B reaches 10 only on the seed 123 checkpoint, at update 274).
   - λ ≥ 1 keeps most or all of A (5 to 10 of 10 over the 300 updates at λ=1; always 8 or
     more at λ=3; 10 at λ ≥ 10) but learns essentially none of B: at most 1 of 10 at any
     single update.
3. *Pre-registered decision: no λ matches replay.* No λ gives A ≥ 9 together with B = 10
   on any of the three checkpoints at 100 or at 300 updates, so every λ fails both the
   "matches" and the "nearly matches" test
   ([`e1_decision.json`](../../results/e1_decision.json)). The best L2-SP compromise,
   the largest A + B on each checkpoint, is 11 to 13 of 20 entries after 100 updates and
   12 to 14 after 300, against 20 of 20 for balanced replay.
4. *Balanced replay reaches 10 of 10 on both books* at update 51, 41 and 33, after A
   dips to 8 of 10 for a few updates (lowest A at update 4, 3 and 2).

**Checks.**

- [`audit_experiment.py`](../../audit_experiment.py) passes on both evidence folders
  with local artifacts: it re-generated all 660 + 60 completions from the saved
  checkpoints and they equal the saved strings; spec and config equal the versions
  committed at `e877957`; Phase-A checkpoint hashes equal the pins.
- The trainer's logged teacher-forced recall equals strict generation recall at the
  endpoint of all 33 runs (checked inside the audit).
- `e1_prefix_check`: the 100-update runs equal the first 100 rows of the 300-update
  runs on all three conditions (100 epochs compared each, no mismatch, timing column
  ignored), so the 100-update numbers above come from the long runs legitimately.

**What this does not show.** One fixed pair of phonebooks (E2 tests new ones); one
learning rate; the trainer's AdamW defaults including weight decay 0.1; ten entries
per book, so counts move in tenths. L2-SP here is the isotropic penalty from the
mitigation notes; a Fisher-weighted penalty (EWC) was not tested. A "no" for L2-SP
at these settings is not a claim about every regularizer.

## E1b: a finer λ grid and a longer horizon (follow-up to E1)

*Added Saturday October 10, after E1's runs finished and before any E1b run. Not part of E1's original protocol.*

E1's half-decade λ grid is coarse next to the range where L2-SP stops learning B,
and 300 updates may be short for strong penalties. A "no L2-SP setting matches
replay" conclusion should survive a finer grid and a longer horizon. E1b runs
λ ∈ {0.3, 0.4, …, 1.0} on the same three checkpoints for 1000 updates, logged every
5 updates ([`e1b_fine_grid_long_horizon.json`](../../specs/e1b_fine_grid_long_horizon.json)).

**Decision rule.** E1's rule, unchanged, applied at 100, 300 and 1000 updates. The
100- and 300-update budgets use the union of E1's and E1b's λ values; the 1000-update
budget uses E1b's. **Cross-check:** the λ=0.3 and λ=1.0 rows at updates 5, 10, …, 300
must equal E1's rows at the same updates.

## E2: does E1 hold on new phonebooks?

*Added Saturday October 10, before any E2 run.*

E1 used one fixed pair of phonebooks. E2 repeats E1 exactly (same 11 conditions,
learning rate, updates, per-update logging and decision rules) on three new 20-entry
phonebooks. The whole grid is repeated, so no λ is carried over and E1's
"λ for replication" rule is not needed. This changes no E1 result or rule.

**Phonebooks.** Made with the README's own generator (`create_phonebook.py 20 --seed S`);
the first ten entries are book A and the last ten book B. A seed qualifies only if A
and B have the same longest line, because the trainer sizes its context from the
training book alone. Seed 1 crashes Phase A (longest lines 28 and 31); seed 3 trains,
but its checkpoint cannot be loaded for Phase B (28 versus 29). Seeds 2, 4 and 7 are
the first three that qualify; the choice depends only on name lengths, never on a
result. [`make_heldout_dataset.py`](../../make_heldout_dataset.py) enforces the rule.

**Phase A.** One fresh Phase-A model per phonebook (default config, lr=0.001, 100
updates, seed 42): [`e2_phase_a.json`](../../specs/e2_phase_a.json). Gate: each must
recall all ten entries of its own book A exactly; otherwise it is retrained from
scratch with 200, then 400 updates, keeping every attempt. The E2 spec pins the
resulting checkpoint hashes.

**Replication criteria (fixed now).** E1's conclusions replicate if, on at least 2 of
the 3 new phonebooks: (i) balanced replay recalls 10/10 of both books after 100
updates; (ii) plain fine-tuning ends 300 updates with B=10/10 and at most 2 of A
recalled; and (iii) the L2-SP decision rule, applied to the 3 new phonebooks over the
same grid, gives the same verdict as E1 at both 100 and 300 updates.

## E3: the README's replay question, A once plus B repeated k times

*Added Saturday October 10, before any E3 run.*

The phonebook README asks: if the replay data holds one copy of A and k copies of B,
how does k affect forgetting, and which k permits the fastest learning of 100%
accuracy on all data? E1's `replay_a1b1` is the k=1 case.

**Design.** The three E1 checkpoints (hash-pinned) × k ∈ {1, 2, 3, 5, 9} ×
lr ∈ {0.003, 0.001, 0.0003, 0.0001} = 60 runs, 500 updates, batch size 128 (so even
k=9, 100 sequences, is one update per epoch), per-update logging
([`e3_replay_ratio.json`](../../specs/e3_replay_ratio.json)).

**Metrics and rules (fixed now).**

- *first_joint*: first update at which every training sequence, hence both books, is
  recalled exactly. *stable_joint*: first update from which every logged update through
  update 500 has both books recalled exactly; runs that never get there are censored and
  count as infinitely slow.
- For each lr, k\* is the k with the smallest median stable_joint over the three
  checkpoints; ties go to the smaller k; if every k is censored there is no k\*. The
  question is whether k\*>1 for any lr.
- *Dip*: the minimum number of A entries recalled at any update, and the update where it
  first occurs.
- **Cross-checks.** k=1 at lr=0.001 must reproduce E1's `replay_a1b1` curves exactly
  (batch size 64 versus 128 is irrelevant for a single batch). k=1 at lr=0.0001 must
  reproduce last week's 100-update results: B recalled 0/10, 1/10 and 0/10 on the three
  checkpoints.
- Every cell is reported, including censored ones.

**Reproduce.**

```bash
python phonebook/run_experiment.py --spec phonebook/specs/e1b_fine_grid_long_horizon.json --out-root phonebook/experiments/week_2026-10-07/e1b_fine_grid_long_horizon
python phonebook/run_experiment.py --spec phonebook/specs/e2_phase_a.json --out-root phonebook/experiments/week_2026-10-07/e2_phase_a
python phonebook/run_experiment.py --spec phonebook/specs/e2_heldout_phonebooks.json --out-root phonebook/experiments/week_2026-10-07/e2_heldout_phonebooks
python phonebook/run_experiment.py --spec phonebook/specs/e3_replay_ratio.json --out-root phonebook/experiments/week_2026-10-07/e3_replay_ratio
```

**Audit and derived results** (after the runs; every command reads only the evidence folders):

```bash
python phonebook/audit_experiment.py --root phonebook/experiments/week_2026-10-07/e1_matched_lr --require-local-artifacts
python phonebook/report_experiments.py grid --root phonebook/experiments/week_2026-10-07/e1_matched_lr --out phonebook/results --prefix e1
python phonebook/report_experiments.py e1b --e1 phonebook/experiments/week_2026-10-07/e1_matched_lr --e1b phonebook/experiments/week_2026-10-07/e1b_fine_grid_long_horizon --out phonebook/results
python phonebook/report_experiments.py e2 --root phonebook/experiments/week_2026-10-07/e2_heldout_phonebooks --reference phonebook/results/e1_decision.json --out phonebook/results
python phonebook/report_experiments.py e3 --root phonebook/experiments/week_2026-10-07/e3_replay_ratio --e1 phonebook/experiments/week_2026-10-07/e1_matched_lr --out phonebook/results
```

## d1: does the logging interval change the training trajectory? (diagnostic)

*Added Sunday October 11, after E1b's runs and before any d1 run. Not part of the
original protocol.*

E1b's pre-registered cross-check against E1 failed (see the E1b results). An E1b run and
the E1 run it should repeat differ only in `--accuracy-interval` (5 versus 1) and
`--epochs` (1000 versus 300); the code is identical. Reading `train.py`: each accuracy
evaluation iterates two extra `DataLoader`s, and every new `DataLoader` iterator draws a
base seed from the global random generator even with `shuffle=False` (checked directly
with `torch.get_rng_state()`). So the interval changes the shuffle order of the single
training batch. That order is mathematically irrelevant but changes floating-point
rounding.

**Not blind.** Before writing this spec I re-ran one E1b setting outside the repository
(seed 42, λ=0.3, 100 updates) twice, once at interval 5 and once at interval 1; the first
reproduced E1b's rows and the second reproduced E1's. d1 repeats the interval-5 case under
the runner on all three checkpoints, so that the evidence is committed and audited.

**Design.** λ=0.3, lr=0.001, 100 updates, logged every 5 updates, training on B only,
the three E1 checkpoints (hash-pinned), 3 runs
([`d1_logging_interval.json`](../../specs/d1_logging_interval.json)).

**Predictions (fixed now).**

- P1: each d1 run equals the first 100 updates of the corresponding E1b run (same
  interval 5, 1000 updates in E1b) in every logged column except timing.
- P2: each d1 run differs from E1's λ=0.3 run (interval 1) at the shared updates.

If P1 holds, the number of epochs is irrelevant and the interval alone explains why runs
are identical or not. If P2 also holds, the interval is sufficient to explain E1b's failed
cross-check. If P1 fails, something else differs between runs and the explanation is
wrong. Every outcome is reported.

**Reproduce.**

```bash
python phonebook/run_experiment.py --spec phonebook/specs/d1_logging_interval.json --out-root phonebook/experiments/week_2026-10-07/d1_logging_interval
python phonebook/audit_experiment.py --root phonebook/experiments/week_2026-10-07/d1_logging_interval --require-local-artifacts --output phonebook/experiments/week_2026-10-07/d1_logging_interval/audit.json
python phonebook/report_experiments.py compare --a phonebook/experiments/week_2026-10-07/d1_logging_interval --b phonebook/experiments/week_2026-10-07/e1b_fine_grid_long_horizon --pair d0_seed42:l2sp_lam0.3_iv5_ep100:l2sp_lam0.3 --pair d0_seed123:l2sp_lam0.3_iv5_ep100:l2sp_lam0.3 --pair d0_seed999:l2sp_lam0.3_iv5_ep100:l2sp_lam0.3 --max-epoch 100 --out phonebook/experiments/week_2026-10-07/d1_logging_interval/compare_with_e1b.json
python phonebook/report_experiments.py compare --a phonebook/experiments/week_2026-10-07/d1_logging_interval --b phonebook/experiments/week_2026-10-07/e1_matched_lr --pair d0_seed42:l2sp_lam0.3_iv5_ep100:l2sp_lam0.3 --pair d0_seed123:l2sp_lam0.3_iv5_ep100:l2sp_lam0.3 --pair d0_seed999:l2sp_lam0.3_iv5_ep100:l2sp_lam0.3 --max-epoch 100 --out phonebook/experiments/week_2026-10-07/d1_logging_interval/compare_with_e1.json
```

The last command is expected to exit with status 1 (it reports the mismatches P2 predicts).
