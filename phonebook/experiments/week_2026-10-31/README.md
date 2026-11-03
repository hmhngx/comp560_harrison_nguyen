# Week of October 31 to November 3, 2026: replay ratio (E5) and capacity (E4)

Two experiments answer the two course-README bullets that were still open: *which replay ratio k reaches 100% on all
the data fastest* (E5, which finishes E3 properly) and *what is the capacity of a very small model* (E4). Protocols, rules
and predictions were fixed and committed before any run of the experiment they govern; results are scored against those
predictions mechanically. Derived tables are in [`../../results/`](../../results/), the rules and predictions in
[`../../preregistration/`](../../preregistration/) and [`../../specs/`](../../specs/).

| What | Commit | Runs | Audit |
|---|---|---|---|
| E5 Phase A (18 models, 6 phonebooks x 3 inits) | `ba50291` | 18 | passed (360 answers regenerated) |
| E5 main grid, noise arm, E3 anchor | `6d8cc86` (lock) | 306 | 12 folders passed (6,120 answers) |
| E4 capacity | `0fe8f6f` (lock) | 96 | 24 folders passed (17,202 answers) |

All weights stay local; the committed evidence per run is `metrics.csv` and the strict evaluation JSON, plus protocol,
prepared-data hashes, manifest, summary and audit for each folder.

---

## E5: the replay ratio k, confirmed on six phonebooks

**Question (course README).** Replay data holds one copy of A and k copies of B. How does k change forgetting, and which k
reaches 100% exact recall of both books in the fewest updates?

**Design.** Six independent phonebooks (seeds 2, 4, 7 from E2 and the first three qualifying seeds above 7: 8, 9, 12), each with
three Phase-A models (init seeds 42, 123, 999); every Phase-A model recalled all ten entries of its own book. At lr 0.001 the
fine-tuning grid is k = 1, 2, 3, 4, 5, 9 plus a control with no copy of A (plain fine-tuning); at lr 0.003 and 0.0003,
k = 1, 2, 3, 5 (270 main runs). A noise arm repeats k = 1 and 3 on phonebooks 2, 4, 7 under five other shuffle seeds (30 runs),
and an anchor reruns E3's k = 1 and 3 under the historical loader (6 runs). One update is one full batch (A once plus B k times);
an update counts as joint recall only if every logged update from then on also recalls both books exactly (`stable_joint`, as in E3).

**Pre-registered rule (lr 0.001 only).** The unit is the phonebook: the median of its three inits, compared with k = 1 within
the phonebook. k beats k = 1 if it is faster in at least 5 of 6 phonebooks and its median paired gain exceeds the noise band,
the largest spread between runs that differ only in the shuffle seed.

### Results

1. **The noise band is 0.** Changing the shuffle seed never moved the joint-recall update: spread 0 in all six noise
   configurations (six runs each). The shuffle only reorders rows inside the single batch, which changes floating-point
   rounding in the last digits and nothing observable here. With a band of 0 the rule reduces to a sign rule, satisfied by
   any consistent gain, however small.
2. **k = 2 beats k = 1 by the rule, by about 1.5 updates (4%).** Median time to joint recall over the six phonebooks is 40.0,
   38.5, 38.5, 40.5, 44.0, 54.0 updates for k = 1, 2, 3, 4, 5, 9. k = 2 is faster than k = 1 in 5 of 6 phonebooks (median gain 1.5;
   phonebook 2 is 1 update slower); no other k meets the rule (k = 3: 3 of 6, k = 4: 1, k = 5: 1, k = 9: 1). Per single run,
   k = 2 is faster in 15 of 18 runs, tied in 1, slower in 2, by 1.2 updates on average; k = 3 is a coin flip (10 faster, 2 tied,
   6 slower, mean 0.1); k = 9 is slower in 17 of 18 (mean 15 updates). The pattern 5 of 6 has a one-sided sign-test probability of
   7/64 = 0.11 under no effect, so this is a small, consistent shift, not a strong result.
3. **Which k is fastest is a tie between 2 and 3** at the phonebook level (both 38.5); k = 2 is the more reliable. In sequences
   processed (10 + 10k per update) k = 1 is cheapest by far: 800, 1155, 1540, 2025, 2640, 5400 for k = 1, 2, 3, 4, 5, 9.
4. **Forgetting.** The fewest entries of A recalled at any update (median over phonebooks) is 9, 6.5, 4, 3, 2, 0.5 for k = 1, 2, 3, 4, 5, 9:
   more copies of B means deeper forgetting of A, monotonically. Plain fine-tuning loses A completely in all 18 runs (0 of 10) and never
   reaches joint recall.
5. **Why the gains are small (exploratory, after the fact).** Joint recall waits for two things: B memorized and A back to 10 of 10.
   B is memorized only 2 updates sooner at k = 2 and 4.5 updates sooner at k = 9 than at k = 1 (median over phonebooks of the
   per-phonebook saving; it saturates), while A's return time (median over the 18 runs per k) grows from 18 to 56 updates. A is the later of
   the two in 1, 2, 9, 12, 15, 18 of 18 runs for k = 1, 2, 3, 4, 5, 9. The most any k can save is the small B speed-up, which is why k = 2
   gains little and larger k loses. (That joint recall equals the later of the two is true by definition; the evidence is in the trends.)
6. **Other learning rates (descriptive).** lr 0.003: medians 19.0, 17.5, 19.0, 25.5 for k = 1, 2, 3, 5 (k = 2 faster in 5 of 6 phonebooks,
   k = 5 in none). lr 0.0003: 96.5, 93.5, 91.5, 96.5; k = 2 and k = 3 are each faster in 5 of 6 phonebooks (median gains 7 and 9), k = 5 in 3.
   Lower learning rates favour a larger k, but the learning rate moves the time far more than k does (19, 40 and 96.5 updates for k = 1 at
   lr 0.003, 0.001 and 0.0003).
7. **The anchor reproduces E3 exactly:** six runs of 500 updates, no differing logged value except timing.

### Predictions against outcomes (rules fixed before any result was read)

13 of 15 scored lines hit (`results/e5_prediction_scores.json`). Misses: (a) k = 9 was predicted faster than k = 1 in exactly 0 phonebooks
and was in 1 (phonebook 7: medians 46 against 48, from run times 44, 46, 76 against 38, 48, 50, a median-of-three effect on the slowest
phonebook, and 1 of 18 single runs); (b) the plateau was predicted as k = 2 alone and is {2, 3} because the two medians tie at 38.5. Two hits
are tie-dependent: "fastest k is 2" (k = 3 ties, the smaller k wins by rule) and "best k is larger at lr 0.0003" (k = 3 at 0.0003 against the
tied 2 and 3 at 0.001). The k = 1 median was predicted 38 (34 to 43) and is 40.

---

## E4: the capacity of a small transformer

**Question (course README).** "Create an extremely small model ... Determine the capacity of your model by training on larger and larger
phonebooks until it is impossible to get 100 percent accuracy."

**Design.** Four models that differ only in width: 2 layers, 2 attention heads, embedding width d = 8, 16, 32, 64, which is 3,072, 9,216,
30,720 and 110,592 parameters at the smallest context (a few rungs have a different context and carry 8 to 64 parameters fewer or more; the
audit checks every logged count against the closed form). Each model is trained from scratch, one run per (book size N, book seed), on N
random name-to-number entries from the course generator. Books are nested: the book of N entries is the first N lines of one large random
book per seed. Three book seeds (1, 2, 3), the training seed equal to the book seed; learning rate 0.001 (the trainer default), batch size 64,
1000 epochs, accuracy logged every 10 epochs. A held-out control book of 20 entries that share no name and no number with any training entry
is evaluated at every logged epoch. Each width has its own ladder of book sizes, chosen from a scratch calibration on a throwaway book seed
(57 short runs, not evidence): d = 8: 10, 14, 20, 28, 40; d = 16: 10, 20, 40, 57, 80, 113, 160, 226; d = 32: 10, 20, 40, 80, 113, 160, 226, 320,
453; d = 64: 10, 20, 40, 80, 160, 226, 320, 453, 640, 905.

**Pre-registered rules** (in the specs and the analysis code before any E4 run).
- A run is *perfect by budget E* if every one of its N entries is recalled exactly at some logged epoch at or before E (the trainer's
  teacher-forced sequence accuracy is 100%; the audit checks that this equals strict greedy generation).
- A rung *passes* at budget E if at least 2 of its 3 book seeds are perfect by E. The *capacity* at E is the largest rung such that it and
  every smaller rung pass; a pass above the first failing rung is reported but never counted (none means no rung passes).
- Budgets are read at 100, 300, 500 and 1000 epochs from one 1000-epoch log per run (training is deterministic with an isolated loader
  stream, so a shorter budget is a prefix of a longer run).
- The *boundary* is `stable` if the capacity at 500 epochs equals the capacity at 1000, `moving` if not (then it is a budget limit, not a
  capacity), `ceiling` if the top rung passes and `none` if no rung passes.
- Controls: exact recall of the control book must stay 0; the logged parameter count must equal the closed form.

### Results

1. **Capacity at 1000 epochs: 14, 113, 226 and 453 entries for d = 8, 16, 32, 64.** By budget:

   | width | parameters | 100 epochs | 300 | 500 | 1000 | boundary | bits per parameter at 1000 |
   |---|---|---|---|---|---|---|---|
   | 8 | 3,064 | none | none | none | 14 | moving | 0.149 |
   | 16 | 9,216 | none | 20 | 40 | 113 | moving | 0.399 |
   | 32 | 30,720 | none | 160 | 226 | 226 | stable | 0.240 |
   | 64 | 110,656 | 20 | 320 | 320 | 453 | moving | 0.133 |

   An entry is 32.575 bits (a name-to-number pair drawn from 800 x 800 x 10,000 equally likely numbers). Capacity rises strictly with width,
   and doubles when the width doubles from 16 to 32 and from 32 to 64 (113 to 226 to 453), while the parameter count roughly triples to
   quadruples; no model reached the top of its ladder. All four are far below one bit per parameter.
2. **Capacity here is limited by training time more than by room in the weights.** The first full recall of the smallest book (10 entries)
   comes at epochs 470 to 790 for d = 8, 200 to 220 for d = 16, 110 to 120 for d = 32 and 60 to 70 for d = 64, and the epoch of first full
   recall rises with the number of entries at every width, apart from small reversals between neighbouring rungs (for d = 64 from 60 to 70 at
   10 entries to 710 to 860 at 453; for d = 16 the median at 57 entries, 600, is above that at 80, 590). The boundary moves with the
   budget for d = 8, 16 and 64 (the capacity at 500 epochs is smaller than at 1000) and is the same at 500 and 1000 for d = 32. This is a
   capacity at 1000 epochs with this learning rate and this exact-recall criterion, not an asymptotic capacity.
3. **The rung above each capacity stalls just short of 100%.** Final training accuracy at epoch 1000 for seeds 1/2/3: d = 8 at 20
   entries 0.90/0.90/0.70; d = 16 at 160 entries 0.80/0.98/0.72; d = 32 at 320 entries 0.97/0.98 for two seeds while the third reached 100% at epoch
   940 (and fell back to 0.99); d = 64 at 640 entries 0.98/0.97/0.96. For the two narrow models accuracy was still rising steeply at the
   end (+0.07 to +0.26 over the last 200 epochs). For the two wide models it had flattened at 0.96 to 0.99 with a few entries
   moving in and out, so "needs more epochs" is established for d = 8 and 16 only; for d = 32 and 64 the last entries may need a lower
   learning rate or a different schedule, which was not tried.
4. **No control entry was ever recalled** (0 of 20 in all 96 runs), so the recall is memorization of the training pairs.
5. **Several capacities rest on cells at the edge of a budget.** d = 8 passes at 14 entries only because one seed finished at epoch 1000
   (the last logged epoch; the others at 730 and never, 0.93); d = 16 passes at 113 entries with seeds at 970, 1000 and 920; d = 32's capacity at
   300 epochs is 160 only because 226 entries passes in 1 of 3 seeds (finishing at epochs 330, 500 and 300); d = 64's capacity at 300 epochs
   passes at 320 entries with seeds at 220, 330 and 240, exactly the rule's 2 of 3. A rung ladder with ratio 1.4 and a 10-epoch logging grid cannot resolve these better.

### Predictions against outcomes (rules fixed before any result was read)

23 of 23 scored lines hit (`results/e4_prediction_scores.json`), including all four capacities at 1000 epochs exactly on the predicted rung. These lines are
strongly correlated (the same four capacities read at different budgets), so this is not 23 independent confirmations; the calibration that
produced the predictions used a different book seed and different training seeds, which is why it generalized. One part of the locked
explanation was only half right: it said the rung above each boundary was "still climbing at epoch 1000", which is true for d = 8 and 16 and not for
d = 32 and 64 (result 3).

---

## Incidents and deviations (all disclosed)

- **The E4 batch was interrupted.** The first batch was killed at 03:36 local on November 3, the minute an interrupted tool call ended
  (the last run in any manifest finished at 08:36:00 UTC; the background process tree died with that call, which is the likely cause but was
  not tested). 17 of 24 specs (78 of 96 runs) were complete, all with return code 0; 7 specs were cut off. Their partial folders
  were moved aside (`..__interrupted`, kept locally, not committed, never analyzed) and the 7 specs were rerun in full under their original names from the same locked commit
  `0fe8f6f` with a Python driver. The 21 runs that had finished before the interruption reproduced exactly in the rerun: identical in
  every logged column except timing and in every generated answer. All 96 manifest entries of the analyzed folders record commit `0fe8f6f` and a clean tree.
- **One audit failed at default thread count and passes at the training thread count.** Auditing `e4_capacity_d64_seed1_a` with PyTorch's default 10 threads
  failed on the check "saved evaluation equals a fresh evaluation": every generated answer was identical, but the training-set loss of the 10-entry run
  (0.0013339654 saved, 0.0013339636 fresh) differs by a relative 1.4e-6, just above the audit's 1e-6 tolerance, which is tight for a loss this near zero.
  With one thread, the environment of training and of the saved evaluation, the fresh evaluation reproduces the saved one exactly and the audit passes
  (machine output of both attempts: [`results/e4_audit_thread_check.txt`](../../results/e4_audit_thread_check.txt)).
  The audit code was not changed. A later fix could use an absolute tolerance for near-zero losses or pin the thread count.
- **Threads.** The E5 anchor ran at PyTorch's default thread count, as E3 did; all other E5 runs and all E4 runs ran with one thread each, chosen for
  throughput after a measurement (stable_joint was identical at one and ten threads in scratch pilots, which are not evidence).
- **Analysis calls.** `analyze_replay_k.py` and `independent_check.py` cannot take one `--ks` list across learning rates that have different k, so each was called
  twice (lr 0.001 with k = 1, 2, 3, 4, 5, 9; lr 0.003 and 0.0003 with k = 1, 2, 3, 5); no analysis code was changed.
- **A predictions file in the wrong folder.** The E5 predictions file was first committed in `specs/`, which a test treats as runner specs; it was moved to
  `preregistration/` (commit `33c1d5f`) without touching the lock commit.
- **Who wrote the predictions.** Both sets of predictions were drafted by Claude (Sonnet 5.5) at the researcher's request, from scratch pilots on phonebooks and book seeds
  that the experiments do not use, and confirmed by the researcher before any run. The scorers (`score_e5_predictions.py`, `score_e4_predictions.py`) have tests that plant bugs in
  the scoring rules and confirm that the tests catch them.

## Reproduce

From the repository root; the weights are needed for `--require-local-artifacts`.

```bash
# E5
python phonebook/run_experiment.py --spec phonebook/specs/e5_main_pb2.json --out-root phonebook/experiments/week_2026-10-31/e5_main_pb2
python phonebook/audit_experiment.py --root phonebook/experiments/week_2026-10-31/e5_main_pb2 --require-local-artifacts
python phonebook/analyze_replay_k.py --root ROOT [--root ROOT ...] --lrs 0.001 --primary-lr 0.001 --ks 1 2 3 4 5 9 --phonebooks 2 4 7 8 9 12 --out phonebook/results --prefix e5
python phonebook/analyze_replay_k.py --root ROOT [--root ROOT ...] --lrs 0.003 0.0003 --primary-lr 0.003 --ks 1 2 3 5 --phonebooks 2 4 7 8 9 12 --out phonebook/results --prefix e5_robust
python phonebook/independent_check.py e5 --root ROOT [--root ROOT ...] --lrs 0.001 --primary-lr 0.001 --ks 1 2 3 4 5 9 --phonebooks 2 4 7 8 9 12 --against phonebook/results/e5_analysis.json
python phonebook/score_e5_predictions.py --primary phonebook/results/e5_analysis.json --robust phonebook/results/e5_robust_analysis.json --cells phonebook/results/e5_cells.csv --predictions phonebook/preregistration/e5_predictions.json
# E4
python phonebook/run_experiment.py --spec phonebook/specs/e4_capacity_d64_seed1_a.json --out-root phonebook/experiments/week_2026-10-31/e4_capacity_d64_seed1_a
python phonebook/analyze_capacity.py --root ROOT [--root ROOT ...] --budgets 100 300 500 1000 --out phonebook/results --prefix e4
python phonebook/independent_check.py e4 --root ROOT [--root ROOT ...] --budgets 100 300 500 1000 --against phonebook/results/e4_capacity.json
python phonebook/score_e4_predictions.py --capacity phonebook/results/e4_capacity.json --predictions phonebook/preregistration/e4_predictions.json
```

Without the weights, `python phonebook/audit_experiment.py --root ROOT` still reconciles manifests, protocols, specs at their recorded commits,
summaries, logged metrics and the saved raw answers (each re-scored from the strings).

## AI assistance

Claude Code (Sonnet 5.5) assisted with the code, the experiments, the audits, the predictions (see above) and this write-up.
