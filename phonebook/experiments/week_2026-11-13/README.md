# Week of November 13, 2026: what stops the wide models from holding 100% (E7)

E6 trained the first failing rung of each width for 4000 epochs and found that at widths 16 and 32 the 100% appears only on isolated logged epochs and that width 64's
640 entries never reach it. E7 asks what stops them. It retrains the same three rungs (160, 320 and 640 entries at widths 16, 32 and 64; the same books, book seeds, model,
batch size and 4000 epochs) with exactly one optimizer setting changed from E6, which is the control (AdamW weight decay 0.1, constant learning rate 0.001):
**arm W** sets the weight decay to 0.0, **arm S** multiplies the learning rate by 0.1 after epoch 2000. Rules and predictions were fixed and committed before any run;
results are scored against them mechanically. Derived tables are in [`../../results/`](../../results/), the rules and predictions in
[`../../preregistration/e7_predictions.json`](../../preregistration/e7_predictions.json) and the 18 specs in [`../../specs/`](../../specs/).

| What | Commit | Runs | Audit |
|---|---|---|---|
| E7 weight decay 0.0 (arm W) and learning-rate step-down (arm S), widths 16, 32, 64 | `40a73f2` (lock) | 18 | 18 folders passed (7,080 answers, regenerated from the saved weights), and the same 18 pass without the weights in a fresh clone of `d3fe2dd` |

The trainer options (`--weight-decay`, `--lr-schedule`) and the analysis flags came in `e8081a1`; the predictions file and its scorer in `a9db2aa`; the specs in `40a73f2`, all before any run;
the results and evidence in `d3fe2dd`. All weights stay local; the committed evidence per run is `metrics.csv` and the strict evaluation JSON, plus protocol, prepared-data hashes, manifest,
summary and audit for each folder, as for E6.

---

## E7: the weight decay and the learning-rate step

**Question.** E6 varied only the length of training. At constant learning rate 0.001 with AdamW weight decay 0.1, the wide models reached 100% only on isolated epochs, none of the nine
runs at widths 16, 32 and 64 was at 100% at epoch 4000, and width 64 never passed. Is that the weight decay, the size of the learning-rate step, or neither?

**Design.** The rungs are E6's: 160 entries at d = 16, 320 at d = 32 and 640 at d = 64 (2 layers, 2 heads), book seeds 1, 2 and 3, training seed equal to the book seed, batch size 64,
learning rate 0.001, 4000 epochs logged every 10, budgets 1000, 2000, 3000 and 4000 read from one log per run. One setting differs from E6 in each arm:

- **Arm W:** AdamW weight decay 0.0 (E6: 0.1), learning rate constant. It isolates the decay.
- **Arm S:** the learning rate is multiplied by 0.1 after epoch 2000 (`--lr-schedule step:2000:0.1`), weight decay 0.1 as in E6. AdamW decays every weight by learning rate times
  weight decay at each step, so arm S lowers the step size and the per-step decay together; only arm W isolates the decay. Its first 2000 epochs must equal the E6 run point for point.

That is 18 runs, one spec each, run 7 at a time with one thread each from the locked commit `40a73f2`, from 14:26 to 17:18 on November 13 (2 h 51 min; the six width-64 runs took 2.5 h each).

**Pre-registered rules** (the predictions file and its scorer are committed in `a9db2aa`, the specs in `40a73f2`, all before any E7 run).
- A run is perfect by budget E if all N entries are recalled exactly at some logged epoch at or before E; a rung passes at E if at least 2 of the 3 book seeds are perfect by E (as in E4 and E6).
  A run is perfect at epoch 4000 if all N entries are recalled at the last logged epoch.
- A run *holds* the book if at least 50 of its 100 logged epochs in 3001 to 4000 have all N entries recalled (the three width-8 runs of E6 do, with 95, 96 and 94; none of E6's nine runs at
  widths 16, 32 and 64 does).
- Accuracy at a logged epoch is the number of entries recalled exactly divided by N; a window mean is its mean over the logged epochs in the window, pooled over the 3 seeds with equal weight;
  the *gain* is the mean over 3001 to 4000 minus the mean over 1001 to 2000; the trend is *rising* if the gain is at least +0.005, *falling* if at most -0.005, *flat* otherwise (E6's rule).
  An arm *beats E6* at a width when its 3001 to 4000 mean exceeds E6's by at least 0.005.
- A scored line is a HIT when the actual value lies inside the locked range (or equals the locked answer) and an exact HIT when it equals the point; nothing else counts. 44 lines are scored
  (nine questions); the arm S prefix check, the six trends and ten tripwires are reported, not scored. Control recall must be 0.

### Results

1. **Every rung passes in both arms, including width 64, and all 18 runs end at 100%.**

   | arm | width | rung | first epoch at 100% (seeds 1/2/3) | passes at 4000 (first budget) | at 100% at epoch 4000 |
   |---|---|---|---|---|---|
   | W | 16 | 160 | 2020/1310/1830 | yes (2000) | 3 of 3 |
   | W | 32 | 320 | 820/1660/1770 | yes (2000) | 3 of 3 |
   | W | 64 | 640 | 1310/1270/1500 | yes (2000) | 3 of 3 |
   | S | 16 | 160 | 2010/1210/2510 | yes (3000) | 3 of 3 |
   | S | 32 | 320 | 940/2010/2010 | yes (3000) | 3 of 3 |
   | S | 64 | 640 | 2010/2010/2010 | yes (3000) | 3 of 3 |
   | E6 | 16 | 160 | 3270/1210/3200 | yes (4000) | 0 of 3 |
   | E6 | 32 | 320 | 940/2960/never | yes (3000) | 0 of 3 |
   | E6 | 64 | 640 | never/never/never | no | 0 of 3 |

   Six of arm S's nine runs first reach 100% at epoch 2010, the first logged epoch after the cut. The capacity read at 4000 epochs is now 160, 320 and 640 entries in both arms (arm W from 2000
   epochs, arm S from 3000), at least 0.566, 0.339 and 0.188 bits per parameter; the rung above each pass (226, 453 and 905 entries) was not run, so these are lower bounds. At width 64 that
   is above E4's 0.133.
2. **The models hold, with one exception and two dips.** Logged epochs at 100% among the last 100 (3001 to 4000), seeds 1/2/3, with the window means (1001 to 2000, then 3001 to 4000) and the trend:

   | arm | width | logged epochs at 100% of the last 100 | holding seeds | mean accuracy | trend (gain) |
   |---|---|---|---|---|---|
   | W | 16 | **13**/100/100 | 2 of 3 | 0.9666 then 0.9937 | rising (+0.0271) |
   | W | 32 | 100/100/100 | 3 of 3 | 0.9802 then 1.0000 | rising (+0.0198) |
   | W | 64 | 95/90/100 | 3 of 3 | 0.9928 then 0.9928 | flat (+0.0000) |
   | S | 16 | 100/100/100 | 3 of 3 | 0.9351 then 1.0000 | rising (+0.0649) |
   | S | 32 | 100/100/100 | 3 of 3 | 0.9636 then 1.0000 | rising (+0.0364) |
   | S | 64 | 100/100/100 | 3 of 3 | 0.9684 then 1.0000 | rising (+0.0316) |

   E6 held in none of these nine runs (its window means: 0.9351 then 0.9526 at width 16, 0.9636 then 0.9733 at 32, 0.9684 then 0.9754 at 64). Arm W at width 16 with seed 1 is the exception: it first reaches 100%
   at epoch 2020 but only on isolated logged epochs (24 of 400, the longest stretch 2), as in E6, although it ends at 100%. The pre-registration says to list from the curves every dip of an arm W run below 100%
   after its first stretch of 500 epochs at 100% (read as 50 consecutive logged epochs); here they are, with the epoch from which each run is settled:

   | arm W run | settled from | dips below 100% afterwards |
   |---|---|---|
   | d16 seed 1 | never settled | |
   | d16 seed 2 | 1360 | none |
   | d16 seed 3 | 2990 | none |
   | d32 seed 1 | 1560 | none |
   | d32 seed 2 | 2560 | none |
   | d32 seed 3 | 2130 | none |
   | d64 seed 1 | 1310 | epochs 3700 to 3740, down to 390 of 640 |
   | d64 seed 2 | 1270 | epochs 3900 to 3990, down to 25 of 640 |
   | d64 seed 3 | 1500 | none |

   Both width-64 dips recover by epoch 4000. The seed 2 dip ends at the last logged epoch, so its "perfect at epoch 4000" (and the locked line that hit on it) rests on a recovery in the final 10 epochs. No arm S run dips
   after it settles (all nine settle at epoch 2000 or later and stay at 100% to epoch 4000).
3. **Which of the ten tripwires fired.** Two fired, and their readings were written into the predictions file before any run. `w_d64_passes` (arm W passes at width 64): "the decay was the binding limit even at the widest
   model, E6's width-64 shortfall was not a time or capacity limit, and the weight-decay explanation of the 96 to 99 percent plateau is supported". `both_large` (both arms hold in more seeds than E6 at widths 16 and 32):
   "either setting is enough to help; arm S cannot say whether the step size or the lower decay did it (it changes both), and only arm W shows that the decay alone is enough". The other eight did not fire:
   `w_d64_fails_to_pass`, `neither_arm_lifts_d64`, `w_null_s_large`, `w_large_s_null`, `w_hurts`, `s_prefix_fails`, `s_d64_does_not_hold` and `real_books_do_not_replicate`.
4. **Controls and reproduction.** No control entry was recalled (0 of 20 in all 18 runs). The first 2000 epochs of every arm S run equal the E6 run at every logged epoch (the check passed for all nine). All 18 audits pass with
   the weights (7,080 answers regenerated at one thread), and the 18 folders pass without them in a fresh clone of `d3fe2dd`, where `analyze_capacity.py`, `analyze_extension.py`, `independent_check.py e4` and the scorer, run on
   the tracked files alone, reproduce all seven committed result files (line endings ignored).

### Predictions against outcomes (rules fixed before any result was read)

41 of 44 scored lines hit (`results/e7_scores.json`): 25 exact hits on the locked point value and 16 inside the locked range. Treat the count with care: the lines are correlated (the passing, first-budget and perfect-at-4000 lines all read
the same 18 runs), and 24 of the 44 are about whether a rung passes, by when, and whether the runs end at 100%, all of which hit. The three misses are all about arm W: (1) width 16 holding seeds, predicted 3 (probability about 0.91 that all three hold), found 2, because of seed 1;
(2) the arm W width-64 mean accuracy over 3001 to 4000, predicted 0.9988 in a range of 0.9934 to 1.0, found 0.9928, 0.0006 below the range, from the two dips above; (3) which arm has more holding seeds at width 16, predicted a tie (confidence 0.90),
found arm S, 3 against 2, which follows from (1). The locked predictions put the chance that an arm W run dips at width 64 at 0.30 per run (about two in three that at least one of the three dips); two of the three dipped, deeper than the one
dip of the calibrations (a spike down to 0.767 of the entries). Reported but not scored: the six trends (arm W rising at 16 and 32 and flat at 64; arm S rising at all three) and the dips above.

---

## Incidents and deviations (all disclosed)

- **The first launch attempt failed before it started anything.** The launcher script (`launch_e7.ps1`, a scratch file) called the Windows power-state function with a hexadecimal literal that PowerShell reads as a negative 32-bit integer and stopped at that line. No run, folder or log existed (checked: no python process, no log folder, no out-root, a clean tree); the literal was replaced by its decimal value and the batch was launched at 14:26:31.
- **No run failed, was interrupted or was rerun.** All 18 returned 0, the driver ended with `failed: []`, `not_started: []` and `tree_clean: true` at 17:18:06, and all 18 protocols record commit `40a73f2` with a clean tree and the same environment as E6 (Python 3.12.10, torch 2.12.0+cpu, 1 thread, Windows 11 10.0.26300).
- **The repository was not written while the runs were going** (no git write command, no new file inside it); the plans, drafts and scratch work were kept outside it.
- **Other work shared the machine.** During the runs a TeX distribution was installed and test documents were compiled on the same laptop, and file contents were read. Training is deterministic with one thread, and the arm S prefix check (first 2000 epochs equal to E6's at every logged epoch, for all nine runs) passed, but timing was not controlled: the width-64 runs took 8,874 to 9,044 s each, against about 6,400 s for E6's and the 1.8 h planned.
- **Partial logs were read once.** To estimate finishing times, the last line of each run's `metrics.csv` (epoch, loss and accuracy columns) was read while the runs went, at 16:32 and 17:01. No analysis or scoring was run before the driver ended, and nothing was changed.
- **Scratch scripts, not committed.** The driver that started the 18 runs (`e7_run_driver.py`: 7 workers, widest runs first, no new run after a failure), its launcher, the chain script that ran the audits, analyses and scorer (`run_e77.sh`), the staging script and the fresh-clone check are kept outside the repository; the exact commands are under Reproduce.
- **The `audit.json` files** are copies of the with-weights audit reports made after the 18 runs, as for E6.
- **Spec text.** As in E6, the spec text for the rule and the description is the E4 generator's, so it speaks of a ladder and nested books although each spec holds one rung; the E7 rules are in the predictions file (embedded in each spec) and above.
- **Who wrote the E7 predictions.** Drafted by Claude (Sonnet 5.5) on November 12 at the researcher's request, from the E6 logs, the E6 scratch calibration and the E7 plan, and revised on November 13 after throwaway calibrations of the two settings
  (12 runs on a throwaway book, seed 100, widths 16, 32 and 64; not evidence, and not archived in the repository). They were then settled with the researcher in a chat session: the researcher chose how far the calibrations carry to the real books at each width
  (the "more sure" option) and how quickly arm W passes, kept the other numeric lines and the tripwire readings as they stood, and left the chance that an arm W run dips at width 64 to Claude, who set it to 0.30 from one dip in six calibration runs.
  Asked whether to adopt, the researcher answered "go for the most optimal approach", and re-confirmed by asking for the commit that locked the file and its scorer (`a9db2aa`).
  The scorer (`score_e7_predictions.py`) has 270 tests; a scratch run that planted 513 one-line changes in it was caught by the tests 509 times (the four left are a rounding digit in a float guard, which is equivalent, and three help texts); that script is not committed.

## Reproduce

From the repository root; the weights are needed for `--require-local-artifacts`. `--root` and `--earlier-root` are repeatable options (one flag per folder).

```bash
# E7 (eighteen specs e7_{w,s}_d{16,32,64}_seed{1,2,3}; the runs were started 7 at a time, one thread each)
python phonebook/run_experiment.py --spec phonebook/specs/e7_w_d64_seed1.json --out-root phonebook/experiments/week_2026-11-13/e7_w_d64_seed1
python phonebook/audit_experiment.py --root phonebook/experiments/week_2026-11-13/e7_w_d64_seed1 --require-local-artifacts
python phonebook/analyze_capacity.py  --root ROOT [--root ROOT ...] --budgets 1000 2000 3000 4000 --out phonebook/results --prefix e7w     # the nine arm W roots; --prefix e7s for the nine arm S roots
python phonebook/analyze_extension.py --root ROOT [--root ROOT ...] --budgets 1000 2000 3000 4000 --epochs-at-100 --out phonebook/results --prefix e7w
python phonebook/analyze_extension.py --root ROOT [--root ROOT ...] --earlier-root E6ROOT [--earlier-root E6ROOT ...] --earlier-label E6 --prefix-epochs 2000 --budgets 1000 2000 3000 4000 --epochs-at-100 --out phonebook/results --prefix e7s
python phonebook/independent_check.py e4 --root ROOT [--root ROOT ...] --budgets 1000 2000 3000 4000 --against phonebook/results/e7w_capacity.json    # and e7s_capacity.json for arm S
python phonebook/score_e7_predictions.py --w phonebook/results/e7w_extension.json --s phonebook/results/e7s_extension.json --predictions phonebook/preregistration/e7_predictions.json --json phonebook/results/e7_scores.json
```

Without the weights, `python phonebook/audit_experiment.py --root ROOT` still reconciles manifests, protocols, specs at their recorded commits, summaries, logged metrics and the saved raw answers (each re-scored from the strings).

## AI assistance

Claude Code (Sonnet 5.5) assisted with the code, the experiments, the audits, the predictions (see above) and this write-up.
