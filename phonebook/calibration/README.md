# Calibration and pilots behind the locked predictions

**This is not evidence.** Nothing here has a spec, a manifest, an audit or a lock commit. It is the scratch work that fixed the
E4 ladders and that the E4 and E5 predictions
([`../preregistration/`](../preregistration/)) were drafted from. It is committed so that the predictions can be reproduced and
inspected, not so that anything can be cited as a result. The results are in [`../results/`](../results/) and
[`../experiments/week_2026-10-31/README.md`](../experiments/week_2026-10-31/README.md).

## What is here

| File | What it is |
|---|---|
| [`e4_calibration.py`](e4_calibration.py) | writes the throwaway books (book seed 100) and trains calibration cells: width, entries, epochs, training seed |
| [`e4_calibration_results.jsonl`](e4_calibration_results.jsonl) | the 57 recorded cells (46 distinct once a 300-epoch run is merged into its 1000-epoch twin), with full accuracy curves |
| [`e4_montecarlo.py`](e4_montecarlo.py) | the Monte Carlo prior that gave the E4 prediction ranges |
| [`e5_pilots.py`](e5_pilots.py) | the E5 pilots: the E3 phonebook with several shuffle seeds, and ten other phonebooks |
| [`e5_pilot_results/`](e5_pilot_results/) | the six raw result files of those pilots, as the pilots wrote them |
| [`e5_pilot_report.py`](e5_pilot_report.py) | the noise, censoring, flicker and bootstrap numbers the E5 predictions quote |
| [`throughput_benchmark.py`](throughput_benchmark.py) | the measurement that chose 7 single-thread runs at a time for the E5 and E4 batches |

## E4 calibration

Widths 8, 16, 32, 64 (2 layers, 2 heads), trained from scratch at lr 0.001, batch 64, accuracy every 10 epochs, on nested random books
from the course generator (book seed 100, which no experiment uses) with training seeds 100 and 101. `python
phonebook/calibration/e4_calibration.py summary` prints the table below from the committed results.

```text
first epoch with 100% exact recall of all n entries; no(x) = never, x = accuracy at the last epoch; * = only run for 300 epochs
d=8   n=10: 520/450 | n=14: 710 | n=20: no(0.95)/680 | n=40: no(0.28) | n=80: no(0.00)* | n=160: no(0.00)*
d=16  n=10: 230* | n=20: 280 | n=40: 440 | n=57: 500 | n=80: 680/610 | n=113: 770 | n=160: no(0.97)/no(0.97) | n=320: no(0.00)*
d=32  n=10: 130* | n=20: 170* | n=40: 210* | n=80: 220* | n=113: 240 | n=160: 250*/260 | n=226: 390/330 | n=320: no(0.98)/570 | n=640: no(0.80) | n=1280: no(0.04)
d=64  n=10: 60*/70* | n=20: 80*/90* | n=40: 110* | n=80: 130* | n=160: 110* | n=226: 200 | n=320: 280*/290 | n=453: 850/770 | n=640: no(0.97)/no(0.97) | n=1280: no(0.82)
```

What the table shows: at 100 epochs only the 64-wide model finishes even 10 to 20 entries; the boundary at 1000 epochs sits near 14 to 20
entries (d=8), 113 (d=16), 226 to 320 (d=32) and 453 (d=64); and the first rung that fails ends at 0.95 to 0.98 exact recall rather than far
below it. From that came the per-width ladders, the 1000-epoch run length and the budgets 100, 300, 500, 1000.

**The Monte Carlo** (`e4_montecarlo.py`, run with `--sigma 0.25` and `--sigma 0.4`) treats the first full-recall epoch as lognormal around a
curve through the calibration medians, with independent noise per book seed, applies the pre-registered rule (2 of 3 seeds, contiguous
rungs) and reports the distribution of the capacity at each budget. The twelve point predictions of `e4_predictions.json` (4 widths at
100, 300 and 1000 epochs) are its modes, which are the same at both sigmas. Each predicted range equals the 80% range at one of the two
sigmas, except d=32 at 300 epochs, [80, 226]: the lower end is the one at sigma 0.25 and the upper end the one at sigma 0.4. The
boundary-status predictions (moving, moving, stable, moving) are its most likely outcomes (0.91, 0.97, 0.48, 0.76 at sigma 0.4). The model is
a guess fitted to the 57 cells: a run that stalled below 100% at epoch 1000 is assumed to finish at epoch 1500 (3000 if it was far off), and
the calibration seeds share one book, so the book-to-book spread was not measured (hence the two sigmas). E4 later scored 23 of 23 lines as
hits, with all four 1000-epoch capacities equal to the modes; that is one confirmation of this prior, not a validation of the model.

## E5 pilots

`e5_pilots.py d0` reruns E3's recipe on the E3 phonebook's three pinned Phase-A checkpoints (hashes checked against
`specs/e3_replay_ratio.json`) under several shuffle seeds; `e5_pilots.py phonebooks` generates phonebooks that E5 does not use (generator seeds 15, 17,
18, 20, 26, 32, 34, 36, 38, 39), trains three Phase-A models for each, and replays A once plus B x k. `python
phonebook/calibration/e5_pilot_report.py` prints, from the raw files:

- **Timing noise.** 56 groups of runs that differ only in the shuffle seed (18 + 6 + 12 on the E3 phonebook, 20 on the other
  phonebooks): none had a nonzero spread in the joint-recall epoch. This is why E5's prediction for the noise band was 0. (The locked
  prediction also says the final weights differ by about 0.002 between shuffle seeds; those weights are not kept here, so that figure
  cannot be rechecked from this folder.)
- **Censoring and flicker.** On the ten other phonebooks, 0 of 340 runs at lr 0.001 and 0.0003 and 0 of 120 at lr 0.003 failed to reach a stable joint
  recall; 2 of the 340 recalled both books once and then lost one again (a flicker), which is why E5 uses `stable_joint`.
- **Pooled tables.** At lr 0.001, k = 2 was faster than k = 1 in 9 of 11 phonebooks (the E3 phonebook plus the ten), k = 3 in 4, k = 4 in 1, k = 5 and
  k = 9 in none; at lr 0.0003, k = 2 was faster in 11 of 11.
- **Bootstrap.** Drawing 6 of the 11 phonebooks with replacement and applying the E5 rule with a noise band of 0: k = 2 beats k = 1 in 70%
  of draws, k = 2 is the fastest k in 80% and k = 1 in 17%, and the median time at k = 1 averages 36.8 (10% to 90%: 34 to 40).

The locked E5 predictions drew on these numbers: Q1 (the k = 1 time) on the phonebook-level times of 31 to 41, Q2 and Q4 to Q6 (counts, the
rule, the plateau, the fastest k) on the pooled tables and the bootstrap, Q3 on the noise groups, Q7 on the dip medians, Q8 on the censoring
census (the 18 plain fine-tuning runs are censored by construction) and Q9 on the lr 0.0003 arm. A bootstrap of 11 phonebooks overstates
how well the pool represents new phonebooks; E5 hit 13 of its 15 scored lines and missed Q2 for k = 9 and the Q5 plateau.

## Throughput benchmark

500-epoch replay runs (k = 5) on the E3 checkpoint, measured once while choosing how to run the batches:

| Torch threads per run | Runs at a time | Time per run | Wall time | Wall time per run |
|---|---|---|---|---|
| default | 1 | 48 s | 48 s | 48 s |
| default | 7 | about 293 s | 294 s | 42 s |
| 1 | 1 | 76 s | 76 s | 76 s |
| 1 | 7 | about 174 s | 175 s | 25 s |
| 1 | 11 | about 288 s | 291 s | 26 s |

Seven single-thread runs at a time was the best of these: seven runs finished in 175 s, against 336 s (7 x 48 s) one after another on
default threads, a factor of 1.9, while seven concurrent runs on default threads took 294 s, a factor of only 1.14. Eleven at a time was
no better than seven. The numbers belong to one machine and one measurement.

## How each raw file was made

All with one torch thread unless noted; `WORK` is any scratch folder.

```bash
python phonebook/calibration/e4_calibration.py books --work WORK
python phonebook/calibration/e4_calibration.py run --work WORK --cells D:N:EPOCHS[:SEED],... --workers 7      # appends to WORK/results.jsonl, of which e4_calibration_results.jsonl is a copy
python phonebook/calibration/e5_pilots.py d0 --work WORK --out phonebook/calibration/e5_pilot_results/d0_lr0.001.json --lr 0.001 --ks 1,2,3,4,5,9 --seeds 42,1,2,3,4 --epochs 120
python phonebook/calibration/e5_pilots.py d0 --work WORK --out phonebook/calibration/e5_pilot_results/d0_lr0.0003.json --lr 0.0003 --ks 1,2,3,5 --seeds 42,1,2 --epochs 200
python phonebook/calibration/e5_pilots.py d0 --work WORK --out phonebook/calibration/e5_pilot_results/d0_lr0.001_default_threads.json --lr 0.001 --ks 1,3 --seeds 42,1,2,3 --epochs 120 --threads default --tag _mt
python phonebook/calibration/e5_pilots.py phonebooks --work WORK --out phonebook/calibration/e5_pilot_results/phonebooks_a.json --pbs 15,17,18,20 --stage main
python phonebook/calibration/e5_pilots.py phonebooks --work WORK --out phonebook/calibration/e5_pilot_results/phonebooks_b.json --pbs 26,32,34,36,38,39 --stage main
python phonebook/calibration/e5_pilots.py phonebooks --work WORK --out phonebook/calibration/e5_pilot_results/phonebooks_lr0.003.json --pbs 15,17,18,20,26,32,34,36,38,39 --stage lr003
python phonebook/calibration/e4_montecarlo.py --sigma 0.25
python phonebook/calibration/e4_montecarlo.py --sigma 0.4
python phonebook/calibration/e5_pilot_report.py
```

The `d0` pilots need the E3 Phase-A checkpoints named in `e5_pilots.py` (`D0_CHECKPOINTS`), which are local files and not committed; the
hash of each is checked against E3's pin first. The calibration cells were run as a 300-epoch coarse ladder first and then 1000-epoch
cells, with training seeds 100 and 101; the order was not recorded beyond that.

## What was checked when this code was cleaned up

The scripts were first written as scratch files with absolute paths. They were rewritten to be portable, and then compared with the originals
and with the recorded data:

- the 18 calibration books regenerate byte for byte;
- six calibration cells (widths 8, 16, 32, 64; 300 and 1000 epochs; both training seeds) retrained from scratch reproduce the recorded
  first-full-recall epoch and the whole 30- or 100-point accuracy curve exactly;
- all 12 phonebook-15 cells at lr 0.003, including three Phase-A models trained from scratch, and the three E3-phonebook cells at k = 3 reproduce exactly;
- the Monte Carlo output is byte-identical to the original script's at sigma 0.25 and 0.4, and the report's tables and bootstrap equal the original's line for line;
- `tests/test_calibration.py` (24 tests) pins the logic and the quoted numbers. A one-line bug planted in these files, one at a time, was
  caught for each of 25 plants. The first round caught 13 of 17; the four misses (a redundant guard, the epochs of the lr 0.003 arm, and the
  tie-break and shuffle-seed selection of the report) became tests that pin each job grid to the cells recorded in the result files, and
  the shuffle-seed and tie-break choices to a synthetic table and the printed lines.

Two things went wrong during the original work and are worth knowing. The first batch driver swallowed the error for seven calibration
cells whose books did not exist (the intermediate rungs), so the first results file had 43 rows instead of 50; the cells were rerun once the
books existed, and `run_batch` now reports a failed cell and exits 1. And the audit of one E4 evidence folder failed at the default thread
count; that is documented with the E4 results, not here.
