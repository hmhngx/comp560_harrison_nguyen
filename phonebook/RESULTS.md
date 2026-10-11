# Phonebook forgetting: what the experiments show

*Status as of Sunday October 11, 2026. Protocol, per-experiment results and audits:
[`experiments/week_2026-10-07/README.md`](experiments/week_2026-10-07/README.md). Derived tables and
figures: [`results/`](results/).*

## The question

A small transformer (64 dimensions, 2 layers, 2 attention heads, 110,528 parameters) memorizes the
phone numbers of one phonebook (phase A, ten entries) and is then adapted to a second one (phase B,
ten entries). Plain adaptation forgets the first book. Two remedies are compared: **balanced
replay** (keep training on A while learning B) and **L2-SP** (penalize drifting from the phase-A
weights; needs no old data). A second question comes from the course README: if the replay data
holds one copy of A and k copies of B, which k reaches 100% on all the data fastest?

An entry counts as *recalled* only if greedy generation from its name produces the whole phone
number and the newline, with no length hint. Counts are entries recalled out of ten per book.

## Findings

1. **Replay learns both books.** At the trainer's default learning rate (0.001), balanced replay
   recalls 10 of 10 entries of A and of B after 100 and after 300 updates on all three phase-A
   checkpoints of the original phonebook (E1) and on three new phonebooks (E2). Both books are
   first recalled together at update 33 to 51 (E1) and 36 to 48 (E2), after A
   dips to 8 to 9 of 10. Plain fine-tuning ends with B at 10 and A at 0: B is first complete
   at update 31 to 44 and A is down to 0 by update 12 to 19.
2. **No L2-SP strength matches replay.** On the original phonebook (E1, nine strengths from
   λ = 0.01 to 100), on a finer grid trained for 1000 updates (E1b) and on the new phonebooks
   (E2), the pre-registered rule (one λ with A = 10 and B = 10, or A ≥ 9 and B = 10, on at least two
   of three checkpoints) gives "no" every time. L2-SP trades one book for the other along λ, with
   the change between λ = 0.3 and λ = 1: whenever a run kept at least 8 of A's entries it recalled 0
   of B's, and whenever it recalled at least 8 of B's it kept at most 4 of A's.
3. **Repeating B (the README's question).** With one copy of A and k copies of B, the smallest median
   time to joint recall is at k = 3 for lr = 0.003 and 0.001 and at k = 5 for lr = 0.0003 and
   0.0001, so k > 1 is faster at every learning rate, but only by 2 to 22 updates (5% to 18%): the
   choice among k = 2, 3, 5 is not resolved, and k = 9 is not faster. The temporary dip in A deepens
   with k (at lr = 0.001 the median fewest A entries recalled is 8, 5, 3, 1, 1 for
   k = 1, 2, 3, 5, 9). The learning rate matters far more than k.
4. **Counts in the regime-change region are noisy by about one entry.** Two runs that differ only
   in how often accuracy is logged have different shuffle orders of the training batch, and exact
   counts at the same update differ by one entry at about one in five updates there, rarely more.
   This failed a pre-registered cross-check (E1b against E1); its cause was then tested (d1).

## Claims and evidence

| # | Claim | Status | Evidence |
|---|---|---|---|
| 1 | At lr = 0.001, balanced replay recalls both books exactly at 100 and 300 updates | Established on this task, replicated on 3 new phonebooks | [`e1_endpoints.csv`](results/e1_endpoints.csv), [`e2_endpoints.csv`](results/e2_endpoints.csv) |
| 2 | Plain fine-tuning at lr = 0.001 ends with B 10 of 10 and A 0 of 10 | Established, replicated | same files; timing in the experiment README |
| 3 | No tested L2-SP strength matches replay at lr = 0.001 | Established for isotropic L2-SP, λ from 0.01 to 100, 100 to 1000 updates | [`e1_decision.json`](results/e1_decision.json), [`e1b_decision.json`](results/e1b_decision.json), [`e2_decision.json`](results/e2_decision.json) |
| 4 | L2-SP shows a trade-off with no λ that keeps A and learns B (A ≥ 8 implies B = 0; B ≥ 8 implies A ≤ 4) | Established on these runs | [`e1b_fine_grid.png`](results/e1b_fine_grid.png), [`e2_dose_response.png`](results/e2_dose_response.png) |
| 5 | k > 1 reaches joint recall slightly sooner than k = 1 at every learning rate; k = 9 does not | Established as a small effect; the paired sign tests are exploratory | [`e3_table.csv`](results/e3_table.csv), [`e3_cells.csv`](results/e3_cells.csv) |
| 6 | Any particular k among 2, 3, 5 is best | **Not established** (medians within 3 to 8 updates; timing noise not measured) | [`e3_heatmaps.png`](results/e3_heatmaps.png) |
| 7 | Counts near the L2-SP regime change carry about one entry of numerical noise | Established (d1: the logging interval alone changes the trajectory) | [`d1_logging_interval/`](experiments/week_2026-10-07/d1_logging_interval/) |
| 8 | L2-SP is the more promising direction; replay's trade-off is binary | **Retracted** (see below) | [`sweep_lr/README.md`](sweep_lr/README.md) Parts 9 and 10 |
| 9 | Other regularizers (EWC), other model sizes, larger or other kinds of data, capacity | **Not tested** | |

## What was retracted

The September 29 synthesis (sweep README Parts 8 and 9) concluded that replay learns none of book
B while L2-SP's partial learning of B made L2-SP the direction to pursue. On October 4 an audit found
that it compared L2-SP's B *token* accuracy with replay's B *exact* recall, and that replay had
been run at lr = 0.0001 for 100 updates, which is too little for B (Part 10). This week's
experiments tested the question properly: at lr = 0.001 replay recalls both books and no L2-SP
strength does (findings 1 and 2).

## Figures

![Exact recall per update, three checkpoints (E1)](results/e1_curves.png)

![Recall after 100 and 300 updates as lambda grows (E1)](results/e1_dose_response.png)

![Finer lambda grid and longer horizon (E1b)](results/e1b_fine_grid.png)

![Recall as lambda grows, three new phonebooks (E2)](results/e2_dose_response.png)

![Time to joint recall and depth of the dip in A (E3)](results/e3_heatmaps.png)

## Limits

- One task type (20 random names and phone numbers, ten entries per book), one model size, one
  optimizer setting apart from the learning rate; counts move in tenths, so a one-entry difference
  is small.
- Three Phase-A checkpoints of one phonebook (E1, E1b, E3) and three new phonebooks (E2), one
  shuffle stream per run. Medians in E3 are over three checkpoints.
- L2-SP here is the isotropic penalty from the mitigation notes. A "no" is not a claim about every
  regularizer; a Fisher-weighted penalty (EWC) was not tried.
- E3 counts updates, not work: at k = 5 each update processes 60 sequences against 20 at k = 1.
- Replay uses old data and L2-SP does not, so "replay wins" says nothing about the case where old
  data are unavailable, only that isotropic L2-SP does not rescue it here.

## Next steps (candidates, not promises)

1. A Fisher-weighted penalty (EWC), and replay of only a fraction of A, to see what is needed
   when old data are limited.
2. Capacity: how many entries fit before 100% accuracy is impossible (the README's other bullet).
3. Decouple the training shuffle from evaluation (a dedicated `torch.Generator`), then measure
   run-to-run noise in the timing so that differences among k can be resolved.
4. Other kinds of data than random phone numbers.

## Checking this

Every experiment is a committed spec in [`specs/`](specs/) with predictions or rules fixed before
the runs, and an independent auditor covered every run: 159 training runs and 3180
generated answers, all consistent. From a fresh clone, the committed files alone can be re-audited
(run from the repository root):

```bash
python phonebook/audit_experiment.py --root phonebook/experiments/week_2026-10-07/e3_replay_ratio
```

This reconciles the manifest, protocol, the spec at its recorded commit, the summaries, the logged
metrics and the saved raw answers (each re-scored from the strings). The model weights are not
committed; with them in place, `--require-local-artifacts` also re-generates every answer from the
saved weights, which is how the audits recorded in each folder were run. One pre-registered check
failed, E1b against E1; it is reported as failed in the experiment README with its verified cause
(d1).

## AI assistance

Claude Code assisted with the code, experiments, audit and writing.
