# Defense sheet: how many copies of the new book to replay (E3 and E5)

*Status as of November 4, 2026. A study sheet for explaining the replay-ratio experiments without help. Every number in it is rechecked against the committed results by `tests/test_defense_sheets.py`, and the numbers table says where each one comes from. Companion sheet: [capacity (E4 to E6)](capacity-e4-e6.md). All claims and their evidence: [RESULTS.md](../../RESULTS.md).*

## 1. The question

If the replay data holds one copy of the old phonebook A and k copies of the new phonebook B, how does k change the forgetting of A, and which k gets the model to exact recall of both books in the fewest updates? This is the course README's replay question. E3 asked it on one phonebook; E5 repeated it on six independent ones and fixed the decision rule before the runs.

## 2. The claims and their status

| # | claim | status |
|---|---|---|
| 5 | k > 1 reaches joint recall slightly sooner than k = 1 at every learning rate; k = 9 does not | Observed in the medians at every learning rate; small; run-to-run timing noise not measured; the paired sign tests are exploratory |
| 6 | Any particular k among 2, 3, 5 is best | **Not established in E3; E5 narrows it**: k = 2 and k = 3 tie in the median (38.5), k = 2 is faster than k = 1 in more single runs (15 of 18 against 10 of 18), k = 5 is slower |
| 7 | Counts near the L2-SP regime change carry about one entry of numerical noise | Measured in E1b against E1; d1 shows the logging interval alone changes the trajectory. For replay timing at lr = 0.001, E5 found no variation at all under five other shuffle seeds (noise band 0) |
| 10 | At lr = 0.001, k = 2 reaches joint recall slightly sooner than k = 1 (about 1.5 updates, 5 of 6 phonebooks); no other k does; k = 9 is much slower | Established by the pre-registered rule on six phonebooks (E5); small, and the rule is a sign rule because the noise band is 0 |

Read the status words literally: "established" means the pre-registered rule was met on these runs, not that the effect is large or holds for other models, data or learning rates.

## 3. The design

- **What varied.** Inside a phonebook only k changed, the number of copies of B per copy of A, over the values in [R04] (E3 used five values of k, listed in [R26]). The learning rate also varied: the E5 verdict is read at lr 0.001 [R04], and the other rates are descriptive only [R05].
- **What was held fixed.** Each E5 run starts from a Phase-A model that already recalls its own book A exactly and has never seen book B, then trains for the updates and batch size in [R02], one full batch per update. Every book holds the entries in [R32]. The model size and the optimizer (full-batch AdamW) are the same in every run.
- **The unit of replication.** The phonebook, not the run. The phonebooks are 2, 4, 7, 8, 9 and 12, each with Phase-A inits 42, 123 and 999 [R03]. The three inits of a phonebook are reduced to their median first, and every k is compared with k = 1 inside the same phonebook [R04].
- **The measure.** stable_joint: the first update from which every logged update has both books recalled exactly; a run that never gets there would count as infinitely slow, and none in E3 did [R31]. Also the dip (the fewest A entries recalled at any update) and the work in sequences processed [R17].
- **The rule, fixed before the runs.** The verdict rule was written into the spec before the first E5 run: at lr 0.001 only, k beats k = 1 if it is faster in at least 5 of 6 phonebooks and its median paired gain exceeds the noise band [R33]. The noise band is the largest spread of stable_joint among runs that differ only in the shuffle seed [R07].
- **Controls and checks.** A control with no copy of A must lose A [R06]. A noise arm measures the shuffle noise [R07]. An anchor reruns part of E3 under the historical loader and must reproduce it exactly [R34]. The runs were audited, and the pre-registered predictions were scored mechanically [R22].

## 4. Numbers

| id | statement | value | source |
|---|---|---|---|
| R01 | E3 design: the grid of runs | 60 runs = 3 checkpoints x 5 values of k x 4 learning rates | [e3_cells.csv](../../results/e3_cells.csv) |
| R02 | E3 and E5 schedule: updates per run, batch size, logging | 500 updates per run, batch size 128, logged every 1 update, one update per epoch | [e3_replay_ratio.json](../../specs/e3_replay_ratio.json), [e5_main_pb2.json](../../specs/e5_main_pb2.json) |
| R03 | E5 design: independent phonebooks and Phase-A starting models | 6 phonebooks (2, 4, 7, 8, 9, 12) x 3 Phase-A inits (42, 123, 999) | [e5_cells.csv](../../results/e5_cells.csv) |
| R04 | E5 grid at lr 0.001 | 108 runs = 6 phonebooks x 3 inits x 6 values of k (1, 2, 3, 4, 5, 9) | [e5_cells.csv](../../results/e5_cells.csv) |
| R05 | E5 other learning rates (descriptive) | learning rates 0.003 and 0.0003 use k = 1, 2, 3, 5, 72 runs each | [e5_cells.csv](../../results/e5_cells.csv) |
| R06 | E5 control with no copy of A (plain fine-tuning), lr 0.001 | 18 runs without a copy of A: A ended at 0 of 10 in 18, B at 10 of 10 in 18, joint recall reached in 0 | [e5_cells.csv](../../results/e5_cells.csv) |
| R07 | E5 noise arm: runs that differ only in the shuffle seed, lr 0.001 | 6 configurations x 6 runs (shuffle seeds 1, 2, 3, 4, 5, 42), largest spread of stable_joint 0 | [e5_cells.csv](../../results/e5_cells.csv) |
| R08 | Median over the six phonebooks of updates to stable joint recall, lr 0.001, k = 1, 2, 3, 4, 5, 9 | 40.0, 38.5, 38.5, 40.5, 44.0, 54.0 | [e5_cells.csv](../../results/e5_cells.csv) |
| R09 | Median over the six phonebooks of the fewest A entries recalled at any update (the dip), lr 0.001, k = 1, 2, 3, 4, 5, 9 | 9.0, 6.5, 4.0, 3.0, 2.0, 0.5 | [e5_cells.csv](../../results/e5_cells.csv) |
| R10 | Phonebooks (of six) in which k is faster than k = 1, lr 0.001, k = 2, 3, 4, 5, 9 | 5, 3, 1, 1, 1 of 6 | [e5_cells.csv](../../results/e5_cells.csv) |
| R11 | Median over the six phonebooks of the updates saved against k = 1 (negative means slower), lr 0.001, k = 2, 3, 4, 5, 9 | 1.5, 0.5, -1.5, -6.0, -15.5 | [e5_cells.csv](../../results/e5_cells.csv) |
| R12 | Probability under no effect (a fair coin per phonebook) of at least 5 faster phonebooks out of 6, one-sided | 7/64 = 0.109 | arithmetic (binomial tail) |
| R13 | Single runs (18 pairs of phonebook and init): k = 2 against k = 1, lr 0.001 | 15 faster, 1 tied, 2 slower of 18; mean gain 1.2 updates | [e5_cells.csv](../../results/e5_cells.csv) |
| R14 | Single runs (18 pairs of phonebook and init): k = 3 against k = 1, lr 0.001 | 10 faster, 2 tied, 6 slower of 18; mean gain 0.1 updates | [e5_cells.csv](../../results/e5_cells.csv) |
| R15 | Single runs (18 pairs of phonebook and init): k = 9 against k = 1, lr 0.001 | 1 faster, 0 tied, 17 slower of 18; mean gain -15.4 updates | [e5_cells.csv](../../results/e5_cells.csv) |
| R16 | Updates saved by k = 2 against k = 1 in each phonebook (2, 4, 7, 8, 9, 12), lr 0.001 | -1, 3, 2, 1, 2, 1 | [e5_cells.csv](../../results/e5_cells.csv) |
| R17 | Sequences processed to stable joint recall (median updates x (10 + 10k)), lr 0.001, k = 1, 2, 3, 4, 5, 9 | 800, 1155, 1540, 2025, 2640, 5400 | [e5_cells.csv](../../results/e5_cells.csv) |
| R18 | Median over the six phonebooks of updates to stable joint recall at lr 0.003 | 19.0, 17.5, 19.0, 25.5 (k = 1, 2, 3, 5) | [e5_cells.csv](../../results/e5_cells.csv) |
| R19 | Median over the six phonebooks of updates to stable joint recall at lr 0.0003 | 96.5, 93.5, 91.5, 96.5 (k = 1, 2, 3, 5) | [e5_cells.csv](../../results/e5_cells.csv) |
| R20 | Phonebooks (of six) in which k is faster than k = 1, and the median updates saved, at the other learning rates | lr 0.003: k = 2, 3, 5 faster in 5, 1, 0 of 6, median gains 1.0, -0.5, -6.0; lr 0.0003: k = 2, 3, 5 faster in 5, 5, 3 of 6, median gains 7.0, 9.0, 2.0 | [e5_cells.csv](../../results/e5_cells.csv) |
| R21 | Median updates to stable joint recall for k = 1 at lr 0.003, 0.001, 0.0003 | 19.0, 40.0, 96.5 | [e5_cells.csv](../../results/e5_cells.csv) |
| R22 | E5 pre-registered predictions, scored mechanically | 13 of 15 scored lines hit (4 exact); misses Q2 and Q5 | [e5_prediction_scores.json](../../results/e5_prediction_scores.json) |
| R23 | B clock: median over the six phonebooks of how much sooner B is first fully recalled than at k = 1, lr 0.001, k = 2, 3, 4, 5, 9 (exploratory) | 2.0, 3.0, 3.5, 4.0, 4.5 | [experiments/week_2026-10-31/](../../experiments/week_2026-10-31/) |
| R24 | A clock: median over the 18 runs of the update from which A stays fully recalled, lr 0.001, k = 1, 2, 3, 4, 5, 9 (exploratory) | 18.0, 30.0, 36.0, 39.5, 44.0, 56.0 | [experiments/week_2026-10-31/](../../experiments/week_2026-10-31/) |
| R25 | Runs (of 18) in which A's return is strictly later than B's first full recall, lr 0.001, k = 1, 2, 3, 4, 5, 9 (exploratory) | 1, 2, 9, 12, 15, 18 of 18 | [experiments/week_2026-10-31/](../../experiments/week_2026-10-31/) |
| R26 | E3 median updates to stable joint recall, lr 0.001, k = 1, 2, 3, 5, 9 | 41, 40, 39, 44, 54 | [e3_cells.csv](../../results/e3_cells.csv) |
| R27 | E3 best k at lr 0.003, 0.001, 0.0003, 0.0001 and the updates saved against k = 1 | k = 3, 3, 5, 5; updates saved against k = 1: 4 (18%), 2 (5%), 6 (6%), 22 (8%) | [e3_cells.csv](../../results/e3_cells.csv) |
| R28 | E3 k = 2 against k = 1 over all learning rates and checkpoints (exploratory) | 11 faster, 0 tied, 1 slower of 12 pairs; two-sided sign test p = 0.0063 | [e3_cells.csv](../../results/e3_cells.csv) |
| R29 | E3 k = 9 against k = 1 over all learning rates and checkpoints (exploratory) | 4 faster, 0 tied, 8 slower of 12 pairs; two-sided sign test p = 0.3877 | [e3_cells.csv](../../results/e3_cells.csv) |
| R30 | E3 median dip (fewest A entries recalled), lr 0.001, k = 1, 2, 3, 5, 9 | 8, 5, 3, 1, 1 | [e3_cells.csv](../../results/e3_cells.csv) |
| R31 | E3 completeness | 0 censored of 60; 60 of 60 end with A and B at 10 of 10; first_joint equals stable_joint in 60 of 60 | [e3_cells.csv](../../results/e3_cells.csv) |
| R32 | E5 book sizes | every book holds 10 entries (A and B of each of the 6 phonebooks) | [specs/](../../specs/), [inputs/](../../inputs/) |
| R33 | E5 verdict rule, quoted from the spec | fixed before the runs: at lr 0.001 only, k beats k = 1 if it is faster in at least 5 of 6 phonebooks and its median paired gain exceeds the noise band | [e5_main_pb2.json](../../specs/e5_main_pb2.json) |
| R34 | E5 anchor: E3's k = 1 and k = 3 rerun under the historical loader | 6 anchor runs compared with E3 over 500 updates each, 0 mismatches | [compare_with_e3.json](../../experiments/week_2026-10-31/e5_anchor_e3/compare_with_e3.json) |
| R35 | E5 tripwires of the locked story | k = 4 or 5 faster than k = 1 in 3 or more phonebooks: no; noise band of 3 or more updates: no | [e5_prediction_scores.json](../../results/e5_prediction_scores.json) |

## 5. The mechanism

**Exploratory.** Joint recall waits for two things, the two clocks: B must be memorized and A must come back. The update at which both books stay recalled is the later of the two, which is true by definition, so only the trends below carry evidence. The picture was written into the locked predictions before E5 ran, from scratch pilots on other phonebooks; the numbers below were computed after the E5 results were read, and no run separated the two clocks. Two tripwires written into the locked predictions, which would have broken the picture, did not fire [R35].

- The B clock barely moves with k: against k = 1 the median saving in the update at which B is first fully recalled is 2.0 at k = 2 and 4.5 at k = 9 [R23]. A reason offered in the locked predictions, not tested: Adam takes steps of roughly fixed size, so extra copies of B change the direction of the step more than its speed.
- The A clock grows with k: the median update from which A stays fully recalled is 18.0 at k = 1 and 56.0 at k = 9 [R24]. A is the later clock in 1 of 18 runs at k = 1 and in 18 of 18 at k = 9 [R25].
- The most any k can save is therefore about the small B saving, which is why k = 2 gains little and larger k loses: the median change against k = 1 is 1.5, 0.5, -1.5, -6.0 and -15.5 updates for k = 2, 3, 4, 5 and 9 [R11].
- Why the dip deepens: A supplies 10 of the 10 + 10k sequences in every update [R17], and the loss is a mean over all supervised tokens in the batch, so A's share of the gradient is roughly its share of the sequences and shrinks as k grows. The measured dip is 9.0, 6.5, 4.0, 3.0, 2.0 and 0.5 entries for k = 1, 2, 3, 4, 5 and 9 [R09]. No run in E3 or E5 changed that share without changing k, so this reading is untested.

## 6. What was not shown

- That k = 2 is better than k = 1 in any way that matters: the median saving is 1.5 updates [R11], seeing at least 5 faster phonebooks out of 6 has probability 7/64 = 0.109 under no effect [R10, R12], and one phonebook goes the other way [R16].
- Which k is best in general: E5 is read at lr 0.001 [R04], and the other learning rates are descriptive only [R05].
- That k = 2 is cheaper: counted in sequences processed, k = 1 is the cheapest by far, 800 against 1155 for k = 2 [R17].
- That the noise band of 0 means the runs have no variation: it measures only the shuffle order [R07], while phonebooks and Phase-A inits do vary (the gain of k = 2 over k = 1 across the six phonebooks is -1, 3, 2, 1, 2, 1 [R16]).
- Anything beyond one task type, one model size and one optimizer: random books of 10 entries only [R32], no other data, no other model size, no replay of only part of A.
- That the two-clocks mechanism is the cause: it fits the medians and its tripwires did not fire [R35], but no run held one clock fixed while changing the other.
- Anything about L2-SP or other regularizers; this sheet covers only the replay ratio, and RESULTS.md has the rest.

## 7. Twelve hard questions

**Q1. What is the unit of replication, and why is it not the individual run?**
Answer. The phonebook. The three Phase-A inits of one phonebook share the same two books, so their runs are not independent evidence about phonebooks; each phonebook's inits are reduced to a median first, and every k is compared with k = 1 inside the same phonebook [R03, R04]. With only six phonebooks the best possible sign pattern is 6 of 6, and a pattern at least as strong as the observed 5 of 6 has probability 7/64 = 0.109 under no effect [R10, R12].
Must mention. the inits of one phonebook share the books; a median per phonebook, then paired with k = 1; six phonebooks is a small sample; 5 of 6 or better has probability 7/64 [R10, R12].

**Q2. The noise band is 0 [R07]. Does that mean the runs have no noise?**
Answer. No. A different shuffle seed only reorders the rows inside the one full batch of every update [R02], and it never moved the update at which joint recall is reached: every configuration of runs that differ only in the shuffle seed has spread 0 [R07]. That is rounding-level noise from the shuffle, not the variation between phonebooks and Phase-A inits, which is real: k = 2 is slower than k = 1 in one phonebook [R16]. A noise band of 0 turns the pre-registered rule into a sign rule [R07, R33].
Must mention. the shuffle only reorders rows inside one batch; spread 0 in every noise configuration; the real spread is between phonebooks and inits; band 0 makes the rule a sign rule [R07].

**Q3. k = 2 is said to beat k = 1 by 1.5 updates [R11]. Is that a real effect?**
Answer. By the rule fixed before the runs, yes: k = 2 is faster in 5 of 6 phonebooks and the median gain is 1.5 updates, above a band of 0 [R07, R10, R11]. It is a small, consistent shift and not strong evidence: 5 of 6 or better has probability 7/64 = 0.109 under no effect, phonebook 2 goes the other way with a gain of -1, and in single runs k = 2 is faster in 15 of 18 with a mean gain of 1.2 updates [R12, R13, R16]. The claims table calls it established by the rule and small; it does not say the effect is large or that it holds elsewhere.
Must mention. the pre-registered sign rule was met; 5 of 6 phonebooks; median gain 1.5 updates; probability 7/64 under no effect; small and not general [R10, R11, R12].

**Q4. Why is k = 9 slower than k = 1 [R15]?**
Answer. The two clocks (exploratory). B is memorized sooner at k = 9 than at k = 1, but only by 4.5 updates [R23], while the update from which A stays recalled has a median of 56.0 against 18.0 [R24], and A is the later clock in 18 of 18 runs [R25]. The net result is a median of 54.0 updates against 40.0 [R08], and k = 9 is slower than k = 1 in 17 of 18 single runs [R15].
Must mention. the B clock saves only a few updates; A's return time grows with k; A is the later clock in every run at k = 9; slower in 17 of 18 runs [R15, R25].

**Q5. Why does the dip in A deepen as k grows?**
Answer. A supplies 10 of the 10 + 10k sequences in every update [R17], and the loss is a mean over all supervised tokens in the batch, so A's share of the gradient is roughly its share of the sequences and shrinks as k grows. No run isolated that share, so it is a reading, not a result. What was measured: the median fewest A entries recalled is 9.0, 6.5, 4.0, 3.0, 2.0 and 0.5 for k = 1, 2, 3, 4, 5 and 9 [R09], and plain fine-tuning, with no copy of A, ends at 0 of 10 for A in 18 of 18 runs [R06]. E3 found the same pattern on its own phonebook: 8, 5, 3, 1 and 1 for k = 1, 2, 3, 5 and 9 [R30].
Must mention. A's share is 10 of 10 + 10k sequences; the dip goes from 9.0 to 0.5 as k goes from 1 to 9; plain fine-tuning loses A completely in 18 of 18 runs; the share argument is untested [R06, R09, R17].

**Q6. Is the number of updates the right cost? What about the work done?**
Answer. An update is one optimizer step, but it processes 10 + 10k sequences, so a larger k costs more per update [R17]. In sequences processed to reach stable joint recall k = 1 is the cheapest by far: 800, 1155, 1540, 2025, 2640 and 5400 for k = 1, 2, 3, 4, 5 and 9 [R17]. The claim that k = 2 is faster is about updates only; in sequences it costs more than k = 1 (1155 against 800 [R17]).
Must mention. an update counts steps, not work; 10 + 10k sequences per update; k = 1 is the cheapest in sequences (800 against 1155 for k = 2) [R17].

**Q7. E3 picked k = 3 and E5 picks k = 2 at lr 0.001 [R08, R26]. Which one is right?**
Answer. They used different data and rules, and they agree on the shape. E3 had one phonebook with three checkpoints; its rule (smallest median, ties to the smaller k) picked k = 3, 3, 5, 5 at lr 0.003, 0.001, 0.0003, 0.0001, saving 4, 2, 6 and 22 updates [R27], and the E3 medians at lr 0.001 were 41, 40, 39, 44, 54 for k = 1, 2, 3, 5, 9 [R26]. E5 used six phonebooks: the medians are 40.0, 38.5, 38.5, 40.5, 44.0, 54.0 [R08], so k = 2 and k = 3 tie and the sign rule passes only k = 2 [R10]. In single runs k = 3 is hard to tell from a coin flip against k = 1 (10 faster, 2 tied, 6 slower [R14]). Both show that, at lr 0.001, k of 2 or 3 is a little faster than k = 1 and that k = 9 is slower [R11]; E3's one phonebook is a single sample, which is why E5 was run.
Must mention. E3 used one phonebook with three checkpoints and E5 six phonebooks; the medians tie at 38.5 for k = 2 and k = 3; both show a small gain for k of 2 or 3 and k = 9 slower [R08].

**Q8. What did the E3 sign tests show, and why were they not enough?**
Answer. E3 compared each k with k = 1 over 12 pairs: k = 2 was faster in 11 of 12 with a two-sided sign test p = 0.0063, and k = 9 was faster in 4 and slower in 8 [R28, R29]. They were exploratory: the 12 pairs are not independent (the same 3 checkpoints appear at all 4 learning rates [R01, R28]), the comparisons were made without correction for several tests, and they were not pre-registered. E5 fixed the rule first and made the phonebook the unit.
Must mention. 12 pairs that are not independent; 11 of 12 and p = 0.0063 for k = 2; not pre-registered and no correction for several tests; E5 fixed the rule first [R28].

**Q9. Does the answer change with the learning rate?**
Answer. The learning rate moves the time far more than k does: the median for k = 1 is 19.0, 40.0 and 96.5 updates at lr 0.003, 0.001 and 0.0003 [R21]. The other rates are descriptive. At lr 0.003 the medians are 19.0, 17.5, 19.0, 25.5 for k = 1, 2, 3, 5 [R18]; at lr 0.0003 they are 96.5, 93.5, 91.5, 96.5 [R19]; and at lr 0.0003 k = 2 and k = 3 are each faster than k = 1 in 5 of 6 phonebooks, with median gains of 7.0 and 9.0 updates [R20]. In the medians a lower learning rate favours a somewhat larger k, but no rule was pre-registered for these rates.
Must mention. the k = 1 median is 19.0, 40.0 and 96.5 at the three rates; the learning rate matters more than k; lower rates favour a larger k only descriptively; no pre-registered rule for the other rates [R21].

**Q10. How did the pre-registered predictions do?**
Answer. 13 of 15 scored lines hit [R22]. The two misses: k = 9 was predicted faster than k = 1 in no phonebook and was faster in one, and the plateau was predicted as k = 2 alone but came out as k = 2 and k = 3 because their medians tie [R10, R22]. Two further hits depend on that tie. Do not read 13 of 15 as 13 independent confirmations: the lines share the same runs [R22]. The tripwires meant to break the story did not fire [R35].
Must mention. 13 of 15 hit; the two misses (k = 9 in one phonebook, the plateau {2, 3}); the lines are correlated and some hits depend on a tie; the tripwires did not fire [R10, R22].

**Q11. Why a control with no copy of A, and why "stable" joint recall?**
Answer. The control shows what replay is fixing: training on B alone drops A to 0 of 10 in all 18 runs and joint recall is never reached [R06]. Stable joint recall (the update from which both books stay recalled) is used instead of the first update at which both are recalled, so that a brief touch of both books at 10 of 10 does not count; in E3 the two coincide in all 60 runs [R31].
Must mention. plain fine-tuning loses A in every run; stable means both books stay recalled; first and stable coincide in E3 [R06, R31].

**Q12. What is the one-sentence takeaway you would defend, and what would you refuse to claim?**
Answer. On this task, with small random phonebooks and one small model, replaying two copies of B per copy of A reaches joint recall about 1.5 updates sooner than one copy in the median [R11], nine copies change it by -15.5 updates (slower) with a deeper dip in A [R09, R11], and the learning rate matters far more than k [R21]. I would refuse to claim an optimal k in general, a large effect, that replay is cheaper in work (k = 1 is the cheapest in sequences [R17]), or anything beyond this model size, task and optimizer.
Must mention. a small gain at k = 2; k = 9 slower with a deeper dip; the learning rate matters more; the limits (one task, one model size, one optimizer) and no claim of an optimal k [R11].

## 8. Glossary

- **exact recall**: an entry counts as recalled only if greedy generation from its name produces the whole phone number and the newline, with no length hint.
- **update**: one optimizer step. In E3 and E5 every update is one full batch holding one copy of A and k copies of B, and a run lasts the number of updates in [R02].
- **k**: the number of copies of B per copy of A in the replay data; k = 1 is balanced replay [R04].
- **joint recall**: exact recall of both books at the same time.
- **stable_joint**: the first update from which every logged update through the end of the run has both books recalled exactly; a run that never gets there would be censored and count as infinitely slow, and none of E3's did [R31].
- **dip**: the fewest entries of A recalled exactly at any update, a measure of how much of A is forgotten on the way [R09].
- **noise band**: the largest spread of stable_joint among runs that differ only in the shuffle seed; it is 0 here [R07].
- **sign rule**: what the pre-registered verdict rule becomes when the noise band is 0: k beats k = 1 if it is faster in at least 5 of 6 phonebooks and the median gain is above 0, however small [R07, R33].
- **unit of replication**: the thing whose repeats count as independent evidence; here the phonebook, with its three Phase-A inits reduced to a median [R03].
