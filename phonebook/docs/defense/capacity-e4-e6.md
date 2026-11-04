# Defense sheet: how many entries a tiny transformer can memorize (E4 to E6)

*Status as of November 4, 2026. A study sheet for explaining the capacity experiments without help. Every number in it is rechecked against the committed results by `tests/test_defense_sheets.py`, and the numbers table says where each one comes from. Companion sheet: [replay ratio (E3 to E5)](replay-ratio-e3-e5.md). All claims and their evidence: [RESULTS.md](../../RESULTS.md).*

## 1. The question

How many random name-to-number entries can a very small transformer memorize exactly, and what limits it: the size of the model or the time it is trained? The course README asks for the capacity of a very small model, found by training on larger and larger phonebooks until the model can no longer recall every entry. E4 measured the capacity at four training budgets for four widths; E6 trained the first rung that failed for longer, 4000 epochs instead of 1000 [C02].

## 2. The claims and their status

| # | claim | status |
|---|---|---|
| 11 | A 2-layer, 2-head model memorizes 14, 113, 226 and 453 random entries at widths 8, 16, 32, 64 within 1000 epochs | Established for this budget, learning rate and exact-recall criterion, with three book seeds per rung |
| 12 | Capacity is limited by training time more than by parameter count | Supported: E4 (narrow models still improving, boundaries moving, 0.13 to 0.40 bits per parameter) and E6 (at 4000 epochs the rung above the E4 capacity passes at widths 8, 16, 32); not shown for width 64, whose 640-entry rung still fails |
| 13 | Capacity is an asymptotic property of the model, or some rung is impossible for a width | **Not established**: 1000 and 4000 epochs, constant lr 0.001, weight decay 0.1; width 64's 640-entry rung is still rising slowly at 4000 (best 636 to 638 of 640) and the rungs above E6's passes were not run |
| 14 | At 4000 epochs the first rung that failed in E4 passes at widths 8, 16, 32 (20, 160, 320 entries) and not at width 64 (640 entries) | Established by the pre-registered rule (2 of 3 book seeds, first epoch at 100%) on E4's books; 17 of 17 locked predictions hit, 8 exactly |
| 15 | A wide model keeps 100% exact recall of a book at the rung where it first reaches it | **Not true in E6**: at widths 16 and 32 the 100% is isolated (1 to 21 of 400 logged epochs for the seeds that reach it) and none of the six runs is at 100% at epoch 4000; width 8 holds it (all three runs) |

Read the status words literally: every capacity here is a statement about this budget, this learning rate, this weight decay and the exact-recall criterion, and a lower bound, not an asymptotic limit.

## 3. The design

- **What varied.** The width of the model (embedding width 8, 16, 32 and 64, always with 2 layers and 2 heads [C02]); the size of the phonebook, a ladder of rungs for each width [C03]; and the training budget, read at 100, 300, 500 and 1000 epochs from one 1000-epoch log per run in E4 [C04] and extended to 4000 epochs for one rung per width in E6 [C02].
- **What was held fixed.** Learning rate 0.001, batch size 64, AdamW with weight decay 0.1 and gradient clip 1.0, and no learning-rate schedule [C02, C28]. None of these was varied, so a capacity here is a capacity at these settings.
- **The unit of replication.** The book seed: three book seeds per rung with the training seed equal to the book seed, 96 runs in E4 [C03], and a rung is judged by at least 2 of its 3 seeds [C29]. Books are nested: the book of N entries is the first N lines of one large random book per seed.
- **The rule, fixed before the runs.** A run is perfect by a budget if every entry is recalled exactly at some logged epoch up to it (logged every 10 epochs [C02]); a rung passes if at least 2 of its 3 book seeds are perfect; the capacity is the largest rung such that it and every smaller rung pass [C29]. Consecutive rungs differ by a factor of 1.40 to 2.00, so a capacity is known only to the nearest rung [C27].
- **The measure.** Exact recall. A run counts as perfect from the first logged epoch at which every entry is recalled (first_perfect), even if the model falls back afterwards, and at widths 16 and 32 in E6 it does, because the 100% appears only on isolated logged epochs [C19].
- **Controls and checks.** A control book of 20 entries that share no name or number with training must stay at exact recall 0 [C13]. The parameter count of every run must equal the closed form for its model, and E6 must reproduce the first 1000 epochs of E4 [C24]. The runs were audited, and the pre-registered predictions were scored mechanically [C25].

## 4. Numbers

| id | statement | value | source |
|---|---|---|---|
| C01 | Information in one phone number (the model must store it; the name is the question) | 32.575 bits = log2(800 x 800 x 10,000) | [create_phonebook.py](../../create_phonebook.py) |
| C02 | Training setup of every E4 and E6 run | widths 8, 16, 32, 64, 2 layers, 2 heads, learning rate 0.001, batch size 64, logged every 10 epochs; 1000 epochs in E4 and 4000 in E6 | [specs/](../../specs/) |
| C03 | E4 design: rungs per width (widths 8, 16, 32, 64) and book seeds | rungs per width 5, 8, 9, 10; largest rung 40, 226, 453, 905; 3 book seeds (1, 2, 3); 96 runs | [e4_cells.csv](../../results/e4_cells.csv) |
| C04 | E4 capacity (entries) at 100, 300, 500, 1000 epochs, width 8 (none: no rung passed) | none, none, none, 14 | [e4_cells.csv](../../results/e4_cells.csv) |
| C05 | E4 capacity (entries) at 100, 300, 500, 1000 epochs, width 16 (none: no rung passed) | none, 20, 40, 113 | [e4_cells.csv](../../results/e4_cells.csv) |
| C06 | E4 capacity (entries) at 100, 300, 500, 1000 epochs, width 32 (none: no rung passed) | none, 160, 226, 226 | [e4_cells.csv](../../results/e4_cells.csv) |
| C07 | E4 capacity (entries) at 100, 300, 500, 1000 epochs, width 64 (none: no rung passed) | 20, 320, 320, 453 | [e4_cells.csv](../../results/e4_cells.csv) |
| C08 | Parameters at the 1000-epoch capacity rung (median over seeds), widths 8, 16, 32, 64 | 3,064 / 9,216 / 30,720 / 110,656 | [e4_cells.csv](../../results/e4_cells.csv) |
| C09 | Bits per parameter at the 1000-epoch capacity, widths 8, 16, 32, 64 | 0.149 / 0.399 / 0.240 / 0.133 | [e4_cells.csv](../../results/e4_cells.csv) |
| C10 | E4 boundary (capacity at 500 against 1000 epochs), widths 8, 16, 32, 64 | moving / moving / stable / moving | [e4_cells.csv](../../results/e4_cells.csv) |
| C11 | First epoch at which all 10 entries are recalled, range over the 3 seeds, widths 8, 16, 32, 64 | 470 to 790 / 200 to 220 / 110 to 120 / 60 to 70 | [e4_cells.csv](../../results/e4_cells.csv) |
| C12 | Growth of capacity and of parameters from width 16 to 32 and from 32 to 64 | capacity ratios 2.00 and 2.00 (width 16 to 32, 32 to 64); parameter ratios 3.33 and 3.60 | [e4_cells.csv](../../results/e4_cells.csv) |
| C13 | Control book that was never trained on | 0 of 20 control entries recalled in all 96 E4 runs and all 12 E6 runs | [e4_cells.csv](../../results/e4_cells.csv), [e6_cells.csv](../../results/e6_cells.csv), [inputs/](../../inputs/) |
| C14 | First epoch at full recall by seed (1, 2, 3) at the 1000-epoch capacity rung, widths 8, 16, 32, 64 | 14 entries: 1000/none/730; 113 entries: 970/1000/920; 226 entries: 330/500/300; 453 entries: 840/710/860 | [e4_cells.csv](../../results/e4_cells.csv) |
| C15 | E4 entries recalled at epoch 1000 by seed (1, 2, 3) at the first rung that failed, widths 8, 16, 32, 64 | 18/18/14 of 20; 128/157/116 of 160; 316/309/314 of 320; 628/620/613 of 640 | [e4_cells.csv](../../results/e4_cells.csv) |
| C16 | E6 first logged epoch at full recall by seed (1, 2, 3), widths 8, 16, 32, 64 | 1050/1050/1730; 3270/1210/3200; 940/2960/never; never/never/never | [e6_cells.csv](../../results/e6_cells.csv), [fig_e6_curves.csv](../../results/fig_e6_curves.csv) |
| C17 | E6 pass at 4000 epochs and first passing budget (1000, 2000, 3000, 4000), widths 8, 16, 32, 64 | yes (2000) / yes (4000) / yes (3000) / no | [e6_cells.csv](../../results/e6_cells.csv) |
| C18 | E6 entries recalled at epoch 4000 by seed (1, 2, 3), widths 8, 16, 32, 64 | 20/20/20 of 20; 147/158/154 of 160; 316/311/316 of 320; 610/633/623 of 640 | [fig_e6_curves.csv](../../results/fig_e6_curves.csv) |
| C19 | E6 logged epochs at 100% by seed (1, 2, 3), widths 8, 16, 32, 64 | 283/290/219; 3/21/2; 6/1/0; 0/0/0 of 400 logged epochs | [fig_e6_curves.csv](../../results/fig_e6_curves.csv) |
| C20 | E6 width 64: best recall at any logged epoch, by seed (1, 2, 3) | 637/638/636 of 640 at logged epochs 3920/1760/1660 | [fig_e6_curves.csv](../../results/fig_e6_curves.csv) |
| C21 | E6 pooled mean accuracy over epochs 1001 to 2000 and 3001 to 4000 (gain), widths 8, 16, 32, 64 | 0.9542 to 0.9955 (+0.0413) / 0.9351 to 0.9526 (+0.0175) / 0.9636 to 0.9733 (+0.0097) / 0.9684 to 0.9754 (+0.0070) | [fig_e6_curves.csv](../../results/fig_e6_curves.csv) |
| C22 | E6 trend by the pre-registered threshold (rising if the gain is at least +0.005), widths 8, 16, 32, 64 | rising / rising / rising / rising | [fig_e6_curves.csv](../../results/fig_e6_curves.csv) |
| C23 | Bits per parameter as a lower bound after E6, widths 8, 16, 32, 64 | 0.213 / 0.566 / 0.339 / 0.133 | [e4_cells.csv](../../results/e4_cells.csv), [e6_cells.csv](../../results/e6_cells.csv) |
| C24 | E6 reproduces the first 1000 epochs of E4 | 12 of 12 runs | [e6_extension.json](../../results/e6_extension.json) |
| C25 | Pre-registered predictions, scored mechanically | E4: 23 of 23 scored lines hit (13 exact); E6: 17 of 17 scored lines hit (8 exact) | [e4_prediction_scores.json](../../results/e4_prediction_scores.json), [e6_prediction_scores.json](../../results/e6_prediction_scores.json) |
| C26 | E6 width 32 seed 3: the closest it came to 100% | 319 of 320 at logged epoch 740 | [fig_e6_curves.csv](../../results/fig_e6_curves.csv) |
| C27 | Spacing of the rung ladders | consecutive rungs differ by a factor of 1.40 to 2.00 | [e4_cells.csv](../../results/e4_cells.csv) |
| C28 | Optimizer and schedule of E4 and E6 | AdamW, weight decay 0.1, gradient clip 1.0, constant learning rate (the trainer has no schedule) | [protocol.json](../../experiments/week_2026-10-31/e4_capacity_d8_seed1_a/protocol.json), [protocol.json](../../experiments/week_2026-10-31/e6_extend_d64_seed1/protocol.json), [train.py](../../../train.py) |
| C29 | Capacity rule, quoted from the spec | fixed before the runs: a rung passes if at least 2 of its 3 book seeds are perfect by the budget; the capacity is the largest rung such that it and every smaller rung pass | [e4_capacity_d8_seed1_a.json](../../specs/e4_capacity_d8_seed1_a.json) |

## 5. The mechanism

**Time, not room.** The reading of E4 and E6 is that the limit they found is mostly training time rather than room in the weights. It is a reading with limits, stated here; width 64 is where it is not shown [C20].

- The first epoch at which the 10-entry book is fully recalled falls as the width grows: 470 to 790, 200 to 220, 110 to 120 and 60 to 70 epochs for widths 8, 16, 32 and 64 [C11].
- The boundary moves with the budget at three of the four widths, so those capacities are budget limits; it is stable only at width 32 [C10].
- E6 supports the reading where it could: at 4000 epochs the first rung that failed at 1000 epochs passes at widths 8, 16 and 32, so their capacities are at least 20, 160 and 320 entries [C15, C17].
- Width 64 is not shown, for a specific reason: its 640-entry rung did not pass in 4000 epochs, the best recall at any logged epoch was 637, 638 and 636 of 640 [C20], and the pooled mean accuracy rose from 0.9684 (epochs 1001 to 2000) to 0.9754 (epochs 3001 to 4000), a gain of +0.0070 [C21], which the pre-registered threshold calls rising [C22]. A slow rise is no evidence that 100% will be reached or that it will not [C19]. Width 64 at 640 entries is not shown to be possible and not shown to be impossible [C20].
- At widths 16 and 32 the 100% is reached only on isolated logged epochs (3, 21, 2 and 6, 1, 0 of 400 per seed [C19]), and at epoch 4000 the runs recall 147 to 158 of 160 and 311 to 316 of 320 [C18]; so "passes" there means a visit to full recall, not a held state.
- In E4 the evidence that the rung above each capacity needs more epochs was strong only for widths 8 and 16; at widths 32 and 64 accuracy had flattened just short of full recall (the E4 results in the [week README](../../experiments/week_2026-10-31/README.md)) [C04, C05, C06, C07].
- What this does not separate: E4 and E6 changed the width and the training length only, never the learning rate, the schedule or the weight decay [C28], so "time" means epochs at these settings, and a lower learning rate or weight decay could change the picture.

## 6. What was not shown

- Whether any rung is out of reach for any width. No width has been shown to have an impossible rung, and a run of finite length cannot show it; what E4 and E6 show is that a rung was or was not reached within a budget, under the settings in [C28].
- An asymptotic capacity: every capacity here is a lower bound. E6 lifted the E4 capacities at three widths [C17], and the rung above each of E6's passes was not run.
- That the models hold the full recall they reach: at widths 16 and 32 the 100% is isolated and none of the six runs ends at it [C18, C19].
- The effect of the learning rate, the schedule or the weight decay: none was varied [C28].
- The effect of depth or of the number of heads: only the width was reduced, with 2 layers and 2 heads throughout [C02].
- Anything beyond memorization of the training pairs: the control book was never recalled, 0 of 20 in every run [C13], so this says nothing about generalization to new entries.
- Anything about other data than random name-to-number entries.

## 7. Twelve hard questions

**Q1. What exactly is "capacity" here, and why is it a lower bound?**
Answer. The capacity at a budget is the largest rung of the width's ladder such that it and every smaller rung pass, and a rung passes when at least 2 of its 3 book seeds recalled every entry exactly at some logged epoch up to the budget [C29]. It is a lower bound because the rung above it was only observed not to pass within that budget, because rungs are spaced by a factor of 1.40 to 2.00 [C27], and because a longer run can pass more: E6 lifted three of the four capacities [C17].
Must mention. the largest passing rung with all smaller rungs passing; at least 2 of 3 seeds; recalled at some logged epoch up to the budget; the spacing of the rungs; E6 lifted three capacities [C17, C29].

**Q2. What are the capacities, and how do they scale with width?**
Answer. At 1000 epochs they are 14, 113, 226 and 453 entries at widths 8, 16, 32 and 64 [C04, C05, C06, C07], with 3,064 / 9,216 / 30,720 / 110,656 parameters [C08], which is 0.149 / 0.399 / 0.240 / 0.133 bits per parameter [C09], far below one. From width 16 to 32 and from 32 to 64 the capacity doubles (ratios 2.00 and 2.00) while the parameter count grows by 3.33 and 3.60 [C12]. A capacity is known only to the nearest rung, and neighbouring rungs differ by a factor of 1.40 to 2.00 [C27].
Must mention. 14, 113, 226 and 453 at widths 8, 16, 32 and 64; bits per parameter 0.133 to 0.399, far below one; capacity doubles when the width doubles while parameters grow faster; the resolution is one rung [C04, C05, C06, C07, C09, C12].

**Q3. Why do you say the limit is training time, not room in the weights?**
Answer. Three observations. The first epoch at which the 10-entry book is fully recalled falls with width: 470 to 790, 200 to 220, 110 to 120 and 60 to 70 epochs for widths 8, 16, 32 and 64 [C11]. The boundary moves with the budget at three widths and is stable only at width 32 [C10]. And at 4000 epochs the first rung that failed at 1000 epochs passes at widths 8, 16 and 32 [C17]. All of this is at one learning rate, weight decay and schedule [C28], so "time" is shorthand for epochs at these settings; E6 does not separate longer training from the learning rate, the schedule or the weight decay.
Must mention. the first full recall of the smallest book comes sooner with width; the boundary moves with the budget at three widths; E6 passes at widths 8, 16 and 32; one learning rate, weight decay and schedule only [C10, C11, C17, C28].

**Q4. What did E6 change, and what did it leave open?**
Answer. E6 trained the first rung that failed at 1000 epochs for 4000 epochs: it passes at widths 8, 16 and 32 (first passing budgets 2000, 4000 and 3000) and not at width 64 [C17]. So E4's capacities of 14, 113 and 226 entries at those widths were budget limits [C04, C05, C06], and the capacities are now at least 20, 160 and 320 entries [C15], at least 0.213, 0.566 and 0.339 bits per parameter, still under one bit [C23]. It left open width 64, the rungs above E6's passes (not run), and everything about the learning rate, schedule and weight decay [C17, C28].
Must mention. the first failing rung trained for 4000 epochs; passes at widths 8, 16 and 32 but not 64; E4's capacities were budget limits; lower bounds; width 64 and the unvaried settings stay open [C17, C23].

**Q5. Some passes rest on isolated epochs. Is that really a pass?**
Answer. By the rule, yes: a rung passes when every entry was recalled at some logged epoch, and that happened, in 3, 21, 2 and 6, 1, 0 of 400 logged epochs per seed at widths 16 and 32 [C19]. But the models do not hold it: at epoch 4000 they recall 147 to 158 of 160 and 311 to 316 of 320 [C18], none of the six runs ends at full recall, and width 32 seed 3 never reached it (closest: 319 of 320 at logged epoch 740 [C26]). The first epochs at full recall are 3270, 1210 and 3200 at width 16 and 940, 2960 and never at width 32 [C16]. So the passes are visits, not stable states, which is why the claims table marks the claim that a wide model keeps its 100% as not true in E6 [C19].
Must mention. the rule counts the first epoch at full recall; 3, 21, 2 and 6, 1, 0 of 400 logged epochs; the recall at epoch 4000 is below full; a pass is a visit, not a held state [C18, C19].

**Q6. Is the widest model's last tested rung out of reach, and what is the evidence either way?**
Answer. It is not shown either way. No seed reached 640 of 640 in 4000 epochs: the best recall at any logged epoch was 637, 638 and 636 [C20], and at epoch 4000 it was 610, 633 and 623 [C18]. The pooled mean accuracy rose from 0.9684 to 0.9754, a gain of +0.0070, which the pre-registered threshold calls rising [C21, C22]. A slow rise is no evidence that 100% will be reached or that it will not [C19]; a finite run cannot show a rung impossible, and E6 did not vary the learning rate, the schedule or the weight decay [C28].
Must mention. no seed reached 640 of 640; best 637, 638 and 636; a slow rise, +0.0070; a finite run cannot show impossibility; learning rate, schedule and weight decay not varied [C20, C21, C28].

**Q7. What do the prediction hit counts show?**
Answer. E4: 23 of 23 scored lines hit (13 exact); E6: 17 of 17 (8 exact) [C25]. That is less than it sounds: the lines are strongly correlated (the same four capacities read at different budgets; the seeds-perfect, pass and first-budget lines of E6 all read the same 12 runs [C13]), and the calibration behind the predictions used a different book seed and different training seeds, which is why they generalized. One part of the locked E4 explanation was only half right: the rung above each boundary was said to be still climbing at the end, which holds for widths 8 and 16 and not for widths 32 and 64 [C04, C05, C06, C07].
Must mention. 23 of 23 and 17 of 17 scored lines hit; the lines are correlated; the calibration used other seeds; one explanation was half right [C25].

**Q8. Why only three book seeds, and what does a majority rule cost?**
Answer. There are three seeds per rung, 96 runs in E4 [C03], and the rule is a majority, at least 2 of 3 [C29]. A majority is coarse: a rung whose third seed fails still passes, and several capacities rest on cells at the edge of a budget. Width 8 passes at 14 entries only because one seed finished at epoch 1000, the last logged epoch, while another never did (first full recall at epochs 1000, none and 730 for seeds 1, 2, 3 [C14]). So the claims are stated for this budget, rule and criterion, and as lower bounds.
Must mention. three seeds and a 2-of-3 majority; a rung can pass with a failing seed; width 8 at 14 entries rests on a seed that finished at the last logged epoch; the claims are lower bounds [C03, C14, C29].

**Q9. Why bits per parameter, and what does the value at width 16 mean [C09]?**
Answer. An entry's phone number is one of 800 x 800 x 10,000 equally likely values, so about 32.575 bits [C01]. Bits per parameter is the entries times 32.575 divided by the parameters [C01]: 0.149 / 0.399 / 0.240 / 0.133 at 1000 epochs [C09], and at least 0.213 / 0.566 / 0.339 / 0.133 after E6 [C23]. It is a yardstick, not a measured storage: the name is the question and is not counted, and the number says how far below one bit per parameter these models stayed, not that they could not go higher.
Must mention. 32.575 bits per entry from 800 x 800 x 10,000 numbers; entries times bits per entry over parameters; 0.399 at width 16; far below one; a yardstick, not a measured storage [C01, C09].

**Q10. What was checked to make sure the control and the pipeline were not fooling you?**
Answer. The control book (20 entries never trained on, sharing no name or number with training) was never recalled: 0 of 20 in all 96 E4 runs and all 12 E6 runs [C13], so the recall is memorization of the training pairs. The first 1000 epochs of every E6 run equal the E4 run [C24], so E6 continues E4's runs instead of drawing new ones. The runs were audited, and the analysis was recomputed with separate code.
Must mention. 0 of 20 control entries in all 96 and 12 runs; the first 1000 epochs of E6 equal E4's for 12 of 12 runs; audits and a separate recomputation [C13, C24].

**Q11. How would you find out whether the widest model is just slow?**
Answer. Train it for longer than 4000 epochs [C02] and separate longer training from the learning rate, the schedule and the weight decay, which E6 did not vary [C28]. Fix the pass rule and write the predictions before any run, as E4 and E6 did. The best possible answer is still "reached" or "not reached within the budget under these settings"; a finite run cannot show a rung impossible.
Must mention. more than 4000 epochs; vary the learning rate, schedule and weight decay one at a time; rule and predictions fixed first; the answer is reached or not reached within a budget [C02, C28].

**Q12. What is the one-sentence takeaway you would defend, and what would you refuse to claim?**
Answer. A 2-layer, 2-head transformer memorizes at least 20, 160 and 320 random name-to-number entries at widths 8, 16 and 32 after 4000 epochs, and at least 453 at width 64 after 1000 epochs [C15, C17, C07], all far below one bit per parameter [C23], and the limit found is mostly training time at these settings [C02]. I would refuse to claim an asymptotic capacity, an impossible rung, that these models hold full recall once they reach it [C19], or anything about other learning rates, schedules, weight decays, depths or head counts [C28].
Must mention. at least 20, 160, 320 and 453 entries; far below one bit per parameter; mostly training time at these settings; no asymptotic capacity, no impossible rung, no claim about unvaried settings [C07, C15, C23].

## 8. Glossary

- **exact recall**: an entry counts as recalled only if greedy generation from its name produces the whole phone number and the newline, with no length hint.
- **entry**: one name-to-phone-number pair; its number carries about 32.575 bits [C01].
- **epoch**: one pass over the training book; the batch size is 64, so a book larger than one batch takes several updates per epoch [C02].
- **first_perfect**: the first logged epoch at which every entry of the book is recalled exactly (logged every 10 epochs [C02]); none if it never happens.
- **budget**: the number of epochs a run is allowed; E4 reads 100, 300, 500 and 1000 epochs from one 1000-epoch log per run [C04], E6 reads 1000, 2000, 3000 and 4000 [C17].
- **rung**: one book size N on a width's ladder; books are nested, so a larger rung contains the smaller ones [C03].
- **capacity**: at a budget, the largest rung such that it and every smaller rung pass; a rung passes when at least 2 of its 3 book seeds are perfect by the budget [C29].
- **lower bound**: a capacity that is at least this much, because the rung above was not run or did not pass within the budget [C17].
- **boundary**: "stable" if the capacity at 500 epochs equals the capacity at 1000, otherwise "moving", which means the number is a budget limit and not a capacity [C10].
- **bits per parameter**: entries times the bits per entry, divided by the parameters; a yardstick of how full the model is [C09].
