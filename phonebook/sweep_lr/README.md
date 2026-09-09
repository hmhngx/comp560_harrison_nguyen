# Learning-rate sweep (phonebook Phase B adaptation)

Twelve local runs exploring how the Phase-B adaptation learning rate affects
forgetting of phonebook A.

**The exact commands that produced these runs were not logged anywhere and
could not be recovered from shell history** (checked both PowerShell and bash
history on 2026-09-09 — history stops at the initial `create_phonebook.py` /
train-eval split step, before any training was run). The protocol below is
reconstructed with high confidence, not verified from a log. Re-run at least
one point to confirm before citing these numbers in a report.

## How the protocol was reconstructed

`out_lr_0.001/metrics.csv` is decimal-identical, across all 8 metric columns,
to [`phonebook/out_phaseB/metrics.csv`](../out_phaseB/metrics.csv) (both end
`...,0.0107,2.4490,1.0000,0.4154,1.0000,0.0000,...`). `out_phaseB` is the
Phase-B adaptation command documented in
[`phonebook/README.md`](../README.md):

```bash
py -u train.py --data-dir phonebook/data_phaseB --out-dir phonebook/out_phaseB \
    --adapt-from phonebook/out_phaseA/model.pth phonebook/config/phonebook.py
```

which uses `phonebook/config/phonebook.py`'s epochs=100 and `train.py`'s
default `--lr` (1e-3). The sweep runs are almost certainly this same command
with `--lr` (and, for the 500-epoch runs below, `--epochs 500`) overridden:

```bash
py -u train.py --data-dir phonebook/data_phaseB --out-dir phonebook/sweep_lr/<name> \
    --adapt-from phonebook/out_phaseA/model.pth phonebook/config/phonebook.py \
    --lr <value> [--epochs 500]
```

## Runs

Learning rate and epoch count below are read directly from each folder's own
`metrics.csv` (row count / final-row epoch), not reconstructed — only the
*command* that produced them is inferred.

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

## Finding (single seed — not yet statistically confirmed)

Lower learning rate and fewer adaptation epochs both preserve more of
phonebook A (compare `out_lr_0.0001` at 100 epochs, 93.85% retained, against
`diag_lr_0.0001_e500` at the same learning rate but 500 epochs, 65.38%
retained). No repeated seeds have been run yet — treat this as a single-seed
pilot result, not a confirmed finding, until re-run with multiple seeds.

## Not committed

`model.pth` checkpoints are intentionally excluded (matches the repo's
existing `*.pth` gitignore rule). Each run retrains in well under a minute on
CPU (see the `Training Time (sec)` column in each `metrics.csv`), so
re-training is cheaper than storing the checkpoints.
