# Portion 3 hand-off: baselines and supervised experiments

**Status (2026-11-27): code complete and tested; no Portion 3 runs executed yet.** This
machine has the repo and the Portion 1 data but neither the embeddings nor GPU access.
The next agent runs everything on the powerful PC (RTX 5080, 16 GB) or a Colab T4, using
this document. Portion 2's reference fine-tuning run already exists on the shared Drive
folder.

Read this whole file once before running anything. The run matrix is in
[`RUN_PLAN.md`](RUN_PLAN.md), the code is in this folder, and the measurement rules are in
[`../results/README.md`](../results/README.md).

---

## 0. TL;DR runbook

```bash
# --- one-time setup (section 3) ---
git clone --branch portion3 https://github.com/ted-cc6/TabBERT_AML.git
cd TabBERT_AML && git lfs pull
python -m venv .venv && source .venv/bin/activate      # Windows: .\.venv\Scripts\Activate.ps1
pip install -r Portion3/requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu128   # RTX 5080; skip on Colab

# --- verify (section 4) ---
python -m pytest Portion3/tests -q                     # expect: 25 passed
python -m Portion3.portion3_baselines plan             # expect: 60 runs, all pending

# --- baselines, no embeddings (section 5) ---
python -m Portion3.portion3_baselines rules
python -m Portion3.portion3_baselines isolation-forest
python -m Portion3.portion3_baselines lightgbm

# --- embeddings, then LightGBM + embeddings (sections 6 and 7) ---
python -m Portion3.prepare_embeddings check --source-dir <drive>/portion2_runs
python -m Portion3.prepare_embeddings slice --source-dir <drive>/portion2_runs --out-dir Portion3/outputs/embeddings_256
python -m Portion3.portion3_baselines tune --embeddings-dir Portion3/outputs/embeddings_256
python -m Portion3.portion3_baselines lightgbm-embeddings --embeddings-dir Portion3/outputs/embeddings_256

# --- fine-tuning grid on the GPU (section 8) ---
python -m Portion3.portion3_baselines reference-check --scores-root <drive>/finetune
python Portion3/run_finetune.py --checkpoint-dir <drive> --out-dir <drive>

# --- score the fine-tune runs into the log (section 9) ---
python -m Portion3.portion3_baselines score-finetune --scores-root <drive>/finetune
python -m Portion3.portion3_baselines summary
```

Stop after the fine-tuning step and run the reference check in section 12 before spending
the full 2.5 GPU hours if anything looks inconsistent.

---

## 1. What is done, what is left

| Work | Owner | Status |
| --- | --- | --- |
| Rules, Isolation Forest, LightGBM on features | Portion 3 | Code ready, not run |
| LightGBM + frozen embeddings (main and ablations) | Portion 3 | Code ready, not run |
| Fine-tuning grid, 20 remaining runs | Portion 3 | Runner ready, not run |
| Scoring every run into `results/runs.csv` | Portion 3 | Code ready |
| Published reference numbers | Portion 3 | Done: [`REFERENCE_NUMBERS.md`](REFERENCE_NUMBERS.md), verified against the papers |
| Account-history diagnostic (not a run) | Portion 3 | Done: `diagnose_account_history.py`, numbers in section 14 |
| Portion 2 checkpoints + embeddings + reference run | Portion 2 | Done |
| Reconstruction scores for seeds 43/44, Experiment B analysis | Portion 4 | Not started |

Portion 3 produces **60 new runs** (40 tabular + 20 fine-tune) and logs **61 rows** once
the reference run is scored: `results/runs.csv` should end with one row per run in
`RUN_PLAN.md`, including `ft_main_seed42_5pct_seed42`.

---

## 2. Prerequisites

**Human (only they can do these):**
- Editor access to the shared Drive folder `TabBERT_AML` (the `portion2_runs` folder).
- Getting the five embedding files onto the PC: either 26 GB of originals, or 13 GB of
  slices made by `Portion3_PrepareEmbeddings.ipynb` on Colab (section 6).
- Push rights to `ted-cc6/TabBERT_AML`.

**Agent:**
- Everything else. No interactive steps, no Colab required if the PC has CUDA.

Paths used below:

```
<drive> = the folder that contains main_seed42/, main_seed43/, main_seed44/,
          abl_whole_mask_seed42/, abl_no_dt_seed42/
          (on Colab: /content/drive/MyDrive/TabBERT_AML/portion2_runs)
```

The loaders accept both the Drive layout (`<dir>/<run>/embeddings_<run>.npy`) and a flat
directory (`<dir>/embeddings_<run>.npy`), with or without the `_256` suffix.

---

## 3. Setup (once)

```bash
git clone --branch portion3 https://github.com/ted-cc6/TabBERT_AML.git
cd TabBERT_AML
git lfs pull                      # 592 MB; Portion 1 processed data must be present
python -m venv .venv
source .venv/bin/activate         # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r Portion3/requirements.txt
```

PyTorch is only needed for fine-tuning. For the RTX 5080 (Blackwell, compute capability
12.0) install a CUDA 12.8 build:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability())"
```

Expected: `True (12, 0)`. If CUDA is unavailable, fix the install or use the Colab
notebook; do not run the grid on CPU.

---

## 4. Stage 0: verify (about 2 minutes)

```bash
python -m pytest Portion3/tests -q
```

Expected: **25 passed**. The tests use a synthetic fixture, not the real data.

```bash
python -m Portion3.portion3_baselines plan
```

Expected: 60 runs, all `pending`; log path `results/runs.csv`.

Sanity-check the data contract once:

```bash
python -c "from Portion3 import portion3_baselines as pb; from pathlib import Path; d=pb.load_data(Path('Portion1/processed')); print(d.features.shape, d.valid_positions.size, d.test_positions.size, int(d.labels[d.valid_positions].sum()), int(d.labels[d.test_positions].sum()))"
```

Expected: `(5078345, 21) 965524 862792 1036 956`. Loading takes about 30 seconds; every
command reloads the data once, which is included in the timings below.

---

## 5. Stage 1: tabular baselines, no embeddings (about 1 hour)

```bash
python -m Portion3.portion3_baselines rules
python -m Portion3.portion3_baselines isolation-forest
python -m Portion3.portion3_baselines lightgbm
```

- `rules`: train-only 99.9th-percentile thresholds on log amount, `events_1h`,
  `events_24h`, `distinct_counterparties_24h`; score = number of rules exceeded plus a
  small tie-break. Under a minute after loading.
- `isolation-forest`: 200 trees, 262,144 max samples, fit on train features only, one run
  per seed. 5 to 10 minutes each.
- `lightgbm`: 5 fractions x 3 seeds = 15 runs, early stopping on validation PR AUC.
  Roughly 5 to 15 minutes each; 20 to 40 minutes total.

Each run appends one row to `results/runs.csv` and writes
`Portion3/outputs/scores/<run_id>.csv.gz` (1,828,316 rows: 965,524 valid + 862,792 test).

Reference points from a local smoke run (200-round cap, 100% labels, seed 42): valid PR
AUC **0.0123**, test **0.0050**; rules test PR AUC **0.0018**. The engineered-feature
baseline is weak on this dataset. That is a finding to report. Section 12 explains why.

If a command is interrupted, rerun it; `results/runs.csv` is checked and finished runs are
skipped. `--force` reruns anyway; avoid it unless you mean to overwrite a row.

---

## 6. Stage 2: embeddings (30–60 minutes, mostly download)

First check the files:

```bash
python -m Portion3.prepare_embeddings check --source-dir <drive>
```

Expected for each of the five runs: `5,078,345 x 512 float16`, ids match `window_index`,
and a line `same order: ...`. `check` fails on duplicate ids, missing ids, wrong row
count, or non-finite values.

Then make the 256-column copies LightGBM uses:

```bash
python -m Portion3.prepare_embeddings slice --source-dir <drive> --out-dir Portion3/outputs/embeddings_256
```

Each output is about 2.6 GB (13 GB total). **Slow connection:** run
`Portion3_PrepareEmbeddings.ipynb` on Colab instead; it slices on Drive and you download
13 GB instead of 26 GB. Afterwards point `--embeddings-dir` at the downloaded folder.

The embedding files are 5,078,345 rows in `window_index` order; `EmbeddingStore` still
joins by `transaction_id`, never by position.

---

## 7. Stage 3: LightGBM plus embeddings (about 1 hour)

Optional but recommended, validation only (5% labels, seed 42; four trainings):

```bash
python -m Portion3.portion3_baselines tune --embeddings-dir Portion3/outputs/embeddings_256
```

It prints validation PR AUC for `num_leaves` 31 vs 63 and 256 vs 512 embedding columns,
writes `Portion3/configs/lightgbm_tuning.json`, and recommends a pair. If you accept the
recommendation, edit `Portion3/configs/lightgbm.json` (`num_leaves`,
`embedding_columns`) and commit it. Then:

```bash
python -m Portion3.portion3_baselines lightgbm-embeddings --embeddings-dir Portion3/outputs/embeddings_256
```

15 main runs plus 6 ablation runs (Experiments A and C at 5%). Memory: the 100% subset
uses all 3.25M train rows; 21 features + 256 embeddings is about 3.6 GB for the training
matrix and 5 to 7 GB peak. On a 16 GB machine close other applications. If it still runs
out of memory, set `embedding_columns` to 128 and note the change in the results file
(the tuning comparison no longer applies, so say so).

---

## 8. Stage 4: fine-tuning grid (GPU, about 2 to 2.5 hours)

```bash
python Portion3/run_finetune.py --checkpoint-dir <drive> --out-dir <drive>
```

The runner refuses to start without a passing reference check (section 12); it looks for
`Portion3/outputs/reference_check.json`. Run the check first. `--skip-reference-check`
overrides the gate, and should be used only after the team has agreed to proceed despite a
mismatch.

- 20 runs: 1/2/5/20/100% x 3 seeds on the main checkpoints, plus the two ablations at 5%
  x 3 seeds. The reference run `ft_main_seed42_5pct_seed42` is skipped automatically.
- Downstream seed `s` uses `main_seed<s>`; ablations use their seed-42 checkpoint with
  seeds 42, 43, 44.
- Each run writes `<out-dir>/finetune/ft_<run>_<pct>pct_seed<seed>/` with `scores.csv`,
  `finetune_log.csv`, `summary.json`. Finished runs are skipped, so a disconnect costs at
  most the run in progress.
- Expected: 45 to 90 seconds per epoch, 5 to 8 minutes per run. On an RTX 5080 expect the
  faster end; the 16 GB of VRAM is far more than this 4.29M-parameter model needs.
- `--max-epochs 3` gives a quick end-to-end check. `--runs <id> ...` runs a subset.
- The Colab notebook `Portion3_Finetune.ipynb` does the same through Drive. On Colab,
  copy the resulting `finetune/` folder back to the PC before section 9.

Before the full grid, or right after the first run, do the check in section 12.

---

## 9. Stage 5: score the fine-tune runs into the log

```bash
python -m Portion3.portion3_baselines score-finetune --scores-root <drive>/finetune
```

This reads every `ft_*/scores.csv`, checks the split column against `window_index`,
applies the same metrics module as every tabular run, and appends one row per run. It
also picks up the reference run. Expected per file: 1,829,424 rows (965,524 valid +
862,792 test + 1,108 tail). If a fine-tune folder is missing, it is simply absent from
the log; `plan` will still show it pending.

---

## 10. Stage 6: verify completeness and commit

```bash
python -m Portion3.portion3_baselines plan      # every run should say "done"
python -m Portion3.portion3_baselines summary
python -c "import pandas as pd; f=pd.read_csv('results/runs.csv'); print(len(f), f.run_id.duplicated().sum())"
```

Expected: **61 rows, 0 duplicated run ids**. `summary` prints mean/std/min/max of test PR
AUC by method and fraction; that is the table Portion 4 needs for the report.

Commit and push:

```bash
git add results/runs.csv Portion3/configs/lightgbm.json Portion3/configs/lightgbm_tuning.json
git commit -m "Add Portion 3 results log for Experiment A and C runs"
git push origin portion3
```

Do not commit `Portion3/outputs/` (git-ignored). Fine-tune scores stay on Drive.

---

## 11. Cut line for a short deadline

Run in this order; each stage leaves a clean, usable boundary:

1. `rules`, `isolation-forest`, `lightgbm` (the RQ1 baseline, ~1 h).
2. Fine-tuning grid (~2.5 h). This is the other half of RQ1 and cheap per run.
3. Embeddings + `lightgbm-embeddings` (~1 h plus download). Do this while the fine-tune
   grid runs or while the download finishes.
4. `tune` (~10 min).
5. Ablations (already inside steps 2 and 3; drop them last if time disappears).
6. Published reference numbers (no compute).

If you have only a few hours total: rules, Isolation Forest, LightGBM at 5/20/100% x 3
seeds, fine-tuning at 5/20/100% x 3 seeds, LightGBM + embeddings at 5% x 3 seeds, and the
reference section. Mark the skipped runs as skipped in the log (do not fabricate rows).

---

## 12. Sanity checks and one open anomaly

**Reproduce the fine-tune reference first.** Portion 2 reports validation PR AUC **0.428**
for `ft_main_seed42_5pct_seed42` (5%, seed 42) against a positive rate of 0.0011.

```bash
python -m Portion3.portion3_baselines reference-check --scores-root <drive>/finetune
```

Expected: `valid AP 0.428 +/- 0.02`, then `PASS`. The command reads the existing
`scores.csv`, applies the same labels and metrics as every other run, and writes
`Portion3/outputs/reference_check.json`. `run_finetune.py` refuses to start without a
passing marker (section 8). If the check fails, stop and report: the score file, the
labels, or the reported number disagree, and the whole RQ1 table would be invalid.

**Why this matters.** Every simple signal measured on 2026-11-27 clusters between 0.006
and 0.015 validation PR AUC:

| Signal | Valid PR AUC | x base rate |
| --- | ---: | ---: |
| 21 engineered features, LightGBM, 100% labels | 0.0123 | 11x |
| Target-token encodings | 0.0147 | 14x |
| Naive Bayes over the 32-event token window | 0.0133 | 12x |
| 10 hand-made sequence features, LightGBM | 0.0064 | 6x |
| Portion 2 reconstruction score (`nll_sum`) | 0.0113 | 11x |
| Account laundering rate from the 5% subset | 0.0127 | 12x |
| Account laundering rate from all train labels | 0.0723 | 67x |
| Reference fine-tune, 5% labels | **0.428** | **400x** |

Two readings follow. Either the fine-tune found a real sequence signal that none of these
captures, or the 0.428 is an artefact of that run or of how it was reported. A separate
diagnostic (section 14) shows laundering is heavily account-concentrated, so an encoder
that recognises account behaviour could in principle do much better than the aggregate
features. Do not "fix" either number to make the curve look better.

**Checks to run on the reference score file** (all read-only, none affect a choice):

- Row count 1,829,424 and unique transaction ids.
- Valid top-1% precision and recall. AP 0.428 requires a large share of the 1,036
  positives ranked near the very top, not merely a few percent precision in the top
  10,000 rows. The `reference-check` output prints both; compare them with the
  account-history diagnostic, whose AP 0.072 comes with top-1% precision 0.019.
- Test AP from the same file. If valid stays at 0.43 while test collapses toward 0.01,
  the ranking is tied to the validation period or to particular accounts.
- Join scores to `from_account_key` and report what share of the top 1% comes from the
  110 accounts that hold a 5% positive. A high share means account recognition, not
  generalisation.
- Re-run one seed at 5% after the grid starts (seed 43 is already in it). If 0.43 does
  not recur, check the original run's command and output.


**Other checks:**
- Each tabular `scores.csv.gz`: 1,828,316 rows.
- Each fine-tune `scores.csv`: 1,829,424 rows.
- `results/runs.csv`: 61 rows, no duplicate `run_id`, `created_at` timestamps in order.
- Positive counts per subset: 1% = 26, 2% = 51, 5% = 127, 20% = 506, 100% = 2,530. If the
  log disagrees, the fraction was loaded wrongly.

---

## 13. Troubleshooting

| Symptom | Fix |
| --- | --- |
| `FileNotFoundError: no embeddings for ...` | Pass `--embeddings-dir` as the folder that contains `<run>/` subfolders, or the flat folder with `_256` files. Run `prepare_embeddings check` first. |
| `KeyError: N transaction ids have no embedding` | Wrong embedding file for that checkpoint (the ids must cover all 5,078,345 transactions). Re-run `check`. |
| `MemoryError` or the machine swaps at 100% | Close other apps; keep `embedding_columns` at 256; if still short, 128. Do not silently drop the 100% runs: report them as not run. |
| `torch.cuda.is_available()` is False on the 5080 | Install the cu128 build (section 3) and a driver from 2025 or later. Capability should be `(12, 0)`. |
| Fine-tuning stops immediately with a CUDA attention error | Same cause as above: old torch build. Portion 2 hit this on the export path; a current cu128 wheel fixes it. |
| Colab disconnects | Rerun the fine-tune cell; finished folders are skipped. Do not delete partial folders; an interrupted run restarts from scratch. |
| `plan` still shows runs pending after a stage | Either the run failed (check the printed error) or it was run with a different `--results` path. The default is `results/runs.csv`. |
| A command is much slower than the table | Disk or antivirus scanning the 221 MB feature CSV; try copying `Portion1/processed` to a local SSD folder and pass `--processed-dir`. |
| `ValueError: fraction must be one of ...` | Only 0.01, 0.02, 0.05, 0.2, 1.0 exist. The 3% subset was never built. |

---

## 14. Decisions this code makes (for the report's method section)

- **2% is included** because the 1% subset has 26 positives, below the plan's ~30 check.
  1% is still run and reported; 2% is the primary low-label cell. Both counts are quoted.
- **Metrics**: PR AUC, minority F1 at a validation-chosen threshold, precision/recall at
  the top 0.5% and 1% of test rows, and per-typology recall at 1%. No bootstrap CIs here;
  Portion 4 owns those.
- **The metrics module is interim.** Portion 4 should either adopt
  `Portion3/portion3_metrics.py` or replace it and re-score every `scores.csv`; the score
  file format is deliberately the same for all methods.
- **Embeddings**: first 256 columns (target-transaction vector). `tune` checks 256 vs 512
  at 5%/seed 42 and freezes the choice. Seeds are paired: downstream seed `s` uses
  `main_seed<s>`.
- **Fine-tuning** uses Portion 2's code unchanged: class-weighted BCE, every positive plus
  at most 200,000 negatives per epoch, at most 10 epochs, early stop after 3 epochs with
  no validation-PR-AUC gain. Test is scored but never used for a choice.
- **Rules** use train-only 99.9th-percentile thresholds. Row-weighted percentiles in this
  dataset are dominated by a few hub accounts (for example `events_24h` 99.9th percentile
  is about 26,000), so the rules fire rarely and the baseline is deliberately weak. Say
  this in the limitations section; it is not a bug.
- **Isolation Forest** is label-free and fraction-independent: one fit per seed on train
  features only, scored on valid/test. It appears once in the log, not once per fraction.
- **Ablations** use a single pretraining seed (Portion 2's constraint) with three
  downstream seeds. Experiment C goes through both paths: fine-tuned encoder and
  LightGBM + ablation embeddings, at 5% labels.
- **Published numbers are not comparable.** HI-Small reference results use graph features
  and a different split; see [`REFERENCE_NUMBERS.md`](REFERENCE_NUMBERS.md) before quoting
  any of them.
- **Account history is the strongest simple signal, and the baselines deliberately lack
  it.** `diagnose_account_history.py` scores each transaction by its sender account's
  laundering rate in the labelled subset: 0.0723 valid PR AUC with all train labels,
  0.0127 with the 5% subset. It is not part of Experiment A (the plan keeps labels out of
  the features), but it is the right number to quote when interpreting any encoder
  advantage: 0.072 is what account history alone buys at 100% labels, so a fine-tuned
  encoder above that is doing more than recognising known accounts. Write the numbers into
  the limitations section. The diagnostic does not write to `results/runs.csv`.

---

## 15. File map

| Path | Purpose |
| --- | --- |
| `Portion3/portion3_baselines.py` | Rules, Isolation Forest, LightGBM, embedding store, CLI |
| `Portion3/portion3_metrics.py` | Metrics, threshold protocol, results-log writer |
| `Portion3/runplan.py` | Single source of truth for all run ids |
| `Portion3/run_finetune.py` | Fine-tuning grid runner (wraps Portion 2) |
| `Portion3/prepare_embeddings.py` | Verify and slice the 26 GB embedding exports |
| `Portion3/diagnose_account_history.py` | Account-history diagnostic (not part of the log) |
| `Portion3/configs/lightgbm.json` | Frozen LightGBM / Isolation Forest settings |
| `Portion3/tests/` | 25 synthetic-fixture tests |
| `Portion3/Portion3_Baselines.ipynb` | Tabular stages, readable wrappers around the CLI |
| `Portion3/Portion3_Finetune.ipynb` | GPU grid, local or Colab |
| `Portion3/Portion3_PrepareEmbeddings.ipynb` | Colab slicing for a slow connection |
| `Portion3/RUN_PLAN.md` | All 61 runs and costs |
| `Portion3/REFERENCE_NUMBERS.md` | Verified published HI-Small numbers and protocol caveats |
| `results/runs.csv` | The shared log (append-only) |
| `Portion3/outputs/` | Score files and temporary artefacts (git-ignored) |

---

## 16. When you finish

Update section 1's table and this header: set the status to the completion date, note
which runs were skipped and why, and record any anomaly found (especially section 12).
Push `results/runs.csv`, the config, and this file. Then tell Portion 4 the log is ready
and point them at `results/README.md` for the column definitions.
