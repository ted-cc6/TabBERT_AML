# Portion 2 hand-off: how Portions 3 and 4 use the encoder

**Status (2026-11-25): Portion 2 is complete.** The encoder, all 5 pretrained checkpoints, their embeddings, one reference fine-tuning run and the reconstruction scores for `main_seed42` are on the shared Drive folder. The code, tests and notebook ([`Portion2_Encoder.ipynb`](Portion2_Encoder.ipynb)) are on `main`. Portion 3 runs the fine-tuning grid for Experiments A and C; Portion 4 runs the remaining reconstruction scores for Experiment B. Design details: [`PORTION2_BUILD_GUIDE.md`](PORTION2_BUILD_GUIDE.md).

## 1. Who runs what

| Work | Owner | Notebook section | Status |
| --- | --- | --- | --- |
| Model, pretraining code, tests | Portion 2 | 2b to 5 | Done |
| 5 pretrained checkpoints | Portion 2 | 3 and 5b | Done |
| Embeddings for all 5 checkpoints | Portion 2 | 6 | Done |
| One reference fine-tuning run (`main_seed42`, 5%, seed 42) | Portion 2 | 7 | Done |
| Reconstruction scores for `main_seed42` | Portion 2 | 8 | Done |
| LightGBM plus frozen embeddings (Experiment A) | Portion 3 | none (uses the embedding files, section 5 below) | To do |
| Fine-tuning grid for Experiments A and C (18 runs, 17 left) | Portion 3 | 7 (`FT_RUNS`) | To do |
| Reconstruction scores for `main_seed43` and `main_seed44` | Portion 4 | 8 (`RECON_RUNS`) | To do |
| Experiment B analysis (normalisation, Isolation Forest comparison, per typology) | Portion 4 | none (uses the `recon_*.csv` files, section 7 below) | To do |

## 2. Next steps

### Portion 3

1. Do the one-time setup (section 3).
2. **LightGBM plus embeddings:** load the embeddings as in section 5 and join them to the engineered features on `transaction_id`. No notebook run is needed.
3. **Fine-tuning grid:** in a Colab session, run sections 1, 2 and 2b, skip 3 to 6 (nothing left to do there), then edit and run section 7 as in section 6 below. About 2 to 2.5 hours for 17 runs; split it over several sessions if needed.
4. Check it is complete: the table printed at the end of section 7 has 18 rows (the reference run plus 17), and every folder in `portion2_runs/finetune/` has a `scores.csv`.
5. Copy the rows of that table (run, fraction, seed, best epoch, validation PR AUC, subset counts) into the shared results file (project plan, section 9).

### Portion 4

1. Do the one-time setup (section 3).
2. In a Colab session, run sections 1, 2 and 2b, skip 3 to 7, then add `"main_seed43", "main_seed44"` to `RECON_RUNS` in section 8 and run it. About 5 minutes per run.
3. Check it is complete: `portion2_runs/main_seed43/recon_main_seed43.csv` and the seed 44 file exist, each with 1,829,424 rows (valid 965,524, test 862,792, tail 1,108).
4. Analyse the three `recon_*.csv` files for Experiment B as in section 7.

## 3. Getting set up (once per person)

1. The Portion 2 owner has shared the Drive folder `TabBERT_AML` with you as Editor. In Google Drive, open **Shared with me**, right-click `TabBERT_AML` and choose **Organize → Add shortcut → My Drive**. The notebook expects everything at `/content/drive/MyDrive/TabBERT_AML/portion2_runs`.
2. Open `Portion2/Portion2_Encoder.ipynb` in Colab (File → Open notebook → GitHub → `ted-cc6/TabBERT_AML`, branch `main`) and pick an **L4 GPU** runtime. The VS Code Colab extension also works.
3. In every session, run sections 1, 2 and 2b first. Section 1 clones the repo and pulls the Portion 1 data (about 180 MB), section 2 mounts Drive (allow access when asked), and 2b runs the 49 tests (about 5.5 minutes). Do not continue if a test fails; send the output to the Portion 2 owner.

**Do not save the notebook back to GitHub from Colab** (File → Save a copy in GitHub) unless you opened it from the latest `main` in the same session. Colab overwrites the whole file on `main` with your copy, and an older copy silently deletes newer code (this has happened once). To keep your edits, use File → Save a copy in Drive instead.

Every section skips work that is already on Drive, so a disconnected session can be rerun from the top without losing finished runs. A run that was interrupted halfway starts again from the beginning.

## 4. What is in the Drive folder

Git holds the code; the shared Drive folder `TabBERT_AML/portion2_runs/` holds everything the notebook produces, because the files are large and Colab runtimes are wiped after each session.

```
portion2_runs/
├── main_seed42/, main_seed43/,     checkpoint.pt, run_config.json, metrics.csv,
│   main_seed44/,                   embeddings_<run>.npy (+ embeddings_<run>_ids.npy),
│   abl_whole_mask_seed42/,         recon_<run>.csv (main seeds only, section 8)
│   abl_no_dt_seed42/
├── finetune/ft_<run>_<pct>pct_seed<seed>/   scores.csv, finetune_log.csv, summary.json (section 7)
├── capacity_check_seed42/          model-size check: evidence only, not used downstream
└── small_seed42/                   original 1.5M model: evidence only, not used downstream
```

Do not delete or rename anything under `portion2_runs/`: the notebook finds files by these exact names, and a missing checkpoint would be retrained.

### Checkpoints

All 5 use the same model: two-level TabBERT, 4 sequence layers, d_model 256, 4.29M parameters (4.27M without Δt), pretrained on train-period windows only.

| Run | Masking | Δt | Pretraining seed | Used for |
| --- | --- | --- | --- | --- |
| `main_seed42`, `main_seed43`, `main_seed44` | field-level | with | 42, 43, 44 | Experiments A and B |
| `abl_whole_mask_seed42` | whole-transaction | with | 42 | Experiment C |
| `abl_no_dt_seed42` | field-level | without | 42 | Experiment C |

The ablations have one pretraining seed each; the report must say so. `small_seed42` and `capacity_check_seed42` are the model-size check (notebook section 5) and are not used downstream.

Pretraining results (L4, 3 epochs, about 7 minutes each; validation set; accuracy per masked field):

| Run | Valid masked loss | payment | currency | bank | direction | amount | time gap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `main_seed42` | 0.6690 | 0.907 | 1.000 | 0.853 | 0.992 | 0.792 | 0.258 |
| `main_seed43` | 0.6618 | 0.907 | 1.000 | 0.852 | 0.992 | 0.800 | 0.269 |
| `main_seed44` | 0.6623 | 0.908 | 1.000 | 0.851 | 0.992 | 0.801 | 0.271 |
| `abl_whole_mask_seed42` | 0.8056 | 0.819 | 0.999 | 0.782 | 0.959 | 0.705 | 0.308 |
| `abl_no_dt_seed42` | 0.4077 | 0.889 | 0.999 | 0.832 | 0.982 | 0.777 | n/a |
| Majority-class baseline | | 0.408 | 0.372 | 0.025 | 0.736 | 0.057 | 0.137 |

The three main seeds agree within about 1% on loss and accuracy, so pretraining is stable across seeds. **Do not compare masked loss across the ablations:** whole-transaction masking hides the whole row, which is a harder task, and the no-Δt run does not predict time gap at all (the hardest field), so its average loss is lower by construction. Experiment C compares the ablations on downstream PR AUC, not on pretraining loss. Per-epoch numbers are in each run's `metrics.csv`.

## 5. Portion 3: using the embeddings (LightGBM plus embeddings)

- One file per checkpoint, next to it: `embeddings_<run>.npy` (float16, 5,078,345 × 512, 5.2 GB) and `embeddings_<run>_ids.npy` (the `transaction_id` of each row).
- Columns 0 to 255 are the target transaction's vector; columns 256 to 511 are the mean over its window. For a smaller feature set, use the first 256 columns.
- Rows cover every transaction (train, valid, test, tail). **Join on `transaction_id`**, never on row order of other files.
- Reading scattered rows straight from Drive is slow. Copy the two files to the runtime's disk first (about 1 to 2 minutes), then open the array memory-mapped so only the rows you take are read:

```python
import shutil, numpy as np, pandas as pd

run = "main_seed42"
src = f"/content/drive/MyDrive/TabBERT_AML/portion2_runs/{run}"
for name in (f"embeddings_{run}.npy", f"embeddings_{run}_ids.npy"):
    shutil.copyfile(f"{src}/{name}", f"/content/{name}")

emb = np.load(f"/content/embeddings_{run}.npy", mmap_mode="r")
ids = pd.Index(np.load(f"/content/embeddings_{run}_ids.npy"))

def embedding_features(transaction_ids, columns=slice(0, 256)):
    """Embedding rows for these transaction ids, in the same order, as float32."""
    pos = ids.get_indexer(transaction_ids)
    assert (pos >= 0).all(), "some transaction ids have no embedding"
    return np.asarray(emb[np.sort(pos)][np.argsort(np.argsort(pos))][:, columns], dtype=np.float32)
```

- Memory: the 100% train subset is 3.25M rows, so 256 float32 columns take about 3.3 GB of RAM (6.6 GB for all 512). The smaller fractions are far below that.
- Train LightGBM on the same labelled subsets as everything else (`Portion1/processed/label_subsets.csv.gz` and `label_subset_100pct.csv.gz`, the same fraction and seed), and pair downstream seed s with `main_seed<s>`, as in the fine-tuning grid. For Experiment C, repeat with the two ablation files.
- Features must exclude `transaction_id`, `split`, `is_laundering` and `typology`.

## 6. Portion 3: fine-tuning grid (section 7)

- In section 7, uncomment the two `FT_RUNS` lines under the comment "Portion 3, Experiments A and C". They replace the reference list, so you get 12 main runs (label fractions 1, 5, 20 and 100%, downstream seeds 42, 43, 44) plus 6 ablation runs (5%, seeds 42, 43, 44). The reference run is part of the grid and is skipped because it already exists, so 17 runs are trained. If Experiment A also uses the 2% subset, add `0.02` to the fractions (3 more runs).
- Seed pairing: downstream seed s uses `main_seed<s>`; ablation runs use their seed-42 checkpoint with downstream seeds 42, 43 and 44.
- What a run does: adds a small MLP head on the target-row vector and trains encoder and head on that labelled subset only. Class-weighted loss, every positive plus at most 200,000 negatives per epoch, at most 10 epochs, early stopping after 3 epochs without a gain in validation PR AUC.
- Output in `portion2_runs/finetune/ft_<run>_<pct>pct_seed<seed>/` (for example `ft_main_seed43_20pct_seed43`):
  - `scores.csv`: `transaction_id, split, score` for every valid and test transaction, from the best epoch. This is the file to evaluate.
  - `finetune_log.csv`: train loss, validation PR AUC and seconds per epoch.
  - `summary.json`: best epoch, validation PR AUC, subset counts. Section 7 prints all of them as one table at the end.
- Timing: the reference run took about 45 seconds per epoch on an L4 (most of it scoring the 965K validation windows) and stopped after 7 epochs, about 5 minutes. Expect at most about 8 minutes per run, so the grid is about 2 to 2.5 hours. A session that disconnects loses only the run in progress.
- Reference result: `main_seed42`, 5% labels (127 positives), best validation PR AUC 0.428 at epoch 4 (validation positive rate 0.0011). Validation PR AUC moved a lot between epochs (0.29 to 0.43), so expect noticeable spread across seeds; this is why every cell has 3 seeds. Report the mean and spread over seeds, not the best seed.
- Test is scored but never used for any choice; thresholds and tuning use validation only.
- The PR AUC in the log comes from `sklearn.metrics.average_precision_score`. For the report, score every run's `scores.csv` with Portion 4's evaluation module so all methods are measured the same way.

## 7. Portion 4: reconstruction scores (section 8)

- `main_seed42` is done. For seed variation, add `"main_seed43", "main_seed44"` to `RECON_RUNS` (about 5 minutes each on an L4).
- Output: `portion2_runs/<run>/recon_<run>.csv` with `transaction_id, split`, then `nll_payment_format, nll_currency, nll_counterparty_bank, nll_amount_bucket, nll_time_gap_bucket` and `nll_sum`, for every valid, test and tail transaction. There are no train rows: the encoder was pretrained on them.
- Each `nll_` value is how surprised the encoder is by the true value of that field of the target transaction when that field is hidden. Higher means more unusual. Direction is not scored (the target is always "sent").
- Labels and typology for the analysis are in `Portion1/processed/window_index.csv.gz` (`transaction_id, split, is_laundering, typology`; an empty typology means not in the pattern file, not a negative label). Join on `transaction_id`. There is no activity group column yet; Portion 4 defines the groups.
- First look at `main_seed42` (validation only, no labels used to build the score): mean `nll_sum` 10.19 for laundering against 3.84 for normal transactions, and PR AUC 0.0113 against a positive rate of 0.0011 (about 10 times random). Bank, payment format and amount are much more surprising for laundering; **time gap goes the other way** (1.56 for laundering against 2.26), so normalising and choosing fields matters more than the raw sum.
- The bank field has 30,531 possible values and dominates `nll_sum`. Normalise per field, then by account activity group, as Experiment B describes. Choose the normalisation and fields on validation, then report test once.

## 8. If something goes wrong

- **A test fails in 2b, or a cell raises an error about an unknown key or a missing function:** Colab kept an old Python session. Use Runtime → Restart session, then run from section 1 again. If it still fails, send the output to the Portion 2 owner.
- **"No CUDA GPU" in section 2:** change the runtime type to L4 GPU and start again from section 1.
- **A file is not found under `/content/drive/MyDrive/TabBERT_AML`:** the Drive shortcut is missing or has a different name (setup step 1).
- **Section 1 says a file "is still an LFS pointer":** the Git LFS download failed; rerun section 1.
- **The session disconnected:** reconnect and run from the top; finished runs are skipped.

## 9. Things to know before analysing

- **Twin rows.** A transfer between two accounts of the same customer appears twice in that account's window (sent and received, adjacent, gap 0), so pretraining accuracy partly reflects copying from the twin row. This does not involve labels, but say it when quoting pretraining accuracy.
- **Known data trade-offs** (row order, the ACH share of laundering, `prior_events_total`, the sender-side view): see [`../TRADEOFFS.md`](../TRADEOFFS.md).
- **Model size.** The 4.29M model was chosen over the 1.5M one because its validation loss was 17.4% lower, against a rule fixed in advance (keep it only if at least 5% lower). See notebook section 5 and the build guide.
