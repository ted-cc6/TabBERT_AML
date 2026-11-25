# Portion 2 hand-off: how Portions 3 and 4 use the encoder

Portion 2 delivers the encoder, the pretraining, export and scoring code, all 5 pretrained checkpoints with their embeddings, and one verified end-to-end run on `main_seed42`, all runnable from [`Portion2_Encoder.ipynb`](Portion2_Encoder.ipynb) on Colab. Portion 3 runs the fine-tuned encoder for Experiments A and C (project plan, Portion 3). Portion 4 uses the reconstruction scores for Experiment B. Design details: [`PORTION2_BUILD_GUIDE.md`](PORTION2_BUILD_GUIDE.md).

## 1. Who runs what

| Work | Owner | Notebook section |
| --- | --- | --- |
| Model, pretraining code, tests; `main_seed42` pretrained, its embeddings, one reference fine-tuning run (5%, seed 42) and its reconstruction scores | Portion 2 | 3 to 8 |
| Pretrain the other 4 checkpoints (about 30 minutes) | Portion 2 | 5b (`PRETRAIN_RUNS`) |
| Embeddings for those 4 checkpoints | Portion 2 | 6 (`EXPORT_RUNS`) |
| Fine-tuning grid for Experiments A and C (18 runs) | Portion 3 | 7 (`FT_RUNS`) |
| Reconstruction scores for `main_seed43` and `main_seed44` | Portion 4 | 8 (`RECON_RUNS`) |

All 5 checkpoints and their embeddings are on Drive before Portions 3 and 4 start. For Portion 3, one session can run the notebook from the top: sections 1 to 6 take a few minutes (finished work is skipped), then 7 fine-tunes.

## 2. Getting set up (once)

1. Ask the Portion 2 owner for **Editor** access to the shared Drive folder `TabBERT_AML`.
2. In Google Drive, open the shared folder and choose **Organize → Add shortcut → My Drive**. The notebook then finds everything at `/content/drive/MyDrive/TabBERT_AML/portion2_runs`.
3. Open `Portion2/Portion2_Encoder.ipynb` in Colab (File → Open notebook → GitHub → `ted-cc6/TabBERT_AML`, branch `main`) and pick an **L4 GPU** runtime. The VS Code Colab extension also works.
4. Run sections 1, 2 and 2b (tests, about 5 minutes) first in every session. Do not continue if a test fails.

Every section skips work that is already on Drive, so a disconnected session can be rerun from the top without losing anything. Sections 3 and 5b do not retrain checkpoints that exist.

## 2b. What is in the Drive folder

Git holds the code; the shared Drive folder `TabBERT_AML/portion2_runs/` holds everything the notebook produces, because the files are large and Colab runtimes are wiped after each session.

```
portion2_runs/
├── main_seed42/                    checkpoint.pt, run_config.json, metrics.csv, amount_embedding_cosine.png,
│                                   embeddings_main_seed42.npy (+ _ids.npy), recon_main_seed42.csv
├── main_seed43/, main_seed44/,     added by section 5b (checkpoint.pt, metrics.csv, ...)
│   abl_whole_mask_seed42/,         embeddings_<run>.npy added by section 6
│   abl_no_dt_seed42/               recon_<run>.csv (main seeds only) added by section 8
├── finetune/ft_<run>_<pct>pct_seed<seed>/   scores.csv, finetune_log.csv, summary.json (section 7)
├── capacity_check_seed42/          model-size check: evidence only, not used downstream
└── small_seed42/                   original 1.5M model: evidence only, not used downstream
```

If `small_seed42/` still contains a file called `embeddings_main_seed42.npy`, ignore it: it comes from the dropped 1.5M model (256 columns). The real one is in `main_seed42/` (512 columns).

## 3. Checkpoints (on Drive under `portion2_runs/<run>/checkpoint.pt`; all but `main_seed42` come from section 5b)

All use the same model: two-level TabBERT, 4 sequence layers, d_model 256, 4.29M parameters, pretrained on train-period windows only.

| Run | Masking | Δt | Pretraining seed | Used for |
| --- | --- | --- | --- | --- |
| `main_seed42`, `main_seed43`, `main_seed44` | field-level | with | 42, 43, 44 | Experiments A and B |
| `abl_whole_mask_seed42` | whole-transaction | with | 42 | Experiment C |
| `abl_no_dt_seed42` | field-level | without | 42 | Experiment C |

The ablations have one pretraining seed each; the report must say so. `small_seed42` and `capacity_check_seed42` are the model-size check (section 5 of the notebook) and are not used downstream.

## 4. Embeddings for Portion 3 (section 6)

- All 5 checkpoints are exported by Portion 2 (`EXPORT_RUNS` lists them). Each export takes about 3 minutes on an L4 plus the copy to Drive.
- Output per run, next to its checkpoint: `embeddings_<run>.npy` (float16, 5,078,345 × 512, about 5.2 GB) and `embeddings_<run>_ids.npy` (the `transaction_id` of each row).
- Columns 0 to 255 are the target transaction's vector; columns 256 to 511 are the mean over the window. For a smaller feature set, use the first 256 columns.
- Rows cover every transaction (train, valid, test, tail). **Join on `transaction_id`**, never on row order of other files.
- Open with `np.load(path, mmap_mode="r")` and take only the rows you need; the full array does not need to fit in RAM.

## 5. Portion 3: fine-tuning grid (section 7)

- In section 7, uncomment the two `FT_RUNS` lines. That is 12 main runs (label fractions 1, 5, 20 and 100%, downstream seeds 42, 43, 44) plus 6 ablation runs (5%, seeds 42, 43, 44). Add fraction 0.02 if Experiment A uses the 2% subset.
- Seed pairing: downstream seed s uses `main_seed<s>`; ablation runs use their seed-42 checkpoint with downstream seeds 42, 43 and 44.
- What a run does: adds a small MLP head on the target-row vector and trains encoder and head on that labelled subset only. Class-weighted loss, every positive plus at most 200,000 negatives per epoch, at most 10 epochs, early stopping after 3 epochs without a gain in validation PR AUC.
- Output in `portion2_runs/finetune/ft_<run>_<pct>pct_seed<seed>/`:
  - `scores.csv`: `transaction_id, split, score` for every valid and test transaction, from the best epoch.
  - `finetune_log.csv`: train loss, validation PR AUC and seconds per epoch.
  - `summary.json`: best epoch, validation PR AUC, subset counts.
- Timing: the reference run (5%, seed 42) took about 45 seconds per epoch on an L4, most of it scoring the 965K validation windows, and stopped early after 7 epochs (about 5 minutes). Expect at most about 8 minutes per run, since the 20% and 100% runs see at most 200K negatives per epoch, so the 18-run grid is about 2 to 2.5 hours. Runs can be spread over several sessions.
- Reference result: `main_seed42`, 5% labels (127 positives), best validation PR AUC 0.428 at epoch 4 (validation positive rate 0.0011). Validation PR AUC moved a lot between epochs (0.29 to 0.43), so expect noticeable spread across seeds; this is why every cell has 3 seeds.
- Test is scored but never used for any choice; thresholds and tuning use validation only.
- PR AUC here comes from `sklearn.metrics.average_precision_score`. Switch to Portion 4's evaluation module when it exists by reading `scores.csv`.

## 6. Portion 4: reconstruction scores (section 8)

- `main_seed42` is done by Portion 2. For seed variation, add `"main_seed43", "main_seed44"` to `RECON_RUNS` (about 5 minutes each on an L4).
- First look at `main_seed42` (validation only, no labels used to build the score): mean `nll_sum` 10.19 for laundering against 3.84 for normal transactions, and PR AUC 0.0113 against a positive rate of 0.0011 (about 10 times random). Bank, payment format and amount are much more surprising for laundering; **time gap goes the other way** (1.56 for laundering against 2.26), so normalising and choosing fields matters more than the raw sum.
- Output: `portion2_runs/<run>/recon_<run>.csv` with `transaction_id, split`, then `nll_payment_format, nll_currency, nll_counterparty_bank, nll_amount_bucket, nll_time_gap_bucket` and `nll_sum`, for every valid, test and tail transaction.
- Each `nll_` value is how surprised the encoder is by the true value of that field of the target transaction when that field is hidden. Higher means more unusual. Direction is not scored (the target is always "sent").
- The bank field has 30,531 possible values and dominates `nll_sum`. Normalise per field, then by account activity group, as Experiment B describes.

## 7. Things to know before analysing

- **Twin rows.** A transfer between two accounts of the same customer appears twice in that account's window (sent and received, adjacent, gap 0), so pretraining accuracy partly reflects copying from the twin row. This does not involve labels, but say it when quoting pretraining accuracy.
- **Known data trade-offs** (row order, the ACH share of laundering, `prior_events_total`, the sender-side view): see [`../TRADEOFFS.md`](../TRADEOFFS.md). LightGBM features must exclude `transaction_id` and `split`.
- Pretraining results for `main_seed42` are in its `metrics.csv` on Drive; the model-size decision (17.4% lower validation loss for the 4.29M model, rule fixed in advance) is in notebook section 5 and the build guide.
