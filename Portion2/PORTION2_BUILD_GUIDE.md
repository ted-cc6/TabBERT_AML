# Portion 2 build guide: encoder and pretraining

Design for Portion 2 of the TabBERT AML project. Read this first when resuming work. Project plan: [`../bert_aml_project_plan.md`](../bert_aml_project_plan.md). Data inputs: [`../Portion1/DATA_CONTRACT.md`](../Portion1/DATA_CONTRACT.md). Known design trade-offs: [`../TRADEOFFS.md`](../TRADEOFFS.md).

## 0. Status and where to resume

| Item | State (2026-11-20) |
| --- | --- |
| Portion 1 (data) | Done and verified. Walkthrough of all six themes done; findings in `TRADEOFFS.md`. |
| This design | **Approved 2026-11-20** with the section 12 defaults recorded in the implementation plan |
| Implementation plan | [`PORTION2_IMPLEMENTATION_PLAN.md`](PORTION2_IMPLEMENTATION_PLAN.md), approved 2026-11-20 |
| Next step | Seed-42 end-to-end run done 2026-11-20 (49 tests passed on Colab). Left: notebook section 9 (other 4 checkpoints), share the Drive folder, send the hand-off message |
| Code written | Tasks 1 to 7 code and tests. Tests run on Colab (notebook cell 2b) |

## 1. Intent

Portion 2 builds a small TabBERT-style encoder, pretrains it with masked field modelling on unlabelled training-period transactions, and hands three things to the rest of the team:

1. **Frozen embeddings** per transaction, for LightGBM plus embeddings (Portion 3, Experiment A).
2. **A fine-tuning head** and script, for the fine-tuned encoder runs (Portion 3, Experiments A and C).
3. **A reconstruction score** per transaction, for zero-shot anomaly detection (Portion 4, Experiment B).

It also produces the RQ3 ablation checkpoints (masking type, with or without Δt).

**Goals**
- A model the owner can explain line by line in the report and in Q&A.
- Outputs in agreed formats that Portions 3 and 4 can use without asking.
- Every run reproducible from a config and a seed.

**Non-goals**
- No new architecture. This is a lean reimplementation informed by TabFormer and FATA-Trans.
- Beating graph models (Multi-GNN) is not a goal.
- No hyperparameter search beyond light checks on validation.

**Anti-goals (would make the work a failure even if it runs)**
- Any valid, test or tail event used in pretraining.
- Labels or typology used as encoder inputs.
- Any choice tuned on the test split.

## 2. Decisions made

| Decision | Choice | Decided by |
| --- | --- | --- |
| Working mode | The builder writes all code, including the data loader, and explains it; the owner runs it locally and on Colab and checks the results. (Changed 2026-11-20; originally the owner was to write the data loader.) | Owner |
| Pretraining seeds | 3 seeds for the main checkpoint, 1 seed per ablation (5 pretraining runs). Downstream runs use 3 seeds everywhere. State this in the report. | Owner |
| Model source | Built from scratch and pretrained on this data. No pretrained weights: Hugging Face text BERTs have an English vocabulary that means nothing for transaction tokens, and no published checkpoint exists for this dataset. | Owner approved |
| Framework | Plain PyTorch (`nn.TransformerEncoder`), not Hugging Face `transformers` | Owner approved |
| Compute | **All runs on Colab GPUs** (decided 2026-11-20): pretraining, exports, fine-tuning and scoring, through the VS Code Colab extension or the browser, on the Google AI Pro allowance (200 units a month; an L4 is available). The owner runs nothing heavy on the laptop. Small unit tests run on CPU during development only. | Owner |
| Split of downstream work | Decided 2026-11-20, following the project plan: Portion 2 delivers the 5 checkpoints, the export and scoring code, and one verified end-to-end run on `main_seed42` (embeddings, one 5% fine-tuning run, reconstruction scores). **Portion 3** runs the embedding exports for the other 4 checkpoints and the fine-tuning grid (Experiments A and C). **Portion 4** runs the reconstruction scores for `main_seed43` and `main_seed44`. Instructions for them are in [`HANDOFF.md`](HANDOFF.md). | Owner |
| Local GPU | Not used. The Intel Arc 130T (Core Ultra 5 225H) is not visible to PyTorch in WSL2. Tested again on 2026-11-20 on native Windows with PyTorch 2.15.0+xpu: the GPU is detected and the forward pass works, but the backward pass fails with `UR_RESULT_ERROR_OUT_OF_RESOURCES` even at batch 64, probably because the driver (32.0.101.6554) is too old. A driver update might fix this, but Colab GPUs are much faster for the full runs. The Windows environment `C:\Users\<user>\.venvs\tabbert-xpu` is kept for possible inference-only use. | Owner |

## 3. Approaches considered

| Area | Option | Trade-off | Outcome |
| --- | --- | --- | --- |
| Model weights | Pretrained text BERT from Hugging Face | Vocabulary and knowledge do not transfer to transaction tokens | Rejected |
| | Train from scratch on our data | This is what RQ1 tests | **Chosen** |
| Building blocks | Hugging Face `BertModel` with random init | Matches TabFormer naming, but heavy and version sensitive; the two-level structure and per-field heads still need custom code | Rejected |
| | Plain PyTorch | Fewer dependencies, easy to explain, matches the plan's "lean PyTorch reimplementation" | **Chosen** |
| Compute | Intel Arc locally | Unsupported in WSL2; slow integrated GPU | Rejected for real runs |
| | Colab in VS Code | Fast NVIDIA GPUs, reproducible for teammates; data must be fetched onto the runtime and outputs saved off it | **Chosen** |

## 4. Inputs (from Portion 1)

All under `Portion1/processed/`.

| File | Use |
| --- | --- |
| `event_tokens.npz` (key `tokens`) | 10,156,690 × 6 int32, about 240 MB in RAM. Row = `event_index`. |
| `window_index.csv.gz` | One row per transaction: `transaction_id, target_event_index, window_start_index, window_length, split, is_laundering, typology`. Window = `tokens[window_start_index : target_event_index + 1]`, left-padded with 0 to 32 rows. |
| `events_meta.csv.gz` | One row per event, same order as the token matrix, with `split` per event |
| `vocab_and_bins.json` | Vocabulary sizes and special IDs |
| `label_subsets.csv.gz`, `label_subset_100pct.csv.gz` | Labelled training subsets for fine-tuning (columns `seed, label_fraction, transaction_id, is_laundering`) |

**Token fields, in column order, with vocabulary size (including special IDs):**

| # | Field | Size |
| --- | --- | --- |
| 0 | payment_format | 10 |
| 1 | currency | 18 |
| 2 | counterparty_bank | 30,531 |
| 3 | direction | 5 (3 = sent, 4 = received) |
| 4 | amount_bucket | 68 (IDs 4 to 67) |
| 5 | time_gap_bucket | 36 (IDs 4 to 35) |

Special IDs, shared by all fields: `PAD=0`, `UNK=1`, `MASK=2`, `NO_HISTORY=3`.

**Split sizes (windows = transactions):** train 3,248,921 (2,530 positive), valid 965,524 (1,036), test 862,792 (956), tail 1,108 (655). The tail is never used for headline metrics.

## 5. Model

Two levels, as in TabBERT:

```
Input: B × 32 events × 6 field tokens (int)
  1. Field embeddings: one table per field, 32 dims each, plus a learned field-position embedding
  2. Field transformer: 1 layer over the 6 fields of each event, then flatten 6 × 32 and project to 128
  3. Add a learned sequence position embedding (positions 0 to 31)
  4. Sequence transformer: 2 layers, d_model 128, 4 heads, feed-forward 512, dropout 0.1,
     bidirectional attention with a key padding mask for PAD rows
  5. Output: one contextual vector per event (128)
  6. Pretraining heads: one per field. Each maps the event vector to 32 dims (linear, GELU, LayerNorm),
     then scores every token of that field against the field's input embedding table (weight tying)
```

- **Changed 2026-11-20 (Task 2):**
  - **Weight tying.** A plain linear head for the bank field alone would add 3.9M parameters. Reusing the input embedding table as the output layer (the BERT convention) keeps the whole model at 1,475,692 parameters, of which 981,376 are embedding tables.
  - **Logits only at masked positions.** Full bank logits for a 512-window batch would need about 2 GB.
  - **Embeddings start at std 0.02, as in BERT.** With PyTorch's default std 1, the first masked loss is about 12 instead of the uniform-guess value of about 4.15.
  - The field transformer uses 2 heads and a 128-wide feed-forward layer; the guide did not fix these.

- Size: about 1 to 2M parameters (about half is the bank embedding table).
- Go to 4 sequence layers only if 2 layers clearly underfit (training loss plateaus high and per-field accuracy stays near baseline).
- The main run (2026-11-20) did not underfit. Because Colab L4 runs are cheap (4.8 min), one larger-model check is run anyway: 4 layers, d_model 256, 4.29M parameters. It is kept only if its validation masked loss is at least 5% lower.
- **Result (2026-11-20): larger model adopted.** Validation masked loss 0.669 against 0.810 (17.4% lower), with higher accuracy on every field (bank 0.852 vs 0.811, amount 0.791 vs 0.730, payment format 0.906 vs 0.869, time gap 0.259 vs 0.244). It trains in 7.1 min on an L4. All main and ablation runs use 4 sequence layers, d_model 256, feed-forward 1024 (4.29M parameters); embeddings are therefore 512 values (target row 256 + mean 256). The 1.5M run is kept on record as `small_seed42`. Only validation data was used for this choice. Larger sizes were not tried, to avoid open-ended tuning.
- Mixed precision on GPU.
- Without-Δt variant: drop field 5 from the input and from the loss (5 fields instead of 6).

## 6. Pretraining

**Data**
- Train-split windows only (`window_index.split == "train"`). Their windows contain only train-period events, so the leakage guard holds automatically.
- Neighbouring windows from one account overlap heavily, so sample about 1M windows per epoch instead of all 3.25M.
- A fixed sample of valid windows is used only for monitoring the masked loss (no labels).

**Field-level masking (main)**
- Choose 15% of non-PAD field tokens. Of these: 80% become `MASK=2`, 10% a random ID from the same field's real vocabulary, 10% unchanged.
- Loss: cross entropy on chosen positions only, averaged over the fields.

**Whole-transaction masking (ablation 1)**
- Choose 15% of non-PAD events and mask all fields of each, with the same 80/10/10 rule per event.

**Logged per epoch**
- Training and validation masked loss.
- Exact accuracy per field on masked positions, next to a "most frequent token" baseline per field.
- Within-one-bucket accuracy for amount and time gap.
- Time gap accuracy with and without gap-0 events.
- Ignore direction accuracy at the last position (always "sent").
- After training: a cosine similarity heatmap of the amount bucket embeddings.

**Variants and runs**

| Run | Masking | Δt | Pretraining seeds |
| --- | --- | --- | --- |
| main | field-level | with | 42, 43, 44 |
| abl_whole_mask | whole-transaction | with | 42 |
| abl_no_dt | field-level | without | 42 |

Downstream seed pairing: downstream seed s uses the main checkpoint pretrained with seed s.

## 7. Uses of the trained encoder (handoffs)

### 7a. Frozen embeddings (to Portion 3)
- Run the encoder without masking on every window (all splits, including tail).
- Embedding = target-position vector (d_model) concatenated with the mean over non-PAD positions (d_model): 512 values with the adopted d_model 256 (256 with the original 128).
- File per checkpoint: `embeddings_<run_name>.npy` (float16, shape 5,078,345 × 2·d_model: 512 values and about 5.2 GB with the adopted d_model 256) plus `embeddings_<run_name>_ids.npy` (transaction_id per row). Run names carry the seed, for example `embeddings_main_seed42.npy`.
- Confirm the format with Portion 3. The fallback, target-position only, is simply the first d_model columns, so the export does not wait for this answer.

### 7b. Fine-tuning (Experiments A and C)
- Head: a small MLP on the target-position vector, giving one laundering logit.
- Train end to end on the labelled subset only (fractions 1, 2, 5, 20 and 100% from Portion 1; 3 seeds). Unlabelled training transactions are not negatives.
- Loss: class-weighted cross entropy (about 1:1,280 imbalance). Early stopping on valid PR AUC.
- Output: a score per valid and test transaction, keyed by `transaction_id`. Portion 4's evaluation module does not exist yet, so use an interim file format: `transaction_id, split, score` (CSV, one file per run) and compute PR AUC locally with `sklearn.metrics.average_precision_score`. Switch to Portion 4's module once it exists. *(Builder's call.)*
- Experiment C: fine-tune each ablation checkpoint at 5% labels with 3 downstream seeds.

### 7c. Reconstruction score (to Portion 4)
- For each valid, test and tail target: make 5 copies of the window, each with one target field masked (all fields except direction), and take the negative log likelihood of the true token.
- Output per transaction: `transaction_id`, one NLL column per field, and the sum.
- Per-field columns let Portion 4 normalise; the 30,531-way bank field would otherwise dominate the sum.
- Activity-group normalisation is done by Portion 4.

### 7d. Which checkpoints get which outputs, and who runs them

| Output | Checkpoints | Run by |
| --- | --- | --- |
| Embeddings (7a) | `main_seed42` | Portion 2 |
| | `main_seed43`, `main_seed44`, both ablations | Portion 3 (Experiment C goes through the Experiment A pipeline, which includes LightGBM plus embeddings) |
| Fine-tuning (7b) | one reference run: `main_seed42`, 5%, seed 42 | Portion 2 |
| | the grid: 4 fractions × 3 seeds on the main checkpoints, plus 2 ablations × 3 seeds at 5% (18 runs; add 2% if used) | Portion 3 |
| Reconstruction scores (7c) | `main_seed42` | Portion 2 |
| | `main_seed43`, `main_seed44` (RQ2 needs seed variation; ablations are not part of Experiment B) | Portion 4 |

Storage: about 5 × 5.2 GB of embeddings on the shared Drive folder.

## 8. Code layout and environment

```
Portion2/
  PORTION2_BUILD_GUIDE.md       this file
  portion2_encoder.py           all logic: dataset, model, masking, pretrain, export, finetune, score
  Portion2_Encoder.ipynb        Colab notebook that calls the module step by step (same pattern as Portion 1)
  configs/                      one small config per run (variant, seed, sizes, epochs)
```

- **Colab:** use the VS Code Colab extension (select kernel, then Colab, then a GPU runtime). The notebook file is local but the kernel and disk are remote, so the notebook must clone the repo and run `git lfs pull` (copy the first cell of `Portion1_Data_Pipeline.ipynb`).
- **Outputs:** checkpoints, embeddings and score files go to a shared Google Drive folder, not to Git. Only code, configs, small logs and figures are committed.
- **Local:** WSL with a CPU PyTorch venv, for unit checks on a few thousand windows. Limit PyTorch to 8 threads (`torch.set_num_threads(8)`): with all 14 logical cores, a training step under WSL is about 40 times slower (1.8 s vs 40 ms).
- Group rule: commit only when the owner asks.

## 9. Build order

| Step | Work | Done when |
| --- | --- | --- |
| 1 | Data loader (owner writes, with guidance): window slicing, left padding, padding mask, train-only sampling | The window for transaction 2120422 matches the reference below |
| 2 | Model and masking | Shapes are right; the model overfits a single small batch to near-zero loss |
| 3 | Pretraining loop and logging, on a 50K-window sample on CPU | Loss falls; per-field metrics are logged |
| 4 | Colab notebook; full main pretraining, seed 42 | Per-field accuracy clearly above the most-frequent baseline |
| 5 | Embedding export; agree the format with Portion 3 | Embedding file and id file on Drive |
| 6 | Fine-tuning head; first run at 5% labels | Valid PR AUC reported through Portion 4's module |
| 7 | Reconstruction score | Score file for valid, test and tail on Drive |
| 8 | Main seeds 43 and 44; the two ablation checkpoints | All 5 checkpoints on Drive |

**Step 1 reference window.** Transaction 2120422 (train, laundering, FAN-OUT), sender `0022155:808D89450`, `target_event_index` 1213442, `window_start_index` 1213435, `window_length` 8. These values were read directly from `event_tokens.npz` and decoded with `vocab_and_bins.json` on 2026-11-19. After padding, rows 0 to 23 are all zeros and rows 24 to 31 are:

```
row  pay cur bank dir amt gap   meaning
24    5   9  516   3  10   3    Reinvestment, GBP, self, sent, 12.67, first event (NO_HISTORY)
25    5   9  516   4  10   4    same transaction, received side, gap 0
26    6   9  516   3  18  32    ACH, GBP 521.55 to self, about 2.5 days later
27    6   3  516   4  19   4    received side, converted to USD 673.66
28    6   3  311   3  19   4    ACH USD 673.66 sent to bank 017425 (laundering)
29    6   9  516   3  23  27    ACH, GBP 5,584.44 to self, about 13 hours later
30    6   4  516   4  24   4    received side, converted to EUR 6,155.73
31    6   4  338   3  24   4    target: ACH EUR 6,155.73 sent to bank 016354 (laundering)
```

The padding mask is False for rows 0 to 23 and True for rows 24 to 31.

The plan's timeline puts steps 1 to 3 in week 2 and steps 4 to 6 in week 3.

## 10. Coordination with other portions

- **Portion 3:** embedding format (section 7a); drop `transaction_id` and `split` from features; the ACH shortcut and the `prior_events_total` shift (see `TRADEOFFS.md`).
- **Portion 4:** the evaluation module's input format for scores; the score file layout (section 7c); the account-level positive definition.
- **Group:** location of the shared Drive folder; the report must state that ablations use one pretraining seed.

## 11. Left to the builder

- Time-aware position embedding (FATA-Trans) is out of scope for now: Δt is already a token field. It is a stretch goal only if the main results are done. *(Builder's call.)*

- Learning rate, batch size and number of epochs (start with AdamW, learning rate 1e-3 with warmup, batch size 512, 1 to 3 epochs; tune on the validation masked loss only).
- Exact MLP size of the fine-tuning head.
- Logging format (CSV or JSON per run is enough).

## 12. Open questions (answer before planning)

1. **Who runs the fine-tuning grid?** *Answered 2026-11-20:* Portion 3, as the project plan says. Portion 2 provides the script (notebook section 7) and one verified reference run. See section 2 and `HANDOFF.md`.
2. **What is the deadline, and whose compute pays for the runs?** The 200 Colab units a month belong to the owner's Google AI Pro plan. With a deadline and a budget, we can decide on epoch caps (for example, subsampling negatives for the 100% fine-tuning runs) before the plan is written.
3. **Embedding format** (section 7a): confirm 256 vs 128 values, and one file vs one file per split, with Portion 3. This can wait until step 5, but must be settled before the first full export.
