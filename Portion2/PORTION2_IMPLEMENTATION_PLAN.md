# Portion 2 implementation plan

**Goal:** Build, test and run the TabBERT-style encoder: data loader, model, masked field pretraining, and the three handoffs (embeddings, fine-tuning scores, reconstruction scores).

**Architecture:** The data loader is in its own file, which the owner writes. All the other logic goes in one module, following the Portion 1 pattern, and a Colab notebook calls it step by step. Batches are built with vectorised NumPy indexing straight from the token matrix, so no `DataLoader` workers are needed.

**Testing (changed 2026-11-20):** the suite runs on Colab as notebook cell 2b, including GPU smoke tests (`tests/test_gpu.py`) that skip without CUDA. The owner's laptop runs nothing heavy.

**Tech stack:** Python 3.12, PyTorch 2.x (CPU locally, CUDA on Colab), NumPy, pandas, scikit-learn (`average_precision_score`), pytest.

**Spec:** [`PORTION2_BUILD_GUIDE.md`](PORTION2_BUILD_GUIDE.md). Each task below names the guide section it implements.

## Defaults for the guide's section 12 (approved 2026-11-20)

1. **Portion 3 runs the fine-tuning grid** (changed 2026-11-20 to match the project plan; the first default had the owner running it). Portion 2 provides the script and one reference run. The grid is 4 fractions (1, 5, 20, 100%) × 3 seeds, plus 2 ablations × 3 seeds at 5%: 18 runs, or 21 with the 2% subset.
2. **Compute is the owner's 200 Colab units a month.** Epoch caps: pretraining 1M windows per epoch, at most 3 epochs. Fine-tuning at most 10 epochs with patience 3. Each fine-tuning epoch uses all positives in the subset plus at most 200,000 negatives, resampled every epoch. The cap only bites at 20% and 100%.
3. **Embedding format:** 256 values in one file per checkpoint (guide section 7a). Confirm with Portion 3 before the first full export (Task 5).

## Global constraints

- Pretraining reads only windows with `split == "train"`. Valid windows are used only for the masked loss, never for labels.
- Labels and typology never go into the encoder.
- Nothing is tuned on test.
- Field order: `payment_format, currency, counterparty_bank, direction, amount_bucket, time_gap_bucket`. Sizes: `10, 18, 30531, 5, 68, 36`.
- Special IDs: `PAD=0, UNK=1, MASK=2, NO_HISTORY=3`.
- Real ID range per field, used for the random 10% replacement:
  - Categorical fields: `3 .. size-1`.
  - `amount_bucket`: `4 .. 67`.
  - `time_gap_bucket`: `4 .. 35`.
  - Read these from `vocab_and_bins.json` (`first_bin_id`, `bins`). Do not hard-code them.
- Time gap 0 is bucket ID 4.
- Window length 32, left-padded.
- Mask convention used everywhere in code: `event_mask` is True for a real event and False for padding. PyTorch's `src_key_padding_mask` takes the opposite, so pass `~event_mask`.
- Every run takes a seed and a config. The same config and seed give the same metrics on CPU.
- Checkpoints, embeddings and score files go to Drive, never to Git.
- No commits unless the owner asks. No attribution lines in commit messages.

## Review focus

1. **Window length 1** (431,184 transactions, the account's first event): row 31 only, time gap `NO_HISTORY`. Masking must still be able to pick it. Tests in Task 1 and Task 2.
2. **Window length 32** (2,309,714 transactions): no padding row at all. Make sure no off-by-one drops the oldest event. Test in Task 1.
3. **Leakage through window contents:** every event inside a sampled train window must have `split == "train"` in `events_meta`. Test in Task 1.
4. **Masking never touches PAD rows**, and the random replacement never produces a special ID or an ID outside the field's range (the bank field is the risky one). Test in Task 2.
5. **Fine-tuning subset loading:** 100% comes from a different file (`label_subset_100pct.csv.gz`, no `seed` column) than the other fractions (`label_subsets.csv.gz`). A fraction typo must raise an error, not silently return 0 rows. Test in Task 6.

---

### Task 1: Data loader. Guide sections 4, 6 and step 1

**Files:**
- Create: `Portion2/portion2_data.py`
- Test: `Portion2/tests/test_data.py` 
- Setup: `uv pip install --python ~/.venvs/week3/bin/python pytest` (done)

**Interfaces (produces):**
- `load_tokens(processed_dir: Path) -> np.ndarray`: shape `(10_156_690, 6)`, dtype int32.
- `load_window_index(processed_dir: Path) -> pd.DataFrame`: all columns of `window_index.csv.gz`, row order as stored.
- `make_window(tokens, start: int, target: int, max_len: int = 32) -> tuple[np.ndarray, np.ndarray]`: returns `(window (32, 6) int64, event_mask (32,) bool)`. This is the readable single-window version.
- `build_batch(tokens, starts: np.ndarray, targets: np.ndarray, max_len: int = 32) -> tuple[np.ndarray, np.ndarray]`: returns `(B, 32, 6)` and `(B, 32)`. This is the vectorised version.
  - Build `idx = targets[:, None] - np.arange(max_len - 1, -1, -1)`.
  - Set `valid = idx >= starts[:, None]`.
  - Gather `tokens[np.where(valid, idx, 0)]`, then zero the rows where `valid` is False.
- `sample_epoch_rows(window_index, split: str, n: int | None, seed: int, epoch: int) -> np.ndarray`: positions into `window_index` for rows of that split, drawn without replacement with `np.random.default_rng([seed, epoch])`. Use `n=None` for all rows in a fixed order.
- `iterate_batches(tokens, window_index, rows: np.ndarray, batch_size: int) -> Iterator[dict]`: yields `{"tokens": LongTensor (B,32,6), "event_mask": BoolTensor (B,32), "transaction_id": LongTensor (B,)}`.

- [x] **Step 1: Write the tests.** The tests load the real `processed/` files once per module through a fixture, so each function can be tested on its own.
  - `test_load_tokens` and `test_load_window_index`: shape, dtype, columns, row count and train count.
  - `test_reference_window_2120422`: rows 0 to 23 are zeros and rows 24 to 31 equal the table in guide section 9. The mask is False for rows 0 to 23 and True for rows 24 to 31.
  - `test_length_one_window`: the first row of `window_index` (`213094`, start 1, target 1). Only row 31 is real, and `window[31, 5] == 3`.
  - `test_length_32_window`: any row with `window_length == 32`. The mask is all True and `window[0]` equals `tokens[start]`.
  - `test_build_batch_matches_make_window`: 1,000 random rows give identical arrays from both functions.
  - `test_sample_is_train_only_and_reproducible`: `sample_epoch_rows(..., "train", 10_000, 42, 0)` gives only train rows, the same result twice, a different result for epoch 1, and no duplicates.
  - `test_train_windows_contain_only_train_events`: for those 10,000 rows, every event index in `[start, target]` has `events_meta.split == "train"`. Read `events_meta` with `usecols=["split"]`.
  - `test_iterate_batches_shapes_and_ids`: the last batch can be short, and `transaction_id` follows `rows` order.
- [x] **Step 2: Run the tests to confirm they fail.** `cd Portion2 && ~/.venvs/week3/bin/python -m pytest tests/test_data.py -v`. Expected: ImportError or NameError.
- [x] **Step 3: Implement `make_window`, then `build_batch`, then the rest.** (2026-11-20: working mode changed; the builder writes all code and the owner runs and checks it.)
- [x] **Step 4: Run the tests again.** Expected: all 9 pass. (9 passed on 2026-11-20; building batches takes about 1 s per 1M windows on CPU.)
- [ ] **Step 5: Commit, only if the owner asks.**

### Task 2: Model and masking. Guide sections 5 and 6

**Files:**
- Create: `Portion2/portion2_encoder.py`
- Test: `Portion2/tests/test_model.py`

**Interfaces:**
- Consumes: `build_batch` and `iterate_batches` from Task 1.
- Produces:
  - `FieldSpec` (dataclass: `name, size, real_low, real_high`).
  - `load_field_specs(processed_dir, use_dt: bool = True) -> list[FieldSpec]`. This returns 5 specs when `use_dt` is False.
  - `EncoderConfig` (dataclass, defaults from guide section 5):
    - `field_dim=32, field_layers=1`
    - `d_model=128, seq_layers=2, heads=4, ff=512, dropout=0.1`
    - `max_len=32, use_dt=True`
  - `TabBERT(nn.Module)`:
    - `__init__(specs, cfg)`
    - `encode(tokens (B,L,F), event_mask (B,L)) -> (B,L,128)`
    - `field_logits(hidden (..., 128), field: int) -> (..., size_f)` uses a tied head (the field's input embedding table). Call it only on selected positions.
  - `masked_field_loss(model, hidden, targets, chosen) -> (loss, list[(logits, targets) | None])` gives the mean over fields of the cross entropy on chosen positions.
  - (Changed 2026-11-20: there is no `forward` returning full logits, because full bank logits for 512 windows need about 2 GB.)
  - `select_fields(tokens, use_dt) -> tokens` drops column 5 when `use_dt` is False.
  - `mask_fields(tokens, event_mask, specs, rate=0.15, generator) -> (masked_tokens, chosen (B,L,F) bool)`.
  - `mask_events(...)` has the same signature. It chooses 15% of real events and applies one 80/10/10 draw per event to all of its fields.

- [x] **Step 1: Write the tests.**
  - `test_encode_shape`: `(4,32,6)` in gives `(4,32,128)` out. With `use_dt=False`, the model takes `(4,32,5)`.
  - `test_param_count`: between 1M and 2.5M parameters.
  - `test_padding_does_not_change_real_outputs`: change the token IDs in PAD rows. The outputs at the real rows stay the same (`allclose`, eval mode).
  - `test_mask_fields_rates`: on a 2,048-window real batch, the share of chosen real tokens is in `[0.14, 0.16]`. Among chosen tokens, the MASK share is in `[0.77, 0.83]` and the unchanged share is in `[0.08, 0.12]`.
  - `test_mask_never_touches_pad`: `chosen & ~event_mask[..., None]` is all False.
  - `test_random_replacement_in_range`: every replaced token that is neither MASK nor the original lies in `[real_low, real_high]` for its field.
  - `test_mask_events_whole_rows`: in each chosen event, all F fields are chosen.
  - `test_length_one_window_maskable`: over 200 draws on a batch of length-1 windows, row 31 is chosen at least once.
  - `test_field_logits_shape` and `test_masking_reproducible`.
  - `test_overfit_one_batch`: 64 windows, dropout 0, AdamW 1e-3, 1,000 steps on CPU, with masking redrawn on every step. The first loss is within 0.5 of the uniform-guess value (about 4.15), and the mean of the last 10 losses is below 0.3 (it reached 0.07). The original rule, "below 0.1 times the first loss", passed only because PyTorch's default embedding scale inflated the first loss to 12.
- [x] **Step 2: Run the tests and confirm they fail.**
- [x] **Step 3: Implement.**
  - The field transformer runs over the F fields of each event. Reshape to `(B*L, F, field_dim)` with a learned field-position embedding, flatten to `F*field_dim`, then apply a linear layer to 128.
  - Add a learned sequence-position embedding, then run `nn.TransformerEncoder(batch_first=True)` with `src_key_padding_mask=~event_mask`.
  - Draw all randomness from the `generator` argument so tests are reproducible.
- [x] **Step 4: Run the tests and confirm they pass.** Walk the owner through each part of `TabBERT`. (24 passed in 60 s on 2026-11-20, with tests limited to 8 CPU threads in `tests/conftest.py`.)
- [ ] **Step 5: Commit, only if the owner asks.**

### Task 3: Pretraining loop and logging. Guide section 6 and step 3

**Files:**
- Modify: `Portion2/portion2_encoder.py`
- Create: `Portion2/configs/main_seed42.json`, `main_seed43.json`, `main_seed44.json`, `abl_whole_mask_seed42.json`, `abl_no_dt_seed42.json`, and `smoke_cpu.json` (50K windows, 1 epoch, valid sample 5K)
- Test: `Portion2/tests/test_pretrain.py`

**Interfaces:**
- Consumes: Tasks 1 and 2.
- Produces:
  - `RunConfig` (dataclass):
    - `run_name, seed, masking ("field" | "event"), use_dt`
    - `windows_per_epoch=1_000_000, epochs=3, valid_windows=50_000`
    - `batch_size=512, lr=1e-3, warmup_steps=1000, weight_decay=0.01, amp=True`
  - `load_run_config(path) -> RunConfig`.
  - `FieldMetrics(specs, baselines)` with `update(outputs, chosen)` and `result() -> dict[str, float]`. It accumulates over batches; `outputs` comes from `masked_field_loss`. (Changed 2026-11-20 from a single `field_metrics` call, so that metrics add up across validation batches.) Keys per field: `acc_<field>`, `baseline_<field>`, `n_<field>`. It also returns:
    - `acc1_amount_bucket` and `acc1_time_gap_bucket` (within one bucket).
    - `acc_time_gap_bucket_nonzero` (true token is not 4).
    - Direction accuracy that excludes position 31.
  - `fit_baselines(tokens, event_split, specs) -> dict[str, int]` gives the most frequent non-PAD token per field over train events (`event_split` comes from `events_meta.csv.gz`).
  - `pick_device()` tries CUDA, then Intel XPU, then CPU (capped at 8 threads). `lr_factor(step, warmup_steps, total_steps)` gives warmup, then cosine decay.
  - Masking always runs on CPU with a seeded generator, and the masked batch is then moved to the device, so masks are identical on every device.
  - `pretrain(cfg, processed_dir, out_dir) -> Path`. It writes `out_dir/<run>/checkpoint.pt` (holding `state_dict`, `EncoderConfig`, `RunConfig`, `specs`), `metrics.csv` (one row per epoch with train and valid loss and all metric keys) and `amount_embedding_cosine.png`. It returns the checkpoint path.
  - `load_encoder(ckpt_path, device) -> (TabBERT, specs, RunConfig)`.

- [x] **Step 1: Write the tests.** (Also added: `test_fit_baselines`, `test_lr_schedule`, `test_configs_load`, and a check that both ablation variants run.)
  - `test_field_metrics_known_values`: hand-made logits and targets produce exact accuracies, including a within-one-bucket hit and the exclusion of direction at position 31.
  - `test_pretrain_smoke_reproducible`: run `smoke_cpu.json` shrunk to 2K windows twice into temporary folders. Both give identical `metrics.csv`, and `checkpoint.pt` reloads through `load_encoder`.
  - `test_pretrain_never_reads_non_train_rows`: patch `sample_epoch_rows` to record its `split` argument. Only `"train"` is used for training batches, and `"valid"` only for loss monitoring.
- [x] **Step 2: Run the tests and confirm they fail.**
- [x] **Step 3: Implement.**
  - Loss is the mean over fields of the cross entropy on chosen positions.
  - Use linear warmup, then cosine decay.
  - Use `torch.autocast` only on CUDA.
  - Keep the valid sample fixed with `sample_epoch_rows(..., "valid", valid_windows, seed, 0)`.
- [x] **Step 4: Run the tests and confirm they pass. Then run the full 50K `smoke_cpu.json` locally.** Expected: the loss falls across logged steps, and `metrics.csv` has every key. (2026-11-20: 8 tests pass. Smoke run on WSL CPU: loss 3.84 to 3.06 in 98 steps, about 0.6 s per 512-window step; currency 0.745 against a baseline of 0.377.)
- [ ] **Step 5: Commit, only if the owner asks.**

### Task 4: Colab notebook and main pretraining, seed 42. Step 4

**Files:**
- Create: `Portion2/Portion2_Encoder.ipynb`

- [x] **Step 1: Write the notebook cells.** (2026-11-20: executed locally end to end in quick-check mode. On Colab the notebook clones `main` and pulls only the 5 Portion 1 files it needs, 183 MB.)
  1. Clone the repo and run `git lfs pull`. Copy this from the first code cell of `Portion1/Portion1_Data_Pipeline.ipynb`.
  2. `pip install` whatever is missing, then check `torch.cuda.is_available()`.
  3. Mount Drive and set `OUT_DIR` to the shared folder.
  4. `pretrain(load_run_config("configs/main_seed42.json"), ...)`.
  5. Show `metrics.csv` as a table and display the cosine heatmap.
  - Keep each later task's cells in their own section, so one notebook covers every step.
- [x] **Step 2: Owner runs cells 1 to 5 on a Colab GPU.** Note the units used per epoch, so we can budget the other 4 pretraining runs.
- [x] **Step 2 (run).** Done 2026-11-20 on a Colab L4 in 4.8 min.
- [x] **Step 3: Check the result.** Expected: `acc_<field>` is clearly above `baseline_<field>` for currency, bank, amount and time gap. If it is not, check the underfitting rule in guide section 5 before considering 4 layers. (Result: valid masked loss 0.810; accuracy against baseline: payment 0.869/0.408, currency 0.998/0.372, bank 0.811/0.025, direction 0.984/0.736, amount 0.730/0.057, time gap 0.244/0.138. No underfitting.)
- [x] **Step 4: Larger-model check (added 2026-11-20).** Adopted: 0.669 against 0.810. Retrained as `main_seed42`: valid masked loss 0.6690. Compute is cheap on the L4, so run `configs/capacity_check_seed42.json` once: 4 sequence layers, d_model 256, feed-forward 1024, 4.29M parameters (RunConfig gained an `encoder` dict of EncoderConfig overrides). Rule, fixed before the run: keep the larger model only if its final validation masked loss is at least 5% lower than 0.810. Otherwise keep the 1.5M model for every checkpoint and downstream run.

### Task 5: Embedding export. Guide section 7a and step 5

**Files:**
- Modify: `Portion2/portion2_encoder.py`, and add a notebook section
- Test: `Portion2/tests/test_export.py`

**Interfaces:**
- Consumes: `load_encoder` and `iterate_batches`.
- Produces: `export_embeddings(ckpt_path, processed_dir, out_dir, batch_size=2048, rows=None, device=None) -> tuple[Path, Path]`. Files are named `embeddings_<run_name>.npy` and `embeddings_<run_name>_ids.npy`; run names already carry the seed, for example `main_seed42`.
  - It writes `embeddings_<run>_seed<s>.npy`, float16, `(n, 256)`. Each row is the target-position vector (row 31) concatenated with the mean over real rows.
  - It also writes `embeddings_<run>_seed<s>_ids.npy` (int64 `transaction_id`) in `window_index` order.
  - It runs in eval mode, with no masking and no gradients, and writes through `np.lib.format.open_memmap` so the 2.6 GB array never sits in RAM.

- [x] **Step 1: Write the tests.** Use a 2K-window smoke checkpoint and `rows` set to the first 3,000 rows.
  - `test_export_shapes_and_dtype`: `(3000, 256)` float16, ids equal `window_index.transaction_id[:3000]`, and no NaN values.
  - `test_export_deterministic`: two exports are identical.
  - `test_export_matches_encode`: one row equals a manual `encode` of that window, within float16 tolerance.
- [x] **Step 2: Run the tests, implement, run them again until they pass.** (2026-11-20: 4 export tests, including a no-Δt checkpoint; 37 tests in total.)
- [ ] **Step 3: Agree the format with Portion 3** (in the hand-off message). The 128-value fallback is the first 128 columns of the 256-value file, so the export does not need to wait for this answer. Send them the shapes, the dtype and the id file. This settles question 3 of guide section 12.
- [x] **Step 4: Owner runs the full export for main seed 42 on Colab.** (2026-11-20, 4.29M model: 5,078,345 × 512 in 169 s on an L4.) Expected: both files are on Drive, with 5,078,345 rows.

### Task 6: Fine-tuning head and the first 5% run. Guide section 7b and step 6

**Files:**
- Modify: `Portion2/portion2_encoder.py`, and add a notebook section
- Test: `Portion2/tests/test_finetune.py`

**Interfaces:**
- Consumes: `load_encoder`, `iterate_batches` and `sample_epoch_rows`.
- Produces:
  - `load_label_subset(processed_dir, fraction: float, seed: int) -> pd.DataFrame` gives `transaction_id, is_laundering`. Valid fractions are `{0.01, 0.02, 0.05, 0.2, 1.0}`. Any other value raises `ValueError`. A fraction of 1.0 reads `label_subset_100pct.csv.gz`.
  - `finetune_epoch_sample(labels, max_negatives, seed, epoch) -> np.ndarray` gives every positive plus at most `max_negatives` negatives, shuffled.
  - `FineTuneModel(nn.Module)` wraps the encoder with an MLP head (`d_model → 64 → 1`, ReLU, dropout 0.1) on the target-position vector. The head's learning rate is 5 times the encoder's.
  - `finetune(ckpt_path, processed_dir, out_dir, fraction, seed, max_epochs=10, patience=3, max_negatives_per_epoch=200_000, lr=2e-4, batch_size=512, eval_rows=None, device=None) -> Path`.
    - It uses `BCEWithLogitsLoss(pos_weight=n_neg/n_pos)`, where the counts come from that epoch's sample.
    - It early-stops on valid PR AUC.
    - It writes to `out_dir/ft_<run>_<pct>pct_seed<seed>/`: `scores.csv` (`transaction_id, split, score`, valid and test rows only, from the best epoch), `finetune_log.csv` (one row per epoch: train loss, valid PR AUC) and `summary.json`.
    - It returns the score path. Test AP is never computed here.

- [x] **Step 1: Write the tests.** (2026-11-20, in `tests/test_finetune.py`, plus `tests/test_gpu.py`.)
  - `test_label_subset_counts`: seed 42 at 0.05 gives 127 positives and 162,320 negatives. A fraction of 1.0 gives 2,530 positives. A fraction of 0.03 raises `ValueError`.
  - `test_label_subset_train_only`: every `transaction_id` has `split == "train"` in `window_index`.
  - `test_negative_cap`: with `max_negatives_per_epoch=100`, an epoch sees every positive and exactly 100 negatives.
  - `test_finetune_smoke`: a smoke checkpoint, a tiny valid and test sample, 1 epoch. The score file has the right columns, contains only `valid` and `test` splits, and has finite scores.
- [x] **Step 2: Implement.** Tests run on Colab (cell 2b), not locally.
- [x] **Step 3: Owner runs main seed 42 at 5% on Colab.** (2026-11-20: best valid PR AUC 0.428 at epoch 4 of 7, about 45 s per epoch.) Expected: the valid PR AUC is logged and computed with `sklearn.metrics.average_precision_score`. Compare it with Portion 3's LightGBM at 5% once that exists. Switch to Portion 4's evaluation module when it lands.

### Task 7: Reconstruction score. Guide section 7c and step 7

**Files:**
- Modify: `Portion2/portion2_encoder.py`, and add a notebook section
- Test: `Portion2/tests/test_recon.py`

**Interfaces:**
- Produces: `reconstruction_scores(ckpt_path, processed_dir, out_dir, splits=("valid", "test", "tail"), batch_size=1024, rows=None, device=None) -> Path`. It runs one forward pass per scored field, with only that target-row field masked, instead of building 5 copies of each window at once.
  - For each window, it builds 5 copies, each with one target-row field set to MASK. That covers every field except direction (and except Δt for a no-Δt checkpoint).
  - It records `-log p(true token)`.
  - It writes `recon_<run>_seed<s>.csv` with columns `transaction_id, split, nll_payment_format, nll_currency, nll_counterparty_bank, nll_amount_bucket, nll_time_gap_bucket, nll_sum`.

- [x] **Step 1: Write the tests.** Use a smoke checkpoint and 500 valid rows. (2026-11-20, in `tests/test_recon.py`.)
  - `test_recon_columns_and_splits`: exact column list, `split` values within the requested splits, and NLL values that are finite and at least 0.
  - `test_recon_masks_only_target_field`: patch the model to record its inputs. In each copy, exactly one token differs from the original window, at row 31 and the expected field.
  - `test_recon_sum`: `nll_sum` equals the sum of the 5 columns.
- [x] **Step 2: Implement.** Tests run on Colab (cell 2b), not locally.
- [x] **Step 3: Owner runs it for main seed 42 on Colab.** (2026-11-20: 1,829,424 rows in 254 s; valid PR AUC of nll_sum 0.0113 against a positive rate of 0.0011.) Expected: the file is on Drive with 965,524 + 862,792 + 1,108 rows.

### Task 8: Remaining checkpoints and hand-off. Step 8 (changed 2026-11-20)

**Files:**
- Modify: notebook section 9 (`PRETRAIN_RUNS`); sections 6 to 8 take lists of runs (`EXPORT_RUNS`, `FT_RUNS`, `RECON_RUNS`)
- Create: `Portion2/HANDOFF.md` (instructions for Portions 3 and 4)

- [ ] **Step 1: Owner runs notebook section 9.** It pretrains `main_seed43`, `main_seed44`, `abl_whole_mask_seed42` and `abl_no_dt_seed42`, about 7 minutes each. Expected: 5 checkpoints on Drive, each with `metrics.csv`.
- [ ] **Step 2: Owner shares the `TabBERT_AML` Drive folder** with Portions 3 and 4 as Editors, and sends the hand-off message pointing to `HANDOFF.md`.
- [ ] **Step 3 (Portion 3): embeddings for the other 4 checkpoints and the fine-tuning grid**, from sections 6 and 7.
- [ ] **Step 4 (Portion 4): reconstruction scores for `main_seed43` and `main_seed44`**, from section 8.
