# Shared results log

`runs.csv` is the one results file the project plan (section 9) asks for: one row per
run, append-only, committed to Git. Every Portion 3 command writes to it through
`Portion3/portion3_metrics.py`; do not edit it by hand.

## Columns

| Column | Meaning |
| --- | --- |
| `run_id` | Unique id from `Portion3/runplan.py` |
| `portion` | Producing portion (3 here) |
| `method` | `rules`, `isolation_forest`, `lgbm_features`, `lgbm_embeddings`, `finetune` |
| `pretrain_run` | Checkpoint used for the encoder embeddings / fine-tuning (`main_seed<s>` or an ablation) |
| `label_fraction` | Fraction of train labels used; empty for label-free methods |
| `seed` | Downstream seed; empty for rules |
| `valid_ap` | PR AUC (average precision) of the laundering class on validation |
| `test_ap` | PR AUC on test |
| `test_f1` | Minority-class F1 on test at the threshold chosen on validation |
| `threshold` | Score threshold that maximised validation F1; frozen before test |
| `test_precision_at_0.5pct` / `test_recall_at_0.5pct` | Top 0.5% of scored test rows (4,314 rows) |
| `test_precision_at_1pct` / `test_recall_at_1pct` | Top 1% of scored test rows (8,628 rows) |
| `per_typology_recall_at_1pct` | JSON: per-typology positives, hits and recall within the 1% budget |
| `train_positives` / `train_negatives` | Labelled training rows used (fine-tune and LightGBM only) |
| `runtime_s` | Wall-clock seconds for the run |
| `git_commit` | Short commit hash at run time; `-dirty` means tracked files were modified |
| `created_at` | UTC timestamp |
| `notes` | Config details (thresholds, `best_iter`, embedding width, best epoch) |

## Protocol

- Thresholds and every other choice are made on validation. Test is scored once.
- Alert-budget metrics use the top k percent by score, `k = ceil(fraction * n)`; no
  threshold is involved.
- Per-typology recall covers only laundering test rows tagged in the pattern file; an
  empty typology means "not in the file", not a negative label.
- The 1% subset has 26 positives and is unstable; 2% (51) is the primary low-label cell.
  Report both, and quote the positive counts.
- Seed pairing: downstream seed `s` uses `main_seed<s>`. Ablation runs use their single
  seed-42 checkpoint with downstream seeds 42, 43 and 44.

## Reading it

```bash
python -m Portion3.portion3_baselines summary
```

prints mean and spread of test PR AUC by method and label fraction. A complete Portion 3
run leaves 61 rows: the 60 runs in `Portion3/RUN_PLAN.md` plus the reference fine-tune
`ft_main_seed42_5pct_seed42`.

Per-run score files (`transaction_id, split, score`) are written to
`Portion3/outputs/scores/` (git-ignored) so any run can be re-scored. Fine-tune score
files stay in `portion2_runs/finetune/` on the shared Drive.
