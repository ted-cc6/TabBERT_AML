# Portion 1 data contract

This document explains how to use the processed datasets, including file formats, field definitions, join keys, split dates, window loading, and leakage guards.

For an overview and visualizations of the source data, see the [EDA report](reports/EDA_REPORT.md).

## Fixed data and splits

- Read raw/HI-Small_Trans.csv.gz and raw/HI-Small_Patterns.txt; retain both source files.
- Set transaction_id to the zero-based data-row number in the source CSV.
- Identify each account as bank:account; create one sender event and one receiver event per transaction.
- Keep the source timestamps at minute precision; do not infer a time zone.
- Assign train to [2022-09-01 00:00, 2022-09-07 00:00), valid to [2022-09-07 00:00, 2022-09-09 00:00), test to [2022-09-09 00:00, 2022-09-11 00:00), and all remaining rows to tail.
- Count 5,078,345 transactions, 10,156,690 events, and 515,088 accounts.
- Fix the window at 32 events. Include the current target event and earlier account events only.
- Use each transaction's sender event as its single prediction target; retain the receiver event in account history.
- Break same-minute ties by transaction_id, then sent(0) / received(1). This is a stable ordering convention, not evidence of actual transaction order.

## Handoff files

| File | Content | Use |
| --- | --- | --- |
| [`processed/transactions_clean.csv.gz`](processed/transactions_clean.csv.gz) | One row per transaction | Inspect normalized account keys, split, label, and typology |
| [`processed/event_tokens.npz`](processed/event_tokens.npz) | 10,156,690 × 6 int32 values | Read field tokens by event_index; stored in compressed form |
| [`processed/events_meta.csv.gz`](processed/events_meta.csv.gz) | One row per event | Inspect event position, account, time, and direction |
| [`processed/window_index.csv.gz`](processed/window_index.csv.gz) | One row per transaction | Read the target event and causal window start |
| [`processed/engineered_features.csv.gz`](processed/engineered_features.csv.gz) | One row per transaction | Train LightGBM / Isolation Forest; excludes labels |
| [`processed/account_offsets.csv.gz`](processed/account_offsets.csv.gz) | One row per account | Locate each account's event range |
| [`processed/vocab_and_bins.json`](processed/vocab_and_bins.json) | JSON | Inspect training-period vocabularies, numeric bins, and special tokens |
| [`processed/label_subsets.csv.gz`](processed/label_subsets.csv.gz) | Subset membership rows | Find train IDs for 1%, 5%, and 20% across seeds; add 2% when needed |
| [`processed/label_subset_100pct.csv.gz`](processed/label_subset_100pct.csv.gz) | One row per train transaction | Reuse the full train ID and label list for every seed |
| [`processed/label_subset_manifest.json`](processed/label_subset_manifest.json) | JSON | Inspect subset counts and the full-train list path |

## Read a window

Load np.load("processed/event_tokens.npz")["tokens"]. Slice
tokens[window_start_index:target_event_index+1] and left-pad with PAD=0 to 32 events.
For lower memory use, extract event_tokens.npy from the archive and open it with mmap_mode="r".

## Prevent leakage

- Build category vocabularies and amount/time-gap bins from train events only.
- Build windows from the target and earlier events only; compute baseline history features from earlier account events only.
- Keep is_laundering and typology in target files; exclude both from tokens and baseline features.
- Sample label subsets within train, stratify by class, and nest 1% within 5% and 5% within 20% for each seed.
- Add a 2% subset if the 1% subset contains fewer than 30 positive examples; record this in the manifest.
- Use the same 100% train list for every seed; do not duplicate identical rows.
- Choose decision thresholds on valid before computing test metrics.

## Field definitions

- event_tokens.npz tokens column order: payment_format, currency, counterparty_bank, direction, amount_bucket, time_gap_bucket.
- Join windows, labels, and features by transaction_id. Their row orders are not guaranteed to match.
- engineered_features.csv.gz column order: transaction_id, split, current_amount_paid, current_log_amount_paid, current_payment_currency, current_payment_format, same_bank, self_transfer, previous_event_gap_min, prior_events_total, events_1h, events_24h, events_7d, sent_24h, received_24h, distinct_counterparties_24h, distinct_counterparties_7d, same_currency_amount_count_24h, same_currency_amount_sum_24h, same_currency_amount_mean_24h, amount_to_prior_mean_ratio_24h, current_format_share_24h, mean_interevent_gap_24h.
- Use 1-hour, 24-hour, and 7-day windows over events before the current target event.
- Aggregate monetary amounts only within the same currency; cross-currency history contributes counts, directions, and counterparties only.
- Match typology exactly against pattern-file rows. An empty typology means absent from that file, not a negative label.

## References

- Dataset: https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml
- Adapt the bank/account node-key and temporal-ordering ideas from IBM Multi-GNN's format_kaggle_files.py. Build a separate sequence pipeline; do not run the graph-model formatter directly.
- Keep the post-2022-09-10 tail separate from the main test metrics, consistent with the Kaggle dataset note.
