# IBM AML HI-Small · Portion 1 Guide

## Open and rerun

1. Open [`Portion1_Data_Pipeline.ipynb`](Portion1_Data_Pipeline.ipynb) in Colab; `REPO_URL` is already set to the project repository.  
2. Run the eight steps in order. Keep both source files in `raw/`.  
3. Read [`DATA_CONTRACT.md`](DATA_CONTRACT.md) for columns, time boundaries, and window rules.  
4. Compare the generated outputs with [`reports/EDA_REPORT.md`](reports/EDA_REPORT.md), [`figures/`](figures/), and [`reports/verification.json`](reports/verification.json).

Run [`portion1_pipeline.py`](portion1_pipeline.py) through the notebook. Each notebook stage calls the same Python module, so the processing logic has one source of truth.

## Record sources and choices

- Download `HI-Small_Trans.csv` and `HI-Small_Patterns.txt` from [IBM Transactions for Anti Money Laundering (AML)](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml).  
- Adapt the bank/account node key and time ordering ideas in [IBM Multi-GNN's `format_kaggle_files.py`](https://github.com/IBM/Multi-GNN/blob/main/format_kaggle_files.py). Build a separate sequence pipeline; do not run the graph-model formatter directly.  
- Hold transactions after September 10, 2022 in a separate `tail` audit set. Split the main September 1–10 period into train, valid, and test to avoid validation and test sets dominated by the sparse tail.  
- Use the sender event as each transaction's single prediction target. Keep sender and receiver events in account history.  
- Fit vocabularies and numeric bins on train only. Build historical features from past events. Keep labels and typologies out of tokens and baseline features.  
- Add a 2% label subset because the 1% training subset contains only 26 laundering positives, below the plan's approximate 30-positive check.

## Apply limits

- Treat HI-Small as synthetic data; do not generalize results to real bank transactions.  
- Use source row order to break same-minute ties without claiming it reflects real event order.  
- Aggregate historical monetary amounts only within the same currency.  
- Keep laundering transactions absent from the patterns file as positive labels.  
- Choose the alert threshold on valid, then freeze it before test evaluation.

## Files Guide

### 1\. root

| Item | Purpose |
| :---- | :---- |
| [`Portion1_Data_Pipeline.ipynb`](Portion1_Data_Pipeline.ipynb) | Run the pipeline in Colab, with brief notes for each stage |
| [`portion1_pipeline.py`](portion1_pipeline.py) | Ingest, inspect, tokenize, build windows and features, sample labels, and verify |
| [`PORTION1_FILE_GUIDE.md`](PORTION1_FILE_GUIDE.md) | Setup instructions and the file guide of portion1 |
| [`DATA_CONTRACT.md`](DATA_CONTRACT.md) | Check columns, split boundaries, prediction unit, window loading, and leakage guards |
| [`.gitattributes`](.gitattributes) | Track compressed datasets and token archives with Git LFS |
| [`.gitignore`](.gitignore) | Exclude temporary runtime files |

### 2. [raw/](raw/): source data

| File | Purpose |
| :---- | :---- |
| [`HI-Small_Trans.csv.gz`](raw/HI-Small_Trans.csv.gz) | Compressed source transaction CSV |
| [`HI-Small_Patterns.txt`](raw/HI-Small_Patterns.txt) | Source laundering-pattern descriptions |

### 3. [processed/](processed/): model-ready outputs

| File | Purpose |
| :---- | :---- |
| [`transactions_clean.csv.gz`](processed/transactions_clean.csv.gz) | One row per transaction, with normalized accounts, split, label, and typology |
| [`events_meta.csv.gz`](processed/events_meta.csv.gz) | Account-event positions, account keys, times, directions, and splits |
| [`event_tokens.npz`](processed/event_tokens.npz) | Six-field int32 tokens for 10,156,690 account events |
| [`window_index.csv.gz`](processed/window_index.csv.gz) | Causal history-window indices for each target transaction |
| [`account_offsets.csv.gz`](processed/account_offsets.csv.gz) | Event range for each account |
| [`engineered_features.csv.gz`](processed/engineered_features.csv.gz) | History features for baseline models |
| [`vocab_and_bins.json`](processed/vocab_and_bins.json) | Training-only vocabularies, amount bins, and time-gap bins |
| [`label_subsets.csv.gz`](processed/label_subsets.csv.gz) | Stratified 1%, 2%, 5%, and 20% train-subset membership across seeds |
| [`label_subset_100pct.csv.gz`](processed/label_subset_100pct.csv.gz) | Complete train IDs and labels, reused for every seed |
| [`label_subset_manifest.json`](processed/label_subset_manifest.json) | Subset fractions and positive/negative counts |
| [`pipeline_state.json`](processed/pipeline_state.json) | Source-file hashes and pipeline-stage status |

### 4. [reports/](reports/): notes and checks

| File | Purpose |
| :---- | :---- |
| [`EDA_REPORT.md`](reports/EDA_REPORT.md) | Readable EDA findings and split rationale |
| [`eda_summary.json`](reports/eda_summary.json) | Machine-readable EDA statistics |
| [`verification.json`](reports/verification.json) | Automated integrity-check results |

### 5. [figures/](figures/): EDA charts

| File | Purpose |
| :---- | :---- |
| [`daily_transactions.svg`](figures/daily_transactions.svg) | Daily transaction counts |
| [`daily_laundering.svg`](figures/daily_laundering.svg) | Daily laundering-positive counts |
| [`daily_laundering_rate.svg`](figures/daily_laundering_rate.svg) | Daily laundering-positive rates |
| [`account_sequence_lengths.svg`](figures/account_sequence_lengths.svg) | Account-event sequence length distribution and recommended 32-event window |
| [`typology_counts.svg`](figures/typology_counts.svg) | Laundering typology counts |

## Validate a rerun

In Colab, choose **Runtime → Restart session and run all**. Check that all eight code steps finish, five EDA charts render, and [`reports/verification.json`](reports/verification.json) reports `status: passed`, `full_train_list_rows: 3248921`, and `full_train_list_positives: 2530`.

## Submit to GitHub

Install Git LFS and run `git lfs install` before adding files. Commit [`.gitattributes`](.gitattributes) with the code and datasets; it routes `raw/*.gz`, `processed/*.gz`, and `processed/*.npz` through Git LFS. GitHub blocks regular Git files over [100 MiB](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github). Clone the repository with Git LFS enabled to retrieve the data. In Colab, run the notebook with its preconfigured `REPO_URL`.

