# Does BERT Style Pretraining Help Detect Money Laundering When Labels Are Scarce?

Team project plan: scope, resources, tasks, evaluation and team split

## 1. Project summary

Anti money laundering (AML) teams face two problems at once: confirmed laundering labels are rare, and laundering behaviour keeps changing. A popular idea is to borrow from NLP: treat each transaction as a "token" and each account's transaction history as a "sentence", pretrain a BERT style encoder with masked modelling on unlabelled data, then use what it learns to detect laundering with few labels.

This idea already exists for card fraud and credit (TabBERT, FATA Trans, UniTTab, and industry models at Nubank, Stripe and Featurespace). However, almost all published evidence comes from card fraud, card testing and credit tasks. **There is very little published evidence for AML specifically.** That gap is what this project addresses.

We will not propose a new architecture. We will build a small TabBERT style encoder, run a careful and honest evaluation on a public AML benchmark, and report what works and what does not.

## 2. Research questions

**RQ1, label efficiency (core question).** Does self supervised pretraining improve laundering detection when only a small fraction of labels is available? We compare a gradient boosted tree baseline with and without pretrained embeddings, and a fine tuned encoder, at 1%, 5%, 20% and 100% of training labels.

**RQ2, zero shot anomaly detection.** Can the encoder's masked reconstruction error flag laundering with no labels at all? How does it compare with Isolation Forest, and which laundering typologies does it catch or miss?

**RQ3, design ablations.** Does field level masking beat whole transaction masking? Does adding the time gap between transactions (Δt) help?

### Hypotheses

* H1: Pretrained embeddings help most at 1% and 5% labels, and the gap shrinks or disappears at 100%.
* H2: Reconstruction error catches bursty or unusual amount patterns (for example fan out bursts) but misses regular, low variance patterns (for example cycles and scatter gather), which are only visible across accounts.
* H3: Field level masking and Δt both improve downstream results.

A result that rejects a hypothesis is still a valid finding and will be reported as such.

## 3. Data

**Dataset:** IBM Transactions for Anti Money Laundering (AML), from Altman et al., NeurIPS 2023 (AMLworld generator).

* Kaggle: https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml
* Six variants: HI (higher illicit ratio) and LI (lower illicit ratio), each in Small, Medium and Large. Large variants hold about 175 to 180 million transactions.
* **We use HI Small:** about 515,000 accounts, about 5.08 million transactions, about 0.1% labelled as laundering. It fits on a single Colab GPU.
* Each transaction has a timestamp, sending bank and account, receiving bank and account, amounts paid and received, currencies, payment format, and an "Is Laundering" label.
* A separate patterns file lists laundering attempts by typology: fan in, fan out, bipartite, stack, random, cycle, scatter gather and gather scatter.

**Known caveats (put these in the report):**

* Data is synthetic. Results show how methods behave on a realistic simulation, not on real bank data.
* Many laundering transactions are not tagged with any typology, so the per typology analysis covers only the tagged subset.
* The small variants cover a short time window (reportedly about 10 days). Account sequences may be short, and the time based split has limited room. Check this during EDA.
* Labels are per transaction, while our model reads sequences per account. Section 6 defines how we bridge this.

## 4. Code repositories and papers

### Repositories

* **IBM TabFormer** (official TabBERT code): https://github.com/IBM/TabFormer
  Use as a **reference to read**, not to run as is. It was tested on Python 3.7, PyTorch 1.6 and HuggingFace Transformers 3.2, and its data loader is built for the credit card dataset. Key parts to study: field quantization into vocabularies, the field level transformer feeding the sequence transformer, and the tabular masking data collator.
* **FATA Trans** (Visa Research, builds on TabBERT): https://github.com/zdy93/FATA-Trans
  Reference for the time aware position embedding (Δt) and the split between static and dynamic fields.
* **UniTTab**: https://github.com/fabriziogaruti/UniTTab
  Optional reference for handling longer sequences and mixed numerical and categorical fields.
* **IBM Multi GNN**: https://github.com/IBM/Multi-GNN
  Built for the AML dataset paper and the directed multigraph GNN paper. Use its Kaggle preprocessing script and its published results as an external reference point. Rerunning it is optional.

**Our implementation:** a lean, modern PyTorch reimplementation of a small hierarchical encoder (2 to 4 layers), informed by TabFormer and FATA Trans.

### Papers to cite

* Padhi et al., "Tabular Transformers for Modeling Multivariate Time Series", ICASSP 2021 (TabBERT). arXiv 2011.01843
* Zhang et al., "FATA Trans: Field And Time Aware Transformer for Sequential Tabular Data", CIKM 2023. arXiv 2310.13818
* Luetto et al., "One Transformer for All Time Series" (UniTTab), Machine Learning journal 2025. arXiv 2302.06375
* Altman et al., "Realistic Synthetic Financial Transactions for Anti Money Laundering Models", NeurIPS 2023. arXiv 2306.16424
* Egressy et al., "Provably Powerful Graph Neural Networks for Directed Multigraphs", AAAI 2024 (Multi GNN). arXiv 2306.11586
* Industry context (cite as self reported): Nubank nuFormer (arXiv 2507.23267), Stripe Payments Foundation Model, Featurespace TallierLTM.

## 5. Method overview

1. **Tokenize transactions.** Each transaction becomes a set of field tokens: payment format, currency, counterparty bank, amount bucket (log scale quantization), Δt bucket since the account's previous transaction, and direction (sent or received). Vocabularies are built on the training period only.
2. **Build account sequences.** For each account, order its transactions (both sent and received) by time and cut sliding windows of fixed length (choose 16, 32 or 64 after EDA on sequence lengths).
3. **Pretrain.** Train a hierarchical encoder with masked field modelling on unlabelled training period data: mask 15% of field tokens and predict them with cross entropy. The label column is never an input.
4. **Use the encoder three ways:**
   * Frozen embeddings added as features to LightGBM (RQ1).
   * Fine tuned end to end with a classification head (RQ1).
   * Reconstruction error as an unsupervised anomaly score (RQ2).

## 6. Evaluation plan

This is the most important section. The goal is results a reviewer would trust.

### 6.1 Prediction unit and leakage guards

* **Unit:** each window ends at a target transaction, and we predict whether that target transaction is laundering. This gives transaction level scores comparable to the Multi GNN literature. Account level results (an account is positive if any of its transactions is laundering) are reported as a secondary view.
* **No future context:** the window only contains the target transaction and transactions before it. The encoder is bidirectional inside the window, but never sees anything after the target.
* **Time based split:** train on the earliest 60% of the time range, validate on the next 20%, test on the final 20%. Pretraining, vocabularies, quantization bins and feature statistics use the training period only.
* **Thresholds** are chosen on validation and frozen before touching test.

### 6.2 Metrics

* **Primary:** minority class F1 (laundering class), so we can place results next to published AML benchmarks.
* **PR AUC:** threshold free and suitable for extreme imbalance.
* **Precision and recall at a fixed alert budget:** flag the top 0.5% and 1% of scored transactions, which mirrors how investigation teams work with limited capacity.
* **Per typology recall at the alert budget:** for each of the eight tagged patterns.
* **Not used as a headline metric:** ROC AUC, because it can look near perfect under heavy imbalance while saying little.

### 6.3 Baselines

* **Rules baseline:** simple thresholds (large amounts, high transaction counts in a short period).
* **LightGBM on engineered features:** amount statistics, counts and velocity over recent windows, number of distinct counterparties, fan in and fan out counts, share of each payment format, Δt statistics.
* **Isolation Forest** on the same engineered features (unsupervised comparison for RQ2).
* **External reference:** Multi GNN's published HI Small result (an aggregator lists a minority class F1 of about 68; verify against the paper before quoting). We expect our per account sequence model to score lower, because graph models see the whole network. That gap is itself a finding.

### 6.4 Experiments

**Experiment A, label efficiency (RQ1).** For each label fraction (1%, 5%, 20%, 100%), train three methods: LightGBM on features, LightGBM on features plus frozen embeddings, and the fine tuned encoder. Subsample labels stratified by class. Pretraining always uses all training period transactions without labels.

**Experiment B, zero shot (RQ2).** Score each target transaction by masking each of its fields in turn and summing the negative log likelihood. Normalize scores within account activity groups (low, medium, high transaction counts) so busy accounts are not flagged just for being busy. Compare against Isolation Forest on PR AUC, recall at budget, and per typology recall.

**Experiment C, ablations (RQ3).** Field level versus whole transaction masking; with versus without Δt tokens. Evaluate each variant through the Experiment A pipeline at 5% labels to keep compute small.

### 6.5 Statistical rigour

* Run every configuration with **3 random seeds** and report mean and standard deviation.
* Report **95% bootstrap confidence intervals** on test metrics (resample test transactions 1,000 times).
* Treat a difference as meaningful only when intervals do not overlap or a paired bootstrap test supports it.
* Log every run (config, seed, metrics, runtime) to a shared results file so all figures are reproducible.

### 6.6 Error analysis and cost

* Review 30 to 50 top scoring false positives and describe what they look like.
* Review missed laundering transactions by typology.
* Report pretraining time, fine tuning time and inference throughput on the hardware used.

### 6.7 Figures and tables for the report

* Figure 1: label efficiency curve (x axis label fraction, y axis minority class F1, one line per method, error bars from seeds).
* Figure 2: PR curves at 5% labels.
* Figure 3: per typology recall at the 1% alert budget, reconstruction score versus Isolation Forest.
* Table 1: all methods by label fraction with F1, PR AUC and precision at budget, with confidence intervals.
* Table 2: ablation results.
* Table 3: external reference results from published papers, clearly marked as not directly comparable where protocols differ.

### 6.8 What counts as success

Success is a clear, trustworthy answer to RQ1 to RQ3, whichever way it goes. Beating graph models is not a goal.

## 7. Team split: four portions

Each portion has a clear owner, clear outputs and a defined handoff. Portion 1 unblocks everyone, so its first deliverable (the data contract) is due in week 1.

### Portion 1: Data pipeline and feature engineering

Owner: ____________

* Download HI Small and the patterns file; run the Multi GNN preprocessing script as a starting point.
* EDA: transactions per account, time span, laundering rate over time, typology counts. Recommend the sequence length.
* Implement the time based split and the leakage guards in 6.1.
* Build tokenization: vocabularies, log amount buckets, Δt buckets, direction field.
* Build the sequence window generator, plus the stratified label subsets (1%, 5%, 20%, 100%) for every seed.
* Build engineered features for LightGBM and Isolation Forest.
* **Handoff:** a written data contract (file formats, field list, split dates) in week 1, then the processed datasets.

### Portion 2: Encoder model and pretraining

Owner: ____________

* Study TabFormer and FATA Trans; write the lean hierarchical encoder in PyTorch.
* Implement field level masking and the alternative whole transaction masking.
* Implement Δt handling (as a token, and optionally as a time aware position embedding).
* Pretrain on the training period; track reconstruction accuracy per field.
* Export frozen embeddings per target transaction; implement the fine tuning head.
* Implement the reconstruction score function for Experiment B.
* **Handoff:** trained checkpoints, an embedding export script, and a scoring function.

### Portion 3: Baselines and supervised experiments

Owner: ____________

* Rules baseline, LightGBM on features, Isolation Forest.
* LightGBM plus frozen embeddings, and runs of the fine tuned encoder.
* Run Experiment A across all label fractions and seeds, plus the Experiment C ablations.
* Light hyperparameter tuning on validation only.
* Collect published reference numbers (Multi GNN and the AML dataset paper) and verify them.
* **Handoff:** all runs logged to the shared results file.

### Portion 4: Evaluation, analysis and communication

Owner: ____________

* Build the shared evaluation harness early (week 1 to 2): metrics, alert budget scoring, per typology recall, bootstrap confidence intervals. Everyone uses this one module, so numbers are comparable.
* Run Experiment B analysis: reconstruction score versus Isolation Forest, per typology breakdown, activity group normalization.
* Error analysis (6.6) and all figures and tables (6.7).
* Lead the final report structure and the limitations section; draft the LinkedIn post.
* **Handoff:** the final report draft and figure set.

**Workload note:** Portion 1 is front loaded and Portion 4 is back loaded. Portion 1's owner can help Portion 4 with error analysis in the final weeks; Portion 4's owner can help Portion 1 with EDA in week 1.

## 8. Timeline (six weeks; adjust to our deadline)

* **Week 1:** data contract agreed; EDA done; repo structure, environment and evaluation harness skeleton set up; everyone reads TabBERT and the AML dataset paper.
* **Week 2:** processed datasets and features ready; baselines running; encoder code complete on a small sample.
* **Week 3:** full pretraining run; embeddings exported; first Experiment A results at 5% labels.
* **Week 4:** full Experiment A grid (all fractions, all seeds); Experiment B scoring.
* **Week 5:** ablations; error analysis; figures and tables.
* **Week 6:** report writing, review and polish; LinkedIn post.

## 9. Shared conventions

* One GitHub repository with folders: data (scripts only, no raw data committed), src, configs, notebooks, results, report.
* Fixed seeds; every run driven by a config file.
* One evaluation module (owned by Portion 4) used by everyone.
* One results log (CSV) with columns: run id, method, label fraction, seed, metrics, runtime, commit hash.
* Weekly 30 minute sync; blockers raised in the group chat as soon as they appear.

## 10. Risks and mitigations

* **Short account sequences** (short dataset time span): use shorter windows with padding; report the sequence length distribution.
* **Compute limits:** keep the encoder small; use HI Small only; cap pretraining epochs; use mixed precision.
* **Old TabFormer dependencies:** do not depend on running it; reimplement the needed parts.
* **Too few positives at 1% labels:** check positive counts per subset; if under about 30, add a 2% fraction and report the instability honestly.
* **Results show no gain from pretraining:** that is a valid, publishable finding for RQ1. Report it with the confidence intervals.

## 11. Deliverables

* Clean, documented code repository that reproduces every figure.
* Final report: introduction, related work, data, method, evaluation, results, limitations, conclusion.
* A short LinkedIn post with one headline figure (the label efficiency curve) and three honest takeaways.
