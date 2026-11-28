# Published reference numbers (HI-Small)

Collected for the report's external-reference table (plan section 6.3 and Figure/Table 3).
All numbers below are **transaction/edge-level minority-class F1 on IBM AML HI-Small**,
unless stated otherwise. They are **not directly comparable** with our results: the
published models use graph features and a different split protocol, and F1 depends on a
chosen threshold while our headline metric is PR AUC. Quote them with these caveats next
to the table, never as a target to beat.

## Verified values

| Method | HI-Small minority F1 (%) | Source |
| --- | ---: | --- |
| GIN | 28.70 ± 1.13 | Altman et al., NeurIPS 2023, Table 2 |
| GIN + edge updates | 47.73 ± 7.86 | Altman et al., NeurIPS 2023, Table 2 |
| PNA | 56.77 ± 2.41 | Altman et al., NeurIPS 2023, Table 2 |
| GFP + LightGBM | 62.86 ± 0.25 | Altman et al., NeurIPS 2023, Table 2 |
| GFP + XGBoost | 63.23 ± 0.17 | Altman et al., NeurIPS 2023, Table 2 |
| Multi-GIN | 64.79 ± 1.22 | Egressy et al., AAAI 2024 (as tabulated in MEGA-GNN, Table 2) |
| Multi-PNA | **68.16 ± 2.65** | Egressy et al., AAAI 2024 (as tabulated in MEGA-GNN, Table 2) |

The plan's "aggregator lists about 68" is the Multi-PNA number. It is real, but it belongs
to the **Multi-GNN paper** (Egressy et al.), not to the AML dataset paper, whose best listed
HI-Small result is 63.23% (GFP + XGBoost).

## References

- Altman, E., Blanuša, J., von Niederhäusern, M., Egressy, B., Anghel, A., and P. K.
  "Realistic Synthetic Financial Transactions for Anti-Money Laundering Models."
  NeurIPS 2023 Datasets and Benchmarks. arXiv:2306.16424.
  <https://arxiv.org/html/2306.16424> (Table 2)
- Egressy, B., Blanuša, J., von Niederhäusern, M., and Wattenhofer, R.
  "Provably Powerful Graph Neural Networks for Directed Multigraphs." AAAI 2024.
  arXiv:2306.11586. <https://arxiv.org/abs/2306.11586>
- MEGA-GNN (Bilgi and Atasu), "Multigraph Message Passing with Bi-Directional Multi-Edge
  Aggregations." arXiv:2412.00241, Table 2, which tabulates the Multi-GIN and Multi-PNA
  HI-Small results. <https://ar5iv.labs.arxiv.org/html/2412.00241>

## Why these are not comparable with Portion 3

1. **Graph features.** GFP (Graph Feature Preprocessor) and the GNNs see the whole
   transaction network: communities, subgraph patterns, multi-hop flows. Our LightGBM sees
   21 per-account aggregate features and no other accounts. This is the main reason the
   plan expects a lower score from a sequence model (TRADEOFFS, Theme 5).
2. **Split protocol.** The published numbers come from the dataset's own train/valid/test
   split (the MEGA-GNN paper reports a 64/19/17 split). Portion 1 uses a strict **time**
   split (Sep 1–6 / 7–8 / 9–10) with a separate tail. A time split is harder: laundering
   schemes change over the period and the test days are later than training. Some of the
   gap is protocol, not model quality.
3. **Metric.** Published tables quote F1 at an unspecified threshold; we choose the
   threshold on validation and report PR AUC alongside F1. F1 at a fixed operating point
   is not the same statistic as PR AUC.
4. **Class ratio.** HI-Small has about 0.1% laundering; small differences in the split's
   positive rate move F1 noticeably.
5. **Prediction unit.** The papers classify edges (transactions). We also predict
   transaction-level, from the sender's history only; the account-level view is a secondary
   analysis in Portion 4.

## How to use this in the report

- Put these in their own table, labelled "published, not directly comparable", with the
  protocol notes above.
- Do not rerun Multi-GNN. The plan explicitly treats it as an external reference point.
- If our fine-tuned encoder lands far below 68% F1, that is expected and is itself the
  finding (a per-account sequence model cannot see the network).
- Report our numbers with the split dates, positive counts, and PR AUC next to F1.
