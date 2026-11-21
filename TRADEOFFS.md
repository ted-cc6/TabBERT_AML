# Design trade-offs

Running list of design choices in the pipeline and model, what they cost, and how to check or mitigate them. Useful for the limitations section of the report and for Q&A.

## Portion 1, Theme 2: turning numbers into tokens

Context: `bucket()` in `Portion1/portion1_pipeline.py` maps amounts to 64 log-scale buckets (token IDs 4 to 67) and time gaps to 32 log-scale buckets (IDs 4 to 35), using the training-period maximum as the top edge. Each amount bucket spans about 1.54x the previous one.

### 1. The model does not know that neighbouring buckets are close

- **What:** each bucket ID gets its own embedding vector. At the start, bucket 18 is as far from bucket 19 as it is from bucket 60.
- **Cost:** the model has to learn the ordering from data, the same way BERT learns that "big" and "large" are related. Rare buckets (IDs above about 60, with only tens of events each) may never get good embeddings.
- **Alternative:** feed the number as a continuous value next to the token, or replace the bucket with a continuous numeric embedding, which is the approach UniTTab takes for numeric fields.
- **How to check:** after pretraining, plot the cosine similarity between amount bucket embeddings. A clear diagonal band means the ordering was learned.

### 1b. The training loss gives no partial credit for a near miss

- **What:** masked pretraining uses cross entropy over the 64 amount buckets. If the true bucket is 24, a guess of 25 (about 9K) and a guess of 60 (about 100B) cost the same, because the loss only looks at the probability placed on 24.
- **Cost:** the loss gives no direct push toward learning that nearby buckets mean similar amounts. The ordering has to come indirectly from context, and the reconstruction accuracy for amount will look harsh, since nearly right counts as wrong.
- **Alternatives:** soft labels (most weight on the true bucket, a little on its neighbours), or an extra regression loss on log amount.
- **How to report:** besides exact accuracy, report "within one bucket" accuracy for amount and time gap. It shows whether the model is close even when it misses.

### 2. Amounts are bucketed in their own currency

- **What:** the amount is bucketed in the transaction's native currency with no conversion. ¥1,000 (about $7) and $1,000 fall in the same bucket.
- **Cost:** the model has to learn exchange rates from how the currency token and the amount token occur together. Where it doesn't, one amount token means very different real values.
- **Mitigation:** the currency token is always in the same event, so the information is there. An option is to add a second amount field converted to USD (rates can be estimated from the self-conversion transactions in the data).
- **How to check:** count how often each bucket appears per currency. If per-field reconstruction accuracy for amount is poor in pretraining, look here first.

### 3. Information is lost inside a bucket

- **What:** values in the same bucket are identical to the model. $5,703 and $8,788 are both token 24.
- **Cost:** patterns that depend on exact amounts get blurred:
  - structuring, where amounts are kept just under a reporting threshold
  - forwarding exactly the amount received
  - In the FAN-OUT example (account `0022155:808D89450`, Sep 4), "convert and then send the exact converted amount" appears only as the same bucket repeated (24 then 24), not as an exact match.
- **Mitigation:** the engineered features for LightGBM keep exact amounts (`current_amount_paid`, `amount_to_prior_mean_ratio_24h`), so the baselines can still use this signal. Worth noting when comparing the encoder with LightGBM.

### 4. Bins are fixed on the training period

- **What:** `train_max` (amount) and `max_train_gap_minutes` (8,627 minutes) come from Sep 1 to 6 only. Larger values in valid or test are clamped into the top bucket (amount ID 67, gap ID 35).
- **Why:** this is a leakage guard. Recomputing bins on all data would let future information shape the inputs.
- **Cost:** extreme values later in time all look the same. Gap ID 35 holds about 145K events, partly because of this clamping.
- **Note:** this choice is correct and should stay. Report it as a leakage guard, not as a weakness.

### Smaller observations

- Gap ID 5 (0.3 to 0.8 minutes) is never used, because timestamps are at minute precision. Gaps are whole minutes, so 0 maps to ID 4 and 1 maps to ID 6. Harmless.
- About 1.2M events have gap 0 (ID 4), mostly sent and received pairs from self-transfers in the same minute. The model can predict these easily during masked pretraining, so the time-gap reconstruction accuracy will look better than it really is. Consider reporting gap accuracy with and without gap 0 events.

## Portion 1, Theme 3: how the time split was chosen

Context: the main period Sep 1 to 10 is split 60/20/20 by elapsed time: train Sep 1 to 6, valid Sep 7 to 8, test Sep 9 to 10. Sep 11 to 18 is held out as a separate tail (1,108 transactions, 59% laundering), because normal traffic stops after Sep 10. A literal 60/20/20 over Sep 1 to 18 would have left valid and test with only a few hundred, mostly laundering, transactions.

### 1. Base rate drifts across splits

- **What:** the laundering rate is 0.078% in train, 0.107% in valid and 0.111% in test.
- **Cost:** precision and F1 depend on the base rate, so train-period numbers are not directly comparable with test numbers.
- **Mitigation:** valid is close to test, so choosing thresholds on valid is reasonable. State the rates next to the results.

### 2. The test set is short

- **What:** two days and 956 positives. Per typology, some groups will have only a few dozen test positives.
- **Cost:** wide confidence intervals, especially for per typology recall and for the 1% label runs.
- **Mitigation:** the plan's 1,000 bootstrap resamples and three seeds. Report positive counts next to each per typology number.

### 3. Weekday mix differs between splits

- **What:** train has one weekend (Sep 3 to 4), valid has none, and test has a Saturday (Sep 10). Laundering per day is steady (about 320 to 540), but normal volume drops on weekends, so weekend laundering rates roughly double.
- **Cost:** part of any train to test difference may come from the calendar rather than the model.
- **How to check:** report test metrics for Sep 9 and Sep 10 separately as a sanity check.

### 4. The tail is set aside, not thrown away

- **What:** Sep 11 to 18 never enters the headline metrics.
- **Option:** a short appendix stress test on how each model behaves when normal traffic disappears. Keep it clearly separate from the main tables.

### 5. Accounts are not held out

- **What:** the split is by time, so the same account can appear in train and test. Account IDs are not tokens, but the model can still learn the habits of busy accounts.
- **Cost:** results describe "known accounts, future transactions", not unseen accounts. This is normal for transaction monitoring, but worth one sentence in the limitations section.

### 6. Test windows reach back into the training period

- **What:** the target transaction's date decides its split. A Sep 9 target can have history events from Sep 5 in its window.
- **Note:** this is correct. It is the account's real past, and only tokens (never labels) are used. Portion 2 must still pretrain only on train-period events and should not mask-train on valid or test windows.

## Portion 1, Theme 4: leakage watch-list

Context: Portion 1's five leakage guards hold (train-only vocabularies and bins, causal windows, past-only features, labels never used as inputs, nested train-only label subsets). Independent checks on the processed files: 20,000 random windows had 0 cross-account events and 0 future events; 0 label subset rows fall outside train; nesting holds for all three seeds. The items below are risks for the downstream portions, not defects in Portion 1.

### 1. Source row order clusters laundering

- **What:** `transaction_id` is the row number in the source CSV, and the generator writes the legs of a laundering scheme next to each other. If a row is laundering, the next or previous row is laundering 28.6% of the time, against 0.13% for a non-laundering row. Row ID also tracks time almost exactly (Spearman 0.999). Example: the FAN-OUT legs of account `0022155:808D89450` are consecutive IDs 2120419 to 2120422.
- **Risk:** any feature built from row position or neighbouring IDs (for example "was the previous row laundering" or "how many nearby rows share this account") would score very well and be pure leakage.
- **Rules:**
  - Never use `transaction_id` as a model input. It is a join key only.
  - Never build features from row adjacency.
  - Shuffling or sampling by ID is fine.
- **Related:** same-minute ties inside a window are ordered by `transaction_id`, so the order of a scheme's legs follows how the generator wrote them. Only tokens are visible, never labels, so this is not label leakage, but it is a simulator artifact. The data contract already says this order is not evidence of actual event order.

### 2. Handoff reminder for Portion 2 (pretraining)

- `processed/event_tokens.npz` holds all 10,156,690 events, including valid, test and tail.
- Pretraining must use train-period events only. Filter with the `split` column of `processed/events_meta.csv.gz` (same row order as the token matrix, keyed by `event_index`).
- Masked modelling on valid or test windows is leakage even without labels: the encoder would already have seen how accounts behave in the evaluation period.
- Valid and test windows can still use train-period history as context at inference time (see Theme 3, item 6).

### 3. Handoff reminder for Portion 3 (baselines)

- Join `processed/engineered_features.csv.gz` to labels on `transaction_id`; row orders across files are not guaranteed to match.
- Drop `transaction_id` and `split` before training LightGBM or Isolation Forest (see item 1).
- `current_payment_currency` and `current_payment_format` are text columns and need categorical encoding.
- Choose thresholds on valid and freeze them before computing any test metric.

## Portion 1, Theme 5: the prediction unit

Context: labels are per transaction, but the model reads per-account sequences. Each transaction is predicted once, from the sender's side: the window is the sender's last (up to) 32 events, ending at the sent event. The receiver event is kept as context in the receiver's history but gets no prediction. This gives exactly 5,078,345 targets for 5,078,345 transactions. For self-transfers, the tie-break (sent before received) keeps the target's own mirror event out of its window.

Window facts: median window length 27 events; 45.5% of windows are full (32); 8.5% have length 1 (the account's first event). Laundering targets have a median sender history of 21 events, and only 1.6% have length 1.

### 1. Some typologies are mostly invisible from the sender's side

- **What:** a sender window shows only that account's own history.
  - FAN-OUT (one to many) is visible to the sender.
  - FAN-IN (many to one) and CYCLE (A to B to C to A) are spread across accounts, so each sender sees one ordinary-looking leg. SCATTER-GATHER, GATHER-SCATTER and BIPARTITE are partly visible to the middle accounts.
- **Cost:** a per-account sequence model has a built-in ceiling on these typologies. This is the mechanism behind hypothesis H2, and the main reason graph models such as Multi-GNN are expected to score higher.
- **Note:** sender histories are not short (median 14 to 30 events per typology), so the gap comes from whose history is read, not from missing history.
- **How to report:** per typology recall, with this limitation stated next to FAN-IN and CYCLE.

### 2. The encoder has a shorter memory than the LightGBM baseline

- **What:** the encoder sees only the last 32 events. For busy accounts that can cover a few minutes. The engineered features include `prior_events_total` (full history) plus 7-day counts and counterparties.
- **Cost:** in RQ1 the baseline can have a longer view of busy accounts than the encoder, so a LightGBM win there may partly reflect memory length rather than the value of pretraining.
- **How to check:** break results down by account activity group (low, medium, high), as the plan already suggests for Experiment B.

### 3. Window length itself carries a weak signal

- **What:** laundering accounts rarely act without prior history (1.6% of laundering targets have length 1, against 8.5% of all targets). The model can read window length from the amount of left padding.
- **Note:** this is legitimate (account age is a real signal), but it is a simulator property and worth keeping in mind during error analysis.

### 4. The account-level view needs a definition

- **What:** 3,376 accounts sent at least one laundering transaction and 3,984 received at least one.
- **Decision for Portion 4:** does an account that only received laundering money count as positive in the account-level view? Under the sender-side unit, such accounts never have a laundering-labelled target of their own. State the definition used.
