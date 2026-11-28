"""Synthetic processed-data fixture for the Portion 3 tests.

The tests must not depend on the 221 MB real feature file. They build a tiny
copy of the Portion 1 contract in a temporary directory: same columns, same
join keys, a shuffled feature row order (so the join is actually exercised), a
currency value that only appears in validation, and a label subset layout that
matches the real files.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from Portion3 import portion3_baselines as pb

N_TRAIN = 60
N_VALID = 15
N_TEST = 15


def _make_frame(rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = np.arange(1000, 1000 + N_TRAIN + N_VALID + N_TEST)
    split = ["train"] * N_TRAIN + ["valid"] * N_VALID + ["test"] * N_TEST

    features = pd.DataFrame({"transaction_id": ids, "split": split})
    for column in pb.NUMERIC_COLUMNS:
        features[column] = rng.uniform(0.1, 100.0, len(ids)).astype(np.float32)
    features["current_log_amount_paid"] = rng.uniform(0.0, 12.0, len(ids)).astype(np.float32)
    features["events_1h"] = rng.integers(0, 10, len(ids))
    features["events_24h"] = rng.integers(0, 40, len(ids))
    features["distinct_counterparties_24h"] = rng.integers(0, 8, len(ids))
    features["previous_event_gap_min"] = rng.integers(-1, 500, len(ids))
    features["mean_interevent_gap_24h"] = rng.uniform(-1.0, 200.0, len(ids))
    # One currency appears only outside train, to check the train-only vocabulary.
    currency = np.where(np.array(split) == "train", "USD", "USD")
    currency[np.array(split) == "train"] = np.where(rng.random(N_TRAIN) < 0.5, "USD", "GBP")
    currency[N_TRAIN] = "EUR"
    features["current_payment_currency"] = currency
    features["current_payment_format"] = np.where(rng.random(len(ids)) < 0.5, "ACH", "Cheque")
    features = features.sample(frac=1.0, random_state=7).reset_index(drop=True)  # shuffle row order

    labels = np.zeros(len(ids), dtype=np.int8)
    train_positions = np.arange(0, N_TRAIN, 3)  # 20 positives
    labels[train_positions] = 1
    labels[N_TRAIN + 2] = 1  # one validation positive
    labels[N_TRAIN + 6] = 1
    labels[N_TRAIN + N_VALID + 1] = 1  # two test positives
    labels[N_TRAIN + N_VALID + 9] = 1
    typology = np.array([""] * len(ids), dtype=object)
    typology[train_positions[0]] = "FAN-OUT"
    typology[N_TRAIN + 2] = "FAN-OUT"
    typology[N_TRAIN + 6] = "CYCLE"
    typology[N_TRAIN + N_VALID + 1] = "FAN-OUT"
    typology[N_TRAIN + N_VALID + 9] = "CYCLE"

    targets = pd.DataFrame({
        "transaction_id": ids,
        "target_event_index": np.arange(len(ids)) + 10,
        "window_start_index": np.zeros(len(ids), dtype=int),
        "window_length": np.full(len(ids), 5),
        "split": split,
        "is_laundering": labels,
        "typology": typology,
    })
    return features, targets


def _write_label_subsets(processed_dir: Path, targets: pd.DataFrame) -> None:
    train = targets[targets.split == "train"]
    positives = train[train.is_laundering == 1].transaction_id
    negatives = train[train.is_laundering == 0].transaction_id.to_numpy()
    rows = []
    for seed in (42, 43, 44):
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(negatives)
        for fraction in (0.01, 0.02, 0.05, 0.2):
            picked = shuffled[: max(1, int(len(negatives) * fraction))]
            for transaction_id in np.concatenate([positives.to_numpy(), picked]):
                rows.append({
                    "seed": seed,
                    "label_fraction": fraction,
                    "transaction_id": int(transaction_id),
                    "is_laundering": int(transaction_id in set(positives)),
                })
    pd.DataFrame(rows).to_csv(processed_dir / "label_subsets.csv.gz", index=False, compression="gzip")
    train[["transaction_id", "is_laundering"]].to_csv(
        processed_dir / "label_subset_100pct.csv.gz", index=False, compression="gzip"
    )
    manifest = {
        "seeds": [42, 43, 44],
        "counts": {"42:0.05": {"positive": int(positives.size), "negative": 32},
                   "42:1.0": {"positive": int(positives.size), "negative": int((train.is_laundering == 0).sum())}},
    }
    (processed_dir / "label_subset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


@pytest.fixture(scope="session")
def processed_dir(tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("processed")
    rng = np.random.default_rng(1234)
    features, targets = _make_frame(rng)
    features.to_csv(directory / "engineered_features.csv.gz", index=False, compression="gzip")
    targets.to_csv(directory / "window_index.csv.gz", index=False, compression="gzip")
    _write_label_subsets(directory, targets)
    return directory


@pytest.fixture(scope="session")
def data(processed_dir: Path):
    return pb.load_data(processed_dir)
