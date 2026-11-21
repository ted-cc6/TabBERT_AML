"""Checks for the Portion 2 data loader against the real Portion 1 outputs.

The fixtures read the files directly, so each loader function can be tested on its own.
"""

import numpy as np
import pandas as pd
import pytest
import torch

import portion2_data as pdata
from conftest import PROCESSED_DIR

N_EVENTS = 10_156_690
N_TRANSACTIONS = 5_078_345
N_TRAIN = 3_248_921
WINDOW_COLUMNS = [
    "transaction_id", "target_event_index", "window_start_index",
    "window_length", "split", "is_laundering", "typology",
]

# Transaction 2120422 (train, laundering, FAN-OUT), decoded on 2026-11-19.
# Columns: payment_format, currency, counterparty_bank, direction, amount_bucket, time_gap_bucket.
REFERENCE_ROWS = np.array([
    [5, 9, 516, 3, 10, 3],
    [5, 9, 516, 4, 10, 4],
    [6, 9, 516, 3, 18, 32],
    [6, 3, 516, 4, 19, 4],
    [6, 3, 311, 3, 19, 4],
    [6, 9, 516, 3, 23, 27],
    [6, 4, 516, 4, 24, 4],
    [6, 4, 338, 3, 24, 4],
])


@pytest.fixture(scope="module")
def tokens():
    return np.load(PROCESSED_DIR / "event_tokens.npz")["tokens"]


@pytest.fixture(scope="module")
def window_index():
    return pd.read_csv(PROCESSED_DIR / "window_index.csv.gz", keep_default_na=False)


def window_args(window_index, row):
    record = window_index.iloc[row]
    return int(record.window_start_index), int(record.target_event_index)


def test_load_tokens():
    loaded = pdata.load_tokens(PROCESSED_DIR)
    assert loaded.shape == (N_EVENTS, 6)
    assert loaded.dtype == np.int32


def test_load_window_index():
    loaded = pdata.load_window_index(PROCESSED_DIR)
    assert list(loaded.columns) == WINDOW_COLUMNS
    assert len(loaded) == N_TRANSACTIONS
    assert (loaded.split == "train").sum() == N_TRAIN


def test_reference_window_2120422(tokens, window_index):
    row = window_index.index[window_index.transaction_id == 2120422][0]
    start, target = window_args(window_index, row)
    assert (start, target) == (1213435, 1213442)

    window, event_mask = pdata.make_window(tokens, start, target)

    assert window.shape == (32, 6)
    assert window.dtype == np.int64
    assert event_mask.dtype == np.bool_
    assert (window[:24] == 0).all()
    np.testing.assert_array_equal(window[24:], REFERENCE_ROWS)
    assert not event_mask[:24].any()
    assert event_mask[24:].all()


def test_length_one_window(tokens, window_index):
    record = window_index.iloc[0]
    assert (record.transaction_id, record.window_start_index, record.target_event_index) == (213094, 1, 1)

    window, event_mask = pdata.make_window(tokens, 1, 1)

    assert (window[:31] == 0).all()
    np.testing.assert_array_equal(window[31], tokens[1])
    assert window[31, 5] == 3  # NO_HISTORY: first event of the account
    assert event_mask.sum() == 1
    assert event_mask[31]


def test_length_32_window(tokens, window_index):
    row = int(np.flatnonzero(window_index.window_length.to_numpy() == 32)[0])
    start, target = window_args(window_index, row)

    window, event_mask = pdata.make_window(tokens, start, target)

    assert event_mask.all()
    np.testing.assert_array_equal(window, tokens[start:target + 1])
    np.testing.assert_array_equal(window[0], tokens[start])
    np.testing.assert_array_equal(window[31], tokens[target])


def test_build_batch_matches_make_window(tokens, window_index):
    rows = np.random.default_rng(0).choice(len(window_index), size=1000, replace=False)
    lengths = window_index.window_length.to_numpy()
    rows = np.concatenate([rows, [0, int(np.flatnonzero(lengths == 32)[0])]])
    starts = window_index.window_start_index.to_numpy()[rows]
    targets = window_index.target_event_index.to_numpy()[rows]

    batch, batch_mask = pdata.build_batch(tokens, starts, targets)

    assert batch.shape == (len(rows), 32, 6)
    assert batch.dtype == np.int64
    assert batch_mask.shape == (len(rows), 32)
    for i, (start, target) in enumerate(zip(starts, targets)):
        window, event_mask = pdata.make_window(tokens, int(start), int(target))
        np.testing.assert_array_equal(batch[i], window)
        np.testing.assert_array_equal(batch_mask[i], event_mask)


def test_sample_is_train_only_and_reproducible(window_index):
    rows = pdata.sample_epoch_rows(window_index, "train", 10_000, seed=42, epoch=0)

    assert len(rows) == 10_000
    assert len(np.unique(rows)) == 10_000
    assert (window_index.split.to_numpy()[rows] == "train").all()
    np.testing.assert_array_equal(rows, pdata.sample_epoch_rows(window_index, "train", 10_000, seed=42, epoch=0))
    assert not np.array_equal(rows, pdata.sample_epoch_rows(window_index, "train", 10_000, seed=42, epoch=1))
    assert not np.array_equal(rows, pdata.sample_epoch_rows(window_index, "train", 10_000, seed=43, epoch=0))

    all_rows = pdata.sample_epoch_rows(window_index, "train", None, seed=42, epoch=0)
    assert len(all_rows) == N_TRAIN
    assert (np.diff(all_rows) > 0).all()  # n=None keeps the stored row order


def test_train_windows_contain_only_train_events(window_index):
    rows = pdata.sample_epoch_rows(window_index, "train", 10_000, seed=42, epoch=0)
    starts = window_index.window_start_index.to_numpy()[rows]
    targets = window_index.target_event_index.to_numpy()[rows]
    np.testing.assert_array_equal(targets - starts + 1, window_index.window_length.to_numpy()[rows])

    event_split = pd.read_csv(PROCESSED_DIR / "events_meta.csv.gz", usecols=["split"]).split.to_numpy()
    window_events = np.concatenate([np.arange(s, t + 1) for s, t in zip(starts, targets)])
    assert (event_split[window_events] == "train").all()


def test_iterate_batches_shapes_and_ids(tokens, window_index):
    rows = np.arange(0, 2200, 2)  # 1,100 rows: batches of 512, 512 and 76
    batches = list(pdata.iterate_batches(tokens, window_index, rows, batch_size=512))

    assert [len(b["transaction_id"]) for b in batches] == [512, 512, 76]
    first = batches[0]
    assert set(first) == {"tokens", "event_mask", "transaction_id"}
    assert first["tokens"].dtype == torch.long and first["tokens"].shape == (512, 32, 6)
    assert first["event_mask"].dtype == torch.bool and first["event_mask"].shape == (512, 32)
    assert first["transaction_id"].dtype == torch.long

    ids = torch.cat([b["transaction_id"] for b in batches]).numpy()
    np.testing.assert_array_equal(ids, window_index.transaction_id.to_numpy()[rows])

    expected, expected_mask = pdata.build_batch(
        tokens,
        window_index.window_start_index.to_numpy()[rows[:512]],
        window_index.target_event_index.to_numpy()[rows[:512]],
    )
    np.testing.assert_array_equal(first["tokens"].numpy(), expected)
    np.testing.assert_array_equal(first["event_mask"].numpy(), expected_mask)
