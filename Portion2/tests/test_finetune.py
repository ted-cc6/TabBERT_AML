"""Checks for label subsets and fine-tuning (Task 6) on a tiny checkpoint."""

import numpy as np
import pandas as pd
import pytest

import portion2_data as pdata
import portion2_encoder as enc
from conftest import PROCESSED_DIR, tiny_checkpoint


@pytest.fixture(scope="module")
def window_index():
    return pdata.load_window_index(PROCESSED_DIR)


def test_label_subset_counts():
    five = enc.load_label_subset(PROCESSED_DIR, 0.05, seed=42)
    assert list(five.columns) == ["transaction_id", "is_laundering"]
    assert (five.is_laundering.sum(), (five.is_laundering == 0).sum()) == (127, 162_320)
    full = enc.load_label_subset(PROCESSED_DIR, 1.0, seed=43)
    assert (full.is_laundering.sum(), (full.is_laundering == 0).sum()) == (2_530, 3_246_391)
    with pytest.raises(ValueError):
        enc.load_label_subset(PROCESSED_DIR, 0.03, seed=42)


def test_label_subsets_are_train_only_and_nested(window_index):
    train_ids = set(window_index.transaction_id[window_index.split == "train"])
    one = enc.load_label_subset(PROCESSED_DIR, 0.01, seed=44)
    five = enc.load_label_subset(PROCESSED_DIR, 0.05, seed=44)
    assert set(five.transaction_id) <= train_ids
    assert set(one.transaction_id) <= set(five.transaction_id)


def test_finetune_epoch_sample_caps_negatives():
    labels = np.array([1] * 30 + [0] * 1000)
    picked = enc.finetune_epoch_sample(labels, max_negatives=100, seed=42, epoch=0)
    assert labels[picked].sum() == 30 and (labels[picked] == 0).sum() == 100
    assert len(np.unique(picked)) == 130
    np.testing.assert_array_equal(picked, enc.finetune_epoch_sample(labels, 100, seed=42, epoch=0))
    assert not np.array_equal(picked, enc.finetune_epoch_sample(labels, 100, seed=42, epoch=1))
    assert len(enc.finetune_epoch_sample(labels, max_negatives=5000, seed=42, epoch=0)) == 1030  # cap above count


def test_finetune_smoke(tmp_path, window_index):
    ckpt = tiny_checkpoint(tmp_path / "ckpt")
    split = window_index.split.to_numpy()
    positive = window_index.is_laundering.to_numpy() == 1
    rng = np.random.default_rng(0)
    valid_rows = np.concatenate([np.flatnonzero((split == "valid") & positive)[:20],
                                 rng.choice(np.flatnonzero((split == "valid") & ~positive), 980, replace=False)])
    test_rows = rng.choice(np.flatnonzero(split == "test"), 300, replace=False)

    scores_path = enc.finetune(ckpt, PROCESSED_DIR, tmp_path / "ft", fraction=0.01, seed=42, max_epochs=2,
                               max_negatives_per_epoch=2000, eval_rows={"valid": valid_rows, "test": test_rows})

    assert scores_path.name == "scores.csv" and scores_path.parent.name == "ft_tiny_1pct_seed42"
    scores = pd.read_csv(scores_path)
    assert list(scores.columns) == ["transaction_id", "split", "score"]
    assert (scores.split == "valid").sum() == 1000 and (scores.split == "test").sum() == 300
    assert np.isfinite(scores.score).all() and scores.score.between(0, 1).all()
    log = pd.read_csv(scores_path.parent / "finetune_log.csv")
    assert list(log.epoch) == [1, 2]
    assert log.valid_ap.between(0, 1).all()
    assert "test_ap" not in log.columns  # test is scored, never used for choices
