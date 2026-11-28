from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from Portion3 import portion3_baselines as pb


def test_features_are_aligned_to_window_index(data, processed_dir):
    targets = pd.read_csv(processed_dir / "window_index.csv.gz")
    assert np.array_equal(data.ids, targets.transaction_id.to_numpy())
    assert data.features.shape == (len(targets), len(pb.FEATURE_COLUMNS))
    assert data.feature_names == pb.FEATURE_COLUMNS
    assert not (pb.FORBIDDEN_COLUMNS & set(data.feature_names))
    assert np.isfinite(data.features).all()


def test_categories_are_built_from_train_only(data):
    column = data.feature_names.index("current_payment_currency")
    eur_rows = data.targets.index[data.targets.transaction_id == 1000 + 60]
    assert len(eur_rows) == 1
    assert data.features[eur_rows[0], column] == -1  # unseen in train -> missing


def test_label_subset_contract(processed_dir):
    subset = pb.load_label_subset(processed_dir, 0.05, 42)
    assert subset.is_laundering.sum() == 20
    assert len(subset) == 22
    full = pb.load_label_subset(processed_dir, 1.0, 42)
    assert len(full) == 60
    with pytest.raises(ValueError):
        pb.load_label_subset(processed_dir, 0.03, 42)


def test_rule_thresholds_ignore_valid_and_test(data):
    original = pb.rule_thresholds(data)
    features = data.features.copy()
    for column in pb.RULE_COLUMNS:
        index = data.feature_names.index(column)
        features[~data.train_mask, index] = 1e9
    altered = pb.rule_thresholds(dataclasses.replace(data, features=features))
    assert np.array_equal(original, altered)


def test_rule_score_counts_exceeded_rules(data):
    thresholds = pb.rule_thresholds(data)
    features = data.features.copy()
    positions = data.valid_positions[:2]
    all_indices = [data.feature_names.index(column) for column in pb.RULE_COLUMNS]
    features[positions[0], all_indices] = thresholds * 1000 + 1
    features[positions[1], all_indices] = 0
    scores = pb.rule_scores(dataclasses.replace(data, features=features), thresholds)
    assert scores[positions[0]] >= 4
    assert scores[positions[0]] < 5
    assert scores[positions[1]] == 0
