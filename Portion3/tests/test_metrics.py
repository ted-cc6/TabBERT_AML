from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, precision_recall_curve

from Portion3 import portion3_metrics as metrics


def test_pr_auc_matches_sklearn():
    y = [1, 1, 0, 0, 0]
    score = [0.9, 0.4, 0.8, 0.3, 0.1]
    assert metrics.pr_auc(y, score) == pytest.approx(average_precision_score(y, score))


def test_best_f1_threshold_maximises_f1():
    y = np.array([1, 1, 0, 0, 1, 0, 0, 0])
    score = np.array([0.9, 0.7, 0.8, 0.6, 0.55, 0.5, 0.2, 0.1])
    threshold = metrics.best_f1_threshold(y, score)
    brute = [metrics.f1_at(y, score, t) for t in np.unique(score)]
    assert metrics.f1_at(y, score, threshold) == pytest.approx(max(brute))


def test_budget_precision_and_recall():
    y = [1, 0, 0, 1]
    score = [0.9, 0.8, 0.1, 0.05]
    precision, recall = metrics.precision_recall_at_budget(y, score, 0.5)
    assert precision == pytest.approx(0.5)
    assert recall == pytest.approx(0.5)


def test_budget_uses_ceiling():
    y = [1, 1, 1, 1, 1]
    score = [0.5, 0.4, 0.3, 0.2, 0.1]
    precision, recall = metrics.precision_recall_at_budget(y, score, 0.1)
    assert precision == 1.0
    assert recall == pytest.approx(1 / 5)


def test_per_typology_recall():
    y = [1, 1, 0]
    score = [0.9, 0.1, 0.8]
    typology = ["FAN-OUT", "CYCLE", ""]
    result = metrics.per_typology_recall(y, score, typology, 1 / 3)
    assert result["FAN-OUT"]["recall"] == 1.0
    assert result["CYCLE"]["recall"] == 0.0
    assert result["CYCLE"]["n_positives"] == 1


def test_evaluate_scores_uses_validation_threshold():
    y_valid = [1, 1, 0, 0, 0]
    s_valid = [0.9, 0.8, 0.7, 0.6, 0.5]
    y_test = [1, 1, 0]
    s_test = [0.85, 0.75, 0.6]
    out = metrics.evaluate_scores(y_valid, s_valid, y_test, s_test, ["FAN-OUT", "CYCLE", ""])
    assert out["threshold"] == pytest.approx(0.8)
    assert out["test_f1"] == pytest.approx(2 / 3)
    assert out["test_ap"] == pytest.approx(1.0)
    detail = json.loads(out["per_typology_recall_at_1pct"])
    assert detail["FAN-OUT"]["recall"] == 1.0
    assert detail["CYCLE"]["recall"] == 0.0


def test_result_columns_match_writer(tmp_path):
    path = tmp_path / "runs.csv"
    metrics.append_result(path, {"run_id": "a", "portion": 3, "method": "rules"})
    metrics.append_result(path, {"run_id": "b", "portion": 3, "method": "rules"})
    frame = metrics.read_results(path)
    assert list(frame.columns) == metrics.RESULT_COLUMNS
    assert frame.run_id.tolist() == ["a", "b"]
    assert metrics.run_exists(path, "a")
    assert not metrics.run_exists(path, "c")
