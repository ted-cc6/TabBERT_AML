from __future__ import annotations

import json

import numpy as np
import pandas as pd

from Portion3 import portion3_baselines as pb
from Portion3 import runplan


def test_train_lightgbm_smoke(data):
    positions = np.flatnonzero(data.train_mask & (data.labels == 1))
    negatives = np.flatnonzero(data.train_mask & (data.labels == 0))
    rng = np.random.default_rng(0)
    positions = np.concatenate([positions, rng.choice(negatives, size=10, replace=False)])
    y_train = data.labels[positions]
    X_train = data.features[positions]
    X_valid = data.features[data.valid_positions]
    y_valid = data.labels[data.valid_positions]
    booster, best_iteration = pb.train_lightgbm(
        X_train, y_train, X_valid, y_valid, data.cat_indices, pb.DEFAULT_CONFIG, num_threads=2, seed=42, max_rounds=20
    )
    scores = pb.predict(booster, X_valid, best_iteration)
    assert scores.shape == (len(y_valid),)
    assert np.isfinite(scores).all()


def test_execute_rules_run_writes_log_and_scores(data, tmp_path):
    results = tmp_path / "runs.csv"
    run = runplan.Run("rules", "tabular", "rules", None, None, None)
    row = pb.execute_tabular_run(
        run, data, pb.DEFAULT_CONFIG, pb.REPO_ROOT / "Portion1" / "processed",
        None, results, tmp_path / "scores", num_threads=2, max_rounds=None,
    )
    assert row["run_id"] == "rules"
    assert 0.0 <= row["test_ap"] <= 1.0
    frame = pd.read_csv(results)
    assert len(frame) == 1
    scores = pd.read_csv(tmp_path / "scores" / "rules.csv.gz")
    assert len(scores) == len(data.valid_positions) + len(data.test_positions)
    assert set(scores.split) == {"valid", "test"}


def _write_reference_scores(processed_dir, root):
    """Perfect scores on the synthetic fixture, in Portion 2's folder layout."""
    targets = pd.read_csv(processed_dir / "window_index.csv.gz")
    folder = root / pb.REFERENCE_RUN
    folder.mkdir(parents=True)
    frame = targets.loc[targets.split.isin(["valid", "test"]), ["transaction_id", "split", "is_laundering"]]
    frame = frame.rename(columns={"is_laundering": "score"})
    frame.to_csv(folder / "scores.csv", index=False)


def test_reference_check_pass_and_fail(processed_dir, tmp_path):
    scores_root = tmp_path / "finetune"
    _write_reference_scores(processed_dir, scores_root)
    marker = tmp_path / "check.json"

    class Args:
        pass

    args = Args()
    args.scores_root = scores_root
    args.processed_dir = processed_dir
    args.expected_ap = 1.0
    args.tolerance = 0.01
    args.out = marker
    assert pb.reference_check(args) == 0
    assert json.loads(marker.read_text())["status"] == "pass"

    args.expected_ap = 0.0
    assert pb.reference_check(args) == 1
    assert json.loads(marker.read_text())["status"] == "fail"

    args.scores_root = tmp_path / "empty"
    assert pb.reference_check(args) == 2
