"""Evaluation metrics, the shared results log, and small reporting helpers.

Portion 4 owns the final evaluation harness. Until that exists, every Portion 3
method (rules, Isolation Forest, LightGBM, frozen-embedding LightGBM, fine-tuned
encoder) is scored through this one module so the numbers in results/runs.csv are
directly comparable. The file format is deliberately boring: one row per run.

Protocol (project plan section 6):
- The minority-class F1 threshold is chosen on validation and frozen before test.
- Alert-budget precision and recall use the top k percent of scored test rows; no
  threshold is involved.
- Test is never used for any choice.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

# One row per run. Keep this list in one place: the writer, the tests and the
# README all rely on it.
RESULT_COLUMNS = [
    "run_id",
    "portion",
    "method",
    "pretrain_run",
    "label_fraction",
    "seed",
    "valid_ap",
    "test_ap",
    "test_f1",
    "threshold",
    "test_precision_at_0.5pct",
    "test_recall_at_0.5pct",
    "test_precision_at_1pct",
    "test_recall_at_1pct",
    "per_typology_recall_at_1pct",
    "train_positives",
    "train_negatives",
    "runtime_s",
    "git_commit",
    "created_at",
    "notes",
]

BUDGETS = (0.005, 0.01)


def _arrays(y, score) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y).astype(np.int8).ravel()
    score = np.asarray(score, dtype=np.float64).ravel()
    if y.shape != score.shape:
        raise ValueError(f"labels and scores differ in length: {y.shape} vs {score.shape}")
    if len(y) == 0:
        raise ValueError("empty input")
    return y, score


def pr_auc(y, score) -> float:
    """Average precision of the positive (laundering) class."""
    y, score = _arrays(y, score)
    return float(average_precision_score(y, score))


def best_f1_threshold(y, score) -> float:
    """Validation threshold that maximises minority-class F1.

    Ties are broken toward the lower threshold (higher recall), which is what
    ``np.argmax`` returns because sklearn lists thresholds in ascending order.
    Returns ``inf`` when the input has no usable threshold, so nothing is flagged.
    """
    y, score = _arrays(y, score)
    precision, recall, thresholds = precision_recall_curve(y, score)
    if len(thresholds) == 0:
        return float("inf")
    f1 = np.divide(
        2 * precision[:-1] * recall[:-1],
        precision[:-1] + recall[:-1],
        out=np.zeros(len(thresholds)),
        where=(precision[:-1] + recall[:-1]) > 0,
    )
    return float(thresholds[int(np.argmax(f1))])


def f1_at(y, score, threshold: float) -> float:
    """Minority-class F1 of the frozen-threshold rule ``score >= threshold``."""
    y, score = _arrays(y, score)
    predicted = score >= threshold
    tp = int(np.sum(predicted & (y == 1)))
    fp = int(np.sum(predicted & (y == 0)))
    fn = int(np.sum(~predicted & (y == 1)))
    denominator = 2 * tp + fp + fn
    return 0.0 if denominator == 0 else 2 * tp / denominator


def _top_k(n: int, fraction: float) -> int:
    return max(1, int(math.ceil(fraction * n)))


def precision_recall_at_budget(y, score, fraction: float) -> tuple[float, float]:
    """Precision and recall when the top ``fraction`` of rows by score is flagged."""
    y, score = _arrays(y, score)
    k = _top_k(len(y), fraction)
    flagged = np.argsort(-score, kind="stable")[:k]
    positives = int(y.sum())
    precision = float(y[flagged].mean())
    recall = 0.0 if positives == 0 else float(y[flagged].sum() / positives)
    return precision, recall


def per_typology_recall(y, score, typology, fraction: float) -> dict[str, dict]:
    """Recall at the alert budget for each typology among test positives.

    ``typology`` may contain empty strings for laundering transactions that are
    not in the pattern file; those are reported under ``(untagged)`` only if the
    caller includes them, and are never treated as negatives.
    """
    y, score = _arrays(y, score)
    typology = np.asarray([str(t) for t in np.asarray(typology, dtype=object).ravel()], dtype="U64")
    if typology.shape != y.shape:
        raise ValueError("typology and labels differ in length")
    k = _top_k(len(y), fraction)
    in_budget = np.zeros(len(y), dtype=bool)
    in_budget[np.argsort(-score, kind="stable")[:k]] = True
    result: dict[str, dict] = {}
    for name in sorted(set(typology[y == 1].tolist()) - {""}):
        mask = (y == 1) & (typology == name)
        n = int(mask.sum())
        hits = int(np.sum(mask & in_budget))
        result[name] = {"n_positives": n, "hits": hits, "recall": (hits / n) if n else 0.0}
    return result


def evaluate_scores(y_valid, s_valid, y_test, s_test, typology_test=None) -> dict:
    """All headline metrics for one run; the threshold comes from validation only."""
    threshold = best_f1_threshold(y_valid, s_valid)
    out = {
        "valid_ap": pr_auc(y_valid, s_valid),
        "test_ap": pr_auc(y_test, s_test),
        "test_f1": f1_at(y_test, s_test, threshold),
        "threshold": threshold,
    }
    for fraction in BUDGETS:
        precision, recall = precision_recall_at_budget(y_test, s_test, fraction)
        label = f"{fraction * 100:g}pct"
        out[f"test_precision_at_{label}"] = precision
        out[f"test_recall_at_{label}"] = recall
    if typology_test is not None:
        detail = per_typology_recall(y_test, s_test, typology_test, 0.01)
        out["per_typology_recall_at_1pct"] = json.dumps(detail, sort_keys=True)
    return out


def git_revision(repo_dir: Path) -> str:
    """Short commit hash, suffixed with -dirty when tracked files are modified."""
    repo_dir = Path(repo_dir)

    def run(*args):
        return subprocess.run(args, cwd=repo_dir, capture_output=True, text=True, check=False)

    try:
        sha = run("git", "rev-parse", "--short", "HEAD").stdout.strip()
        if not sha:
            return ""
        dirty = bool(run("git", "diff", "--quiet").returncode) or bool(
            run("git", "diff", "--cached", "--quiet").returncode
        )
        return f"{sha}-dirty" if dirty else sha
    except OSError:
        return ""


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def append_result(results_csv: Path, row: dict) -> None:
    """Append one run to the shared log, writing the header exactly once."""
    results_csv = Path(results_csv)
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not results_csv.exists() or results_csv.stat().st_size == 0
    with results_csv.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in RESULT_COLUMNS})


def read_results(results_csv: Path):
    """Load the shared log; returns an empty frame with the right columns if absent."""
    import pandas as pd

    results_csv = Path(results_csv)
    if not results_csv.exists() or results_csv.stat().st_size == 0:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    return pd.read_csv(results_csv)


def run_exists(results_csv: Path, run_id: str) -> bool:
    frame = read_results(results_csv)
    if frame.empty:
        return False
    return bool((frame.run_id.astype(str) == run_id).any())


def summary_frame(frame):
    """Mean and spread of test AP by method and label fraction."""
    import pandas as pd

    if frame.empty:
        return frame
    working = frame.copy()
    working["label_fraction"] = pd.to_numeric(working["label_fraction"], errors="coerce")
    working["test_ap"] = pd.to_numeric(working["test_ap"], errors="coerce")
    grouped = (
        working.groupby(["method", "label_fraction"], dropna=False)["test_ap"]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
    )
    return grouped.sort_values(["method", "label_fraction"]).reset_index(drop=True)
