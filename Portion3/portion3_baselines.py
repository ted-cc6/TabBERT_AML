"""Tabular baselines and the LightGBM experiments for Portion 3.

Everything here runs on CPU. The fine-tuned encoder grid lives in
``Portion3_Finetune.ipynb`` and reuses Portion 2's ``finetune`` unchanged; this
module only scores those runs (``score-finetune``) so every method is measured
by the same metrics module.

Leakage rules enforced here:
- ``transaction_id`` is a join key only; it never enters a feature matrix.
- Categorical vocabularies are built from train rows only; unseen values in
  valid/test become missing (code -1).
- Validation chooses thresholds; test is read once, after the threshold is set.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import portion3_metrics as metrics
from . import runplan

try:  # LightGBM is imported lazily by the tests that do not need it, but keep a clear error.
    import lightgbm as lgb
except ImportError:  # pragma: no cover - guidance instead of a traceback
    lgb = None

from sklearn.ensemble import IsolationForest

REPO_ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_COLUMNS = {"transaction_id", "split", "is_laundering", "typology"}

FEATURE_COLUMNS = [
    "current_amount_paid",
    "current_log_amount_paid",
    "current_payment_currency",  # categorical
    "current_payment_format",  # categorical
    "same_bank",
    "self_transfer",
    "previous_event_gap_min",
    "prior_events_total",
    "events_1h",
    "events_24h",
    "events_7d",
    "sent_24h",
    "received_24h",
    "distinct_counterparties_24h",
    "distinct_counterparties_7d",
    "same_currency_amount_count_24h",
    "same_currency_amount_sum_24h",
    "same_currency_amount_mean_24h",
    "amount_to_prior_mean_ratio_24h",
    "current_format_share_24h",
    "mean_interevent_gap_24h",
]
CATEGORICAL_COLUMNS = ("current_payment_currency", "current_payment_format")
NUMERIC_COLUMNS = [column for column in FEATURE_COLUMNS if column not in CATEGORICAL_COLUMNS]
CATEGORICAL_INDICES = [FEATURE_COLUMNS.index(column) for column in CATEGORICAL_COLUMNS]

LABEL_FRACTIONS = (0.01, 0.02, 0.05, 0.2, 1.0)

# The four simple rules; thresholds are train-only percentiles.
RULE_COLUMNS = (
    "current_log_amount_paid",
    "events_1h",
    "events_24h",
    "distinct_counterparties_24h",
)

DEFAULT_CONFIG = {
    "num_leaves": 31,
    "learning_rate": 0.05,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "num_boost_round": 2000,
    "early_stopping_rounds": 100,
    "embedding_columns": 256,
    "rules_percentile": 99.9,
    "isolation_forest": {"n_estimators": 200, "max_samples": 262144},
}


# --------------------------------------------------------------------------- data


@dataclass
class Data:
    targets: pd.DataFrame  # transaction_id, split, is_laundering, typology (window order)
    features: np.ndarray  # float32 (n, len(FEATURE_COLUMNS)), categoricals as codes
    feature_names: list[str]
    cat_indices: list[int]
    valid_positions: np.ndarray
    test_positions: np.ndarray
    train_mask: np.ndarray
    labels: np.ndarray
    typology: np.ndarray
    ids: np.ndarray

    @property
    def n_features(self) -> int:
        return self.features.shape[1]


def load_targets(processed_dir: Path) -> pd.DataFrame:
    targets = pd.read_csv(
        Path(processed_dir) / "window_index.csv.gz",
        usecols=["transaction_id", "split", "is_laundering", "typology"],
        dtype={"transaction_id": "int64", "split": "string", "is_laundering": "int8", "typology": "string"},
    )
    targets["typology"] = targets["typology"].fillna("").astype(str)
    return targets


def load_feature_frame(processed_dir: Path) -> pd.DataFrame:
    path = Path(processed_dir) / "engineered_features.csv.gz"
    frame = pd.read_csv(
        path,
        usecols=["transaction_id"] + FEATURE_COLUMNS,
        dtype={"transaction_id": "int64", **{column: "float32" for column in NUMERIC_COLUMNS}},
    )
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing feature columns: {missing}")
    if not frame.transaction_id.is_unique:
        raise ValueError("engineered_features has duplicate transaction_id rows; refusing to join")
    return frame


def load_data(processed_dir: Path) -> Data:
    """Load features and targets once, aligned in window_index order."""
    targets = load_targets(processed_dir)
    frame = load_feature_frame(processed_dir)
    if len(frame) != len(targets):
        raise ValueError(f"row count mismatch: features {len(frame):,}, window_index {len(targets):,}")

    order = pd.Index(frame.transaction_id).get_indexer(targets.transaction_id.to_numpy())
    if (order < 0).any():
        raise ValueError("window_index contains transaction ids absent from engineered_features")
    frame = frame.iloc[order].reset_index(drop=True)
    if not np.array_equal(frame.transaction_id.to_numpy(), targets.transaction_id.to_numpy()):
        raise ValueError("feature alignment failed")

    train_mask = (targets.split == "train").to_numpy()
    features = np.empty((len(targets), len(FEATURE_COLUMNS)), dtype=np.float32)
    for j, column in enumerate(FEATURE_COLUMNS):
        series = frame[column]
        if column in CATEGORICAL_COLUMNS:
            categories = sorted(series[train_mask].dropna().astype(str).unique())
            mapping = {category: index for index, category in enumerate(categories)}
            # Unseen values become missing (code -1); LightGBM treats negative codes as NaN.
            features[:, j] = series.astype(str).map(mapping).fillna(-1).to_numpy(dtype=np.float32)
        else:
            features[:, j] = series.to_numpy(dtype=np.float32, copy=False)
    if np.isinf(features).any():
        raise ValueError("feature matrix contains infinite values")

    split = targets.split.to_numpy()
    return Data(
        targets=targets,
        features=features,
        feature_names=list(FEATURE_COLUMNS),
        cat_indices=list(CATEGORICAL_INDICES),
        valid_positions=np.flatnonzero(split == "valid"),
        test_positions=np.flatnonzero(split == "test"),
        train_mask=train_mask,
        labels=targets.is_laundering.to_numpy(np.int8),
        typology=targets.typology.to_numpy(dtype=object),
        ids=targets.transaction_id.to_numpy(np.int64),
    )


def load_label_subset(processed_dir: Path, fraction: float, seed: int) -> pd.DataFrame:
    """Labelled train transactions for one fraction and seed (same contract as Portion 2)."""
    if not any(abs(fraction - allowed) < 1e-12 for allowed in LABEL_FRACTIONS):
        raise ValueError(f"fraction must be one of {LABEL_FRACTIONS}, got {fraction}")
    processed_dir = Path(processed_dir)
    if abs(fraction - 1.0) < 1e-12:
        return pd.read_csv(processed_dir / "label_subset_100pct.csv.gz")[
            ["transaction_id", "is_laundering"]
        ]
    subsets = pd.read_csv(processed_dir / "label_subsets.csv.gz")
    keep = (subsets.seed == seed) & np.isclose(subsets.label_fraction, fraction)
    return subsets.loc[keep, ["transaction_id", "is_laundering"]].reset_index(drop=True)


# --------------------------------------------------------------------- embeddings


def embedding_paths(directory: Path, run_name: str) -> tuple[Path, Path]:
    """Prefer the 256-column slice produced by prepare_embeddings, else the full export.

    Accepts both the Drive layout (``<dir>/<run>/embeddings_<run>.npy``) and a flat
    directory (``<dir>/embeddings_<run>.npy``).
    """
    directory = Path(directory)
    candidates = [directory / run_name, directory]
    for suffix in ("_256", ""):
        for folder in candidates:
            emb = folder / f"embeddings_{run_name}{suffix}.npy"
            ids = folder / f"embeddings_{run_name}{suffix}_ids.npy"
            if emb.exists() and ids.exists():
                return emb, ids
    raise FileNotFoundError(
        f"no embeddings for {run_name} in {directory}; expected "
        f"embeddings_{run_name}_256.npy (preferred) or embeddings_{run_name}.npy, "
        f"either flat or inside a {run_name}/ subfolder"
    )


class EmbeddingStore:
    """Memory-mapped read access to one embedding file, joined by transaction id."""

    def __init__(self, emb_path: Path, ids_path: Path, columns: int = 256):
        self.array = np.load(Path(emb_path), mmap_mode="r")
        self.ids = np.load(Path(ids_path))
        if len(self.ids) != self.array.shape[0]:
            raise ValueError(f"embedding rows {self.array.shape[0]:,} != ids {len(self.ids):,}")
        if not pd.Index(self.ids).is_unique:
            raise ValueError("embedding ids are not unique")
        if columns > self.array.shape[1]:
            raise ValueError(f"asked for {columns} columns but the file has {self.array.shape[1]}")
        self.columns = columns
        self.index = pd.Index(self.ids)

    def matrix(self, transaction_ids: np.ndarray) -> np.ndarray:
        """Rows for these ids, in the same order, read from disk in sorted order."""
        ids = np.asarray(transaction_ids, dtype=np.int64).ravel()
        positions = self.index.get_indexer(ids)
        if (positions < 0).any():
            missing = ids[positions < 0][:5]
            raise KeyError(f"{int((positions < 0).sum()):,} transaction ids have no embedding, e.g. {missing}")
        out = np.empty((len(ids), self.columns), dtype=np.float32)
        order = np.argsort(positions, kind="stable")
        sorted_positions = positions[order]
        chunk = 250_000
        for start in range(0, len(ids), chunk):
            target = order[start:start + chunk]
            out[target] = self.array[sorted_positions[start:start + chunk], : self.columns]
        return out


# -------------------------------------------------------------------------- rules


def rule_thresholds(data: Data, percentile: float = 99.9) -> np.ndarray:
    """Train-only percentile thresholds for the four rules."""
    indices = [data.feature_names.index(column) for column in RULE_COLUMNS]
    return np.percentile(data.features[data.train_mask][:, indices], percentile, axis=0)


def rule_scores(data: Data, thresholds: np.ndarray) -> np.ndarray:
    """Number of exceeded rules, with a tiny continuous tie-break for alert budgets."""
    indices = [data.feature_names.index(column) for column in RULE_COLUMNS]
    values = data.features[:, indices]
    counts = (values > thresholds).sum(axis=1).astype(np.float64)
    max_ratio = (values / np.maximum(thresholds, 1e-12)).max(axis=1)
    return counts + 1e-4 * np.log1p(max_ratio)


# ----------------------------------------------------------------------- lightgbm


def _config(path: Path | None) -> dict:
    config = json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy
    if path and Path(path).exists():
        loaded = json.loads(Path(path).read_text())
        for key, value in loaded.items():
            if key == "isolation_forest" and isinstance(value, dict):
                config["isolation_forest"].update(value)
            else:
                config[key] = value
    return config


def lightgbm_params(config: dict, n_pos: int, n_neg: int, num_threads: int, seed: int) -> dict:
    return {
        "objective": "binary",
        "learning_rate": float(config["learning_rate"]),
        "num_leaves": int(config["num_leaves"]),
        "min_data_in_leaf": int(config["min_data_in_leaf"]),
        "feature_fraction": float(config["feature_fraction"]),
        "bagging_fraction": float(config["bagging_fraction"]),
        "bagging_freq": int(config["bagging_freq"]),
        "scale_pos_weight": float(n_neg / max(n_pos, 1)),
        "metric": "None",
        "verbosity": -1,
        "num_threads": num_threads,
        "seed": seed,
        "deterministic": True,
    }


def _pr_auc_feval(preds: np.ndarray, dataset) -> tuple[str, float, bool]:
    from sklearn.metrics import average_precision_score

    return "pr_auc", float(average_precision_score(dataset.get_label(), preds)), True


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    cat_indices: list[int],
    config: dict,
    num_threads: int,
    seed: int,
    max_rounds: int | None = None,
):
    if lgb is None:
        raise ImportError("lightgbm is not installed; run: pip install -r Portion3/requirements.txt")
    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    rounds = int(config["num_boost_round"])
    if max_rounds is not None:
        rounds = min(rounds, int(max_rounds))
    params = lightgbm_params(config, n_pos, n_neg, num_threads, seed)
    train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_indices, free_raw_data=True)
    valid_set = lgb.Dataset(
        X_valid, label=y_valid, reference=train_set, categorical_feature=cat_indices, free_raw_data=True
    )
    callbacks = [
        lgb.early_stopping(int(config["early_stopping_rounds"]), first_metric_only=True, verbose=False),
        lgb.log_evaluation(period=100),
    ]
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=rounds,
        valid_sets=[valid_set],
        feval=_pr_auc_feval,
        callbacks=callbacks,
    )
    best_iteration = int(booster.best_iteration or booster.current_iteration())
    return booster, best_iteration


def predict(booster, X: np.ndarray, best_iteration: int) -> np.ndarray:
    return np.asarray(booster.predict(X, num_iteration=best_iteration), dtype=np.float64)


# ------------------------------------------------------------------ run plumbing


def _matrix(data: Data, embeddings: EmbeddingStore | None, positions: np.ndarray) -> np.ndarray:
    if embeddings is None:
        return data.features[positions]
    out = np.empty((len(positions), data.n_features + embeddings.columns), dtype=np.float32)
    out[:, : data.n_features] = data.features[positions]
    out[:, data.n_features:] = embeddings.matrix(data.ids[positions])
    return out


def _save_scores(path: Path, data: Data, valid_scores: np.ndarray, test_scores: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({
        "transaction_id": np.concatenate([data.ids[data.valid_positions], data.ids[data.test_positions]]),
        "split": ["valid"] * len(data.valid_positions) + ["test"] * len(data.test_positions),
        "score": np.concatenate([valid_scores, test_scores]),
    })
    frame.to_csv(path, index=False, compression="gzip")


def _metric_row(data: Data, valid_scores: np.ndarray, test_scores: np.ndarray) -> dict:
    return metrics.evaluate_scores(
        data.labels[data.valid_positions],
        valid_scores,
        data.labels[data.test_positions],
        test_scores,
        data.typology[data.test_positions],
    )


def execute_tabular_run(
    run: runplan.Run,
    data: Data,
    config: dict,
    processed_dir: Path,
    embeddings_dir: Path | None,
    results_csv: Path,
    scores_dir: Path,
    num_threads: int,
    max_rounds: int | None,
    save_scores: bool = True,
) -> dict:
    start = time.perf_counter()
    notes = ""
    train_positives = train_negatives = ""

    if run.method == "rules":
        thresholds = rule_thresholds(data, float(config["rules_percentile"]))
        all_scores = rule_scores(data, thresholds)
        valid_scores = all_scores[data.valid_positions]
        test_scores = all_scores[data.test_positions]
        notes = "thresholds=" + ", ".join(
            f"{column}:{value:.6g}" for column, value in zip(RULE_COLUMNS, thresholds)
        )
    elif run.method == "isolation_forest":
        forest = IsolationForest(
            n_estimators=int(config["isolation_forest"]["n_estimators"]),
            max_samples=int(config["isolation_forest"]["max_samples"]),
            random_state=int(run.seed),
            n_jobs=-1,
        )
        forest.fit(data.features[data.train_mask])
        all_scores = -forest.score_samples(data.features)  # higher = more anomalous
        valid_scores = all_scores[data.valid_positions]
        test_scores = all_scores[data.test_positions]
        notes = (
            f"n_estimators={config['isolation_forest']['n_estimators']}, "
            f"max_samples={config['isolation_forest']['max_samples']}"
        )
        del forest
        gc.collect()
    elif run.method in ("lgbm_features", "lgbm_embeddings"):
        if run.fraction is None or run.seed is None:
            raise ValueError(f"{run.run_id} needs a fraction and seed")
        subset = load_label_subset(processed_dir, run.fraction, run.seed)
        positions = pd.Index(data.ids).get_indexer(subset.transaction_id.to_numpy())
        if (positions < 0).any():
            raise ValueError(f"{run.run_id}: label subset ids missing from window_index")
        y_train = subset.is_laundering.to_numpy(np.int8)
        train_positives, train_negatives = int(y_train.sum()), int((y_train == 0).sum())

        embeddings = None
        width = 0
        if run.method == "lgbm_embeddings":
            if embeddings_dir is None:
                raise ValueError(f"{run.run_id} needs --embeddings-dir")
            emb_path, ids_path = embedding_paths(embeddings_dir, run.pretrain_run)
            width = int(config["embedding_columns"])
            embeddings = EmbeddingStore(emb_path, ids_path, columns=width)

        X_train = _matrix(data, embeddings, positions)
        X_valid = _matrix(data, embeddings, data.valid_positions)
        y_valid = data.labels[data.valid_positions]
        booster, best_iteration = train_lightgbm(
            X_train, y_train, X_valid, y_valid, data.cat_indices, config, num_threads, int(run.seed), max_rounds
        )
        del X_train
        gc.collect()
        valid_scores = predict(booster, X_valid, best_iteration)
        del X_valid
        X_test = _matrix(data, embeddings, data.test_positions)
        test_scores = predict(booster, X_test, best_iteration)
        del X_test, booster
        gc.collect()
        notes = (
            f"num_leaves={config['num_leaves']}, lr={config['learning_rate']}, "
            f"best_iter={best_iteration}"
            + (f", embedding_columns={width}" if run.method == "lgbm_embeddings" else "")
        )
    else:
        raise ValueError(f"unknown tabular method {run.method}")

    row = {
        "run_id": run.run_id,
        "portion": 3,
        "method": run.method,
        "pretrain_run": run.pretrain_run or "",
        "label_fraction": "" if run.fraction is None else run.fraction,
        "seed": "" if run.seed is None else run.seed,
        **_metric_row(data, valid_scores, test_scores),
        "train_positives": train_positives,
        "train_negatives": train_negatives,
        "runtime_s": round(time.perf_counter() - start, 1),
        "git_commit": metrics.git_revision(REPO_ROOT),
        "created_at": metrics.utc_now(),
        "notes": notes,
    }
    metrics.append_result(results_csv, row)
    if save_scores:
        _save_scores(scores_dir / f"{run.run_id}.csv.gz", data, valid_scores, test_scores)
    return row


def run_selection(
    method: str,
    fractions: tuple[float, ...] | None,
    seeds: tuple[int, ...] | None,
) -> list[runplan.Run]:
    selected = [run for run in runplan.tabular_runs() if run.method == method]
    if fractions is not None:
        selected = [run for run in selected if run.fraction is not None and any(abs(run.fraction - f) < 1e-12 for f in fractions)]
    if seeds is not None:
        selected = [run for run in selected if run.seed in seeds]
    return selected


def print_plan(runs: list[runplan.Run], results_csv: Path, force: bool) -> None:
    done = set()
    if results_csv.exists():
        frame = metrics.read_results(results_csv)
        done = set(frame.run_id.astype(str)) if not frame.empty else set()
    print(f"{'run_id':<58} {'fraction':>9} {'seed':>5}  status")
    for run in runs:
        fraction = "-" if run.fraction is None else f"{run.fraction * 100:g}%"
        seed = "-" if run.seed is None else str(run.seed)
        status = "done" if run.run_id in done else "pending"
        if status == "done" and force:
            status = "done (will rerun)"
        print(f"{run.run_id:<58} {fraction:>9} {seed:>5}  {status}")
    print(f"{len(runs)} run(s); log: {results_csv}")


# ------------------------------------------------------------------------- tuning


def tune(args) -> None:
    """Two validation-only choices: num_leaves 31 vs 63, embeddings 256 vs 512."""
    combos = [(leaves, width) for leaves in (31, 63) for width in (256, 512)]
    if args.dry_run:
        for num_leaves, width in combos:
            print(f"[dry-run] would train num_leaves={num_leaves}, embedding_columns={width}")
        return
    data = load_data(args.processed_dir)
    config = _config(args.config)
    results = []
    for num_leaves, width in combos:
        start = time.perf_counter()
        trial = json.loads(json.dumps(config))
        trial["num_leaves"] = num_leaves
        trial["embedding_columns"] = width
        subset = load_label_subset(args.processed_dir, 0.05, 42)
        positions = pd.Index(data.ids).get_indexer(subset.transaction_id.to_numpy())
        emb_path, ids_path = embedding_paths(args.embeddings_dir, "main_seed42")
        store = EmbeddingStore(emb_path, ids_path, columns=width)
        X_train = _matrix(data, store, positions)
        X_valid = _matrix(data, store, data.valid_positions)
        booster, best_iteration = train_lightgbm(
            X_train, subset.is_laundering.to_numpy(np.int8),
            X_valid, data.labels[data.valid_positions],
            data.cat_indices, trial, args.num_threads, 42, args.max_rounds,
        )
        valid_scores = predict(booster, X_valid, best_iteration)
        ap = metrics.pr_auc(data.labels[data.valid_positions], valid_scores)
        results.append({
            "num_leaves": num_leaves,
            "embedding_columns": width,
            "valid_ap": ap,
            "best_iteration": best_iteration,
            "seconds": round(time.perf_counter() - start, 1),
        })
        print(results[-1])
        del X_train, X_valid, store, booster
        gc.collect()
    if not results:
        return
    frame = pd.DataFrame(results).sort_values("valid_ap", ascending=False)
    best = frame.iloc[0]
    print("\nValidation-only comparison (5% labels, seed 42):")
    print(frame.to_string(index=False))
    print(
        f"\nRecommended: num_leaves={int(best.num_leaves)}, "
        f"embedding_columns={int(best.embedding_columns)} "
        f"(valid PR AUC {best.valid_ap:.4f})"
    )
    out = Path(args.config).with_name("lightgbm_tuning.json")
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}. Update {args.config} with the recommended values when you are sure.")


# --------------------------------------------------------------- fine-tune scores

FT_PATTERN = re.compile(r"^ft_(?P<checkpoint>.+)_(?P<pct>\d+(?:\.\d+)?)pct_seed(?P<seed>\d+)$")


def parse_finetune_folder(name: str) -> tuple[str, float, int]:
    match = FT_PATTERN.match(name)
    if not match:
        raise ValueError(f"unexpected fine-tuning folder name: {name}")
    return match.group("checkpoint"), float(match.group("pct")) / 100.0, int(match.group("seed"))


def score_finetune(args) -> None:
    scores_root = Path(args.scores_root)
    folders = sorted(path.parent for path in scores_root.glob("ft_*/scores.csv"))
    if not folders:
        raise FileNotFoundError(f"no ft_*/scores.csv under {scores_root}")
    targets = load_targets(args.processed_dir).set_index("transaction_id")
    targets = targets.rename(columns={"split": "target_split"})
    for folder in folders:
        run_id = folder.name
        if not args.force and metrics.run_exists(args.results, run_id):
            print(f"already logged: {run_id}")
            continue
        checkpoint, fraction, seed = parse_finetune_folder(run_id)
        scores = pd.read_csv(folder / "scores.csv")
        scores = scores.join(targets[["is_laundering", "typology", "target_split"]], on="transaction_id")
        if scores.is_laundering.isna().any():
            raise ValueError(f"{run_id}: score rows not found in window_index")
        if not (scores.split == scores.target_split).all():
            raise ValueError(f"{run_id}: scores.csv split column disagrees with window_index")
        valid = scores[scores.split == "valid"]
        test = scores[scores.split == "test"]
        summary_path = folder / "summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        log_path = folder / "finetune_log.csv"
        runtime = ""
        if log_path.exists():
            runtime = float(pd.read_csv(log_path).seconds.iloc[-1])
        row = {
            "run_id": run_id,
            "portion": 3,
            "method": "finetune",
            "pretrain_run": checkpoint,
            "label_fraction": fraction,
            "seed": seed,
            **metrics.evaluate_scores(
                valid.is_laundering.to_numpy(np.int8), valid.score.to_numpy(),
                test.is_laundering.to_numpy(np.int8), test.score.to_numpy(),
                test.typology.to_numpy(dtype=object),
            ),
            "train_positives": summary.get("positives", ""),
            "train_negatives": summary.get("negatives", ""),
            "runtime_s": runtime,
            "git_commit": metrics.git_revision(REPO_ROOT),
            "created_at": metrics.utc_now(),
            "notes": f"best_epoch={summary.get('best_epoch', '')}, summary_valid_ap={summary.get('valid_ap', '')}",
        }
        metrics.append_result(args.results, row)
        print(f"logged {run_id}: test AP {row['test_ap']:.4f}, F1 {row['test_f1']:.4f}")


# ------------------------------------------------------------------------- CLI


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--processed-dir", type=Path, default=REPO_ROOT / "Portion1" / "processed")
    parser.add_argument("--results", type=Path, default=REPO_ROOT / "results" / "runs.csv")
    parser.add_argument("--outputs-dir", type=Path, default=REPO_ROOT / "Portion3" / "outputs")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "Portion3" / "configs" / "lightgbm.json")
    parser.add_argument("--num-threads", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--max-rounds", type=int, default=None, help="cap boosting rounds (smoke tests)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="rerun runs already present in the log")
    parser.add_argument("--no-scores", action="store_true", help="do not write per-run score files")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("plan", help="print the run plan and completion status")
    add_common(p)

    for name, method in (
        ("rules", "rules"),
        ("isolation-forest", "isolation_forest"),
        ("lightgbm", "lgbm_features"),
        ("lightgbm-embeddings", "lgbm_embeddings"),
    ):
        p = sub.add_parser(name)
        add_common(p)
        p.add_argument("--fractions", nargs="+", type=float, default=None)
        p.add_argument("--seeds", nargs="+", type=int, default=None)
        if method == "lgbm_embeddings":
            p.add_argument("--embeddings-dir", type=Path, required=True)

    p = sub.add_parser("tune", help="validation-only choice of num_leaves and embedding width")
    add_common(p)
    p.add_argument("--embeddings-dir", type=Path, required=True)

    p = sub.add_parser("score-finetune", help="score fine-tuning score files and append to the log")
    add_common(p)
    p.add_argument("--scores-root", type=Path, required=True)

    p = sub.add_parser("summary", help="mean and spread of test AP by method and fraction")
    add_common(p)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "summary":
        frame = metrics.read_results(args.results)
        if frame.empty:
            print(f"no results yet at {args.results}")
            return 0
        print(metrics.summary_frame(frame).to_string(index=False))
        return 0

    if args.command == "plan":
        runs = runplan.new_runs()
        print_plan(runs, args.results, args.force)
        return 0

    if args.command == "score-finetune":
        score_finetune(args)
        return 0

    if args.command == "tune":
        tune(args)
        return 0

    method = {
        "rules": "rules",
        "isolation-forest": "isolation_forest",
        "lightgbm": "lgbm_features",
        "lightgbm-embeddings": "lgbm_embeddings",
    }[args.command]
    runs = run_selection(
        method,
        tuple(args.fractions) if getattr(args, "fractions", None) else None,
        tuple(args.seeds) if getattr(args, "seeds", None) else None,
    )
    embeddings_dir = getattr(args, "embeddings_dir", None)
    print_plan(runs, args.results, args.force)
    if args.dry_run:
        return 0
    if method == "lgbm_embeddings" and embeddings_dir is None:
        parser.error("--embeddings-dir is required")
    if method == "lgbm_embeddings" and not Path(embeddings_dir).exists():
        parser.error(f"embeddings directory does not exist: {embeddings_dir}")

    config = _config(args.config)
    data = load_data(args.processed_dir)
    scores_dir = Path(args.outputs_dir) / "scores"
    for run in runs:
        if not args.force and metrics.run_exists(args.results, run.run_id):
            print(f"skip (already logged): {run.run_id}")
            continue
        print(f"\n=== {run.run_id} ===", flush=True)
        row = execute_tabular_run(
            run, data, config, args.processed_dir, embeddings_dir,
            args.results, scores_dir, args.num_threads, args.max_rounds,
            save_scores=not args.no_scores,
        )
        print(
            f"valid AP {row['valid_ap']:.4f} | test AP {row['test_ap']:.4f} | "
            f"F1 {row['test_f1']:.4f} | {row['runtime_s']:.0f}s",
            flush=True,
        )
    print(f"\nWrote {args.results}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
