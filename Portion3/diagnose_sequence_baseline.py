"""Fairness diagnostic: LightGBM with hand-made sequence features, 100% labels.

The main Experiment A baseline uses the 21 engineered features. This diagnostic adds 14
window features (pass-through flags, burst shape, bank and amount diversity, gap
statistics) to see how much of the fine-tuned encoder's advantage a handful of
hand-made sequence features can reproduce. It trains one model at 100% labels, seed 42,
and writes a JSON report. It is not part of results/runs.csv.

    python Portion3/diagnose_sequence_baseline.py
    python Portion3/diagnose_sequence_baseline.py --max-rounds 300

Expected (2026-11-28): valid PR AUC 0.0411, test 0.0326, far below the fine-tuned
encoder's 0.428 at 5% labels. The extra features help against the 0.0123 feature-only
baseline and remain an order of magnitude short, so the gap is representation rather
than one missing feature.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from Portion3 import portion3_metrics as metrics  # noqa: E402
from Portion3 import portion3_baselines as pb  # noqa: E402
from Portion3 import sequence_features  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--processed-dir", type=Path, default=REPO_ROOT / "Portion1" / "processed")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "Portion3" / "configs" / "lightgbm.json")
    parser.add_argument("--max-rounds", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-threads", type=int, default=None)
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "Portion3" / "outputs" / "sequence_baseline_diagnostic.json")
    args = parser.parse_args(argv)

    start = time.perf_counter()
    data = pb.load_data(args.processed_dir)
    windows = pd.read_csv(
        Path(args.processed_dir) / "window_index.csv.gz",
        usecols=["transaction_id", "target_event_index", "window_start_index"],
        dtype={"transaction_id": "int64"},
    )
    order = pd.Index(windows.transaction_id).get_indexer(data.ids)
    if (order < 0).any():
        raise SystemExit("window_index and features disagree on transaction ids")
    windows = windows.iloc[order]
    tokens = np.load(Path(args.processed_dir) / "event_tokens.npz")["tokens"]
    seq = sequence_features.build_features(
        tokens, windows.window_start_index.to_numpy(), windows.target_event_index.to_numpy()
    )
    print(f"built {len(sequence_features.FEATURE_NAMES)} sequence features in "
          f"{time.perf_counter() - start:.0f}s", flush=True)

    X = np.empty((len(data.ids), data.n_features + len(sequence_features.FEATURE_NAMES)), dtype=np.float32)
    X[:, : data.n_features] = data.features
    for j, name in enumerate(sequence_features.FEATURE_NAMES):
        X[:, data.n_features + j] = seq[name]

    train = np.flatnonzero(data.train_mask)
    valid, test = data.valid_positions, data.test_positions
    config = pb._config(args.config)
    num_threads = args.num_threads or (os.cpu_count() or 4)
    train_start = time.perf_counter()
    booster, best_iteration = pb.train_lightgbm(
        X[train], data.labels[train], X[valid], data.labels[valid],
        data.cat_indices, config, num_threads, args.seed, args.max_rounds,
    )
    valid_scores = pb.predict(booster, X[valid], best_iteration)
    test_scores = pb.predict(booster, X[test], best_iteration)
    summary = metrics.evaluate_scores(
        data.labels[valid], valid_scores, data.labels[test], test_scores, data.typology[test]
    )
    summary.update({
        "seed": args.seed,
        "label_fraction": 1.0,
        "features": data.n_features + len(sequence_features.FEATURE_NAMES),
        "sequence_features": sequence_features.FEATURE_NAMES,
        "best_iteration": best_iteration,
        "num_leaves": config["num_leaves"],
        "runtime_s": round(time.perf_counter() - start, 1),
        "train_s": round(time.perf_counter() - train_start, 1),
        "git_commit": metrics.git_revision(REPO_ROOT),
        "created_at": metrics.utc_now(),
        "note": "diagnostic only; not in results/runs.csv",
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"valid AP {summary['valid_ap']:.4f} | test AP {summary['test_ap']:.4f} | "
        f"test F1 {summary['test_f1']:.4f} | best iter {best_iteration} | "
        f"{summary['runtime_s']:.0f}s total"
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
