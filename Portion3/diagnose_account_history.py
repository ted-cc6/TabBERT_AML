"""Diagnostic: how much of the supervised signal is just account history?

For each sender account, compute its laundering rate in the labelled subset, map that
rate to every valid/test transaction of the same account, and evaluate the ranking. This
is a label-history baseline, not a model, and it is not part of the main Experiment A
table (it is not in the project plan). It exists to interpret how much of any encoder
advantage could come from "this account has laundered before" rather than from
representation learning.

    python Portion3/diagnose_account_history.py --fraction 1.0
    python Portion3/diagnose_account_history.py --fraction 0.05 --seed 42

Writes Portion3/outputs/account_history_diagnostic.json and prints a small table. It does
not touch results/runs.csv: this is a diagnostic, not a run.

Reference values from 2026-11-27 (seed 42):
- 100% labels: valid AP 0.0723 (67x the 0.00107 base rate)
- 5% labels:   valid AP 0.0127 (12x)
For comparison, the 21 engineered features reach 0.0123 at 100% labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:  # python -m Portion3.diagnose_account_history
    from . import portion3_metrics as metrics
    from . import portion3_baselines as pb
except ImportError:  # python Portion3/diagnose_account_history.py
    sys.path.insert(0, str(REPO_ROOT))
    from Portion3 import portion3_metrics as metrics
    from Portion3 import portion3_baselines as pb


def diagnose(processed_dir: Path, fraction: float, seed: int, smooth: float = 20.0) -> dict:
    windows = pb.load_targets(processed_dir)
    accounts = pd.read_csv(
        Path(processed_dir) / "transactions_clean.csv.gz",
        usecols=["transaction_id", "from_account_key"],
    )
    frame = windows.merge(accounts, on="transaction_id")
    subset = pb.load_label_subset(processed_dir, fraction, seed).merge(accounts, on="transaction_id")
    prior = float(frame.loc[frame.split == "train", "is_laundering"].mean())

    by_account = subset.groupby("from_account_key").is_laundering.agg(["sum", "count"])
    by_account["rate"] = (by_account["sum"] + smooth * prior) / (by_account["count"] + smooth)
    positive_accounts = set(subset.loc[subset.is_laundering == 1, "from_account_key"])

    report: dict = {
        "fraction": fraction,
        "seed": seed,
        "smoothing": smooth,
        "labelled_positives": int(subset.is_laundering.sum()),
        "labelled_positive_accounts": len(positive_accounts),
        "base_rate_train": prior,
    }
    for split in ("valid", "test"):
        rows = frame[frame.split == split]
        scores = rows.from_account_key.map(by_account["rate"]).fillna(prior).to_numpy()
        labels = rows.is_laundering.to_numpy(np.int8)
        entry = {
            "rows": int(len(rows)),
            "positives": int(labels.sum()),
            "ap": metrics.pr_auc(labels, scores),
            "seen_positive_account_row_share": float(rows.from_account_key.isin(positive_accounts).mean()),
        }
        for budget in (0.005, 0.01):
            precision, recall = metrics.precision_recall_at_budget(labels, scores, budget)
            entry[f"precision_at_{budget * 100:g}pct"] = precision
            entry[f"recall_at_{budget * 100:g}pct"] = recall
        report[split] = entry
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--processed-dir", type=Path, default=REPO_ROOT / "Portion1" / "processed")
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smooth", type=float, default=20.0)
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "Portion3" / "outputs" / "account_history_diagnostic.json")
    args = parser.parse_args(argv)

    report = diagnose(args.processed_dir, args.fraction, args.seed, args.smooth)
    print(
        f"fraction {args.fraction:.0%}, seed {args.seed}: "
        f"{report['labelled_positives']} positives on {report['labelled_positive_accounts']} accounts"
    )
    for split in ("valid", "test"):
        entry = report[split]
        print(
            f"  {split}: AP {entry['ap']:.4f} ({entry['ap'] / (entry['positives'] / entry['rows']):.1f}x)  "
            f"P@1% {entry['precision_at_1pct']:.4f}  R@1% {entry['recall_at_1pct']:.4f}  "
            f"rows from labelled-positive accounts {entry['seen_positive_account_row_share']:.3f}"
        )

    previous = []
    if args.out.exists():
        try:
            previous = json.loads(args.out.read_text())
            if isinstance(previous, dict):
                previous = [previous]
        except json.JSONDecodeError:
            previous = []
    previous = [item for item in previous if not (
        abs(item.get("fraction", -1) - args.fraction) < 1e-12 and item.get("seed") == args.seed
    )]
    previous.append(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(previous, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
