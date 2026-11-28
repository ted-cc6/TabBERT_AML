"""Measure the pass-through pattern behind the fine-tuned encoder's scores.

Pass-through here means: within the account's last N events before the target, the
account received money in the same amount bucket that the target sends. The fine-tune
score concentrates on these rows (HANDOFF.md section 12).

    python Portion3/diagnose_pass_through.py
    python Portion3/diagnose_pass_through.py --scores C:/path/to/scores.csv

Without --scores it reports label-only statistics (prevalence, PR AUC of the flag).
With --scores it also reports how many pass-through positives land in the top rows, how
many other positives do, and per-typology recall at the 1% budget on test.

Writes Portion3/outputs/pass_through_diagnostic.json. This is a diagnostic: it does not
touch results/runs.csv.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from Portion3 import sequence_features  # noqa: E402


def _stats(frame: pd.DataFrame, flag: str) -> dict:
    y = frame.is_laundering.to_numpy(np.int8)
    f = frame[flag].to_numpy(dtype=bool)
    return {
        "rows": int(len(frame)),
        "positives": int(y.sum()),
        "prevalence_positives": float(f[y == 1].mean()),
        "prevalence_negatives": float(f[y == 0].mean()),
        "pr_auc_of_flag": float(average_precision_score(y, f.astype(np.int8))),
        "flag_positive_count": int(f.sum()),
    }


def _capture(frame: pd.DataFrame, flag: str, k: int) -> dict:
    order = np.argsort(-frame.score.to_numpy(), kind="stable")[:k]
    top = np.zeros(len(frame), dtype=bool)
    top[order] = True
    y = frame.is_laundering.to_numpy(np.int8)
    f = frame[flag].to_numpy(dtype=bool)
    out = {"k": int(k), "top_positives": int(y[top].sum()), "top_precision": float(y[top].mean())}
    for name, group in (("pass", f), ("other", ~f)):
        positives = int(((y == 1) & group).sum())
        captured = int(((y == 1) & group & top).sum())
        out[f"{name}_positives"] = positives
        out[f"{name}_captured"] = captured
        out[f"{name}_recall"] = captured / positives if positives else 0.0
    out["top_share_with_flag"] = float(f[top].mean())
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--processed-dir", type=Path, default=REPO_ROOT / "Portion1" / "processed")
    parser.add_argument("--scores", type=Path, default=None, help="a fine-tune scores.csv (optional)")
    parser.add_argument("--recent", type=int, default=5)
    parser.add_argument("--flag", default="pass_recent", choices=sequence_features.FEATURE_NAMES)
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "Portion3" / "outputs" / "pass_through_diagnostic.json")
    args = parser.parse_args(argv)

    windows = pd.read_csv(
        Path(args.processed_dir) / "window_index.csv.gz",
        usecols=["transaction_id", "split", "is_laundering", "typology",
                 "target_event_index", "window_start_index"],
        dtype={"transaction_id": "int64", "split": "string", "is_laundering": "int8", "typology": "string"},
    )
    windows["typology"] = windows["typology"].fillna("")
    rows = windows[windows.split.isin(["valid", "test"])].reset_index(drop=True)
    tokens = np.load(Path(args.processed_dir) / "event_tokens.npz")["tokens"]
    built = sequence_features.build_features(
        tokens, rows.window_start_index.to_numpy(), rows.target_event_index.to_numpy(), recent=args.recent
    )
    frame = rows.drop(columns=["target_event_index", "window_start_index"]).copy()
    for name, values in built.items():
        frame[name] = values

    report: dict = {
        "definition": f"{args.flag}: received event with the target's amount bucket within the last "
                      f"{args.recent} events",
        "recent": args.recent,
        "splits": {},
    }
    for split in ("valid", "test"):
        part = frame[frame.split == split]
        report["splits"][split] = {"flag_stats": _stats(part, args.flag)}

    if args.scores is not None:
        scores = pd.read_csv(args.scores)
        merged = frame.merge(scores[["transaction_id", "split", "score"]], on="transaction_id",
                             how="left", suffixes=("", "_scores"))
        if merged.score.isna().any():
            raise SystemExit("some window rows have no score; check the score file")
        if not (merged.split == merged.split_scores).all():
            raise SystemExit("score split column disagrees with window_index")
        valid = merged[merged.split == "valid"]
        test = merged[merged.split == "test"]
        report["splits"]["valid"]["capture"] = [
            _capture(valid, args.flag, 1000),
            _capture(valid, args.flag, int(np.ceil(0.01 * len(valid)))),
        ]
        order = np.argsort(-test.score.to_numpy(), kind="stable")
        k = int(np.ceil(0.01 * len(test)))
        top = np.zeros(len(test), dtype=bool)
        top[order[:k]] = True
        typologies = {}
        for name in sorted(set(test.typology) - {""}):
            mask = (test.typology == name) & (test.is_laundering == 1)
            typologies[name] = {
                "positives": int(mask.sum()),
                "recall_at_1pct": float((mask.to_numpy() & top).sum() / max(mask.sum(), 1)),
            }
        report["per_typology_recall_at_1pct"] = typologies

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    for split, entry in report["splits"].items():
        stats = entry["flag_stats"]
        print(
            f"{split}: {stats['positives']} positives, flag prevalence "
            f"{stats['prevalence_positives']:.3f} (positives) / {stats['prevalence_negatives']:.4f} "
            f"(negatives), PR AUC {stats['pr_auc_of_flag']:.5f}"
        )
        for capture in entry.get("capture", []):
            print(
                f"  top {capture['k']}: {capture['top_positives']} positives, precision "
                f"{capture['top_precision']:.3f}; pass recall {capture['pass_recall']:.3f}, "
                f"other recall {capture['other_recall']:.3f}; top rows with flag "
                f"{capture['top_share_with_flag']:.3f}"
            )
    if "per_typology_recall_at_1pct" in report:
        print("  test recall@1% per typology:")
        for name, entry in report["per_typology_recall_at_1pct"].items():
            print(f"    {name:<15} n {entry['positives']:>4}  recall {entry['recall_at_1pct']:.3f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
