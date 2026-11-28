"""Run the Portion 3 fine-tuning grid, reusing Portion 2's ``finetune`` unchanged.

This is the agent-friendly equivalent of notebook section 7 in Portion 2. It
needs PyTorch and a GPU (or CPU, which is far too slow). Finished runs are
skipped, so the grid is resumable across sessions.

    python Portion3/run_finetune.py \
        --checkpoint-dir /path/to/portion2_runs \
        --out-dir /path/to/portion2_runs

The checkpoint directory must contain ``<run>/checkpoint.pt`` for
``main_seed42``, ``main_seed43``, ``main_seed44``, ``abl_whole_mask_seed42`` and
``abl_no_dt_seed42``. Outputs go to ``<out-dir>/finetune/ft_<run>_<pct>pct_seed<seed>/``
(scores.csv, finetune_log.csv, summary.json), exactly where
``portion3_baselines score-finetune`` expects them.

Afterwards, on the machine that holds results/runs.csv:

    python -m Portion3.portion3_baselines score-finetune --scores-root <out-dir>/finetune
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "Portion2"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:  # normal path: python -m Portion3.run_finetune
    from . import runplan
except ImportError:  # direct path: python Portion3/run_finetune.py
    import runplan  # type: ignore[no-redef]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-dir", type=Path, required=True,
                        help="directory holding <run>/checkpoint.pt (the shared Drive folder)")
    parser.add_argument("--processed-dir", type=Path, default=REPO_ROOT / "Portion1" / "processed")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="writes <out-dir>/finetune/... (usually the same as --checkpoint-dir)")
    parser.add_argument("--runs", nargs="+", default=None, help="subset of run ids (default: all)")
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--device", default="auto", help="auto | cuda | cpu")
    parser.add_argument("--force", action="store_true", help="rerun runs that already have scores.csv")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runs = [run for run in runplan.finetune_runs() if run.run_id not in runplan.EXISTING_FINETUNE_RUNS]
    if args.runs:
        wanted = set(args.runs)
        unknown = wanted - {run.run_id for run in runplan.all_runs()}
        if unknown:
            raise SystemExit(f"unknown run ids: {sorted(unknown)}")
        runs = [run for run in runs if run.run_id in wanted]
    print(f"{len(runs)} fine-tuning run(s), out: {Path(args.out_dir) / 'finetune'}")
    for run in runs:
        print(f"  {run.run_id:<44} {run.pretrain_run:<24} {run.fraction:.0%} seed {run.seed}")
    if args.dry_run:
        return 0

    try:
        import torch
    except ImportError:
        raise SystemExit(
            "PyTorch is not installed. For an RTX 50-series card:\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cu128\n"
            "For Colab, use the preinstalled CUDA build."
        )
    import portion2_encoder as enc

    if args.device == "auto":
        device = enc.pick_device()
    else:
        device = torch.device(args.device)
    print(f"device: {device}")

    finetune_dir = Path(args.out_dir) / "finetune"
    finetune_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    for index, run in enumerate(runs, start=1):
        finished = finetune_dir / run.run_id / "scores.csv"
        if finished.exists() and not args.force:
            print(f"[{index}/{len(runs)}] skip (already done): {run.run_id}")
            continue
        checkpoint = Path(args.checkpoint_dir) / run.pretrain_run / "checkpoint.pt"
        if not checkpoint.exists():
            raise SystemExit(f"missing checkpoint: {checkpoint}")
        print(f"\n[{index}/{len(runs)}] {run.run_id}", flush=True)
        enc.finetune(
            checkpoint,
            args.processed_dir,
            finetune_dir,
            fraction=run.fraction,
            seed=run.seed,
            max_epochs=args.max_epochs,
            device=device,
        )
    elapsed = time.perf_counter() - start
    print(f"\nfinished in {elapsed / 60:.1f} min")

    summaries = []
    for folder in sorted(finetune_dir.glob("ft_*/summary.json")):
        summaries.append(json.loads(folder.read_text()))
    if summaries:
        print(pd.DataFrame(summaries).to_string(index=False))
    print("\nNext: python -m Portion3.portion3_baselines score-finetune --scores-root", finetune_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
