"""Verify and slice the Portion 2 embedding files before LightGBM uses them.

Portion 2 exports ``embeddings_<run>.npy`` (float16, 5,078,345 x 512) plus
``embeddings_<run>_ids.npy`` for five checkpoints on the shared Drive folder.
LightGBM only needs the first 256 columns (the target-transaction vector), which
halves both the download and the RAM. This module:

    python -m Portion3.prepare_embeddings check --source-dir DIR
    python -m Portion3.prepare_embeddings slice --source-dir DIR --out-dir DIR

``check`` fails loudly on a wrong row count, duplicate ids, ids that are not in
window_index, or non-finite values. ``slice`` writes
``embeddings_<run>_256.npy`` and ``embeddings_<run>_256_ids.npy`` in chunks, so it
never holds the full array in memory.

If your connection to Drive is slow, run the small Colab notebook
``Portion3_PrepareEmbeddings.ipynb`` first: it does the same slicing on Drive and
you download 13 GB instead of 26 GB.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = ["main_seed42", "main_seed43", "main_seed44", "abl_whole_mask_seed42", "abl_no_dt_seed42"]


def window_ids(processed_dir: Path) -> np.ndarray:
    frame = pd.read_csv(Path(processed_dir) / "window_index.csv.gz", usecols=["transaction_id"])
    return frame.transaction_id.to_numpy(np.int64)


def check_one(emb_path: Path, ids_path: Path, expected_ids: np.ndarray, sample_rows: int = 10_000) -> dict:
    array = np.load(Path(emb_path), mmap_mode="r")
    ids = np.load(Path(ids_path))
    report: dict = {
        "file": str(emb_path),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "rows": int(array.shape[0]),
        "ids": int(len(ids)),
    }
    if array.shape[0] != len(ids):
        raise ValueError(f"{emb_path.name}: {array.shape[0]:,} rows but {len(ids):,} ids")
    if len(expected_ids) != len(ids):
        raise ValueError(f"{emb_path.name}: {len(ids):,} ids but window_index has {len(expected_ids):,} rows")
    if not pd.Index(ids).is_unique:
        duplicates = int(len(ids) - len(pd.Index(ids).unique()))
        raise ValueError(f"{emb_path.name}: {duplicates:,} duplicate transaction ids")
    missing = np.setdiff1d(expected_ids, ids, assume_unique=True)
    extra = np.setdiff1d(ids, expected_ids, assume_unique=True)
    if len(missing) or len(extra):
        raise ValueError(
            f"{emb_path.name}: id set differs from window_index ({len(missing):,} missing, {len(extra):,} extra)"
        )
    report["same_order_as_window_index"] = bool(np.array_equal(ids, expected_ids))
    sample = np.asarray(array[: min(sample_rows, len(ids))])
    if not np.isfinite(sample).all():
        raise ValueError(f"{emb_path.name}: non-finite values in the first {len(sample):,} rows")
    report["finite_sample_rows"] = int(len(sample))
    return report


def slice_one(
    emb_path: Path,
    ids_path: Path,
    out_dir: Path,
    run_name: str,
    columns: int = 256,
    chunk_rows: int = 250_000,
) -> tuple[Path, Path]:
    source = np.load(Path(emb_path), mmap_mode="r")
    if columns > source.shape[1]:
        raise ValueError(f"{emb_path.name}: asked for {columns} columns but the file has {source.shape[1]}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_emb = out_dir / f"embeddings_{run_name}_256.npy"
    out_ids = out_dir / f"embeddings_{run_name}_256_ids.npy"
    target = np.lib.format.open_memmap(out_emb, mode="w+", dtype=np.float16, shape=(source.shape[0], columns))
    for start in range(0, source.shape[0], chunk_rows):
        stop = min(start + chunk_rows, source.shape[0])
        target[start:stop] = source[start:stop, :columns]
    target.flush()
    del target
    np.save(out_ids, np.load(Path(ids_path)))
    return out_emb, out_ids


def _resolve(source_dir: Path, run_name: str) -> tuple[Path, Path]:
    for folder in (Path(source_dir) / run_name, Path(source_dir)):
        for suffix in ("", "_256"):
            emb = folder / f"embeddings_{run_name}{suffix}.npy"
            ids = folder / f"embeddings_{run_name}{suffix}_ids.npy"
            if emb.exists() and ids.exists():
                return emb, ids
    raise FileNotFoundError(f"no embeddings for {run_name} in {source_dir}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="verify row counts, id coverage and finiteness")
    check.add_argument("--source-dir", type=Path, required=True)
    check.add_argument("--processed-dir", type=Path, default=REPO_ROOT / "Portion1" / "processed")
    check.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)

    slice_cmd = sub.add_parser("slice", help="write the 256-column copies used by LightGBM")
    slice_cmd.add_argument("--source-dir", type=Path, required=True)
    slice_cmd.add_argument("--out-dir", type=Path, required=True)
    slice_cmd.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    slice_cmd.add_argument("--columns", type=int, default=256)
    slice_cmd.add_argument("--processed-dir", type=Path, default=REPO_ROOT / "Portion1" / "processed")
    args = parser.parse_args(argv)

    expected = window_ids(args.processed_dir)
    reports = []
    for run_name in args.runs:
        emb_path, ids_path = _resolve(args.source_dir, run_name)
        report = check_one(emb_path, ids_path, expected)
        report["run"] = run_name
        reports.append(report)
        print(
            f"{run_name}: {report['rows']:,} x {report['shape'][1]} {report['dtype']}, "
            f"ids match window_index: yes, same order: {report['same_order_as_window_index']}"
        )

    if args.command == "slice":
        for run_name in args.runs:
            emb_path, ids_path = _resolve(args.source_dir, run_name)
            out_emb, out_ids = slice_one(emb_path, ids_path, args.out_dir, run_name, columns=args.columns)
            print(f"sliced {run_name}: {out_emb} ({out_emb.stat().st_size / 1e9:.2f} GB)")
        manifest = Path(args.out_dir) / "embedding_manifest.json"
        manifest.write_text(json.dumps({
            "columns": args.columns,
            "source_dir": str(Path(args.source_dir).resolve()),
            "runs": reports,
        }, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
