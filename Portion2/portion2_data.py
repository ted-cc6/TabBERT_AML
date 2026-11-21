"""Load causal account windows from the Portion 1 outputs for the Portion 2 encoder.

A window is tokens[window_start_index : target_event_index + 1], left-padded with PAD=0 to 32 events,
so the target event always sits in the last row. event_mask is True for a real event and False for padding.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import torch

MAX_LEN = 32
PAD = 0


def load_tokens(processed_dir: Path) -> np.ndarray:
    """Return the (10,156,690, 6) int32 token matrix; row = event_index."""
    return np.load(Path(processed_dir) / "event_tokens.npz")["tokens"]


def load_window_index(processed_dir: Path) -> pd.DataFrame:
    """Return window_index.csv.gz with all columns, in stored row order."""
    # An empty typology means "not in the pattern file"; keep it as "" rather than NaN.
    return pd.read_csv(Path(processed_dir) / "window_index.csv.gz", keep_default_na=False)


def make_window(tokens: np.ndarray, start: int, target: int, max_len: int = MAX_LEN) -> tuple[np.ndarray, np.ndarray]:
    """Return (window (max_len, 6) int64, event_mask (max_len,) bool) for one transaction."""
    events = tokens[start:target + 1]
    n_real = len(events)
    window = np.full((max_len, tokens.shape[1]), PAD, dtype=np.int64)
    window[max_len - n_real:] = events
    event_mask = np.arange(max_len) >= max_len - n_real
    return window, event_mask


def build_batch(tokens: np.ndarray, starts: np.ndarray, targets: np.ndarray, max_len: int = MAX_LEN) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised make_window: return (B, max_len, 6) int64 and (B, max_len) bool."""
    starts = np.asarray(starts, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.int64)
    # Row j of each window holds event target - (max_len - 1 - j); rows before the window start are padding.
    event_index = targets[:, None] - np.arange(max_len - 1, -1, -1)
    event_mask = event_index >= starts[:, None]
    batch = tokens[np.where(event_mask, event_index, 0)].astype(np.int64)
    batch[~event_mask] = PAD
    return batch, event_mask


def sample_epoch_rows(window_index: pd.DataFrame, split: str, n: int | None, seed: int, epoch: int) -> np.ndarray:
    """Return row positions of `split` windows; n=None gives all of them in stored order."""
    candidates = np.flatnonzero(window_index.split.to_numpy() == split)
    if n is None:
        return candidates
    rng = np.random.default_rng([seed, epoch])
    return rng.choice(candidates, size=n, replace=False)


def iterate_batches(tokens: np.ndarray, window_index: pd.DataFrame, rows: np.ndarray, batch_size: int) -> Iterator[dict]:
    """Yield {"tokens", "event_mask", "transaction_id"} tensors for `rows`, in order."""
    starts = window_index.window_start_index.to_numpy()
    targets = window_index.target_event_index.to_numpy()
    transaction_ids = window_index.transaction_id.to_numpy()
    for begin in range(0, len(rows), batch_size):
        chunk = rows[begin:begin + batch_size]
        batch, event_mask = build_batch(tokens, starts[chunk], targets[chunk])
        yield {
            "tokens": torch.from_numpy(batch),
            "event_mask": torch.from_numpy(event_mask),
            "transaction_id": torch.from_numpy(transaction_ids[chunk].astype(np.int64)),
        }
