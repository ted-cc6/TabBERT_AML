"""Window features built from the token matrix, for diagnostics only.

These are not part of the Experiment A feature set. They exist to measure how much of
the fine-tuned encoder's signal a handful of hand-made sequence features can reproduce,
in particular the pass-through pattern (money arrives, then leaves in the same amount
bucket within a few events).

All functions take the window matrix ``W`` (n, 32, 6) and mask ``M`` (n, 32) produced by
the same indexing rule as Portion 2's ``build_batch``: the target is always at position
31 and padding sits on the left.
"""

from __future__ import annotations

import numpy as np

MAX_LEN = 32
FEATURE_NAMES = [
    "pass_any",
    "pass_recent",
    "pass_prev",
    "window_length",
    "sent_share",
    "zero_gap_share",
    "distinct_banks",
    "distinct_amounts",
    "amount_repeat",
    "bank_repeat",
    "mean_gap_bucket",
    "target_amount_rank",
    "recent_received",
    "gap_since_received",
]


def _row_distinct(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Number of distinct non-padding values in each row (padding is -1)."""
    v = np.where(mask, values, -1)
    s = np.sort(v, axis=1)
    has_real = s[:, -1] >= 0
    previous, following = s[:, :-1], s[:, 1:]
    both_real = (previous >= 0) & (following >= 0)
    changes = ((previous != following) & both_real).sum(axis=1)
    return np.where(has_real, changes + 1, 0).astype(np.float32)


def window_features(W: np.ndarray, M: np.ndarray, recent: int = 5) -> dict[str, np.ndarray]:
    """Sequence features for one batch of windows; see FEATURE_NAMES."""
    n, length_axis, _ = W.shape
    length = M.sum(axis=1).astype(np.float32)
    safe = np.maximum(length, 1.0)
    direction = W[:, :, 3]
    amount = W[:, :, 4]
    bank = W[:, :, 2]
    gap = W[:, :, 5]

    target_amount = amount[:, -1]
    target_bank = bank[:, -1]
    received = M & (direction == 4)
    sent = M & (direction == 3)
    same_bucket = received & (amount == target_amount[:, None])

    recent_mask = np.zeros_like(M)
    recent_mask[:, -recent:] = True
    last_received_position = np.max(np.where(received, np.arange(length_axis), -1), axis=1)

    return {
        "pass_any": same_bucket.any(axis=1).astype(np.float32),
        "pass_recent": (same_bucket & recent_mask).any(axis=1).astype(np.float32),
        "pass_prev": same_bucket[:, -2].astype(np.float32) if length_axis > 1 else np.zeros(n, np.float32),
        "window_length": length,
        "sent_share": (sent.sum(axis=1) / safe).astype(np.float32),
        "zero_gap_share": ((M & (gap == 4)).sum(axis=1) / safe).astype(np.float32),
        "distinct_banks": _row_distinct(bank, M),
        "distinct_amounts": _row_distinct(amount, M),
        "amount_repeat": (M & (amount == target_amount[:, None])).sum(axis=1).astype(np.float32) - 1,
        "bank_repeat": (M & (bank == target_bank[:, None])).sum(axis=1).astype(np.float32) - 1,
        "mean_gap_bucket": (np.where(M, gap, 0).sum(axis=1) / safe).astype(np.float32),
        "target_amount_rank": (M & (amount <= target_amount[:, None])).sum(axis=1) / safe,
        "recent_received": (received & recent_mask).sum(axis=1).astype(np.float32),
        "gap_since_received": np.where(
            last_received_position >= 0, length_axis - 1 - last_received_position, 63
        ).astype(np.float32),
    }


def windows_for(tokens: np.ndarray, starts: np.ndarray, targets: np.ndarray, max_len: int = MAX_LEN):
    """Build the padded window matrix and mask for these target event indices."""
    positions = targets[:, None] - np.arange(max_len - 1, -1, -1)
    mask = positions >= starts[:, None]
    return tokens[np.where(mask, positions, 0)], mask


def build_features(
    tokens: np.ndarray,
    starts: np.ndarray,
    targets: np.ndarray,
    max_len: int = MAX_LEN,
    recent: int = 5,
    chunk_rows: int = 500_000,
) -> dict[str, np.ndarray]:
    """Features for every row, computed in chunks so the window matrix stays small."""
    parts: dict[str, list[np.ndarray]] = {name: [] for name in FEATURE_NAMES}
    for begin in range(0, len(targets), chunk_rows):
        stop = min(begin + chunk_rows, len(targets))
        W, M = windows_for(tokens, starts[begin:stop], targets[begin:stop], max_len)
        batch = window_features(W, M, recent=recent)
        for name in FEATURE_NAMES:
            parts[name].append(batch[name])
    return {name: np.concatenate(chunks) for name, chunks in parts.items()}
