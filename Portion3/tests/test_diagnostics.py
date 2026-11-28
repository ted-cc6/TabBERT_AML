from __future__ import annotations

import numpy as np

from Portion3 import sequence_features


def _window(events):
    W = np.zeros((1, 32, 6), dtype=np.int64)
    M = np.zeros((1, 32), dtype=bool)
    length = len(events)
    W[0, 32 - length:] = events
    M[0, 32 - length:] = True
    return W, M


def test_pass_through_flags():
    W, M = _window([
        [1, 1, 5, 4, 10, 4],   # received, same bucket, three events before target
        [1, 1, 7, 3, 5, 4],    # sent, different bucket
        [1, 1, 7, 3, 10, 4],   # target
    ])
    features = sequence_features.window_features(W, M)
    assert features["pass_any"][0] == 1
    assert features["pass_recent"][0] == 1
    assert features["pass_prev"][0] == 0
    assert features["window_length"][0] == 3
    assert features["distinct_banks"][0] == 2
    assert features["amount_repeat"][0] == 1
    assert features["sent_share"][0] == 2 / 3
    assert features["zero_gap_share"][0] == 1
    assert features["gap_since_received"][0] == 2

    W, M = _window([
        [1, 1, 7, 4, 10, 4],
        [1, 1, 7, 3, 10, 4],
    ])
    features = sequence_features.window_features(W, M)
    assert features["pass_prev"][0] == 1
    assert features["recent_received"][0] == 1
    assert features["gap_since_received"][0] == 1

    W, M = _window([[1, 1, 7, 3, 9, 4]])
    features = sequence_features.window_features(W, M)
    assert features["pass_any"][0] == 0
    assert features["gap_since_received"][0] == 63
    assert features["window_length"][0] == 1


def test_build_features_chunking_matches_full():
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, 6, size=(20, 6), dtype=np.int32)
    starts = np.array([1, 2, 3, 4])
    targets = np.array([5, 6, 7, 8])
    full = sequence_features.build_features(tokens, starts, targets, chunk_rows=100)
    chunked = sequence_features.build_features(tokens, starts, targets, chunk_rows=1)
    for name in sequence_features.FEATURE_NAMES:
        assert np.array_equal(full[name], chunked[name]), name
