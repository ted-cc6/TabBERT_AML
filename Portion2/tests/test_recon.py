"""Checks for the zero-shot reconstruction score (Task 7) on a tiny checkpoint."""

import numpy as np
import pandas as pd
import pytest
import torch

import portion2_data as pdata
import portion2_encoder as enc
from conftest import PROCESSED_DIR, tiny_checkpoint

FIELDS = ["payment_format", "currency", "counterparty_bank", "amount_bucket", "time_gap_bucket"]


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    return tiny_checkpoint(tmp_path_factory.mktemp("ckpt"))


@pytest.fixture(scope="module")
def window_index():
    return pdata.load_window_index(PROCESSED_DIR)


def valid_rows(window_index, n=500):
    return np.flatnonzero(window_index.split.to_numpy() == "valid")[:n]


def test_recon_columns_and_values(checkpoint, window_index, tmp_path):
    rows = valid_rows(window_index)
    path = enc.reconstruction_scores(checkpoint, PROCESSED_DIR, tmp_path, rows=rows, batch_size=128)
    assert path.name == "recon_tiny.csv"
    scores = pd.read_csv(path)
    assert list(scores.columns) == ["transaction_id", "split"] + [f"nll_{f}" for f in FIELDS] + ["nll_sum"]
    np.testing.assert_array_equal(scores.transaction_id, window_index.transaction_id.to_numpy()[rows])
    assert (scores.split == "valid").all()
    nll = scores[[f"nll_{f}" for f in FIELDS]]
    assert np.isfinite(nll.to_numpy()).all() and (nll.to_numpy() >= 0).all()
    np.testing.assert_allclose(scores.nll_sum, nll.sum(axis=1), rtol=1e-5)


def test_recon_default_rows_cover_requested_splits(checkpoint, window_index, tmp_path, monkeypatch):
    seen = {}
    original = pdata.iterate_batches

    def recording(tokens, wi, rows, batch_size):
        seen["rows"] = rows
        return original(tokens, wi, rows[:64], batch_size)  # score only a few rows

    monkeypatch.setattr(pdata, "iterate_batches", recording)
    enc.reconstruction_scores(checkpoint, PROCESSED_DIR, tmp_path, splits=("tail",))
    assert len(seen["rows"]) == 1_108
    assert (window_index.split.to_numpy()[seen["rows"]] == "tail").all()


def test_recon_masks_only_target_field(checkpoint, window_index, tmp_path, monkeypatch):
    rows = valid_rows(window_index, 16)
    inputs = []
    original_encode = enc.TabBERT.encode

    def recording(self, tokens, event_mask):
        inputs.append(tokens.cpu().clone())
        return original_encode(self, tokens, event_mask)

    monkeypatch.setattr(enc.TabBERT, "encode", recording)
    enc.reconstruction_scores(checkpoint, PROCESSED_DIR, tmp_path, rows=rows, batch_size=16)

    clean = next(pdata.iterate_batches(pdata.load_tokens(PROCESSED_DIR), window_index, rows, 16))["tokens"]
    field_index = [enc.TOKEN_FIELDS.index(f) for f in FIELDS]
    assert len(inputs) == len(FIELDS)
    for masked, f in zip(inputs, field_index):
        changed = masked != clean
        assert changed[:, -1, f].all() and (masked[:, -1, f] == enc.MASK).all()
        changed[:, -1, f] = False
        assert not changed.any()


def test_recon_no_dt_checkpoint(window_index, tmp_path):
    ckpt = tiny_checkpoint(tmp_path / "ckpt", use_dt=False, run_name="tiny_no_dt")
    path = enc.reconstruction_scores(ckpt, PROCESSED_DIR, tmp_path, rows=valid_rows(window_index, 100))
    assert "nll_time_gap_bucket" not in pd.read_csv(path).columns
