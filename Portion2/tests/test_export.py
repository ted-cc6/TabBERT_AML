"""Checks for the frozen embedding export (Task 5) on a tiny CPU checkpoint."""

import numpy as np
import pytest
import torch

import portion2_data as pdata
import portion2_encoder as enc
from conftest import PROCESSED_DIR, tiny_checkpoint

N_ROWS = 3000


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    return tiny_checkpoint(tmp_path_factory.mktemp("ckpt"), run_name="tiny_export")


@pytest.fixture(scope="module")
def window_index():
    return pdata.load_window_index(PROCESSED_DIR)


@pytest.fixture(scope="module")
def exported(checkpoint, tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("export")
    return enc.export_embeddings(checkpoint, PROCESSED_DIR, out_dir, batch_size=512,
                                 rows=np.arange(N_ROWS), device=torch.device("cpu"))


def test_export_shapes_and_dtype(exported, window_index):
    emb_path, ids_path = exported
    assert emb_path.name == "embeddings_tiny_export.npy"
    assert ids_path.name == "embeddings_tiny_export_ids.npy"
    embeddings = np.load(emb_path)
    ids = np.load(ids_path)
    assert embeddings.shape == (N_ROWS, 256)
    assert embeddings.dtype == np.float16
    assert ids.dtype == np.int64
    np.testing.assert_array_equal(ids, window_index.transaction_id.to_numpy()[:N_ROWS])
    assert np.isfinite(embeddings).all()


def test_export_deterministic(checkpoint, exported, tmp_path):
    again, _ = enc.export_embeddings(checkpoint, PROCESSED_DIR, tmp_path, batch_size=1000,
                                     rows=np.arange(N_ROWS), device=torch.device("cpu"))
    np.testing.assert_array_equal(np.load(again), np.load(exported[0]))  # batch size does not matter


def test_export_matches_encode(checkpoint, exported, window_index):
    tokens = pdata.load_tokens(PROCESSED_DIR)
    model, specs, cfg = enc.load_encoder(checkpoint, torch.device("cpu"))
    row = int(np.flatnonzero(window_index.window_length.to_numpy()[:N_ROWS] == 5)[0])
    batch = next(pdata.iterate_batches(tokens, window_index, np.array([row]), batch_size=1))
    with torch.no_grad():
        hidden = model.encode(enc.select_fields(batch["tokens"], cfg.use_dt), batch["event_mask"])[0]
    expected = torch.cat([hidden[-1], hidden[-5:].mean(0)]).numpy()  # target row, then mean of the 5 real rows
    np.testing.assert_allclose(np.load(exported[0])[row].astype(np.float32), expected, rtol=1e-2, atol=1e-2)


def test_export_no_dt_checkpoint(tmp_path):
    ckpt = tiny_checkpoint(tmp_path / "ckpt", use_dt=False, run_name="tiny_no_dt")
    emb_path, _ = enc.export_embeddings(ckpt, PROCESSED_DIR, tmp_path, rows=np.arange(500), device=torch.device("cpu"))
    assert np.load(emb_path).shape == (500, 256)
