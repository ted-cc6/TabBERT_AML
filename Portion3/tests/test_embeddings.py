from __future__ import annotations

import numpy as np
import pytest

from Portion3 import prepare_embeddings
from Portion3 import portion3_baselines as pb


def _write_synthetic_embeddings(directory, n=50, columns=512, seed=0):
    rng = np.random.default_rng(seed)
    array = rng.standard_normal((n, columns)).astype(np.float16)
    array[:, 0] = np.arange(n)  # row marker in column 0
    ids = np.arange(5000, 5000 + n)
    rng.shuffle(ids)
    emb_path = directory / "embeddings_main_seed42.npy"
    ids_path = directory / "embeddings_main_seed42_ids.npy"
    np.save(emb_path, array)
    np.save(ids_path, ids)
    return emb_path, ids_path, array, ids


def test_embedding_store_joins_in_requested_order(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    emb_path, ids_path, array, ids = _write_synthetic_embeddings(tmp_path)
    store = pb.EmbeddingStore(emb_path, ids_path, columns=256)
    requested = np.array([ids[3], ids[0], ids[7]])
    block = store.matrix(requested)
    assert block.shape == (3, 256)
    assert block.dtype == np.float32
    assert np.allclose(block[:, 0], [3, 0, 7])


def test_embedding_store_rejects_unknown_ids(tmp_path):
    emb_path, ids_path, _, _ = _write_synthetic_embeddings(tmp_path)
    store = pb.EmbeddingStore(emb_path, ids_path, columns=256)
    with pytest.raises(KeyError):
        store.matrix(np.array([1]))


def test_embedding_paths_prefer_sliced_copy(tmp_path):
    full = tmp_path / "embeddings_main_seed42.npy"
    full_ids = tmp_path / "embeddings_main_seed42_ids.npy"
    sliced = tmp_path / "embeddings_main_seed42_256.npy"
    sliced_ids = tmp_path / "embeddings_main_seed42_256_ids.npy"
    for path in (full, full_ids):
        path.write_bytes(b"")
    np.save(sliced, np.zeros((2, 256), dtype=np.float16))
    np.save(sliced_ids, np.arange(2))
    emb, ids = pb.embedding_paths(tmp_path, "main_seed42")
    assert emb == sliced and ids == sliced_ids


def test_embedding_paths_accept_drive_layout(tmp_path):
    folder = tmp_path / "main_seed42"
    folder.mkdir()
    np.save(folder / "embeddings_main_seed42.npy", np.zeros((2, 512), dtype=np.float16))
    np.save(folder / "embeddings_main_seed42_ids.npy", np.arange(2))
    emb, ids = pb.embedding_paths(tmp_path, "main_seed42")
    assert emb == folder / "embeddings_main_seed42.npy"
    assert prepare_embeddings._resolve(tmp_path, "main_seed42")[0] == emb


def test_prepare_check_and_slice(tmp_path):
    source = tmp_path / "source"
    out = tmp_path / "out"
    source.mkdir()
    emb_path, ids_path, array, ids = _write_synthetic_embeddings(source)
    report = prepare_embeddings.check_one(emb_path, ids_path, np.sort(ids))
    assert report["rows"] == 50
    assert report["same_order_as_window_index"] is False

    out_emb, out_ids = prepare_embeddings.slice_one(emb_path, ids_path, out, "main_seed42")
    sliced = np.load(out_emb)
    assert sliced.shape == (50, 256)
    assert sliced.dtype == np.float16
    assert np.array_equal(sliced[:, 0].astype(int), array[:, 0].astype(int))
    assert np.array_equal(np.load(out_ids), ids)


def test_prepare_check_fails_on_missing_id(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    emb_path, ids_path, _, ids = _write_synthetic_embeddings(source)
    with pytest.raises(ValueError):
        prepare_embeddings.check_one(emb_path, ids_path, np.append(np.sort(ids), 999_999))
