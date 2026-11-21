"""GPU smoke tests: run every Portion 2 step on CUDA at real batch sizes. Skipped without a GPU.

CPU tests cannot catch GPU-only failures such as the CUDA attention limit hit by the export at batch 2048.
"""

import numpy as np
import pandas as pd
import pytest
import torch

import portion2_data as pdata
import portion2_encoder as enc
from conftest import PROCESSED_DIR, tiny_checkpoint

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
CUDA = torch.device("cuda")


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    # The adopted 4.29M model, with mixed precision as in the real runs.
    return tiny_checkpoint(tmp_path_factory.mktemp("ckpt"), device=CUDA, amp=True, batch_size=512,
                           windows_per_epoch=2048, encoder={"d_model": 256, "seq_layers": 4, "ff": 1024})


def test_gpu_export_at_full_batch(checkpoint, tmp_path):
    emb_path, _ = enc.export_embeddings(checkpoint, PROCESSED_DIR, tmp_path, batch_size=2048,
                                        rows=np.arange(6000), device=CUDA)
    embeddings = np.load(emb_path)
    assert embeddings.shape == (6000, 512) and np.isfinite(embeddings).all()


def test_gpu_reconstruction(checkpoint, tmp_path):
    window_index = pdata.load_window_index(PROCESSED_DIR)
    rows = np.flatnonzero(window_index.split.to_numpy() == "valid")[:3000]
    scores = pd.read_csv(enc.reconstruction_scores(checkpoint, PROCESSED_DIR, tmp_path, rows=rows,
                                                   batch_size=1024, device=CUDA))
    assert len(scores) == 3000 and np.isfinite(scores.nll_sum).all()


def test_gpu_finetune(checkpoint, tmp_path):
    window_index = pdata.load_window_index(PROCESSED_DIR)
    split = window_index.split.to_numpy()
    positive = window_index.is_laundering.to_numpy() == 1
    valid_rows = np.concatenate([np.flatnonzero((split == "valid") & positive)[:20],
                                 np.flatnonzero((split == "valid") & ~positive)[:2000]])
    test_rows = np.flatnonzero(split == "test")[:500]
    path = enc.finetune(checkpoint, PROCESSED_DIR, tmp_path, fraction=0.01, seed=42, max_epochs=1,
                        max_negatives_per_epoch=5000, eval_rows={"valid": valid_rows, "test": test_rows}, device=CUDA)
    assert np.isfinite(pd.read_csv(path).score).all()
