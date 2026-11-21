"""Checks for the pretraining loop, metrics and checkpoints, on tiny CPU runs."""

import dataclasses

import numpy as np
import pandas as pd
import pytest
import torch

import portion2_data as pdata
import portion2_encoder as enc
from conftest import PORTION2_DIR, PROCESSED_DIR

TINY = dict(windows_per_epoch=1024, epochs=1, valid_windows=256, batch_size=256, warmup_steps=2, amp=False)


def tiny_config(name="tiny", **overrides):
    cfg = enc.load_run_config(PORTION2_DIR / "configs" / "smoke_cpu.json")
    return dataclasses.replace(cfg, run_name=name, **{**TINY, **overrides})


def toy_specs():
    return [
        enc.FieldSpec("payment_format", 10, 3, 9),
        enc.FieldSpec("direction", 5, 3, 4),
        enc.FieldSpec("amount_bucket", 68, 4, 67),
        enc.FieldSpec("time_gap_bucket", 36, 4, 35),
    ]


def one_hot_logits(predictions, size):
    return torch.nn.functional.one_hot(torch.tensor(predictions), size).float()


def test_field_metrics_known_values():
    specs = toy_specs()
    baselines = {"payment_format": 6, "direction": 3, "amount_bucket": 10, "time_gap_bucket": 4}
    # One window of length 4; field f is chosen at the positions listed below.
    chosen = torch.zeros(1, 4, 4, dtype=torch.bool)
    chosen[0, [0, 1, 2], 0] = True   # payment_format: targets 6, 6, 7; predictions 6, 7, 7
    chosen[0, [2, 3], 1] = True      # direction: position 3 is the last row and is ignored
    chosen[0, [0, 1, 2, 3], 2] = True  # amount: exact, off by one, off by one, off by five
    chosen[0, [0, 1, 2], 3] = True   # time gap: targets 4 (gap 0), 9, 20
    outputs = [
        (one_hot_logits([6, 7, 7], 10), torch.tensor([6, 6, 7])),
        (one_hot_logits([4, 3], 5), torch.tensor([3, 3])),
        (one_hot_logits([10, 11, 9, 15], 68), torch.tensor([10, 10, 10, 10])),
        (one_hot_logits([4, 9, 30], 36), torch.tensor([4, 9, 20])),
    ]

    metrics = enc.FieldMetrics(specs, baselines)
    metrics.update(outputs, chosen)
    result = metrics.result()

    assert result["acc_payment_format"] == pytest.approx(2 / 3)
    assert result["baseline_payment_format"] == pytest.approx(2 / 3)
    assert result["acc_direction"] == 0.0  # only position 2 counts; predicted 4, true 3
    assert result["n_direction"] == 1
    assert result["acc_amount_bucket"] == pytest.approx(1 / 4)
    assert result["acc1_amount_bucket"] == pytest.approx(3 / 4)
    assert result["baseline_amount_bucket"] == pytest.approx(1.0)
    assert result["acc_time_gap_bucket"] == pytest.approx(2 / 3)
    assert result["acc_time_gap_bucket_nonzero"] == pytest.approx(1 / 2)
    assert result["baseline_time_gap_bucket"] == pytest.approx(1 / 3)


def test_fit_baselines():
    specs = enc.load_field_specs(PROCESSED_DIR)
    tokens = np.array([[6, 3, 5, 3, 10, 3], [6, 3, 7, 4, 10, 4], [7, 4, 7, 3, 11, 4], [9, 9, 9, 4, 12, 30]])
    baselines = enc.fit_baselines(tokens, np.array(["train", "train", "train", "valid"]), specs)
    assert baselines == {
        "payment_format": 6, "currency": 3, "counterparty_bank": 7,
        "direction": 3, "amount_bucket": 10, "time_gap_bucket": 4,
    }


def test_lr_schedule():
    assert enc.lr_factor(0, warmup_steps=10, total_steps=100) == pytest.approx(0.1)
    assert enc.lr_factor(9, warmup_steps=10, total_steps=100) == pytest.approx(1.0)
    assert enc.lr_factor(99, warmup_steps=10, total_steps=100) == pytest.approx(0.0, abs=1e-3)


def test_configs_load():
    names = {p.stem for p in (PORTION2_DIR / "configs").glob("*.json")}
    assert names == {"smoke_cpu", "main_seed42", "main_seed43", "main_seed44",
                     "abl_whole_mask_seed42", "abl_no_dt_seed42"}
    for name in names:
        cfg = enc.load_run_config(PORTION2_DIR / "configs" / f"{name}.json")
        assert cfg.run_name == name
        assert cfg.masking in ("field", "event")
    assert enc.load_run_config(PORTION2_DIR / "configs" / "abl_no_dt_seed42.json").use_dt is False
    assert enc.load_run_config(PORTION2_DIR / "configs" / "abl_whole_mask_seed42.json").masking == "event"


def test_pretrain_smoke_reproducible(tmp_path):
    first = enc.pretrain(tiny_config(), PROCESSED_DIR, tmp_path / "a", device=torch.device("cpu"))
    second = enc.pretrain(tiny_config(), PROCESSED_DIR, tmp_path / "b", device=torch.device("cpu"))

    metrics_a = pd.read_csv(first.parent / "metrics.csv").drop(columns="seconds")
    metrics_b = pd.read_csv(second.parent / "metrics.csv").drop(columns="seconds")
    pd.testing.assert_frame_equal(metrics_a, metrics_b)
    assert {"train_loss", "valid_loss", "acc_counterparty_bank", "baseline_counterparty_bank",
            "acc1_amount_bucket", "acc_time_gap_bucket_nonzero"} <= set(metrics_a.columns)
    assert (first.parent / "amount_embedding_cosine.png").exists()

    model, specs, cfg = enc.load_encoder(first, torch.device("cpu"))
    assert cfg.run_name == "tiny" and len(specs) == 6
    saved = torch.load(first, weights_only=False)["state_dict"]
    assert all(torch.equal(saved[k], v) for k, v in model.state_dict().items())


@pytest.mark.parametrize("overrides", [dict(masking="event"), dict(use_dt=False)])
def test_pretrain_ablation_variants_run(tmp_path, overrides):
    ckpt = enc.pretrain(tiny_config(**overrides), PROCESSED_DIR, tmp_path, device=torch.device("cpu"))
    model, specs, cfg = enc.load_encoder(ckpt, torch.device("cpu"))
    assert len(specs) == (5 if overrides.get("use_dt") is False else 6)
    assert np.isfinite(pd.read_csv(ckpt.parent / "metrics.csv").valid_loss).all()


def test_pretrain_never_reads_non_train_rows(tmp_path, monkeypatch):
    calls = []
    original = pdata.sample_epoch_rows

    def recording(window_index, split, n, seed, epoch):
        calls.append((split, n))
        return original(window_index, split, n, seed, epoch)

    monkeypatch.setattr(pdata, "sample_epoch_rows", recording)
    enc.pretrain(tiny_config(epochs=2), PROCESSED_DIR, tmp_path, device=torch.device("cpu"))
    assert calls == [("valid", 256), ("train", 1024), ("train", 1024)]
