"""Checks for the Portion 2 encoder and masking on real windows from the Portion 1 outputs."""

import numpy as np
import pytest
import torch

import portion2_data as pdata
import portion2_encoder as enc
from conftest import PROCESSED_DIR

SIZES = [10, 18, 30531, 5, 68, 36]
REAL_LOW = [3, 3, 3, 3, 4, 4]
REAL_HIGH = [9, 17, 30530, 4, 67, 35]


@pytest.fixture(scope="module")
def data():
    return pdata.load_tokens(PROCESSED_DIR), pdata.load_window_index(PROCESSED_DIR)


@pytest.fixture(scope="module")
def specs():
    return enc.load_field_specs(PROCESSED_DIR)


def real_batch(data, n, seed=0, max_length=None):
    tokens, window_index = data
    rows = pdata.sample_epoch_rows(window_index, "train", None, seed=0, epoch=0)
    if max_length is not None:
        rows = rows[window_index.window_length.to_numpy()[rows] <= max_length]
    rows = np.random.default_rng(seed).choice(rows, size=n, replace=False)
    batch = next(pdata.iterate_batches(tokens, window_index, rows, batch_size=n))
    return batch["tokens"], batch["event_mask"]


def test_field_specs(specs):
    assert [s.name for s in specs] == enc.TOKEN_FIELDS
    assert [s.size for s in specs] == SIZES
    assert [s.real_low for s in specs] == REAL_LOW
    assert [s.real_high for s in specs] == REAL_HIGH
    no_dt = enc.load_field_specs(PROCESSED_DIR, use_dt=False)
    assert [s.name for s in no_dt] == enc.TOKEN_FIELDS[:5]


def test_encode_shape(data, specs):
    tokens, event_mask = real_batch(data, 4)
    model = enc.TabBERT(specs, enc.EncoderConfig())
    assert model.encode(tokens, event_mask).shape == (4, 32, 128)

    no_dt_specs = enc.load_field_specs(PROCESSED_DIR, use_dt=False)
    no_dt_model = enc.TabBERT(no_dt_specs, enc.EncoderConfig(use_dt=False))
    no_dt_tokens = enc.select_fields(tokens, use_dt=False)
    assert no_dt_tokens.shape == (4, 32, 5)
    assert no_dt_model.encode(no_dt_tokens, event_mask).shape == (4, 32, 128)


def test_field_logits_shape(data, specs):
    tokens, event_mask = real_batch(data, 4)
    model = enc.TabBERT(specs, enc.EncoderConfig())
    hidden = model.encode(tokens, event_mask)
    for f, spec in enumerate(specs):
        assert model.field_logits(hidden[event_mask], f).shape == (int(event_mask.sum()), spec.size)


def test_param_count(specs):
    n_params = sum(p.numel() for p in enc.TabBERT(specs, enc.EncoderConfig()).parameters())
    assert 1_000_000 <= n_params <= 2_500_000


def test_padding_does_not_change_real_outputs(data, specs):
    tokens, event_mask = real_batch(data, 16, max_length=20)
    torch.manual_seed(0)
    model = enc.TabBERT(specs, enc.EncoderConfig()).eval()
    noisy = tokens.clone()
    noisy[~event_mask] = 3  # a real ID in every field, written into the padding rows
    with torch.no_grad():
        clean_out = model.encode(tokens, event_mask)
        noisy_out = model.encode(noisy, event_mask)
    torch.testing.assert_close(clean_out[event_mask], noisy_out[event_mask])


def test_mask_fields_rates(data, specs):
    tokens, event_mask = real_batch(data, 2048)
    generator = torch.Generator().manual_seed(0)
    masked, chosen = enc.mask_fields(tokens, event_mask, specs, generator=generator)

    real = event_mask[..., None].expand_as(tokens)
    assert 0.14 <= chosen[real].float().mean().item() <= 0.16
    became_mask = (masked == enc.MASK)[chosen].float().mean().item()
    unchanged = (masked == tokens)[chosen].float().mean().item()
    assert 0.77 <= became_mask <= 0.83
    assert 0.08 <= unchanged <= 0.13  # 10% kept, plus random draws that hit the original ID
    assert torch.equal(masked[~chosen], tokens[~chosen])


@pytest.mark.parametrize("mask_fn", [enc.mask_fields, enc.mask_events])
def test_mask_never_touches_pad(data, specs, mask_fn):
    tokens, event_mask = real_batch(data, 512, max_length=20)
    masked, chosen = mask_fn(tokens, event_mask, specs, generator=torch.Generator().manual_seed(0))
    assert not (chosen & ~event_mask[..., None]).any()
    assert (masked[~event_mask] == enc.PAD).all()


@pytest.mark.parametrize("mask_fn", [enc.mask_fields, enc.mask_events])
def test_random_replacement_in_range(data, specs, mask_fn):
    tokens, event_mask = real_batch(data, 2048)
    masked, chosen = mask_fn(tokens, event_mask, specs, generator=torch.Generator().manual_seed(1))
    replaced = chosen & (masked != enc.MASK) & (masked != tokens)
    assert replaced.any()
    for f, spec in enumerate(specs):
        ids = masked[..., f][replaced[..., f]]
        assert ((ids >= spec.real_low) & (ids <= spec.real_high)).all(), spec.name


def test_mask_events_whole_rows(data, specs):
    tokens, event_mask = real_batch(data, 2048)
    masked, chosen = enc.mask_events(tokens, event_mask, specs, generator=torch.Generator().manual_seed(2))

    chosen_events = chosen.any(-1)
    assert torch.equal(chosen_events, chosen.all(-1))
    assert 0.14 <= chosen_events[event_mask].float().mean().item() <= 0.16
    # One 80/10/10 draw per event: a chosen event is either fully MASK or has no MASK at all.
    mask_count = (masked == enc.MASK).sum(-1)[chosen_events]
    assert ((mask_count == 0) | (mask_count == tokens.shape[-1])).all()


@pytest.mark.parametrize("mask_fn", [enc.mask_fields, enc.mask_events])
def test_length_one_window_maskable(data, specs, mask_fn):
    tokens, window_index = data
    window, event_mask = pdata.make_window(tokens, 1, 1)
    batch = torch.from_numpy(window)[None].repeat(8, 1, 1)
    batch_mask = torch.from_numpy(event_mask)[None].repeat(8, 1)
    generator = torch.Generator().manual_seed(3)
    hits = sum(mask_fn(batch, batch_mask, specs, generator=generator)[1][:, 31].any().item() for _ in range(200))
    assert hits > 0


def test_masking_reproducible(data, specs):
    tokens, event_mask = real_batch(data, 64)
    first = enc.mask_fields(tokens, event_mask, specs, generator=torch.Generator().manual_seed(7))
    second = enc.mask_fields(tokens, event_mask, specs, generator=torch.Generator().manual_seed(7))
    assert torch.equal(first[0], second[0]) and torch.equal(first[1], second[1])


def test_overfit_one_batch(data, specs):
    tokens, event_mask = real_batch(data, 64, seed=5)
    torch.manual_seed(0)
    model = enc.TabBERT(specs, enc.EncoderConfig(dropout=0.0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(0)
    losses = []
    for _ in range(1000):
        masked, chosen = enc.mask_fields(tokens, event_mask, specs, generator=generator)
        loss, _ = enc.masked_field_loss(model, model.encode(masked, event_mask), tokens, chosen)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    uniform_guess = np.mean([np.log(s.size) for s in specs])  # about 4.15
    assert abs(losses[0] - uniform_guess) < 0.5
    assert np.mean(losses[-10:]) < 0.3  # reached 0.07 on 2026-11-20
