"""TabBERT-style encoder for account transaction windows: model, masking, pretraining and handoffs.

The data loader lives in portion2_data.py. Field order and special IDs follow Portion1/DATA_CONTRACT.md.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import portion2_data as pdata

TOKEN_FIELDS = [
    "payment_format", "currency", "counterparty_bank", "direction",
    "amount_bucket", "time_gap_bucket",
]
PAD, UNK, MASK, NO_HISTORY = 0, 1, 2, 3
DT_FIELD = "time_gap_bucket"


@dataclass(frozen=True)
class FieldSpec:
    name: str
    size: int
    real_low: int
    real_high: int


@dataclass
class EncoderConfig:
    field_dim: int = 32
    field_heads: int = 2
    field_ff: int = 128
    field_layers: int = 1
    d_model: int = 128
    seq_layers: int = 2
    heads: int = 4
    ff: int = 512
    dropout: float = 0.1
    max_len: int = 32
    use_dt: bool = True


def load_field_specs(processed_dir: Path, use_dt: bool = True) -> list[FieldSpec]:
    """Read vocabulary sizes and the real (non-special) ID range of each field from vocab_and_bins.json."""
    vocab = json.loads((Path(processed_dir) / "vocab_and_bins.json").read_text())
    assert vocab["token_field_order"] == TOKEN_FIELDS
    specs = []
    for name in TOKEN_FIELDS:
        if name in vocab["categorical"]:
            real_ids = [i for token, i in vocab["categorical"][name].items() if not token.startswith("<")]
            low, high = min(real_ids), max(real_ids)
        else:
            low = vocab[name]["first_bin_id"]
            high = low + vocab[name]["bins"] - 1
        specs.append(FieldSpec(name, high + 1, low, high))
    if not use_dt:
        specs = [s for s in specs if s.name != DT_FIELD]
    return specs


def select_fields(tokens: torch.Tensor, use_dt: bool) -> torch.Tensor:
    """Drop the time gap column for the without-Δt variant (it is the last column)."""
    return tokens if use_dt else tokens[..., :TOKEN_FIELDS.index(DT_FIELD)]


class TabBERT(nn.Module):
    """Two-level encoder: a field transformer inside each event, then a sequence transformer across events."""

    # The field transformer sees B * L short sequences. CUDA attention kernels fail ("invalid argument") once
    # sequences x heads passes about 65,535 in one call (export batch 2048 x 32 rows x 2 heads = 131,072),
    # so feed it in chunks. Each event is encoded independently, so chunking does not change the output.
    FIELD_CHUNK = 8192

    def __init__(self, specs: list[FieldSpec], cfg: EncoderConfig):
        super().__init__()
        assert len(specs) == (6 if cfg.use_dt else 5)
        self.specs = list(specs)
        self.cfg = cfg
        n_fields = len(specs)

        self.field_embeddings = nn.ModuleList(nn.Embedding(s.size, cfg.field_dim, padding_idx=PAD) for s in specs)
        self.field_position = nn.Parameter(torch.randn(n_fields, cfg.field_dim) * 0.02)
        self.field_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(cfg.field_dim, cfg.field_heads, cfg.field_ff, cfg.dropout,
                                       activation="gelu", batch_first=True),
            cfg.field_layers, enable_nested_tensor=False,
        )
        self.event_projection = nn.Linear(n_fields * cfg.field_dim, cfg.d_model)

        self.sequence_position = nn.Parameter(torch.randn(cfg.max_len, cfg.d_model) * 0.02)
        self.input_norm = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.sequence_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(cfg.d_model, cfg.heads, cfg.ff, cfg.dropout,
                                       activation="gelu", batch_first=True),
            cfg.seq_layers, enable_nested_tensor=False,
        )

        # Output heads reuse each field's input embedding table (BERT-style weight tying),
        # so the 30,531-way bank head adds 4K parameters instead of 3.9M.
        self.head_transforms = nn.ModuleList(
            nn.Sequential(nn.Linear(cfg.d_model, cfg.field_dim), nn.GELU(), nn.LayerNorm(cfg.field_dim))
            for _ in specs
        )
        self.head_biases = nn.ParameterList(nn.Parameter(torch.zeros(s.size)) for s in specs)

        # BERT initialisation (std 0.02). PyTorch's default std 1 makes the tied heads start with logits of
        # std about 5.7, so the first masked loss is about 12 instead of the uniform-guess value of about 4.
        for embedding in self.field_embeddings:
            nn.init.normal_(embedding.weight, std=0.02)
            with torch.no_grad():
                embedding.weight[PAD].zero_()

    def encode(self, tokens: torch.Tensor, event_mask: torch.Tensor) -> torch.Tensor:
        """(B, L, F) token IDs and (B, L) event mask -> (B, L, d_model) contextual event vectors."""
        batch, length, n_fields = tokens.shape
        fields = torch.stack([emb(tokens[..., f]) for f, emb in enumerate(self.field_embeddings)], dim=2)
        fields = fields.reshape(batch * length, n_fields, -1) + self.field_position
        fields = torch.cat([self.field_encoder(chunk) for chunk in fields.split(self.FIELD_CHUNK)])
        events = self.event_projection(fields.reshape(batch, length, -1))
        events = self.dropout(self.input_norm(events + self.sequence_position[:length]))
        return self.sequence_encoder(events, src_key_padding_mask=~event_mask)

    def field_logits(self, hidden: torch.Tensor, field: int) -> torch.Tensor:
        """(..., d_model) event vectors -> (..., vocabulary size) logits for one field."""
        projected = self.head_transforms[field](hidden)
        return projected @ self.field_embeddings[field].weight.T + self.head_biases[field]


def _corrupt(tokens: torch.Tensor, chosen: torch.Tensor, draw: torch.Tensor,
             specs: list[FieldSpec], generator: torch.Generator | None) -> torch.Tensor:
    """BERT 80/10/10 rule on chosen positions: MASK, a random real ID of the same field, or unchanged."""
    low = torch.tensor([s.real_low for s in specs], device=tokens.device)
    span = torch.tensor([s.real_high - s.real_low + 1 for s in specs], device=tokens.device)
    random_ids = low + (torch.rand(tokens.shape, generator=generator, device=tokens.device) * span).long()
    random_ids = torch.minimum(random_ids, low + span - 1)

    masked = tokens.clone()
    masked[chosen & (draw < 0.8)] = MASK
    use_random = chosen & (draw >= 0.8) & (draw < 0.9)
    masked[use_random] = random_ids[use_random]
    return masked


def mask_fields(tokens: torch.Tensor, event_mask: torch.Tensor, specs: list[FieldSpec],
                rate: float = 0.15, generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Field-level masking: choose `rate` of real field tokens independently. Returns (masked tokens, chosen)."""
    real = event_mask[..., None].expand_as(tokens)
    chosen = (torch.rand(tokens.shape, generator=generator, device=tokens.device) < rate) & real
    draw = torch.rand(tokens.shape, generator=generator, device=tokens.device)
    return _corrupt(tokens, chosen, draw, specs, generator), chosen


def mask_events(tokens: torch.Tensor, event_mask: torch.Tensor, specs: list[FieldSpec],
                rate: float = 0.15, generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Whole-transaction masking: choose `rate` of real events and corrupt all their fields with one draw per event."""
    batch, length, n_fields = tokens.shape
    chosen_events = (torch.rand((batch, length), generator=generator, device=tokens.device) < rate) & event_mask
    chosen = chosen_events[..., None].expand_as(tokens)
    draw = torch.rand((batch, length, 1), generator=generator, device=tokens.device).expand_as(tokens)
    return _corrupt(tokens, chosen, draw, specs, generator), chosen.clone()


def masked_field_loss(model: TabBERT, hidden: torch.Tensor, targets: torch.Tensor,
                      chosen: torch.Tensor) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor] | None]]:
    """Cross entropy on chosen positions only, averaged over fields.

    Logits are computed for chosen positions only: full bank logits for a 512-window batch would need about 2 GB.
    Returns (loss, per-field (logits, targets) or None when a field had no chosen positions).
    """
    losses, outputs = [], []
    for f in range(len(model.specs)):
        selected = chosen[..., f]
        if not selected.any():
            outputs.append(None)
            continue
        logits = model.field_logits(hidden[selected], f)
        field_targets = targets[..., f][selected]
        losses.append(F.cross_entropy(logits.float(), field_targets))
        outputs.append((logits, field_targets))
    return torch.stack(losses).mean(), outputs


# ---------------------------------------------------------------- pretraining

@dataclass
class RunConfig:
    run_name: str
    seed: int
    masking: str = "field"  # "field" (main) or "event" (whole-transaction ablation)
    use_dt: bool = True
    windows_per_epoch: int = 1_000_000
    epochs: int = 3
    valid_windows: int = 50_000
    batch_size: int = 512
    lr: float = 1e-3
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    amp: bool = True
    encoder: dict = field(default_factory=dict)  # EncoderConfig overrides, e.g. {"d_model": 256, "seq_layers": 4}


def load_run_config(path: Path) -> RunConfig:
    values = json.loads(Path(path).read_text())
    unknown = set(values) - {f.name for f in fields(RunConfig)}
    if unknown:
        raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")
    return RunConfig(**values)


def pick_device() -> torch.device:
    """CUDA (Colab), then Intel XPU, then CPU. On CPU, cap threads: all 14 cores under WSL is about 40x slower."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    torch.set_num_threads(min(8, torch.get_num_threads()))
    return torch.device("cpu")


def lr_factor(step: int, warmup_steps: int, total_steps: int) -> float:
    """Linear warmup to 1, then cosine decay to 0."""
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def fit_baselines(tokens: np.ndarray, event_split: np.ndarray, specs: list[FieldSpec]) -> dict[str, int]:
    """Most frequent non-PAD token per field over train-period events: the "always guess the mode" baseline."""
    train_tokens = tokens[event_split == "train"]
    baselines = {}
    for f, spec in enumerate(specs):
        counts = np.bincount(train_tokens[:, f], minlength=spec.size)
        counts[PAD] = 0
        baselines[spec.name] = int(counts.argmax())
    return baselines


class FieldMetrics:
    """Accumulates masked-position accuracy per field over batches, next to the most-frequent-token baseline.

    Direction at the last position is skipped (the target event is always "sent"). Bucket fields also get
    within-one-bucket accuracy on real bucket IDs, and the time gap gets accuracy without gap-0 (bucket 4) targets.
    """

    GAP_ZERO = 4

    def __init__(self, specs: list[FieldSpec], baselines: dict[str, int]):
        self.specs = specs
        self.baselines = baselines
        self.sums: dict[str, float] = {}

    def _add(self, key: str, value: float) -> None:
        self.sums[key] = self.sums.get(key, 0.0) + float(value)

    def update(self, outputs: list, chosen: torch.Tensor) -> None:
        for f, (spec, output) in enumerate(zip(self.specs, outputs)):
            if output is None:
                continue
            logits, targets = output
            predictions = logits.argmax(-1).cpu()
            targets = targets.cpu()
            positions = chosen[..., f].nonzero()[:, 1].cpu()  # same order as boolean indexing
            keep = positions != chosen.shape[1] - 1 if spec.name == "direction" else torch.ones_like(targets, dtype=torch.bool)
            predictions, targets = predictions[keep], targets[keep]
            name = spec.name
            self._add(f"n_{name}", len(targets))
            self._add(f"correct_{name}", (predictions == targets).sum())
            self._add(f"base_{name}", (targets == self.baselines[name]).sum())
            if name in ("amount_bucket", "time_gap_bucket"):
                real = targets >= spec.real_low
                self._add(f"n1_{name}", real.sum())
                self._add(f"correct1_{name}", ((predictions - targets).abs() <= 1)[real].sum())
            if name == "time_gap_bucket":
                nonzero = targets != self.GAP_ZERO
                self._add("n_gap_nonzero", nonzero.sum())
                self._add("correct_gap_nonzero", (predictions == targets)[nonzero].sum())

    def result(self) -> dict[str, float]:
        def ratio(num: str, den: str) -> float:
            return self.sums.get(num, 0.0) / self.sums[den] if self.sums.get(den) else float("nan")

        out = {}
        for spec in self.specs:
            name = spec.name
            out[f"acc_{name}"] = ratio(f"correct_{name}", f"n_{name}")
            out[f"baseline_{name}"] = ratio(f"base_{name}", f"n_{name}")
            out[f"n_{name}"] = int(self.sums.get(f"n_{name}", 0))
            if name in ("amount_bucket", "time_gap_bucket"):
                out[f"acc1_{name}"] = ratio(f"correct1_{name}", f"n1_{name}")
            if name == "time_gap_bucket":
                out["acc_time_gap_bucket_nonzero"] = ratio("correct_gap_nonzero", "n_gap_nonzero")
        return out


def _autocast(device: torch.device, enabled: bool):
    # bfloat16 on Intel XPU; float16 on CUDA (T4s lack fast bfloat16), paired with a GradScaler.
    dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    return torch.autocast(device.type, dtype=dtype, enabled=enabled and device.type in ("cuda", "xpu"))


def _masked_step(model, batch, specs, mask_fn, generator, device, use_dt, amp):
    """Mask on CPU (same masks on every device), move to the device, return (loss, outputs, chosen)."""
    tokens = select_fields(batch["tokens"], use_dt)
    event_mask = batch["event_mask"]
    masked, chosen = mask_fn(tokens, event_mask, specs, generator=generator)
    masked, event_mask, targets, chosen_dev = (t.to(device, non_blocking=True) for t in (masked, event_mask, tokens, chosen))
    with _autocast(device, amp):
        hidden = model.encode(masked, event_mask)
        loss, outputs = masked_field_loss(model, hidden, targets, chosen_dev)
    return loss, outputs, chosen


@torch.no_grad()
def evaluate_masked(model, tokens, window_index, rows, specs, baselines, cfg: RunConfig, device) -> dict[str, float]:
    """Masked loss and field metrics on fixed rows, with a fixed mask seed so epochs are comparable."""
    model.eval()
    mask_fn = mask_fields if cfg.masking == "field" else mask_events
    generator = torch.Generator().manual_seed(cfg.seed + 1)
    metrics = FieldMetrics(specs, baselines)
    loss_sum, n_batches = 0.0, 0
    for batch in pdata.iterate_batches(tokens, window_index, rows, cfg.batch_size):
        loss, outputs, chosen = _masked_step(model, batch, specs, mask_fn, generator, device, cfg.use_dt, cfg.amp)
        metrics.update(outputs, chosen)
        loss_sum += loss.item()
        n_batches += 1
    model.train()
    return {"valid_loss": loss_sum / n_batches, **metrics.result()}


def save_amount_heatmap(model: TabBERT, path: Path) -> None:
    """Cosine similarity between the amount bucket embeddings; neighbouring buckets should end up similar."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = [s.name for s in model.specs].index("amount_bucket")
    spec = model.specs[f]
    weights = model.field_embeddings[f].weight[spec.real_low:spec.real_high + 1].detach().float().cpu()
    weights = F.normalize(weights, dim=1)
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow((weights @ weights.T).numpy(), cmap="viridis", vmin=-1, vmax=1)
    ax.set_title("Amount bucket embeddings: cosine similarity")
    ax.set_xlabel("amount bucket (low to high)")
    ax.set_ylabel("amount bucket (low to high)")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def pretrain(cfg: RunConfig, processed_dir: Path, out_dir: Path, device: torch.device | None = None,
             log_every: int = 100) -> Path:
    """Masked field pretraining on train windows only. Writes checkpoint.pt, metrics.csv and the heatmap."""
    device = device or pick_device()
    torch.manual_seed(cfg.seed)
    run_dir = Path(out_dir) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")

    specs = load_field_specs(processed_dir, cfg.use_dt)
    tokens = pdata.load_tokens(processed_dir)
    window_index = pdata.load_window_index(processed_dir)
    event_split = pd.read_csv(Path(processed_dir) / "events_meta.csv.gz", usecols=["split"]).split.to_numpy()
    baselines = fit_baselines(select_fields(torch.from_numpy(tokens), cfg.use_dt).numpy(), event_split, specs)
    valid_rows = pdata.sample_epoch_rows(window_index, "valid", cfg.valid_windows, cfg.seed, 0)

    encoder_cfg = EncoderConfig(**cfg.encoder, use_dt=cfg.use_dt)
    model = TabBERT(specs, encoder_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = math.ceil(cfg.windows_per_epoch / cfg.batch_size)
    total_steps = steps_per_epoch * cfg.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_factor(s, cfg.warmup_steps, total_steps))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    mask_fn = mask_fields if cfg.masking == "field" else mask_events
    generator = torch.Generator().manual_seed(cfg.seed)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{cfg.run_name}: {n_params:,} parameters on {device}; {total_steps:,} steps", flush=True)

    rows_log, step, start = [], 0, time.time()
    for epoch in range(cfg.epochs):
        rows = pdata.sample_epoch_rows(window_index, "train", cfg.windows_per_epoch, cfg.seed, epoch)
        loss_sum = 0.0
        for batch in pdata.iterate_batches(tokens, window_index, rows, cfg.batch_size):
            loss, _, _ = _masked_step(model, batch, specs, mask_fn, generator, device, cfg.use_dt, cfg.amp)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_sum += loss.item()
            step += 1
            if step % log_every == 0:
                print(f"  epoch {epoch + 1} step {step:,}/{total_steps:,}  loss {loss.item():.4f}  "
                      f"lr {scheduler.get_last_lr()[0]:.2e}  {time.time() - start:.0f}s", flush=True)
        valid = evaluate_masked(model, tokens, window_index, valid_rows, specs, baselines, cfg, device)
        row = {"epoch": epoch + 1, "step": step, "train_loss": loss_sum / steps_per_epoch, **valid,
               "seconds": round(time.time() - start, 1)}
        rows_log.append(row)
        pd.DataFrame(rows_log).to_csv(run_dir / "metrics.csv", index=False)
        print(f"  epoch {epoch + 1}: train {row['train_loss']:.4f}  valid {row['valid_loss']:.4f}  "
              + "  ".join(f"{s.name} {valid[f'acc_{s.name}']:.3f}/{valid[f'baseline_{s.name}']:.3f}" for s in specs),
              flush=True)

    ckpt_path = run_dir / "checkpoint.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "encoder_config": asdict(encoder_cfg),
        "run_config": asdict(cfg),
        "specs": [asdict(s) for s in specs],
        "baselines": baselines,
    }, ckpt_path)
    save_amount_heatmap(model, run_dir / "amount_embedding_cosine.png")
    return ckpt_path


def load_encoder(ckpt_path: Path, device: torch.device | None = None) -> tuple[TabBERT, list[FieldSpec], RunConfig]:
    device = device or pick_device()
    saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    specs = [FieldSpec(**s) for s in saved["specs"]]
    model = TabBERT(specs, EncoderConfig(**saved["encoder_config"]))
    model.load_state_dict(saved["state_dict"])
    return model.to(device).eval(), specs, RunConfig(**saved["run_config"])


# ---------------------------------------------------------------- handoff 1: frozen embeddings (Task 5)

@torch.no_grad()
def export_embeddings(ckpt_path: Path, processed_dir: Path, out_dir: Path, batch_size: int = 2048,
                      rows: np.ndarray | None = None, device: torch.device | None = None) -> tuple[Path, Path]:
    """Write one embedding per window: the target-row vector, then the mean over real rows (2 x d_model, float16).

    Rows follow window_index order (all splits, including tail) unless `rows` is given. The first d_model columns
    alone are the target-position-only fallback. Writes through a memory map, so the full array never sits in RAM.
    """
    model, specs, cfg = load_encoder(ckpt_path, device)
    device = next(model.parameters()).device
    tokens = pdata.load_tokens(processed_dir)
    window_index = pdata.load_window_index(processed_dir)
    rows = np.arange(len(window_index)) if rows is None else np.asarray(rows)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_path = out_dir / f"embeddings_{cfg.run_name}.npy"
    ids_path = out_dir / f"embeddings_{cfg.run_name}_ids.npy"
    embeddings = np.lib.format.open_memmap(emb_path, mode="w+", dtype=np.float16,
                                           shape=(len(rows), 2 * model.cfg.d_model))
    begin, start = 0, time.time()
    for batch in pdata.iterate_batches(tokens, window_index, rows, batch_size):
        event_mask = batch["event_mask"].to(device)
        with _autocast(device, cfg.amp):
            hidden = model.encode(select_fields(batch["tokens"], cfg.use_dt).to(device), event_mask).float()
        weights = event_mask.unsqueeze(-1).float()
        mean = (hidden * weights).sum(1) / weights.sum(1)
        out = torch.cat([hidden[:, -1], mean], dim=1)
        embeddings[begin:begin + len(out)] = out.cpu().numpy().astype(np.float16)
        begin += len(out)
        if (begin // batch_size) % 500 == 0:
            print(f"  {begin:,}/{len(rows):,} windows  {time.time() - start:.0f}s", flush=True)
    embeddings.flush()
    del embeddings
    np.save(ids_path, window_index.transaction_id.to_numpy()[rows].astype(np.int64))
    print(f"Wrote {emb_path.name} ({len(rows):,} x {2 * model.cfg.d_model}) in {time.time() - start:.0f}s", flush=True)
    return emb_path, ids_path


# ---------------------------------------------------------------- handoff 2: fine-tuning (Task 6)

LABEL_FRACTIONS = (0.01, 0.02, 0.05, 0.2, 1.0)


def load_label_subset(processed_dir: Path, fraction: float, seed: int) -> pd.DataFrame:
    """Labelled train transactions for one fraction and seed: columns transaction_id, is_laundering.

    1.0 reads the shared full-train list (the same for every seed); other fractions read label_subsets.csv.gz.
    """
    if not any(math.isclose(fraction, f) for f in LABEL_FRACTIONS):
        raise ValueError(f"fraction must be one of {LABEL_FRACTIONS}, got {fraction}")
    if math.isclose(fraction, 1.0):
        return pd.read_csv(Path(processed_dir) / "label_subset_100pct.csv.gz")[["transaction_id", "is_laundering"]]
    subsets = pd.read_csv(Path(processed_dir) / "label_subsets.csv.gz")
    keep = (subsets.seed == seed) & np.isclose(subsets.label_fraction, fraction)
    return subsets.loc[keep, ["transaction_id", "is_laundering"]].reset_index(drop=True)


def finetune_epoch_sample(labels: np.ndarray, max_negatives: int, seed: int, epoch: int) -> np.ndarray:
    """Positions for one epoch: every positive plus at most `max_negatives` negatives, shuffled."""
    rng = np.random.default_rng([seed, epoch, 6])
    positives = np.flatnonzero(labels == 1)
    negatives = np.flatnonzero(labels == 0)
    if len(negatives) > max_negatives:
        negatives = rng.choice(negatives, size=max_negatives, replace=False)
    return rng.permutation(np.concatenate([positives, negatives]))


class FineTuneModel(nn.Module):
    """Pretrained encoder plus a small MLP on the target-row vector, giving one laundering logit."""

    def __init__(self, encoder: TabBERT, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(nn.Linear(encoder.cfg.d_model, hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, 1))

    def forward(self, tokens: torch.Tensor, event_mask: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder.encode(tokens, event_mask)[:, -1]).squeeze(-1)


@torch.no_grad()
def score_rows(model: FineTuneModel, tokens, window_index, rows, cfg: RunConfig, device, batch_size: int) -> np.ndarray:
    """Laundering probability for each row, in `rows` order."""
    model.eval()
    scores = []
    for batch in pdata.iterate_batches(tokens, window_index, rows, batch_size):
        with _autocast(device, cfg.amp):
            logits = model(select_fields(batch["tokens"], cfg.use_dt).to(device), batch["event_mask"].to(device))
        scores.append(torch.sigmoid(logits.float()).cpu().numpy())
    model.train()
    return np.concatenate(scores)


def finetune(ckpt_path: Path, processed_dir: Path, out_dir: Path, fraction: float, seed: int,
             max_epochs: int = 10, patience: int = 3, max_negatives_per_epoch: int = 200_000,
             lr: float = 2e-4, batch_size: int = 512, eval_rows: dict | None = None,
             device: torch.device | None = None) -> Path:
    """Fine-tune encoder and head end to end on one labelled subset; early stopping on valid PR AUC.

    Only the labelled subset is used for training (unlabelled train transactions are not negatives). Writes
    scores.csv (transaction_id, split, score for valid and test, from the best epoch) and finetune_log.csv to
    out_dir/ft_<run>_<pct>pct_seed<seed>/ and returns the scores path. Test is scored but never used for choices.
    """
    from sklearn.metrics import average_precision_score

    torch.manual_seed(seed)
    encoder, specs, cfg = load_encoder(ckpt_path, device)
    device = next(encoder.parameters()).device
    model = FineTuneModel(encoder).to(device).train()
    run_dir = Path(out_dir) / f"ft_{cfg.run_name}_{fraction * 100:g}pct_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    tokens = pdata.load_tokens(processed_dir)
    window_index = pdata.load_window_index(processed_dir)
    subset = load_label_subset(processed_dir, fraction, seed)
    train_rows = pd.Index(window_index.transaction_id).get_indexer(subset.transaction_id)
    assert (train_rows >= 0).all() and (window_index.split.to_numpy()[train_rows] == "train").all()
    labels = subset.is_laundering.to_numpy()
    split = window_index.split.to_numpy()
    if eval_rows is None:
        eval_rows = {name: np.flatnonzero(split == name) for name in ("valid", "test")}
    valid_labels = window_index.is_laundering.to_numpy()[eval_rows["valid"]]

    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": lr},
        {"params": model.head.parameters(), "lr": lr * 5},
    ], weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    print(f"Fine-tuning {cfg.run_name} on {fraction:.0%} labels, seed {seed}: "
          f"{labels.sum():,} positives, {(labels == 0).sum():,} negatives", flush=True)

    best_ap, best_state, best_epoch, log, start = -1.0, None, 0, [], time.time()
    for epoch in range(max_epochs):
        picked = finetune_epoch_sample(labels, max_negatives_per_epoch, seed, epoch)
        epoch_labels = torch.from_numpy(labels[picked]).float()
        n_pos = float(epoch_labels.sum())
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(picked) - n_pos) / n_pos, device=device))
        loss_sum, begin = 0.0, 0
        for batch in pdata.iterate_batches(tokens, window_index, train_rows[picked], batch_size):
            y = epoch_labels[begin:begin + len(batch["transaction_id"])].to(device)
            begin += len(y)
            with _autocast(device, cfg.amp):
                logits = model(select_fields(batch["tokens"], cfg.use_dt).to(device), batch["event_mask"].to(device))
            loss = loss_fn(logits.float(), y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * len(y)

        valid_scores = score_rows(model, tokens, window_index, eval_rows["valid"], cfg, device, batch_size * 4)
        valid_ap = float(average_precision_score(valid_labels, valid_scores))
        log.append({"epoch": epoch + 1, "train_loss": loss_sum / len(picked), "valid_ap": valid_ap,
                    "seconds": round(time.time() - start, 1)})
        pd.DataFrame(log).to_csv(run_dir / "finetune_log.csv", index=False)
        print(f"  epoch {epoch + 1}: train loss {log[-1]['train_loss']:.4f}  valid PR AUC {valid_ap:.4f}  "
              f"{log[-1]['seconds']:.0f}s", flush=True)
        if valid_ap > best_ap:
            best_ap, best_epoch = valid_ap, epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch + 1 - best_epoch >= patience:
            print(f"  early stop: no gain for {patience} epochs", flush=True)
            break

    model.load_state_dict(best_state)
    frames = []
    for name in ("valid", "test"):
        rows = eval_rows[name]
        frames.append(pd.DataFrame({
            "transaction_id": window_index.transaction_id.to_numpy()[rows],
            "split": name,
            "score": score_rows(model, tokens, window_index, rows, cfg, device, batch_size * 4),
        }))
    scores_path = run_dir / "scores.csv"
    pd.concat(frames).to_csv(scores_path, index=False)
    (run_dir / "summary.json").write_text(json.dumps({
        "run": cfg.run_name, "fraction": fraction, "seed": seed, "best_epoch": best_epoch,
        "valid_ap": best_ap, "positives": int(labels.sum()), "negatives": int((labels == 0).sum()),
    }, indent=2) + "\n")
    print(f"Best epoch {best_epoch}: valid PR AUC {best_ap:.4f}. Wrote {scores_path}", flush=True)
    return scores_path


# ---------------------------------------------------------------- handoff 3: reconstruction score (Task 7)

@torch.no_grad()
def reconstruction_scores(ckpt_path: Path, processed_dir: Path, out_dir: Path,
                          splits: tuple[str, ...] = ("valid", "test", "tail"), batch_size: int = 1024,
                          rows: np.ndarray | None = None, device: torch.device | None = None) -> Path:
    """Zero-shot anomaly score: for each target transaction, mask one target-row field at a time and record the
    negative log likelihood of the true token. Direction is skipped (the target is always "sent").

    Writes recon_<run>.csv with transaction_id, split, nll_<field> per scored field and nll_sum. Per-field columns
    let Portion 4 normalise; the 30,531-way bank field would otherwise dominate the sum.
    """
    model, specs, cfg = load_encoder(ckpt_path, device)
    device = next(model.parameters()).device
    tokens = pdata.load_tokens(processed_dir)
    window_index = pdata.load_window_index(processed_dir)
    split = window_index.split.to_numpy()
    if rows is None:
        rows = np.flatnonzero(np.isin(split, splits))
    scored = [f for f, s in enumerate(specs) if s.name != "direction"]

    ids, nll_parts, start = [], [], time.time()
    for batch in pdata.iterate_batches(tokens, window_index, rows, batch_size):
        batch_tokens = select_fields(batch["tokens"], cfg.use_dt).to(device)
        event_mask = batch["event_mask"].to(device)
        columns = []
        for f in scored:
            masked = batch_tokens.clone()
            masked[:, -1, f] = MASK
            with _autocast(device, cfg.amp):
                hidden = model.encode(masked, event_mask)[:, -1]
                logits = model.field_logits(hidden, f)
            log_probs = F.log_softmax(logits.float(), dim=-1)
            columns.append(-log_probs.gather(1, batch_tokens[:, -1, f:f + 1]).squeeze(1))
        nll_parts.append(torch.stack(columns, dim=1).cpu().numpy())
        ids.append(batch["transaction_id"].numpy())
        done = sum(len(i) for i in ids)
        if (done // batch_size) % 500 == 0:
            print(f"  {done:,}/{len(rows):,} windows  {time.time() - start:.0f}s", flush=True)

    nll = np.concatenate(nll_parts)
    out = pd.DataFrame({"transaction_id": np.concatenate(ids), "split": split[rows][:len(nll)]})
    for j, f in enumerate(scored):
        out[f"nll_{specs[f].name}"] = nll[:, j]
    out["nll_sum"] = nll.sum(axis=1)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"recon_{cfg.run_name}.csv"
    out.to_csv(path, index=False)
    print(f"Wrote {path.name} ({len(out):,} rows) in {time.time() - start:.0f}s", flush=True)
    return path
