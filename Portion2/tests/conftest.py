import sys
from pathlib import Path

import torch

PORTION2_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PORTION2_DIR.parent / "Portion1" / "processed"

sys.path.insert(0, str(PORTION2_DIR))

# Using every logical core under WSL makes CPU training steps about 40x slower (1.8 s vs 40 ms on a 14-core laptop),
# because threads wait on cores that Windows has taken back.
torch.set_num_threads(min(8, torch.get_num_threads()))


def tiny_checkpoint(tmp_dir, device=None, **overrides):
    """Pretrain a 4-step model on 1,024 train windows; enough to exercise every code path."""
    import dataclasses

    import portion2_encoder as enc

    cfg = enc.load_run_config(PORTION2_DIR / "configs" / "smoke_cpu.json")
    settings = dict(run_name="tiny", windows_per_epoch=1024, epochs=1, valid_windows=256,
                    batch_size=256, warmup_steps=2, amp=False)
    cfg = dataclasses.replace(cfg, **{**settings, **overrides})
    return enc.pretrain(cfg, PROCESSED_DIR, tmp_dir, device=device or torch.device("cpu"))
