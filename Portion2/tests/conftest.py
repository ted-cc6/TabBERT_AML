import sys
from pathlib import Path

import torch

PORTION2_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PORTION2_DIR.parent / "Portion1" / "processed"

sys.path.insert(0, str(PORTION2_DIR))

# Using every logical core under WSL makes CPU training steps about 40x slower (1.8 s vs 40 ms on a 14-core laptop),
# because threads wait on cores that Windows has taken back.
torch.set_num_threads(min(8, torch.get_num_threads()))
