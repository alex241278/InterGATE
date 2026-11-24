"""Utility helpers: seeding, memory management, formatting."""

import gc
import os
import random
import time

import numpy as np
import torch


# ────────────────────────────────────────────────────────────
# Seeding
# ────────────────────────────────────────────────────────────
def set_seed(seed: int = 1234):
    """Set random seed for reproducibility (basic version)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_all_seeds(seed: int):
    """Set random seed for full reproducibility (deterministic mode)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ────────────────────────────────────────────────────────────
# Memory management
# ────────────────────────────────────────────────────────────
def cleanup_memory(tag: str = ""):
    """Release references and caches (does NOT change training)."""
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()


# ────────────────────────────────────────────────────────────
# Formatting
# ────────────────────────────────────────────────────────────
def fmt_time(sec: float) -> str:
    sec = int(max(0, sec))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


# ────────────────────────────────────────────────────────────
# Device
# ────────────────────────────────────────────────────────────
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
