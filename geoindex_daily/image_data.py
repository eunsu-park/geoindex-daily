"""Windows of cached frames for the lead-1 image arms.

A sample issued on day t is the stack of frames at the slot hours of days t−L+1 .. t, for
each image channel, read from the uint8 cache (`frames1024/<channel>/<YYYY>/<YYYYMMDD_HH>.npy`)
built by `scripts/build_frame_cache.py`, plus the matching time-series window and target.
Every sample is complete by construction (the issue days come from `lead1.common_issue_days`
with the image completeness mask), so a missing file is an error, not a skip.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .frames import cache_path

DEFAULT_CHANNELS = ("aia193", "aia211", "hmi_m45", "lasco_c2")


def frame_paths(cache_dir: Path, issue_day, L: int, slot_hours, channels) -> list[list[Path]]:
    """[channel][frame] paths, frames ordered in time over days t−L+1 .. t."""
    t = pd.Timestamp(issue_day)
    times = [t - pd.Timedelta(days=L - 1 - d) + pd.Timedelta(hours=h) for d in range(L) for h in slot_hours]
    return [[cache_path(cache_dir, c, s) for s in times] for c in channels]


def check_cache(cache_dir: Path, issue_days, L: int, slot_hours, channels) -> pd.DataFrame:
    """Which issue days have every frame on disk; returns a frame (issue_day, ok, n_missing)."""
    rows = []
    for d in issue_days:
        missing = sum(1 for ch in frame_paths(cache_dir, d, L, slot_hours, channels) for p in ch if not p.exists())
        rows.append({"issue_day": d, "ok": missing == 0, "n_missing": missing})
    return pd.DataFrame(rows)


class FrameWindowDataset(Dataset):
    """Returns (images uint8 (C, 4L, H, W), ts float32 (L, F), y float32) per issue day."""

    def __init__(self, cache_dir: Path, issue_days, L: int, slot_hours, channels, ts: np.ndarray,
                 y: np.ndarray, size: int = 1024):
        assert len(issue_days) == len(ts) == len(y)
        self.cache_dir, self.days, self.L = Path(cache_dir), list(issue_days), L
        self.slot_hours, self.channels = tuple(slot_hours), tuple(channels)
        self.ts, self.y, self.size = ts.astype(np.float32), y.astype(np.float32), size

    def __len__(self):
        return len(self.days)

    def __getitem__(self, i):
        paths = frame_paths(self.cache_dir, self.days[i], self.L, self.slot_hours, self.channels)
        img = np.empty((len(self.channels), self.L * len(self.slot_hours), self.size, self.size), np.uint8)
        for c, ch in enumerate(paths):
            for f, p in enumerate(ch):
                a = np.load(p)
                if a.shape[0] != self.size:
                    step = a.shape[0] // self.size
                    a = a.reshape(self.size, step, self.size, step).mean(axis=(1, 3)).astype(np.uint8)
                img[c, f] = a
        return torch.from_numpy(img), torch.from_numpy(self.ts[i]), torch.tensor(self.y[i])
