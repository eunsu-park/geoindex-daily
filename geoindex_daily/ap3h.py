"""Samples, loss and scores of phase l6: the eight 3-hour ap of the next day (vault `planning/ap3h-plan.md`).

Used by scripts/ap3h_train.py. A sample issued on day t has the 8L three-hour bins of days t−L+1 .. t as input
and the bins of day t + 1 (and later days for a longer horizon) as target.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

TS_COLS = ["ap_t", "xray_mean_log", "xray_max_log"]
BINS = 8                                                 # 3-hour bins a day


def windows_3h(f3: pd.DataFrame, days: pd.DatetimeIndex, L: int, H: int):
    """X (N, 8L, F) over the bins of days t−L+1..t and Y (N, H) = log1p ap of the H bins after day t."""
    pos = f3.index.get_indexer(days + pd.Timedelta(days=1))       # first target bin: 00 UT of t + 1
    assert (pos >= 0).all(), "issue day outside the 3-hour table"
    F, A = f3[TS_COLS].to_numpy(float), f3["ap_t"].to_numpy(float)
    X = np.stack([F[p - BINS * L: p] for p in pos])
    Y = np.stack([A[p: p + H] for p in pos])
    return X, Y


def cme_days(days: pd.DatetimeIndex, catalog: Path, window_days: int = 3) -> np.ndarray:
    """True where an Earth-directed CME started on days t−window_days+1 .. t (the 72 h before issue)."""
    c = pd.read_parquet(catalog)
    starts = set(c.loc[c.earth_directed, "cme_date"].dt.normalize())
    return np.array([any((d - pd.Timedelta(days=k)) in starts for k in range(window_days)) for d in days])


def gaussian_nll(out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mean Gaussian NLL (constant dropped); out = [mean (H) | log-variance (H)] in standardised units."""
    mu, lv = out.float().chunk(2, dim=-1)
    lv = lv.clamp(-10, 10)
    return 0.5 * (lv + (y - mu) ** 2 * torch.exp(-lv)).mean()


def scores(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> dict:
    """3-hour and daily scores in ap units; y (N, H) ap, mu / sigma (N, H) log1p ap."""
    p = np.expm1(mu)
    ly = np.log1p(y)
    day_o, day_p = y.reshape(len(y), -1, BINS).mean(-1).ravel(), p.reshape(len(p), -1, BINS).mean(-1).ravel()
    out = {"mae_3h": np.abs(p - y).mean(), "rmse_3h": np.sqrt(((p - y) ** 2).mean()),
           "cc_3h": np.corrcoef(p.ravel(), y.ravel())[0, 1],
           "mae_day": np.abs(day_p - day_o).mean(), "cc_day": np.corrcoef(day_p, day_o)[0, 1],
           "cover68_3h": (np.abs(ly - mu) <= sigma).mean(), "cover95_3h": (np.abs(ly - mu) <= 1.96 * sigma).mean()}
    out.update({f"mae_h{h + 1}": np.abs(p[:, h] - y[:, h]).mean() for h in range(y.shape[1])})
    return out
