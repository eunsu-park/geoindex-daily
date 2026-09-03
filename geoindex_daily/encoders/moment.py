"""MOMENT (AutonLab) as the time-series encoder for the index history.

MOMENT takes a fixed 512-step context; a 30-day window is placed at the *end* of the
context with `input_mask` zero elsewhere, which is the model's own recipe for short
series. Two uses:

- `embed(...)`: frozen backbone, one 1024-d (large) / 512-d (small) embedding per window.
  Cheap to cache; a linear probe on top is the first time-series-only model.
- `forecaster(...)`: the forecasting head (`forecast_horizon` outputs) — NOT pretrained,
  must be fine-tuned (MOMENT warns so at load time).

MOMENT normalises each instance internally (RevIN), so the caller only needs to pick the
transform that makes the series roughly Gaussian (log1p for Ap; see config `ts_transform`).
"""
from __future__ import annotations

import numpy as np
import torch


def device_auto() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load(model: str = "AutonLab/MOMENT-1-large", task: str = "embedding",
         horizon: int = 60, device: str | None = None):
    """Load a MOMENT pipeline in `task` mode ('embedding' | 'forecasting'), eval, on device."""
    from momentfm import MOMENTPipeline
    kw = {"task_name": task}
    if task == "forecasting":
        kw.update({"forecast_horizon": horizon, "head_dropout": 0.1})
    m = MOMENTPipeline.from_pretrained(model, model_kwargs=kw)
    m.init()
    device = device or device_auto()
    if device == "mps":
        disable_attention_dropout(m)  # MPS scaled_dot_product_attention has no dropout kernel
    return m.to(device).eval()


def disable_attention_dropout(model) -> int:
    """Zero the float `dropout` attribute of attention modules (T5's SDPA path reads it).

    Needed to train on Apple MPS, where `scaled_dot_product_attention` raises on
    dropout > 0; regular nn.Dropout layers are left alone. Returns the count changed.
    """
    n = 0
    for mod in model.modules():
        if isinstance(getattr(mod, "dropout", None), float) and mod.dropout > 0:
            mod.dropout = 0.0
            n += 1
    return n


def pack(X: np.ndarray, seq_len: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
    """`(N, L)` windows → `(x_enc (N,1,seq_len), input_mask (N,seq_len))`, window right-aligned."""
    n, L = X.shape
    if L > seq_len:
        raise ValueError(f"window {L} longer than MOMENT context {seq_len}")
    x = torch.zeros(n, 1, seq_len, dtype=torch.float32)
    x[:, 0, seq_len - L:] = torch.as_tensor(X, dtype=torch.float32)
    mask = torch.zeros(n, seq_len, dtype=torch.float32)
    mask[:, seq_len - L:] = 1.0
    return x, mask


@torch.no_grad()
def embed(model, X: np.ndarray, seq_len: int = 512, batch: int = 64) -> np.ndarray:
    """Frozen MOMENT embeddings `(N, D)` for windows `X (N, L)`."""
    dev = next(model.parameters()).device
    out = []
    for i in range(0, len(X), batch):
        x, mask = pack(X[i: i + batch], seq_len)
        o = model(x_enc=x.to(dev), input_mask=mask.to(dev))
        out.append(o.embeddings.float().cpu().numpy())
    return np.concatenate(out) if out else np.empty((0, 0))


@torch.no_grad()
def forecast(model, X: np.ndarray, seq_len: int = 512, batch: int = 64) -> np.ndarray:
    """Forecast-head outputs `(N, horizon)` (meaningful only after fine-tuning)."""
    dev = next(model.parameters()).device
    out = []
    for i in range(0, len(X), batch):
        x, mask = pack(X[i: i + batch], seq_len)
        o = model(x_enc=x.to(dev), input_mask=mask.to(dev))
        out.append(o.forecast[:, 0, :].float().cpu().numpy())
    return np.concatenate(out) if out else np.empty((0, 0))
