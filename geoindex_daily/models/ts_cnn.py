"""A small 1-D CNN over a daily feature window, the time-series branch of the lead-1 models.

Input `(N, L, F)` (days × features), output one value: the standardised log1p Ap of the
next day. Two Conv1d layers with GELU, a global average pool over the L days, and a linear
head. The same block is the time-series branch of the image fusion model, so the
time-series-only run of this network is the matched neural reference for "images help".
"""
import numpy as np
import torch
from torch import nn


class TSCNN(nn.Module):
    def __init__(self, n_features: int, width: int = 32, kernel: int = 3, dropout: float = 0.1,
                 out_dim: int = 1):
        super().__init__()
        pad = kernel // 2
        self.net = nn.Sequential(
            nn.Conv1d(n_features, width, kernel, padding=pad), nn.GELU(),
            nn.Conv1d(width, width, kernel, padding=pad), nn.GELU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(dropout),
        )
        self.head = nn.Linear(width, out_dim)
        self.embed_dim = width

    def features(self, x: torch.Tensor) -> torch.Tensor:  # x: (N, L, F)
        return self.net(x.transpose(1, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x)).squeeze(-1)


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_ts_cnn(Xtr, ytr, Xva, yva, seed: int = 0, width: int = 32, kernel: int = 3,
                 dropout: float = 0.1, lr: float = 1e-3, weight_decay: float = 1e-4,
                 batch_size: int = 128, max_epochs: int = 200, patience: int = 15,
                 device=None, val_metric=None):
    """Fit on standardised inputs/targets; early stopping on `val_metric(pred_val)` (lower is better).

    `val_metric` defaults to the MSE on the standardised targets. Returns (model, history).
    """
    device = device or default_device()
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = TSCNN(Xtr.shape[-1], width, kernel, dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    Xt = torch.tensor(Xtr, dtype=torch.float32, device=device)
    yt = torch.tensor(ytr, dtype=torch.float32, device=device)
    Xv = torch.tensor(Xva, dtype=torch.float32, device=device)
    yv = np.asarray(yva, dtype=float)
    n = len(Xt)
    best, best_state, bad, hist = np.inf, None, 0, []
    g = torch.Generator(device="cpu").manual_seed(seed)
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n, generator=g).to(device)
        for i in range(0, n, batch_size):
            idx = perm[i: i + batch_size]
            loss = nn.functional.mse_loss(model(Xt[idx]), yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xv).cpu().numpy()
        v = float(((pv - yv) ** 2).mean()) if val_metric is None else float(val_metric(pv))
        hist.append(v)
        if v < best - 1e-6:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, {"best_val": best, "epochs": len(hist), "best_epoch": int(np.argmin(hist)) + 1}


def predict(model, X, device=None, batch_size: int = 1024) -> np.ndarray:
    device = device or next(model.parameters()).device
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i: i + batch_size], dtype=torch.float32, device=device)
            out.append(model(xb).cpu().numpy())
    return np.concatenate(out) if out else np.empty(0)
