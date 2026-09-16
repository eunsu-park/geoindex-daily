"""SINet — the Solar Index Network of Wang, Abduallah & Wang (JGR Space Physics, 2026).

Port of the co-author's `utils.py` + `layers/{Embed,Conv_Blocks}.py` (shared 2026-09), kept
parameter-for-parameter compatible with their checkpoints (`SINet_60_interval_F10.pth` loads
with `strict=True`). Architecturally it is TimesNet (Wu et al., ICLR 2023) in its
long-term-forecast configuration: instance normalisation, a conv token embedding plus
positional encoding, a linear map from `seq_len` to `seq_len + pred_len` positions, then
`e_layers` TimesBlocks (FFT picks the top-k periods, the sequence is folded into a 2-D
period × phase image and filtered by an Inception block of 1×1…11×11 convolutions), and a
linear projection back to one channel. Only the forecasting path is kept.

Defaults are the paper's 60-day F10.7 model: d_model 32, d_ff 64, 6 kernels, 2 layers,
top-k 3, dropout 0.3.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SINetConfig:
    seq_len: int = 30
    pred_len: int = 60
    enc_in: int = 1
    c_out: int = 1
    d_model: int = 32
    d_ff: int = 64
    num_kernels: int = 6
    e_layers: int = 2
    top_k: int = 3
    dropout: float = 0.3
    freq: str = "d"          # only sizes the (unused when x_mark is None) time-feature embedding
    embed: str = "timeF"


# --- layers/Embed.py --------------------------------------------------------------------

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int):
        super().__init__()
        self.tokenConv = nn.Conv1d(c_in, d_model, kernel_size=3, padding=1, padding_mode="circular", bias=False)
        nn.init.kaiming_normal_(self.tokenConv.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x):
        return self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)


class TimeFeatureEmbedding(nn.Module):
    """Present for checkpoint compatibility; never called when `x_mark` is None."""

    def __init__(self, d_model: int, freq: str = "d"):
        super().__init__()
        freq_map = {"h": 4, "t": 5, "s": 6, "m": 1, "a": 1, "w": 2, "d": 3, "b": 3}
        self.embed = nn.Linear(freq_map[freq], d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int, freq: str = "d", dropout: float = 0.1):
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in, d_model)
        self.position_embedding = PositionalEmbedding(d_model)
        self.temporal_embedding = TimeFeatureEmbedding(d_model, freq)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mark=None):
        x = self.value_embedding(x) + self.position_embedding(x)
        if x_mark is not None:
            x = x + self.temporal_embedding(x_mark)
        return self.dropout(x)


# --- layers/Conv_Blocks.py --------------------------------------------------------------

class InceptionBlockV1(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_kernels: int = 6):
        super().__init__()
        self.num_kernels = num_kernels
        self.kernels = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i)
                                      for i in range(num_kernels)])
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return torch.stack([k(x) for k in self.kernels], dim=-1).mean(-1)


# --- utils.py: TimesBlock / Model --------------------------------------------------------

def fft_for_period(x: torch.Tensor, k: int = 2):
    """Top-k periods of a (B, T, C) batch by mean FFT amplitude, and the per-sample amplitudes."""
    xf = torch.fft.rfft(x, dim=1)
    amp = abs(xf).mean(0).mean(-1)
    amp[0] = 0
    _, top = torch.topk(amp, k)
    top = top.detach().cpu().numpy()
    period = x.shape[1] // top
    return period, abs(xf).mean(-1)[:, top]


class TimesBlock(nn.Module):
    def __init__(self, cfg: SINetConfig):
        super().__init__()
        self.seq_len, self.pred_len, self.k = cfg.seq_len, cfg.pred_len, cfg.top_k
        self.conv = nn.Sequential(InceptionBlockV1(cfg.d_model, cfg.d_ff, cfg.num_kernels), nn.GELU(),
                                  InceptionBlockV1(cfg.d_ff, cfg.d_model, cfg.num_kernels))

    def forward(self, x):
        B, T, N = x.size()
        total = self.seq_len + self.pred_len
        periods, weight = fft_for_period(x, self.k)
        res = []
        for i in range(self.k):
            period = int(periods[i])
            if total % period != 0:
                length = (total // period + 1) * period
                out = torch.cat([x, torch.zeros(B, length - total, N, device=x.device, dtype=x.dtype)], dim=1)
            else:
                length, out = total, x
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :total, :])
        res = torch.stack(res, dim=-1)
        weight = F.softmax(weight, dim=1).unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        return torch.sum(res * weight, -1) + x


class SINet(nn.Module):
    """Forecast `pred_len` steps from `seq_len` steps of a univariate (or multivariate) series."""

    def __init__(self, cfg: SINetConfig | None = None, **overrides):
        super().__init__()
        cfg = cfg or SINetConfig(**overrides)
        self.cfg = cfg
        self.seq_len, self.pred_len = cfg.seq_len, cfg.pred_len
        self.model = nn.ModuleList([TimesBlock(cfg) for _ in range(cfg.e_layers)])
        self.enc_embedding = DataEmbedding(cfg.enc_in, cfg.d_model, cfg.freq, cfg.dropout)
        self.layer_norm = nn.LayerNorm(cfg.d_model)
        self.predict_linear = nn.Linear(cfg.seq_len, cfg.pred_len + cfg.seq_len)
        self.projection = nn.Linear(cfg.d_model, cfg.c_out, bias=True)

    def forward(self, x_enc: torch.Tensor, x_mark_enc=None, *_args) -> torch.Tensor:
        """`x_enc` (B, seq_len, enc_in) → (B, pred_len, c_out)."""
        means = x_enc.mean(1, keepdim=True).detach()
        x = x_enc - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev
        enc = self.enc_embedding(x, x_mark_enc)
        enc = self.predict_linear(enc.permute(0, 2, 1)).permute(0, 2, 1)
        for block in self.model:
            enc = self.layer_norm(block(enc))
        dec = self.projection(enc)
        total = self.pred_len + self.seq_len
        dec = dec * stdev[:, 0, :].unsqueeze(1).repeat(1, total, 1) + means[:, 0, :].unsqueeze(1).repeat(1, total, 1)
        return dec[:, -self.pred_len:, :]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
