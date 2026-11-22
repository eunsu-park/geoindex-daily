"""ImageFusion with an optional-CME residual (Model A) and a footpoint residual (Model B).

Co-authors' work order of 2026-11-22: Model 0 is ImageFusion as trained in phase l6; Model A adds the CME branch,
Model B adds the footpoint branch on top of A and nothing else. Both additions are residuals on the base head:

    z      = [SDO | LASCO | time series] embeddings                    (as ImageFusion)
    out    = head(z)                                                     (Model 0)
           + g · R_cme([z, E_cme])                                       (A, B)   g = 1 if any CME slot is filled
           + R_fp([z, E_fp])                                             (B)

E_cme is the masked mean over the CME slots of a per-slot MLP; with every mask 0 it is 0 and g = 0, so the CME term is
exactly zero and the forecast is the base head's. E_fp is a small MLP of the six footpoint values. The last layer of
each residual starts at zero, so A starts as Model 0 and B as A. The base modules are created first, from the same
seed, so their initial weights equal Model 0's; training is joint (the base weights then differ from Model 0's).

`aux` (N, S·F + S + P): the S slots' F features (zero where empty), the S slot masks, then the P footpoint values.
"""
import torch
from torch import nn

from .image_fusion import ImageFusion


def _residual(d_in: int, hidden: int, out_dim: int, dropout: float) -> nn.Sequential:
    r = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, out_dim))
    nn.init.zeros_(r[-1].weight); nn.init.zeros_(r[-1].bias)
    return r


class OptionalInputFusion(ImageFusion):
    def __init__(self, *args, cme_slots: int = 3, cme_features: int = 5, fp_features: int = 0,
                 cme_dim: int = 32, fp_dim: int = 16, **kw):
        super().__init__(*args, **kw)
        d = self.head[1].in_features
        hidden, dropout, out_dim = self.head[1].out_features, self.head[0].p, self.head[-1].out_features
        self.S, self.F, self.P = cme_slots, cme_features, fp_features
        self.cme_enc = nn.Sequential(nn.Linear(cme_features, cme_dim), nn.GELU(), nn.Linear(cme_dim, cme_dim), nn.GELU())
        self.cme_res = _residual(d + cme_dim, hidden, out_dim, dropout)
        if fp_features:
            self.fp_enc = nn.Sequential(nn.Linear(fp_features, fp_dim), nn.GELU(), nn.Linear(fp_dim, fp_dim), nn.GELU())
            self.fp_res = _residual(d + fp_dim, hidden, out_dim, dropout)

    def split_aux(self, aux: torch.Tensor):
        n = self.S * self.F
        return (aux[:, :n].reshape(-1, self.S, self.F), aux[:, n: n + self.S], aux[:, n + self.S: n + self.S + self.P])

    def forward(self, img, ts=None, lasco=None, aux=None):
        z = []
        if self.image is not None:
            z.append(self.image(img))
        if self.lasco is not None:
            z.append(self.lasco(lasco))
        if self.ts is not None:
            z.append(self.ts.features(ts))
        z = torch.cat(z, dim=1)
        out = self.head(z)
        x, m, fp = self.split_aux(aux.to(z.dtype))
        e = self.cme_enc(x) * m.unsqueeze(-1)
        e = e.sum(1) / m.sum(1, keepdim=True).clamp(min=1)
        g = (m.sum(1, keepdim=True) > 0).to(out.dtype)
        out = out + g * self.cme_res(torch.cat([z, e], dim=1))
        if self.P:
            out = out + self.fp_res(torch.cat([z, self.fp_enc(fp)], dim=1))
        return out.squeeze(-1) if out.shape[-1] == 1 else out
