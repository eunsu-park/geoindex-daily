"""(2+1)D image branch + 1-D time-series branch, late fusion, one output: the lead-1 model.

Image branch (a 3D CNN in the factorised "(2+1)D" form: a 2-D spatial convolution followed
by a 1-D temporal convolution, each with its own normalisation and activation):

    input (N, C, T, 1024, 1024) uint8  →  /255, per-channel standardisation
    stem   Conv3d k=(1,4,4) s=(1,4,4)        → (N, 32, T, 256, 256)
    stage1 (2+1)D block, spatial stride 2    → (N, 32, T, 128, 128)
    stage2 (2+1)D block, spatial stride 2    → (N, 64, T, 64, 64)
    stage3 (2+1)D block, spatial stride 2    → (N, 128, T, 32, 32)
    stage4 (2+1)D block, spatial stride 2    → (N, 256, T, 16, 16)
    global average pool over (T, H, W)       → (N, 256)

Time-series branch: `TSCNN.features` → (N, 32). Head: concat → MLP → 1.
"""
import torch
from torch import nn

from .ts_cnn import TSCNN


class Block2p1D(nn.Module):
    """Spatial 3×3 conv (stride s in space) + temporal 3 conv, with a residual projection."""

    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv3d(cin, cout, (1, 3, 3), stride=(1, stride, stride), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(cout), nn.GELU())
        self.temporal = nn.Sequential(
            nn.Conv3d(cout, cout, (3, 1, 1), padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(cout))
        self.proj = (nn.Identity() if cin == cout and stride == 1 else
                     nn.Conv3d(cin, cout, 1, stride=(1, stride, stride), bias=False))
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.temporal(self.spatial(x)) + self.proj(x))


class ImageBranch(nn.Module):
    def __init__(self, n_channels: int = 4, widths=(32, 64, 128, 256), stem_patch: int = 4):
        super().__init__()
        self.register_buffer("mean", torch.full((1, n_channels, 1, 1, 1), 0.5))
        self.register_buffer("std", torch.full((1, n_channels, 1, 1, 1), 0.25))
        self.stem = nn.Sequential(
            nn.Conv3d(n_channels, widths[0], (1, stem_patch, stem_patch), stride=(1, stem_patch, stem_patch), bias=False),
            nn.BatchNorm3d(widths[0]), nn.GELU())
        stages, cin = [], widths[0]
        for w in widths:
            stages.append(Block2p1D(cin, w, stride=2))
            cin = w
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.embed_dim = widths[-1]

    def set_normalisation(self, mean, std):
        self.mean.copy_(torch.as_tensor(mean, dtype=torch.float32).view(1, -1, 1, 1, 1))
        self.std.copy_(torch.as_tensor(std, dtype=torch.float32).view(1, -1, 1, 1, 1))

    def forward(self, img):                     # img: (N, C, T, H, W) uint8 or float in [0, 1]
        x = img.float() / 255.0 if img.dtype == torch.uint8 else img
        x = (x - self.mean) / self.std
        x = self.stages(self.stem(x))
        return self.pool(x).flatten(1)


class ImageFusion(nn.Module):
    """Late fusion of the image branch and the time-series branch; `ts_features` may be 0 (images only)."""

    def __init__(self, n_channels: int = 4, ts_features: int = 1, widths=(32, 64, 128, 256),
                 ts_width: int = 32, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.image = ImageBranch(n_channels, widths)
        self.ts = TSCNN(ts_features, ts_width) if ts_features > 0 else None
        d = self.image.embed_dim + (ts_width if ts_features > 0 else 0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, hidden), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, img, ts=None):
        z = self.image(img)
        if self.ts is not None:
            z = torch.cat([z, self.ts.features(ts)], dim=1)
        return self.head(z).squeeze(-1)
