"""Surya (NASA HelioSpectFormer, 366M) as a frozen image encoder over SuryaBench frames.

Surya's pretrained input is a pair of 13-channel 4096² frames at (t−60 min, t). In
`finetune=True` mode the forward returns the backbone tokens `(B, L, 1280)`; this module
pools them into fixed embeddings that can be cached once per day:

    mean   (1280,)        token average — the E5 probe's embedding, kept for continuity
    first  (1280,)        first token
    grid   (G, G, 1280)   tokens averaged over time and average-pooled to a G×G spatial grid,
                          so a downstream head can still attend over the disk

The model is imported from a local Surya clone (`surya_repo`), weights + scalers from a
checkpoint dir laid out like `~/Projects/GeoIndex/surya_probe/ckpt` (config.yaml,
scalers.yaml, surya.366m.v1.pt). The per-channel transform (signum-log, standardize) is
re-implemented here in numpy so the heavy `surya.datasets` stack is not needed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import yaml

SURYA_CHANNELS = ["aia94", "aia131", "aia171", "aia193", "aia211", "aia304", "aia335", "aia1600",
                  "hmi_m", "hmi_bx", "hmi_by", "hmi_bz", "hmi_v"]
PAIR_MINUTES = (-60, 0)


def read_frame(path: Path, channels=SURYA_CHANNELS) -> np.ndarray:
    """Raw `(C, 4096, 4096)` float32 from a SuryaBench .nc (h5netcdf; HDF5 is not thread-safe)."""
    import h5netcdf
    with h5netcdf.File(str(path), "r") as nc:
        return np.stack([np.asarray(nc.variables[ch][:]) for ch in channels], axis=0).astype(np.float32)


def scaler_arrays(scalers: dict, channels=SURYA_CHANNELS) -> dict[str, np.ndarray]:
    """Per-channel constants from scalers.yaml as `(C,1,1)` arrays: mean, std, epsilon, sl."""
    def vec(key):
        return np.array([scalers[ch][key] for ch in channels], dtype=np.float64)[:, None, None]
    return {"mean": vec("mean"), "std": vec("std"), "eps": vec("epsilon"), "sl": vec("sl_scale_factor")}


def transform(raw: np.ndarray, sc: dict[str, np.ndarray]) -> np.ndarray:
    """Surya's transform: `sign(x·sl)·log1p(|x·sl|)`, then `(· − mean) / (std + eps)`, per channel."""
    x = raw.astype(np.float64) * sc["sl"]
    x = np.sign(x) * np.log1p(np.abs(x))
    return ((x - sc["mean"]) / (sc["std"] + sc["eps"])).astype(np.float32)


def pool_tokens(tokens: torch.Tensor, grid: int = 8, patch: int = 16, img: int = 4096) -> dict[str, np.ndarray]:
    """`(1, L, D)` backbone tokens → {mean (D,), first (D,), grid (G,G,D)} as float32 numpy.

    L must be a multiple of the (img/patch)² patch grid; the leading factor is the number of
    time steps, which is averaged out before spatial pooling.
    """
    t = tokens.float()[0]
    L, D = t.shape
    side = img // patch
    if L % (side * side) != 0:
        raise ValueError(f"{L} tokens is not a multiple of the {side}x{side} patch grid")
    T = L // (side * side)
    spatial = t.view(T, side, side, D).mean(0).permute(2, 0, 1)[None]  # (1, D, side, side)
    g = torch.nn.functional.adaptive_avg_pool2d(spatial, grid)[0].permute(1, 2, 0)  # (G, G, D)
    return {"mean": t.mean(0).cpu().numpy(), "first": t[0].cpu().numpy(), "grid": g.cpu().numpy()}


def device_auto() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load(ckpt_dir: Path, surya_repo: Path, device: str | None = None):
    """Frozen Surya backbone in token-return mode. Returns `(model, config, scaler arrays, device)`."""
    ckpt_dir, surya_repo = Path(ckpt_dir), Path(surya_repo)
    if str(surya_repo) not in sys.path:
        sys.path.insert(0, str(surya_repo))
    from surya.models.helio_spectformer import HelioSpectFormer
    cfg = yaml.safe_load((ckpt_dir / "config.yaml").read_text())
    m, d = cfg["model"], cfg["data"]
    channels = d["sdo_channels"]
    device = device or device_auto()
    model = HelioSpectFormer(
        img_size=m["img_size"], patch_size=m["patch_size"], in_chans=len(channels),
        embed_dim=m["embed_dim"],
        time_embedding={"type": "linear", "time_dim": len(d["time_delta_input_minutes"])},
        depth=m["depth"], n_spectral_blocks=m["n_spectral_blocks"], num_heads=m["num_heads"],
        mlp_ratio=m["mlp_ratio"], drop_rate=m["drop_rate"], dtype=torch.bfloat16,
        window_size=m["window_size"], dp_rank=m["dp_rank"], learned_flow=m["learned_flow"],
        use_latitude_in_learned_flow=m["learned_flow"], init_weights=False, checkpoint_layers=None,
        rpe=m["rpe"], ensemble=m.get("ensemble"), finetune=True)
    state = torch.load(ckpt_dir / "surya.366m.v1.pt", map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    state = {k.removeprefix("module."): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    # finetune=True drops the forecast head, so the checkpoint's `unembed.*` weights are
    # unexpected by construction; anything else means a wrong checkpoint
    bad_missing = [k for k in missing if not k.startswith("unembed")]
    bad_unexpected = [k for k in unexpected if not k.startswith("unembed")]
    if bad_missing or bad_unexpected:
        raise RuntimeError(f"state dict mismatch: missing {bad_missing[:5]}, unexpected {bad_unexpected[:5]}")
    model = model.to(device).eval()
    sc = scaler_arrays(yaml.safe_load((ckpt_dir / "scalers.yaml").read_text()), channels)
    return model, cfg, sc, device


@torch.no_grad()
def embed_pair(model, frame_prev: np.ndarray, frame_now: np.ndarray, sc: dict, device: str,
               grid: int = 8, cfg: dict | None = None) -> dict[str, np.ndarray]:
    """Embeddings for one (t−60, t) pair of raw frames `(C, H, W)`."""
    ts = np.stack([transform(frame_prev, sc), transform(frame_now, sc)], axis=1)[None]  # (1,C,2,H,W)
    batch = {"ts": torch.from_numpy(ts).to(device),
             "time_delta_input": torch.tensor([[1.0, 0.0]], dtype=torch.float32, device=device)}
    tokens = model(batch)
    if device == "mps":
        torch.mps.synchronize()
    m = cfg["model"] if cfg else {"patch_size": 16, "img_size": 4096}
    return pool_tokens(tokens, grid, m["patch_size"], m["img_size"])
