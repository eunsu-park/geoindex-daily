"""Frame cache for the lead-1 image arms: one uint8 1024 × 1024 array per slot and channel.

Sources and transforms (all fixed, recorded in `calibration.json` next to the cache):

- ``aia193``, ``aia211``  SDO/AIA level-1 FITS (4096², DN): divide by EXPTIME, clip at 0,
  log1p, map [0, hi] → 0–255 where hi is the calibrated 99.9th percentile of log1p(DN/s);
  4 × 4 block mean → 1024². AIA's 0.06° roll is ignored.
- ``hmi_m45``  SDO/HMI 45 s magnetogram (4096², gauss, NaN off the disk): NaN → 0, rotated
  180° when CROTA2 ≈ 180 so that solar north is up as in AIA, clip ±1,500 G,
  (B / 1,500 + 1) / 2 → 0–255; 4 × 4 block mean. HMI's plate scale (0.504″) is kept, so the
  disk is 19 % larger than in AIA (0.6″); the network sees each channel separately.
- ``lasco_c2``  SOHO/LASCO C2 level-0.5 FITS (1024², counts): divide by EXPTIME, clip at 0,
  log1p, map [0, hi] → 0–255 with a calibrated hi.
- ``lasco_c2_rd``  running difference: (counts/s of the slot frame) − (counts/s of the
  previous C2 frame 6–30 min earlier), clipped at ±scale and mapped so that 0 → 128.

Every function here is pure NumPy on arrays so it can be unit-tested without FITS files;
`load_frame` is the only reader.
"""
import json
from pathlib import Path

import numpy as np

SIZE = 1024
SOURCES = ("aia193", "aia211", "hmi_m45", "lasco_c2", "lasco_c2_rd")
DEFAULT_CALIBRATION = {           # overwritten by `calibrate` on real frames
    "aia193": {"hi": 7.6}, "aia211": {"hi": 7.0}, "lasco_c2": {"hi": 6.5},
    "lasco_c2_rd": {"scale": 60.0}, "hmi_m45": {"bmax": 1500.0},
}


def block_mean(a: np.ndarray, k: int) -> np.ndarray:
    """(H, W) → (H/k, W/k) mean over k × k blocks (H and W must be multiples of k)."""
    h, w = a.shape
    return a.reshape(h // k, k, w // k, k).mean(axis=(1, 3))


def to_uint8(a: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((a - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def resize_to(a: np.ndarray, size: int = SIZE) -> np.ndarray:
    """Block-mean down to `size` when larger; nearest-neighbour up when smaller."""
    if a.shape[0] == size:
        return a
    if a.shape[0] > size and a.shape[0] % size == 0:
        return block_mean(a, a.shape[0] // size)
    idx = (np.arange(size) * a.shape[0] / size).astype(int)
    return a[idx][:, idx]


def preprocess_aia(data: np.ndarray, exptime: float, hi: float) -> np.ndarray:
    rate = np.clip(np.nan_to_num(data.astype(np.float32), nan=0.0), 0, None) / max(exptime, 1e-3)
    return to_uint8(resize_to(np.log1p(rate)), 0.0, hi)


def preprocess_hmi(data: np.ndarray, crota2: float, bmax: float = 1500.0) -> np.ndarray:
    b = np.nan_to_num(data.astype(np.float32), nan=0.0)
    if abs(((crota2 + 180) % 360) - 180) > 90:      # CROTA2 near 180: rotate so north is up
        b = b[::-1, ::-1]
    b = np.clip(b, -bmax, bmax)
    return to_uint8(resize_to(b), -bmax, bmax)


def lasco_rate(data: np.ndarray, exptime: float) -> np.ndarray:
    return np.clip(data.astype(np.float32), 0, None) / max(exptime, 1e-3)


def preprocess_lasco(data: np.ndarray, exptime: float, hi: float) -> np.ndarray:
    return to_uint8(resize_to(np.log1p(lasco_rate(data, exptime))), 0.0, hi)


def lasco_running_difference(cur: np.ndarray, cur_exptime: float, prev: np.ndarray,
                             prev_exptime: float, scale: float) -> np.ndarray:
    d = resize_to(lasco_rate(cur, cur_exptime)) - resize_to(lasco_rate(prev, prev_exptime))
    return to_uint8(d, -scale, scale)


# ---------------------------------------------------------------- readers and paths
def load_frame(path: Path):
    """(data, header-dict) for an AIA / HMI (extension 1) or LASCO (primary) FITS file."""
    from astropy.io import fits
    with fits.open(path) as f:
        hdu = f[1] if len(f) > 1 and f[1].data is not None else f[0]
        data = np.asarray(hdu.data)
        h = hdu.header
        keys = {k: h.get(k) for k in ("EXPTIME", "CROTA2", "NAXIS1", "NAXIS2", "QUALITY", "DATE-OBS", "T_OBS")}
    return data, keys


def cache_path(cache_dir: Path, source: str, slot_time) -> Path:
    import pandas as pd
    t = pd.Timestamp(slot_time)
    return Path(cache_dir) / source / f"{t.year:04d}" / f"{t.strftime('%Y%m%d_%H')}.npy"


def save_calibration(cache_dir: Path, calib: dict) -> None:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    (Path(cache_dir) / "calibration.json").write_text(json.dumps(calib, indent=2))


def load_calibration(cache_dir: Path) -> dict:
    p = Path(cache_dir) / "calibration.json"
    return json.loads(p.read_text()) if p.exists() else dict(DEFAULT_CALIBRATION)


def calibrate(frames: dict[str, list[tuple[np.ndarray, dict]]],
              rd_pairs: list[tuple[np.ndarray, float, np.ndarray, float]] | None = None) -> dict:
    """Per-channel scale constants from sample frames: the median over frames of the per-frame
    99.9th percentile of the transformed values (AIA / LASCO: log1p rate; rd: |difference|)."""
    calib = dict(DEFAULT_CALIBRATION)
    for src in ("aia193", "aia211"):
        if frames.get(src):
            p = [np.percentile(np.log1p(np.clip(np.nan_to_num(d, nan=0.0), 0, None) / max(h["EXPTIME"], 1e-3)), 99.9)
                 for d, h in frames[src]]
            calib[src] = {"hi": float(np.median(p)), "n": len(p)}
    if frames.get("lasco_c2"):
        p = [np.percentile(np.log1p(lasco_rate(d, h["EXPTIME"])), 99.9) for d, h in frames["lasco_c2"]]
        calib["lasco_c2"] = {"hi": float(np.median(p)), "n": len(p)}
    if rd_pairs:
        p = [np.percentile(np.abs(lasco_rate(c, ce) - lasco_rate(q, qe)), 99.0) for c, ce, q, qe in rd_pairs]
        calib["lasco_c2_rd"] = {"scale": float(np.median(p)), "n": len(p)}
    return calib
