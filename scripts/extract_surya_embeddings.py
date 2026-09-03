"""Frozen Surya embeddings for daily anchors from the SuryaBench archive tree → one .npz per day.

For each anchor day at `--hour` UT, the pair (t−60 min, t) is read from the archive
(`<archive>/YYYY/MM/YYYYMMDD_HHMM.nc`). If either frame is missing the anchor is shifted
forward in 1-h steps up to `--max-shift` hours (the bucket has ~1 % gaps). Output:
`<out>/<YYYYMMDD_HHMM>.npz` with `mean (1280,)`, `first (1280,)`, `grid (G,G,1280)` and
`effective_time`. Resume-safe: existing files are skipped. Progress goes to `<out>/progress.log`.

    python scripts/extract_surya_embeddings.py --start 2011-01-01 --end 2011-01-10 \
        --archive /Volumes/archive2/solar_images/suryabench --limit 3          # smoke
    python scripts/extract_surya_embeddings.py --start 2010-05-13 --end 2024-12-31   # full, on the server

Reads over NFS/SMB dominate on a laptop (2 × 590 MB per anchor); the pair is prefetched
in a thread while the previous pair runs through the model.
"""
import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geoindex_daily.daily_index import default_data_dir  # noqa: E402
from geoindex_daily.encoders import surya as su  # noqa: E402

DEFAULT_CKPT = Path.home() / "Projects" / "GeoIndex" / "surya_probe" / "ckpt"
DEFAULT_REPO = Path.home() / "GitHub" / "surya-upstream"


def frame_path(archive: Path, ts: pd.Timestamp) -> Path:
    return archive / f"{ts:%Y}" / f"{ts:%m}" / f"{ts:%Y%m%d_%H%M}.nc"


def resolve_pair(archive: Path, anchor: pd.Timestamp, max_shift: int):
    """First hour ≥ anchor (within max_shift h) whose (t−60, t) frames both exist, else None."""
    for shift in range(max_shift + 1):
        t = anchor + timedelta(hours=shift)
        a, b = frame_path(archive, t - timedelta(minutes=60)), frame_path(archive, t)
        if a.exists() and b.exists():
            return t, a, b
    return None


def read_pair(paths):
    a, b = paths
    return su.read_frame(a), su.read_frame(b)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--hour", type=int, default=0, help="anchor hour UT (default 0)")
    p.add_argument("--max-shift", type=int, default=6)
    p.add_argument("--archive", default=str(Path.home() / "NAS/archive/solar_images/suryabench"))
    p.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    p.add_argument("--surya-repo", default=str(DEFAULT_REPO))
    p.add_argument("--out", default=None, help="embedding dir (default data dir/surya/emb/13ch)")
    p.add_argument("--grid", type=int, default=8)
    p.add_argument("--device", default=None)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    archive = Path(args.archive)
    if not archive.is_dir():
        raise SystemExit(f"archive not found (mount?): {archive}")
    out = Path(args.out) if args.out else default_data_dir() / "surya" / "emb" / "13ch"
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "progress.log"

    def log(msg):
        line = f"{pd.Timestamp.now('UTC').isoformat()} {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    anchors = [t + timedelta(hours=args.hour) for t in pd.date_range(args.start, args.end, freq="D")]
    todo = [a for a in anchors if not (out / f"{a:%Y%m%d_%H%M}.npz").exists()]
    if args.limit:
        todo = todo[: args.limit]
    log(f"{len(anchors)} anchors, {len(todo)} to do, archive {archive}, out {out}")
    if not todo:
        return 0

    model, cfg, sc, device = su.load(args.ckpt, args.surya_repo, args.device)
    log(f"model ready on {device}")

    ok = fail = 0
    t0 = time.time()
    with ThreadPoolExecutor(1) as pool:
        pending = None  # (anchor, effective, future)
        i = 0
        while i < len(todo) or pending is not None:
            # prefetch the next resolvable anchor
            while pending is None and i < len(todo):
                a = todo[i]; i += 1
                r = resolve_pair(archive, a, args.max_shift)
                if r is None:
                    log(f"SKIP {a:%Y-%m-%d} no pair within +{args.max_shift}h"); fail += 1
                    continue
                pending = (a, r[0], pool.submit(read_pair, (r[1], r[2])))
            if pending is None:
                break
            a, eff, fut = pending
            pending = None
            # queue the following one before running the model on this one
            while pending is None and i < len(todo):
                a2 = todo[i]; i += 1
                r2 = resolve_pair(archive, a2, args.max_shift)
                if r2 is None:
                    log(f"SKIP {a2:%Y-%m-%d} no pair within +{args.max_shift}h"); fail += 1
                    continue
                pending = (a2, r2[0], pool.submit(read_pair, (r2[1], r2[2])))
            try:
                prev, now = fut.result()
                e = su.embed_pair(model, prev, now, sc, device, args.grid, cfg)
                np.savez(out / f"{a:%Y%m%d_%H%M}.npz", effective_time=str(eff), **e)
                ok += 1
            except Exception as ex:  # noqa: BLE001
                log(f"FAIL {a:%Y-%m-%d}: {str(ex)[:120]}"); fail += 1
            done = ok + fail
            if done % 10 == 0 or done == len(todo):
                rate = (time.time() - t0) / max(done, 1)
                log(f"progress {done}/{len(todo)} ok={ok} fail={fail} {rate:.0f}s/anchor "
                    f"eta={(len(todo) - done) * rate / 3600:.1f}h")
    return 0


if __name__ == "__main__":
    sys.exit(main())
