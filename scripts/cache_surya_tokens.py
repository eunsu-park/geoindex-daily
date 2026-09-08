"""Cache Surya's patch-embedding output per day for LoRA training (lora-plan §2).

For each issue day, read the (t−60 min, t) SuryaBench pair, apply Surya's transform and
the frozen front end (16×16 patch embedding + positional encoding) and save the resulting
65,536 × 1,280 token map. LoRA later trains the backbone from these files instead of
re-reading 1.2 GB of frames per sample. Frames are read in a prefetch thread while the GPU
embeds the previous pair. Resumable: existing files are skipped; `index.csv` lists them.

    python scripts/cache_surya_tokens.py --start 2016-01-01 --end 2016-01-20 --limit 20 --validate 20   # smoke
    python scripts/cache_surya_tokens.py --start 2016-01-01 --end 2024-12-31                             # reduced set

`--validate N` runs the frozen backbone on N cached token maps and compares the mean token
with the cached embeddings in emb/13ch (expect cosine ≥ 0.999 after the bf16 rounding).
"""
import argparse
import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_surya_embeddings import DEFAULT_CKPT, DEFAULT_REPO, read_pair, resolve_pair  # noqa: E402
from geoindex_daily.daily_index import default_data_dir  # noqa: E402
from geoindex_daily.encoders import surya as su  # noqa: E402

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


@torch.no_grad()
def front_end(model, prev, now, sc, device):
    """Transform + patch embedding + positional encoding → (65,536, 1,280) in the model dtype."""
    ts = np.stack([su.transform(prev, sc), su.transform(now, sc)], axis=1)[None]  # (1, C, 2, H, W)
    dtype = next(model.parameters()).dtype
    x = torch.from_numpy(ts).to(device=device, dtype=dtype)
    dt = torch.tensor([[1.0, 0.0]], dtype=torch.float32, device=device)
    return model.embedding(x, dt)[0]


@torch.no_grad()
def validate(model, out: Path, emb_dir: Path, n: int, device: str) -> None:
    files = sorted(out.glob("2*.pt"))
    picked = [f for f in files if (emb_dir / (f.stem + ".npz")).exists()][:n]
    if not picked:
        print("validate: no cached day has a reference embedding"); return
    cos = []
    for f in picked:
        tok = torch.load(f)["tokens"].to(device=device, dtype=next(model.parameters()).dtype)[None]
        mean = model.backbone(tok)[0].float().mean(0).cpu().numpy()
        ref = np.load(emb_dir / (f.stem + ".npz"))["mean"]
        cos.append(float(mean @ ref / np.linalg.norm(mean) / np.linalg.norm(ref)))
    print(f"validate: {len(cos)} days, cosine(mean token from cache, emb/13ch) min {min(cos):.5f} "
          f"median {float(np.median(cos)):.5f}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--hour", type=int, default=0)
    p.add_argument("--max-shift", type=int, default=6)
    p.add_argument("--archive", default=str(Path.home() / "NAS/archive2/solar_images/suryabench"))
    p.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    p.add_argument("--surya-repo", default=str(DEFAULT_REPO))
    p.add_argument("--out", default=None, help="token dir (default data dir/surya/tokens/13ch)")
    p.add_argument("--emb-dir", default=None, help="reference embeddings for --validate (default data dir/surya/emb/13ch)")
    p.add_argument("--dtype", choices=list(DTYPES), default="bf16")
    p.add_argument("--device", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--validate", type=int, default=0)
    args = p.parse_args()

    archive = Path(args.archive)
    if not archive.is_dir():
        raise SystemExit(f"archive not found (mount?): {archive}")
    out = Path(args.out) if args.out else default_data_dir() / "surya" / "tokens" / "13ch"
    out.mkdir(parents=True, exist_ok=True)
    emb_dir = Path(args.emb_dir) if args.emb_dir else default_data_dir() / "surya" / "emb" / "13ch"
    log_path, index_path = out / "progress.log", out / "index.csv"

    def log(msg):
        line = f"{pd.Timestamp.now('UTC').isoformat()} {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    anchors = [t + timedelta(hours=args.hour) for t in pd.date_range(args.start, args.end, freq="D")]
    todo = [a for a in anchors if not (out / f"{a:%Y%m%d_%H%M}.pt").exists()]
    if args.limit:
        todo = todo[: args.limit]
    log(f"{len(anchors)} anchors, {len(todo)} to do, dtype {args.dtype}, archive {archive}, out {out}")

    model, cfg, sc, device = su.load(args.ckpt, args.surya_repo, args.device, DTYPES[args.dtype])
    log(f"model ready on {device} ({next(model.parameters()).dtype})")

    ok = fail = 0
    t0 = time.time()
    new_index = not index_path.exists()
    with open(index_path, "a", newline="") as idx, ThreadPoolExecutor(1) as pool:
        w = csv.writer(idx)
        if new_index:
            w.writerow(["date", "effective_time", "file"])
        pending, i = None, 0

        def queue_next():
            nonlocal i, fail
            while i < len(todo):
                a = todo[i]; i += 1
                r = resolve_pair(archive, a, args.max_shift)
                if r is None:
                    log(f"SKIP {a:%Y-%m-%d} no pair within +{args.max_shift}h"); fail += 1
                    continue
                return (a, r[0], pool.submit(read_pair, (r[1], r[2])))
            return None

        pending = queue_next()
        while pending is not None:
            a, eff, fut = pending
            pending = queue_next()  # prefetch the next pair while this one is embedded
            try:
                prev, now = fut.result()
                tok = front_end(model, prev, now, sc, device)
                f = out / f"{a:%Y%m%d_%H%M}.pt"
                torch.save({"tokens": tok.to(DTYPES[args.dtype]).cpu().contiguous(),
                            "effective_time": str(eff), "dtype": args.dtype}, f)
                w.writerow([f"{a:%Y-%m-%d}", str(eff), f.name]); idx.flush()
                ok += 1
            except Exception as ex:  # noqa: BLE001
                log(f"FAIL {a:%Y-%m-%d}: {str(ex)[:120]}"); fail += 1
            done = ok + fail
            if done % 10 == 0 or done == len(todo):
                rate = (time.time() - t0) / max(done, 1)
                log(f"progress {done}/{len(todo)} ok={ok} fail={fail} {rate:.1f}s/anchor eta={(len(todo)-done)*rate/3600:.1f}h")

    if args.validate:
        validate(model, out, emb_dir, args.validate, device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
