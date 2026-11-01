"""Interim scores of the l5 ablation (work order 2026-10-23) from a snapshot of the remote run folders.

For every (combination, L) with at least two seeds: seed-mean forecast MAE / CC on the split-A test days,
and the monthly block bootstrap of the difference against B2 (the no-image 1-D CNN on Ap + X-ray at the
same L, from l1_preds_A.npz) and against persistence. Writes a CSV and one figure (ΔMAE vs B2 per
combination, one panel per L).

    python scripts/lead1_l5_interim.py ~/Projects/GeoIndex/postbox/remote/l5-full-gpu*-interim-20261031 \
        --csv ~/Projects/GeoIndex/daily/lead1/l5-interim-20261031/l5_interim_scores.csv \
        --fig ~/Vaults/GeoIndex/experiments/figures/l5_interim_vs_b2.png
"""
import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from geoindex_daily.daily_index import default_data_dir  # noqa: E402
from geoindex_daily.lead1 import block_bootstrap_diff  # noqa: E402

# work-order IDs → image inputs (A1–A15: Ap + X-ray + images; A16 = the image-only model of Q5)
INPUTS = {"a01": "HMI", "a02": "AIA193", "a03": "AIA211", "a04": "LASCO", "a05": "HMI+AIA193",
          "a06": "HMI+AIA211", "a07": "HMI+LASCO", "a08": "AIA193+AIA211", "a09": "AIA193+LASCO",
          "a10": "AIA211+LASCO", "a11": "HMI+AIA193+AIA211", "a12": "HMI+AIA193+LASCO",
          "a13": "HMI+AIA211+LASCO", "a14": "AIA193+AIA211+LASCO", "a15": "all four",
          "a16img": "all four, images only"}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dirs", nargs="+", type=Path)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--fig", type=Path, default=None)
    p.add_argument("--min-seeds", type=int, default=2)
    args = p.parse_args()

    ref = np.load(default_data_dir() / "lead1" / "l1_preds_A.npz", allow_pickle=True)
    y, dates = ref["y"], pd.DatetimeIndex(ref["dates"])
    runs = {}
    for d in args.dirs:
        for f in d.glob("l5-*_preds_A_*_L*_s*.npz"):
            m = re.fullmatch(r"l5-([a-z0-9]+)_preds_A_([A-Z])_L(\d+)_s(\d)\.npz", f.name)
            z = np.load(f)
            assert (z["y"] == y).all() and (z["dates"] == ref["dates"]).all(), f"{f}: not the l1 test days"
            runs.setdefault((m[1], m[2], int(m[3])), {})[int(m[4])] = z["pred"]

    rows = []
    for (cid, arm, L), s in sorted(runs.items()):
        if len(s) < args.min_seeds:
            continue
        pm = np.mean([s[k] for k in sorted(s)], axis=0)
        row = {"id": cid.upper().replace("A0", "A").replace("IMG", ""), "inputs": INPUTS[cid], "arm": arm,
               "L": L, "seeds": len(s), "seed_mae": " / ".join(f"{np.abs(s[k] - y).mean():.2f}" for k in sorted(s)),
               "mae": np.abs(pm - y).mean(), "cc": np.corrcoef(pm, y)[0, 1]}
        for name, b in (("b2", ref[f"B_cnn_L{L}"]), ("pers", ref["persistence"])):
            r = block_bootstrap_diff(y, pm, b, dates, n=2000)
            row.update({f"{name}_{k}": v for k, v in r.items()})
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(["L", "mae"], ascending=[False, True])
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.csv, index=False)
    for L in sorted(df.L.unique(), reverse=True):
        b = ref[f"B_cnn_L{L}"]
        print(f"L={L} B2 MAE {np.abs(b - y).mean():.3f} CC {np.corrcoef(b, y)[0, 1]:.3f}")
    print(df[["id", "inputs", "L", "seeds", "seed_mae", "mae", "cc", "b2_d_mae", "b2_d_mae_lo", "b2_d_mae_hi",
              "b2_d_corr", "b2_d_corr_lo", "b2_d_corr_hi"]].round(3).to_string(index=False))

    if args.fig:
        full = df[df.seeds == 3]
        Ls = sorted(full.L.unique(), reverse=True)
        fig, axes = plt.subplots(1, len(Ls), figsize=(5.2 * len(Ls), 4.2), sharex=True)
        for ax, L in zip(np.atleast_1d(axes), Ls):
            t = full[full.L == L].sort_values("b2_d_mae", ascending=False)
            yy = np.arange(len(t))
            sig = (t.b2_d_mae_lo > 0) | (t.b2_d_mae_hi < 0)
            for i, (_, r) in enumerate(t.iterrows()):
                c = "#1f4e79" if sig.iloc[i] else "#8c8c8c"
                ax.plot([r.b2_d_mae_lo, r.b2_d_mae_hi], [i, i], color=c, lw=2, solid_capstyle="round")
                ax.plot(r.b2_d_mae, i, "o", ms=8, color=c, mfc=c if sig.iloc[i] else "white", mew=2)
            ax.axvline(0, color="#333333", lw=1)
            ax.set_yticks(yy, [f"{r.id}  {r.inputs}" for _, r in t.iterrows()], fontsize=8)
            ax.set_title(f"L = {L} days", fontsize=10)
            ax.set_xlabel("MAE(combination) − MAE(B2), Ap  (negative = images help)", fontsize=8)
            ax.grid(axis="x", color="#e5e5e5", lw=0.8); ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        fig.suptitle("l5 interim (split A, 1,273 test days): image combinations against Ap + X-ray without images\n"
                     "seed-mean of 3; 95 % monthly block bootstrap; filled = interval excludes zero", fontsize=9)
        fig.tight_layout()
        args.fig.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.fig, dpi=150); plt.close(fig)
        print("wrote", args.fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
