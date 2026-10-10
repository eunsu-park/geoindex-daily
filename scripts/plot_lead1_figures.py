"""Report figures for the lead-1 track (l1 + l2): length curve, image effect, cases, architecture.

Reads the saved forecasts in $GEOINDEX_DAILY_DATA/lead1/ (l1_preds_<split>.npz,
l2_preds_<split>_<arm>_L<L>_s<seed>.npz, l2_summary_bootstrap.csv) and writes PNGs to --out
(default: the vault's experiments/figures):

* ``lead1_length_curve.png`` — MAE and CC against input length L, split A, the two image arms
  next to their no-image l1 twins, persistence as a line.
* ``lead1_image_effect.png`` — image arm − no-image arm at each L, 95 % bootstrap interval.
* ``lead1_case_<YYYYMMDD>.png`` — observed daily Ap and the L = 2 forecasts over a test window.
* ``arch_lead1_fusion.png`` — the l2 model, boxes and arrows.

    python scripts/plot_lead1_figures.py
    python scripts/plot_lead1_figures.py --cases 2024-04-20:2024-06-10 2023-02-01:2023-03-31
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from geoindex_daily.daily_index import default_data_dir  # noqa: E402

VAULT_FIG = Path.home() / "Vaults/Research/GeoIndex/experiments/figures"
LENGTHS = [1, 2, 3, 4, 5, 6, 7]
# (image arm, no-image l1 key prefix, label, colour)
PAIRS = [("D", "A_cnn", "Ap", "#1f77b4"), ("B", "B_cnn", "Ap + X-ray", "#2ca02c"),
         ("E", "E_cnn", "Ap + X-ray + wind", "#d62728")]


def seed_mean(d: Path, split: str, arm: str, L: int):
    zs = [np.load(f, allow_pickle=True) for f in sorted(d.glob(f"l2_preds_{split}_{arm}_L{L}_s*.npz"))]
    if not zs:
        return None
    return pd.DatetimeIndex(zs[0]["dates"]), zs[0]["y"], np.mean([z["pred"] for z in zs], axis=0)


def length_curve(d: Path, out: Path):
    l1 = np.load(d / "l1_preds_A.npz", allow_pickle=True)
    y1 = l1["y"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for arm, ref, label, c in PAIRS:
        mae_i, cc_i, mae_r, cc_r = [], [], [], []
        for L in LENGTHS:
            dates, y, p = seed_mean(d, "A", arm, L)
            r = l1[f"{ref}_L{L}"][pd.DatetimeIndex(l1["dates"]).get_indexer(dates)]
            mae_i.append(np.abs(y - p).mean()); cc_i.append(np.corrcoef(y, p)[0, 1])
            mae_r.append(np.abs(y - r).mean()); cc_r.append(np.corrcoef(y, r)[0, 1])
        for ax, vi, vr in [(axes[0], mae_i, mae_r), (axes[1], cc_i, cc_r)]:
            ax.plot(LENGTHS, vi, "-o", color=c, label=f"{label} + images")
            ax.plot(LENGTHS, vr, "--s", color=c, alpha=0.55, mfc="white", label=f"{label}, no images")
    pers = l1["persistence"]
    axes[0].axhline(np.abs(y1 - pers).mean(), color="grey", ls=":", label="persistence")
    axes[1].axhline(np.corrcoef(y1, pers)[0, 1], color="grey", ls=":")
    axes[0].set_ylabel("MAE (Ap) — lower is better")
    axes[1].set_ylabel("CC — higher is better")
    for ax in axes:
        ax.set_xlabel("input length L (days)"); ax.set_xticks(LENGTHS); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="center right")
    fig.suptitle("Daily Ap one day ahead, split A (test 2022–2025, 1,273 days), seed-mean of 3", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "lead1_length_curve.png", dpi=150); plt.close(fig)


def image_effect(d: Path, out: Path):
    bt = pd.read_csv(d / "l2_summary_bootstrap.csv")
    bt = bt[bt.comparison.str.contains("no images")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for k, (arm, ref, label, c) in enumerate(PAIRS):
        for sp, mk, dx in [("A", "o", -0.12), ("B", "D", 0.12)]:
            s = bt[(bt.arm == arm) & (bt.split == sp)].sort_values("L")
            if s.empty:
                continue
            x = np.array([LENGTHS.index(L) for L in s.L]) + (k - 1) * 0.25 + (dx if sp == "B" else 0)
            for ax, v, lo, hi in [(axes[0], "d_mae", "d_mae_lo", "d_mae_hi"),
                                  (axes[1], "d_corr", "d_corr_lo", "d_corr_hi")]:
                ax.errorbar(x, s[v], yerr=[s[v] - s[lo], s[hi] - s[v]], fmt=mk, color=c, capsize=3,
                            mfc=c if sp == "A" else "white",
                            label=f"{label}: images − none, split {sp}")
    for ax, t in [(axes[0], "ΔMAE (Ap) — below 0: images help"), (axes[1], "ΔCC — above 0: images help")]:
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(len(LENGTHS))); ax.set_xticklabels([f"L = {L}" for L in LENGTHS])
        ax.set_ylabel(t); ax.grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=7.5, loc="upper left")
    fig.suptitle("What the images add at the same input length (95 % monthly block bootstrap)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "lead1_image_effect.png", dpi=150); plt.close(fig)


def case(d: Path, out: Path, start: str, end: str):
    split = "A" if pd.Timestamp(start).year >= 2022 else "B"
    l1 = np.load(d / f"l1_preds_{split}.npz", allow_pickle=True)
    dates, y, pe = seed_mean(d, split, "E", 2)
    r = l1["E_cnn_L2"][pd.DatetimeIndex(l1["dates"]).get_indexer(dates)]
    pers = l1["persistence"][pd.DatetimeIndex(l1["dates"]).get_indexer(dates)]
    tgt = dates + pd.Timedelta(days=1)          # forecasts are issued on day t for day t + 1
    m = (tgt >= start) & (tgt <= end)
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.plot(tgt[m], y[m], "-o", color="green", ms=3, label="observed daily Ap")
    ax.plot(tgt[m], pe[m], "--x", color="red", ms=4, label="E + images, L = 2")
    ax.plot(tgt[m], r[m], "-.", color="#1f77b4", label="E, no images, L = 2")
    ax.plot(tgt[m], pers[m], ":", color="grey", label="persistence")
    ax.axhline(30, color="k", lw=0.6, alpha=0.5)
    txt = "\n".join(f"{n}: MAE {np.abs(y[m] - p[m]).mean():.2f}"
                    for n, p in [("E + images", pe), ("E, no images", r), ("persistence", pers)])
    ax.text(0.01, 0.97, txt, transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6))
    ax.set_ylabel("daily Ap"); ax.set_yscale("symlog", linthresh=30); ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(f"One-day-ahead forecasts by target day, {start} → {end} (split {split} test)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / f"lead1_case_{pd.Timestamp(start):%Y%m%d}.png", dpi=150); plt.close(fig)


def architecture(out: Path):
    fig, ax = plt.subplots(figsize=(10, 3.6)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 4)

    def box(x, y, w, h, t, c):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=c, ec="k", lw=0.8))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=8)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="->", lw=0.9))

    box(0.1, 2.3, 1.9, 1.4, "frames\n4 channels × 4L\n1024 × 1024\n(AIA 193, 211,\nHMI, LASCO C2)", "#fde9c8")
    box(2.4, 2.5, 1.3, 1.0, "stem\nConv3d 1×4×4\nstride 4 → 256²", "#f3f3f3")
    box(4.1, 2.5, 1.9, 1.0, "4 × (2+1)D block\n3×3 spatial + 3 temporal\n32→64→128→256, /2 each", "#f3f3f3")
    box(6.4, 2.5, 1.0, 1.0, "pool\n(T, H, W)\n→ 256", "#f3f3f3")
    box(0.1, 0.4, 1.9, 1.2, "daily features × L\nD: Ap\nE: Ap, X-ray (2),\nwind (5)", "#d9e8f5")
    box(2.4, 0.5, 3.6, 1.0, "1-D CNN (the l1 model)\n2 × Conv1d k = 3, 32 ch, GELU, pool → 32", "#f3f3f3")
    box(7.8, 1.5, 1.0, 1.0, "concat 288\nMLP 128\n→ 1", "#e2f0d9")
    box(9.0, 1.6, 0.9, 0.8, "log1p Ap\nday t + 1", "#e2f0d9")
    for a in [(2.0, 3.0, 2.4, 3.0), (3.7, 3.0, 4.1, 3.0), (6.0, 3.0, 6.4, 3.0), (7.4, 3.0, 7.8, 2.2),
              (2.0, 1.0, 2.4, 1.0), (6.0, 1.0, 7.8, 1.8), (8.8, 2.0, 9.0, 2.0)]:
        arrow(*a)
    ax.text(5.0, 3.85, "image branch — a 3D CNN in factorised (2+1)D form", ha="center", fontsize=9)
    ax.text(4.2, 0.1, "time-series branch", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "arch_lead1_fusion.png", dpi=150); plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=VAULT_FIG)
    p.add_argument("--cases", nargs="*", default=["2024-04-20:2024-06-10", "2023-02-01:2023-03-31"])
    args = p.parse_args()
    d = default_data_dir() / "lead1"
    args.out.mkdir(parents=True, exist_ok=True)
    length_curve(d, args.out)
    image_effect(d, args.out)
    for c in args.cases:
        case(d, args.out, *c.split(":"))
    architecture(args.out)
    print(f"wrote figures to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
