"""Report figures for W42 (l3: LASCO C2 running difference at 12 slots/day, own image branch).

Reads $GEOINDEX_DAILY_DATA/lead1/ (l3_* and l2_* forecasts, l1_preds_<split>.npz,
l3_summary_bootstrap.csv from lead1_l3_summary.py) and the frame cache, and writes PNGs to --out
(default: the vault's experiments/figures):

* ``l3_length_curve.png`` — MAE and CC against L, split A, per arm: W42, W41 on the same days, no images.
* ``l3_vs_w41.png`` — W42 − W41 and W42 − no images at each L, 95 % bootstrap interval.
* ``l3_case_<YYYYMMDD>.png`` — observed daily Ap and the W42 / W41 / no-image forecasts over a test window.
* ``l3_lasco_input.png`` — one day of LASCO C2 as W41 saw it (raw, 4 slots) and as W42 sees it (RD, 12 slots).
* ``arch_lead1_fusion_l3.png`` — the l3 model, boxes and arrows.
* ``l4_length_curve.png`` — arm B without AIA / HMI (l4) next to l3, W41 and no images (when l4 has run).

    python scripts/plot_lead1_l3_figures.py [--arms B] [--case-L 1] [--input-day 2024-05-08]
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
from geoindex_daily.frames import cache_path  # noqa: E402

VAULT_FIG = Path.home() / "Vaults/GeoIndex/experiments/figures"
LENGTHS = [1, 2, 3, 4, 5, 6, 7]
ARMS = {"D": ("A_cnn", "Ap", "#1f77b4"), "B": ("B_cnn", "Ap + X-ray", "#2ca02c"),
        "E": ("E_cnn", "Ap + X-ray + wind", "#d62728")}


def seed_mean(d: Path, tag: str, split: str, arm: str, L: int):
    fs = sorted(d.glob(f"{tag}_preds_{split}_{arm}_L{L}_s*.npz"))
    if len(fs) < 3:
        return None
    zs = [np.load(f, allow_pickle=True) for f in fs]
    return pd.DatetimeIndex(zs[0]["dates"]), zs[0]["y"], np.mean([z["pred"] for z in zs], axis=0)


def length_curve(d: Path, out: Path, arms):
    l1 = np.load(d / "l1_preds_A.npz", allow_pickle=True)
    l1d = pd.DatetimeIndex(l1["dates"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for arm in arms:
        ref, label, c = ARMS[arm]
        v = {k: [] for k in ("m3", "c3", "m2", "c2", "mr", "cr")}
        for L in LENGTHS:
            dates, y, p3 = seed_mean(d, "l3", "A", arm, L)
            d2, _, p2 = seed_mean(d, "l2", "A", arm, L)
            p2 = p2[d2.get_indexer(dates)]
            r = l1[f"{ref}_L{L}"][l1d.get_indexer(dates)]
            for k, p in (("3", p3), ("2", p2), ("r", r)):
                v["m" + k].append(np.abs(y - p).mean()); v["c" + k].append(np.corrcoef(y, p)[0, 1])
        for ax, a in ((axes[0], "m"), (axes[1], "c")):
            ax.plot(LENGTHS, v[a + "3"], "-o", color=c, label=f"{label}: W42 (LASCO RD, 12/day)")
            ax.plot(LENGTHS, v[a + "2"], "--^", color=c, alpha=0.6, mfc="white", label=f"{label}: W41 (LASCO raw, 4/day)")
            ax.plot(LENGTHS, v[a + "r"], ":s", color="k", alpha=0.45, ms=4, mfc="white", label=f"{label}: no images")
    pers = l1["persistence"][l1d.get_indexer(dates)]
    axes[0].axhline(np.abs(y - pers).mean(), color="grey", ls="-.", lw=0.8, label="persistence")
    axes[1].axhline(np.corrcoef(y, pers)[0, 1], color="grey", ls="-.", lw=0.8)
    axes[0].set_ylabel("MAE (Ap) — lower is better")
    axes[1].set_ylabel("CC — higher is better")
    for ax in axes:
        ax.set_xlabel("input length L (days)"); ax.set_xticks(LENGTHS); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7.5, loc="center right")
    fig.suptitle(f"Daily Ap one day ahead, split A (test 2022–2025, {len(y):,} days), seed-mean of 3", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "l3_length_curve.png", dpi=150); plt.close(fig)


def l4_curve(d: Path, out: Path):
    """Arm B, split A: without AIA / HMI (l4) next to with them (l3), W41 and no images; MAE and CC against L."""
    l1 = np.load(d / "l1_preds_A.npz", allow_pickle=True)
    l1d = pd.DatetimeIndex(l1["dates"])
    rows = {"l4": [], "l3": [], "l2": [], "none": []}
    for L in LENGTHS:
        dates, y, p4 = seed_mean(d, "l4", "A", "B", L)
        preds = {"l4": p4, "none": l1[f"B_cnn_L{L}"][l1d.get_indexer(dates)]}
        for tag in ("l3", "l2"):
            dd, _, p = seed_mean(d, tag, "A", "B", L)
            preds[tag] = p[dd.get_indexer(dates)]
        for k, p in preds.items():
            rows[k].append((np.abs(y - p).mean(), np.corrcoef(y, p)[0, 1]))
    style = {"l4": ("-o", "#9467bd", "LASCO RD only (no AIA / HMI)"), "l3": ("-s", "#2ca02c", "W42: AIA / HMI + LASCO RD"),
             "l2": ("--^", "orange", "W41: AIA / HMI + LASCO raw"), "none": (":s", "grey", "no images")}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for k, (fmt, c, label) in style.items():
        v = np.array(rows[k])
        for j, ax in enumerate(axes):
            ax.plot(LENGTHS, v[:, j], fmt, color=c, mfc="white" if k != "l4" else c, label=label)
    axes[0].set_ylabel("MAE (Ap) — lower is better"); axes[1].set_ylabel("CC — higher is better")
    for ax in axes:
        ax.set_xlabel("input length L (days)"); ax.set_xticks(LENGTHS); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7.5, loc="upper left")
    fig.suptitle(f"Arm B (Ap + X-ray + images), split A ({len(y):,} test days), seed-mean of 3", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "l4_length_curve.png", dpi=150); plt.close(fig)


def vs_w41(d: Path, out: Path, arms):
    bt = pd.read_csv(d / "l3_summary_bootstrap.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    comps = [("vs W41 image (l2)", "W42 − W41", "o", -0.15), ("vs no images", "W42 − no images", "s", 0.15)]
    for arm in arms:
        _, label, c = ARMS[arm]
        for comp, name, mk, dx in comps:
            for sp, off in (("A", 0.0), ("B", 0.06)):
                s = bt[(bt.arm == arm) & (bt.split == sp) & bt.comparison.str.startswith(comp)].sort_values("L")
                if s.empty:
                    continue
                x = np.array([LENGTHS.index(L) for L in s.L]) + dx + off
                col = c if comp.startswith("vs W41") else "k"
                for ax, v in ((axes[0], "d_mae"), (axes[1], "d_corr")):
                    ax.errorbar(x, s[v], yerr=[s[v] - s[v + "_lo"], s[v + "_hi"] - s[v]], fmt=mk, color=col,
                                capsize=3, mfc=col if sp == "A" else "white", label=f"{label}: {name}, split {sp}")
    for ax, t in ((axes[0], "ΔMAE (Ap) — below 0: W42 better"), (axes[1], "ΔCC — above 0: W42 better")):
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(len(LENGTHS))); ax.set_xticklabels([f"L = {L}" for L in LENGTHS])
        ax.set_ylabel(t); ax.grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=7.5, loc="lower left")
    fig.suptitle("W42 against W41 (same days) and against no images (95 % monthly block bootstrap)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "l3_vs_w41.png", dpi=150); plt.close(fig)


def case(d: Path, out: Path, start: str, end: str, arm: str, L: int):
    split = "A" if pd.Timestamp(start).year >= 2022 else "B"
    l1 = np.load(d / f"l1_preds_{split}.npz", allow_pickle=True)
    l1d = pd.DatetimeIndex(l1["dates"])
    dates, y, p3 = seed_mean(d, "l3", split, arm, L)
    d2, _, p2 = seed_mean(d, "l2", split, arm, L)
    p2 = p2[d2.get_indexer(dates)]
    r = l1[f"{ARMS[arm][0]}_L{L}"][l1d.get_indexer(dates)]
    pers = l1["persistence"][l1d.get_indexer(dates)]
    tgt = dates + pd.Timedelta(days=1)          # forecasts are issued on day t for day t + 1
    m = (tgt >= start) & (tgt <= end)
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.plot(tgt[m], y[m], "-o", color="green", ms=3, label="observed daily Ap")
    ax.plot(tgt[m], p3[m], "--x", color="red", ms=4, label=f"W42 (LASCO RD, 12/day), L = {L}")
    ax.plot(tgt[m], p2[m], "-.", color="orange", label=f"W41 (LASCO raw, 4/day), L = {L}")
    ax.plot(tgt[m], r[m], ":", color="#1f77b4", label=f"no images, L = {L}")
    ax.plot(tgt[m], pers[m], ":", color="grey", label="persistence")
    ax.axhline(30, color="k", lw=0.6, alpha=0.5)
    txt = "\n".join(f"{n}: MAE {np.abs(y[m] - p[m]).mean():.2f}"
                    for n, p in (("W42", p3), ("W41", p2), ("no images", r), ("persistence", pers)))
    ax.text(0.01, 0.97, txt + f"\n{int(m.sum())} days in window", transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6))
    ax.set_ylabel("daily Ap"); ax.set_yscale("symlog", linthresh=30); ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(f"Arm {arm}, one-day-ahead forecasts by target day, {start} → {end} (split {split} test)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / f"l3_case_{pd.Timestamp(start):%Y%m%d}.png", dpi=150); plt.close(fig)


def lasco_input(cache: Path, out: Path, day: str):
    t0 = pd.Timestamp(day)
    fig, axes = plt.subplots(3, 6, figsize=(12, 6.6))
    for k, h in enumerate((0, 6, 12, 18)):
        ax = axes[0, k]
        ax.imshow(np.load(cache_path(cache, "lasco_c2", t0 + pd.Timedelta(hours=h))), cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"raw {h:02d} UT", fontsize=8)
    for ax in axes[0, 4:]:
        ax.set_visible(False)
    for k, h in enumerate(range(0, 24, 2)):
        ax = axes[1 + k // 6, k % 6]
        ax.imshow(np.load(cache_path(cache, "lasco_c2_rd12", t0 + pd.Timedelta(hours=h))), cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"RD {h:02d} UT", fontsize=8)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"LASCO C2 on {t0:%Y-%m-%d}: W41 input (top: raw brightness, 4 slots) and "
                 "W42 input (rows 2–3: running difference, 12 slots; grey = no change)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "l3_lasco_input.png", dpi=120); plt.close(fig)


def architecture(out: Path):
    fig, ax = plt.subplots(figsize=(10, 4.6)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 5.2)

    def box(x, y, w, h, t, c):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=c, ec="k", lw=0.8))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=8)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="->", lw=0.9))

    box(0.1, 3.6, 1.9, 1.3, "SDO frames\n3 channels × 4L\n1024 × 1024\n(AIA 193, 211, HMI)", "#fde9c8")
    box(0.1, 2.0, 1.9, 1.3, "LASCO C2 running\ndifference\n1 channel × 12L\n1024 × 1024", "#f8d7d0")
    for y0, lab in ((3.75, "SDO branch"), (2.15, "LASCO branch")):
        box(2.4, y0, 1.3, 1.0, "stem\nConv3d 1×4×4\nstride 4 → 256²", "#f3f3f3")
        box(4.1, y0, 1.9, 1.0, "4 × (2+1)D block\n3×3 spatial + 3 temporal\n32→64→128→256, /2 each", "#f3f3f3")
        box(6.4, y0, 1.0, 1.0, "pool\n(T, H, W)\n→ 256", "#f3f3f3")
        ax.text(5.05, y0 + 1.13, lab, ha="center", fontsize=8.5)
        for a in ((2.0, y0 + 0.5, 2.4, y0 + 0.5), (3.7, y0 + 0.5, 4.1, y0 + 0.5), (6.0, y0 + 0.5, 6.4, y0 + 0.5),
                  (7.4, y0 + 0.5, 7.8, 2.4)):
            arrow(*a)
    box(0.1, 0.3, 1.9, 1.2, "daily features × L\nD: Ap · B: Ap, X-ray (2)\nE: Ap, X-ray (2),\nwind (5)", "#d9e8f5")
    box(2.4, 0.4, 3.6, 1.0, "1-D CNN (the l1 model)\n2 × Conv1d k = 3, 32 ch, GELU, pool → 32", "#f3f3f3")
    box(7.8, 1.9, 1.0, 1.0, "concat 544\nMLP 128\n→ 1", "#e2f0d9")
    box(9.0, 2.0, 0.9, 0.8, "log1p Ap\nday t + 1", "#e2f0d9")
    for a in ((2.0, 0.9, 2.4, 0.9), (6.0, 0.9, 7.8, 2.1), (8.8, 2.4, 9.0, 2.4)):
        arrow(*a)
    fig.tight_layout()
    fig.savefig(out / "arch_lead1_fusion_l3.png", dpi=150); plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=VAULT_FIG)
    p.add_argument("--arms", nargs="*", default=["B"])
    p.add_argument("--case-arm", default="B")
    p.add_argument("--case-L", type=int, default=1)
    # May 2024 is not in the l3 test set (2024-05-09 lacks two LASCO slots), so the cases are windows that are in it
    p.add_argument("--cases", nargs="*", default=["2023-02-01:2023-03-31", "2025-10-25:2025-12-10"])
    p.add_argument("--input-day", default="2024-05-08")
    args = p.parse_args()
    data = default_data_dir()
    d = data / "lead1"
    args.out.mkdir(parents=True, exist_ok=True)
    length_curve(d, args.out, args.arms)
    if seed_mean(d, "l4", "A", "B", LENGTHS[-1]) is not None:
        l4_curve(d, args.out)
    vs_w41(d, args.out, args.arms)
    for c in args.cases:
        case(d, args.out, *c.split(":"), args.case_arm, args.case_L)
    lasco_input(data / "frames1024", args.out, args.input_day)
    architecture(args.out)
    print(f"wrote figures to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
