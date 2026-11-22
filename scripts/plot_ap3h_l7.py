"""Figure of phase l7 (Models 0 / A / B, L = 4): 3-hourly ap MAE per seed and of the seed ensemble, by horizon and day.

Reads <compare>/l7_absolute.csv (scripts/ap3h_l7_compare.py) and writes <compare>/fig_l7_seed_mae.png.

    python scripts/plot_ap3h_l7.py --compare ~/Projects/GeoIndex/daily/ap3h-l7/compare
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

MODELS = [("0", "Model 0", "#8c8c8c"), ("A", "Model A (+ CME)", "#1f4e79"), ("B", "Model B (+ CME + footpoint)", "#b04a2f")]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--compare", type=Path, required=True)
    args = p.parse_args()
    a = pd.read_csv(args.compare / "l7_absolute.csv")
    cases = [(24, 1), (48, 1), (48, 2), (72, 1), (72, 2), (72, 3)]
    fig, ax = plt.subplots(figsize=(10, 4))
    for j, (m, lab, c) in enumerate(MODELS):
        for i, (h, k) in enumerate(cases):
            g = a[(a.horizon_h == h) & (a.target_day == k) & (a.model == m)]
            x = i + (j - 1) * 0.22
            seeds = g[g.forecast != "ensemble"].mae_3h
            ax.scatter([x] * len(seeds), seeds, s=22, color=c, alpha=0.55, edgecolor="none")
            ax.scatter([x], g[g.forecast == "ensemble"].mae_3h, s=60, marker="_", color=c, linewidths=2.2,
                       label=lab if i == 0 else None)
    ax.set_xticks(range(len(cases)), [f"{h} h, day {k}" for h, k in cases])
    ax.set_ylabel("3-hourly ap MAE")
    ax.grid(axis="y", color="#e5e5e5", lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("Models 0 / A / B at L = 4, 1,273 test issues: dots = seeds 0–2, bar = seed ensemble", fontsize=9)
    fig.tight_layout(); fig.savefig(args.compare / "fig_l7_seed_mae.png", dpi=150); plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
