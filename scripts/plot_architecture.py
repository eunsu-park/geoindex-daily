"""Draw the fusion-architecture figures for the weekly reports (boxes and arrows, no weights).

Two panels are produced:

* ``arch_lora_fusion.png`` — the i3 model: Surya backbone (frozen or LoRA) → 8×8 grid →
  attention pooling → concat with the 30-day Ap history → MLP → 60 daily Ap values.
* ``arch_frozen_fusion.png`` — the f1 model: frozen Surya mean token → PCA-16 → ridge
  together with the 30-day Ap history (and the MOMENT branch tried in t1/t2).

Usage:
    python scripts/plot_architecture.py [--out DIR]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

VAULT_FIG = Path.home() / "Vaults/Research/GeoIndex/experiments/figures"

C_IMG = "#d9e8f5"     # image branch
C_TS = "#dff0d8"      # time-series branch
C_FUSE = "#fbe3c8"    # fusion / head
C_OUT = "#f2f2f2"
C_FROZEN = "#e8e8e8"
C_LORA = "#f8d0d0"


def box(ax, x, y, w, h, text, fc, fs=9, ec="black", lw=1.0, ls="-", weight=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc=fc, ec=ec, lw=lw, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight=weight)


def arrow(ax, x0, y0, x1, y1, text=None, fs=8, color="black"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12,
                                 color=color, lw=1.2, shrinkA=0, shrinkB=0))
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.22, text, ha="center", va="bottom", fontsize=fs, color="dimgray")


def lora_panel(out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(14, 5.4))
    ax.set_xlim(0, 14); ax.set_ylim(0, 5.2); ax.axis("off")

    # image branch (top row)
    box(ax, 0.3, 3.2, 2.0, 1.4, "SuryaBench frames\n13 channels × 2 times\n(t − 60 min, t)\n4096 × 4096 each", C_IMG)
    arrow(ax, 2.3, 3.9, 2.9, 3.9)
    box(ax, 2.9, 3.2, 2.0, 1.4, "Surya patch embedding\n26 ch → 16 × 16 patches\n65,536 tokens × 1,280\n(frozen, cached per day)", C_FROZEN)
    arrow(ax, 4.9, 3.9, 5.5, 3.9)
    box(ax, 5.5, 3.2, 2.6, 1.4, "Surya backbone (366M)\n2 spectral blocks: frozen\n8 attention blocks:\nfrozen  or  LoRA r=8 on qkv/proj", C_LORA)
    arrow(ax, 8.1, 3.9, 8.7, 3.9)
    box(ax, 8.7, 3.2, 2.0, 1.4, "8 × 8 grid pool\nLN → Linear 1,280→128\n1 learned query\nattention over 64 cells", C_FUSE)
    ax.text(9.7, 3.05, "image vector 128-d", ha="center", va="top", fontsize=8, color="dimgray")

    # time-series branch (bottom row)
    box(ax, 0.3, 0.9, 2.0, 1.2, "Daily Ap history\n30 values (t−29 … t)\nlog1p, standardised", C_TS)
    arrow(ax, 2.3, 1.5, 2.9, 1.5)
    box(ax, 2.9, 0.9, 2.0, 1.2, "used as-is\n(30-d vector)\n[MOMENT branch: t2,\nno gain over raw]", C_TS, fs=8.5)
    ax.text(4.9 + 0.3, 1.25, "TS 30-d", ha="left", va="top", fontsize=8, color="dimgray")

    # fusion
    box(ax, 11.7, 1.9, 2.1, 1.5, "concat\n128 + 30 = 158\nMLP 256 → 256 → 60\n(GELU, dropout 0.1)", C_FUSE, weight="bold")
    arrow(ax, 10.7, 3.6, 12.75, 3.4)
    arrow(ax, 4.9, 1.5, 12.75, 1.9)
    arrow(ax, 12.75, 1.9, 12.75, 0.85)
    box(ax, 11.7, 0.15, 2.1, 0.7, "Ap(t+1) … Ap(t+60)", C_OUT, weight="bold")

    # arms legend
    ax.text(0.3, 2.65, "Arms:  LoRA + TS = backbone adapters trained;   Frozen + grid head = backbone fixed, only pool + MLP trained;\n"
            "TS-only head = image branch removed, same MLP on the 30-d vector.", fontsize=8.5, va="center")
    ax.text(0.3, 0.35, "Loss: MSE on standardised log1p Ap, 60 outputs.   Trainable: LoRA 0.49M + pool/head 0.31M.\n"
            "Batch 1 × 16-step accumulation, bf16 backbone, fp32 head.", fontsize=8.5, va="center", color="dimgray")
    ax.set_title("i3 — LoRA fusion: Surya (image FM) + 30-day Ap history → 60-day daily Ap", fontsize=11, fontweight="bold")
    f = out / "arch_lora_fusion.png"
    fig.savefig(f, dpi=140, bbox_inches="tight"); plt.close(fig)
    return f


def frozen_panel(out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(14, 4.6))
    ax.set_xlim(0, 14); ax.set_ylim(0, 4.6); ax.axis("off")

    box(ax, 0.3, 2.7, 2.0, 1.4, "SuryaBench frames\n13 channels × 2 times\n(t − 60 min, t)", C_IMG)
    arrow(ax, 2.3, 3.4, 2.9, 3.4)
    box(ax, 2.9, 2.7, 2.6, 1.4, "Surya, all frozen\npatch embed + 10 blocks\n65,536 tokens × 1,280", C_FROZEN)
    arrow(ax, 5.5, 3.4, 6.1, 3.4)
    box(ax, 6.1, 2.7, 2.0, 1.4, "mean over tokens\n1,280-d\n(cached: 5,258 days)", C_FUSE)
    arrow(ax, 8.1, 3.4, 8.7, 3.4)
    box(ax, 8.7, 2.7, 1.6, 1.4, "PCA → 16\n(fit on train)", C_FUSE)

    box(ax, 0.3, 0.7, 2.0, 1.2, "Daily Ap history\n30 values, log1p", C_TS)
    arrow(ax, 2.3, 1.3, 2.9, 1.3)
    box(ax, 2.9, 0.7, 2.6, 1.2, "raw 30-d vector\n— or —\nMOMENT-large embedding\n(1,024-d, frozen; t1)", C_TS, fs=8.5)

    box(ax, 11.5, 1.4, 2.2, 1.6, "Ridge regression\n(stage 1)\n— or —\nMLP 256→256→60\n(stage 2)", C_FUSE, weight="bold")
    arrow(ax, 10.3, 3.4, 12.6, 3.0)
    arrow(ax, 5.5, 1.3, 12.6, 1.4)
    arrow(ax, 12.6, 1.4, 12.6, 0.75)
    box(ax, 11.5, 0.1, 2.2, 0.65, "Ap(t+1) … Ap(t+60)", C_OUT, weight="bold")
    ax.text(0.3, 2.3, "Controls: TS only (ridge on the 30-d vector);   PHASE = TS + log F10.7 + sunspot number;\nIMG1 / IMG7 = token of the issue day / mean over the last 7 days.",
            fontsize=8.5, va="center")
    ax.set_title("f1 — frozen fusion: Surya mean token (PCA-16) + Ap history → ridge / MLP", fontsize=11, fontweight="bold")
    f = out / "arch_frozen_fusion.png"
    fig.savefig(f, dpi=140, bbox_inches="tight"); plt.close(fig)
    return f


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(VAULT_FIG))
    args = p.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for f in (lora_panel(out), frozen_panel(out)):
        print(f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
