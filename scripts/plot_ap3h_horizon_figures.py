"""Report figures for the l6 48 / 72-hour runs (vault planning/ap3h-plan.md §9).

* ``ap3h_by_day.png`` — daily-Ap MAE and CC of target days 1–3: 72-h full model, 72-h time-series model, RSGA
  day 1–3, persistence, and the pilot blend with RSGA (from ap3h_pilot_rsga_enlil.py).
* ``ap3h_image_effect_by_day.png`` — full − time-series model (72 h) per L and target day, 95 % intervals.

    python scripts/plot_ap3h_horizon_figures.py --full ~/Projects/GeoIndex/daily/ap3h-l6h/gpu3 \
        --ts ~/Projects/GeoIndex/daily/ap3h-h --pilot ~/Projects/GeoIndex/daily/ap3h-l6h/pilot_rsga_enlil.csv
"""
import argparse
import glob
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

VAULT_FIG = Path.home() / "Vaults/GeoIndex/experiments/figures"
C_FULL, C_TS, C_REF, C_BLEND = "#1f4e79", "#8c8c8c", "#b04a2f", "#6a3d9a"


def load(dirs, tag, model, L):
    fs = sorted(f for d in dirs for f in glob.glob(f"{d}/{tag}_preds_A_{model}_all_L{L}_s*.npz"))
    z = [np.load(f) for f in fs]
    assert len(z) == 3, (tag, model, L, len(z))
    return pd.DatetimeIndex(z[0]["dates"]), np.round(z[0]["y"], 6), np.mean([np.expm1(x["mu"]) for x in z], axis=0)


def style(a):
    a.grid(color="#e5e5e5", lw=0.8); a.set_axisbelow(True)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--full", nargs="+", required=True)
    p.add_argument("--ts", nargs="+", required=True)
    p.add_argument("--pilot", type=Path, required=True)
    p.add_argument("--L", type=int, default=4)
    p.add_argument("--out", type=Path, default=VAULT_FIG)
    args = p.parse_args()
    data = default_data_dir()
    rsga = pd.read_parquet(data / "swpc" / "rsga_ap.parquet")
    f3 = pd.read_parquet(data / "features_3h.parquet")["ap"]
    pilot = pd.read_csv(args.pilot).set_index("day")

    d, y, pf = load(args.full, "l6h3", "full", args.L)
    _, _, pt = load(args.ts, "l6h3", "ts", args.L)
    i0 = f3.index.get_indexer(d)
    pers = np.array([f3.iloc[i: i + 8].mean() for i in i0])                    # the issue day's Ap
    days = [1, 2, 3]
    res = {k: {} for k in ("full", "ts", "rsga", "pers")}
    for k in days:
        sl = slice((k - 1) * 8, k * 8)
        yo = y[:, sl].mean(1)
        r = rsga[(rsga.kind == "predicted") & (rsga.lead == k)].drop_duplicates("target_date", keep="last")
        rg = r.set_index(pd.DatetimeIndex(r.target_date).as_unit("ns"))["ap"].reindex(d + pd.Timedelta(days=k)).to_numpy()
        ok = ~np.isnan(rg)
        for name, f in (("full", pf[:, sl].mean(1)), ("ts", pt[:, sl].mean(1)), ("rsga", rg), ("pers", pers)):
            res[name][k] = (np.abs(f[ok] - yo[ok]).mean(), np.corrcoef(f[ok], yo[ok])[0, 1])
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for j, lab in enumerate(("daily Ap MAE", "daily Ap CC")):
        a = ax[j]
        for name, c, mk, txt in (("full", C_FULL, "o-", f"full model (72 h, L = {args.L})"),
                                 ("ts", C_TS, "s--", "time series only"), ("rsga", C_REF, "D-.", "SWPC RSGA day k"),
                                 ("pers", "#333333", "^:", "persistence (issue day's Ap)")):
            a.plot(days, [res[name][k][j] for k in days], mk, color=c, lw=2, ms=6, label=txt)
        col = "blend_mae" if j == 0 else "blend_cc"
        a.plot(days, [pilot.loc[k, col] for k in days], "v-", color=C_BLEND, lw=1.5, ms=6,
               label="pilot: model–RSGA blend (§7)")
        a.set_xticks(days, ["day 1", "day 2", "day 3"]); a.set_ylabel(lab); style(a)
    ax[1].legend(fontsize=7.5, frameon=False)
    fig.suptitle("l6 72-hour model by target day, split A test (1,261 days with RSGA); seed mean of 3", fontsize=9)
    fig.tight_layout(); fig.savefig(args.out / "ap3h_by_day.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for a, k in zip(ax, days):
        sl = slice((k - 1) * 8, k * 8)
        for L in range(1, 8):
            dd, yy, f = load(args.full, "l6h3", "full", L)
            _, _, t = load(args.ts, "l6h3", "ts", L)
            yo = yy[:, sl].mean(1)
            b = block_bootstrap_diff(yo, f[:, sl].mean(1), t[:, sl].mean(1), dd, 2000, 0)
            sig = b["d_mae_lo"] > 0 or b["d_mae_hi"] < 0
            c = C_FULL if sig else "#8c8c8c"
            a.plot([L, L], [b["d_mae_lo"], b["d_mae_hi"]], color=c, lw=2)
            a.plot(L, b["d_mae"], "o", ms=7, color=c, mfc=c if sig else "white", mew=2)
        a.axhline(0, color="#333333", lw=1); a.set_title(f"target day {k}", fontsize=9)
        a.set_xticks(range(1, 8)); a.set_xlabel("input length L (days)"); style(a)
    ax[0].set_ylabel("Δ daily Ap MAE, full − time series", fontsize=8)
    fig.suptitle("Image effect of the 72-hour model by target day; 95 % monthly block bootstrap, filled = excludes zero",
                 fontsize=9)
    fig.tight_layout(); fig.savefig(args.out / "ap3h_image_effect_by_day.png", dpi=150); plt.close(fig)
    for k in days:
        print(k, {n: tuple(round(v, 3) for v in res[n][k]) for n in res})
    print("wrote figures to", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
