"""Report figures for phase l6 (3-hour ap, the next 24 h; vault planning/ap3h-plan.md).

Reads the l6 forecasts (full and CME-only models from the remote runs, the time-series model from this device)
and writes PNGs to --out (default: the vault's experiments/figures):

* ``ap3h_length_curve.png`` — daily-Ap MAE and CC against L: full model, time-series model, RSGA day 1, persistence.
* ``ap3h_effects.png`` — full − time-series model (all test days) and CME-only − full model (CME days), 95 % intervals.
* ``ap3h_lead_profile.png`` — MAE of each 3-hour bin of the target day: full and time-series model, persistence.
* ``ap3h_case_<YYYYMMDD>.png`` — observed 3-hour ap with the full model's median and 68 / 95 % bands.

    python scripts/plot_ap3h_figures.py --full ~/Projects/GeoIndex/daily/ap3h-l6/gpu2 ~/Projects/GeoIndex/daily/ap3h-l6/gpu3 \
        --ts ~/Projects/GeoIndex/daily/ap3h --case 2024-05-01 2024-05-21 --case-L 4
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
LENGTHS = range(1, 8)
C_FULL, C_TS, C_REF = "#1f4e79", "#8c8c8c", "#b04a2f"


def load(dirs, tag, model, subset, L):
    """dates, y (N, 8) ap, list of (mu, sigma) per seed."""
    fs = sorted(f for d in dirs for f in glob.glob(f"{d}/{tag}_preds_A_{model}_{subset}_L{L}_s*.npz"))
    z = [np.load(f) for f in fs]
    assert len(z) == 3, (tag, model, subset, L, len(z))
    return pd.DatetimeIndex(z[0]["dates"]), np.round(z[0]["y"], 6), [(x["mu"], x["sigma"]) for x in z]


def point(seeds):
    return np.mean([np.expm1(m) for m, _ in seeds], axis=0)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--full", nargs="+", required=True)
    p.add_argument("--ts", nargs="+", required=True)
    p.add_argument("--case", nargs=2, action="append", default=[], metavar=("START", "END"))
    p.add_argument("--case-L", type=int, default=4)
    p.add_argument("--out", type=Path, default=VAULT_FIG)
    args = p.parse_args()
    data = default_data_dir()
    args.out.mkdir(parents=True, exist_ok=True)
    rsga = pd.read_parquet(data / "swpc" / "rsga_ap.parquet")
    rsga = rsga[(rsga.kind == "predicted") & (rsga.lead == 1)].drop_duplicates("target_date", keep="last")
    rsga = rsga.set_index(pd.DatetimeIndex(rsga.target_date).as_unit("ns"))["ap"]
    f3 = pd.read_parquet(data / "features_3h.parquet")["ap"]

    # ---- length curve and effects ---------------------------------------------------------------
    rows = []
    for L in LENGTHS:
        d, y, sf = load(args.full, "l6", "full", "all", L)
        _, _, st = load(args.ts, "l6", "ts", "all", L)
        dc, yc, sc = load(args.full, "l6cme", "full", "cme", L)
        yo, pf, pt = y.mean(1), point(sf).mean(1), point(st).mean(1)
        pos = d.get_indexer(dc)
        b1 = block_bootstrap_diff(yo, pf, pt, d, 2000, 0)
        b2 = block_bootstrap_diff(yc.mean(1), point(sc).mean(1), pf[pos], dc, 2000, 0)
        rg = rsga.reindex(d + pd.Timedelta(days=1)).to_numpy()
        ok = ~np.isnan(rg)
        pers = np.array([f3.iloc[f3.index.get_indexer([t])[0]: f3.index.get_indexer([t])[0] + 8].mean() for t in d])
        rows.append({"L": L, "full_mae": np.abs(pf - yo).mean(), "ts_mae": np.abs(pt - yo).mean(),
                     "full_cc": np.corrcoef(pf, yo)[0, 1], "ts_cc": np.corrcoef(pt, yo)[0, 1],
                     "rsga_mae": np.abs(rg[ok] - yo[ok]).mean(), "rsga_cc": np.corrcoef(rg[ok], yo[ok])[0, 1],
                     "pers_mae": np.abs(pers - yo).mean(), "pers_cc": np.corrcoef(pers, yo)[0, 1],
                     **{f"img_{k}": v for k, v in b1.items()}, **{f"cme_{k}": v for k, v in b2.items()}})
    t = pd.DataFrame(rows)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for a, m, lab in ((ax[0], "mae", "daily Ap MAE"), (ax[1], "cc", "daily Ap CC")):
        a.plot(t.L, t[f"full_{m}"], "o-", color=C_FULL, lw=2, ms=6, label="full model (images + time series)")
        a.plot(t.L, t[f"ts_{m}"], "s--", color=C_TS, lw=2, ms=6, label="time series only")
        a.axhline(t[f"rsga_{m}"].iloc[0], color=C_REF, lw=1.5, ls="-.", label="SWPC RSGA day 1 (1,261 days)")
        a.axhline(t[f"pers_{m}"].iloc[0], color="#333333", lw=1, ls=":", label="persistence")
        a.set_xlabel("input length L (days)"); a.set_ylabel(lab); a.set_xticks(list(LENGTHS))
        a.grid(color="#e5e5e5", lw=0.8); a.set_axisbelow(True)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    ax[0].legend(fontsize=8, frameon=False)
    fig.suptitle("l6, split A test 2022–2025 (1,273 days): daily Ap = mean of the eight 3-hour forecasts; seed mean of 3",
                 fontsize=9)
    fig.tight_layout(); fig.savefig(args.out / "ap3h_length_curve.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=False)
    for a, k, title in ((ax[0], "img", "full − time series, all 1,273 test days"),
                        (ax[1], "cme", "CME-only − full model, 859 CME test days")):
        for _, r in t.iterrows():
            sig = r[f"{k}_d_mae_lo"] > 0 or r[f"{k}_d_mae_hi"] < 0
            c = C_FULL if sig else "#8c8c8c"
            a.plot([r.L, r.L], [r[f"{k}_d_mae_lo"], r[f"{k}_d_mae_hi"]], color=c, lw=2)
            a.plot(r.L, r[f"{k}_d_mae"], "o", ms=8, color=c, mfc=c if sig else "white", mew=2)
        a.axhline(0, color="#333333", lw=1)
        a.set_title(title, fontsize=9); a.set_xlabel("input length L (days)"); a.set_xticks(list(LENGTHS))
        a.set_ylabel("Δ daily Ap MAE (negative = first model better)", fontsize=8)
        a.grid(axis="y", color="#e5e5e5", lw=0.8); a.set_axisbelow(True)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.suptitle("95 % monthly block bootstrap; filled = interval excludes zero", fontsize=9)
    fig.tight_layout(); fig.savefig(args.out / "ap3h_effects.png", dpi=150); plt.close(fig)

    # ---- lead profile at the case L ----------------------------------------------------------------
    L = args.case_L
    d, y, sf = load(args.full, "l6", "full", "all", L)
    _, _, st = load(args.ts, "l6", "ts", "all", L)
    i0 = f3.index.get_indexer(d)
    pers = np.stack([f3.to_numpy()[i: i + 8] for i in i0])            # the issue day's eight bins
    fig, a = plt.subplots(figsize=(6.4, 3.6))
    x = np.arange(8) * 3 + 1.5
    for arr, c, lab, mk in ((point(sf), C_FULL, "full model", "o-"), (point(st), C_TS, "time series only", "s--"),
                            (pers, "#333333", "persistence (issue day's bins)", "^:")):
        a.plot(x, np.abs(arr - y).mean(0), mk, color=c, lw=2, ms=6, label=lab)
    a.set_xticks(np.arange(0, 25, 3)); a.set_xlabel("UT of the target day (h)"); a.set_ylabel("3-hour ap MAE")
    a.legend(fontsize=8, frameon=False); a.grid(color="#e5e5e5", lw=0.8); a.set_axisbelow(True)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    a.set_title(f"l6, L = {L}: error of each 3-hour bin (1,273 test days)", fontsize=9)
    fig.tight_layout(); fig.savefig(args.out / "ap3h_lead_profile.png", dpi=150); plt.close(fig)

    # ---- cases: 3-hour ap with the mixture bands -------------------------------------------------
    rng = np.random.default_rng(0)
    for start, end in args.case:
        tgt = d + pd.Timedelta(days=1)
        sel = (tgt >= pd.Timestamp(start)) & (tgt <= pd.Timestamp(end))
        dd = d[sel]
        smp = np.concatenate([np.expm1(m[sel][..., None] + s[sel][..., None] * rng.standard_normal(m[sel].shape + (300,)))
                              for m, s in sf], axis=-1)
        q = np.quantile(smp, [0.025, 0.16, 0.5, 0.84, 0.975], axis=-1)
        tt = (dd + pd.Timedelta(days=1)).to_numpy()[:, None] + (np.arange(8) * np.timedelta64(3, "h"))[None, :]
        tt = tt.ravel()
        fig, a = plt.subplots(figsize=(10, 3.6))
        a.fill_between(tt, q[0].ravel(), q[4].ravel(), color=C_FULL, alpha=0.15, step="post", lw=0, label="95 % band")
        a.fill_between(tt, q[1].ravel(), q[3].ravel(), color=C_FULL, alpha=0.3, step="post", lw=0, label="68 % band")
        a.step(tt, q[2].ravel(), where="post", color=C_FULL, lw=1.5, label="forecast median")
        a.step(tt, y[sel].ravel(), where="post", color="#000000", lw=1.2, label="observed 3-hour ap")
        a.set_ylabel("3-hour ap"); a.legend(fontsize=8, frameon=False, ncol=4, loc="upper left")
        a.grid(color="#e5e5e5", lw=0.8); a.set_axisbelow(True)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.set_title(f"l6 full model, L = {L}: each day's eight bins forecast at 00 UT ({start} → {end} target days)",
                    fontsize=9)
        fig.tight_layout(); fig.savefig(args.out / f"ap3h_case_{pd.Timestamp(start):%Y%m%d}.png", dpi=150); plt.close(fig)
    print(t.round(3).to_string(index=False))
    print("wrote figures to", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
