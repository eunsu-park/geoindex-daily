"""Forecasts issued before a CME arrives (physical analysis, Task 5 Q2; work order 2026-11-17).

For every shock at Earth that DONKI links to a CME (2022–2025, with test forecasts) and every forecast origin O (00 UT,
issue date + 1 day) with O < shock time S ≤ O + horizon, the bins of that forecast in [S, min(S + 24 h, O + horizon))
— the first post-arrival day the pre-arrival issue covers — are scored: MAE and bias against the observed ap, the
maximum P(ap ≥ 48), and the anticipated rise = the forecast's mean in those bins minus its mean in its own bins before S
(observed rise alongside). Lead to arrival S − O is grouped 0–24 h (24-h model), and 0–24 / 24–48 / 48–72 h (72-h
model). One forecast per shock and lead group, so a storm counts once per group; differences full − time series with
95 % intervals from resampling shocks (2,000). As a contrast, the issue at the first 00 UT after S (post-arrival) is
scored on its own first day.

    python scripts/ap3h_pre_arrival.py --out ~/Projects/GeoIndex/daily/physical_analysis
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
from scipy.stats import norm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from geoindex_daily.daily_index import default_data_dir  # noqa: E402

RUNS = {(24, "full"): (["ap3h-l6/gpu2", "ap3h-l6/gpu3"], "l6", "full"), (24, "ts"): (["ap3h"], "l6", "ts"),
        (72, "full"): (["ap3h-l6h/gpu2", "ap3h-l6h/gpu3"], "l6h3", "full"), (72, "ts"): (["ap3h-h"], "l6h3", "ts")}


def load(data, h, model, L):
    dirs, tag, m = RUNS[(h, model)]
    fs = sorted(f for d in dirs for f in glob.glob(str(data / d / f"{tag}_preds_A_{m}_all_L{L}_s*.npz")))
    z = [np.load(f) for f in fs]
    assert len(z) == 3
    mu, sg = np.stack([x["mu"] for x in z]), np.stack([x["sigma"] for x in z])
    origin = pd.DatetimeIndex(z[0]["dates"]).normalize() + pd.Timedelta(days=1)
    return {o: (np.expm1(mu[:, i]).mean(0), (1 - norm.cdf((np.log1p(48.0) - mu[:, i]) / sg[:, i])).mean(0))
            for i, o in enumerate(origin)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--L", type=int, default=4)
    p.add_argument("--suffix", default="", help="appended to the output names, e.g. _L1")
    args = p.parse_args()
    data = default_data_dir()
    f3 = pd.read_parquet(data / "features_3h.parquet")["ap"]
    ips = pd.read_parquet(data / "events" / "donki_ips_earth.parquet")
    ips = ips[(ips.linked_cmes != "") & (ips.shock_time >= "2022-01-01") & (ips.shock_time < "2026-01-01")]
    fc = {(h, m): load(data, h, m, args.L) for (h, m) in RUNS}
    rows = []
    for s in ips.itertuples():
        S = s.shock_time
        obs_day = f3.loc[S.floor("3h"): S.floor("3h") + pd.Timedelta(hours=21)]
        storm = bool(obs_day.max() >= 48)
        for h in (24, 72):
            H = h // 3
            # pre-arrival issues, one per lead group; and the first post-arrival issue
            cands = [(S.normalize() - pd.Timedelta(days=g), f"{24 * g}–{24 * (g + 1)} h") for g in range(h // 24)]
            cands.append((S.normalize() + pd.Timedelta(days=1), "after arrival"))
            for O, grp in cands:
                if not all(O in fc[(h, m)] for m in ("full", "ts")) or (grp != "after arrival" and O >= S):
                    continue
                t = O + pd.to_timedelta(3 * np.arange(H), unit="h")
                lo = S if grp != "after arrival" else O
                post = (t + pd.Timedelta(hours=3) > lo) & (t < min(lo + pd.Timedelta(hours=24), O + pd.Timedelta(hours=h)))
                pre = t + pd.Timedelta(hours=3) <= S
                if grp == "after arrival":
                    post = (t >= O) & (t < O + pd.Timedelta(hours=24))
                if post.sum() == 0:
                    continue
                y = f3.reindex(t).to_numpy()
                row = {"shock": S, "ips_id": s.ips_id, "storm": storm, "max_ap_after": obs_day.max(), "horizon_h": h,
                       "lead_group": grp, "origin": O, "lead_to_arrival_h": (S - O) / pd.Timedelta(hours=1),
                       "n_bins": int(post.sum()), "obs_rise": (y[post].mean() - y[pre].mean()) if pre.any() else np.nan}
                for m in ("full", "ts"):
                    pr, p48 = fc[(h, m)][O]
                    row.update({f"{m}_mae": np.abs(pr[post] - y[post]).mean(), f"{m}_bias": (pr[post] - y[post]).mean(),
                                f"{m}_p48_max": p48[post].max(),
                                f"{m}_rise": (pr[post].mean() - pr[pre].mean()) if pre.any() else np.nan})
                rows.append(row)
    D = pd.DataFrame(rows)
    D.to_csv(args.out / f"pre_arrival_forecasts{args.suffix}.csv", index=False)

    rng = np.random.default_rng(0)
    summ = []
    for (h, grp, sub), g in D.groupby(["horizon_h", "lead_group", D.storm.map({True: "storm", False: "no storm"})]):
        r = {"horizon_h": h, "lead_group": grp, "subset": sub, "n_shocks": len(g)}
        for col in ("mae", "bias", "p48_max", "rise"):
            d = (g[f"full_{col}"] - g[f"ts_{col}"]).dropna().to_numpy()
            r[f"full_{col}"], r[f"ts_{col}"] = g[f"full_{col}"].mean(), g[f"ts_{col}"].mean()
            if len(d) > 2:
                bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
                r.update({f"d_{col}": d.mean(), f"d_{col}_lo": np.percentile(bs, 2.5), f"d_{col}_hi": np.percentile(bs, 97.5)})
        r["obs_rise"] = g.obs_rise.mean()
        r["full_detect_p30"] = (g.full_p48_max >= 0.3).mean()
        r["ts_detect_p30"] = (g.ts_p48_max >= 0.3).mean()
        summ.append(r)
    S_ = pd.DataFrame(summ)
    S_.to_csv(args.out / f"pre_arrival_summary{args.suffix}.csv", index=False)

    order = ["0–24 h", "24–48 h", "48–72 h", "after arrival"]
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    x = S_[(S_.horizon_h == 72) & (S_.subset == "storm")].set_index("lead_group").reindex(order)
    for a, (col, lab) in zip(ax, (("rise", "anticipated rise of the forecast (ap)"), ("p48_max", "max P(ap ≥ 48) after arrival"),
                                   ("mae", "MAE after arrival"))):
        a.plot(range(4), x[f"full_{col}"], "o-", color="#1f4e79", label="full")
        a.plot(range(4), x[f"ts_{col}"], "s--", color="#8c8c8c", label="time series")
        if col == "rise":
            a.plot(range(4), x["obs_rise"], "^:", color="#000000", label="observed rise")
        a.set_xticks(range(4), ["issued 0–24 h\nbefore", "24–48 h\nbefore", "48–72 h\nbefore", "first issue\nafter"], fontsize=8)
        a.set_ylabel(lab, fontsize=8); a.grid(color="#e5e5e5", lw=0.8); a.set_axisbelow(True)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    ax[0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"72-h model, CME-linked shocks followed by ap ≥ 48 within 24 h (n = {int(x.n_shocks.max())}), L = {args.L}; "
                 "forecast bins of the first day after arrival", fontsize=9)
    fig.tight_layout(); fig.savefig(args.out / f"fig4_pre_arrival{args.suffix}.png", dpi=150); plt.close(fig)
    pd.set_option("display.width", 300)
    cols = ["horizon_h", "lead_group", "subset", "n_shocks", "obs_rise", "full_rise", "ts_rise", "d_rise", "d_rise_lo", "d_rise_hi",
            "full_p48_max", "ts_p48_max", "full_detect_p30", "ts_detect_p30", "full_mae", "ts_mae", "d_mae", "d_mae_lo", "d_mae_hi",
            "full_bias", "ts_bias"]
    print(S_[cols].round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
