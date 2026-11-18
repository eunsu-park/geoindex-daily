"""Figures of the SWPC 3-hourly Kp → ap benchmark (outputs of ap3h_swpc_benchmark.py), into <out>/figures/.

* ap3h_lead_step_mae.png — MAE (top) and number of issue dates by 3-hour lead step 1–24: GeoIndex full models
  (24 / 48 / 72 h, L = 4, seed-ensemble forecast) and SWPC Kp → ap, on each horizon's matched issue dates.
* ap3h_lead_day_mae_cc.png — 3-hourly MAE and CC by target day: GeoIndex full models (range over L = 1–7 as bars,
  L = 4 as markers) and SWPC.
* 24_48_72_day1_common_cohort.png — day 1 of the 24 / 48 / 72-h full models per L against SWPC day 1 (same issue dates).
* storm_case_72h_issue_fixed.png — May 2024: the 72-h forecasts issued on four consecutive dates, GeoIndex and SWPC,
  each issue's three days fixed.
* storm_case_24h_chained.png — May 2024: the day-1 forecasts of each daily issue chained (24-h model and SWPC day 1).

    python scripts/plot_ap3h_swpc_benchmark.py --out ~/Projects/GeoIndex/daily/benchmark_swpc_ap3h
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

C = {24: "#1f4e79", 48: "#4a90c2", 72: "#8fbce0", "swpc": "#b04a2f", "obs": "#000000"}


def style(a):
    a.grid(color="#e5e5e5", lw=0.8); a.set_axisbelow(True)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--L", type=int, default=4)
    p.add_argument("--case", nargs=2, default=["2024-05-08", "2024-05-11"], help="first and last issue date of the case")
    args = p.parse_args()
    m, fig_dir = args.out / "metrics", args.out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    st = pd.read_csv(m / "metrics_3hourly_by_step.csv")
    bd = pd.read_csv(m / "metrics_3hourly_by_target_day.csv")

    # lead step
    fig, ax = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for h in (24, 48, 72):
        g = st[(st.model_horizon_h == h) & (st.model_group == "full") & (st.input_length_days == args.L)]
        ge, sw = g[g.forecast == "geoi_ens"], g[g.forecast == "swpc"]
        ax[0].plot(ge.lead_step_3h, ge.mae, "o-", color=C[h], ms=4, lw=1.8, label=f"GeoIndex {h} h (L = {args.L})")
        ax[0].plot(sw.lead_step_3h, sw.mae, "--", color=C["swpc"], lw=1.2, alpha=0.5 + h / 200,
                   label="SWPC Kp → ap" if h == 72 else None)
        ax[1].plot(ge.lead_step_3h, ge.n_issue_dates, "-", color=C[h], lw=1.5)
    for x in (8.5, 16.5):
        ax[0].axvline(x, color="#999999", lw=0.8); ax[1].axvline(x, color="#999999", lw=0.8)
    ax[0].set_ylabel("3-hourly ap MAE"); ax[1].set_ylabel("issue dates"); ax[1].set_xlabel("lead step (3 h; 1–8 day 1, 9–16 day 2, 17–24 day 3)")
    ax[1].set_xticks(range(1, 25)); ax[0].legend(fontsize=8, frameon=False, ncol=2)
    for a in ax:
        style(a)
    fig.suptitle("SWPC 3-Day Geomagnetic Forecast (Kp → ap) vs GeoIndex, matched issue dates", fontsize=9)
    fig.tight_layout(); fig.savefig(fig_dir / "ap3h_lead_step_mae.png", dpi=150); plt.close(fig)

    # lead day
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for j, col in enumerate(("mae", "cc")):
        a = ax[j]
        for h, off in ((24, -0.15), (48, 0), (72, 0.15)):
            g = bd[(bd.model_horizon_h == h) & (bd.model_group == "full") & (bd.forecast == "geoi_ens")]
            for k in range(1, h // 24 + 1):
                gk = g[g.target_day == f"day{k}"]
                a.plot([k + off] * 2, [gk[col].min(), gk[col].max()], color=C[h], lw=4, alpha=0.6)
                a.plot(k + off, gk[gk.input_length_days == args.L][col].iloc[0], "o", color=C[h], ms=6,
                       label=f"GeoIndex {h} h" if k == 1 else None)
        sw = bd[(bd.model_horizon_h == 72) & (bd.model_group == "full") & (bd.forecast == "swpc") & (bd.input_length_days == 1)]
        a.plot([1, 2, 3], [sw[sw.target_day == f"day{k}"][col].iloc[0] for k in (1, 2, 3)], "D-.", color=C["swpc"],
               ms=6, label="SWPC Kp → ap")
        a.set_xticks([1, 2, 3], ["day 1", "day 2", "day 3"]); a.set_ylabel(f"3-hourly ap {col.upper()}"); style(a)
    ax[1].legend(fontsize=8, frameon=False)
    fig.suptitle(f"3-hourly ap by target day: bars = range over L = 1–7, dots = L = {args.L}; matched issue dates", fontsize=9)
    fig.tight_layout(); fig.savefig(fig_dir / "ap3h_lead_day_mae_cc.png", dpi=150); plt.close(fig)

    # day 1, common cohort
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for j, col in enumerate(("mae", "cc")):
        a = ax[j]
        for h in (24, 48, 72):
            g = bd[(bd.model_horizon_h == h) & (bd.model_group == "full") & (bd.forecast == "geoi_ens") & (bd.target_day == "day1")]
            a.plot(g.input_length_days, g[col], "o-", color=C[h], lw=1.8, ms=5, label=f"GeoIndex {h} h, day 1")
        sw = bd[(bd.model_horizon_h == 24) & (bd.model_group == "full") & (bd.forecast == "swpc") & (bd.target_day == "day1")]
        a.axhline(sw[col].iloc[0], color=C["swpc"], ls="-.", lw=1.5, label="SWPC day 1")
        a.set_xlabel("input length L (days)"); a.set_ylabel(f"3-hourly ap {col.upper()}"); a.set_xticks(range(1, 8)); style(a)
    ax[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Day 1 of the three horizons on the same matched issue dates", fontsize=9)
    fig.tight_layout(); fig.savefig(fig_dir / "24_48_72_day1_common_cohort.png", dpi=150); plt.close(fig)

    # storm case
    mt = pd.read_parquet(args.out / "processed" / "matched_ap3h_24_48_72.parquet",
                         filters=[("model_group", "==", "full"), ("input_length_days", "==", args.L)])
    mt["issue_date"] = pd.to_datetime(mt.issue_date)
    mt["t"] = mt.issue_date + pd.Timedelta(days=1) + pd.to_timedelta(3 * (mt.lead_step_3h - 1), unit="h")
    issues = pd.date_range(*args.case)
    fig, ax = plt.subplots(len(issues), 1, figsize=(9, 2.1 * len(issues)), sharex=True)
    for a, iss in zip(np.atleast_1d(ax), issues):
        g = mt[(mt.model_horizon_h == 72) & (mt.issue_date == iss)]
        if g.empty:
            a.text(0.5, 0.5, f"issue {iss:%Y-%m-%d}: not in the matched set", transform=a.transAxes, ha="center"); continue
        e = g.groupby("t").agg(obs=("observed_ap", "first"), geo=("geoi_ap_pred", "mean"), sw=("swpc_ap_converted", "first"))
        a.step(e.index, e.obs, where="post", color=C["obs"], lw=1.2, label="observed")
        a.step(e.index, e.geo, where="post", color=C[24], lw=1.5, label=f"GeoIndex 72 h (L = {args.L}, seed mean)")
        a.step(e.index, e.sw, where="post", color=C["swpc"], lw=1.5, ls="--", label="SWPC Kp → ap")
        a.set_ylabel("ap"); a.set_title(f"issued {iss:%Y-%m-%d} (GeoIndex origin 00 UT next day; SWPC {g.swpc_issue_utc.iloc[0]})", fontsize=8)
        style(a)
    np.atleast_1d(ax)[0].legend(fontsize=7, frameon=False, ncol=3)
    fig.tight_layout(); fig.savefig(fig_dir / "storm_case_72h_issue_fixed.png", dpi=150); plt.close(fig)

    g = mt[(mt.model_horizon_h == 24) & mt.issue_date.between(issues[0] - pd.Timedelta(days=3), issues[-1] + pd.Timedelta(days=3))]
    e = g.groupby("t").agg(obs=("observed_ap", "first"), geo=("geoi_ap_pred", "mean"), sw=("swpc_ap_converted", "first"))
    fig, a = plt.subplots(figsize=(9, 3.2))
    a.step(e.index, e.obs, where="post", color=C["obs"], lw=1.2, label="observed")
    a.step(e.index, e.geo, where="post", color=C[24], lw=1.5, label=f"GeoIndex 24 h (L = {args.L}), each day's issue")
    a.step(e.index, e.sw, where="post", color=C["swpc"], lw=1.5, ls="--", label="SWPC day 1, each day's issue")
    a.set_ylabel("ap"); a.legend(fontsize=8, frameon=False); style(a)
    a.set_title("Day-1 forecasts of consecutive daily issues, chained (not one issue)", fontsize=9)
    fig.tight_layout(); fig.savefig(fig_dir / "storm_case_24h_chained.png", dpi=150); plt.close(fig)
    print("figures in", fig_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
