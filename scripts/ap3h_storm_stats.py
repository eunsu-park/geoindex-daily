"""Onset / peak statistics over every eligible CME storm, at every L, and the relation to CME speed (physical analysis,
Task 3 extended and Task 4 Analysis A; work order 2026-11-17).

Storms: shocks at Earth that DONKI links to a CME (2022–2025) with an observed 3-hourly ap ≥ 48 in [shock − 6 h,
shock + 48 h); shocks within 48 h of a kept one belong to the same storm (the earliest is kept), so each storm counts
once. For every storm, L = 1–7, lead (24-h day 1; 72-h days 1–3: each target day from the issue k − 1 days before its
00 UT) and model (full, time series; seed ensemble): detection = P(ap ≥ 48) ≥ 0.3 in a bin of the storm interval;
onset error = first such bin from shock − 24 h minus the observed onset (first ap ≥ 48); peak-amplitude and peak-time
errors; MAE over the storm interval. Summaries per L and lead with full − time-series differences (95 % intervals from
resampling storms). CME speed (fastest linked CME) against observed peak, MAE, peak-amplitude error and the full −
time-series MAE difference: Spearman rank correlation over all CME-linked shock days (storm or not), merged likewise.

    python scripts/ap3h_storm_stats.py --out ~/Projects/GeoIndex/daily/physical_analysis
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
from scipy.stats import norm, spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from geoindex_daily.daily_index import default_data_dir  # noqa: E402

RUNS = {(24, "full"): (["ap3h-l6/gpu2", "ap3h-l6/gpu3"], "l6", "full"), (24, "ts"): (["ap3h"], "l6", "ts"),
        (72, "full"): (["ap3h-l6h/gpu2", "ap3h-l6h/gpu3"], "l6h3", "full"), (72, "ts"): (["ap3h-h"], "l6h3", "ts")}
LEADS = [(24, 1), (72, 1), (72, 2), (72, 3)]


def chained(data, h, model, L, k):
    """Series (index = bin start) of the day-k forecast: median-like point (seed-ensemble expm1(mu)) and P(ap ≥ 48)."""
    dirs, tag, m = RUNS[(h, model)]
    fs = sorted(f for d in dirs for f in glob.glob(str(data / d / f"{tag}_preds_A_{m}_all_L{L}_s*.npz")))
    z = [np.load(f) for f in fs]
    mu, sg = np.stack([x["mu"] for x in z])[:, :, (k - 1) * 8: k * 8], np.stack([x["sigma"] for x in z])[:, :, (k - 1) * 8: k * 8]
    origin = pd.DatetimeIndex(z[0]["dates"]).normalize() + pd.Timedelta(days=1)
    t = (origin + pd.Timedelta(days=k - 1)).to_numpy()[:, None] + (np.arange(8) * np.timedelta64(3, "h"))[None, :]
    idx = pd.DatetimeIndex(t.ravel())
    pt = np.expm1(mu).mean(0).ravel()
    p48 = (1 - norm.cdf((np.log1p(48.0) - mu) / sg)).mean(0).ravel()
    return pd.DataFrame({"pt": pt, "p48": p48}, index=idx).sort_index()


def merge(shocks: pd.DataFrame, hours=48) -> pd.DataFrame:
    keep, last = [], None
    for r in shocks.itertuples():
        if last is None or (r.shock_time - last) > pd.Timedelta(hours=hours):
            keep.append(r.Index); last = r.shock_time
    return shocks.loc[keep]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    data = default_data_dir()
    f3 = pd.read_parquet(data / "features_3h.parquet")["ap"]
    ips = pd.read_parquet(data / "events" / "donki_ips_earth.parquet")
    ips = ips[(ips.linked_cmes != "") & (ips.shock_time >= "2022-01-01") & (ips.shock_time < "2026-01-01")].sort_values("shock_time")
    cme = pd.read_parquet(data / "events" / "donki_cme.parquet").set_index("cme_id")
    ips["speed"] = [np.nanmax([cme.speed.get(i, np.nan) for i in s.split(";")] or [np.nan]) for s in ips.linked_cmes]
    sh = merge(ips.reset_index(drop=True))
    win = lambda S: f3.loc[S.floor("3h") - pd.Timedelta(hours=6): S.floor("3h") + pd.Timedelta(hours=45)]  # noqa: E731
    sh["max_ap"] = [win(S).max() for S in sh.shock_time]
    sh["storm"] = sh.max_ap >= 48

    rows = []
    for L in range(1, 8):
        for (h, k) in LEADS:
            fc = {m: chained(data, h, m, L, k) for m in ("full", "ts")}
            for s in sh.itertuples():
                S = s.shock_time
                w = win(S)
                if not all(w.index.isin(fc["full"].index)) or not all(w.index.isin(fc["ts"].index)):
                    continue
                on_obs = w.index[w >= 48][0] if s.storm else pd.NaT
                pre = fc["full"].loc[S.floor("3h") - pd.Timedelta(hours=24): w.index[-1]]
                row = {"shock": S, "storm": s.storm, "max_ap": s.max_ap, "speed": s.speed, "L": L, "lead": f"{h} h day {k}",
                       "observed_onset": on_obs, "observed_peak_time": w.idxmax()}
                for m in ("full", "ts"):
                    f = fc[m].reindex(w.index)
                    fe = fc[m].reindex(pre.index)
                    hit = fe.index[fe.p48 >= 0.3]
                    row.update({f"{m}_detected": bool((f.p48 >= 0.3).any()), f"{m}_mae": float(np.abs(f.pt - w).mean()),
                                f"{m}_peak_amp_err": f.pt.max() - w.max(),
                                f"{m}_peak_time_err_h": (f.pt.idxmax() - w.idxmax()) / pd.Timedelta(hours=1),
                                f"{m}_onset_err_h": (hit[0] - on_obs) / pd.Timedelta(hours=1) if (len(hit) and s.storm) else np.nan})
                rows.append(row)
    E = pd.DataFrame(rows)
    E.to_csv(args.out / "storm_stats_events.csv", index=False)

    rng = np.random.default_rng(0)
    summ = []
    for (L, lead, storm), g in E.groupby(["L", "lead", "storm"]):
        r = {"L": L, "lead": lead, "subset": "storm" if storm else "no storm", "n": len(g)}
        for m in ("full", "ts"):
            r.update({f"{m}_detect": g[f"{m}_detected"].mean(), f"{m}_mae": g[f"{m}_mae"].mean(),
                      f"{m}_peak_amp_err": g[f"{m}_peak_amp_err"].mean(), f"{m}_onset_err_h_median": g[f"{m}_onset_err_h"].median(),
                      f"{m}_early": int((g[f"{m}_onset_err_h"] < 0).sum())})
        for col in ("mae", "detected"):
            d = (g[f"full_{col}"].astype(float) - g[f"ts_{col}"].astype(float)).to_numpy()
            bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
            r.update({f"d_{col}": d.mean(), f"d_{col}_lo": np.percentile(bs, 2.5), f"d_{col}_hi": np.percentile(bs, 97.5)})
        summ.append(r)
    S_ = pd.DataFrame(summ)
    S_.to_csv(args.out / "storm_stats_summary.csv", index=False)

    # CME speed relation (L = 4)
    rel = []
    for lead in ("24 h day 1", "72 h day 2"):
        g = E[(E.L == 4) & (E.lead == lead)].dropna(subset=["speed"])
        for col, v in (("observed peak ap", g.max_ap), ("full MAE", g.full_mae), ("ts MAE", g.ts_mae),
                       ("full − ts MAE", g.full_mae - g.ts_mae), ("full peak-amplitude error", g.full_peak_amp_err)):
            rho, pv = spearmanr(g.speed, v)
            rel.append({"lead": lead, "quantity": col, "n": len(g), "spearman_rho": rho, "p_value": pv})
    R = pd.DataFrame(rel); R.to_csv(args.out / "cme_speed_relation.csv", index=False)
    g = E[(E.L == 4) & (E.lead == "24 h day 1")].dropna(subset=["speed"])
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for a, (v, lab) in zip(ax, ((g.max_ap, "observed peak ap (storm interval)"), (g.full_mae, "full-model MAE, storm interval (day 1)"),
                                (g.full_mae - g.ts_mae, "full − time-series MAE (day 1)"))):
        a.scatter(g.speed, v, c=np.where(g.storm, "#1f4e79", "#8c8c8c"), s=18)
        rho = spearmanr(g.speed, v)[0]
        a.set_xlabel("fastest linked CME speed (km/s)"); a.set_ylabel(lab, fontsize=8); a.set_title(f"Spearman ρ = {rho:.2f}", fontsize=9)
        a.grid(color="#e5e5e5", lw=0.8); a.set_axisbelow(True)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    ax[2].axhline(0, color="#333333", lw=0.8)
    fig.suptitle(f"CME-linked shocks 2022–2025, one per storm (n = {len(g)}; blue = followed by ap ≥ 48), L = 4", fontsize=9)
    fig.tight_layout(); fig.savefig(args.out / "fig5_cme_speed.png", dpi=150); plt.close(fig)
    pd.set_option("display.width", 300)
    print(f"merged CME shocks {len(sh)}, storms {int(sh.storm.sum())}")
    x = S_[(S_.subset == "storm")]
    print(x[["L", "lead", "n", "full_detect", "ts_detect", "d_detected", "d_detected_lo", "d_detected_hi", "full_mae", "ts_mae",
             "d_mae", "d_mae_lo", "d_mae_hi", "full_onset_err_h_median", "full_early", "ts_early", "full_peak_amp_err"]].round(2).to_string(index=False))
    print(R.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
