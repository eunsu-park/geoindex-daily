"""Image contribution by physical event type (physical analysis, Task 4 / Figure 3 / Table 3; work order 2026-11-17).

Every split-A test target day (issue date + k, k = 1 for the 24-h model, 1–3 for the 72-h model) is labelled from
DONKI records (descriptive only, not model input):
* CME — a shock observed at Earth (DONKI IPS) linked to a CME during the target day or the day before;
* HSS — otherwise, a high-speed stream seen at L1 (DONKI HSS) during the target day or the two days before;
* quiet — neither, and observed daily Ap < 15; other — neither, daily Ap ≥ 15.
Per label, lead, L and model (full, time series; seed-ensemble forecast): daily-Ap MAE; 3-hourly MAE; the 3-hourly
error split into the daily level (|forecast daily mean − observed daily mean|) and the intra-day shape (MAE of the
deviations from each day's mean, and the CC of those deviations); peak-amplitude error (max forecast − max observed
ap of the day); on days with an observed bin ≥ 48 the share detected by the forecast median ≥ 48 or by
P(ap ≥ 48) ≥ 0.3 in any bin. Full − time series differences of daily and 3-hourly MAE with monthly block bootstrap.

    python scripts/ap3h_event_type.py --out ~/Projects/GeoIndex/daily/physical_analysis
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
from geoindex_daily.lead1 import block_bootstrap_diff  # noqa: E402

RUNS = {(24, "full"): (["ap3h-l6/gpu2", "ap3h-l6/gpu3"], "l6", "full"), (24, "ts"): (["ap3h"], "l6", "ts"),
        (72, "full"): (["ap3h-l6h/gpu2", "ap3h-l6h/gpu3"], "l6h3", "full"), (72, "ts"): (["ap3h-h"], "l6h3", "ts")}
LEADS = [(24, 1), (72, 1), (72, 2), (72, 3)]
LABELS = ["CME", "HSS", "quiet", "other"]


def load(data, h, model, L):
    dirs, tag, m = RUNS[(h, model)]
    fs = sorted(f for d in dirs for f in glob.glob(str(data / d / f"{tag}_preds_A_{m}_all_L{L}_s*.npz")))
    z = [np.load(f) for f in fs]
    assert len(z) == 3
    mu, sg = np.stack([x["mu"] for x in z]), np.stack([x["sigma"] for x in z])
    p48 = (1 - norm.cdf((np.log1p(48.0) - mu) / sg)).mean(0)
    return pd.DatetimeIndex(z[0]["dates"]).normalize(), np.round(z[0]["y"], 6), np.expm1(mu).mean(0), p48, np.expm1(mu)


def boot_mean(e: np.ndarray, dates: pd.DatetimeIndex, n: int = 2000, seed: int = 0):
    """95 % monthly-block bootstrap interval of the mean of per-day values e."""
    rng = np.random.default_rng(seed)
    months = pd.PeriodIndex(dates, freq="M")
    blocks = [np.flatnonzero(months == m) for m in months.unique()]
    v = [e[np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), len(blocks))])].mean() for _ in range(n)]
    return np.percentile(v, 2.5), np.percentile(v, 97.5)


def labels(days: pd.DatetimeIndex, ips: pd.DataFrame, hss: pd.DataFrame, ap_day: np.ndarray) -> np.ndarray:
    shock = ips[ips.linked_cmes != ""].shock_time.dt.normalize()
    hs = hss.hss_time.dt.normalize()
    out = []
    for d, a in zip(days, ap_day):
        if ((shock >= d - pd.Timedelta(days=1)) & (shock <= d)).any():
            out.append("CME")
        elif ((hs >= d - pd.Timedelta(days=2)) & (hs <= d)).any():
            out.append("HSS")
        else:
            out.append("quiet" if a < 15 else "other")
    return np.array(out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--fig-L", type=int, default=4)
    args = p.parse_args()
    data = default_data_dir()
    ips = pd.read_parquet(data / "events" / "donki_ips_earth.parquet")
    hss = pd.read_parquet(data / "events" / "donki_hss_earth.parquet")
    rows = []
    for (h, k) in LEADS:
        sl = slice((k - 1) * 8, k * 8)
        for L in range(1, 8):
            d, y, pf, qf, sf = load(data, h, "full", L)
            _, _, pt, qt, st_ = load(data, h, "ts", L)
            y, pf, pt, qf, qt, sf, st_ = y[:, sl], pf[:, sl], pt[:, sl], qf[:, sl], qt[:, sl], sf[:, :, sl], st_[:, :, sl]
            tgt = d + pd.Timedelta(days=k)
            lab = labels(tgt, ips, hss, y.mean(1))
            for name in LABELS:
                m = lab == name
                if m.sum() < 5:
                    continue
                yy = y[m]
                row = {"lead": f"{h} h day {k}", "model_horizon_h": h, "target_day": k, "L": L, "label": name,
                       "n_days": int(m.sum()), "n_days_storm": int((yy.max(1) >= 48).sum())}
                for mod, pp, qq in (("full", pf[m], qf[m]), ("ts", pt[m], qt[m])):
                    dm, do = pp.mean(1), yy.mean(1)
                    an_p, an_o = pp - dm[:, None], yy - do[:, None]
                    st = yy.max(1) >= 48
                    row.update({f"{mod}_mae_day": np.abs(dm - do).mean(), f"{mod}_mae_3h": np.abs(pp - yy).mean(),
                                f"{mod}_mae_intraday": np.abs(an_p - an_o).mean(),
                                f"{mod}_cc_intraday": np.corrcoef(an_p.ravel(), an_o.ravel())[0, 1],
                                f"{mod}_peak_amp_error": (pp.max(1) - yy.max(1)).mean(),
                                f"{mod}_detect_median": ((pp.max(1) >= 48) & st).sum() / max(st.sum(), 1),
                                f"{mod}_detect_p30": ((qq.max(1) >= 0.3) & st).sum() / max(st.sum(), 1),
                                f"{mod}_false_p30": ((qq.max(1) >= 0.3) & ~st).sum() / max((~st).sum(), 1)})
                for res, a, b, yo in (("day", pf[m].mean(1), pt[m].mean(1), yy.mean(1)),):
                    bb = block_bootstrap_diff(yo, a, b, tgt[m], 2000, 0)
                    row.update({"d_mae_day": bb["d_mae"], "d_mae_day_lo": bb["d_mae_lo"], "d_mae_day_hi": bb["d_mae_hi"]})
                # the same difference seed by seed (full seed i against time-series seed i)
                row.update({f"d_mae_day_seed{i}": np.abs(sf[i][m].mean(1) - yy.mean(1)).mean() - np.abs(st_[i][m].mean(1) - yy.mean(1)).mean()
                            for i in range(3)})
                e = np.abs(pf[m] - yy).mean(1) - np.abs(pt[m] - yy).mean(1)        # per-day 3-hourly MAE difference
                lo, hi = boot_mean(e, tgt[m])
                row.update({"d_mae_3h": e.mean(), "d_mae_3h_lo": lo, "d_mae_3h_hi": hi})
                rows.append(row)
    T = pd.DataFrame(rows)
    T.to_csv(args.out / "table3_event_type.csv", index=False)

    x = T[T.L == args.fig_L]
    fig, axs = plt.subplots(2, 3, figsize=(13, 7.5))
    ax = axs[0]
    for j, (col, lab) in enumerate((("mae_day", "daily-Ap MAE"), ("mae_3h", "3-hourly ap MAE"), ("mae_intraday", "intra-day shape MAE"))):
        a = ax[j]
        for i, name in enumerate(LABELS):
            g = x[x.label == name]
            a.plot(np.arange(len(g)) + i * 0.0, (g[f"full_{col}"] - g[f"ts_{col}"]).to_numpy(), "o-",
                   label=f"{name} (n {g.n_days.min()}–{g.n_days.max()})")
        a.axhline(0, color="#333333", lw=0.8); a.set_xticks(range(len(LEADS)), [f"{h} h d{k}" for h, k in LEADS])
        a.set_ylabel(f"full − time series, {lab}", fontsize=8)
        a.grid(color="#e5e5e5", lw=0.8); a.set_axisbelow(True)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    ax[0].legend(fontsize=7, frameon=False)
    # second row: peak-amplitude error and the storm-bin detection on days with an observed bin ≥ 48
    for j, (col, lab) in enumerate((("peak_amp_error", "full − time series, peak-amplitude error"),)):
        a = axs[1][0]
        for name in LABELS:
            g = x[x.label == name]
            a.plot(range(len(g)), (g[f"full_{col}"] - g[f"ts_{col}"]).to_numpy(), "o-", label=name)
        a.axhline(0, color="#333333", lw=0.8); a.set_ylabel(lab, fontsize=8)
    for a, mod, ttl in ((axs[1][1], "full", "full model"), (axs[1][2], "ts", "time series")):
        for name in ("CME", "HSS", "other"):
            g = x[x.label == name]
            a.plot(range(len(g)), g[f"{mod}_detect_p30"].to_numpy(), "o-", label=f"{name} (storm days {g.n_days_storm.min()}–{g.n_days_storm.max()})")
        a.set_ylim(0, 0.35); a.set_ylabel(f"share of storm days detected, P ≥ 0.3 ({ttl})", fontsize=8)
    axs[1][1].legend(fontsize=7, frameon=False)
    for a in axs[1]:
        a.set_xticks(range(len(LEADS)), [f"{h} h d{k}" for h, k in LEADS]); a.grid(color="#e5e5e5", lw=0.8); a.set_axisbelow(True)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.suptitle(f"Image contribution by event type (DONKI labels), L = {args.fig_L}; top: negative = the images help", fontsize=9)
    fig.tight_layout(); fig.savefig(args.out / "fig3_event_type.png", dpi=150); plt.close(fig)
    cols = ["lead", "label", "n_days", "n_days_storm", "full_mae_day", "ts_mae_day", "d_mae_day", "d_mae_day_lo", "d_mae_day_hi",
            "d_mae_3h", "d_mae_3h_lo", "d_mae_3h_hi", "full_mae_intraday", "ts_mae_intraday", "full_cc_intraday", "ts_cc_intraday",
            "full_detect_p30", "ts_detect_p30"]
    pd.set_option("display.width", 300)
    print(x[cols].round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
