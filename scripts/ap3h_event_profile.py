"""Physical event profile of one storm for the l6 3-hourly ap models (work order 2026-11-17, physical analysis, Task 2–3).

For a time window: OMNI hourly solar-wind speed, density and IMF Bz (GSM) and Dst; the Earth-directed DONKI CMEs that
started up to --cme-lookback days before the window, the ENLIL arrival predictions of their runs, the DONKI shocks
observed at Earth; the observed 3-hourly ap; and the forecasts at input length L (seed ensemble), arranged by lead:
day 1 from the 24-h models and days 1–3 from the 72-h models, each target day taken from the issue of k days before
(chained), for the full and the time-series model, with the full 24-h model's 68 / 95 % bands (mixture of the three
seeds). OMNI and the shock times are used only to describe the event, never as model inputs.

Event metrics per lead and model (written to --out/<event>_metrics.csv), in the storm interval [shock − 6 h,
shock + 48 h) of the event's main shock (--shock, DONKI IPS): observed onset = first bin with ap ≥ 48; forecast onset =
first bin from shock − 24 h on whose forecast median reaches 48 (missed if none) and first bin with P(ap ≥ 48) ≥ 0.1
and ≥ 0.3; onset errors in hours (negative = early); peak time and amplitude errors in the storm interval; MAE over
the window and over the storm interval.

    python scripts/ap3h_event_profile.py --event 2024-05 --start 2024-05-08 --end 2024-05-14 \
        --out ~/Projects/GeoIndex/daily/physical_analysis
"""
import argparse
import glob
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ap3h_pilot_rsga_enlil import enlil_runs  # noqa: E402
from geoindex_daily.daily_index import default_data_dir  # noqa: E402
from geoindex_daily.db import connect  # noqa: E402

STORM_AP = 48.0
RUNS = {(24, "full"): (["ap3h-l6/gpu2", "ap3h-l6/gpu3"], "l6", "full"), (24, "ts"): (["ap3h"], "l6", "ts"),
        (72, "full"): (["ap3h-l6h/gpu2", "ap3h-l6h/gpu3"], "l6h3", "full"), (72, "ts"): (["ap3h-h"], "l6h3", "ts")}
C = {"obs": "#000000", "full": "#1f4e79", "ts": "#8c8c8c", "cme": "#6a3d9a", "enlil": "#b04a2f", "ips": "#2a7f62"}


def load(data, h, model, L):
    dirs, tag, m = RUNS[(h, model)]
    fs = sorted(f for d in dirs for f in glob.glob(str(data / d / f"{tag}_preds_A_{m}_all_L{L}_s*.npz")))
    z = [np.load(f) for f in fs]
    assert len(z) == 3
    return pd.DatetimeIndex(z[0]["dates"]).normalize(), [x["mu"] for x in z], [x["sigma"] for x in z]


def chained(dates, mus, sigmas, k, t0, t1):
    """Forecast of target day d (issue d − k) for the bins in [t0, t1): times, median, P(ap ≥ 48), mixture quantiles."""
    sl = slice((k - 1) * 8, k * 8)
    rows = []
    for i, d in enumerate(dates):
        day = d + pd.Timedelta(days=k)
        if not (t0 - pd.Timedelta(days=1) <= day < t1):
            continue
        for b in range(8):
            t = day + pd.Timedelta(hours=3 * b)
            mu = np.array([m[i, sl][b] for m in mus]); sg = np.array([s[i, sl][b] for s in sigmas])
            p48 = float(np.mean(1 - norm.cdf((np.log1p(STORM_AP) - mu) / sg)))
            smp = np.concatenate([np.expm1(mu[j] + sg[j] * np.random.default_rng(b).standard_normal(400)) for j in range(3)])
            q = np.quantile(smp, [0.025, 0.16, 0.84, 0.975])
            rows.append({"t": t, "issue": d, "med": float(np.mean(np.expm1(mu))), "p48": p48,
                         "q025": q[0], "q16": q[1], "q84": q[2], "q975": q[3]})
    return pd.DataFrame(rows).set_index("t").sort_index()


def first(ts, cond):
    hit = ts.index[cond]
    return hit[0] if len(hit) else pd.NaT


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--event", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True, help="exclusive")
    p.add_argument("--shock", required=True, help="main shock time (DONKI IPS), UT")
    p.add_argument("--L", type=int, default=4)
    p.add_argument("--cme-lookback", type=int, default=4)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    data = default_data_dir()
    t0, t1 = pd.Timestamp(args.start), pd.Timestamp(args.end)
    args.out.mkdir(parents=True, exist_ok=True)

    with connect("space_weather") as c:
        cur = c.cursor()
        cur.execute("""select datetime, case when plasma_flow_speed_km_s < 9000 then plasma_flow_speed_km_s end,
                       case when proton_density_n_cm3 < 999 then proton_density_n_cm3 end,
                       case when bz_gsm_nt < 999 then bz_gsm_nt end, case when dst_index_nt < 99999 then dst_index_nt end
                       from omni_low_resolution where datetime >= %s and datetime < %s order by 1""", (t0, t1))
        om = pd.DataFrame(cur.fetchall(), columns=["t", "v", "n", "bz", "dst"]).set_index("t").astype(float)
    ap = pd.read_parquet(data / "features_3h.parquet")["ap"].loc[t0: t1 - pd.Timedelta(hours=3)]
    cme = pd.read_parquet(data / "events" / "donki_cme.parquet")
    cme = cme[cme.earth_directed & cme.cme_start_time.between(t0 - pd.Timedelta(days=args.cme_lookback), t1)]
    ips = pd.read_parquet(data / "events" / "donki_ips_earth.parquet")
    ips = ips[ips.shock_time.between(t0, t1)]
    runs = enlil_runs(Path(os.environ["SOLARIS_ARCHIVE_ROOT"]) / "space_weather" / "ccmc" / "donki_cme")
    runs = runs[runs.cme.isin(cme.cme_id) & runs.arrival.between(t0 - pd.Timedelta(days=1), t1)]

    fc = {}
    for (h, model) in RUNS:
        d, mus, sgs = load(data, h, model, args.L)
        for k in ([1] if h == 24 else [1, 2, 3]):
            fc[(h, model, k)] = chained(d, mus, sgs, k, t0, t1).reindex(ap.index)

    # ---- figure ----------------------------------------------------------------------------------------------
    fig, ax = plt.subplots(5, 1, figsize=(10, 12), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1.6, 1.6, 1.4]})
    ax[0].plot(om.index, om.v, color="#333333", lw=1.2); ax[0].set_ylabel("V (km/s)")
    ax[1].plot(om.index, om.bz, color="#333333", lw=1.0); ax[1].axhline(0, color="#999999", lw=0.8)
    ax[1].fill_between(om.index, om.bz, 0, where=om.bz < 0, color="#4a90c2", alpha=0.3, lw=0); ax[1].set_ylabel("IMF Bz GSM (nT)")
    tt = ap.index + pd.Timedelta(hours=1.5)
    ax[2].step(ap.index, ap, where="post", color=C["obs"], lw=1.4, label="observed 3-h ap")
    ax[2].step(ap.index, fc[(24, "full", 1)].med, where="post", color=C["full"], lw=1.5, label="full, day 1 (24-h model)")
    ax[2].step(ap.index, fc[(24, "ts", 1)].med, where="post", color=C["ts"], lw=1.5, ls="--", label="time series, day 1")
    ax[2].set_ylabel("ap"); ax[2].legend(fontsize=7, frameon=False, ncol=3, loc="upper left")
    ax[3].step(ap.index, ap, where="post", color=C["obs"], lw=1.4, label="observed")
    for k, a_ in ((1, 1.0), (2, 0.65), (3, 0.35)):
        ax[3].step(ap.index, fc[(72, "full", k)].med, where="post", color=C["full"], alpha=a_, lw=1.5,
                   label=f"full, day {k} (72-h model, issued {k} day{'s' if k > 1 else ''} before)")
    ax[3].set_ylabel("ap"); ax[3].legend(fontsize=7, frameon=False, loc="upper left")
    f1 = fc[(24, "full", 1)]
    ax[4].fill_between(ap.index, f1.q025, f1.q975, step="post", color=C["full"], alpha=0.15, lw=0, label="95 %")
    ax[4].fill_between(ap.index, f1.q16, f1.q84, step="post", color=C["full"], alpha=0.3, lw=0, label="68 %")
    ax[4].step(ap.index, ap, where="post", color=C["obs"], lw=1.2, label="observed")
    a2 = ax[4].twinx(); a2.step(ap.index, f1.p48, where="post", color=C["enlil"], lw=1.2); a2.set_ylim(0, 1)
    a2.set_ylabel("P(ap ≥ 48), day 1", color=C["enlil"], fontsize=8)
    ax[4].set_ylabel("ap"); ax[4].legend(fontsize=7, frameon=False, loc="upper left")
    for a in ax:
        for d in pd.date_range(t0, t1, freq="D"):
            a.axvline(d, color="#dddddd", lw=0.6, zorder=0)
        for t in cme.cme_start_time:
            if t >= t0:
                a.axvline(t, color=C["cme"], lw=1, ls=":")
        for t in runs.arrival:
            a.axvline(t, color=C["enlil"], lw=0.8, ls="--", alpha=0.6)
        for t in ips.shock_time:
            a.axvline(t, color=C["ips"], lw=1.6)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    ax[0].set_title(f"{args.event}: green = observed shock (DONKI IPS), red dashed = ENLIL predicted arrivals, purple dotted = "
                    f"Earth-directed CME starts; grey = daily issue times (00 UT); L = {args.L}", fontsize=8)
    ax[4].set_xlim(t0, t1)
    fig.tight_layout(); fig.savefig(args.out / f"{args.event}_profile.png", dpi=140); plt.close(fig)

    # ---- event metrics --------------------------------------------------------------------------------------------
    sh = pd.Timestamp(args.shock)
    st = (ap.index >= sh - pd.Timedelta(hours=6)) & (ap.index < sh + pd.Timedelta(hours=48))
    ap_s = ap[st]
    on_obs = first(ap_s, ap_s >= STORM_AP)
    pk_obs, pk_val = ap_s.idxmax(), ap_s.max()
    rows = []
    for (h, model, k), f in fc.items():
        fe = f[ap.index >= sh - pd.Timedelta(hours=24)]
        fs = f[st]
        on_f = first(fe.med, fe.med >= STORM_AP)
        on_p1, on_p3 = first(fe.p48, fe.p48 >= 0.1), first(fe.p48, fe.p48 >= 0.3)
        hrs = lambda t: (t - on_obs) / pd.Timedelta(hours=1) if pd.notna(t) and pd.notna(on_obs) else np.nan  # noqa: E731
        rows.append({"event": args.event, "model_horizon_h": h, "model": model, "target_day": k, "L": args.L,
                     "observed_onset": on_obs, "forecast_onset_median": on_f, "onset_error_h_median": hrs(on_f),
                     "missed_median": pd.isna(on_f), "onset_p48_ge_0.1": on_p1, "onset_error_h_p10": hrs(on_p1),
                     "onset_p48_ge_0.3": on_p3, "onset_error_h_p30": hrs(on_p3),
                     "shock": sh, "observed_peak_time": pk_obs, "observed_peak_ap": pk_val,
                     "forecast_peak_time": fs.med.idxmax(), "peak_time_error_h": (fs.med.idxmax() - pk_obs) / pd.Timedelta(hours=1),
                     "forecast_peak_ap": fs.med.max(), "peak_amplitude_error": fs.med.max() - pk_val,
                     "mae_window": float(np.abs(f.med - ap).mean()), "mae_storm": float(np.abs(fs.med - ap_s).mean()),
                     "max_p48_storm": fs.p48.max()})
    pd.DataFrame(rows).to_csv(args.out / f"{args.event}_metrics.csv", index=False)
    ev = pd.DataFrame({"kind": ["cme"] * len(cme) + ["enlil_arrival"] * len(runs) + ["shock"] * len(ips),
                       "time": list(cme.cme_start_time) + list(runs.arrival) + list(ips.shock_time),
                       "detail": [f"{r.cme_id}, {r.speed} km/s, lon {r.longitude}, half {r.half_angle}" for r in cme.itertuples()]
                       + [f"{r.cme}, completed {r.done}, Kp {r.kp_max}" for r in runs.itertuples()]
                       + [f"{r.ips_id}; linked {r.linked_cmes}" for r in ips.itertuples()]}).sort_values("time")
    ev.to_csv(args.out / f"{args.event}_timeline.csv", index=False)
    print(pd.DataFrame(rows)[["model_horizon_h", "model", "target_day", "onset_error_h_median", "onset_error_h_p10",
                              "onset_error_h_p30", "peak_time_error_h", "forecast_peak_ap", "peak_amplitude_error",
                              "mae_storm", "max_p48_storm"]].round(2).to_string(index=False))
    print(f"observed onset {on_obs}, peak {pk_val} at {pk_obs}; CMEs {len(cme)}, ENLIL arrivals {len(runs)}, shocks {len(ips)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
