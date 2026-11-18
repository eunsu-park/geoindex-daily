"""Event tables and Figure 2 of the physical analysis (work order 2026-11-17), from the ap3h_event_profile.py outputs.

Reads <dir>/event_list_input.txt (event id, window start, end, main shock) and <dir>/<event>_metrics.csv, and writes:

* table1_cme_events.csv — per event: main shock (DONKI IPS, observed arrival), the Earth-directed CMEs DONKI links to
  it (earliest start, maximum speed), the ENLIL arrival prediction for those CMEs that was completed before the shock
  and is nearest to it (with its completion time), maximum observed 3-hourly ap in [shock − 6 h, shock + 48 h), class
  (strong: max ap ≥ 132, Kp ≥ 7; moderate: 48–131; weak: < 48), sources;
* table2_event_performance.csv — the per-event metrics of every lead (24-h day 1, 72-h days 1–3) and model;
* table2_summary.csv — per lead and model over the storm events: detection by the forecast median and by
  P(ap ≥ 48) ≥ 0.3, median onset / peak-time error, mean peak-amplitude error; false alarms on the weak events;
* fig2_onset_peak.png — onset error (P ≥ 0.3) and peak-amplitude error per lead, full vs time series, one dot per event.

    python scripts/ap3h_event_summary.py --dir ~/Projects/GeoIndex/daily/physical_analysis
"""
import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ap3h_pilot_rsga_enlil import enlil_runs  # noqa: E402
from geoindex_daily.daily_index import default_data_dir  # noqa: E402

LEADS = [(24, 1), (72, 1), (72, 2), (72, 3)]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", type=Path, required=True)
    args = p.parse_args()
    data = default_data_dir()
    ev = pd.read_csv(args.dir / "event_list_input.txt", sep=r"\s+", header=None, names=["event", "start", "end", "shock"])
    ev["shock"] = pd.to_datetime(ev.shock)
    ips = pd.read_parquet(data / "events" / "donki_ips_earth.parquet")
    cme = pd.read_parquet(data / "events" / "donki_cme.parquet").set_index("cme_id")
    runs = enlil_runs(Path(os.environ["SOLARIS_ARCHIVE_ROOT"]) / "space_weather" / "ccmc" / "donki_cme")
    f3 = pd.read_parquet(data / "features_3h.parquet")["ap"]

    t1, mets = [], []
    for r in ev.itertuples():
        sh = ips.iloc[(ips.shock_time - r.shock).abs().argmin()]
        ids = [i for i in sh.linked_cmes.split(";") if i in cme.index]
        ed = [i for i in ids if cme.loc[i, "earth_directed"]]
        use = ed or ids
        rr = runs[runs.cme.isin(use) & (runs.done < sh.shock_time)]
        near = rr.iloc[(rr.arrival - sh.shock_time).abs().argmin()] if len(rr) else None
        w = f3.loc[sh.shock_time - pd.Timedelta(hours=6): sh.shock_time + pd.Timedelta(hours=45)]
        mx = w.max()
        t1.append({"event": r.event, "observed_shock_utc": sh.shock_time, "ips_id": sh.ips_id,
                   "linked_cmes": ";".join(use), "earth_directed_linked": len(ed),
                   "cme_onset_utc": min(cme.loc[i, "cme_start_time"] for i in use) if use else pd.NaT,
                   "cme_speed_max_km_s": max(cme.loc[i, "speed"] for i in use) if use else np.nan,
                   "enlil_predicted_arrival_utc": near.arrival if near is not None else pd.NaT,
                   "enlil_completed_utc": near.done if near is not None else pd.NaT,
                   "enlil_arrival_error_h": (near.arrival - sh.shock_time) / pd.Timedelta(hours=1) if near is not None else np.nan,
                   "enlil_kp_max": near.kp_max if near is not None else np.nan,
                   "max_observed_ap": mx, "max_observed_ap_utc": w.idxmax(),
                   "class": "strong" if mx >= 132 else "moderate" if mx >= 48 else "weak",
                   "sources": "DONKI CME / ENLIL / IPS (CCMC), GFZ 3-hour ap"})
        m = pd.read_csv(args.dir / f"{r.event}_metrics.csv")
        m["class"] = t1[-1]["class"]
        mets.append(m)
    T1 = pd.DataFrame(t1); T1.to_csv(args.dir / "table1_cme_events.csv", index=False)
    M = pd.concat(mets, ignore_index=True); M.to_csv(args.dir / "table2_event_performance.csv", index=False)

    rows = []
    for (h, k) in LEADS:
        for model in ("full", "ts"):
            x = M[(M.model_horizon_h == h) & (M.target_day == k) & (M.model == model)]
            s, wk = x[x["class"] != "weak"], x[x["class"] == "weak"]
            rows.append({"lead": f"{h} h day {k}", "model": model, "n_storm_events": len(s),
                         "detected_median": int((~s.missed_median).sum()), "detected_p30": int(s.onset_error_h_p30.notna().sum()),
                         "median_onset_error_h_p30": s.onset_error_h_p30.median(), "median_onset_error_h_p10": s.onset_error_h_p10.median(),
                         "median_peak_time_error_h": s.peak_time_error_h.median(),
                         "mean_peak_amplitude_error": s.peak_amplitude_error.mean(), "mean_mae_storm": s.mae_storm.mean(),
                         "n_weak_events": len(wk), "false_alarm_median": int((wk.forecast_peak_ap >= 48).sum()),
                         "false_alarm_p30": int((wk.max_p48_storm >= 0.3).sum())})
    S = pd.DataFrame(rows); S.to_csv(args.dir / "table2_summary.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    st = M[M["class"] != "weak"]
    for j, (col, lab) in enumerate((("onset_error_h_p30", "onset error, P(ap ≥ 48) ≥ 0.3 (h; − = early)"),
                                    ("peak_amplitude_error", "peak-amplitude error (forecast − observed ap)"))):
        a = ax[j]
        for i, (h, k) in enumerate(LEADS):
            for off, model, c in ((-0.12, "full", "#1f4e79"), (0.12, "ts", "#8c8c8c")):
                v = st[(st.model_horizon_h == h) & (st.target_day == k) & (st.model == model)][col]
                a.scatter(np.full(len(v), i + off) + np.random.default_rng(i).uniform(-0.04, 0.04, len(v)), v, s=22, color=c,
                          label=("full" if model == "full" else "time series") if i == 0 else None)
        a.axhline(0, color="#333333", lw=0.8)
        if col == "onset_error_h_p30":
            miss = lambda h, k, mo: int(st[(st.model_horizon_h == h) & (st.target_day == k) & (st.model == mo)][col].isna().sum())  # noqa: E731
            ticks = [f"{h} h, day {k}\nmissed: full {miss(h, k, 'full')}, ts {miss(h, k, 'ts')}" for h, k in LEADS]
        else:
            ticks = [f"{h} h, day {k}" for h, k in LEADS]
        a.set_xticks(range(len(LEADS)), ticks, fontsize=8); a.set_ylabel(lab, fontsize=8)
        a.grid(color="#e5e5e5", lw=0.8); a.set_axisbelow(True)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    ax[0].legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle(f"{len(st.event.unique())} storm events (strong and moderate), L = 4; onset from the main shock − 24 h", fontsize=9)
    fig.tight_layout(); fig.savefig(args.dir / "fig2_onset_peak.png", dpi=150); plt.close(fig)
    pd.set_option("display.width", 250)
    print(T1[["event", "observed_shock_utc", "cme_onset_utc", "cme_speed_max_km_s", "enlil_predicted_arrival_utc",
              "enlil_arrival_error_h", "max_observed_ap", "class"]].to_string(index=False))
    print(S.round(1).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
