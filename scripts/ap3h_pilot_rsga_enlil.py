"""Pilot for the W46 proposal: can SWPC's RSGA forecast or DONKI's ENLIL arrival predictions add to the l6 model?

Both are known at the issue time (00 UT of t + 1): RSGA day k is issued at 22 UT of day t; an ENLIL run counts if its
model completion time is before 00 UT of t + 1. On the split-A test days, for target days k = 1, 2, 3 of the 72-hour
full model (--L):

* blend: daily Ap = w · model + (1 − w) · RSGA, w chosen on three test years and applied to the fourth
  (leave-one-year-out; no validation-period forecasts of the model were saved);
* ENLIL flag: an Earth-directed ENLIL run completed before issue predicts the shock arrival within target day k;
  hit rate on storm days (Ap > 48), the observed Ap on flagged and unflagged days, and the model's error on both.

    python scripts/ap3h_pilot_rsga_enlil.py --full ~/Projects/GeoIndex/daily/ap3h-l6h/gpu3 --L 4 \
        --csv ~/Projects/GeoIndex/daily/ap3h-l6h/pilot_rsga_enlil.csv
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from geoindex_daily.daily_index import default_data_dir  # noqa: E402
from geoindex_daily.lead1 import block_bootstrap_diff  # noqa: E402

STORM_AP = 48.0


def enlil_runs(raw_dir: Path) -> pd.DataFrame:
    """Every ENLIL run with an Earth arrival in the DONKI CME files: completion time, arrival time, Kp estimate."""
    rows = []
    for f in sorted(raw_dir.glob("*.json")):
        for e in json.loads(f.read_text() or "[]"):
            for a in e.get("cmeAnalyses") or []:
                for x in a.get("enlilList") or []:
                    if x.get("estimatedShockArrivalTime") and x.get("modelCompletionTime") and x.get("isEarthGB"):
                        kp = [x.get(k) for k in ("kp_90", "kp_135", "kp_180") if x.get(k) is not None]
                        rows.append({"cme": e["activityID"],
                                     "done": pd.to_datetime(x["modelCompletionTime"], utc=True).tz_localize(None),
                                     "arrival": pd.to_datetime(x["estimatedShockArrivalTime"], utc=True).tz_localize(None),
                                     "kp_max": max(kp) if kp else np.nan})
    return pd.DataFrame(rows).drop_duplicates()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--full", nargs="+", required=True)
    p.add_argument("--tag", default="l6h3")
    p.add_argument("--L", type=int, default=4)
    p.add_argument("--csv", type=Path, required=True)
    args = p.parse_args()
    data = default_data_dir()
    fs = sorted(f for d in args.full for f in glob.glob(f"{d}/{args.tag}_preds_A_full_all_L{args.L}_s*.npz"))
    z = [np.load(f) for f in fs]
    assert len(z) == 3
    dates, y = pd.DatetimeIndex(z[0]["dates"]), np.round(z[0]["y"], 6)
    pt = np.mean([np.expm1(x["mu"]) for x in z], axis=0)
    rsga = pd.read_parquet(data / "swpc" / "rsga_ap.parquet")
    root = os.environ.get("SOLARIS_ARCHIVE_ROOT")
    if not root:
        raise SystemExit("SOLARIS_ARCHIVE_ROOT is not set (source ~/.solaris_env)")
    runs = enlil_runs(Path(root) / "space_weather" / "ccmc" / "donki_cme")
    issue = dates + pd.Timedelta(days=1)                                      # 00 UT of t + 1
    years = dates.year

    rows = []
    for k in (1, 2, 3):
        sl = slice((k - 1) * 8, k * 8)
        yo, pm = y[:, sl].mean(1), pt[:, sl].mean(1)
        r = rsga[(rsga.kind == "predicted") & (rsga.lead == k)].drop_duplicates("target_date", keep="last")
        rg = r.set_index(pd.DatetimeIndex(r.target_date).as_unit("ns"))["ap"].reindex(dates + pd.Timedelta(days=k)).to_numpy()
        ok = ~np.isnan(rg)
        # blend, leave one test year out
        ws = np.linspace(0, 1, 21)
        pb = np.full(len(yo), np.nan)
        wy = {}
        for yr in np.unique(years):
            fit = ok & (years != yr)
            w = ws[np.argmin([np.abs(w * pm[fit] + (1 - w) * rg[fit] - yo[fit]).mean() for w in ws])]
            app = ok & (years == yr)
            pb[app] = w * pm[app] + (1 - w) * rg[app]
            wy[int(yr)] = float(w)
        b_m = block_bootstrap_diff(yo[ok], pb[ok], pm[ok], dates[ok], 2000, 0)
        b_r = block_bootstrap_diff(yo[ok], pb[ok], rg[ok], dates[ok], 2000, 0)
        # ENLIL: a run finished before issue predicts arrival within target day k
        lo, hi = dates + pd.Timedelta(days=k), dates + pd.Timedelta(days=k + 1)
        flag, kp = np.zeros(len(dates), bool), np.full(len(dates), np.nan)
        for i in range(len(dates)):
            m = (runs.done < issue[i]) & (runs.arrival >= lo[i]) & (runs.arrival < hi[i])
            if m.any():
                flag[i], kp[i] = True, runs.kp_max[m].max()
        storm = yo > STORM_AP
        e = np.abs(pm - yo)
        rows.append({
            "day": k, "n": int(ok.sum()), "model_mae": np.abs(pm[ok] - yo[ok]).mean(), "model_cc": np.corrcoef(pm[ok], yo[ok])[0, 1],
            "rsga_mae": np.abs(rg[ok] - yo[ok]).mean(), "rsga_cc": np.corrcoef(rg[ok], yo[ok])[0, 1],
            "blend_mae": np.abs(pb[ok] - yo[ok]).mean(), "blend_cc": np.corrcoef(pb[ok], yo[ok])[0, 1],
            "blend_w_by_year": json.dumps(wy),
            "blend_minus_model_dmae": b_m["d_mae"], "blend_minus_model_lo": b_m["d_mae_lo"], "blend_minus_model_hi": b_m["d_mae_hi"],
            "blend_minus_model_dcc": b_m["d_corr"], "blend_minus_model_dcc_lo": b_m["d_corr_lo"], "blend_minus_model_dcc_hi": b_m["d_corr_hi"],
            "blend_minus_rsga_dmae": b_r["d_mae"], "blend_minus_rsga_lo": b_r["d_mae_lo"], "blend_minus_rsga_hi": b_r["d_mae_hi"],
            "enlil_flag_days": int(flag.sum()), "storm_days": int(storm.sum()), "storm_days_flagged": int((storm & flag).sum()),
            "ap_obs_flagged": yo[flag].mean(), "ap_obs_unflagged": yo[~flag].mean(),
            "ap_model_flagged": pm[flag].mean(), "model_mae_flagged": e[flag].mean(), "model_mae_unflagged": e[~flag].mean(),
            "model_bias_flagged": (pm - yo)[flag].mean()})
    df = pd.DataFrame(rows)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.csv, index=False)
    pd.set_option("display.width", 250)
    print(df.round(3).T.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
