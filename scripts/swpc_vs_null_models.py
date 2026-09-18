"""What SWPC's 27-day outlook is, measured against the null models on its own issue dates.

Decomposes the outlook into the two things it does — *select* which days will be active
(visible in the correlation, which is invariant to amplitude) and *damp* the amplitude
(visible in MAE and in the forecast/observed standard-deviation ratio) — and compares it
with persistence, recurrence-27, a mechanical persistence+recurrence blend, climatology and
our ridge on the PRF issue dates of the test split, leads 1–26.

    python scripts/swpc_vs_null_models.py --config configs/ap_1985.yaml \\
        --preds ~/Projects/GeoIndex/daily/ts_only_ap_1985_test_preds.npz

Writes `<data dir>/swpc_vs_null_models_<index>.csv` (per-lead CC and MAE for every forecast)
and prints the summary table used in the vault note `reference/swpc-vs-null-models.md`.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geoindex_daily.daily_index import default_data_dir, load_daily  # noqa: E402

MAX_LEAD = 26
SHOW = [1, 2, 3, 5, 7, 10, 14, 20, 26]


def cc(u, v):
    return float(np.corrcoef(u, v)[0, 1]) if np.std(v) > 0 else np.nan


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/ap_1985.yaml")
    p.add_argument("--preds", default=None, help="ts_only test-prediction npz (default: ts_only_<index>_test_preds.npz)")
    p.add_argument("--storm", type=float, default=30.0, help="daily Ap threshold for the event scores")
    p.add_argument("--band", nargs=2, type=int, default=[4, 14], help="lead band summarised as 'judgement leads'")
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    idx = cfg["index"]
    d = default_data_dir()
    z = np.load(Path(args.preds).expanduser() if args.preds else d / f"ts_only_{idx}_test_preds.npz")
    dates = pd.to_datetime(z["dates"])
    y = z["y"]

    prf = pd.read_parquet(d / "swpc" / "prf_outlook.parquet")
    prf = prf[(prf.lead >= 1) & (prf.lead <= MAX_LEAD)]
    sw = prf.pivot_table(index="issue_date", columns="lead", values=idx).reindex(dates)
    has = sw.notna().all(axis=1).to_numpy()
    Y = y[has][:, :MAX_LEAD]
    L = MAX_LEAD
    # persistence = the issue-day value carried forward, from the daily table the npz was built on
    s_daily = load_daily(d / cfg.get("data_file", "daily_index.parquet"))[idx]
    pers = np.repeat(s_daily.reindex(dates[has]).to_numpy(dtype=float)[:, None], L, axis=1)
    F = {"swpc": sw.to_numpy()[has], "persistence": pers, "recurrence27": z["recurrence27"][has][:, :L],
         "climatology": z["climatology"][has][:, :L], "ridge": z["ridge_raw"][has][:, :L]}
    blend = F["recurrence27"].copy(); blend[:, :2] = F["persistence"][:, :2]
    F["persistence1-2+recurrence27"] = blend

    lo, hi = args.band
    rows, long = [], []
    for name, M in F.items():
        r = np.array([cc(Y[:, h], M[:, h]) for h in range(L)])
        m = np.array([np.mean(np.abs(M[:, h] - Y[:, h])) for h in range(L)])
        for h in range(L):
            long.append({"forecast": name, "lead": h + 1, "cc": r[h], "mae": m[h]})
        o = Y[:, lo - 1:hi] >= args.storm; f = M[:, lo - 1:hi] >= args.storm
        hits = int((o & f).sum())
        rows.append({"forecast": name, **{f"cc_l{h}": r[h - 1] for h in SHOW},
                     f"cc_{lo}-{hi}": np.nanmean(r[lo - 1:hi]), "mae_1-26": m.mean(), f"mae_{lo}-{hi}": m[lo - 1:hi].mean(),
                     f"std_ratio_{lo}-{hi}": float(np.std(M[:, lo - 1:hi]) / np.std(Y[:, lo - 1:hi])),
                     f"pod_ge{args.storm:g}_{lo}-{hi}": hits / o.sum() if o.sum() else np.nan,
                     f"far_ge{args.storm:g}_{lo}-{hi}": (f & ~o).sum() / f.sum() if f.sum() else np.nan,
                     "n_forecast_events": int(f.sum())})
    pd.DataFrame(long).to_csv(d / f"swpc_vs_null_models_{idx}.csv", index=False)
    t = pd.DataFrame(rows).set_index("forecast")
    pd.set_option("display.width", 250)
    print(f"{int(has.sum())} PRF issue days {dates[has].min().date()} .. {dates[has].max().date()}, leads 1..{L}; "
          f"{int((Y[:, lo - 1:hi] >= args.storm).sum())} observed days with Ap >= {args.storm:g} in leads {lo}-{hi}")
    print(t.round(3).to_string())
    print(f"per-lead table → {d / f'swpc_vs_null_models_{idx}.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
