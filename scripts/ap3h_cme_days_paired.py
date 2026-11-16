"""Paired differences on the CME test days for the l6 models (24 / 48 / 72 h): one CSV.

On the days the CME-only models are tested on (Earth-directed DONKI CME in the 72 h before issue), per horizon, L and
target day: CME-only full model − all-days full model, and all-days full model − all-days time-series model; daily-Ap
MAE and CC differences with the monthly block bootstrap (2,000 resamples).

    python scripts/ap3h_cme_days_paired.py --out ~/Projects/GeoIndex/daily/ap3h-metrics-3h/ap3h_cme_days_paired.csv
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from geoindex_daily.ap3h import BINS, cme_days  # noqa: E402
from geoindex_daily.daily_index import default_data_dir  # noqa: E402
from geoindex_daily.lead1 import block_bootstrap_diff  # noqa: E402

# horizon (h) → (folders of the full / CME-only runs, folder of the time-series runs, tag all days, tag CME-only)
RUNS = {24: (["ap3h-l6/gpu2", "ap3h-l6/gpu3"], ["ap3h"], "l6", "l6cme"),
        48: (["ap3h-l6h/gpu2", "ap3h-l6h/gpu3"], ["ap3h-h"], "l6h2", "l6cmeh2"),
        72: (["ap3h-l6h/gpu2", "ap3h-l6h/gpu3"], ["ap3h-h"], "l6h3", "l6cmeh3")}


def load(dirs, tag, model, subset, L):
    fs = sorted(f for d in dirs for f in glob.glob(f"{d}/{tag}_preds_A_{model}_{subset}_L{L}_s*.npz"))
    z = [np.load(f) for f in fs]
    assert len(z) == 3, (tag, model, subset, L, len(z))
    return pd.DatetimeIndex(z[0]["dates"]), np.round(z[0]["y"], 6), np.mean([np.expm1(x["mu"]) for x in z], axis=0)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    data = default_data_dir()
    rows = []
    for h, (srv, loc, tf, tc) in RUNS.items():
        srv, loc = [data / d for d in srv], [data / d for d in loc]
        for L in range(1, 8):
            d, y, pf = load(srv, tf, "full", "all", L)
            _, _, pt = load(loc, tf, "ts", "all", L)
            dc, _, pc = load(srv, tc, "full", "cme", L)
            keep = cme_days(d, data / "events" / "donki_cme.parquet")
            assert (d[keep] == dc).all(), "CME days differ from the CME-only test days"
            for k in range(1, h // 24 + 1):
                sl = slice((k - 1) * BINS, k * BINS)
                yo = y[keep][:, sl].mean(1)
                full, ts, cme = pf[keep][:, sl].mean(1), pt[keep][:, sl].mean(1), pc[:, sl].mean(1)
                row = {"horizon_h": h, "L": L, "target_day": k, "n_days": len(dc)}
                for name, a, b in (("cmeonly_minus_full", cme, full), ("full_minus_ts", full, ts)):
                    r = block_bootstrap_diff(yo, a, b, dc, 2000, 0)
                    row.update({f"{name}_dmae": r["d_mae"], f"{name}_lo": r["d_mae_lo"], f"{name}_hi": r["d_mae_hi"],
                                f"{name}_dcc": r["d_corr"], f"{name}_dcc_lo": r["d_corr_lo"], f"{name}_dcc_hi": r["d_corr_hi"]})
                rows.append(row)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).round(4).to_csv(args.out, index=False)
    print("wrote", args.out, len(rows), "rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
