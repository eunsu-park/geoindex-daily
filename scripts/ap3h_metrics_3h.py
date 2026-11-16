"""3-hour-resolution metrics of the l6 forecasts, one CSV per horizon (24, 48, 72 h).

Reads every <tag>_preds_<split>_<model>_<subset>_L<L>_s<seed>.npz under the given folders (three seeds per group),
and for each (model, subset, L) and each target day (1 … H/8) plus all target bins together writes:
n_days, 3-hour MAE / RMSE / CC, CRPS and the 68 % / 95 % coverage of the seed mixture, MAE and count of the bins with
ap ≥ 48, MAE of each of the day's eight bins (00–03 … 21–24 UT), and on the same bins the 3-hour MAE of persistence
(the issue day's eight ap, repeated), 27-day recurrence and the training-period mean. Point forecast = seed mean of
expm1(mu). The CME-only models are scored on their own (CME) test days. Rows with one target day also carry that
day's daily-Ap MAE and CC (daily Ap = mean of its eight bins). With --eval-days cme only the models trained on all days
are scored, on the test days with an Earth-directed DONKI CME in the 72 h before issue (the CME-only models' test days).

    python scripts/ap3h_metrics_3h.py ~/Projects/GeoIndex/daily/ap3h-l6/gpu2 ~/Projects/GeoIndex/daily/ap3h-l6/gpu3 \
        ~/Projects/GeoIndex/daily/ap3h ~/Projects/GeoIndex/daily/ap3h-l6h/gpu2 ~/Projects/GeoIndex/daily/ap3h-l6h/gpu3 \
        ~/Projects/GeoIndex/daily/ap3h-h --out ~/Projects/GeoIndex/daily/ap3h-metrics-3h [--eval-days cme]
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ap3h_score import crps_and_cover  # noqa: E402
from geoindex_daily.ap3h import BINS, cme_days  # noqa: E402
from geoindex_daily.daily_index import default_data_dir  # noqa: E402

STORM_AP = 48.0
NAME = re.compile(r"(\w+?)_preds_([A-Z])_(full|ts)_(all|cme)_L(\d+)_s(\d)\.npz")


def metrics(y, pt, mus, sigmas, refs, samples):
    """Dict of 3-hour metrics over the bins given (arrays (N, B))."""
    crps, c68, c95 = crps_and_cover(y, mus, sigmas, samples)
    storm = y >= STORM_AP
    out = {"mae_3h": np.abs(pt - y).mean(), "rmse_3h": np.sqrt(((pt - y) ** 2).mean()),
           "cc_3h": np.corrcoef(pt.ravel(), y.ravel())[0, 1], "crps_3h": crps, "cover68_3h": c68, "cover95_3h": c95,
           "mae_3h_storm_bins": np.abs(pt - y)[storm].mean() if storm.any() else np.nan, "n_storm_bins": int(storm.sum())}
    out.update({f"{k}_mae_3h": np.abs(r - y).mean() for k, r in refs.items()})
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dirs", nargs="+", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--config", default=str(Path(__file__).resolve().parent.parent / "configs" / "ap_lead1.yaml"))
    p.add_argument("--samples", type=int, default=100)
    p.add_argument("--eval-days", choices=["all", "cme"], default="all",
                   help="cme: score the all-days models on the CME test days only")
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    data = default_data_dir()
    f3 = pd.read_parquet(data / "features_3h.parquet")["ap"]
    tr = cfg["splits"]["A"]["train"]
    clim = float(f3.loc[tr[0]: tr[1]].mean())

    groups = {}
    for d in args.dirs:
        for f in d.glob("*_preds_*.npz"):
            m = NAME.fullmatch(f.name)
            if m:
                groups.setdefault((m[1], m[2], m[3], m[4], int(m[5])), {})[int(m[6])] = f
    rows = []
    for (tag, split, model, subset, L), seeds in sorted(groups.items()):
        if len(seeds) != 3:
            print(f"skip {tag} {model} {subset} L={L}: {len(seeds)} seeds"); continue
        if args.eval_days == "cme" and subset != "all":
            continue
        z = [np.load(seeds[s]) for s in sorted(seeds)]
        dates, y = pd.DatetimeIndex(z[0]["dates"]), np.round(z[0]["y"], 6)
        mus, sigmas = [x["mu"] for x in z], [x["sigma"] for x in z]
        if args.eval_days == "cme":
            keep = cme_days(dates, data / "events" / "donki_cme.parquet")
            dates, y, mus, sigmas = dates[keep], y[keep], [m[keep] for m in mus], [s_[keep] for s_ in sigmas]
        pt = np.mean([np.expm1(m) for m in mus], axis=0)
        H = y.shape[1]
        t0 = f3.index.get_indexer(dates + pd.Timedelta(days=1))
        pers = np.tile(np.stack([f3.to_numpy()[i - BINS: i] for i in t0]), (1, H // BINS))
        rec = np.stack([f3.to_numpy()[i - 27 * BINS: i - 27 * BINS + H] for i in t0])
        refs = {"persistence": pers, "recurrence27": rec, "climatology": np.full_like(y, clim)}
        base = {"horizon_h": H * 3, "tag": tag, "split": split, "model": model, "subset": subset,
                "eval_days": args.eval_days if subset == "all" else "cme", "L": L, "n_days": len(dates)}
        for k in range(0, H // BINS + 1):                                     # 0 = all target bins
            sl = slice(None) if k == 0 else slice((k - 1) * BINS, k * BINS)
            row = {**base, "target_day": "all" if k == 0 else k,
                   **metrics(y[:, sl], pt[:, sl], [m[:, sl] for m in mus], [s[:, sl] for s in sigmas],
                             {n: r[:, sl] for n, r in refs.items()}, args.samples)}
            if k > 0:
                yo, po = y[:, sl].mean(1), pt[:, sl].mean(1)
                row.update({"mae_day": np.abs(po - yo).mean(), "cc_day": np.corrcoef(po, yo)[0, 1]})
                row.update({f"mae_{3 * b:02d}_{3 * b + 3:02d}ut": np.abs(pt[:, sl][:, b] - y[:, sl][:, b]).mean()
                            for b in range(BINS)})
            rows.append(row)
    df = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    for H, g in df.groupby("horizon_h"):
        g = g.sort_values(["model", "subset", "L", "target_day"], key=lambda s: s.astype(str))
        f = args.out / f"ap3h_metrics_3h_{H}h{'_on_cme_days' if args.eval_days == 'cme' else ''}.csv"
        g.round(4).to_csv(f, index=False)
        print(f"wrote {f}: {len(g)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
