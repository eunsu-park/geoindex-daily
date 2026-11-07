"""Score phase-l6 forecasts: 3-hour ap, daily Ap (mean of the eight) against RSGA and baselines, and uncertainty.

Reads <tag>_preds_<split>_<model>_<subset>_L<L>_s<seed>.npz from scripts/ap3h_train.py. Per (model, subset, L)
with three seeds:

* point forecast = seed mean of expm1(mu) per 3-hour bin; daily Ap forecast = mean of its eight bins; the
  observed daily Ap is the unrounded mean of the eight observed ap;
* uncertainty = the equal-weight mixture of the seeds' Gaussians in log1p ap, sampled (`--samples` per seed):
  CRPS in ap units and the coverage of the central 68 % and 95 % intervals;
* references on the same days: persistence (the eight ap of the issue day; its daily Ap), 27-day recurrence
  (the bins 27 days before the target day), the training-period mean, and SWPC's RSGA day-1 Ap (daily only,
  issued 22 UT of the issue day);
* differences against persistence and RSGA (MAE, monthly block bootstrap), and scores on storm days
  (observed daily Ap > 48), on 3-hour bins with ap ≥ 48 (Kp ≥ 5) and on CME days (DONKI, 72 h before issue).

    python scripts/ap3h_score.py ~/Caches/GeoIndex/daily/ap3h --tag l6 --csv ~/Projects/GeoIndex/daily/ap3h/l6_scores_summary.csv
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geoindex_daily.ap3h import BINS, cme_days  # noqa: E402
from geoindex_daily.daily_index import default_data_dir  # noqa: E402
from geoindex_daily.lead1 import block_bootstrap_diff  # noqa: E402

STORM_AP = 48.0


def crps_and_cover(y: np.ndarray, mus: list, sigmas: list, k: int, seed: int = 0):
    """Sample CRPS (ap units) and central 68 / 95 % coverage of the seed mixture; y (N, H) ap."""
    rng = np.random.default_rng(seed)
    s = np.concatenate([np.expm1(m[..., None] + sg[..., None] * rng.standard_normal(m.shape + (k,)))
                        for m, sg in zip(mus, sigmas)], axis=-1)            # (N, H, S)
    s.sort(axis=-1)
    S = s.shape[-1]
    term1 = np.abs(s - y[..., None]).mean(-1)
    w = (2 * np.arange(1, S + 1) - S - 1) / S ** 2                            # E|X − X'| from sorted samples
    crps = term1 - (s * w).sum(-1)
    q = np.quantile(s, [0.025, 0.16, 0.84, 0.975], axis=-1)
    cov68 = ((y >= q[1]) & (y <= q[2])).mean()
    cov95 = ((y >= q[0]) & (y <= q[3])).mean()
    return float(crps.mean()), float(cov68), float(cov95)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dirs", nargs="+", type=Path)
    p.add_argument("--tag", default="l6")
    p.add_argument("--config", default=str(Path(__file__).resolve().parent.parent / "configs" / "ap_lead1.yaml"))
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--samples", type=int, default=100)
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    data = default_data_dir()
    f3 = pd.read_parquet(data / "features_3h.parquet")["ap"]
    rsga = pd.read_parquet(data / "swpc" / "rsga_ap.parquet")
    rsga = rsga[(rsga.kind == "predicted") & (rsga.lead == 1)].drop_duplicates("target_date", keep="last")
    rsga = rsga.set_index(pd.DatetimeIndex(rsga.target_date).as_unit("ns"))["ap"]
    tr = cfg["splits"]["A"]["train"]
    clim = float(f3.loc[tr[0]: tr[1]].mean())

    groups = {}
    for d in args.dirs:
        for f in d.glob(f"{args.tag}_preds_*.npz"):
            m = re.fullmatch(rf"{args.tag}_preds_([A-Z])_(\w+?)_(\w+?)_L(\d+)_s(\d)\.npz", f.name)
            if m:
                groups.setdefault((m[1], m[2], m[3], int(m[4])), {})[int(m[5])] = f

    rows = []
    for (split, model, subset, L), seeds in sorted(groups.items()):
        if len(seeds) < 3:
            print(f"skip {model} {subset} L={L}: {len(seeds)} seeds"); continue
        z = [np.load(seeds[s]) for s in sorted(seeds)]
        dates = pd.DatetimeIndex(z[0]["dates"])
        y = z[0]["y"]
        assert all((x["y"] == y).all() for x in z)
        pt = np.mean([np.expm1(x["mu"]) for x in z], axis=0)
        H = y.shape[1]
        # references on the same bins: issue day's eight ap, the bins 27 days before the target day
        t0 = f3.index.get_indexer(dates + pd.Timedelta(days=1))
        pers = np.stack([f3.to_numpy()[i - BINS: i - BINS + H] for i in t0])
        rec = np.stack([f3.to_numpy()[i - 27 * BINS: i - 27 * BINS + H] for i in t0])
        day = lambda a: a[:, :BINS].mean(1)                                    # noqa: E731 (first target day)
        yo, refs = day(y), {"model": pt, "persistence": pers, "recurrence27": rec, "climatology": np.full_like(y, clim)}
        rg = rsga.reindex(dates + pd.Timedelta(days=1)).to_numpy()
        crps, c68, c95 = crps_and_cover(y, [x["mu"] for x in z], [x["sigma"] for x in z], args.samples)
        storm_day, storm_bin = yo > STORM_AP, y >= STORM_AP
        cme = cme_days(dates, data / "events" / "donki_cme.parquet")
        row = {"split": split, "model": model, "subset": subset, "L": L, "n_days": len(dates),
               "mae_3h": np.abs(pt - y).mean(), "rmse_3h": np.sqrt(((pt - y) ** 2).mean()),
               "cc_3h": np.corrcoef(pt.ravel(), y.ravel())[0, 1],
               "mae_day": np.abs(day(pt) - yo).mean(), "cc_day": np.corrcoef(day(pt), yo)[0, 1],
               "crps_3h": crps, "cover68_3h": c68, "cover95_3h": c95,
               "mae_3h_storm_bins": np.abs(pt - y)[storm_bin].mean() if storm_bin.any() else np.nan,
               "n_storm_bins": int(storm_bin.sum()),
               "mae_day_storm_days": np.abs(day(pt) - yo)[storm_day].mean() if storm_day.any() else np.nan,
               "n_storm_days": int(storm_day.sum()),
               "mae_day_cme_days": np.abs(day(pt) - yo)[cme].mean() if cme.any() else np.nan, "n_cme_days": int(cme.sum())}
        row.update({f"mae_h{h + 1}": np.abs(pt[:, h] - y[:, h]).mean() for h in range(H)})
        for name, r in refs.items():
            if name != "model":
                row[f"{name}_mae_3h"] = np.abs(r - y).mean()
                row[f"{name}_mae_day"] = np.abs(day(r) - yo).mean()
        # differences in daily Ap against persistence and RSGA, monthly block bootstrap on the same days
        for name, ref in (("persistence", day(pers)), ("rsga", rg)):
            ok = ~np.isnan(ref)
            b = block_bootstrap_diff(yo[ok], day(pt)[ok], ref[ok], dates[ok], cfg["bootstrap"]["n"], cfg["bootstrap"]["seed"])
            row.update({f"vs_{name}_n": int(ok.sum()), f"vs_{name}_dmae_day": b["d_mae"],
                        f"vs_{name}_dmae_day_lo": b["d_mae_lo"], f"vs_{name}_dmae_day_hi": b["d_mae_hi"],
                        f"vs_{name}_dcc_day": b["d_corr"], f"vs_{name}_dcc_day_lo": b["d_corr_lo"],
                        f"vs_{name}_dcc_day_hi": b["d_corr_hi"]})
            row[f"{name}_ref_mae_day"] = np.abs(ref[ok] - yo[ok]).mean()
        rows.append(row)

    df = pd.DataFrame(rows)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.csv, index=False)
    cols = ["model", "subset", "L", "n_days", "mae_3h", "cc_3h", "mae_day", "cc_day", "crps_3h", "cover68_3h",
            "cover95_3h", "persistence_mae_day", "vs_rsga_n", "rsga_ref_mae_day", "vs_rsga_dmae_day",
            "vs_rsga_dmae_day_lo", "vs_rsga_dmae_day_hi"]
    pd.set_option("display.width", 250)
    print(df[cols].round(3).to_string(index=False))
    print("wrote", args.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
