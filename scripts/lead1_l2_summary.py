"""Summarise the l2 image arms against their no-image l1 counterparts (seed-mean forecasts).

For every split and image arm (D: Ap + images · B: Ap + X-ray + images · E: all + images) and input length L, averages
the three seed forecasts in l2_preds_<split>_<arm>_L<L>_s<seed>.npz, scores them, and runs the
monthly block bootstrap of
  * image arm vs its no-image l1 CNN at the same L (D vs A_cnn, E vs E_cnn) — the image effect;
  * image arm at L vs the same arm at L = 1, and vs its neighbour L − 1 — the length curve and the
    plan's decision rule (a best L must beat both neighbours with the interval excluding zero);
  * image arm vs persistence.
The l1 forecasts are taken on the l2 test days (a subset of the l1 days).

    python scripts/lead1_l2_summary.py
Outputs in $GEOINDEX_DAILY_DATA/lead1/: l2_summary_scores.csv, l2_summary_bootstrap.csv.
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geoindex_daily.daily_index import default_data_dir  # noqa: E402
from geoindex_daily.lead1 import block_bootstrap_diff, score_lead1  # noqa: E402

NO_IMAGE = {"D": "A", "B": "B", "E": "E"}   # image arm → l1 arm with the same non-image inputs


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/ap_lead1.yaml")
    p.add_argument("--tag", default="l2")
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    thr, n_boot, bseed = cfg["storm_threshold"], cfg["bootstrap"]["n"], cfg["bootstrap"]["seed"]
    out = default_data_dir() / "lead1"

    runs = defaultdict(list)
    pat = re.compile(rf"{args.tag}_preds_([AB])_([A-Z])_L(\d+)_s(\d+)\.npz")
    for f in sorted(out.glob(f"{args.tag}_preds_*.npz")):
        m = pat.fullmatch(f.name)
        if m:
            runs[(m[1], m[2], int(m[3]))].append(f)

    scores, boots = [], []
    for sp in sorted({k[0] for k in runs}):
        l1 = np.load(out / f"l1_preds_{sp}.npz", allow_pickle=True)
        l1_dates = pd.DatetimeIndex(l1["dates"])
        ens = {}
        for (s, arm, L), files in sorted(runs.items()):
            if s != sp:
                continue
            zs = [np.load(f, allow_pickle=True) for f in files]
            dates = pd.DatetimeIndex(zs[0]["dates"])
            assert all((pd.DatetimeIndex(z["dates"]) == dates).all() for z in zs), f"{sp} {arm} L{L}: seed days differ"
            y, pm = zs[0]["y"], np.mean([z["pred"] for z in zs], axis=0)
            ens[(arm, L)] = (dates, y, pm)
            pos = l1_dates.get_indexer(dates)
            assert (pos >= 0).all(), "l2 test days not in l1_preds"
            base = l1[f"{NO_IMAGE[arm]}_cnn_L{L}"][pos]
            seed_mae = [np.abs(y - z["pred"]).mean() for z in zs]
            scores.append({"split": sp, "arm": arm, "L": L, "model": "image", "n_seeds": len(zs),
                           "seed_mae_std": float(np.std(seed_mae)), **score_lead1(y, pm, dates, thr)})
            scores.append({"split": sp, "arm": arm, "L": L, "model": f"l1 {NO_IMAGE[arm]}_cnn",
                           **score_lead1(y, base, dates, thr)})
            for comp, ref in [(f"vs {NO_IMAGE[arm]}_cnn (no images)", base),
                              ("vs persistence", l1["persistence"][pos])]:
                boots.append({"split": sp, "arm": arm, "L": L, "comparison": comp,
                              **block_bootstrap_diff(y, pm, ref, dates, n_boot, bseed)})
        for (arm, L), (dates, y, pm) in ens.items():
            refs = [("vs L1", 1)] + ([("vs L-1", L - 1)] if L - 1 > 1 else [])
            for comp, L0 in refs:
                if L != L0 and (arm, L0) in ens:
                    d0, _, p0 = ens[(arm, L0)]
                    assert (d0 == dates).all()
                    boots.append({"split": sp, "arm": arm, "L": L, "comparison": comp,
                                  **block_bootstrap_diff(y, pm, p0, dates, n_boot, bseed)})

    sc, bt = pd.DataFrame(scores), pd.DataFrame(boots)
    sc.to_csv(out / f"{args.tag}_summary_scores.csv", index=False)
    bt.to_csv(out / f"{args.tag}_summary_bootstrap.csv", index=False)

    pd.set_option("display.width", 200)
    for sp in sc.split.unique():
        sub = sc[sc.split == sp]
        print(f"\n== split {sp}: seed-mean MAE / CC at lead 1 (n days {int(sub.n.iloc[0]) if 'n' in sub else '?'})")
        print(sub.pivot_table(index=["arm", "model"], columns="L", values="mae").round(3).to_string())
        print(sub.pivot_table(index=["arm", "model"], columns="L", values="corr").round(3).to_string())
        b = bt[bt.split == sp].copy()
        b["dMAE [95%]"] = b.apply(lambda r: f"{r.d_mae:+.3f} [{r.d_mae_lo:+.3f}, {r.d_mae_hi:+.3f}]", axis=1)
        b["dCC [95%]"] = b.apply(lambda r: f"{r.d_corr:+.3f} [{r.d_corr_lo:+.3f}, {r.d_corr_hi:+.3f}]", axis=1)
        print(b[["arm", "L", "comparison", "dMAE [95%]", "dCC [95%]"]].to_string(index=False))
    print(f"\nwrote {out}/{args.tag}_summary_scores.csv, {args.tag}_summary_bootstrap.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
