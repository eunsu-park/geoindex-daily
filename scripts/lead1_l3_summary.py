"""Summarise the l3 runs (W42: LASCO C2 running difference at 12 slots/day) against W41 and the no-image models.

For every split, image arm (D: Ap · B: Ap + X-ray · E: all eight daily features, each + images) and input
length L with all three seeds, the seed-mean forecast of l3_preds_<split>_<arm>_L<L>_s<seed>.npz is scored
on the l3 test days, together with — on the same days —
  * the W41 image model (l2, raw LASCO at 4 slots/day), seed mean;
  * the no-image l1 CNN with the same time-series inputs;
  * persistence.
Monthly block bootstrap of the MAE / CC differences:
  * l3 vs l2 (what the LASCO change does), l3 vs the no-image CNN (the image effect), l3 vs persistence;
  * the length rule: every L against the arm's best L (lowest MAE) and against its neighbours L ± 1.

    python scripts/lead1_l3_summary.py
Outputs in $GEOINDEX_DAILY_DATA/lead1/: l3_summary_scores.csv, l3_summary_bootstrap.csv.
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

NO_IMAGE = {"D": "A", "B": "B", "E": "E"}   # image arm → l1 arm with the same time-series inputs


def seed_mean(files):
    zs = [np.load(f, allow_pickle=True) for f in files]
    dates = pd.DatetimeIndex(zs[0]["dates"])
    assert all((pd.DatetimeIndex(z["dates"]) == dates).all() for z in zs), "seed days differ"
    return dates, zs[0]["y"], np.mean([z["pred"] for z in zs], axis=0), [np.abs(zs[0]["y"] - z["pred"]).mean() for z in zs]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/ap_lead1.yaml")
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    thr, n_boot, bseed = cfg["storm_threshold"], cfg["bootstrap"]["n"], cfg["bootstrap"]["seed"]
    out = default_data_dir() / "lead1"

    runs = defaultdict(lambda: defaultdict(list))
    for tag in ("l3", "l2"):
        pat = re.compile(rf"{tag}_preds_([AB])_([A-Z])_L(\d+)_s(\d+)\.npz")
        for f in sorted(out.glob(f"{tag}_preds_*.npz")):
            m = pat.fullmatch(f.name)
            if m:
                runs[tag][(m[1], m[2], int(m[3]))].append(f)
    done = {k: v for k, v in runs["l3"].items() if len(v) == len(cfg["seeds"])}

    scores, boots, ens = [], [], {}
    for sp in sorted({k[0] for k in done}):
        l1 = np.load(out / f"l1_preds_{sp}.npz", allow_pickle=True)
        l1_dates = pd.DatetimeIndex(l1["dates"])
        for (s, arm, L), files in sorted(done.items()):
            if s != sp:
                continue
            dates, y, pm, seed_mae = seed_mean(files)
            ens[(sp, arm, L)] = (dates, y, pm)
            d2, _, p2, _ = seed_mean(runs["l2"][(sp, arm, L)])
            pos2 = d2.get_indexer(dates)
            assert (pos2 >= 0).all(), "l3 test days not in the l2 forecasts"
            pos = l1_dates.get_indexer(dates)
            refs = {"W41 image (l2)": p2[pos2], f"no images ({NO_IMAGE[arm]}_cnn)": l1[f"{NO_IMAGE[arm]}_cnn_L{L}"][pos],
                    "persistence": l1["persistence"][pos]}
            scores.append({"split": sp, "arm": arm, "L": L, "model": "W42 image (l3)", "n_seeds": len(files),
                           "seed_mae_range": float(np.ptp(seed_mae)), **score_lead1(y, pm, dates, thr)})
            for name, ref in refs.items():
                scores.append({"split": sp, "arm": arm, "L": L, "model": name, **score_lead1(y, ref, dates, thr)})
                boots.append({"split": sp, "arm": arm, "L": L, "comparison": f"vs {name}",
                              **block_bootstrap_diff(y, pm, ref, dates, n_boot, bseed)})
    for sp, arm in sorted({(k[0], k[1]) for k in ens}):
        Ls = sorted(L for (s, a, L) in ens if (s, a) == (sp, arm))
        if len(Ls) < 2:
            continue
        mae = {L: np.abs(ens[(sp, arm, L)][1] - ens[(sp, arm, L)][2]).mean() for L in Ls}
        best = min(mae, key=mae.get)
        for L in Ls:
            refs = ([("vs best L", best)] if L != best else []) + [(f"vs L{L0}", L0) for L0 in (L - 1, L + 1) if L0 in Ls]
            for comp, L0 in refs:
                d, y, pm = ens[(sp, arm, L)]
                d0, _, p0 = ens[(sp, arm, L0)]
                assert (d == d0).all(), "lengths are not paired on the same days"
                boots.append({"split": sp, "arm": arm, "L": L, "comparison": comp, "ref_L": L0,
                              **block_bootstrap_diff(y, pm, p0, d, n_boot, bseed)})

    sc, bt = pd.DataFrame(scores), pd.DataFrame(boots)
    sc.to_csv(out / "l3_summary_scores.csv", index=False)
    bt.to_csv(out / "l3_summary_bootstrap.csv", index=False)

    pd.set_option("display.width", 220)
    for sp in sc.split.unique():
        sub = sc[sc.split == sp]
        print(f"\n== split {sp}: seed-mean MAE / CC at lead 1, {int(sub.n.iloc[0])} test days")
        print(sub.pivot_table(index=["arm", "model"], columns="L", values="mae").round(2).to_string())
        print(sub.pivot_table(index=["arm", "model"], columns="L", values="corr").round(3).to_string())
        b = bt[bt.split == sp].copy()
        b["dMAE [95%]"] = b.apply(lambda r: f"{r.d_mae:+.2f} [{r.d_mae_lo:+.2f}, {r.d_mae_hi:+.2f}]", axis=1)
        b["dCC [95%]"] = b.apply(lambda r: f"{r.d_corr:+.3f} [{r.d_corr_lo:+.3f}, {r.d_corr_hi:+.3f}]", axis=1)
        print(b[["arm", "L", "comparison"] + (["ref_L"] if "ref_L" in b else []) + ["dMAE [95%]", "dCC [95%]"]].to_string(index=False))
    print(f"\nwrote {out}/l3_summary_scores.csv, l3_summary_bootstrap.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
