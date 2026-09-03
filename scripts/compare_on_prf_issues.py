"""Like-for-like: our test-split forecasts vs the SWPC PRF outlook on the PRF's own issue dates.

Uses the test predictions saved by `ts_only.py` and `swpc/prf_outlook.parquet`; keeps only
(issue, target) pairs present in both, leads 1–26, and scores MAE / CC per model.

    python scripts/compare_on_prf_issues.py --config configs/ap.yaml
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geoindex_daily.daily_index import default_data_dir  # noqa: E402
from geoindex_daily.metrics import corr, mae  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/ap.yaml")
    p.add_argument("--max-lead", type=int, default=26)
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    idx = cfg["index"]
    d = default_data_dir()

    z = np.load(d / f"ts_only_{idx}_test_preds.npz")
    dates = pd.to_datetime(z["dates"])
    models = [k for k in z.files if k not in ("dates", "y")]
    long = []
    for k in ["y"] + models:
        arr = z[k]
        df = pd.DataFrame(arr[:, : args.max_lead], index=dates, columns=range(1, args.max_lead + 1))
        long.append(df.stack().rename(k))
    ours = pd.concat(long, axis=1)
    ours.index.names = ["issue_date", "lead"]
    ours = ours.reset_index()

    prf = pd.read_parquet(d / "swpc" / "prf_outlook.parquet")[["issue_date", "lead", idx]].rename(columns={idx: "swpc"})
    m = ours.merge(prf, on=["issue_date", "lead"], how="inner")
    m = m[(m.lead >= 1) & (m.lead <= args.max_lead)]
    print(f"common pairs: {len(m)} on {m.issue_date.nunique()} PRF issue dates "
          f"{m.issue_date.min().date()} .. {m.issue_date.max().date()}, leads 1..{args.max_lead}")
    rows = []
    for k in ["swpc"] + models:
        per = m.groupby("lead").apply(lambda g: pd.Series({"mae": mae(g.y, g[k]), "cc": corr(g.y, g[k])}))
        rows.append({"model": k, "mae": per.mae.mean(), "cc": per.cc.mean(),
                     "mae_l1": per.mae.get(1), "cc_l1": per.cc.get(1),
                     "mae_l7": per.mae.get(7), "cc_l7": per.cc.get(7),
                     "mae_l26": per.mae.get(args.max_lead), "cc_l26": per.cc.get(args.max_lead)})
    tab = pd.DataFrame(rows).set_index("model")
    pd.set_option("display.width", 200)
    print(tab.round(3).to_string())
    tab.to_csv(d / f"compare_prf_{idx}.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
