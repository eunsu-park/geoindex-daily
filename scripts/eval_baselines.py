"""Score the reference forecasts (climatology, persistence, 27-day recurrence) by lead time.

    python scripts/eval_baselines.py --config configs/ap.yaml
    python scripts/eval_baselines.py --config configs/ap.yaml --split test --leads 1 7 27 45 60

Leads default to 1..output_days; the table is printed for a few and the full one is
written to `<data dir>/baselines_<index>_<split>.csv`.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geoindex_daily.baselines import BASELINES, target  # noqa: E402
from geoindex_daily.daily_index import default_data_dir, load_daily  # noqa: E402
from geoindex_daily.metrics import score  # noqa: E402

SHOW_LEADS = [1, 3, 7, 14, 27, 45, 60]


def evaluate(s: pd.Series, leads: list[int], split: tuple[str, str] | None, train_end,
             rotation: int, threshold: float) -> pd.DataFrame:
    rows = []
    for h in leads:
        y = target(s, h)
        for name, fn in BASELINES.items():
            f = fn(s, h, train_end=train_end, rotation=rotation)
            if split:
                y_, f_ = y[split[0]: split[1]], f[split[0]: split[1]]
            else:
                y_, f_ = y, f
            rows.append({"lead": h, "baseline": name, **score(y_, f_, threshold)})
    return pd.DataFrame(rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/ap.yaml")
    p.add_argument("--data", default=None, help="daily parquet (default: data dir)")
    p.add_argument("--split", default="all", help="all | train | val | test (issue-date range from config)")
    p.add_argument("--leads", type=int, nargs="+", default=None)
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    d = load_daily(args.data)
    s = d[cfg["index"]]
    leads = args.leads or list(range(1, cfg["output_days"] + 1))
    split = None if args.split == "all" else tuple(cfg["splits"][args.split])
    train_end = cfg["splits"]["train"][1]

    res = evaluate(s, leads, split, train_end, cfg["rotation_days"], cfg["storm_threshold"])
    out = default_data_dir() / f"baselines_{cfg['index']}_{args.split}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)

    shown = res[res.lead.isin([h for h in SHOW_LEADS if h in leads])]
    cols = ["lead", "baseline", "n", "mae", "log_mae", "corr", "pod", "far", "csi"]
    pd.set_option("display.width", 200)
    print(f"index={cfg['index']} split={args.split} (climatology mean from train ≤ {train_end})")
    print(shown[cols].round(3).to_string(index=False))
    print(f"full table → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
