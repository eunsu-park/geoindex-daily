"""Score SWPC's operational daily-Ap forecasts against observed daily Ap, by lead.

Sources (see geoindex_daily/swpc.py): the weekly PRF 27-day outlook (archived
1997–present, `fetch_swpc_prf.py`) and, when present, the 45-day product parquet.
Climatology and 27-day recurrence are scored on exactly the same (issue, target)
pairs so the comparison is like for like.

    python scripts/eval_swpc.py --config configs/ap.yaml
    python scripts/eval_swpc.py --config configs/ap.yaml --start 2022-01-01 --end 2025-12-31
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geoindex_daily.baselines import recurrence_lag  # noqa: E402
from geoindex_daily.daily_index import default_data_dir, load_daily  # noqa: E402
from geoindex_daily.metrics import corr, mae  # noqa: E402

SHOW_LEADS = [1, 3, 7, 14, 21, 26]


def attach_references(fc: pd.DataFrame, s: pd.Series, rotation: int, train_end: str) -> pd.DataFrame:
    """Observed value, climatology and recurrence for every (issue_date, target_date) row."""
    fc = fc.copy()
    fc["obs"] = s.reindex(fc.target_date).to_numpy()
    fc["clim"] = s[:train_end].mean()
    lag = fc.lead.map(lambda h: recurrence_lag(max(h, 1), rotation))
    analogue = fc.issue_date - pd.to_timedelta(lag, unit="D")
    fc["rec27"] = s.reindex(analogue).to_numpy()
    return fc


def by_lead(fc: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for h, g in fc.groupby("lead"):
        r = {"lead": h, "n": int(g.obs.notna().sum())}
        for c in cols:
            r[f"{c}_mae"] = mae(g.obs, g[c])
            r[f"{c}_cc"] = corr(g.obs, g[c])
        rows.append(r)
    return pd.DataFrame(rows).set_index("lead")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/ap.yaml")
    p.add_argument("--prf", default=None, help="prf_outlook.parquet (default data dir/swpc/)")
    p.add_argument("--start", default=None, help="issue-date lower bound")
    p.add_argument("--end", default=None, help="issue-date upper bound")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    s = load_daily()[cfg["index"]]
    prf = pd.read_parquet(args.prf or default_data_dir() / "swpc" / "prf_outlook.parquet")
    if args.start:
        prf = prf[prf.issue_date >= args.start]
    if args.end:
        prf = prf[prf.issue_date <= args.end]
    prf = prf[prf.lead >= 1]  # lead 0 is the issue day itself
    fc = attach_references(prf, s, cfg["rotation_days"], cfg["splits"]["train"][1])
    fc = fc.dropna(subset=["obs"])
    print(f"PRF 27-day outlook: {fc.issue_date.nunique()} issues "
          f"{fc.issue_date.min().date()} .. {fc.issue_date.max().date()}, {len(fc)} forecast days")

    tab = by_lead(fc, [cfg["index"], "rec27", "clim"]).rename(columns=lambda c: c.replace(cfg["index"], "swpc"))
    out = default_data_dir() / f"swpc_prf_{cfg['index']}.csv"
    tab.to_csv(out)
    pd.set_option("display.width", 200)
    print(tab.loc[[h for h in SHOW_LEADS if h in tab.index]].round(3).to_string())
    print("\nmean over leads 1..26:")
    print(tab.drop(columns="n").mean().round(3).to_string())
    print(f"full table → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
