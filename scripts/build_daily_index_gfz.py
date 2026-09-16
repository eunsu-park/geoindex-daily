"""Build the daily index table (Ap, Kp, SN, F10.7) from GFZ's definitive/preliminary file.

`Kp_ap_Ap_SN_F107_since_1932.txt` (GFZ Potsdam, CC BY 4.0) is the official long record:
one row per UT day with Kp1..Kp8, ap1..ap8, the daily Ap (mean of the eight ap values,
rounded to an integer by GFZ), the sunspot number and F10.7. It reaches back to 1932 and is
updated daily (the last ~weeks are preliminary, D=0), so it is the source for training on
the whole record instead of OMNI's 2010–2025 slice.

The output has the same columns as `daily_index.parquet` (`ap kp f107 sn dst n_ap_hours`);
`dst` is NaN (not in the file), `kp` is the daily maximum Kp, `n_ap_hours` is 3 × the number
of valid ap values so the flag semantics match the OMNI build (24 = complete day).

    python scripts/build_daily_index_gfz.py --src /path/to/Kp_ap_Ap_SN_F107_since_1932.txt
    python scripts/build_daily_index_gfz.py --download            # fetch the current file first
    python scripts/build_daily_index_gfz.py --start 1985-01-01    # default; --start 1932-01-01 for all
"""
import argparse
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geoindex_daily.daily_index import DAILY_COLUMNS, default_data_dir, load_daily  # noqa: E402

GFZ_URL = "https://kp.gfz.de/app/files/Kp_ap_Ap_SN_F107_since_1932.txt"
COLUMNS = (["year", "month", "day", "days", "days_m", "bsr", "db"]
           + [f"kp{i}" for i in range(1, 9)] + [f"ap{i}" for i in range(1, 9)]
           + ["Ap", "SN", "f107_obs", "f107_adj", "D"])


def parse_gfz(path: Path) -> pd.DataFrame:
    """GFZ text → daily table indexed by UT day with columns DAILY_COLUMNS (+ `definitive`)."""
    raw = pd.read_csv(path, sep=r"\s+", comment="#", header=None, names=COLUMNS)
    # pandas 2 parses this as datetime64[us]; reindex against a [ns] date_range would then
    # return all-NaN, so pin the unit.
    idx = pd.DatetimeIndex(pd.to_datetime(dict(year=raw.year, month=raw.month, day=raw.day))).as_unit("ns")
    idx.name = "date"
    kp = raw[[f"kp{i}" for i in range(1, 9)]].replace(-1.0, np.nan)
    ap = raw[[f"ap{i}" for i in range(1, 9)]].replace(-1, np.nan).astype(float)
    # .to_numpy(): with an explicit index the constructor would align each Series on its
    # RangeIndex and yield all-NaN columns.
    out = pd.DataFrame({
        "ap": raw["Ap"].replace(-1, np.nan).astype(float).to_numpy(),
        "kp": kp.max(axis=1).to_numpy(),
        "f107": raw["f107_obs"].replace(-1.0, np.nan).to_numpy(),
        "sn": raw["SN"].replace(-1, np.nan).astype(float).to_numpy(),
        "dst": np.nan,
        "n_ap_hours": (ap.notna().sum(axis=1) * 3).astype(int).to_numpy(),
        "ap_mean8": ap.mean(axis=1).to_numpy(),          # unrounded mean of the eight 3-h values
        "definitive": raw["D"].astype(int).to_numpy(),
    }, index=idx)
    full = pd.date_range(out.index.min(), out.index.max(), freq="D", name="date")
    out = out.reindex(full)
    out["n_ap_hours"] = out["n_ap_hours"].fillna(0).astype(int)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default=None, help="local copy of the GFZ file (default: <data dir>/gfz/…)")
    p.add_argument("--download", action="store_true", help="download the current file to --src first")
    p.add_argument("--out", default=None, help="parquet path (default: <data dir>/daily_index_gfz.parquet)")
    p.add_argument("--start", default="1985-01-01")
    p.add_argument("--end", default=None, help="YYYY-MM-DD inclusive (default: last row)")
    p.add_argument("--compare", default=None, help="OMNI-derived parquet to cross-check against")
    args = p.parse_args()

    d = default_data_dir()
    src = Path(args.src).expanduser() if args.src else d / "gfz" / "Kp_ap_Ap_SN_F107_since_1932.txt"
    if args.download or not src.exists():
        src.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {GFZ_URL} → {src}")
        urllib.request.urlretrieve(GFZ_URL, src)

    daily = parse_gfz(src)
    daily = daily.loc[args.start: args.end]
    out = Path(args.out).expanduser() if args.out else d / "daily_index_gfz.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(out)
    n_prelim = int((daily.definitive == 0).sum())
    print(f"wrote {out}: {len(daily)} days {daily.index.min().date()} .. {daily.index.max().date()}, "
          f"{int(daily.ap.isna().sum())} missing Ap, {n_prelim} preliminary days "
          f"(from {daily.index[daily.definitive == 0].min().date() if n_prelim else '—'})")
    print(daily[["ap", "kp", "f107", "sn"]].describe().round(1).to_string())

    cmp_path = Path(args.compare).expanduser() if args.compare else d / "daily_index.parquet"
    if cmp_path.exists():
        omni = load_daily(cmp_path)
        j = daily[["ap", "ap_mean8"]].join(omni[["ap"]].rename(columns={"ap": "ap_omni"}), how="inner").dropna()
        diff_round = (j.ap - j.ap_omni).abs()
        diff_mean = (j.ap_mean8 - j.ap_omni).abs()
        print(f"\ncross-check vs {cmp_path.name}: {len(j)} common days; "
              f"|Ap_gfz − Ap_omni| max {diff_round.max():.2f}, mean {diff_round.mean():.3f} "
              f"(GFZ Ap is rounded); |mean8 − Ap_omni| max {diff_mean.max():.3f}, "
              f"days with mean8 ≠ omni (> 1e-6): {int((diff_mean > 1e-6).sum())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
