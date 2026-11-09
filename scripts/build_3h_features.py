"""Build the 3-hour feature table for phase l6 (3-hour ap, GOES X-ray 3-hour mean / max) → parquet.

Needs the GFZ file (scripts/build_daily_index_gfz.py --download) and the SOLARIS_DB_* env vars for the
`space_weather` database. Checks that each day's eight ap average to GFZ's daily Ap, which GFZ rounds
half to even (6.5 → 6, 7.5 → 8; checked 2010–2025).

    python scripts/build_3h_features.py                  # 2010-05-01 .. 2026-01-04 → $GEOINDEX_DAILY_DATA/features_3h.parquet
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geoindex_daily.daily_index import default_data_dir  # noqa: E402
from geoindex_daily.db import connect  # noqa: E402
from geoindex_daily.features import build_3h_features  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gfz", default=None, help="default: data dir/gfz/Kp_ap_Ap_SN_F107_since_1932.txt")
    p.add_argument("--out", default=None, help="default: data dir/features_3h.parquet")
    p.add_argument("--start", default="2010-05-01")
    p.add_argument("--end", default="2026-01-04", help="exclusive; three days past 2025 so 72-h targets of the last issue days exist")
    args = p.parse_args()

    data = default_data_dir()
    gfz = Path(args.gfz) if args.gfz else data / "gfz" / "Kp_ap_Ap_SN_F107_since_1932.txt"
    out = Path(args.out) if args.out else data / "features_3h.parquet"
    with connect("space_weather") as conn:
        df = build_3h_features(gfz, conn, out, args.start, args.end)
    print(f"wrote {out}: {len(df)} bins {df.index.min()} .. {df.index.max()}")
    print("missing bins per column:\n" + df[["ap", "xray_mean_log", "xray_max_log"]].isna().sum().to_string())
    print(f"bins with < 120 good X-ray minutes: {(df.n_xray_min < 120).sum()} of {len(df)}")

    daily = df["ap"].resample("D").agg(["mean", "count"])
    gd = __import__("pandas").read_parquet(data / "daily_index_gfz.parquet")["ap"].reindex(daily.index)
    full = daily["count"] == 8
    bad = full & (np.round(daily["mean"]) != gd)          # numpy rounds half to even, as GFZ does
    print(f"days with 8 ap: {int(full.sum())}; days whose rounded mean differs from GFZ Ap: {int(bad.sum())}")
    return 0 if not bad.any() else 1


if __name__ == "__main__":
    sys.exit(main())
