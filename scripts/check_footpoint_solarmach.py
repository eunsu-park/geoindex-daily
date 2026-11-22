"""Compare the offline footpoint longitudes of events/ap3h_cme_footpoint.parquet with the Solar-MACH package.

Run in an environment that has `solarmach` (it queries JPL Horizons for Earth's position, so it needs the network);
the geoindex environment does not need the package. Prints, per date and speed, the package's Carrington footpoint
longitude, ours and the difference (deg), and writes the table to --csv.

    python scripts/check_footpoint_solarmach.py --features ~/Projects/GeoIndex/daily/events/ap3h_cme_footpoint.parquet \
        --csv ~/Projects/GeoIndex/daily/events/footpoint_solarmach_check.csv
"""
import argparse

import numpy as np
import pandas as pd
from solarmach import SolarMACH

DATES = ["2010-09-01", "2012-07-15", "2014-02-28", "2016-11-03", "2019-06-21", "2021-03-20", "2022-12-31",
         "2024-05-11", "2024-10-10", "2025-08-15"]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", required=True)
    p.add_argument("--csv", required=True)
    args = p.parse_args()
    f = pd.read_parquet(args.features).set_index("issue_time")
    rows = []
    for d in DATES:
        T = pd.Timestamp(d)
        for v in (300, 400, 600):
            sm = SolarMACH(date=f"{T:%Y-%m-%d %H:%M:%S}", body_list=["Earth"], vsw_list=[v], coord_sys="Carrington",
                           silent=True)
            pkg = float(sm.coord_table["Magnetic footpoint longitude (Carrington)"].iloc[0])
            ours = float(f.loc[T, f"fp{v}_lon"])
            diff = (ours - pkg + 180) % 360 - 180
            rows.append({"issue_time": T, "vsw": v, "solarmach_deg": pkg, "offline_deg": ours, "diff_deg": diff})
    r = pd.DataFrame(rows)
    r.to_csv(args.csv, index=False)
    print(r.round(4).to_string(index=False))
    print(f"max |diff| {np.abs(r.diff_deg).max():.4f} deg over {len(r)} date-speed pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
