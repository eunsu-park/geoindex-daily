"""Models 0 / A / B by physical event type, seed by seed (work order 2026-11-22, §7.3; phase l7).

Target days labelled as in scripts/ap3h_event_type.py (CME / HSS / quiet / other, DONKI, descriptive only). Per model,
seed, lead and label: daily-Ap MAE and 3-hourly MAE. Differences A − 0, B − A, B − 0 as the difference of the seed
means, judged against the seed spread: δ = 2 · pooled seed SD · √(2/3) (the standard error of a difference of two
three-seed means, doubled).

    python scripts/ap3h_l7_event_type.py --out ~/Projects/GeoIndex/daily/ap3h-l7/compare
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ap3h_event_type import labels  # noqa: E402
from ap3h_l7_compare import load  # noqa: E402
from geoindex_daily.daily_index import default_data_dir  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    data = default_data_dir()
    ips = pd.read_parquet(data / "events" / "donki_ips_earth.parquet")
    hss = pd.read_parquet(data / "events" / "donki_hss_earth.parquet")
    rows = []
    for h in (24, 48, 72):
        P = {m: load(data, m, h) for m in ("0", "A", "B")}
        d, y = P["0"][0], P["0"][1]
        for k in range(1, h // 24 + 1):
            sl = slice((k - 1) * 8, k * 8)
            yy = y[:, sl]
            lab = labels(d.normalize() + pd.Timedelta(days=k), ips, hss, yy.mean(1))
            for name in ("CME", "HSS", "quiet", "other"):
                m = lab == name
                for mod, (_, _, mu, _) in P.items():
                    for s in range(3):
                        pp = np.expm1(mu[s][m][:, sl])
                        rows.append({"horizon_h": h, "target_day": k, "label": name, "n_days": int(m.sum()), "model": mod,
                                     "seed": s, "mae_day": np.abs(pp.mean(1) - yy[m].mean(1)).mean(),
                                     "mae_3h": np.abs(pp - yy[m]).mean()})
    T = pd.DataFrame(rows); T.to_csv(args.out / "l7_event_type_seeds.csv", index=False)
    out = []
    for (h, k, name), g in T.groupby(["horizon_h", "target_day", "label"]):
        for a, b in (("A", "0"), ("B", "A"), ("B", "0")):
            for col in ("mae_day", "mae_3h"):
                va, vb = g[g.model == a].sort_values("seed")[col].to_numpy(), g[g.model == b].sort_values("seed")[col].to_numpy()
                sd = np.sqrt((va.var(ddof=1) + vb.var(ddof=1)) / 2)
                dlt = 2 * sd * np.sqrt(2 / 3)
                out.append({"horizon_h": h, "target_day": k, "label": name, "n_days": int(g.n_days.iloc[0]),
                            "comparison": f"{a} − {b}", "metric": col, "d_seedmean": va.mean() - vb.mean(), "delta": dlt,
                            "beyond_seed_spread": abs(va.mean() - vb.mean()) > dlt,
                            "same_sign_all_seeds": bool(np.all(np.sign(va - vb) == np.sign(va.mean() - vb.mean())))})
    O = pd.DataFrame(out); O.to_csv(args.out / "l7_event_type_diff.csv", index=False)
    pd.set_option("display.width", 250)
    x = O[O.metric == "mae_day"]
    print(x.pivot_table(index=["label", "horizon_h", "target_day"], columns="comparison", values="d_seedmean").round(2).to_string())
    print("beyond seed spread:\n", O[O.beyond_seed_spread].round(3).to_string(index=False))
    print("same sign in all seeds (daily Ap):\n", x[x.same_sign_all_seeds].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
