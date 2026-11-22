"""Models 0 / A / B of the work order of 2026-11-22 at L = 4: absolute scores, paired differences, stress test, and the
Model 0 reuse check (phase l7; vault planning/cme-footpoint-plan.md).

Model 0 = the stored l6 / l6h2 / l6h3 runs; A = l7a / l7ah2 / l7ah3 (optional CME); B = l7b / l7bh2 / l7bh3 (+ footpoint);
`<A|B>-off` = the same A / B runs scored with every CME slot emptied (stress test of missing CME data). Every model on
the same 1,273 split-A test issues. Point forecast expm1(mu); the seed ensemble averages the three seeds' expm1(mu)
(and their Gaussian mixtures for CRPS / coverage). Scores by target day: 3-hourly MAE, RMSE, CC, bias (forecast −
observed), daily-Ap MAE and CC (mean of the eight bins), CRPS of log1p ap (mixture of the seeds' Gaussians, by sampling)
and the 68 / 95 % coverage of the seed-mean Gaussian. Paired differences new − reference (A − 0, B − A, B − 0, A-off − 0,
B-off − 0) of 3-hourly MAE and CC and daily-Ap MAE with 95 % monthly-block bootstrap intervals (2,000). Reuse check:
Model 0 retrained on the l7 commit (l7r) against l6 at 24 h, per seed and in the seed spread.

    python scripts/ap3h_l7_compare.py --out ~/Projects/GeoIndex/daily/ap3h-l7/compare
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from geoindex_daily.daily_index import default_data_dir  # noqa: E402

RUNS = {  # (model, horizon h) -> (directories under the data dir, tag, file suffix)
    ("0", 24): (["ap3h-l6/gpu2", "ap3h-l6/gpu3"], "l6", ""), ("0", 48): (["ap3h-l6h/gpu2", "ap3h-l6h/gpu3"], "l6h2", ""),
    ("0", 72): (["ap3h-l6h/gpu2", "ap3h-l6h/gpu3"], "l6h3", ""),
    ("A", 24): (["ap3h-l7/gpu2"], "l7a", ""), ("A", 48): (["ap3h-l7/gpu2"], "l7ah2", ""), ("A", 72): (["ap3h-l7/gpu2"], "l7ah3", ""),
    ("B", 24): (["ap3h-l7/gpu3"], "l7b", ""), ("B", 48): (["ap3h-l7/gpu3"], "l7bh2", ""), ("B", 72): (["ap3h-l7/gpu3"], "l7bh3", ""),
    ("A-off", 24): (["ap3h-l7/gpu2"], "l7a", "_cmeoff"), ("A-off", 48): (["ap3h-l7/gpu2"], "l7ah2", "_cmeoff"),
    ("A-off", 72): (["ap3h-l7/gpu2"], "l7ah3", "_cmeoff"),
    ("B-off", 24): (["ap3h-l7/gpu3"], "l7b", "_cmeoff"), ("B-off", 48): (["ap3h-l7/gpu3"], "l7bh2", "_cmeoff"),
    ("B-off", 72): (["ap3h-l7/gpu3"], "l7bh3", "_cmeoff"),
    ("0r", 24): (["ap3h-l7/gpu2"], "l7r", ""),
}
PAIRS = [("A", "0"), ("B", "A"), ("B", "0"), ("A-off", "0"), ("B-off", "0")]


def load(data, model, h, L=4):
    dirs, tag, suf = RUNS[(model, h)]
    z = []
    for s in range(3):
        f = [p for d in dirs for p in glob.glob(str(data / d / f"{tag}_preds_A_full_all_L{L}_s{s}{suf}.npz"))]
        assert len(f) == 1, (model, h, s, f)
        z.append(np.load(f[0]))
    d = pd.DatetimeIndex(z[0]["dates"])
    assert all((pd.DatetimeIndex(x["dates"]) == d).all() for x in z)
    return d, np.round(z[0]["y"], 6), np.stack([x["mu"] for x in z]), np.stack([x["sigma"] for x in z])


def crps_mixture(mu, sg, y, n=200, seed=0):
    """CRPS of log1p y under the equal mixture of the seeds' Gaussians, by sampling: E|X − y| − ½ E|X − X'|."""
    rng = np.random.default_rng(seed)
    k = rng.integers(0, mu.shape[0], (n,) + mu.shape[1:])
    x = np.take_along_axis(mu, k, 0) + np.take_along_axis(sg, k, 0) * rng.standard_normal(k.shape)
    ly = np.log1p(y)
    return (np.abs(x - ly).mean(0) - 0.5 * np.abs(x[: n // 2] - x[n // 2:]).mean(0)).mean()


def day_scores(y, p, mu=None, sg=None):
    """y, p (N, 8) ap of one target day."""
    yd, pd_ = y.mean(1), p.mean(1)
    r = {"mae_3h": np.abs(p - y).mean(), "rmse_3h": np.sqrt(((p - y) ** 2).mean()),
         "cc_3h": np.corrcoef(p.ravel(), y.ravel())[0, 1], "bias_3h": (p - y).mean(),
         "mae_day": np.abs(pd_ - yd).mean(), "cc_day": np.corrcoef(pd_, yd)[0, 1]}
    if mu is not None:
        ly, m, s = np.log1p(y), mu.mean(0), np.sqrt((sg ** 2).mean(0) + mu.var(0))
        r.update({"crps_log": crps_mixture(mu, sg, y), "cover68": (np.abs(ly - m) <= s).mean(),
                  "cover95": (np.abs(ly - m) <= 1.96 * s).mean()})
    return r


def boot_pair(y, pa, pb, dates, n=2000, seed=0):
    """Monthly-block bootstrap of new − reference: 3-hourly MAE, 3-hourly CC, daily-Ap MAE (pa new, pb reference)."""
    rng = np.random.default_rng(seed)
    months = pd.PeriodIndex(dates, freq="M")
    blocks = [np.flatnonzero(months == m) for m in months.unique()]

    def stat(i):
        yy, a, b = y[i], pa[i], pb[i]
        return (np.abs(a - yy).mean() - np.abs(b - yy).mean(),
                np.corrcoef(a.ravel(), yy.ravel())[0, 1] - np.corrcoef(b.ravel(), yy.ravel())[0, 1],
                np.abs(a.mean(1) - yy.mean(1)).mean() - np.abs(b.mean(1) - yy.mean(1)).mean())
    pt = stat(np.arange(len(y)))
    bs = np.array([stat(np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), len(blocks))])) for _ in range(n)])
    lo, hi = np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)
    return {"d_mae_3h": pt[0], "d_mae_3h_lo": lo[0], "d_mae_3h_hi": hi[0], "d_cc_3h": pt[1], "d_cc_3h_lo": lo[1],
            "d_cc_3h_hi": hi[1], "d_mae_day": pt[2], "d_mae_day_lo": lo[2], "d_mae_day_hi": hi[2]}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data = default_data_dir()
    absolute, pairs = [], []
    for h in (24, 48, 72):
        P = {m: load(data, m, h) for m in ("0", "A", "B", "A-off", "B-off")}
        d0 = P["0"][0]
        assert all((P[m][0] == d0).all() and np.array_equal(P[m][1], P["0"][1]) for m in P)
        y = P["0"][1]
        for k in range(1, h // 24 + 1):
            sl = slice((k - 1) * 8, k * 8)
            ens = {m: np.expm1(P[m][2][:, :, sl]).mean(0) for m in P}
            for m, (_, _, mu, sg) in P.items():
                absolute.append({"horizon_h": h, "target_day": k, "model": m, "forecast": "ensemble",
                                 **day_scores(y[:, sl], ens[m], mu[:, :, sl], sg[:, :, sl])})
                for s in range(3):
                    absolute.append({"horizon_h": h, "target_day": k, "model": m, "forecast": f"seed{s}",
                                     **day_scores(y[:, sl], np.expm1(mu[s, :, sl]))})
            for a, b in PAIRS:
                pairs.append({"horizon_h": h, "target_day": k, "comparison": f"{a} − {b}",
                              **boot_pair(y[:, sl], ens[a], ens[b], d0)})
    A = pd.DataFrame(absolute); A.to_csv(args.out / "l7_absolute.csv", index=False)
    Q = pd.DataFrame(pairs); Q.to_csv(args.out / "l7_paired.csv", index=False)

    # Model 0 reuse check (24 h): l7r against l6, seed by seed
    d, y, mu6, _ = load(data, "0", 24)
    _, _, mur, _ = load(data, "0r", 24)
    rows = []
    for s in range(3):
        e6, er = np.abs(np.expm1(mu6[s]) - y).mean(), np.abs(np.expm1(mur[s]) - y).mean()
        rows.append({"seed": s, "l6_mae_3h": e6, "l7r_mae_3h": er, "max_abs_mu_diff": float(np.abs(mu6[s] - mur[s]).max()),
                     "identical": bool(np.array_equal(mu6[s], mur[s]))})
    R = pd.DataFrame(rows); R.to_csv(args.out / "l7_model0_reuse_check.csv", index=False)

    pd.set_option("display.width", 250)
    e = A[A.forecast == "ensemble"]
    print(e.pivot_table(index=["horizon_h", "target_day"], columns="model", values="mae_3h").round(3).to_string())
    print(e.pivot_table(index=["horizon_h", "target_day"], columns="model", values="cc_3h").round(3).to_string())
    print(e.pivot_table(index=["horizon_h", "target_day"], columns="model", values="mae_day").round(3).to_string())
    print(e.pivot_table(index=["horizon_h", "target_day"], columns="model", values="crps_log").round(4).to_string())
    print(e.pivot_table(index=["horizon_h", "target_day"], columns="model", values="cover95").round(3).to_string())
    print(e.pivot_table(index=["horizon_h", "target_day"], columns="model", values="bias_3h").round(2).to_string())
    s = A[A.forecast != "ensemble"]
    print("seed MAE range:\n", s.groupby(["horizon_h", "target_day", "model"]).mae_3h.agg(["min", "max"]).unstack("model").round(2).to_string())
    print(Q.round(3).to_string(index=False))
    print(R.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
