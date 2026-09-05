"""Stage-1 fusion (linear): Ap history + frozen Surya embedding → 60 daily Ap values.

Arms (see the vault's planning/fusion-plan.md):
  TS          the 30 log1p-Ap values of the window
  PHASE       TS + log F10.7 + SN of the issue day (cycle-phase control)
  IMG1        Surya `mean` token of the issue day (PCA-reduced)
  IMG7        mean of the `mean` token over the last 7 days (PCA-reduced)
  TS+IMG1, TS+IMG7, PHASE+IMG7

Direct multi-output ridge per arm; PCA width k and alpha chosen on val (MAE in log space);
per-lead MAE / CC on test; monthly block bootstrap of the CC gain at leads 3-26 for the
image arms against their matched time-series arm. Test forecasts are saved in the ts_only
layout so `compare_on_prf_issues.py --preds` can score them on the PRF issue dates.

    python scripts/fusion.py --config configs/ap.yaml
    python scripts/fusion.py --config configs/ap.yaml --emb-dir ~/Projects/GeoIndex/daily/surya/emb/13ch
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geoindex_daily.daily_index import default_data_dir, load_daily  # noqa: E402
from geoindex_daily.metrics import corr, mae  # noqa: E402
from geoindex_daily.windows import make_windows, split  # noqa: E402

TRANSFORMS = {"log1p": (np.log1p, np.expm1), "none": (lambda x: x, lambda x: x)}
SHOW_LEADS = [1, 3, 7, 14, 27, 45, 60]
ALPHAS = (0.01, 0.1, 1, 10, 100, 1000, 10000)
PCA_K = (16, 64, 256)
GROUPS = {"1-60": (1, 60), "3-26": (3, 26), "27-60": (27, 60)}


# --- features -------------------------------------------------------------------------

def load_embeddings(emb_dir: Path, key: str = "mean") -> pd.DataFrame:
    """All daily embeddings as a DataFrame indexed by anchor date (from the file name)."""
    rows, idx = [], []
    for f in sorted(emb_dir.glob("2*.npz")):
        rows.append(np.load(f)[key].astype(np.float32))
        idx.append(pd.Timestamp(f.stem[:8]))
    return pd.DataFrame(np.stack(rows), index=pd.DatetimeIndex(idx, name="date"))


def daily_image_features(emb: pd.DataFrame, dates: pd.DatetimeIndex, ffill_days: int = 3):
    """IMG1 (issue day) and IMG7 (mean over t-6..t) per issue date, with forward fill.

    Returns (img1, img7, ok) where ok marks issue dates whose issue-day embedding exists
    within `ffill_days` days back; img7 averages whatever days of the 7 are available after
    filling. Counts of filled days are printed by the caller.
    """
    full = pd.date_range(emb.index.min(), emb.index.max(), freq="D")
    e = emb.reindex(full)
    present = e.notna().all(axis=1)
    filled = e.ffill(limit=ffill_days)
    ok_day = filled.notna().all(axis=1)
    img1 = filled.reindex(dates).to_numpy()
    img7 = filled.rolling(7, min_periods=1).mean().reindex(dates).to_numpy()
    ok = ok_day.reindex(dates).fillna(False).to_numpy(dtype=bool)
    n_filled = int((ok_day & ~present).reindex(dates).fillna(False).astype(bool).sum())
    return img1, img7, ok, n_filled


# --- models ---------------------------------------------------------------------------

def ridge_fit(X, Y, alpha):
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    W = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(X.shape[1]), Xc.T @ Yc)
    return W, my - mx @ W


def ridge_predict(m, X):
    return X @ m[0] + m[1]


class PCA:
    def __init__(self, k):
        self.k = k

    def fit(self, X):
        self.mu = X.mean(0)
        _, _, vt = np.linalg.svd(X - self.mu, full_matrices=False)
        self.comp = vt[: self.k].T
        return self

    def transform(self, X):
        return (X - self.mu) @ self.comp


def build_arm(parts, tr, pca_k):
    """Concatenate feature blocks; blocks tagged as embeddings are PCA-reduced (fit on train).

    parts: list of (array, is_embedding). Returns the standardised design matrix.
    """
    cols = []
    for arr, is_emb in parts:
        if is_emb and pca_k:
            arr = PCA(pca_k).fit(arr[tr]).transform(arr)
        cols.append(arr)
    F = np.concatenate(cols, axis=1)
    mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-8
    return (F - mu) / sd


def fit_arm(parts, Y, tr, va, has_emb):
    """Select (k, alpha) on val MAE (log space); return the fitted design + model + choice."""
    best = None
    for k in (PCA_K if has_emb else (None,)):
        F = build_arm(parts, tr, k)
        for a in ALPHAS:
            m = ridge_fit(F[tr], Y[tr], a)
            err = np.abs(ridge_predict(m, F[va]) - Y[va]).mean()
            if best is None or err < best[0]:
                best = (err, k, a, F, m)
    _, k, a, F, m = best
    return F, m, k, a


# --- scoring --------------------------------------------------------------------------

def per_lead(Y, P, inv):
    rows = []
    for h in range(1, Y.shape[1] + 1):
        y, p = pd.Series(inv(Y[:, h - 1])), pd.Series(inv(P[:, h - 1]))
        rows.append({"lead": h, "mae": mae(y, p), "corr": corr(y, p)})
    return pd.DataFrame(rows).set_index("lead")


def cc_matrix(Y, P):
    """Per-lead Pearson CC for many resamples at once: Y, P (n, leads) → (leads,)."""
    yc = Y - Y.mean(0)
    pc = P - P.mean(0)
    return (yc * pc).sum(0) / np.sqrt((yc ** 2).sum(0) * (pc ** 2).sum(0) + 1e-12)


def block_bootstrap_gain(Y, Pa, Pb, dates, leads=(3, 26), n=1000, seed=0):
    """Monthly block bootstrap of mean CC(Pa) − mean CC(Pb) over `leads` (Ap units)."""
    rng = np.random.default_rng(seed)
    months = pd.PeriodIndex(dates, freq="M")
    blocks = [np.flatnonzero(months == m) for m in months.unique()]
    lo, hi = leads
    sl = slice(lo - 1, hi)
    gains = []
    for _ in range(n):
        idx = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
        gains.append(cc_matrix(Y[idx, sl], Pa[idx, sl]).mean() - cc_matrix(Y[idx, sl], Pb[idx, sl]).mean())
    g = np.array(gains)
    return float(np.mean(g)), float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/ap.yaml")
    p.add_argument("--emb-dir", default=None, help="Surya embedding dir (default data dir/surya/emb/13ch)")
    p.add_argument("--ffill-days", type=int, default=3)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--tag", default="fusion_s1")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    fwd, inv = TRANSFORMS[cfg.get("ts_transform", "log1p")]
    L, H = cfg["input_days"], cfg["output_days"]
    d = load_daily()
    X, Y, dates = make_windows(fwd(d[cfg["index"]]), L, H)

    emb_dir = Path(args.emb_dir) if args.emb_dir else default_data_dir() / "surya" / "emb" / "13ch"
    emb = load_embeddings(emb_dir, "mean")
    img1, img7, ok, n_filled = daily_image_features(emb, dates, args.ffill_days)
    phase = np.stack([np.log(d["f107"].reindex(dates).to_numpy()), d["sn"].reindex(dates).to_numpy()], axis=1)
    keep = ok & np.isfinite(phase).all(axis=1)
    print(f"windows {len(dates)}; with embedding {keep.sum()} (dropped {(~keep).sum()}, "
          f"forward-filled issue days {n_filled}); embeddings {len(emb)} days {emb.index.min().date()}..{emb.index.max().date()}")
    X, Y, dates, img1, img7, phase = X[keep], Y[keep], dates[keep], img1[keep], img7[keep], phase[keep]
    masks = split(dates, cfg["splits"])
    tr, va, te = masks["train"], masks["val"], masks["test"]
    print(f"train {tr.sum()}  val {va.sum()}  test {te.sum()}")

    arms = {
        "TS": ([(X, False)], False),
        "PHASE": ([(X, False), (phase, False)], False),
        "IMG1": ([(img1, True)], True),
        "IMG7": ([(img7, True)], True),
        "TS+IMG1": ([(X, False), (img1, True)], True),
        "TS+IMG7": ([(X, False), (img7, True)], True),
        "PHASE+IMG7": ([(X, False), (phase, False), (img7, True)], True),
    }
    preds, tables, choices = {}, {}, {}
    for name, (parts, has_emb) in arms.items():
        F, m, k, a = fit_arm(parts, Y, tr, va, has_emb)
        P = ridge_predict(m, F[te])
        preds[name] = P
        tables[name] = per_lead(Y[te], P, inv)
        choices[name] = (k, a)
        print(f"{name:11s} k={k} alpha={a}")

    # summary: per-group means
    rows = []
    for name, t in tables.items():
        r = {"arm": name, "k": choices[name][0], "alpha": choices[name][1]}
        for g, (lo, hi) in GROUPS.items():
            r[f"mae_{g}"] = t.loc[lo:hi, "mae"].mean()
            r[f"cc_{g}"] = t.loc[lo:hi, "corr"].mean()
        rows.append(r)
    summary = pd.DataFrame(rows).set_index("arm")
    pd.set_option("display.width", 220)
    print("\nTest 2022-2025, means over lead groups (MAE in Ap units, CC):")
    print(summary.round(3).to_string())

    show = pd.concat({n: t.loc[SHOW_LEADS] for n, t in tables.items()}, axis=1)
    print("\nCC by lead:")
    print(show.xs("corr", axis=1, level=1).round(3).to_string())
    print("\nMAE by lead:")
    print(show.xs("mae", axis=1, level=1).round(2).to_string())

    # bootstrap gains at leads 3-26 (Ap units)
    Yte = inv(Y[te])
    boot = []
    for a_, b_ in (("TS+IMG1", "TS"), ("TS+IMG7", "TS"), ("PHASE", "TS"), ("PHASE+IMG7", "PHASE"), ("IMG7", "TS")):
        g, lo, hi = block_bootstrap_gain(Yte, inv(preds[a_]), inv(preds[b_]), dates[te], (3, 26), args.bootstrap)
        boot.append({"arm": a_, "vs": b_, "cc_gain_3_26": g, "ci_lo": lo, "ci_hi": hi})
    boot = pd.DataFrame(boot)
    print(f"\nMonthly block bootstrap ({args.bootstrap}): mean CC gain over leads 3-26 with 95% CI")
    print(boot.round(4).to_string(index=False))

    out = default_data_dir()
    summary.to_csv(out / f"{args.tag}_{cfg['index']}_summary.csv")
    pd.concat(tables, axis=1).to_csv(out / f"{args.tag}_{cfg['index']}_by_lead.csv")
    boot.to_csv(out / f"{args.tag}_{cfg['index']}_bootstrap.csv", index=False)
    np.savez(out / f"{args.tag}_{cfg['index']}_test_preds.npz",
             dates=dates[te].strftime("%Y-%m-%d").to_numpy().astype("U10"), y=Yte,
             **{n.replace("+", "_"): inv(P) for n, P in preds.items()})
    print(f"\nartifacts → {out}/{args.tag}_{cfg['index']}_*.csv / _test_preds.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
