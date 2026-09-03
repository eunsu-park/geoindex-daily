"""Time-series-only models: what 30 days of the index alone can do for 60 days ahead.

Models (all direct multi-output, one weight set per lead):
  ridge_raw      ridge regression on the 30 transformed values
  ridge_moment   ridge on the frozen MOMENT embedding of the window
  ridge_both     ridge on [embedding, raw values]
plus climatology and 27-day recurrence on the same test samples for reference.

Ridge alpha is chosen on the val split; scores are on the test split, in the index's
own units (the transform is inverted), with MAE and CC per lead.

    python scripts/ts_only.py --config configs/ap.yaml                 # MOMENT-large
    python scripts/ts_only.py --config configs/ap.yaml --moment AutonLab/MOMENT-1-small
    python scripts/ts_only.py --config configs/ap.yaml --no-moment     # raw ridge only

Embeddings are cached under `<data dir>/moment/<model>_<index>_<transform>_L<input>.npz`.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geoindex_daily.baselines import recurrence_lag  # noqa: E402
from geoindex_daily.daily_index import default_data_dir, load_daily  # noqa: E402
from geoindex_daily.metrics import corr, mae  # noqa: E402
from geoindex_daily.windows import make_windows, split  # noqa: E402

SHOW_LEADS = [1, 3, 7, 14, 27, 45, 60]
TRANSFORMS = {"log1p": (np.log1p, np.expm1), "none": (lambda x: x, lambda x: x)}


def ridge_fit(X, Y, alpha):
    """Closed-form ridge with intercept; returns (W, b)."""
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    W = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(X.shape[1]), Xc.T @ Yc)
    return W, my - mx @ W


def ridge_predict(model, X):
    W, b = model
    return X @ W + b


def fit_select(Xtr, Ytr, Xva, Yva, alphas=(0.01, 0.1, 1, 10, 100, 1000, 10000)):
    best = None
    for a in alphas:
        m = ridge_fit(Xtr, Ytr, a)
        err = np.abs(ridge_predict(m, Xva) - Yva).mean()
        if best is None or err < best[0]:
            best = (err, a, m)
    return best[2], best[1]


def score_by_lead(Y, P, leads, inv):
    """MAE / CC per lead in index units."""
    rows = []
    for h in leads:
        y, p = pd.Series(inv(Y[:, h - 1])), pd.Series(inv(P[:, h - 1]))
        rows.append({"lead": h, "mae": mae(y, p), "corr": corr(y, p)})
    return pd.DataFrame(rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/ap.yaml")
    p.add_argument("--data", default=None)
    p.add_argument("--moment", default=None, help="MOMENT model id (default from config)")
    p.add_argument("--no-moment", action="store_true")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    fwd, inv = TRANSFORMS[cfg.get("ts_transform", "log1p")]
    L, H = cfg["input_days"], cfg["output_days"]
    s = load_daily(args.data)[cfg["index"]]
    X, Y, dates = make_windows(fwd(s), L, H)
    masks = split(dates, cfg["splits"])
    tr, va, te = masks["train"], masks["val"], masks["test"]
    print(f"windows: {len(dates)} (train {tr.sum()}, val {va.sum()}, test {te.sum()})")

    leads = list(range(1, H + 1))
    results, preds = {}, {}

    # reference forecasts on the same test samples
    clim = np.full_like(Y[te], Y[tr].mean())
    results["climatology"] = score_by_lead(Y[te], clim, leads, inv); preds["climatology"] = clim
    rec = np.stack([np.array([X_[L - 1 - recurrence_lag(h, cfg["rotation_days"])] if recurrence_lag(h, cfg["rotation_days"]) < L else np.nan
                              for h in leads]) for X_ in X[te]])
    results["recurrence27"] = score_by_lead(Y[te], rec, leads, inv); preds["recurrence27"] = rec

    feats = {"ridge_raw": X}
    if not args.no_moment:
        from geoindex_daily.encoders import moment as mo
        model_id = args.moment or cfg["moment"]["model"]
        seq_len = cfg["moment"]["seq_len"]
        cache = (default_data_dir() / "moment" /
                 f"{model_id.split('/')[-1]}_{cfg['index']}_{cfg.get('ts_transform', 'log1p')}_L{L}.npz")
        if cache.exists() and len(np.load(cache, allow_pickle=True)["dates"]) == len(dates):
            E = np.load(cache)["E"]
            print(f"embeddings from cache {cache.name}: {E.shape}")
        else:
            model = mo.load(model_id, "embedding", device=args.device)
            E = mo.embed(model, X, seq_len)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, E=E, dates=dates.strftime("%Y-%m-%d").to_numpy().astype("U10"))
            print(f"embeddings computed: {E.shape} → {cache}")
        feats["ridge_moment"] = E
        feats["ridge_both"] = np.concatenate([E, X], axis=1)

    for name, F in feats.items():
        mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-8
        Fz = (F - mu) / sd
        model, alpha = fit_select(Fz[tr], Y[tr], Fz[va], Y[va])
        P = ridge_predict(model, Fz[te])
        results[name] = score_by_lead(Y[te], P, leads, inv); preds[name] = P
        print(f"{name}: alpha={alpha}")

    table = pd.concat({k: v.set_index("lead") for k, v in results.items()}, axis=1)
    out = default_data_dir() / f"ts_only_{cfg['index']}.csv"
    table.to_csv(out)
    # test-split forecasts in index units, keyed by issue date, for like-for-like comparisons
    np.savez(default_data_dir() / f"ts_only_{cfg['index']}_test_preds.npz",
             dates=dates[te].strftime("%Y-%m-%d").to_numpy().astype("U10"), y=inv(Y[te]),
             **{k: inv(v) for k, v in preds.items()})
    shown = table.loc[SHOW_LEADS]
    pd.set_option("display.width", 220)
    print("\nMAE (index units) by lead:")
    print(shown.xs("mae", axis=1, level=1).round(2).to_string())
    print("\nCC by lead:")
    print(shown.xs("corr", axis=1, level=1).round(3).to_string())
    print(f"\nmean over leads 1..{H}:")
    print(pd.DataFrame({k: v[["mae", "corr"]].mean() for k, v in results.items()}).T.round(3).to_string())
    print(f"full table → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
