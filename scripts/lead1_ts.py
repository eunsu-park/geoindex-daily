"""Lead-1 time-series arms: ridge and a 1-D CNN over L days of daily features (l1).

For every split (A, B), arm (A: Ap only · B: + X-ray · C: + solar wind · E: all) and input
length L, fits a ridge (alpha on val) and the 1-D CNN (3 seeds, early stopping on val
MAE in Ap units), on the SAME issue days for every (arm, L). Scores at lead 1 next to
persistence and climatology, then a monthly block bootstrap of (L vs L = 1) within each
arm and (arm vs A) at each L.

    python scripts/lead1_ts.py --config configs/ap_lead1.yaml
    python scripts/lead1_ts.py --arms A --splits A --lengths 1 3 7 --no-cnn   # quick look
Outputs in $GEOINDEX_DAILY_DATA/lead1/: l1_scores.csv, l1_bootstrap.csv, l1_samples.csv,
l1_preds_<split>.npz.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geoindex_daily.daily_index import default_data_dir  # noqa: E402
from geoindex_daily.features import load_daily_features  # noqa: E402
from geoindex_daily.lead1 import (  # noqa: E402
    block_bootstrap_diff, common_issue_days, feature_windows, ffill_features,
    image_complete_days, score_lead1, split_masks,
)
from geoindex_daily.models.ts_cnn import default_device, predict, train_ts_cnn  # noqa: E402
from geoindex_daily.slots import load_slot_table  # noqa: E402


def ridge_fit(X, y, alpha):
    mx, my = X.mean(0), y.mean()
    Xc = X - mx
    W = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(X.shape[1]), Xc.T @ (y - my))
    return W, my - mx @ W


def ridge_select(Xtr, ytr, Xva, yva, alphas, inv):
    best = None
    for a in alphas:
        W, b = ridge_fit(Xtr, ytr, a)
        err = np.abs(inv(Xva @ W + b) - inv(yva)).mean()
        if best is None or err < best[0]:
            best = (err, a, (W, b))
    return best[2], best[1]


class Standardiser:
    def __init__(self, X):          # X: (N, L, F) → per-feature stats
        self.m = X.reshape(-1, X.shape[-1]).mean(0)
        self.s = X.reshape(-1, X.shape[-1]).std(0) + 1e-8

    def __call__(self, X):
        return (X - self.m) / self.s


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/ap_lead1.yaml")
    p.add_argument("--arms", nargs="*", default=None)
    p.add_argument("--splits", nargs="*", default=None)
    p.add_argument("--lengths", nargs="*", type=int, default=None)
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    p.add_argument("--no-cnn", action="store_true")
    p.add_argument("--bootstrap", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--tag", default="l1")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    data = default_data_dir()
    out = data / cfg["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    arms = args.arms or list(cfg["features"])
    splits = args.splits or list(cfg["splits"])
    lengths = args.lengths or cfg["input_days"]
    seeds = args.seeds or cfg["seeds"]
    n_boot = args.bootstrap if args.bootstrap is not None else cfg["bootstrap"]["n"]
    device = args.device or default_device()
    thr = cfg["storm_threshold"]
    tf, inv = np.log1p, np.expm1

    # ---- samples: one issue-day set for everything --------------------------------------
    df = load_daily_features(data / cfg["data_file"])
    all_cols = sorted({c for a in arms for c in cfg["features"][a]})
    df = ffill_features(df, all_cols, cfg["ffill_days"])
    df["ap_t"] = tf(df["ap"])                       # the transformed target / input
    slots = load_slot_table(data / cfg["slot_file"])
    img_days = image_complete_days(slots, cfg["image_sources"], tuple(cfg["slot_hours"]),
                                   cfg["sample_window_days"])
    days = common_issue_days(df, all_cols, cfg["history_days"], cfg["lead"], tuple(cfg["period"]), img_days)
    print(f"issue days: {len(days)} ({days.min().date()} .. {days.max().date()}), "
          f"features {all_cols}, device {device}")
    y_all = df["ap"].to_numpy()[df.index.get_indexer(days) + cfg["lead"]]
    pers = df["ap"].to_numpy()[df.index.get_indexer(days)]

    scores, boots, samples = [], [], []
    for sp in splits:
        masks = split_masks(days, cfg["splits"][sp])
        tr, va, te = masks["train"], masks["val"], masks["test"]
        dte = days[te]
        samples.append({"split": sp, "train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())})
        print(f"\n== split {sp}: train {tr.sum()} val {va.sum()} test {te.sum()}")
        preds = {"y": y_all[te], "dates": dte.to_numpy()}
        # references on the same test days
        preds["persistence"] = pers[te]
        preds["climatology"] = np.full(te.sum(), y_all[tr].mean())
        for name in ("persistence", "climatology"):
            scores.append({"split": sp, "arm": "ref", "model": name, "L": 0, "seed": -1,
                           **score_lead1(y_all[te], preds[name], dte, thr)})
        for arm in arms:
            cols = [("ap_t" if c == "ap" else c) for c in cfg["features"][arm]]
            for L in lengths:
                X, yt = feature_windows(df, cols, L, days, target_col="ap_t", lead=cfg["lead"])
                std = Standardiser(X[tr])
                Xs = std(X)
                ym, ys = yt[tr].mean(), yt[tr].std()
                yz = (yt - ym) / ys
                # ridge on the flattened window
                Xf = Xs.reshape(len(Xs), -1)
                (W, b), alpha = ridge_select(Xf[tr], yz[tr], Xf[va], yz[va], cfg["ridge"]["alphas"],
                                             lambda z: inv(z * ys + ym))
                pr = inv((Xf[te] @ W + b) * ys + ym)
                preds[f"{arm}_ridge_L{L}"] = pr
                scores.append({"split": sp, "arm": arm, "model": "ridge", "L": L, "seed": -1, "alpha": alpha,
                               **score_lead1(y_all[te], pr, dte, thr)})
                line = f"  {arm} L={L:2d} ridge(a={alpha:g}) MAE {scores[-1]['mae']:.3f} CC {scores[-1]['corr']:.3f}"
                if not args.no_cnn:
                    t0 = time.time()
                    pcs = []
                    for seed in seeds:
                        vm = lambda pv: np.abs(inv(pv * ys + ym) - inv(yt[va])).mean()  # noqa: E731
                        model, hist = train_ts_cnn(Xs[tr], yz[tr], Xs[va], yz[va], seed=seed, device=device,
                                                   val_metric=vm, **cfg["ts_cnn"])
                        pc = inv(predict(model, Xs[te], device) * ys + ym)
                        pcs.append(pc)
                        preds[f"{arm}_cnn_L{L}_s{seed}"] = pc
                        scores.append({"split": sp, "arm": arm, "model": "cnn", "L": L, "seed": seed,
                                       "epochs": hist["best_epoch"], **score_lead1(y_all[te], pc, dte, thr)})
                    pm = np.mean(pcs, axis=0)
                    preds[f"{arm}_cnn_L{L}"] = pm
                    scores.append({"split": sp, "arm": arm, "model": "cnn", "L": L, "seed": "mean",
                                   **score_lead1(y_all[te], pm, dte, thr)})
                    cc = [s for s in scores if s["split"] == sp and s["arm"] == arm and s["L"] == L
                          and s["model"] == "cnn" and s["seed"] != "mean"]
                    line += (f" | cnn MAE {scores[-1]['mae']:.3f} CC {scores[-1]['corr']:.3f} "
                             f"(seeds MAE {np.mean([s['mae'] for s in cc]):.3f}±{np.std([s['mae'] for s in cc]):.3f}, "
                             f"{time.time() - t0:.0f}s)")
                print(line)
        np.savez(out / f"{args.tag}_preds_{sp}.npz", **preds)

        # ---- bootstrap: L vs L=1 within arm; arm vs A at each L ---------------------------
        models = ["ridge"] + ([] if args.no_cnn else ["cnn"])
        yte = y_all[te]
        for m in models:
            for arm in arms:
                for L in lengths:
                    if L == lengths[0]:
                        continue
                    r = block_bootstrap_diff(yte, preds[f"{arm}_{m}_L{L}"], preds[f"{arm}_{m}_L{lengths[0]}"],
                                             dte, n_boot, cfg["bootstrap"]["seed"])
                    boots.append({"split": sp, "model": m, "comparison": f"L{L}-L{lengths[0]}", "arm": arm, "L": L, **r})
            if "A" in arms:
                for arm in arms:
                    if arm == "A":
                        continue
                    for L in lengths:
                        r = block_bootstrap_diff(yte, preds[f"{arm}_{m}_L{L}"], preds[f"A_{m}_L{L}"],
                                                 dte, n_boot, cfg["bootstrap"]["seed"])
                        boots.append({"split": sp, "model": m, "comparison": f"{arm}-A", "arm": arm, "L": L, **r})
            for arm in arms:
                for L in lengths:
                    r = block_bootstrap_diff(yte, preds[f"{arm}_{m}_L{L}"], preds["persistence"],
                                             dte, n_boot, cfg["bootstrap"]["seed"])
                    boots.append({"split": sp, "model": m, "comparison": "vs persistence", "arm": arm, "L": L, **r})

    sc = pd.DataFrame(scores)
    sc.to_csv(out / f"{args.tag}_scores.csv", index=False)
    bt = pd.DataFrame(boots)
    bt.to_csv(out / f"{args.tag}_bootstrap.csv", index=False)
    pd.DataFrame(samples).to_csv(out / f"{args.tag}_samples.csv", index=False)

    # ---- summary ---------------------------------------------------------------------------
    pd.set_option("display.width", 200)
    for sp in splits:
        print(f"\n== split {sp}: MAE / CC at lead 1 (lower MAE, higher CC is better)")
        ref = sc[(sc.split == sp) & (sc.arm == "ref")]
        print("   " + "  ".join(f"{r.model} MAE {r.mae:.3f} CC {r.corr:.3f}" for r in ref.itertuples()))
        for m in ["ridge"] + ([] if args.no_cnn else ["cnn"]):
            sub = sc[(sc.split == sp) & (sc.model == m) & (sc.seed.astype(str).isin(["-1", "mean"]))]
            print(f"   {m}:")
            print(sub.pivot(index="arm", columns="L", values="mae").round(3).to_string())
            print(sub.pivot(index="arm", columns="L", values="corr").round(3).to_string())
    print(f"\nwrote {out}/{args.tag}_scores.csv, {args.tag}_bootstrap.csv, {args.tag}_preds_<split>.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
