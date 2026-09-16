"""Train SINet (the co-author's F10.7 model) on daily Ap: 30-day input → 60-day output.

Reproduces the recipe of `F107_train.py` on our windows and splits — MinMax scaling fitted
on the training windows, MSE loss, Adam 1e-3, batch 32, the epoch with the lowest validation
MSE kept — and scores the test split per lead in Ap units (MAE, CC) next to the references
on the same samples (climatology, persistence, recurrence-27). Several seeds are trained;
the seed mean is reported as the ensemble.

    python scripts/sinet_train.py --config configs/ap_1985.yaml --tag paper          # faithful: raw Ap, 10 epochs
    python scripts/sinet_train.py --config configs/ap_1985.yaml --tag log1p \\
        --transform log1p --epochs 50 --patience 8                                    # our scaling + longer training

Artifacts under `<data dir>/sinet/<tag>/`: per-seed checkpoint + history; and in the data dir
`sinet_<tag>_<index>.csv` (scores by lead) and `sinet_<tag>_<index>_test_preds.npz`
(test forecasts keyed by issue date, same layout as ts_only.py for the SWPC comparison).
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geoindex_daily.baselines import recurrence_lag  # noqa: E402
from geoindex_daily.daily_index import default_data_dir, load_daily  # noqa: E402
from geoindex_daily.metrics import corr, mae  # noqa: E402
from geoindex_daily.models.sinet import SINet, SINetConfig, count_parameters  # noqa: E402
from geoindex_daily.windows import make_windows, split  # noqa: E402

SHOW_LEADS = [1, 3, 7, 14, 27, 45, 60]
TRANSFORMS = {"log1p": (np.log1p, np.expm1), "none": (lambda x: x, lambda x: x)}


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def score_by_lead(Y, P, leads) -> pd.DataFrame:
    rows = []
    for h in leads:
        y, p = pd.Series(Y[:, h - 1]), pd.Series(P[:, h - 1])
        rows.append({"lead": h, "mae": mae(y, p), "corr": corr(y, p)})
    return pd.DataFrame(rows)


def train_one(seed, Xs, Ys, tr, va, cfg_model, args, device, run_dir):
    set_seed(seed)
    model = SINet(cfg_model).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()
    xtr = torch.tensor(Xs[tr], dtype=torch.float32).unsqueeze(-1)
    ytr = torch.tensor(Ys[tr], dtype=torch.float32).unsqueeze(-1)
    xva = torch.tensor(Xs[va], dtype=torch.float32).unsqueeze(-1).to(device)
    yva = torch.tensor(Ys[va], dtype=torch.float32).unsqueeze(-1).to(device)
    g = torch.Generator().manual_seed(seed)
    best, best_state, best_epoch, bad, history = float("inf"), None, 0, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(xtr), generator=g)
        total, t0 = 0.0, time.time()
        for i in range(0, len(perm), args.batch):
            idx = perm[i: i + args.batch]
            xb, yb = xtr[idx].to(device), ytr[idx].to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            vloss = float(loss_fn(model(xva), yva))
        history.append({"epoch": epoch, "train_mse": total / len(perm), "val_mse": vloss, "seconds": time.time() - t0})
        print(f"  seed {seed} epoch {epoch}/{args.epochs} train {total / len(perm):.6f} val {vloss:.6f} ({time.time() - t0:.0f} s)", flush=True)
        if vloss < best - 1e-7:
            best, best_epoch, bad = vloss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if args.patience and bad >= args.patience:
                print(f"  early stop at epoch {epoch} (best {best_epoch})")
                break
    model.load_state_dict(best_state)
    torch.save(best_state, run_dir / f"sinet_seed{seed}.pt")
    (run_dir / f"history_seed{seed}.json").write_text(json.dumps({"best_epoch": best_epoch, "best_val_mse": best,
                                                                     "history": history}, indent=2))
    return model, best_epoch


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/ap_1985.yaml")
    p.add_argument("--data", default=None, help="daily parquet (default: config data_file in the data dir)")
    p.add_argument("--tag", default="paper")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--transform", default="none", choices=list(TRANSFORMS),
                   help="applied before MinMax scaling; 'none' is the paper's recipe")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=0, help="early-stopping patience on val MSE (0 = off)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--d-model", type=int, default=32)
    p.add_argument("--d-ff", type=int, default=64)
    p.add_argument("--e-layers", type=int, default=2)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--num-kernels", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    L, H, index = cfg["input_days"], cfg["output_days"], cfg["index"]
    data_path = Path(args.data).expanduser() if args.data else default_data_dir() / cfg.get("data_file", "daily_index.parquet")
    fwd, inv = TRANSFORMS[args.transform]
    s = load_daily(data_path)[index]
    X, Y, dates = make_windows(fwd(s), L, H)
    masks = split(dates, cfg["splits"])
    tr, va, te = masks["train"], masks["val"], masks["test"]
    print(f"data {data_path.name}: {s.index.min().date()}..{s.index.max().date()}; windows {len(dates)} "
          f"(train {tr.sum()}, val {va.sum()}, test {te.sum()} {dates[te].min().date()}..{dates[te].max().date()})")

    # MinMax on the training windows (F107_train.py fits it on the training values)
    lo, hi = float(X[tr].min()), float(X[tr].max())
    Xs, Ys = (X - lo) / (hi - lo), (Y - lo) / (hi - lo)
    unscale = lambda a: inv(a * (hi - lo) + lo)  # noqa: E731

    device = torch.device(args.device)
    cfg_model = SINetConfig(seq_len=L, pred_len=H, d_model=args.d_model, d_ff=args.d_ff, e_layers=args.e_layers,
                            top_k=args.top_k, num_kernels=args.num_kernels, dropout=args.dropout)
    print(f"SINet {count_parameters(SINet(cfg_model)):,} parameters; transform={args.transform}, "
          f"minmax [{lo:.3f}, {hi:.3f}], epochs {args.epochs}, patience {args.patience}, device {device}")
    run_dir = default_data_dir() / "sinet" / args.tag
    run_dir.mkdir(parents=True, exist_ok=True)

    leads = list(range(1, H + 1))
    Yte = inv(Y[te])
    preds, results, best_epochs = {}, {}, {}
    xte = torch.tensor(Xs[te], dtype=torch.float32).unsqueeze(-1).to(device)
    for seed in args.seeds:
        model, best_epoch = train_one(seed, Xs, Ys, tr, va, cfg_model, args, device, run_dir)
        best_epochs[seed] = best_epoch
        model.eval()
        with torch.no_grad():
            P = np.clip(unscale(model(xte)[..., 0].cpu().numpy()), 0.0, None)
        preds[f"sinet_seed{seed}"] = P
        results[f"sinet_seed{seed}"] = score_by_lead(Yte, P, leads)
    ens = np.mean([preds[f"sinet_seed{s_}"] for s_ in args.seeds], axis=0)
    preds["sinet"] = ens
    results["sinet"] = score_by_lead(Yte, ens, leads)

    # references on the same test samples (as in ts_only.py)
    clim = np.full_like(Yte, float(inv(Y[tr].mean())))
    pers = np.repeat(inv(X[te][:, -1:]), H, axis=1)
    rec = np.stack([np.array([X_[L - 1 - recurrence_lag(h, cfg["rotation_days"])] if recurrence_lag(h, cfg["rotation_days"]) < L else np.nan
                              for h in leads]) for X_ in X[te]])
    for name, P in {"climatology": clim, "persistence": pers, "recurrence27": inv(rec)}.items():
        preds[name] = P
        results[name] = score_by_lead(Yte, P, leads)

    table = pd.concat({k: v.set_index("lead") for k, v in results.items()}, axis=1)
    out_csv = default_data_dir() / f"sinet_{args.tag}_{index}.csv"
    table.to_csv(out_csv)
    np.savez(default_data_dir() / f"sinet_{args.tag}_{index}_test_preds.npz",
             dates=dates[te].strftime("%Y-%m-%d").to_numpy().astype("U10"), y=Yte, **preds)
    (run_dir / "run.json").write_text(json.dumps({"args": vars(args), "data": str(data_path), "n_train": int(tr.sum()),
                                                  "n_val": int(va.sum()), "n_test": int(te.sum()), "minmax": [lo, hi],
                                                  "best_epochs": best_epochs,
                                                  "mean_over_leads": {k: v[["mae", "corr"]].mean().round(4).to_dict()
                                                                      for k, v in results.items()}}, indent=2))
    shown = table.loc[SHOW_LEADS]
    pd.set_option("display.width", 220)
    print("\nMAE (Ap units) by lead:")
    print(shown.xs("mae", axis=1, level=1).round(2).to_string())
    print("\nCC by lead:")
    print(shown.xs("corr", axis=1, level=1).round(3).to_string())
    print(f"\nmean over leads 1..{H}:")
    print(pd.DataFrame({k: v[["mae", "corr"]].mean() for k, v in results.items()}).T.round(3).to_string())
    print(f"best epochs {best_epochs}; full table → {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
