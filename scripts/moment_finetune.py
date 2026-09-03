"""Fine-tune MOMENT's forecasting head (optionally the encoder too) on 30-day → 60-day windows.

The frozen-embedding probe (`ts_only.py`) showed MOMENT's embedding carries less than the
raw 30 values; MOMENT's own recipe for forecasting is to train the forecasting head, which
is randomly initialised. This script does that with the same windows, splits, transform and
scores as `ts_only.py`, so the numbers are directly comparable, and saves test forecasts in
the same npz layout for `compare_on_prf_issues.py`.

    python scripts/moment_finetune.py --config configs/ap.yaml --moment AutonLab/MOMENT-1-small --epochs 10
    python scripts/moment_finetune.py --config configs/ap.yaml --train-encoder --lr 1e-5
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geoindex_daily.daily_index import default_data_dir, load_daily  # noqa: E402
from geoindex_daily.encoders import moment as mo  # noqa: E402
from geoindex_daily.metrics import corr, mae  # noqa: E402
from geoindex_daily.windows import make_windows, split  # noqa: E402

SHOW_LEADS = [1, 3, 7, 14, 27, 45, 60]
TRANSFORMS = {"log1p": (np.log1p, np.expm1), "none": (lambda x: x, lambda x: x)}


def batches(X, Y, size, shuffle, rng):
    idx = np.arange(len(X))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), size):
        j = idx[i: i + size]
        yield X[j], Y[j]


def predict(model, X, seq_len, device, batch=128):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            x, m = mo.pack(X[i: i + batch], seq_len)
            out.append(model(x_enc=x.to(device), input_mask=m.to(device)).forecast[:, 0, :].float().cpu().numpy())
    return np.concatenate(out)


def score_by_lead(Y, P, leads, inv):
    return pd.DataFrame([{"lead": h, "mae": mae(pd.Series(inv(Y[:, h - 1])), pd.Series(inv(P[:, h - 1]))),
                          "corr": corr(pd.Series(inv(Y[:, h - 1])), pd.Series(inv(P[:, h - 1])))} for h in leads])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/ap.yaml")
    p.add_argument("--moment", default=None)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3, help="head lr (encoder uses lr/10 when trained)")
    p.add_argument("--train-encoder", action="store_true", help="also update the encoder (else head only)")
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default=None, help="name for the saved forecasts (default: moment_<size>[_enc])")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    fwd, inv = TRANSFORMS[cfg.get("ts_transform", "log1p")]
    L, H = cfg["input_days"], cfg["output_days"]
    seq_len = cfg["moment"]["seq_len"]
    model_id = args.moment or cfg["moment"]["model"]
    s = load_daily()[cfg["index"]]
    X, Y, dates = make_windows(fwd(s), L, H)
    masks = split(dates, cfg["splits"])
    tr, va, te = masks["train"], masks["val"], masks["test"]
    print(f"windows: {len(dates)} (train {tr.sum()}, val {va.sum()}, test {te.sum()})")

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = args.device or mo.device_auto()
    model = mo.load(model_id, "forecasting", H, device)
    head_params = [p_ for n, p_ in model.named_parameters() if n.startswith("head")]
    enc_params = [p_ for n, p_ in model.named_parameters() if not n.startswith("head")]
    for p_ in enc_params:
        p_.requires_grad_(args.train_encoder)
    groups = [{"params": head_params, "lr": args.lr}]
    if args.train_encoder:
        groups.append({"params": enc_params, "lr": args.lr / 10})
    opt = torch.optim.AdamW(groups, weight_decay=1e-2)
    print(f"{model_id} on {device}: head {sum(p_.numel() for p_ in head_params)/1e6:.2f}M params, "
          f"encoder {'trained' if args.train_encoder else 'frozen'} ({sum(p_.numel() for p_ in enc_params)/1e6:.0f}M)")

    # targets standardised on train so the MSE is well scaled; inverted for scoring
    ym, ys = Y[tr].mean(), Y[tr].std()
    best, best_state, bad = np.inf, None, 0
    for ep in range(args.epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for xb, yb in batches(X[tr], Y[tr], args.batch, True, rng):
            x, m = mo.pack(xb, seq_len)
            y = torch.as_tensor((yb - ym) / ys, dtype=torch.float32, device=device)
            out = model(x_enc=x.to(device), input_mask=m.to(device)).forecast[:, 0, :]
            loss = torch.nn.functional.mse_loss(out, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(xb); n += len(xb)
        Pva = predict(model, X[va], seq_len, device) * ys + ym
        vmae = np.abs(inv(Pva) - inv(Y[va])).mean()
        print(f"epoch {ep+1:2d} train mse {tot/n:.4f} | val MAE {vmae:.3f} | {time.time()-t0:.0f}s", flush=True)
        if vmae < best - 1e-4:
            best, bad = vmae, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop"); break
    model.load_state_dict(best_state)

    leads = list(range(1, H + 1))
    P = predict(model, X[te], seq_len, device) * ys + ym
    res = score_by_lead(Y[te], P, leads, inv).set_index("lead")
    tag = args.tag or f"moment_{model_id.split('-')[-1]}{'_enc' if args.train_encoder else ''}"
    pd.set_option("display.width", 200)
    print(f"\n{tag} — test MAE / CC by lead:")
    print(res.loc[SHOW_LEADS].round(3).to_string())
    print(f"mean over leads: MAE {res.mae.mean():.3f}  CC {res['corr'].mean():.3f}")
    d = default_data_dir()
    res.to_csv(d / f"{tag}_{cfg['index']}.csv")
    np.savez(d / f"{tag}_{cfg['index']}_test_preds.npz",
             dates=dates[te].strftime("%Y-%m-%d").to_numpy().astype("U10"), y=inv(Y[te]), **{tag: inv(P)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
