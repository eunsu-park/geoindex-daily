"""LoRA fine-tuning of Surya's backbone for daily Ap, 60 days ahead (lora-plan §3-§5).

Inputs per sample: the cached front-end token map of the issue day (65,536 × 1,280, bf16,
from cache_surya_tokens.py) and the 30-day log1p-Ap window (TS) or TS + log F10.7 + SN
(PHASE). The backbone (blocks 0-1 spectral gating, 2-9 windowed attention) runs in bf16
with activation checkpointing; LoRA adapters (rank r) sit on attn.qkv / attn.proj of the
attention blocks. The backbone output is average-pooled to an 8×8 grid, projected to 128-d
per cell and read out by one learned attention query; the image vector is concatenated
with the time-series block and passed through the same MLP head as fusion_mlp.py.

Modes:
  lora     adapters + pooling + head trainable (the experiment)
  frozen   no adapters; backbone under no_grad; pooling + head trainable (control b)
  ts       no image branch; head only (control a; matched time-series model)

    python scripts/lora_train.py --mode lora --ts-block ts --seed 0
    python scripts/lora_train.py --mode lora --smoke 8            # memory / speed check, then exit
    python scripts/lora_train.py --mode frozen --ts-block phase --seed 0
"""
import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_surya_embeddings import DEFAULT_CKPT, DEFAULT_REPO  # noqa: E402
from fusion import GROUPS, SHOW_LEADS, TRANSFORMS, per_lead  # noqa: E402
from geoindex_daily.daily_index import default_data_dir, load_daily  # noqa: E402
from geoindex_daily.encoders import surya as su  # noqa: E402
from geoindex_daily.windows import make_windows, split  # noqa: E402

SIDE = 256  # 4096 / 16 patches per side


# --- data -------------------------------------------------------------------------------

class TokenDataset(Dataset):
    def __init__(self, files, ts, y, load_tokens=True):
        self.files, self.ts, self.y, self.load_tokens = files, ts, y, load_tokens

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        tok = torch.load(self.files[i], map_location="cpu")["tokens"] if self.load_tokens else torch.zeros(0)
        return tok, torch.as_tensor(self.ts[i]), torch.as_tensor(self.y[i])


def build_samples(cfg, token_dir: Path, ts_block: str):
    fwd, inv = TRANSFORMS[cfg.get("ts_transform", "log1p")]
    L, H = cfg["input_days"], cfg["output_days"]
    d = load_daily()
    X, Y, dates = make_windows(fwd(d[cfg["index"]]), L, H)
    have = {pd.Timestamp(f.stem[:8]): f for f in token_dir.glob("2*.pt")}
    keep = np.array([t in have for t in dates])
    if ts_block == "phase":
        phase = np.stack([np.log(d["f107"].reindex(dates).to_numpy()), d["sn"].reindex(dates).to_numpy()], axis=1)
        keep &= np.isfinite(phase).all(axis=1)
        X = np.concatenate([X, phase], axis=1)
    X, Y, dates = X[keep], Y[keep], dates[keep]
    files = [have[t] for t in dates]
    masks = split(dates, cfg["splits"])
    return X, Y, dates, files, masks, inv


# --- model ------------------------------------------------------------------------------

class GridAttnPool(nn.Module):
    """(B, N, D) tokens → 8×8 grid → per-cell projection → one learned query → (B, d_out)."""

    def __init__(self, d_in=1280, d_out=128, grid=8):
        super().__init__()
        self.grid = grid
        self.proj = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d_out), nn.GELU())
        self.query = nn.Parameter(torch.randn(d_out) * 0.02)
        self.key = nn.Linear(d_out, d_out)

    def forward(self, tokens):
        B, N, D = tokens.shape
        x = tokens.float().view(B, SIDE, SIDE, D).permute(0, 3, 1, 2)          # (B, D, 256, 256)
        x = torch.nn.functional.adaptive_avg_pool2d(x, self.grid)               # (B, D, g, g)
        x = x.flatten(2).transpose(1, 2)                                         # (B, g*g, D)
        z = self.proj(x)                                                         # (B, g*g, d_out)
        att = torch.softmax(self.key(z) @ self.query / math.sqrt(z.shape[-1]), dim=1)  # (B, g*g)
        return (att.unsqueeze(-1) * z).sum(1)                                    # (B, d_out)


class Head(nn.Module):
    def __init__(self, n_in, horizon, hidden=256, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, horizon))

    def forward(self, x):
        return self.mlp(x)


def run_backbone(backbone, tokens, freeze_through: int, train: bool):
    """Blocks 0..freeze_through under no_grad (no activations kept), the rest with grad and
    activation checkpointing. Block indices follow Surya's chain: 0-1 spectral, 2-9 attention."""
    from itertools import chain
    from torch.utils.checkpoint import checkpoint
    blocks = list(chain(backbone.blocks_spectral_gating, backbone.blocks_attention))
    x = tokens
    with torch.no_grad():
        for blk in blocks[: freeze_through + 1]:
            x = blk(x, None)
    for blk in blocks[freeze_through + 1:]:
        x = checkpoint(blk, x, None, use_reentrant=False) if train else blk(x, None)
    return x


class SuryaApModel(nn.Module):
    def __init__(self, backbone, mode, n_ts, horizon, d_img=128, grid=8, freeze_through=1):
        super().__init__()
        self.mode, self.freeze_through = mode, freeze_through
        self.backbone = backbone if mode != "ts" else None
        self.pool = GridAttnPool(1280, d_img, grid) if mode != "ts" else None
        self.head = Head(n_ts + (d_img if mode != "ts" else 0), horizon)

    def forward(self, tokens, ts):
        parts = [ts.float()]
        if self.mode != "ts":
            if self.mode == "frozen":
                with torch.no_grad():
                    out = run_backbone(self.backbone, tokens, 9, False)
            else:
                out = run_backbone(self.backbone, tokens, self.freeze_through, self.training)
            parts.append(self.pool(out))
        return self.head(torch.cat(parts, dim=1))


def add_lora(backbone, rank, alpha, dropout, targets="qkv,proj", first_block=2):
    """LoRA on attention blocks with chain index >= first_block (attention block i = chain index i+2)."""
    from peft import LoraConfig, inject_adapter_in_model
    idx = "|".join(str(i) for i in range(max(0, first_block - 2), 8))
    pattern = r".*blocks_attention\.(" + idx + r")\.(attn\.(qkv|proj)" + (r"|mlp\.(fc1|fc2)" if "fc" in targets else "") + ")"
    cfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, target_modules=pattern, bias="none")
    return inject_adapter_in_model(cfg, backbone)


# --- training ---------------------------------------------------------------------------

def predict(model, loader, device):
    model.eval()
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for tok, ts, _ in loader:
            out.append(model(tok.to(device, non_blocking=True), ts.to(device)).float().cpu().numpy())
    return np.concatenate(out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/ap.yaml")
    p.add_argument("--mode", choices=["lora", "frozen", "ts"], default="lora")
    p.add_argument("--ts-block", choices=["ts", "phase"], default="ts")
    p.add_argument("--tokens", default=None, help="token cache dir (default data dir/surya/tokens/13ch)")
    p.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    p.add_argument("--surya-repo", default=str(DEFAULT_REPO))
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--targets", default="qkv,proj", help="'qkv,proj' or 'qkv,proj,fc'")
    p.add_argument("--freeze-through", type=int, default=1,
                   help="run chain blocks 0..K under no_grad; LoRA and gradients only beyond K (1 = spectral blocks frozen, 5 = fallback)")
    p.add_argument("--lr-lora", type=float, default=1e-4)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--accum", type=int, default=8)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", type=int, default=0, help="run N training steps, report memory/speed, exit")
    p.add_argument("--tag", default=None)
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    token_dir = Path(args.tokens) if args.tokens else default_data_dir() / "surya" / "tokens" / "13ch"
    X, Y, dates, files, masks, inv = build_samples(cfg, token_dir, args.ts_block)
    tr, va, te = masks["train"], masks["val"], masks["test"]
    print(f"samples with tokens: {len(dates)} (train {tr.sum()}, val {va.sum()}, test {te.sum()}); ts block {args.ts_block} ({X.shape[1]} values)")

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda"
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
    Xz = ((X - mu) / sd).astype(np.float32)
    ym, ys = Y[tr].mean(), Y[tr].std()
    Yz = ((Y - ym) / ys).astype(np.float32)
    H = Y.shape[1]

    backbone = None
    if args.mode != "ts":
        model_s, scfg, _, _ = su.load(args.ckpt, args.surya_repo, device, torch.bfloat16)
        backbone = model_s.backbone
        for p_ in backbone.parameters():
            p_.requires_grad_(False)
        if args.mode == "lora":
            backbone = add_lora(backbone, args.rank, args.alpha, args.lora_dropout, args.targets, args.freeze_through + 1)
    model = SuryaApModel(backbone, args.mode, Xz.shape[1], H, freeze_through=args.freeze_through).to(device)
    lora_params = [p_ for n, p_ in model.named_parameters() if "lora_" in n]
    other_params = [p_ for n, p_ in model.named_parameters() if p_.requires_grad and "lora_" not in n]
    print(f"trainable: lora {sum(p_.numel() for p_ in lora_params)/1e6:.2f}M, pooling+head {sum(p_.numel() for p_ in other_params)/1e6:.2f}M")

    groups = [{"params": other_params, "lr": args.lr_head}]
    if lora_params:
        groups.append({"params": lora_params, "lr": args.lr_lora})
    opt = torch.optim.AdamW(groups, weight_decay=1e-2)

    load_tok = args.mode != "ts"
    mk = lambda m, shuffle: DataLoader(TokenDataset([f for f, k in zip(files, m) if k], Xz[m], Yz[m], load_tok),  # noqa: E731
                                        batch_size=args.batch, shuffle=shuffle, num_workers=args.workers if load_tok else 0,
                                        pin_memory=True, persistent_workers=load_tok and args.workers > 0)
    dl_tr, dl_va, dl_te = mk(tr, True), mk(va, False), mk(te, False)
    steps_per_epoch = math.ceil(len(dl_tr) / args.accum)
    total_steps = steps_per_epoch * args.epochs
    warm = steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, max(0, s - warm) / max(1, total_steps - warm)))))

    tag = args.tag or f"lora_{args.mode}_{args.ts_block}_r{args.rank}_f{args.freeze_through}_s{args.seed}"
    out_dir = default_data_dir() / "lora"
    out_dir.mkdir(parents=True, exist_ok=True)

    best, best_state, bad, step = np.inf, None, 0, 0
    for ep in range(args.epochs):
        model.train()
        t0, tot, n, nb = time.time(), 0.0, 0, 0
        opt.zero_grad()
        for tok, ts, y in dl_tr:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(tok.to(device, non_blocking=True), ts.to(device))
            loss = nn.functional.mse_loss(out.float(), y.to(device)) / args.accum
            loss.backward()
            nb += 1
            if nb % args.accum == 0:
                torch.nn.utils.clip_grad_norm_([p_ for g in groups for p_ in g["params"]], 1.0)
                opt.step(); sched.step(); opt.zero_grad(); step += 1
            tot += loss.item() * args.accum * len(y); n += len(y)
            if args.smoke and nb >= args.smoke:
                torch.cuda.synchronize()
                print(f"SMOKE: {nb} samples in {time.time()-t0:.1f}s = {(time.time()-t0)/nb:.2f} s/sample (batch {args.batch}); "
                      f"GPU peak {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
                return 0
        Pva = predict(model, dl_va, device) * ys + ym
        vmae = float(np.abs(inv(Pva) - inv(Y[va])).mean())
        print(f"epoch {ep+1:2d} train mse {tot/max(n,1):.4f} | val MAE {vmae:.3f} | {time.time()-t0:.0f}s", flush=True)
        if vmae < best - 1e-4:
            best, bad = vmae, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if v.requires_grad or "lora_" in k or not k.startswith("backbone")}
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop"); break
    model.load_state_dict(best_state, strict=False)

    leads = list(range(1, H + 1))
    P = predict(model, dl_te, device) * ys + ym
    res = per_lead(Y[te], P, inv)
    summ = {f"{k}_{g}": res.loc[lo:hi, "mae" if k == "mae" else "corr"].mean() for g, (lo, hi) in GROUPS.items() for k in ("mae", "cc")}
    pd.set_option("display.width", 200)
    print(f"\n{tag} — test MAE / CC by lead:\n{res.loc[SHOW_LEADS].round(3).to_string()}")
    print("means:", {k: round(v, 3) for k, v in summ.items()})
    res.to_csv(out_dir / f"{tag}_by_lead.csv")
    with open(out_dir / f"{tag}_summary.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(list(summ)); w.writerow([f"{v:.4f}" for v in summ.values()])
    np.savez(out_dir / f"{tag}_test_preds.npz", dates=dates[te].strftime("%Y-%m-%d").to_numpy().astype("U10"),
             y=inv(Y[te]), **{tag: inv(P)})
    torch.save({k: v.cpu() for k, v in best_state.items()}, out_dir / f"{tag}_weights.pt")
    print(f"artifacts → {out_dir}/{tag}_*")
    return 0


if __name__ == "__main__":
    sys.exit(main())
