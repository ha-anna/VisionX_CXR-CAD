"""Обучение одного фолда.

Примеры:
  python train.py --fold 0 --gamma 2 --name d121_g2_f0
  python train.py --fold 0 --gamma 0 --name d121_g0_f0
  python train.py --fold 0 --gamma 2 --epochs 1 --limit 2000 --name smoke   (быстрый тест)
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

import config as C
from dataset import CXRDataset, get_split, load_metadata
from losses import FocalLoss
from metrics import summarize


def build_model(arch: str, pretrained: bool = True) -> nn.Module:
    n = len(C.DISEASES)
    if arch == "densenet121":
        w = torchvision.models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        m = torchvision.models.densenet121(weights=w)
        m.classifier = nn.Linear(m.classifier.in_features, n)
    elif arch == "efficientnet_b0":
        w = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        m = torchvision.models.efficientnet_b0(weights=w)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n)
    elif arch == "efficientnet_b4":
        w = torchvision.models.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        m = torchvision.models.efficientnet_b4(weights=w)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n)
    else:
        raise ValueError(arch)
    return m  # возвращает ЛОГИТЫ; sigmoid применяется снаружи


def compute_pos_weight(train_df, cap: float) -> torch.Tensor:
    pos = train_df[C.DISEASES].sum().values.astype(np.float64)
    neg = len(train_df) - pos
    return torch.tensor(np.minimum(neg / np.maximum(pos, 1), cap), dtype=torch.float32)


def to_dev(x, device):
    x = x.to(device, non_blocking=True)
    return x.contiguous(memory_format=torch.channels_last) if C.CHANNELS_LAST else x


@torch.no_grad()
def predict(model, loader, device, tta: bool = False):
    model.eval()
    probs, ys = [], []
    for x, y in loader:
        x = to_dev(x, device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            p = torch.sigmoid(model(x).float())
            if tta:
                p = 0.5 * (p + torch.sigmoid(model(torch.flip(x, dims=[3])).float()))
        probs.append(p.cpu().numpy()); ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--arch", default="densenet121")
    ap.add_argument("--name", default=None)
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--bs", type=int, default=C.BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=C.LR)
    ap.add_argument("--workers", type=int, default=C.NUM_WORKERS)
    ap.add_argument("--pw-cap", type=float, default=C.POS_WEIGHT_CAP)
    ap.add_argument("--no-clahe", action="store_true")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="для теста: брать N снимков")
    a = ap.parse_args()

    torch.manual_seed(C.SEED); np.random.seed(C.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = a.name or f"{a.arch}_g{a.gamma:g}_f{a.fold}"
    out = C.RUNS_DIR / name
    out.mkdir(parents=True, exist_ok=True)

    df = load_metadata()
    tr_df = get_split(df, a.fold, "train")
    va_df = get_split(df, a.fold, "val")
    if a.limit:
        tr_df, va_df = tr_df.sample(a.limit, random_state=0), va_df.sample(max(a.limit // 4, 200), random_state=0)
    print(f"[{name}] train={len(tr_df):,} val={len(va_df):,} device={device}")

    use_clahe = not a.no_clahe
    mk = lambda d, train: DataLoader(
        CXRDataset(d, train=train, use_clahe=use_clahe), batch_size=a.bs if train else a.bs * 2,
        shuffle=train, num_workers=a.workers, pin_memory=device.type == "cuda",
        persistent_workers=a.workers > 0, drop_last=train)
    tr_loader, va_loader = mk(tr_df, True), mk(va_df, False)

    model = build_model(a.arch, pretrained=not a.no_pretrained).to(device)
    if C.CHANNELS_LAST:
        model = model.to(memory_format=torch.channels_last)
    pos_weight = compute_pos_weight(tr_df, a.pw_cap).to(device)
    criterion = FocalLoss(gamma=a.gamma, pos_weight=pos_weight).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=C.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best, bad, history = -1.0, 0, []
    for ep in range(1, a.epochs + 1):
        model.train(); t0 = time.time(); run = 0.0
        for i, (x, y) in enumerate(tr_loader, 1):
            x = to_dev(x, device)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(x)
            loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            run += loss.item()
            if i % 200 == 0:
                print(f"  ep{ep} it{i}/{len(tr_loader)} loss={run / i:.4f}", flush=True)
        sched.step()

        yt, yp = predict(model, va_loader, device)
        s = summarize(yt, yp)
        rec = {"epoch": ep, "train_loss": run / len(tr_loader), "val_auroc": s["mean_auroc"],
               "val_auprc": s["mean_auprc"], "val_ece": s["ece"], "sec": round(time.time() - t0)}
        history.append(rec)
        print(f"[{name}] ep{ep}: loss={rec['train_loss']:.4f} AUROC={rec['val_auroc']:.4f} "
              f"AUPRC={rec['val_auprc']:.4f} ECE={rec['val_ece']:.4f} ({rec['sec']}s)", flush=True)

        json.dump({"args": vars(a), "history": history}, open(out / "history.json", "w"), indent=2)
        if s["mean_auroc"] > best:
            best, bad = s["mean_auroc"], 0
            torch.save(model.state_dict(), out / "best.pth")
            np.save(out / "val_probs.npy", yp); np.save(out / "val_labels.npy", yt)
            np.save(out / "val_rows.npy", va_df["row"].values)
            json.dump(s, open(out / "best_metrics.json", "w"), indent=2)
        else:
            bad += 1
            if bad >= C.PATIENCE:
                print(f"[{name}] early stopping (patience={C.PATIENCE})"); break

    print(f"[{name}] ГОТОВО. Лучший val AUROC = {best:.4f}  -> {out}")


if __name__ == "__main__":
    main()
