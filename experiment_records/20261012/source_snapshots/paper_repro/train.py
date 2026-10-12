"""
Unified training entry (paper AAS74510R3 Tables 1-3 reproduction).

Usage:
  python train.py --config configs/ours.json --seed 42 --output-dir results
                  [--resume 1] [--gpu-fraction 0.70]

Uniform protocol (identical for every model, per manuscript claim):
  AdamW lr=1e-4, cosine annealing (warmup 5), max 200 epochs,
  early stop on val F1 patience 15 (min 30), SpecAugment-style masking,
  AMP fp16, gradient clipping 1.0. Threshold is tuned on the VALIDATION set
  (max F1, step 0.01), then applied ONCE to the held-out test set.
  Per-seed confusion matrices are always recorded (Reviewer point #1).
"""
import argparse
import json
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from models import build_model
from data.dataset import FRBDataset
from data.splits import load_split, split_stats, generate_split, save_split, load_index


# ---------------- helpers ----------------

def setup_logger(exp_dir):
    os.makedirs(exp_dir, exist_ok=True)
    logger = logging.getLogger("paper_repro")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(exp_dir, "training.log"), encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def get_lr(epoch, cfg):
    warmup = cfg["warmup_epochs"]
    if epoch < warmup:
        return cfg["lr"] * (epoch + 1) / warmup
    progress = (epoch - warmup) / (cfg["epochs"] - warmup)
    return cfg["lr"] * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def predict(model, loader, device, cfg):
    model.eval()
    all_probs, all_labels = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with autocast(enabled=cfg["use_amp"]):
            logits = model(x)
        probs = F.softmax(logits, dim=1)[:, 1]
        all_probs.extend(probs.cpu().tolist())
        all_labels.extend(y.cpu().tolist())
    return np.array(all_probs), np.array(all_labels)


def find_best_threshold(probs, labels):
    best_f1, best_thr = 0, 0.5
    for thr in np.arange(0.1, 0.9, 0.01):
        preds = (probs >= thr).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr, best_f1


def compute_metrics(probs, labels, threshold):
    preds = (probs >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    acc = (tp + tn) / len(labels)
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = 0.0
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "n_positive": int(labels.sum()),
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4), "accuracy": round(acc, 4),
            "auc": round(auc, 4), "threshold": threshold}


def unfreeze_patch_embed_recursive(model):
    for name, module in model.named_modules():
        if hasattr(module, "unfreeze_patch_embed"):
            try:
                module.unfreeze_patch_embed()
                print(f"[unfreeze] patch embed of {name}")
            except Exception:
                pass


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", default="results")
    ap.add_argument("--resume", type=int, default=1)
    ap.add_argument("--gpu-fraction", type=float, default=0.70)
    ap.add_argument("--epochs", type=int, default=None, help="override config epochs (smoke test)")
    args = ap.parse_args()

    cfg = json.load(open(args.config, encoding="utf-8"))
    cfg["seed"] = args.seed
    if args.epochs:
        cfg["epochs"] = args.epochs
    exp_name = cfg.get("exp_name", cfg["model_type"])   # tune runs may override dir name
    exp_dir = os.path.join(args.output_dir, f"{exp_name}_s{args.seed}")
    logger = setup_logger(exp_dir)
    logger.info(f"paper_repro | model={cfg['model_type']} seed={cfg['seed']}")

    # ---- seeds ----
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])

    # ---- GPU memory cap (leave room for desktop use) ----
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.gpu_fraction)
        props = torch.cuda.get_device_properties(0)
        logger.info(f"GPU {props.name} total={props.total_memory/1e9:.1f}GB "
                    f"cap={args.gpu_fraction:.0%}={(props.total_memory*args.gpu_fraction)/1e9:.1f}GB")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ---- split (canonical file, shared by all models/seeds) ----
    split_path = cfg["split_file"]
    if os.path.exists(split_path):
        split = load_split(split_path)
        logger.info(f"Loaded split {split_path} | {split_stats(split)}")
    else:
        pos_paths, neg_paths = load_index(cfg["data_root"])
        split = generate_split(pos_paths, neg_paths, seed=42)
        save_split(split, split_path)
        logger.info(f"Generated split -> {split_path} | {split_stats(split)}")

    # ---- datasets ----
    train_ds = FRBDataset(split["train_pos"], split["train_neg"], cfg, augment=True)
    val_ds = FRBDataset(split["val_pos"], split["val_neg"], cfg, augment=False)
    test_ds = FRBDataset(split["test_pos"], split["test_neg"], cfg, augment=False)
    bs = cfg["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=0, pin_memory=True)
    logger.info(f"Datasets: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    # ---- model ----
    model = build_model(cfg, cfg["model_type"], pretrained=cfg["pretrained"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model {cfg['model_type']}: total params={n_params:,} trainable={n_train:,}")

    # ---- training ----
    loss_type = cfg.get("loss_type", "ce")
    if loss_type == "focal":
        gamma = float(cfg.get("focal_gamma", 2.0))
        alpha = float(cfg.get("focal_alpha", 0.5))

        class FocalLoss(nn.Module):
            def __init__(self, gamma, alpha):
                super().__init__()
                self.gamma, self.alpha = gamma, alpha

            def forward(self, logits, target):
                logp = F.log_softmax(logits, dim=1)
                logp_t = logp.gather(1, target.view(-1, 1)).squeeze(1)
                p_t = logp_t.exp()
                at = torch.where(target == 1, self.alpha, 1 - self.alpha)
                loss = -at * (1 - p_t) ** self.gamma * logp_t
                return loss.mean()

        criterion = FocalLoss(gamma, alpha)
        logger.info(f"Loss: focal gamma={gamma} alpha={alpha}")
    else:
        pos_w = float(cfg.get("pos_weight", 1.0))
        if pos_w != 1.0:
            w = torch.tensor([1.0, pos_w], dtype=torch.float32)
            criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=cfg["label_smoothing"])
            logger.info(f"Loss: CE pos_weight={pos_w}")
        else:
            criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scaler = GradScaler(enabled=cfg["use_amp"])

    best_val_f1, patience_counter, start_epoch = 0, 0, 0
    history = []
    ckpt_path = os.path.join(exp_dir, "last_epoch_model.pt")
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optimizer_state"])
        scaler.load_state_dict(ck["scaler_state"])
        start_epoch, best_val_f1, patience_counter = ck["epoch"], ck["best_val_f1"], ck["patience_counter"]
        if os.path.exists(os.path.join(exp_dir, "history.json")):
            history = json.load(open(os.path.join(exp_dir, "history.json"), encoding="utf-8"))
        logger.info(f"Resumed epoch {start_epoch} (best_val_f1={best_val_f1:.4f} pat={patience_counter})")

    t_start = time.time()
    for epoch in range(start_epoch, cfg["epochs"]):
        lr = get_lr(epoch, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        if epoch == 1:
            unfreeze_patch_embed_recursive(model)

        model.train()
        total_loss, total_correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=cfg["use_amp"]):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"])
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(y)
            total_correct += (logits.argmax(1) == y).sum().item()
            total += len(y)
        train_loss = total_loss / max(total, 1)
        train_acc = total_correct / max(total, 1)

        # validation
        val_probs, val_labels = predict(model, val_loader, device, cfg)
        thr, val_f1 = find_best_threshold(val_probs, val_labels)
        val_metrics = compute_metrics(val_probs, val_labels, thr)
        logger.info(f"ep {epoch:3d} | lr {lr:.2e} | train_loss {train_loss:.4f} acc {train_acc:.4f} "
                    f"| val F1 {val_f1:.4f} (thr {thr:.2f}) | pat {patience_counter}")

        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                        "val_f1": val_f1, "val_threshold": thr})

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(exp_dir, "best_model.pt"))
        else:
            patience_counter += 1

        torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                    "scaler_state": scaler.state_dict(), "epoch": epoch + 1,
                    "best_val_f1": best_val_f1, "patience_counter": patience_counter},
                   ckpt_path)
        with open(os.path.join(exp_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=1)

        if patience_counter >= cfg["patience"] and epoch >= cfg["min_epochs_before_stop"]:
            logger.info(f"Early stop at epoch {epoch} (patience {cfg['patience']})")
            break

    train_time = time.time() - t_start

    # ---- final test evaluation with VAL-tuned threshold ----
    model.load_state_dict(torch.load(os.path.join(exp_dir, "best_model.pt"),
                                     map_location=device, weights_only=False))
    val_probs, val_labels = predict(model, val_loader, device, cfg)
    thr, _ = find_best_threshold(val_probs, val_labels)
    test_probs, test_labels = predict(model, test_loader, device, cfg)
    test_metrics = compute_metrics(test_probs, test_labels, thr)

    # tensor dims actually used (Reviewer point #6)
    tensor_dims = {"input": f"(B,1,{cfg['n_time']},{cfg['n_freq']})",
                   "after_time_pool": "(B,1,960,1024)",
                   "dedispersed": f"(B,{cfg['n_dm_trials']},1,960,1024)",
                   "after_cnn": "(B,256,60,32)",
                   "tokens": f"(B,60,{cfg['d_model']})"}

    result = {
        "model_type": cfg["model_type"], "seed": cfg["seed"],
        "n_params_total": n_params, "n_params_trainable": n_train,
        "best_val_f1": best_val_f1, "val_threshold": round(thr, 4),
        "test": test_metrics,
        "epochs_trained": len(history), "train_time_s": round(train_time, 1),
        "amp": cfg["use_amp"], "batch_size": bs,
        "gradient_checkpointing": False,
        "tensor_dims": tensor_dims,
        "split": split_stats(split),
        "config": cfg,
    }
    with open(os.path.join(exp_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    logger.info(f"TEST  F1={test_metrics['f1']} recall={test_metrics['recall']} "
                f"precision={test_metrics['precision']} AUC={test_metrics['auc']} "
                f"CM=({test_metrics['tp']},{test_metrics['fn']},{test_metrics['fp']},{test_metrics['tn']})")
    logger.info(f"Saved metrics.json | train_time={train_time:.1f}s")


if __name__ == "__main__":
    main()
