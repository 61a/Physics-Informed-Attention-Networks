"""Round-2 diagnostics: AMP + real data loader (what train.py does differently)."""
import json
import os
import sys

import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

sys.path.insert(0, r"D:\FRB_RFI_Detection_AAS74510\paper_repro")
from models import build_model
from data.dataset import FRBDataset
from data.splits import load_split

print("== AMP + dataloader diagnostics ==")
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# GPU cap like train.py
torch.cuda.set_per_process_memory_fraction(0.70)
print("mem cap set")

cfg = json.load(open(r"D:\FRB_RFI_Detection_AAS74510\paper_repro\configs\ours.json", encoding="utf-8"))
split = load_split(cfg["split_file"])
train_ds = FRBDataset(split["train_pos"], split["train_neg"], cfg, augment=True)
loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0,
                    pin_memory=True, drop_last=True)
print("dataset len:", len(train_ds))

m = build_model(cfg, "ours", pretrained=False).cuda()
opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
scaler = GradScaler(enabled=cfg["use_amp"])
criterion = torch.nn.CrossEntropyLoss()

it = iter(loader)
x, y = next(it)
print("batch x:", tuple(x.shape), "y:", tuple(y.shape))
x, y = x.cuda(), y.cuda()

with autocast(enabled=True):
    logits = m(x)
    loss = criterion(logits, y)
print("AMP fwd OK, loss:", loss.item(), "logits dtype:", logits.dtype)

scaler.scale(loss).backward()
scaler.unscale_(opt)
torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
scaler.step(opt)
scaler.update()
print("AMP backward+step OK")

# one full epoch loop
m.train()
total = 0
for i, (xb, yb) in enumerate(loader):
    xb, yb = xb.cuda(), yb.cuda()
    opt.zero_grad(set_to_none=True)
    with autocast(enabled=True):
        out = m(xb)
        loss = criterion(out, yb)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()
    total += 1
    if i % 20 == 0:
        print(f"  step {i} loss={loss.item():.4f}")
print(f"full epoch done: {total} steps, peak mem = "
      f"{torch.cuda.max_memory_allocated()/1e9:.2f} GB")
print("ALL ROUND-2 PASSED")
