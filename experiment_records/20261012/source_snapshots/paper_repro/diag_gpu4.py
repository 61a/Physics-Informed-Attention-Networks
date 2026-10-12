"""Round-4: isolate crash - AMP on/off, batch size, fp32 fft."""
import faulthandler
import sys
import argparse

faulthandler.enable()
sys.path.insert(0, r"D:\FRB_RFI_Detection_AAS74510\paper_repro")

ap = argparse.ArgumentParser()
ap.add_argument("--amp", type=int, default=1)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--model", default="ours")
args = ap.parse_args()

import json
import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from models import build_model
from data.dataset import FRBDataset
from data.splits import load_split

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.cuda.set_per_process_memory_fraction(0.70)
print(f"cfg: amp={args.amp} batch={args.batch} model={args.model}", flush=True)

cfg = json.load(open(r"D:\FRB_RFI_Detection_AAS74510\paper_repro\configs\ours.json", encoding="utf-8"))
split = load_split(cfg["split_file"])
train_ds = FRBDataset(split["train_pos"], split["train_neg"], cfg, augment=True)
loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0,
                    pin_memory=True, drop_last=True)

m = build_model(cfg, args.model, pretrained=False).cuda()
opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
scaler = GradScaler(enabled=bool(args.amp))
criterion = torch.nn.CrossEntropyLoss()

m.train()
for i, (xb, yb) in enumerate(loader):
    xb, yb = xb.cuda(), yb.cuda()
    opt.zero_grad(set_to_none=True)
    if args.amp:
        with autocast(enabled=True):
            out = m(xb)
            loss = criterion(out, yb)
    else:
        out = m(xb)
        loss = criterion(out, yb)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    if args.amp:
        scaler.step(opt)
        scaler.update()
    else:
        opt.step()
    if i % 5 == 0:
        alloc = torch.cuda.memory_allocated() / 1e9
        print(f"   step {i} loss={loss.item():.4f} alloc={alloc:.2f}GB", flush=True)
print("ALL PASSED", flush=True)
