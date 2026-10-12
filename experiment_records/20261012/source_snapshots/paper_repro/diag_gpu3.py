"""Round-3: faulthandler to catch the segfault location."""
import faulthandler
import sys

faulthandler.enable()
sys.path.insert(0, r"D:\FRB_RFI_Detection_AAS74510\paper_repro")

print("s1: imports", flush=True)
import json
import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from models import build_model
from data.dataset import FRBDataset
from data.splits import load_split

print("s2: seeds", flush=True)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

print("s3: mem cap", flush=True)
torch.cuda.set_per_process_memory_fraction(0.70)

print("s4: config+split+dataset", flush=True)
cfg = json.load(open(r"D:\FRB_RFI_Detection_AAS74510\paper_repro\configs\ours.json", encoding="utf-8"))
split = load_split(cfg["split_file"])
train_ds = FRBDataset(split["train_pos"], split["train_neg"], cfg, augment=True)
loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0,
                    pin_memory=True, drop_last=True)
print("   dataset len:", len(train_ds), flush=True)

print("s5: model build", flush=True)
m = build_model(cfg, "ours", pretrained=False).cuda()
opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
scaler = GradScaler(enabled=True)
criterion = torch.nn.CrossEntropyLoss()
print("   model params:", sum(p.numel() for p in m.parameters()), flush=True)

print("s6: first batch", flush=True)
it = iter(loader)
x, y = next(it)
print("   batch:", x.shape, y.shape, flush=True)
x, y = x.cuda(), y.cuda()

print("s7: AMP fwd", flush=True)
with autocast(enabled=True):
    logits = m(x)
    loss = criterion(logits, y)
print("   loss:", loss.item(), flush=True)

print("s8: AMP backward", flush=True)
scaler.scale(loss).backward()
scaler.unscale_(opt)
torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
scaler.step(opt)
scaler.update()
print("   backward ok", flush=True)

print("s9: full epoch", flush=True)
m.train()
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
    if i % 5 == 0:
        alloc = torch.cuda.memory_allocated() / 1e9
        maxa = torch.cuda.max_memory_allocated() / 1e9
        res = torch.cuda.memory_reserved() / 1e9
        print(f"   step {i} loss={loss.item():.4f} alloc={alloc:.2f}GB max={maxa:.2f}GB reserved={res:.2f}GB", flush=True)
print("ALL PASSED", flush=True)
