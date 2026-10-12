"""Round-5: forward-only vs backward, GPU vs CPU, data loading."""
import faulthandler
import sys
import argparse

faulthandler.enable()
sys.path.insert(0, r"D:\FRB_RFI_Detection_AAS74510\paper_repro")

ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="fwd")     # fwd | bwd | cpu
ap.add_argument("--steps", type=int, default=60)
args = ap.parse_args()

import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from models import build_model
from data.dataset import FRBDataset
from data.splits import load_split

print(f"mode={args.mode} steps={args.steps}", flush=True)
torch.manual_seed(42)

cfg = json.load(open(r"D:\FRB_RFI_Detection_AAS74510\paper_repro\configs\ours.json", encoding="utf-8"))
split = load_split(cfg["split_file"])
train_ds = FRBDataset(split["train_pos"], split["train_neg"], cfg, augment=True)
loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0,
                    pin_memory=True, drop_last=True)

use_cuda = args.mode != "cpu"
if use_cuda:
    torch.cuda.set_per_process_memory_fraction(0.70)
    m = build_model(cfg, "ours", pretrained=False).cuda()
else:
    m = build_model(cfg, "ours", pretrained=False)

opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

for i, (xb, yb) in enumerate(loader):
    if i >= args.steps:
        break
    if use_cuda:
        xb, yb = xb.cuda(), yb.cuda()
    if args.mode == "fwd":
        m.eval()
        with torch.no_grad():
            out = m(xb)
        loss = criterion(out, yb)
    else:
        m.train()
        opt.zero_grad(set_to_none=True)
        out = m(xb)
        loss = criterion(out, yb)
        loss.backward()
        opt.step()
    if i % 10 == 0:
        alloc = torch.cuda.memory_allocated() / 1e9 if use_cuda else 0
        print(f"   step {i} loss={loss.item():.4f} alloc={alloc:.2f}GB", flush=True)
print("ALL PASSED", flush=True)
