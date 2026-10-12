"""Round-6: model-component bisection. Same DataLoader loop, different model.
--model ours | variant_a(no_physics) | variant_b(no_dm_attn) | raw_resnet18
Also monitors system RAM to rule out host-memory exhaustion."""
import faulthandler
import sys
import argparse

faulthandler.enable()
sys.path.insert(0, r"D:\FRB_RFI_Detection_AAS74510\paper_repro")

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="variant_a")  # ours | variant_a | variant_b | raw_resnet18
ap.add_argument("--steps", type=int, default=90)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--cpu", action="store_true")
args = ap.parse_args()

import json
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from models import build_model
from data.dataset import FRBDataset
from data.splits import load_split

try:
    import psutil
    HAVE_PSUTIL = True
except ImportError:
    HAVE_PSUTIL = False

print(f"model={args.model} steps={args.steps} batch={args.batch} cpu={args.cpu} PID={os.getpid()}", flush=True)
torch.manual_seed(42)

cfg = json.load(open(r"D:\FRB_RFI_Detection_AAS74510\paper_repro\configs\ours.json", encoding="utf-8"))
split = load_split(cfg["split_file"])
train_ds = FRBDataset(split["train_pos"], split["train_neg"], cfg, augment=True)
loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0,
                    pin_memory=True, drop_last=True)

use_cuda = not args.cpu
if use_cuda:
    torch.cuda.set_per_process_memory_fraction(0.70)
    m = build_model(cfg, args.model, pretrained=False).cuda()
else:
    m = build_model(cfg, args.model, pretrained=False)

n_params = sum(p.numel() for p in m.parameters())
print(f"model={args.model} params={n_params/1e6:.2f}M", flush=True)

opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()
t0 = time.time()
peak_alloc = 0.0

for i, (xb, yb) in enumerate(loader):
    if i >= args.steps:
        break
    if use_cuda:
        xb, yb = xb.cuda(), yb.cuda()
    m.train()
    opt.zero_grad(set_to_none=True)
    out = m(xb)
    loss = criterion(out, yb)
    loss.backward()
    opt.step()
    if use_cuda:
        alloc = torch.cuda.memory_allocated() / 1e9
        peak_alloc = max(peak_alloc, alloc)
    else:
        alloc = 0
    if HAVE_PSUTIL:
        rss = psutil.Process().memory_info().rss / 1e9
        sys_mem = psutil.virtual_memory()
        sys_str = f" rss={rss:.2f}GB sys_used={sys_mem.percent:.0f}%"
    else:
        sys_str = ""
    if i % 10 == 0:
        print(f"   step {i} loss={loss.item():.4f} gpu_alloc={alloc:.2f}GB peak={peak_alloc:.2f}GB{sys_str} "
              f"t={time.time()-t0:.0f}s", flush=True)

print(f"DONE model={args.model} steps={args.steps} peak_gpu={peak_alloc:.2f}GB total_t={time.time()-t0:.0f}s", flush=True)
