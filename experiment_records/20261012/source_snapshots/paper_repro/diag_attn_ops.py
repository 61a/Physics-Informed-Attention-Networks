"""Round-8: pure-op tests for DMAttentionModule's heavy kernels.
Test 1: AdaptiveAvgPool2d((32,32)) on (256,1,960,1024) repeatedly.
Test 2: broadcast multiply (B,K,1,1,1)*(B,K,1,960,1024) + sum(dim=1).
Usage: diag_attn_ops.py [--steps 60]"""
import faulthandler
import sys
import argparse

faulthandler.enable()

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=60)
ap.add_argument("--which", default="pool")  # pool | mul | both
args = ap.parse_args()

import time
import torch
import torch.nn as nn

print(f"attn-op test: which={args.which} steps={args.steps}", flush=True)
torch.manual_seed(42)
torch.cuda.set_per_process_memory_fraction(0.70)

dev = "cuda"
B, K, C, T, F = 8, 32, 1, 960, 1024

if args.which in ("pool", "both"):
    x = torch.randn(B * K, C, T, F, device=dev)
    pool = nn.AdaptiveAvgPool2d((32, 32)).to(dev)
    t0 = time.time()
    peak = 0.0
    for i in range(args.steps):
        y = pool(x)
        assert y.shape == (B * K, C, 32, 32)
        peak = max(peak, torch.cuda.memory_allocated() / 1e9)
        if i % 10 == 0:
            print(f"   [pool] step {i} peak={peak:.2f}GB t={time.time()-t0:.0f}s", flush=True)
    print(f"   [pool] DONE steps={args.steps} peak={peak:.2f}GB", flush=True)

if args.which in ("mul", "both"):
    dedisp = torch.randn(B, K, C, T, F, device=dev)
    attn = torch.softmax(torch.randn(B, K, device=dev), dim=-1)
    attn_exp = attn.view(B, K, 1, 1, 1)
    t0 = time.time()
    peak = 0.0
    for i in range(args.steps):
        weighted = (dedisp * attn_exp).sum(dim=1)   # (B,1,960,1024)
        peak = max(peak, torch.cuda.memory_allocated() / 1e9)
        if i % 10 == 0:
            print(f"   [mul] step {i} peak={peak:.2f}GB t={time.time()-t0:.0f}s", flush=True)
    print(f"   [mul] DONE steps={args.steps} peak={peak:.2f}GB", flush=True)

print("ALL PASSED", flush=True)
