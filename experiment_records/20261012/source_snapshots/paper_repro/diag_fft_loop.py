"""Round-7: pure cuFFT loop. Simulates FourierDedispersion's exact tensor shape
and call pattern WITHOUT any model / autograd / DataLoader.
If this crashes too -> root cause is cuFFT accumulation on this GPU/driver stack.
If it survives -> the problem needs autograd or DataLoader interplay.
Usage: diag_fft_loop.py [--steps 90] [--batch 8] [--cpu]"""
import faulthandler
import sys
import argparse

faulthandler.enable()

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=90)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--cpu", action="store_true")
args = ap.parse_args()

import time
import math
import torch

print(f"pure-FFT loop: steps={args.steps} batch={args.batch} cpu={args.cpu}", flush=True)
torch.manual_seed(42)

T, F, K = 960, 1024, 32
dev = "cpu" if args.cpu else "cuda"
if not args.cpu:
    torch.cuda.set_per_process_memory_fraction(0.70)

x = torch.randn(args.batch, 1, T, F, device=dev)
j = torch.arange(T // 2 + 1, device=dev, dtype=torch.float32)
delay = torch.linspace(0, 3000, K, device=dev).view(-1, 1)
freq = torch.linspace(1000, 1500, F, device=dev).view(1, -1)
delay_samples = delay * 4.148808 * (1.0 / freq**2 - 1.0 / 1500.0**2) * 1e-3 / 9.8304e-5

t0 = time.time()
peak = 0.0
for i in range(args.steps):
    X_fft = torch.fft.rfft(x, dim=2)             # (B,1,481,1024) complex
    outs = []
    for k in range(K):
        phase_k = torch.exp(-2j * math.pi * 1j * j.view(-1, 1) * delay_samples[k].view(1, -1) / T)
        X_shifted = X_fft * phase_k.unsqueeze(0).unsqueeze(0)
        outs.append(torch.fft.irfft(X_shifted, n=T, dim=2))
    dedispersed = torch.stack(outs, dim=1)       # (B,32,1,960,1024) ~1GB
    if not args.cpu:
        peak = max(peak, torch.cuda.memory_allocated() / 1e9)
        if i % 10 == 0:
            print(f"   step {i} peak={peak:.2f}GB t={time.time()-t0:.0f}s", flush=True)
    else:
        if i % 5 == 0:
            print(f"   step {i} t={time.time()-t0:.0f}s", flush=True)
    # free explicitly to mimic per-step lifetime
    del dedispersed
    torch.cuda.empty_cache() if not args.cpu else None

print(f"DONE pure-FFT steps={args.steps} peak={peak:.2f}GB total_t={time.time()-t0:.0f}s", flush=True)
