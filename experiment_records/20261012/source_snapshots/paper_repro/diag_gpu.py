"""Minimal GPU diagnostics to isolate the segfault."""
import sys
import torch

print("py", sys.version.split()[0], "torch", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit(0)
print("device:", torch.cuda.get_device_name(0))

# 1. basic tensor op on GPU
x = torch.randn(4, 1, 64, 1024, device="cuda")
y = x * 2
print("1 basic op OK", tuple(y.shape))

# 2. conv2d on GPU
conv = torch.nn.Conv2d(1, 32, 3, stride=2, padding=1).cuda()
y = conv(x)
print("2 conv2d OK", tuple(y.shape))

# 3. rfft on GPU
xf = torch.randn(4, 1, 960, 1024, device="cuda")
X = torch.fft.rfft(xf, dim=2)
print("3 rfft OK", tuple(X.shape), X.dtype)

# 4. complex phase multiply + irfft
j = torch.arange(481, device="cuda", dtype=torch.float32)
phase = torch.exp(-2j * 3.14159 * 1j * j.view(-1, 1) / 960)
print("4 phase dtype:", phase.dtype)
Xs = X * phase.unsqueeze(0).unsqueeze(0)
xd = torch.fft.irfft(Xs, n=960, dim=2)
print("4 complex mul + irfft OK", tuple(xd.shape), xd.dtype)

# 5. model forward on GPU (Ours)
import json
import sys
sys.path.insert(0, r"D:\FRB_RFI_Detection_AAS74510\paper_repro")
from models import build_model
cfg = json.load(open(r"D:\FRB_RFI_Detection_AAS74510\paper_repro\configs\ours.json", encoding="utf-8"))
m = build_model(cfg, "ours", pretrained=False).cuda().eval()
with torch.no_grad():
    xin = torch.randn(2, 1, 15360, 1024, device="cuda")
    out = m(xin)
print("5 ours fwd OK", tuple(out.shape))

# 6. backward pass
xin = torch.randn(2, 1, 15360, 1024, device="cuda")
yt = torch.randint(0, 2, (2,), device="cuda")
m.train()
opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
loss = torch.nn.functional.cross_entropy(m(xin), yt)
loss.backward()
opt.step()
print("6 backward OK", loss.item())
print("ALL DIAGNOSTICS PASSED")
