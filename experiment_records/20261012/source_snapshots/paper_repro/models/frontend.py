"""
Physics-informed front-end: Fourier-domain dedispersion + DM attention.
Reused verbatim from train_paper_fidelity.py (paper AAS74510R3).

FourierDedispersion: for each of K DM trials, applies a phase rotation in the
Fourier domain along the time axis, i.e. x_dedis[f] = IFFT[FFT(x[f]) * exp(-2pi i j delay[k,f] / T)].
DMAttentionModule: softmax attention over the K dedispersed trials.
"""
import math
import torch
import torch.nn as nn


class FourierDedispersion(nn.Module):
    def __init__(self, n_freq, n_time, n_dm=32, dm_range=(0, 3000),
                 f_min=1000.0, f_max=1500.0, dt=9.8304e-5, k_DM=4.148808):
        super().__init__()
        self.n_dm = n_dm
        self.n_time = n_time
        self.n_freq = n_freq

        # Learnable DM anchors (initialized uniform, can fine-tune)
        dm_anchors = torch.linspace(dm_range[0], dm_range[1], n_dm)
        self.dm_anchors = nn.Parameter(dm_anchors)

        # Frequency grid (MHz)
        freqs = torch.linspace(f_min, f_max, n_freq)
        self.register_buffer("freqs", freqs)
        self.f_ref = f_max
        self.dt = dt
        self.k_DM = k_DM  # ms MHz^2 cm^3 pc^-1

    def forward(self, x):
        """x: (B, C, T, F) -> dedispersed: (B, K, C, T, F), delays: (K, F)"""
        B, C, T, F_dim = x.shape

        # (K, F) delays in samples: delay_ms = DM * k_DM * (f^-2 - f_ref^-2)
        dm_grid = self.dm_anchors.view(-1, 1)          # (K, 1)
        freq_grid = self.freqs.view(1, -1)             # (1, F)
        delay_ms = dm_grid * self.k_DM * (1.0 / freq_grid**2 - 1.0 / self.f_ref**2)
        delay_samples = delay_ms * 1e-3 / self.dt      # (K, F)

        # FFT along time axis
        X_fft = torch.fft.rfft(x, dim=2)               # (B, C, T//2+1, F)

        j = torch.arange(T // 2 + 1, device=x.device, dtype=torch.float32)

        output_list = []
        for k in range(self.n_dm):
            phase_k = torch.exp(-2j * math.pi * 1j * j.view(-1, 1) *
                                delay_samples[k].view(1, -1) / T)
            X_shifted = X_fft * phase_k.unsqueeze(0).unsqueeze(0)
            x_dedis = torch.fft.irfft(X_shifted, n=T, dim=2)
            output_list.append(x_dedis)

        dedispersed = torch.stack(output_list, dim=1)  # (B, K, C, T, F)
        return dedispersed, delay_samples


class DMAttentionModule(nn.Module):
    def __init__(self, n_dm, n_time, n_freq, dropout=0.1):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((32, 32))
        self.feature_extractor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32, 128), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(128, 32), nn.ReLU(inplace=True),
        )
        self.attention_net = nn.Sequential(
            nn.Linear(32 * n_dm, 256), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(256, 64), nn.ReLU(inplace=True),
            nn.Linear(64, n_dm), nn.Softmax(dim=-1)
        )

    def forward(self, dedispersed):
        """(B, K, C, T, F) -> weighted: (B, C, T, F), attn: (B, K)"""
        B, K, C, T, F = dedispersed.shape
        x_flat = dedispersed.view(B * K, C, T, F)
        x_pooled = self.pool(x_flat)
        features = self.feature_extractor(x_pooled).view(B, K, 32)
        features_flat = features.flatten(1)            # (B, K*32)
        attn = self.attention_net(features_flat)       # (B, K)
        attn_exp = attn.view(B, K, 1, 1, 1)
        weighted = (dedispersed * attn_exp).sum(dim=1)  # (B, C, T, F)
        return weighted, attn


class PhysicsFrontEnd(nn.Module):
    """Shared physics-informed front-end used for Table-2 baselines.

    (B,1,15360,1024) -> AvgPool(16,1) -> (B,1,960,1024)
    -> FourierDedispersion(32 DM) -> (B,32,1,960,1024)
    -> DMAttention -> (B,1,960,1024) -> AvgPool(16,1) -> (B,1,60,1024)
    """

    def __init__(self, cfg):
        super().__init__()
        self.time_pool = nn.AvgPool2d(kernel_size=(cfg["time_downsample"], 1))
        t_ds = cfg["n_time"] // cfg["time_downsample"]
        self.dedisp = FourierDedispersion(
            n_freq=cfg["n_freq"], n_time=t_ds,
            n_dm=cfg["n_dm_trials"], dm_range=cfg["dm_range"],
            f_min=cfg["f_min"], f_max=cfg["f_max"],
            dt=cfg["dt"] * cfg["time_downsample"], k_DM=cfg["k_DM"])
        self.dm_attn = DMAttentionModule(
            n_dm=cfg["n_dm_trials"], n_time=t_ds, n_freq=cfg["n_freq"],
            dropout=cfg["dropout"])
        # 960 -> 60 (keeps frequency resolution, aligns with raw mode input)
        self.downsample = nn.AvgPool2d(kernel_size=(16, 1))

    def forward(self, x):
        x = self.time_pool(x)                # (B,1,960,1024)
        dedisp, _ = self.dedisp(x)           # (B,32,1,960,1024)
        weighted, attn = self.dm_attn(dedisp)
        out = self.downsample(weighted)      # (B,1,60,1024)
        return out, attn
