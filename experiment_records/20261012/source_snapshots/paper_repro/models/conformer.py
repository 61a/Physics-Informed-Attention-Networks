"""
Conformer components and the full proposed model (Ours).
Reused verbatim from train_paper_fidelity.py (paper AAS74510R3).

Ours pipeline:
  (B,1,15360,1024) -> AvgPool2d(16,1) -> (B,1,960,1024)
  -> FourierDedispersion(32 DM) -> (B,32,1,960,1024)
  -> DMAttention -> (B,1,960,1024)
  -> 4-layer CNN (t-stride 2,2,2,2 / f-stride 2,2,2,4) -> (B,256,60,32)
  -> proj -> 60 tokens x d_model=256 -> +pos -> 6x Conformer(kernel=31)
  -> GAP -> concat(DM attn) -> 2-class head
"""
import math
import torch
import torch.nn as nn

from .frontend import FourierDedispersion, DMAttentionModule


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Transpose(nn.Module):
    def __init__(self, d1, d2):
        super().__init__()
        self.d1, self.d2 = d1, d2

    def forward(self, x):
        return x.transpose(self.d1, self.d2)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), Swish(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class MHSA(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, self.heads, c // self.heads).permute(0, 2, 1, 3), qkv)
        attn = (q @ k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.to_out(out)


class ConvModule(nn.Module):
    def __init__(self, dim, kernel=31, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), Transpose(1, 2),
            nn.Conv1d(dim, dim * 2, 1), nn.GLU(dim=1),
            nn.Conv1d(dim, dim, kernel, padding=(kernel - 1) // 2, groups=dim),
            nn.BatchNorm1d(dim), Swish(),
            nn.Conv1d(dim, dim, 1), nn.Dropout(dropout),
            Transpose(1, 2))

    def forward(self, x):
        return self.net(x)


class ConformerBlock(nn.Module):
    def __init__(self, dim, heads=4, ffn=1024, kernel=31, dropout=0.1):
        super().__init__()
        self.ff1 = FeedForward(dim, ffn, dropout)
        self.attn = MHSA(dim, heads, dropout)
        self.conv = ConvModule(dim, kernel, dropout)
        self.ff2 = FeedForward(dim, ffn, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + 0.5 * self.ff1(self.norm1(x))
        x = x + self.attn(self.norm2(x))
        x = x + self.conv(self.norm3(x))
        x = x + 0.5 * self.ff2(self.norm4(x))
        return self.final_norm(x)


class PaperFRBDetector(nn.Module):
    """Full proposed model: physics front-end + CNN + Conformer (Ours).

    variant:
      "full"       - time pool + dedisp + DM attention + CNN + Conformer (Table 2 Ours)
      "no_physics" - ablation A: drop dedisp + DM attention (raw 960x1024 -> CNN)
      "no_dm_attn" - ablation B: dedisp kept, GAP (uniform mean over K) instead of attention
    """

    def __init__(self, cfg, variant="full"):
        super().__init__()
        self.cfg = cfg
        self.variant = variant

        # Stage 0: time downsampling 15360 -> 960
        self.time_pool = nn.AvgPool2d(kernel_size=(cfg["time_downsample"], 1))
        t_ds = cfg["n_time"] // cfg["time_downsample"]  # 960

        # Stage 1: Fourier-domain dedispersion (skipped for no_physics)
        if variant in ("full", "no_dm_attn"):
            self.dedisp = FourierDedispersion(
                n_freq=cfg["n_freq"], n_time=t_ds,
                n_dm=cfg["n_dm_trials"], dm_range=cfg["dm_range"],
                f_min=cfg["f_min"], f_max=cfg["f_max"],
                dt=cfg["dt"] * cfg["time_downsample"], k_DM=cfg["k_DM"])

        # Stage 2: DM attention (full only; no_dm_attn uses uniform mean)
        if variant == "full":
            self.dm_attn = DMAttentionModule(
                n_dm=cfg["n_dm_trials"], n_time=t_ds, n_freq=cfg["n_freq"],
                dropout=cfg["dropout"])

        # Stage 3: CNN front-end (time 960->60, freq 1024->32)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=(2, 2), padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=(2, 2), padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=(2, 2), padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=(2, 4), padding=1), nn.BatchNorm2d(256), nn.ReLU(),
        )

        # Stage 4: projection to d_model (256*32 -> 256)
        self.proj = nn.Linear(256 * 32, cfg["d_model"])

        # Positional encoding
        max_len = 256
        pe = torch.zeros(max_len, cfg["d_model"])
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, cfg["d_model"], 2).float() *
                             (-math.log(10000.0) / cfg["d_model"]))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

        # Stage 5: Conformer blocks
        self.conformers = nn.ModuleList([
            ConformerBlock(dim=cfg["d_model"], heads=cfg["n_heads"],
                           ffn=cfg["ffn_dim"], kernel=cfg["conv_kernel"],
                           dropout=cfg["dropout"])
            for _ in range(cfg["n_conformer_blocks"])
        ])

        # Stage 6: classification head (concats DM attention weights in full variant)
        self.gap = nn.AdaptiveAvgPool1d(1)
        head_in = cfg["d_model"] + (cfg["n_dm_trials"] if variant == "full" else 0)
        self.classifier = nn.Sequential(
            nn.Linear(head_in, 128),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 2))

    def forward(self, x, return_attn=False):
        x = self.time_pool(x)                                # (B,1,960,1024)
        if self.variant == "full":
            dedisp, delays = self.dedisp(x)                  # (B,32,1,960,1024)
            weighted, attn = self.dm_attn(dedisp)            # (B,1,960,1024),(B,32)
        elif self.variant == "no_dm_attn":
            dedisp, delays = self.dedisp(x)
            weighted = dedisp.mean(dim=1)                    # GAP over K trials
            attn = None
        else:  # no_physics
            weighted = x
            attn = None
            delays = None
        feat = self.cnn(weighted)                            # (B,256,60,32)
        B, C, T_, F_ = feat.shape
        feat = feat.permute(0, 2, 1, 3).reshape(B, T_, C * F_)  # (B,60,8192)
        feat = self.proj(feat)                               # (B,60,256)
        feat = feat + self.pe[:, :T_, :]
        for block in self.conformers:
            feat = block(feat)
        global_feat = feat.mean(dim=1)                       # (B,256)
        if attn is not None:
            combined = torch.cat([global_feat, attn], dim=1)  # (B,256+32)
        else:
            combined = global_feat
        logits = self.classifier(combined)
        if return_attn:
            return logits, attn, delays
        return logits
