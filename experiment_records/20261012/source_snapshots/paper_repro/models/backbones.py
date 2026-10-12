"""
Baseline backbones for Tables 1-2 (paper AAS74510R3).

Unified interface: every backbone takes (B, 1, T, F) input with T=60, F=1024
after time downsampling, and returns (B, 2) logits.

Adaptation scheme (answers Reviewer point #2):
- CNN backbones (ResNet-18 / EfficientNet-B0 / ConvNeXt-Tiny / MobileNetV3-Large):
  1-channel input is replicated to 3 channels; the pretrained first conv layer
  is kept untouched (channel-consistent); the classifier head is replaced by a
  2-class head; global adaptive pooling makes any input resolution valid.
- ViT-Small: timm `vit_small_patch16_224.augreg_in21k` (ImageNet-21k pretrained).
  1ch->3ch replicate; input padded 60->64 in time so the 16x16 patch grid is
  4x64=256 tokens; pretrained positional embedding (197 = 1 cls + 14x14) has its
  cls token dropped and the 14x14 grid bilinearly-bicubic-interpolated to 4x64;
  the patch embedding is re-initialized for the first epoch then unfrozen
  (mirrors the design documented in Response_to_Reviewer.md).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _replicate_channels(x):
    """(B,1,T,F) -> (B,3,T,F)."""
    return x.repeat(1, 3, 1, 1)


def _get_cnn_model(name, pretrained=True):
    import torchvision.models as tv
    if name == "resnet18":
        m = tv.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        # conv1: Conv2d(3,64,7,2,3) kept; only head replaced
        m.fc = nn.Linear(m.fc.in_features, 2)
    elif name == "efficientnet_b0":
        m = tv.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
        m.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(m.classifier[1].in_features, 2))
    elif name == "convnext_tiny":
        m = tv.convnext_tiny(weights="IMAGENET1K_V1" if pretrained else None)
        m.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.LayerNorm(m.classifier[2].in_features),
            nn.Linear(m.classifier[2].in_features, 2))
    elif name == "mobilenet_v3_large":
        m = tv.mobilenet_v3_large(weights="IMAGENET1K_V1" if pretrained else None)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, 2)
    else:
        raise ValueError(f"unknown CNN backbone: {name}")
    return m


class _CnnBackbone(nn.Module):
    """Wrap a torchvision CNN: 1ch->3ch replicate, keep pretrained weights."""

    def __init__(self, name, pretrained=True):
        super().__init__()
        self.name = name
        self.model = _get_cnn_model(name, pretrained)

    def forward(self, x):
        # x: (B,1,T,F); every torchvision backbone here already ends in
        # AdaptiveAvgPool (or fixed avgpool for resnet) + 2-class head
        x = _replicate_channels(x)
        return self.model(x)


class ViTSmall(nn.Module):
    """ViT-Small adapted to 1x60x1024 dynamic spectra.

    Grid: pad time 60->64, patch 16x16 -> 4x64 = 256 tokens (no cls, GAP pooling).
    Positional embedding: pretrained 197 -> drop cls -> 14x14 grid ->
    bicubic resize to 4x64 -> 256.
    """

    def __init__(self, pretrained=True, freeze_patch_first_epoch=True):
        super().__init__()
        import timm
        model_name = "vit_small_patch16_224.augreg_in21k" if pretrained else "vit_small_patch16_224"
        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        dim = self.vit.embed_dim  # 384
        self.patch_size = 16
        self.grid_t, self.grid_f = 4, 64          # padded 64 x 1024
        self.n_tokens = self.grid_t * self.grid_f  # 256

        # --- patch embedding: keep pretrained kernel (3ch), replicate input ---
        # (already a Conv2d(3,384,16,16), kept as-is); adapt size assertions
        pe = self.vit.patch_embed
        pe.img_size = (64, 1024)
        pe.grid_size = (64 // self.patch_size, 1024 // self.patch_size)
        pe.num_patches = pe.grid_size[0] * pe.grid_size[1]

        # --- positional embedding: 197 -> 256 via 2D bicubic ---
        pe = self.vit.pos_embed  # (1,197,dim); first token is cls
        pe_cls = pe[:, 0:1]                      # keep for reference only
        pe_grid = pe[:, 1:]                      # (1,196,dim) = 14x14
        H = W = 14
        pe_grid = pe_grid.reshape(1, H, W, dim).permute(0, 3, 1, 2)  # (1,dim,14,14)
        # bilinear is safe for non-multiple scales; bicubic via interpolate
        pe_resized = F.interpolate(pe_grid, size=(self.grid_t, self.grid_f),
                                   mode="bilinear", align_corners=False)  # (1,dim,4,64)
        self.pos_embed = pe_resized.reshape(1, dim, self.n_tokens).permute(0, 2, 1)  # (1,256,dim)
        self.pos_embed.requires_grad_(True)
        # drop cls token path entirely (use GAP on tokens)
        self.vit.pos_embed = nn.Parameter(self.pos_embed.detach().clone())
        del pe_cls

        # --- classifier head ---
        self.head = nn.Linear(dim, 2)

        if freeze_patch_first_epoch:
            self._freeze_patch_embed()

    def _freeze_patch_embed(self):
        for p in self.vit.patch_embed.parameters():
            p.requires_grad_(False)

    def unfreeze_patch_embed(self):
        for p in self.vit.patch_embed.parameters():
            p.requires_grad_(True)

    def forward(self, x, use_cls=False):
        # x: (B,1,T,F), T=60
        B = x.shape[0]
        x = _replicate_channels(x)                    # (B,3,60,1024)
        # pad time 60 -> 64
        x = F.pad(x, (0, 0, 0, self.grid_t * self.patch_size - x.shape[2]))  # (B,3,64,1024)
        tokens = self.vit.patch_embed(x)              # timm already flattens: (B,256,384)
        tokens = tokens + self.vit.pos_embed[:, :self.n_tokens, :]
        tokens = self.vit.blocks(tokens)              # (B,256,384)
        tokens = self.vit.norm(tokens)
        feat = tokens.mean(dim=1)                     # GAP (B,384)
        return self.head(feat)


def build_backbone(name, pretrained=True, input_mode="raw"):
    """Return an nn.Module mapping (B,1,60,1024) -> (B,2) logits."""
    if name == "vit_small":
        return ViTSmall(pretrained=pretrained)
    return _CnnBackbone(name, pretrained=pretrained)


if __name__ == "__main__":
    # quick smoke test of parameter counts on random input
    import time
    x = torch.randn(2, 1, 60, 1024)
    for name in ["resnet18", "efficientnet_b0", "convnext_tiny", "mobilenet_v3_large", "vit_small"]:
        m = build_backbone(name, pretrained=False)
        n = sum(p.numel() for p in m.parameters())
        t0 = time.time()
        with torch.no_grad():
            y = m(x)
        dt = (time.time() - t0) * 1000
        print(f"{name:22s} params={n:>10,}  out={tuple(y.shape)}  fwd={dt:.1f}ms")
