"""
Model factory for paper_repro (paper AAS74510R3 Tables 1-3).

build_model(cfg, model_type) returns an nn.Module mapping (B,1,15360,1024) -> (B,2).

model_type:
  ours            - full model (physics front-end + CNN + 6x Conformer)   [Table 2 "Ours"]
  variant_a       - ablation: no physics layer (dedisp + DM attn removed) [Table 3 A]
  variant_b       - ablation: no DM attention (GAP over DM trials)        [Table 3 B]
  raw_resnet18    - raw-input baselines                                   [Table 1]
  raw_efficientnet_b0 / raw_convnext_tiny / raw_mobilenet_v3_large / raw_vit_small
  physics_<backbone> - physics front-end + backbone                       [Table 2]
"""
import torch
import torch.nn as nn

from .frontend import PhysicsFrontEnd
from .conformer import PaperFRBDetector
from .backbones import build_backbone


class RawBackbone(nn.Module):
    """Table 1: raw undedispersed dynamic spectrum, time-downsampled to T=60.

    15360 -> 60 via AvgPool2d((256,1)); frequency resolution 1024 kept.
    """

    def __init__(self, cfg, backbone):
        super().__init__()
        self.downsample = nn.AvgPool2d(kernel_size=(256, 1))
        self.backbone = backbone

    def forward(self, x):
        x = self.downsample(x)          # (B,1,60,1024)
        return self.backbone(x)


class PhysicsBackbone(nn.Module):
    """Table 2: shared physics front-end + backbone."""

    def __init__(self, cfg, backbone):
        super().__init__()
        self.frontend = PhysicsFrontEnd(cfg)
        self.backbone = backbone

    def forward(self, x):
        x, _ = self.frontend(x)         # (B,1,60,1024)
        return self.backbone(x)


RAW_BACKBONES = ["resnet18", "efficientnet_b0", "convnext_tiny",
                 "mobilenet_v3_large", "vit_small"]


def build_model(cfg, model_type, pretrained=True):
    if model_type == "ours":
        return PaperFRBDetector(cfg, variant="full")
    if model_type == "variant_a":
        return PaperFRBDetector(cfg, variant="no_physics")
    if model_type == "variant_b":
        return PaperFRBDetector(cfg, variant="no_dm_attn")

    if model_type.startswith("raw_"):
        name = model_type[4:]
        assert name in RAW_BACKBONES, f"unknown backbone {name}"
        return RawBackbone(cfg, build_backbone(name, pretrained=pretrained))

    if model_type.startswith("physics_"):
        name = model_type[8:]
        assert name in RAW_BACKBONES, f"unknown backbone {name}"
        return PhysicsBackbone(cfg, build_backbone(name, pretrained=pretrained))

    raise ValueError(f"unknown model_type: {model_type}")
