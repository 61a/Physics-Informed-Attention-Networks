"""Generate the 12 experiment config JSONs under paper_repro/configs/."""
import json, os

BASE = {
    "n_time": 15360, "n_freq": 1024, "n_pol": 1,
    "n_dm_trials": 32, "dm_range": [0, 3000],
    "f_min": 1000.0, "f_max": 1500.0, "dt": 9.8304e-5, "k_DM": 4.148808,
    "time_downsample": 16,
    "d_model": 256, "n_heads": 4, "n_conformer_blocks": 6,
    "conv_kernel": 31, "ffn_dim": 1024, "dropout": 0.1,
    "batch_size": 8, "epochs": 200, "lr": 1e-4, "weight_decay": 1e-4,
    "patience": 15, "min_epochs_before_stop": 30, "warmup_epochs": 5,
    "use_amp": True, "gradient_clip": 1.0, "label_smoothing": 0.0,
    "freq_mask_param": 30, "time_mask_param": 50, "augment_prob": 0.8,
    "pretrained": True,
    "data_root": "F:/FRB/npy_data",
    "split_file": "D:/FRB_RFI_Detection_AAS74510/paper_repro/results/split_71515_s42.json",
}

MODELS = [
    ("ours", None),
    ("variant_a", None),
    ("variant_b", None),
    ("raw_resnet18", None),
    ("raw_efficientnet_b0", None),
    ("raw_convnext_tiny", None),
    ("raw_mobilenet_v3_large", None),
    ("raw_vit_small", None),
    ("physics_resnet18", None),
    ("physics_efficientnet_b0", None),
    ("physics_convnext_tiny", None),
    ("physics_mobilenet_v3_large", None),
    ("physics_vit_small", None),
]

HERE = os.path.dirname(os.path.abspath(__file__))
for name, _ in MODELS:
    cfg = dict(BASE)
    cfg["model_type"] = name
    out = os.path.join(HERE, f"{name}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print("wrote", out)
print("done:", len(MODELS), "configs")
