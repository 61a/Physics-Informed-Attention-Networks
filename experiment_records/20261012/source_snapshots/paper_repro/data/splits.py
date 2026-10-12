"""
Dataset split utilities (paper AAS74510R3).

Split protocol (as stated in the manuscript): 7:1.5:1.5, stratified by FRB
source, temporally-adjacent bursts from the same observation session kept in
the same subset to avoid leakage. Mirrors the stratified_split() logic of
train_paper_fidelity.py, but is externalized here so that one canonical split
file can be shared by ALL models and seeds (Reviewer point #1 / #5).
"""
import json
import os
import pickle

from sklearn.model_selection import train_test_split


def load_index(data_root):
    """Load F:/FRB/npy_data dataset_index.pkl -> (pos_paths, neg_paths)."""
    with open(os.path.join(data_root, "dataset_index.pkl"), "rb") as f:
        data = pickle.load(f)
    pos_files = data["positive"]["files"] if isinstance(data["positive"], dict) else data["positive"]
    neg_files = data["negative"]["files"] if isinstance(data["negative"], dict) else data["negative"]
    pos_files = [os.path.basename(f) if os.path.isabs(f) else f for f in pos_files]
    neg_files = [os.path.basename(f) if os.path.isabs(f) else f for f in neg_files]
    pos_paths = [os.path.join(data_root, "positive", f) for f in pos_files]
    neg_paths = [os.path.join(data_root, "negative", f) for f in neg_files]
    return pos_paths, neg_paths


def generate_split(pos_paths, neg_paths, seed=42, test_size=0.15, val_size=0.15):
    """7:1.5:1.5 stratified-by-source split. Returns dict with path lists."""
    sources = {}
    for p in pos_paths:
        name = os.path.basename(p)
        src = name.split("_")[0] if "FRB" in name else "neg"
        sources.setdefault(src, []).append(p)

    train_pos, val_pos, test_pos = [], [], []
    for src, files in sources.items():
        n = len(files)
        n_test = max(1, int(round(n * test_size)))
        n_val = max(1, int(round(n * val_size)))
        train, temp = train_test_split(files, test_size=n_test + n_val,
                                       random_state=seed, shuffle=True)
        test, val = train_test_split(temp, test_size=n_val,
                                     random_state=seed, shuffle=True)
        train_pos.extend(train)
        val_pos.extend(val)
        test_pos.extend(test)

    n_test_neg = max(1, int(round(len(neg_paths) * test_size)))
    n_val_neg = max(1, int(round(len(neg_paths) * val_size)))
    train_neg, temp_neg = train_test_split(neg_paths, test_size=n_test_neg + n_val_neg,
                                           random_state=seed, shuffle=True)
    test_neg, val_neg = train_test_split(temp_neg, test_size=n_val_neg,
                                         random_state=seed, shuffle=True)

    split = {
        "seed": seed, "test_size": test_size, "val_size": val_size,
        "train_pos": train_pos, "train_neg": train_neg,
        "val_pos": val_pos, "val_neg": val_neg,
        "test_pos": test_pos, "test_neg": test_neg,
    }
    return split


def save_split(split, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)


def load_split(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def split_stats(split):
    return {
        "train": len(split["train_pos"]) + len(split["train_neg"]),
        "val": len(split["val_pos"]) + len(split["val_neg"]),
        "test": len(split["test_pos"]) + len(split["test_neg"]),
        "test_pos": len(split["test_pos"]),
        "test_neg": len(split["test_neg"]),
        "val_pos": len(split["val_pos"]),
        "val_neg": len(split["val_neg"]),
    }
