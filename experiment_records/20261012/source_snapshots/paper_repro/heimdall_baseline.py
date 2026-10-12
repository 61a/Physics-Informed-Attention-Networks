"""Heimdall traditional-method baseline (paper Table-1 counterpart).
Pure CPU: dedispersion over 100 DM trials (dm 0-3000) + peak/std SNR scoring,
then SNR-threshold scanning on the test split -> F1/recall/precision/AUC.

Vectorized roll via torch.gather (100x faster than per-channel torch.roll).

Usage: python heimdall_baseline.py [--split results/split_71515_s42.json]
Output: results/heimdall_baseline/metrics.json + snr_scores.npy
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

# Training is CPU-bound (GPU util 1-4%): limit Heimdall threads to avoid
# competing with the GPU training pipeline for CPU cores.
torch.set_num_threads(4)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.splits import load_split

ROOT = os.path.dirname(os.path.abspath(__file__))


class HeimdalLDedispersion:
    """Simulated Heimdall dedispersion (vectorized)."""
    def __init__(self, n_freq, n_time, n_dm_trials=100, dm_range=(0, 3000)):
        self.n_freq = n_freq
        self.n_time = n_time
        self.n_dm_trials = n_dm_trials

        # dispersion_factor as in original code
        freq_indices = torch.linspace(0, 1, n_freq)
        freq_ghz = 1.0 + freq_indices * 0.5
        dispersion_factor = (1.0 / freq_ghz**2) - (1.0 / freq_ghz.max()**2)
        dispersion_factor = dispersion_factor / (dispersion_factor.max() + 1e-8)
        dispersion_factor = dispersion_factor * (n_time * 0.2)
        self.dispersion_factor = dispersion_factor.numpy()

        self.dm_trials = np.linspace(dm_range[0], dm_range[1], n_dm_trials)

    def dedisperse_snr(self, data):
        """data: (n_time, n_freq) float32 -> best SNR over DM trials."""
        T, F = data.shape
        # shift each freq channel by -delay samples (delay depends on DM)
        best_snr = -np.inf
        x = torch.from_numpy(data.T).contiguous().float()  # (F, T)
        for dm in self.dm_trials:
            delays = dm * self.dispersion_factor          # (F,)
            shifts = delays.astype(np.int64)
            # vectorized roll: idx[f,t] = (t - shift[f]) % T  -> (F, T)
            idx = (torch.arange(T).view(1, -1) - torch.from_numpy(shifts).view(-1, 1)) % T
            idx = idx.expand(F, T)                        # (F, T)
            rolled = x.gather(1, idx)                     # roll along time dim
            timeseries = rolled.sum(dim=0).numpy()        # (T,)
            peak = np.max(timeseries)
            std = np.std(timeseries)
            snr = peak / (std + 1e-8)
            if snr > best_snr:
                best_snr = snr
        return best_snr


def load_sample(fpath):
    data = np.load(fpath)
    if data.ndim == 3:
        if data.shape[0] == 1:
            data = data[0]
        elif data.shape[1] == 1:
            data = data[:, 0, :]
        else:
            data = data.mean(axis=0)
    elif data.ndim == 2:
        pass
    return np.ascontiguousarray(data, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=os.path.join(ROOT, "results", "split_71515_s42.json"))
    ap.add_argument("--dm-trials", type=int, default=100)
    ap.add_argument("--n-time", type=int, default=15360)
    ap.add_argument("--n-freq", type=int, default=1024)
    args = ap.parse_args()

    split = load_split(args.split)
    test_files = split["test_pos"] + split["test_neg"]
    labels = [1] * len(split["test_pos"]) + [0] * len(split["test_neg"])
    print(f"Heimdall baseline: {len(test_files)} test samples "
          f"({len(split['test_pos'])} pos / {len(split['test_neg'])} neg)", flush=True)

    hd = HeimdalLDedispersion(n_freq=args.n_freq, n_time=args.n_time,
                              n_dm_trials=args.dm_trials)
    t0 = time.time()
    snrs = []
    for i, (f, lab) in enumerate(zip(test_files, labels)):
        data = load_sample(f)
        snr = hd.dedisperse_snr(data)
        snrs.append(snr)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(test_files)} t={time.time()-t0:.0f}s", flush=True)
    snrs = np.array(snrs)
    labels = np.array(labels)
    print(f"Dedispersion done in {time.time()-t0:.0f}s", flush=True)

    # Threshold sweep
    thrs = np.linspace(snrs.min(), snrs.max(), 400)
    best = {"f1": -1}
    for thr in thrs:
        pred = (snrs > thr).astype(int)
        tp = int(((pred == 1) & (labels == 1)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        tn = int(((pred == 0) & (labels == 0)).sum())
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-8, prec + rec)
        if f1 > best["f1"]:
            best = {"f1": f1, "recall": rec, "precision": prec, "tp": tp, "fn": fn,
                    "fp": fp, "tn": tn, "thr": float(thr)}

    # AUC (rank-based)
    order = np.argsort(-snrs)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(snrs))
    pos_ranks = ranks[labels == 1]
    neg_ranks = ranks[labels == 0]
    auc = (pos_ranks.mean() - neg_ranks.mean() + len(snrs)) / (2 * len(snrs)) if len(pos_ranks) and len(neg_ranks) else 0.5

    out_dir = os.path.join(ROOT, "results", "heimdall_baseline")
    os.makedirs(out_dir, exist_ok=True)
    metrics = {
        "method": "heimdall_dedispersion",
        "n_dm_trials": args.dm_trials, "dm_range": [0, 3000],
        "n_test": len(test_files), "n_pos": int(labels.sum()),
        "test_f1": round(best["f1"], 4), "test_recall": round(best["recall"], 4),
        "test_precision": round(best["precision"], 4),
        "cm": [best["tp"], best["fn"], best["fp"], best["tn"]],
        "best_thr": round(best["thr"], 4),
        "auc": round(float(auc), 4),
        "inference_time_s": round(time.time() - t0, 1),
    }
    json.dump(metrics, open(os.path.join(out_dir, "metrics.json"), "w"), indent=2, ensure_ascii=False)
    np.save(os.path.join(out_dir, "snr_scores.npy"), snrs)
    print("=== Heimdall result ===")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Saved -> {out_dir}/metrics.json")


if __name__ == "__main__":
    main()
