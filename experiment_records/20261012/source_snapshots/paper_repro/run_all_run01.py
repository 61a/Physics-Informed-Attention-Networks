"""Stage-1 scheduler: run all 13 NN configs (seed=42) sequentially.
- Each config: subprocess train.py, check return code.
- Abnormal exit (rc != 0) -> auto-resume retry (up to 3 attempts, 60s apart).
- Writes run_all_s42.log + run_all_s42_summary.json.

Usage: python run_all_s42.py [--seed 42] [--epochs 1(smoke)|full]
Detached launch: python launch_detached.py run_all_s42.py
"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = r"C:\ProgramData\anaconda3\envs\envly61\python.exe"

# Order: Table-1 raw baselines, Table-2 physics baselines, Ours, ablations
CONFIGS = [
    "raw_resnet18", "raw_efficientnet_b0", "raw_convnext_tiny",
    "raw_mobilenet_v3_large", "raw_vit_small",
    "physics_resnet18", "physics_efficientnet_b0", "physics_convnext_tiny",
    "physics_mobilenet_v3_large", "physics_vit_small",
    "ours", "variant_a", "variant_b",
]


def log(msg, f):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {msg}"
    print(line, flush=True)
    f.write(line + "\n")
    f.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=None,
                    help="override epochs (smoke use --epochs 1)")
    ap.add_argument("--gpu-fraction", type=float, default=0.70)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None,
                    help="only run first N configs (scheduler smoke test)")
    ap.add_argument("--only", type=str, default=None,
                    help="comma-separated config names to run ONLY these "
                         "(keeps other configs' previous results in summary)")
    ap.add_argument("--tag", type=str, default="",
                    help="suffix for log/summary filenames (e.g. '_tune1')")
    args = ap.parse_args()

    configs = CONFIGS[:args.limit] if args.limit else CONFIGS
    summary_path = os.path.join(ROOT, f"run_all_s{args.seed}{args.tag}_summary.json")

    # Load previous summary so finished configs not re-run this time are preserved
    old_jobs = {}
    if os.path.exists(summary_path):
        try:
            old = json.load(open(summary_path, encoding="utf-8"))
            old_jobs = {j["config"]: j for j in old.get("jobs", [])}
        except Exception:
            old_jobs = {}

    if args.only:
        only = [c.strip() for c in args.only.split(",") if c.strip()]
        only_set = set(only)
        configs = [c for c in configs if c in only_set]
        # allow config names outside CONFIGS (e.g. tune experiments) if json exists
        extras = [c for c in only if c not in CONFIGS
                  and os.path.exists(os.path.join(ROOT, "configs", f"{c}.json"))]
        configs += extras
        missing = [c for c in only if c not in configs]
        if missing:
            print(f"[warn] unknown configs skipped: {missing}", flush=True)

    logf = open(os.path.join(ROOT, f"run_all_s{args.seed}{args.tag}.log"), "a", encoding="utf-8")
    summary = {"seed": args.seed, "jobs": [], "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    kept = [j for name, j in old_jobs.items() if name not in configs]
    log(f"=== Stage-1 scheduler: {len(configs)} configs to run, seed={args.seed}, "
        f"epochs={args.epochs or 'full'}, gpu_frac={args.gpu_fraction}, "
        f"keeping {len(kept)} previously-completed configs ===", logf)

    for cfg_name in configs:
        cfg_path = os.path.join(ROOT, "configs", f"{cfg_name}.json")
        exp_dir = os.path.join(ROOT, "results", f"{cfg_name}_s{args.seed}")
        log(f"--- [{cfg_name}] start ---", logf)

        job = {"config": cfg_name, "attempts": [], "status": "pending"}
        rc = None
        for attempt in range(1, args.max_retries + 1):
            cmd = [PY, "-u", os.path.join(ROOT, "train.py"),
                   "--config", cfg_path, "--seed", str(args.seed),
                   "--output-dir", os.path.join(ROOT, "results"),
                   "--gpu-fraction", str(args.gpu_fraction)]
            if args.epochs:
                cmd += ["--epochs", str(args.epochs)]
            t0 = time.time()
            log(f"  attempt {attempt}/{args.max_retries}: {' '.join(cmd)}", logf)
            p = subprocess.run(cmd, cwd=ROOT)
            rc = p.returncode
            dur = time.time() - t0
            log(f"  attempt {attempt} finished rc={rc} dur={dur/60:.1f}min", logf)
            job["attempts"].append({"attempt": attempt, "rc": rc, "dur_min": round(dur / 60, 1)})

            metrics_path = os.path.join(exp_dir, "metrics.json")
            done = (rc == 0) and os.path.exists(metrics_path)
            if done:
                break
            if attempt < args.max_retries:
                log(f"  rc={rc} != 0 -> sleep 60s, auto-resume (retry {attempt+1})", logf)
                time.sleep(60)

        # Determine final status
        metrics_path = os.path.join(exp_dir, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, encoding="utf-8") as f:
                m = json.load(f)
            job["status"] = "complete" if rc == 0 else f"complete_rc{rc}"
            t = m.get("test", {})
            job["test_f1"] = t.get("f1")
            job["test_recall"] = t.get("recall")
            job["test_precision"] = t.get("precision")
            job["test_cm"] = [t.get("tp"), t.get("fn"), t.get("fp"), t.get("tn")]
            job["best_val_f1"] = m.get("best_val_f1")
            job["val_threshold"] = m.get("val_threshold")
            job["epochs_trained"] = m.get("epochs_trained")
            job["train_time_s"] = m.get("train_time_s")
            job["params"] = m.get("n_params_total")
        else:
            job["status"] = f"FAILED_rc{rc}"
        summary["jobs"].append(job)
        log(f"--- [{cfg_name}] status={job['status']} "
            f"test_f1={job.get('test_f1')} recall={job.get('test_recall')} ---", logf)
        json.dump(summary, open(summary_path, "w"), ensure_ascii=False, indent=2)

    summary["jobs"] = kept + summary["jobs"]
    summary["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(summary, open(summary_path, "w"), ensure_ascii=False, indent=2)
    log(f"=== ALL DONE: {sum(1 for j in summary['jobs'] if j['status'].startswith('complete'))}"
        f"/{len(summary['jobs'])} ===", logf)
    logf.close()


if __name__ == "__main__":
    main()
