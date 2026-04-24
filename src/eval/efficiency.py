"""Aggregate per-run training stats + eval scores into a single metrics.csv.

Walks results/checkpoints/<method>/shots<N>/trial<T>/ for run_stats.json, and
results/samples/<method>/shots<N>/trial<T>/{dino,clip}.json, joins them, writes
results/metrics.csv with one row per run.
"""
import argparse
import csv
from pathlib import Path

from src.utils.io import load_json

METHODS = ["ti", "dblora"]
SHOTS = [3, 5, 10, 20]
TRIALS = ["trial1", "trial2", "trial3"]


def maybe(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("results/metrics.csv"))
    args = ap.parse_args()

    rows = []
    for method in METHODS:
        for n in SHOTS:
            for trial in TRIALS:
                ckpt_dir = args.results_root / "checkpoints" / method / f"shots{n}" / trial
                samples_dir = args.results_root / "samples" / method / f"shots{n}" / trial
                stats = ckpt_dir / "run_stats.json"
                dino = samples_dir / "dino.json"
                clip_ = samples_dir / "clip.json"
                if not stats.exists():
                    continue
                run = load_json(stats)
                row = {
                    "method": method,
                    "shots": n,
                    "trial": trial,
                    "elapsed_min": maybe(run, "elapsed_min"),
                    "peak_vram_gb": maybe(run, "peak_vram_gb"),
                    "lora_rank": maybe(run, "lora_rank"),
                    "dino_identity": maybe(load_json(dino), "dino_identity_mean") if dino.exists() else None,
                    "dino_identity_std": maybe(load_json(dino), "dino_identity_std") if dino.exists() else None,
                    "clip_alignment": maybe(load_json(clip_), "clip_alignment_mean") if clip_.exists() else None,
                }
                rows.append(row)

    if not rows:
        raise SystemExit("no runs found under results/")

    fields = list(rows[0].keys())
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
