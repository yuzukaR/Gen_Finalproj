"""DreamBooth + LoRA on SDXL.

Wraps the official diffusers example
(`examples/dreambooth/train_dreambooth_lora_sdxl.py`).

Usage:
    python -m src.train.dreambooth_lora \
        --config configs/dblora_sdxl.yaml \
        --splits data/splits/splits.json \
        --prior data/prior \
        --trial trial1 --shots 5 \
        --out results/checkpoints/dblora/shots5/trial1
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

from src.train._shared import materialize_shot_set
from src.utils.io import load_yaml
from src.utils.logging import track_run

DIFFUSERS_SCRIPT_ENV = "DIFFUSERS_DBLORA_SDXL_SCRIPT"  # train_dreambooth_lora_sdxl.py


def build_cmd(cfg: dict, instance_dir: Path, prior_dir: Path, out_dir: Path) -> list[str]:
    script = os.environ.get(DIFFUSERS_SCRIPT_ENV)
    if not script:
        raise SystemExit(
            f"Set {DIFFUSERS_SCRIPT_ENV} to the path of "
            "diffusers/examples/dreambooth/train_dreambooth_lora_sdxl.py"
        )
    cmd = [
        "accelerate", "launch", script,
        f"--pretrained_model_name_or_path={cfg['base_model']}",
        f"--instance_data_dir={instance_dir}",
        f"--instance_prompt={cfg['instance_prompt']}",
        f"--resolution={cfg['resolution']}",
        f"--train_batch_size={cfg['train_batch_size']}",
        f"--gradient_accumulation_steps={cfg['gradient_accumulation_steps']}",
        f"--max_train_steps={cfg['max_train_steps']}",
        f"--learning_rate={cfg['learning_rate']}",
        f"--lr_scheduler={cfg['lr_scheduler']}",
        f"--lr_warmup_steps={cfg['lr_warmup_steps']}",
        f"--mixed_precision={cfg['mixed_precision']}",
        f"--rank={cfg['lora_rank']}",
        f"--checkpointing_steps={cfg['checkpointing_steps']}",
        f"--seed={cfg['seed']}",
        f"--output_dir={out_dir}",
        "--gradient_checkpointing",
        "--enable_xformers_memory_efficient_attention",
    ]
    if cfg.get("use_8bit_adam"):
        cmd.append("--use_8bit_adam")
    if cfg.get("with_prior_preservation"):
        cmd += [
            "--with_prior_preservation",
            f"--prior_loss_weight={cfg['prior_loss_weight']}",
            f"--class_data_dir={prior_dir}",
            f"--class_prompt={cfg['class_prompt']}",
            f"--num_class_images={cfg['num_class_images']}",
        ]
    if cfg.get("train_text_encoder"):
        cmd.append("--train_text_encoder")
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--splits", required=True, type=Path)
    ap.add_argument("--prior", required=True, type=Path)
    ap.add_argument("--trial", required=True)
    ap.add_argument("--shots", required=True, type=int)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    instance_dir = args.out / "_instance"
    materialize_shot_set(args.splits, args.trial, args.shots, instance_dir)

    cmd = build_cmd(cfg, instance_dir, args.prior, args.out)
    log_path = args.out / "run_stats.json"
    extra = {"method": "dreambooth_lora", "trial": args.trial, "shots": args.shots,
             "lora_rank": cfg["lora_rank"]}
    print("Launching:", " ".join(cmd), flush=True)
    with track_run(log_path, extra=extra):
        rc = subprocess.call(cmd)
    if rc != 0:
        sys.exit(rc)


if __name__ == "__main__":
    main()
