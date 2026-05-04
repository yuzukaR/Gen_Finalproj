"""Textual Inversion on SDXL.

Wraps the official diffusers example
(`examples/textual_inversion/textual_inversion_sdxl.py`) so we get a battle-tested
trainer and only own the experiment plumbing.

Usage:
    python -m src.train.textual_inversion \
        --config configs/ti_sdxl.yaml \
        --splits data/splits/splits.json \
        --trial trial1 --shots 5 \
        --out results/checkpoints/ti/shots5/trial1

Requires `diffusers` installed from source or `examples/` on PYTHONPATH; we shell
out via accelerate so memory + multi-GPU behavior matches the upstream script.
"""
import argparse
import subprocess
import sys
from pathlib import Path

from src.train._launch import (
    build_subprocess_env,
    module_available,
    resolve_diffusers_example_script,
)
from src.train._shared import materialize_shot_set
from src.utils.io import load_yaml
from src.utils.logging import track_run

DIFFUSERS_SCRIPT_ENV = "DIFFUSERS_TI_SDXL_SCRIPT"  # path to textual_inversion_sdxl.py


def build_cmd(cfg: dict, instance_dir: Path, out_dir: Path) -> list[str]:
    script = resolve_diffusers_example_script(
        DIFFUSERS_SCRIPT_ENV,
        "examples/textual_inversion/textual_inversion_sdxl.py",
    )
    cmd = [
        "accelerate", "launch", script,
        f"--pretrained_model_name_or_path={cfg['base_model']}",
        f"--train_data_dir={instance_dir}",
        f"--placeholder_token={cfg['placeholder_token']}",
        f"--initializer_token={cfg['initializer_token']}",
        f"--num_vectors={cfg['num_vectors']}",
        f"--resolution={cfg['resolution']}",
        f"--train_batch_size={cfg['train_batch_size']}",
        f"--gradient_accumulation_steps={cfg['gradient_accumulation_steps']}",
        f"--max_train_steps={cfg['max_train_steps']}",
        f"--learning_rate={cfg['learning_rate']}",
        f"--lr_scheduler={cfg['lr_scheduler']}",
        f"--lr_warmup_steps={cfg['lr_warmup_steps']}",
        f"--mixed_precision={cfg['mixed_precision']}",
        f"--save_steps={cfg['save_steps']}",
        f"--checkpointing_steps={cfg['checkpointing_steps']}",
        f"--seed={cfg['seed']}",
        f"--output_dir={out_dir}",
        "--gradient_checkpointing",
    ]
    if cfg.get("scale_lr"):
        cmd.append("--scale_lr")
    if "adam_beta1" in cfg:
        cmd.append(f"--adam_beta1={cfg['adam_beta1']}")
    if "adam_beta2" in cfg:
        cmd.append(f"--adam_beta2={cfg['adam_beta2']}")
    if "adam_weight_decay" in cfg:
        cmd.append(f"--adam_weight_decay={cfg['adam_weight_decay']}")
    if "adam_epsilon" in cfg:
        cmd.append(f"--adam_epsilon={cfg['adam_epsilon']}")
    if module_available("xformers"):
        cmd.append("--enable_xformers_memory_efficient_attention")
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--splits", required=True, type=Path)
    ap.add_argument("--trial", required=True)
    ap.add_argument("--shots", required=True, type=int)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    instance_dir = args.out / "_instance"
    materialize_shot_set(args.splits, args.trial, args.shots, instance_dir)

    cmd = build_cmd(cfg, instance_dir, args.out)
    log_path = args.out / "run_stats.json"
    extra = {"method": "textual_inversion", "trial": args.trial, "shots": args.shots}
    print("Launching:", " ".join(cmd), flush=True)
    if not module_available("xformers"):
        print("Environment: xformers unavailable, training without memory-efficient attention", flush=True)
    with track_run(log_path, extra=extra):
        rc = subprocess.call(cmd, env=build_subprocess_env())
    if rc != 0:
        sys.exit(rc)


if __name__ == "__main__":
    main()
