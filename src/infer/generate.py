from __future__ import annotations

"""Generate the frozen prompt suite from a trained checkpoint, OR class-prior images.

Modes:
    --mode prior    : pure SDXL base, generates class-prior images for DB-LoRA
    --mode ti       : load TI embedding, generate the prompt suite
    --mode dblora   : load LoRA adapter, generate the prompt suite

Outputs PNGs as: <out>/<prompt_id>_<seed>.png and writes a manifest.json.
"""
import argparse
from pathlib import Path

import torch
from tqdm.auto import tqdm

from src.utils.runtime import disable_incompatible_torchao
from src.utils.io import dump_json, load_yaml
from src.utils.seed import set_seed

NEGATIVE = "low quality, blurry, deformed, extra limbs, watermark, text"


def resolve_generation_dtype(device: str, mixed_precision: str | None) -> tuple[torch.dtype, str | None]:
    if device != "cuda":
        return torch.float32, None

    if mixed_precision == "bf16" and torch.cuda.is_bf16_supported():
        # SDXL publishes fp16 weight variants; load those files but run the pipeline in bf16
        # so Colab A100 sampling matches the training config more closely.
        return torch.bfloat16, "fp16"

    return torch.float16, "fp16"


def load_pipeline(
    base_model: str,
    mode: str,
    ckpt: Path | None,
    *,
    mixed_precision: str | None = None,
) -> DiffusionPipeline:
    disable_incompatible_torchao()
    from diffusers import DiffusionPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    kwargs = {"use_safetensors": True}
    torch_dtype, variant = resolve_generation_dtype(device, mixed_precision)
    kwargs["torch_dtype"] = torch_dtype
    if variant is not None:
        kwargs["variant"] = variant
    pipe = DiffusionPipeline.from_pretrained(base_model, **kwargs)
    if mode == "ti":
        # diffusers stores the trained embedding under ckpt; shape is detected automatically.
        pipe.load_textual_inversion(str(ckpt))
    elif mode == "dblora":
        pipe.load_lora_weights(str(ckpt))
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    if device == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
    return pipe


def render_token(mode: str, cfg: dict) -> str:
    if mode == "ti":
        return cfg["placeholder_token"]
    if mode == "dblora":
        return f"{cfg['instance_token']} {cfg['class_word']}"
    raise ValueError(mode)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["prior", "ti", "dblora"])
    ap.add_argument("--base_model", default="stabilityai/stable-diffusion-xl-base-1.0")
    ap.add_argument("--ckpt", type=Path, default=None,
                    help="checkpoint dir (TI embedding or LoRA weights)")
    ap.add_argument("--method_config", type=Path, default=None,
                    help="ti_sdxl.yaml / dblora_sdxl.yaml (for the token spec)")
    ap.add_argument("--prompts", type=Path, default=Path("configs/prompts.yaml"))
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--num", type=int, default=200, help="prior mode: how many to gen")
    ap.add_argument("--class_prompt", default="a photo of a cat", help="prior mode")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    method_cfg = None
    if args.mode != "prior":
        if args.method_config is None:
            raise SystemExit("--method_config is required for ti and dblora generation")
        method_cfg = load_yaml(args.method_config)

    pipe = load_pipeline(
        args.base_model,
        args.mode,
        args.ckpt,
        mixed_precision=(method_cfg or {}).get("mixed_precision"),
    )
    generator_device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.mode == "prior":
        set_seed(0)
        for i in tqdm(range(args.num), desc="prior", unit="img"):
            g = torch.Generator(device=generator_device).manual_seed(i)
            img = pipe(
                args.class_prompt,
                negative_prompt=NEGATIVE,
                num_inference_steps=30,
                guidance_scale=5.0,
                generator=g,
            ).images[0]
            img.save(args.out / f"prior_{i:04d}.png")
        return

    prompts_cfg = load_yaml(args.prompts)
    token = render_token(args.mode, method_cfg)
    seeds = prompts_cfg["seeds"]

    manifest = []
    total = len(prompts_cfg["prompts"]) * len(seeds)
    pbar = tqdm(total=total, desc=f"gen[{args.mode}]", unit="img")
    for prompt in prompts_cfg["prompts"]:
        text = prompt["text"].format(token=token)
        for seed in seeds:
            g = torch.Generator(device=generator_device).manual_seed(seed)
            img = pipe(
                text,
                negative_prompt=NEGATIVE,
                num_inference_steps=30,
                guidance_scale=5.0,
                generator=g,
            ).images[0]
            fname = f"{prompt['id']}_seed{seed}.png"
            img.save(args.out / fname)
            manifest.append({
                "file": fname,
                "prompt_id": prompt["id"],
                "category": prompt["category"],
                "text": text,
                "seed": seed,
            })
            pbar.update(1)
    pbar.close()

    dump_json({"mode": args.mode, "ckpt": str(args.ckpt), "items": manifest},
              args.out / "manifest.json")


if __name__ == "__main__":
    main()
