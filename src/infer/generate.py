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
from diffusers import DiffusionPipeline
from tqdm.auto import tqdm

from src.utils.io import dump_json, load_yaml
from src.utils.seed import set_seed

NEGATIVE = "low quality, blurry, deformed, extra limbs, watermark, text"


def load_pipeline(base_model: str, mode: str, ckpt: Path | None) -> DiffusionPipeline:
    pipe = DiffusionPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    if mode == "ti":
        # diffusers stores the trained embedding under ckpt; shape is detected automatically.
        pipe.load_textual_inversion(str(ckpt))
    elif mode == "dblora":
        pipe.load_lora_weights(str(ckpt))
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
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
    pipe = load_pipeline(args.base_model, args.mode, args.ckpt)

    if args.mode == "prior":
        set_seed(0)
        for i in tqdm(range(args.num), desc="prior", unit="img"):
            g = torch.Generator(device="cuda").manual_seed(i)
            img = pipe(
                args.class_prompt,
                negative_prompt=NEGATIVE,
                num_inference_steps=30,
                guidance_scale=5.0,
                generator=g,
            ).images[0]
            img.save(args.out / f"prior_{i:04d}.png")
        return

    method_cfg = load_yaml(args.method_config)
    prompts_cfg = load_yaml(args.prompts)
    token = render_token(args.mode, method_cfg)
    seeds = prompts_cfg["seeds"]

    manifest = []
    total = len(prompts_cfg["prompts"]) * len(seeds)
    pbar = tqdm(total=total, desc=f"gen[{args.mode}]", unit="img")
    for prompt in prompts_cfg["prompts"]:
        text = prompt["text"].format(token=token)
        for seed in seeds:
            g = torch.Generator(device="cuda").manual_seed(seed)
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
