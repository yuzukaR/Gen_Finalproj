"""DINOv2 identity similarity between reference set and a folder of generations.

For each generated image:
    sim_i = mean cosine sim(DINO(generated_i), DINO(reference_j)) over all j

The run-level score is the mean over all generated images. This rewards
generations that look like *any* reference rather than penalizing pose/scene
variation.

Usage:
    python -m src.eval.dino_identity \
        --refs data/clean \
        --gens results/samples/ti/shots5/trial1 \
        --out results/samples/ti/shots5/trial1/dino.json
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.utils.io import dump_json

MODEL_NAME = "facebook/dinov2-large"
EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def build_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])


@torch.inference_mode()
def embed(model, processor_tf, paths: list[Path], device: str) -> torch.Tensor:
    feats = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        x = processor_tf(img).unsqueeze(0).to(device)
        out = model(pixel_values=x)
        # CLS token from last_hidden_state; shape (1, hidden)
        cls = out.last_hidden_state[:, 0]
        feats.append(F.normalize(cls, dim=-1).cpu())
    return torch.cat(feats, dim=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", required=True, type=Path)
    ap.add_argument("--gens", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model", default=MODEL_NAME)
    args = ap.parse_args()

    from transformers import AutoModel  # lazy import

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(args.model).to(device).eval()
    tf = build_transform()

    ref_paths = sorted(p for p in args.refs.iterdir() if p.suffix.lower() in EXTS)
    gen_paths = sorted(p for p in args.gens.iterdir() if p.suffix.lower() in EXTS)
    if not ref_paths or not gen_paths:
        raise SystemExit("empty refs or gens")

    ref_feats = embed(model, tf, ref_paths, device)   # (R, D)
    gen_feats = embed(model, tf, gen_paths, device)   # (G, D)

    # cosine sim matrix (G, R) — both already L2-normalized
    sim = gen_feats @ ref_feats.T
    per_image = sim.mean(dim=1)        # (G,)
    score = float(per_image.mean())

    dump_json({
        "model": args.model,
        "n_refs": len(ref_paths),
        "n_gens": len(gen_paths),
        "dino_identity_mean": score,
        "dino_identity_std": float(per_image.std()),
        "per_image": [
            {"file": p.name, "score": float(s)}
            for p, s in zip(gen_paths, per_image.tolist())
        ],
    }, args.out)
    print(f"DINO identity = {score:.4f}  (n={len(gen_paths)} gens vs {len(ref_paths)} refs)")


if __name__ == "__main__":
    main()
