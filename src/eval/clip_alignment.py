"""CLIP text–image alignment for a generated set, using its manifest.json.

For each (image, prompt_text) pair, computes cosine sim of CLIP image and text
embeddings. We strip the placeholder/identifier from the prompt before scoring,
since CLIP has no notion of <my-character> or "sks" — measuring alignment to
the *concept* description (e.g., "a photo of a cat on a beach at sunset").

Usage:
    python -m src.eval.clip_alignment \
        --gens results/samples/ti/shots5/trial1 \
        --strip "<my-character>" \
        --out results/samples/ti/shots5/trial1/clip.json
"""
import argparse
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

from src.utils.io import dump_json, load_json

MODEL_NAME = "openai/clip-vit-large-patch14"


def strip_token(text: str, token: str) -> str:
    cleaned = re.sub(re.escape(token), "subject", text)
    return re.sub(r"\s+", " ", cleaned).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", required=True, type=Path)
    ap.add_argument("--strip", default="",
                    help="placeholder string to replace with 'subject' before CLIP scoring")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model", default=MODEL_NAME)
    args = ap.parse_args()

    # Use CLIPModel/CLIPProcessor explicitly. Some environments still surface
    # BaseModelOutputWithPooling through CLIP text helpers, so we score from the
    # full CLIP forward pass and normalize the returned text/image embeddings.
    # Pinned transformers<5.0 in requirements.txt keeps this import path stable.
    from transformers import CLIPModel, CLIPProcessor  # lazy

    manifest = load_json(args.gens / "manifest.json")
    items = manifest["items"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(args.model).to(device).eval()
    proc = CLIPProcessor.from_pretrained(args.model)

    per_item = []
    by_category: dict[str, list[float]] = {}
    with torch.inference_mode():
        for it in tqdm(items, desc="CLIP", unit="img", leave=False):
            img = Image.open(args.gens / it["file"]).convert("RGB")
            text = strip_token(it["text"], args.strip) if args.strip else it["text"]
            inputs = proc(text=[text], images=[img],
                          return_tensors="pt", padding=True, truncation=True).to(device)
            outputs = model(**inputs)
            t = F.normalize(outputs.text_embeds, dim=-1)
            v = F.normalize(outputs.image_embeds, dim=-1)
            score = float((t * v).sum(dim=-1).item())
            per_item.append({**it, "clip_score": score, "scored_text": text})
            by_category.setdefault(it["category"], []).append(score)

    overall = sum(p["clip_score"] for p in per_item) / len(per_item)
    dump_json({
        "model": args.model,
        "n": len(per_item),
        "clip_alignment_mean": overall,
        "by_category": {k: sum(v) / len(v) for k, v in by_category.items()},
        "per_item": per_item,
    }, args.out)
    print(f"CLIP alignment = {overall:.4f} (n={len(per_item)})")


if __name__ == "__main__":
    main()
