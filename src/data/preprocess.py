"""Center-crop and resize raw subject photos to a fixed square resolution.

Usage:
    python -m src.data.preprocess --raw data/raw --out data/clean --size 1024
"""
import argparse
from pathlib import Path

from PIL import Image, ImageOps

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def process_one(src: Path, dst: Path, size: int) -> None:
    img = Image.open(src).convert("RGB")
    img = ImageOps.exif_transpose(img)
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    img = img.resize((size, size), Image.LANCZOS)
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst, format="PNG", optimize=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--size", type=int, default=1024)
    args = ap.parse_args()

    sources = sorted(p for p in args.raw.iterdir() if p.suffix.lower() in EXTS)
    if not sources:
        raise SystemExit(f"No images found in {args.raw}")

    for i, src in enumerate(sources):
        dst = args.out / f"{i:03d}.png"
        process_one(src, dst, args.size)
        print(f"[{i + 1}/{len(sources)}] {src.name} -> {dst.name}")


if __name__ == "__main__":
    main()
