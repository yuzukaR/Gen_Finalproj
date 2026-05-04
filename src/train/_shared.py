"""Helpers shared by both training stubs."""
import shutil
from pathlib import Path

from src.utils.io import load_json


def materialize_shot_set(splits_path: Path, trial: str, shots: int, dst: Path) -> Path:
    """Copy the chosen shot subset into an isolated directory the trainer reads from."""
    splits = load_json(splits_path)
    clean_dir = Path(splits["clean_dir"])
    files = splits["splits"][trial][f"shots{shots}"]
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(clean_dir / f, dst / f)
    return dst
