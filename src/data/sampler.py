"""Build deterministic, *nested* shot-set splits for the experiment grid.

For each trial t in [0, 1, 2]:
    - draw a random permutation of the cleaned image pool (seeded by trial)
    - take prefixes of size 3, 5, 10, 20

Nesting (3 ⊂ 5 ⊂ 10 ⊂ 20) means cross-shot variance reflects added images, not a
totally different subject mix — cleaner interpretation of "how many do we need?".

Usage:
    python -m src.data.sampler --clean data/clean --out data/splits/splits.json
"""
import argparse
import random
from pathlib import Path

from src.utils.io import dump_json

SHOT_COUNTS = [3, 5, 10, 20]
TRIAL_SEEDS = [101, 202, 303]  # one per trial


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    pool = sorted(p.name for p in args.clean.iterdir() if p.suffix == ".png")
    if len(pool) < max(SHOT_COUNTS):
        raise SystemExit(
            f"Need at least {max(SHOT_COUNTS)} cleaned images, found {len(pool)}"
        )

    splits: dict[str, dict[str, list[str]]] = {}
    for trial_idx, seed in enumerate(TRIAL_SEEDS):
        rng = random.Random(seed)
        permuted = pool.copy()
        rng.shuffle(permuted)
        trial_key = f"trial{trial_idx + 1}"
        splits[trial_key] = {
            f"shots{n}": permuted[:n] for n in SHOT_COUNTS
        }

    dump_json(
        {
            "clean_dir": str(args.clean),
            "shot_counts": SHOT_COUNTS,
            "trial_seeds": TRIAL_SEEDS,
            "splits": splits,
        },
        args.out,
    )
    print(f"wrote {args.out}")
    for trial_key, by_shot in splits.items():
        print(f"  {trial_key}: " + ", ".join(f"{k}={len(v)}" for k, v in by_shot.items()))


if __name__ == "__main__":
    main()
