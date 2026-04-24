# Few-Shot Personalization for Consistent Character Generation

CS 5788 final project. Compares **Textual Inversion** vs. **DreamBooth + LoRA** on
SDXL base 1.0 across {3, 5, 10, 20} reference images × 3 trials.

See [`plan.md`](./plan.md) for the full project plan and timeline.

## Quickstart

```bash
# 1. Environment (Python 3.10+, CUDA 11.8+)
pip install -r requirements.txt

# 2. Drop 20–30 curated subject photos in data/raw/, then:
python -m src.data.preprocess --raw data/raw --out data/clean
python -m src.data.sampler --clean data/clean --out data/splits/splits.json

# 3. Generate class-prior images for DB-LoRA (run once)
python -m src.infer.generate \
    --mode prior \
    --class_prompt "a photo of a cat" \
    --num 200 --out data/prior

# 4. Launch the full 24-run grid
bash scripts/run_grid.sh
```

## Layout

```
src/
  data/        preprocess.py, sampler.py
  train/       textual_inversion.py, dreambooth_lora.py
  infer/       generate.py
  eval/        dino_identity.py, clip_alignment.py, efficiency.py
  utils/       seed.py, logging.py, io.py
configs/       ti_sdxl.yaml, dblora_sdxl.yaml, prompts.yaml
scripts/       run_grid.sh
results/       checkpoints/, samples/, figures/, metrics.csv
```

## Reproducibility

- All seeds fixed in `configs/*.yaml` and `prompts.yaml`.
- Splits frozen in `data/splits/splits.json` before any training.
- Prompt suite frozen in `configs/prompts.yaml` before any qualitative inspection.
