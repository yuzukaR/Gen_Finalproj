#!/usr/bin/env bash
# Runs the full 24-cell grid: 2 methods x 4 shot counts x 3 trials.
# Each cell trains, generates the prompt suite, then scores DINO + CLIP.
#
# Required env:
#   DIFFUSERS_TI_SDXL_SCRIPT     -> diffusers/examples/textual_inversion/textual_inversion_sdxl.py
#   DIFFUSERS_DBLORA_SDXL_SCRIPT -> diffusers/examples/dreambooth/train_dreambooth_lora_sdxl.py
#
# Usage: bash scripts/run_grid.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

SPLITS="${SPLITS:-data/splits/splits.json}"
PRIOR="${PRIOR:-data/prior}"
REFS="${REFS:-data/clean}"

TI_CFG="configs/ti_sdxl.yaml"
DB_CFG="configs/dblora_sdxl.yaml"

SHOTS=(3 5 10 20)
TRIALS=(trial1 trial2 trial3)

run_ti () {
  local n=$1 t=$2
  local ckpt="results/checkpoints/ti/shots${n}/${t}"
  local samp="results/samples/ti/shots${n}/${t}"
  echo "=== TI shots=${n} ${t} ==="
  python -m src.train.textual_inversion \
      --config "$TI_CFG" --splits "$SPLITS" \
      --trial "$t" --shots "$n" --out "$ckpt"
  python -m src.infer.generate \
      --mode ti --ckpt "$ckpt" --method_config "$TI_CFG" --out "$samp"
  python -m src.eval.dino_identity --refs "$REFS" --gens "$samp" --out "${samp}/dino.json"
  python -m src.eval.clip_alignment --gens "$samp" --strip "<my-character>" --out "${samp}/clip.json"
}

run_dblora () {
  local n=$1 t=$2
  local ckpt="results/checkpoints/dblora/shots${n}/${t}"
  local samp="results/samples/dblora/shots${n}/${t}"
  echo "=== DB-LoRA shots=${n} ${t} ==="
  python -m src.train.dreambooth_lora \
      --config "$DB_CFG" --splits "$SPLITS" --prior "$PRIOR" \
      --trial "$t" --shots "$n" --out "$ckpt"
  python -m src.infer.generate \
      --mode dblora --ckpt "$ckpt" --method_config "$DB_CFG" --out "$samp"
  python -m src.eval.dino_identity --refs "$REFS" --gens "$samp" --out "${samp}/dino.json"
  python -m src.eval.clip_alignment --gens "$samp" --strip "sks" --out "${samp}/clip.json"
}

for n in "${SHOTS[@]}"; do
  for t in "${TRIALS[@]}"; do
    run_ti     "$n" "$t"
    run_dblora "$n" "$t"
  done
done

python -m src.eval.efficiency --out results/metrics.csv
echo "All done. See results/metrics.csv"
