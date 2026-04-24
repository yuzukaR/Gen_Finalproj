# Few-Shot Personalization for Consistent Character Generation — Project Plan

**Course:** CS 5788 (Spring 2026), Instructor: Andrew Owens
**Team:** Tongjia Rao (tr426), Haoshen Wu (hw862), Jiasheng Zhu (jz2455)
**Posted:** April 21, 2026 · **Due:** Tuesday, May 12, 2026

---

## Project at a Glance

- **Question:** How many reference images does each personalization method need to produce a consistent, prompt-controllable character?
- **Methods compared:** Textual Inversion (TI) vs. DreamBooth + LoRA (DB-LoRA), on **SDXL base 1.0** (`stabilityai/stable-diffusion-xl-base-1.0`).
- **Experiment grid:** 2 methods × 4 shot counts (3, 5, 10, 20) × 3 trials = **24 runs**.
- **Metrics:** DINOv2 identity similarity, CLIP text–image alignment, training time, peak VRAM, human ranking.
- **Compute budget:** ~40–70 GPU-hours on a single A100 (16–24 GB VRAM headroom for SDXL).

---

## Deliverables (per project guidelines)

1. **Written report** — CVPR style, **4 pages** max (excluding refs), `pagenumbers` mode.
2. **Code repository** — zipped.
3. **Jupyter notebook** — 4 pages max, demo only, results pre-rendered in cells, appended to the report PDF.
4. **In-class presentation** — strict 3 minutes; motivation + demo of best results.

Report sections required: Introduction, Method (with at least one figure), Experiments (data, metrics + justification, qualitative + quantitative results), Conclusion.

---

## Repository Structure

```
genfinal_proj/
├── data/
│   ├── raw/                      # 20–30 curated photos of target subject
│   ├── splits/                   # JSON manifests: shot_count × trial → file list
│   └── prior/                    # SDXL-generated class-prior images (DB-LoRA)
├── src/
│   ├── data/preprocess.py        # crop, resize to 1024², caption
│   ├── data/sampler.py           # deterministic subset sampling per (shots, trial)
│   ├── train/textual_inversion.py
│   ├── train/dreambooth_lora.py
│   ├── infer/generate.py         # fixed prompt suite + fixed seeds
│   ├── eval/dino_identity.py     # DINOv2 cosine similarity
│   ├── eval/clip_alignment.py    # CLIP text-image score
│   ├── eval/efficiency.py        # parses logs → time + peak VRAM
│   └── utils/{seed,logging,io}.py
├── configs/
│   ├── ti_sdxl.yaml
│   ├── dblora_sdxl.yaml
│   └── prompts.yaml              # 12–20 prompts × 4 categories
├── scripts/run_grid.sh           # launches all 24 jobs
├── results/
│   ├── checkpoints/{method}/{shots}/{trial}/
│   ├── samples/{method}/{shots}/{trial}/{prompt_id}_{seed}.png
│   ├── metrics.csv
│   └── figures/
├── notebooks/demo.ipynb          # the deliverable notebook
├── report/                       # CVPR LaTeX
└── README.md
```

---

## Phase Plan (Apr 24 → May 12)

### Phase 0 — Setup (Apr 24–26, 3 days)
- Pin environment: `diffusers`, `peft`, `transformers`, `accelerate`, `torch`, `xformers`, `bitsandbytes`.
- Lock model weights: SDXL base 1.0, DINOv2 (`dinov2_vitl14`), CLIP (ViT-L/14).
- Decide compute target and confirm A100 access; budget ~40–70 GPU-hr.
- **Owner:** Jiasheng (compute/logging), Tongjia (eval libs).

### Phase 1 — Data & Protocol (Apr 26–29)
- Collect 20–30 curated subject photos; clean, square-crop, 1024².
- Fixed seeds → `splits.json` with 12 shot-set assignments (4 sizes × 3 trials). Nest 3 ⊂ 5 ⊂ 10 ⊂ 20 when feasible so variance reflects sampling noise, not subject mix.
- Author **prompt suite** in `prompts.yaml`: 4 categories (style / scene / pose / accessory), 4 prompts each → 16 prompts × 4 seeds = 64 generations per run.
- Generate ~200 class-prior images for DB-LoRA prior-preservation.
- **Owner:** all three.

### Phase 2 — Pipelines (Apr 29 – May 3)
- **TI pipeline (Haoshen):** train new token `<my-character>`; sweep LR + steps on the 5-shot trial 1; freeze hyperparams for the grid.
- **DB-LoRA pipeline (Jiasheng):** rare identifier + prior-preservation loss; LoRA rank 16 vs. 32 vs. 64 mini-sweep on 5-shot trial 1; freeze the best.
- Both pipelines log: wallclock, peak `torch.cuda.max_memory_allocated`, step loss.
- Smoke-test each at 3-shot before launching the full grid.
- **Gate:** one TI + one DB-LoRA run produces visibly on-identity samples before scaling up.

### Phase 3 — Full Experiment Grid (May 3–7)
- Launch 24 runs from `scripts/run_grid.sh`, serialized on one GPU.
- Generate the 64-image prompt suite per run (1,536 images total).
- All checkpoints + samples saved under `results/`.
- **Risk buffer:** if compute slips, drop to 2 trials per size (16 runs) and note in the report.

### Phase 4 — Evaluation (May 7–9)
- DINOv2 identity: mean cosine sim between generated samples and the reference set.
- CLIP alignment: per-prompt CLIP score, then averaged.
- Efficiency: parse training logs → minutes + GB.
- Human ranking: each teammate ranks 5 prompt × 4 shot grids per method (blinded filenames); aggregate with Borda count.
- Output `metrics.csv` + plots: identity vs. shots, CLIP vs. shots, identity-vs-CLIP scatter, time/VRAM bar charts.
- **Owner:** Tongjia (drives), all review.

### Phase 5 — Writing & Notebook (May 9–11)
- **Report (4 pages, CVPR `pagenumbers` mode):**
  1. **Introduction** — why few-shot identity matters, related work (LoRA, DreamBooth, Custom Diffusion, DINOv2, CLIP), our contribution: a controlled low-shot benchmark.
  2. **Method** — TI + DB-LoRA descriptions, **figure**: pipeline diagram (data → method → fixed prompt suite → metrics).
  3. **Experiments** — data, metrics + justification, qualitative grids, quantitative tables/plots, failure cases.
  4. **Conclusion** — practical guidance on minimum dataset size per method.
- **Notebook (4 pages PDF):** loads the smallest checkpoint, runs 1 prompt per category, displays cached results from `results/`. Code is thin wrappers over `src/`.
- Append notebook PDF after references; submit single PDF to Gradescope.

### Phase 6 — Presentation & Submit (May 11–12)
- 3-minute slides: motivation → key plot (identity vs. shots) → 1 qualitative grid → take-away.
- Zip the repo (exclude checkpoints if oversized; include a download link or smallest checkpoint).
- Submit by **Tue, May 12**.

---

## Workload (mirrors proposal §6)

| Owner | Primary | Secondary |
|---|---|---|
| **Tongjia** | Eval framework (DINO/CLIP), experiment design, metrics tables, fairness protocol | Report §3 Experiments, figures |
| **Haoshen** | TI pipeline, hyperparameter sweep, qualitative curation | Report §2 Method (TI), notebook |
| **Jiasheng** | DB-LoRA pipeline, prior-preservation, compute/logging | Report §2 Method (DB-LoRA), repo packaging |
| Shared | Dataset, writing, slides | — |

---

## Key Risks and Mitigations

- **Compute overrun.** Drop to 2 trials × 4 shots = 16 runs; keep variance reporting honest.
- **TI fails to converge on 3-shot.** Expected outcome — report as a finding, not a bug.
- **DB-LoRA overfits at high shots.** Tune LoRA rank + LR on a held-out shot-5 sweep before the grid.
- **Prompt-suite leakage.** Fix prompts and seeds *before* any qualitative inspection to avoid cherry-picking.
- **Metric mismatch.** Report DINOv2 and CLIP together; identity often trades against prompt adherence.

---

## Milestone Checklist

- [ ] Phase 0 — env pinned, weights downloaded
- [ ] Phase 1 — dataset curated, splits frozen, prompts frozen, prior set generated
- [ ] Phase 2 — both pipelines pass smoke test on 3-shot
- [ ] Phase 3 — 24-run grid complete (or 16-run fallback)
- [ ] Phase 4 — `metrics.csv` + figures finalized
- [ ] Phase 5 — report draft + notebook draft
- [ ] Phase 6 — slides ready, repo zipped, Gradescope submission
