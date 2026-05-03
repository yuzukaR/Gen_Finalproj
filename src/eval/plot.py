"""Generate the report figures from results/metrics.csv.

Outputs to results/figures/:
    - identity_vs_shots.png   DINO identity vs shot count, mean+std across trials
    - clip_vs_shots.png       CLIP alignment vs shot count, mean+std across trials
    - identity_vs_clip.png    Scatter, one point per (method, shots, trial)
    - time_vs_shots.png       Training minutes per cell
    - vram_vs_shots.png       Peak VRAM per cell

Usage:
    python -m src.eval.plot --csv results/metrics.csv --out results/figures
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

METHOD_LABEL = {"ti": "Textual Inversion", "dblora": "DreamBooth + LoRA"}
METHOD_COLOR = {"ti": "#1f77b4", "dblora": "#d62728"}


def line_with_band(ax, df: pd.DataFrame, ycol: str, ylabel: str, title: str) -> None:
    for method, sub in df.groupby("method"):
        agg = sub.groupby("shots")[ycol].agg(["mean", "std"]).reset_index()
        ax.plot(agg["shots"], agg["mean"],
                marker="o", label=METHOD_LABEL[method], color=METHOD_COLOR[method])
        ax.fill_between(
            agg["shots"],
            agg["mean"] - agg["std"].fillna(0),
            agg["mean"] + agg["std"].fillna(0),
            alpha=0.2, color=METHOD_COLOR[method],
        )
    ax.set_xlabel("Number of training images")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()


def bar_grouped(ax, df: pd.DataFrame, ycol: str, ylabel: str, title: str) -> None:
    pivot = df.groupby(["shots", "method"])[ycol].mean().unstack("method")
    pivot.plot(kind="bar", ax=ax,
               color=[METHOD_COLOR[m] for m in pivot.columns])
    ax.set_xlabel("Number of training images")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, [METHOD_LABEL[m] for m in pivot.columns])


def scatter_id_vs_clip(ax, df: pd.DataFrame) -> None:
    for method, sub in df.groupby("method"):
        ax.scatter(sub["clip_alignment"], sub["dino_identity"],
                   s=80, alpha=0.7, label=METHOD_LABEL[method],
                   color=METHOD_COLOR[method])
    ax.set_xlabel("CLIP text–image alignment")
    ax.set_ylabel("DINOv2 identity similarity")
    ax.set_title("Identity vs. prompt adherence (per run)")
    ax.grid(True, alpha=0.3)
    ax.legend()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=Path("results/metrics.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/figures"))
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    args.out.mkdir(parents=True, exist_ok=True)

    plots = [
        ("identity_vs_shots.png", "dino_identity",
         "DINOv2 identity (mean cos)", "Identity consistency vs. shot count", line_with_band),
        ("clip_vs_shots.png", "clip_alignment",
         "CLIP alignment", "Prompt adherence vs. shot count", line_with_band),
        ("time_vs_shots.png", "elapsed_min",
         "Training time (min)", "Training cost vs. shot count", bar_grouped),
        ("vram_vs_shots.png", "peak_vram_gb",
         "Peak VRAM (GB)", "Peak GPU memory vs. shot count", bar_grouped),
    ]

    for fname, ycol, ylabel, title, fn in plots:
        fig, ax = plt.subplots(figsize=(6, 4))
        fn(ax, df, ycol, ylabel, title)
        fig.tight_layout()
        fig.savefig(args.out / fname, dpi=150)
        plt.close(fig)
        print(f"wrote {args.out / fname}")

    fig, ax = plt.subplots(figsize=(6, 5))
    scatter_id_vs_clip(ax, df)
    fig.tight_layout()
    fig.savefig(args.out / "identity_vs_clip.png", dpi=150)
    plt.close(fig)
    print(f"wrote {args.out / 'identity_vs_clip.png'}")


if __name__ == "__main__":
    main()
