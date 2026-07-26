"""分析 OCR 文本长度和预测分数的分布，输出统计摘要 + 分布图。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scienceplots  # noqa: F401


def analyze(input_path: str, output_dir: str | None = None):
    if output_dir is None:
        output_dir = Path(input_path).stem + "_dist"
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(input_path)
    if "OCR文字结果" not in df.columns:
        print("Error: input xlsx must have 'OCR文字结果' column")
        sys.exit(1)

    plt.style.use(["science", "no-latex"])
    figsize = (8, 5)
    dpi = 150

    # ── 1. OCR 文本长度分布 ──────────────────────────────────
    texts = df["OCR文字结果"].fillna("").astype(str)
    lengths = texts.str.len()

    print("=" * 50)
    print("【OCR 文本长度统计】")
    print(lengths.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_string())
    print(f"零长度文本数: {(lengths == 0).sum()}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=dpi)
    axes[0].hist(lengths, bins=80, color="steelblue", edgecolor="white", linewidth=0.3)
    axes[0].set_xlabel("文本长度")
    axes[0].set_ylabel("频数")
    axes[0].set_title("OCR 文本长度分布")

    axes[1].hist(lengths, bins=80, color="steelblue", edgecolor="white", linewidth=0.3)
    axes[1].set_xlabel("文本长度")
    axes[1].set_ylabel("频数")
    axes[1].set_yscale("log")
    axes[1].set_title("OCR 文本长度分布 (对数 y 轴)")
    fig.tight_layout()
    fig.savefig(out / "ocr_length_dist.png")
    plt.close(fig)
    print(f"→ {out / 'ocr_length_dist.png'}")

    # ── 2. 预测分数分布 ──────────────────────────────────────
    score_col = "预测分数"
    if score_col not in df.columns:
        print(f"Warning: '{score_col}' not found, skipping score analysis.")
    else:
        scores = df[score_col].dropna()
        print()
        print("=" * 50)
        print("【预测分数统计】")
        print(scores.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_string())

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=dpi)
        axes[0].hist(scores, bins=50, color="coral", edgecolor="white", linewidth=0.3)
        axes[0].set_xlabel("预测分数")
        axes[0].set_ylabel("频数")
        axes[0].set_title("预测分数分布")

        h = axes[1].hist(scores, bins=50, color="coral", edgecolor="white", linewidth=0.3)
        # 在柱子上标注频数
        for rect, count in zip(h[2], h[0]):
            if count > 0:
                axes[1].annotate(int(count), xy=(rect.get_x() + rect.get_width() / 2, count),
                                 ha="center", va="bottom", fontsize=6)
        axes[1].set_xlabel("预测分数")
        axes[1].set_ylabel("频数")
        axes[1].set_title("预测分数分布 (带标注)")
        fig.tight_layout()
        fig.savefig(out / "score_dist.png")
        plt.close(fig)
        print(f"→ {out / 'score_dist.png'}")

    # ── 3. 长度 vs 分数散点图（如有） ────────────────────────
    if score_col in df.columns:
        valid = df[scores.notna()]
        valid_texts = valid["OCR文字结果"].fillna("").astype(str)
        valid_scores = valid[score_col]
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.scatter(valid_texts.str.len(), valid_scores, alpha=0.4, s=10, c="steelblue")
        ax.set_xlabel("OCR 文本长度")
        ax.set_ylabel("预测分数")
        ax.set_title("文本长度 vs 预测分数")
        fig.tight_layout()
        fig.savefig(out / "length_vs_score.png")
        plt.close(fig)
        print(f"→ {out / 'length_vs_score.png'}")

    print()
    print(f"所有分布图已保存至: {out.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分析 OCR 文本长度和预测分数分布")
    parser.add_argument("input", type=str, help="输入的 xlsx 文件路径")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录 (默认: {文件名}_dist)")
    args = parser.parse_args()
    analyze(args.input, args.output_dir)
