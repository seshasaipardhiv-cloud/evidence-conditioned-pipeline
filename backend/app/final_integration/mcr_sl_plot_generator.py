"""
mcr_sl_plot_generator.py

Generates all MCR-SL real multimodal experiment plots.
All plots are derived strictly from canonical predictions/results files.
No hardcoded metric arrays.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve

logger = logging.getLogger(__name__)

PLOTS_DIR = Path("evidence/final/submission/New/plots")
RESULTS_PATH = Path("evidence/final/submission/New/results/mcr_sl_multimodal_results.json")
PREDS_PATH = Path("evidence/final/submission/New/predictions/Cohort_MCR_SL_Real_Multimodal_predictions.jsonl")

ARCH_COLORS = {
    "image_only": "#3B82F6",           # Blue
    "context_only": "#10B981",          # Green
    "concatenation_fusion": "#8B5CF6",  # Purple
    "late_fusion": "#F59E0B",           # Amber
    "cross_attention_fusion": "#EF4444",# Red
    "gated_fusion": "#EC4899",          # Pink
}

ARCH_LABELS = {
    "image_only": "Image-Only\n(ResNet-18)",
    "context_only": "Context-Only\n(PubMedBERT)",
    "concatenation_fusion": "Concatenation\nFusion",
    "late_fusion": "Late Fusion",
    "cross_attention_fusion": "Cross-Attention\nFusion",
    "gated_fusion": "Gated Fusion",
}


def _load_results() -> Optional[Dict[str, Any]]:
    if not RESULTS_PATH.exists():
        logger.warning(f"Results file not found: {RESULTS_PATH}")
        return None
    with open(RESULTS_PATH, "r") as f:
        return json.load(f)


def _load_predictions() -> List[Dict[str, Any]]:
    if not PREDS_PATH.exists():
        return []
    records = []
    with open(PREDS_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def plot_fusion_comparison_bar(results: Dict[str, Any]) -> Path:
    """Bar chart comparing ROC-AUC and F1 across all 6 architectures (with error bars)."""
    exp = results.get("experiment_results", {})
    if not exp:
        raise ValueError("No experiment results found.")

    arch_ids = ["image_only", "context_only", "concatenation_fusion",
                "late_fusion", "cross_attention_fusion", "gated_fusion"]

    roc_means = [exp[a]["multi_seed_summary"]["roc_auc_mean"] for a in arch_ids if a in exp]
    roc_stds = [exp[a]["multi_seed_summary"]["roc_auc_std"] for a in arch_ids if a in exp]
    f1_means = [exp[a]["multi_seed_summary"]["f1_mean"] for a in arch_ids if a in exp]
    f1_stds = [exp[a]["multi_seed_summary"]["f1_std"] for a in arch_ids if a in exp]
    labels = [ARCH_LABELS.get(a, a) for a in arch_ids if a in exp]
    colors = [ARCH_COLORS.get(a, "#999999") for a in arch_ids if a in exp]

    x = np.arange(len(labels))
    width = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor("#0F172A")

    for ax in axes:
        ax.set_facecolor("#1E293B")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color("#334155")
        ax.spines["left"].set_color("#334155")
        ax.tick_params(colors="#CBD5E1")
        ax.yaxis.label.set_color("#CBD5E1")
        ax.xaxis.label.set_color("#CBD5E1")
        ax.title.set_color("#F1F5F9")
        ax.grid(axis="y", alpha=0.3, color="#334155")

    # ROC-AUC
    bars = axes[0].bar(x, roc_means, width=0.6, yerr=roc_stds,
                       color=colors, alpha=0.85, capsize=6,
                       error_kw={"ecolor": "#F8FAFC", "capthick": 2, "elinewidth": 1.5})
    axes[0].axhline(0.5, color="#FCD34D", linewidth=1.5, linestyle="--", alpha=0.8, label="Chance (0.5)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=9, color="#CBD5E1")
    axes[0].set_ylabel("ROC-AUC", fontsize=11)
    axes[0].set_title("ROC-AUC by Architecture\n(REAL MCR-SL — Multi-Seed, Subject-Isolated)", fontsize=12)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].legend(fontsize=9, labelcolor="#CBD5E1", facecolor="#1E293B", edgecolor="#334155")
    for bar, val, std in zip(bars, roc_means, roc_stds):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.02,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=8, color="#F1F5F9", fontweight="bold")

    # F1
    bars2 = axes[1].bar(x, f1_means, width=0.6, yerr=f1_stds,
                        color=colors, alpha=0.85, capsize=6,
                        error_kw={"ecolor": "#F8FAFC", "capthick": 2, "elinewidth": 1.5})
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=9, color="#CBD5E1")
    axes[1].set_ylabel("F1 Score", fontsize=11)
    axes[1].set_title("F1 Score by Architecture\n(REAL MCR-SL — Multi-Seed, Subject-Isolated)", fontsize=12)
    axes[1].set_ylim(0.0, 1.0)
    for bar, val, std in zip(bars2, f1_means, f1_stds):
        label = f"{val:.3f}"
        axes[1].text(bar.get_x() + bar.get_width() / 2, max(bar.get_height(), 0) + std + 0.02,
                     label, ha="center", va="bottom", fontsize=8, color="#F1F5F9", fontweight="bold")

    plt.suptitle(
        "MCR-SL Real Multimodal Benchmark — Architecture Comparison\n"
        "234 lesions (42 malignant / 192 non-malignant) | Subject-Level Isolated Splits | Seeds [42, 100, 2026]",
        color="#F1F5F9", fontsize=11, y=1.02,
    )
    plt.tight_layout()

    out = PLOTS_DIR / "mcr_sl_fusion_comparison.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="#0F172A")
    plt.close()
    logger.info(f"Plot saved: {out}")
    return out


def plot_per_seed_stability(results: Dict[str, Any]) -> Path:
    """Line plot showing per-seed ROC-AUC stability per architecture."""
    exp = results.get("experiment_results", {})
    seeds = [42, 100, 2026]
    arch_ids = ["image_only", "context_only", "concatenation_fusion",
                "late_fusion", "cross_attention_fusion", "gated_fusion"]

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#1E293B")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#334155")
    ax.spines["left"].set_color("#334155")
    ax.tick_params(colors="#CBD5E1")
    ax.grid(alpha=0.3, color="#334155")

    for arch_id in arch_ids:
        if arch_id not in exp:
            continue
        seed_rocs = []
        for run in exp[arch_id]["seed_runs"]:
            seed_rocs.append(run["metrics"]["roc_auc"])
        color = ARCH_COLORS.get(arch_id, "#999999")
        label = ARCH_LABELS.get(arch_id, arch_id).replace("\n", " ")
        ax.plot(seeds, seed_rocs, "o-", color=color, linewidth=2, markersize=8, label=label)

    ax.axhline(0.5, color="#FCD34D", linewidth=1.5, linestyle="--", alpha=0.7, label="Chance")
    ax.set_xticks(seeds)
    ax.set_xticklabels([str(s) for s in seeds], color="#CBD5E1", fontsize=11)
    ax.set_xlabel("Seed", color="#CBD5E1", fontsize=12)
    ax.set_ylabel("ROC-AUC", color="#CBD5E1", fontsize=12)
    ax.set_title(
        "Per-Seed ROC-AUC Stability — REAL MCR-SL Multimodal Benchmark\n"
        "Subject-Level Isolated Splits | 3 Random Seeds",
        color="#F1F5F9", fontsize=12,
    )
    ax.set_ylim(0.0, 1.0)
    ax.legend(fontsize=9, labelcolor="#CBD5E1", facecolor="#1E293B",
              edgecolor="#334155", loc="upper right")

    plt.tight_layout()
    out = PLOTS_DIR / "mcr_sl_per_seed_stability.png"
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="#0F172A")
    plt.close()
    logger.info(f"Plot saved: {out}")
    return out


def plot_roc_pr_curves(predictions: List[Dict[str, Any]]) -> Path:
    """ROC and PR curves per architecture (using seed 42 predictions)."""
    arch_ids = ["image_only", "context_only", "concatenation_fusion",
                "late_fusion", "cross_attention_fusion", "gated_fusion"]
    arch_name_map = {exp_id: ARCH_LABELS.get(exp_id, exp_id) for exp_id in arch_ids}

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("#0F172A")
    for ax in axes:
        ax.set_facecolor("#1E293B")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color("#334155")
        ax.spines["left"].set_color("#334155")
        ax.tick_params(colors="#CBD5E1")
        ax.grid(alpha=0.25, color="#334155")

    # Group predictions by model_name and seed=42
    from collections import defaultdict
    grouped: dict = defaultdict(lambda: {"y_true": [], "y_prob": []})
    for pred in predictions:
        if pred.get("seed") == 42:
            grouped[pred["model_name"]]["y_true"].append(pred["true_label"])
            grouped[pred["model_name"]]["y_prob"].append(pred["predicted_probability"])

    # Map model names back to arch_id
    full_name_to_arch = {
        "Image-Only (ResNet-18)": "image_only",
        "Clinical-Context-Only (PubMedBERT)": "context_only",
        "Feature Concatenation Fusion (ResNet-18 + PubMedBERT)": "concatenation_fusion",
        "Late Fusion (Probability-Weighted)": "late_fusion",
        "Cross-Attention Fusion (Candidate Selected)": "cross_attention_fusion",
        "Gated Multimodal Fusion": "gated_fusion",
    }

    for model_name, data in grouped.items():
        y_true = np.array(data["y_true"])
        y_prob = np.array(data["y_prob"])
        if len(np.unique(y_true)) < 2:
            continue

        arch_id = full_name_to_arch.get(model_name, "image_only")
        color = ARCH_COLORS.get(arch_id, "#999")
        label_short = ARCH_LABELS.get(arch_id, model_name).replace("\n", " ")

        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, color=color, linewidth=2, alpha=0.9,
                     label=f"{label_short} (AUC={roc_auc:.3f})")

        # PR
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(rec, prec)
        axes[1].plot(rec, prec, color=color, linewidth=2, alpha=0.9,
                     label=f"{label_short} (AUC={pr_auc:.3f})")

    # ROC diagonal
    axes[0].plot([0, 1], [0, 1], "--", color="#FCD34D", alpha=0.6, linewidth=1.2, label="Chance")
    axes[0].set_xlabel("False Positive Rate", color="#CBD5E1", fontsize=11)
    axes[0].set_ylabel("True Positive Rate", color="#CBD5E1", fontsize=11)
    axes[0].set_title("ROC Curves — REAL MCR-SL (Seed 42)", color="#F1F5F9", fontsize=12)
    axes[0].legend(fontsize=8, labelcolor="#CBD5E1", facecolor="#1E293B", edgecolor="#334155")

    # PR baseline
    pos_rate = 42 / 234
    axes[1].axhline(pos_rate, color="#FCD34D", alpha=0.6, linewidth=1.2, linestyle="--",
                    label=f"Baseline Prevalence ({pos_rate:.2f})")
    axes[1].set_xlabel("Recall", color="#CBD5E1", fontsize=11)
    axes[1].set_ylabel("Precision", color="#CBD5E1", fontsize=11)
    axes[1].set_title("PR Curves — REAL MCR-SL (Seed 42)", color="#F1F5F9", fontsize=12)
    axes[1].legend(fontsize=8, labelcolor="#CBD5E1", facecolor="#1E293B", edgecolor="#334155")

    plt.suptitle(
        "ROC & PR Curves — REAL MCR-SL Dataset | Subject-Level Isolated Splits",
        color="#F1F5F9", fontsize=12, y=1.01,
    )
    plt.tight_layout()
    out = PLOTS_DIR / "mcr_sl_roc_pr_curves.png"
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="#0F172A")
    plt.close()
    logger.info(f"Plot saved: {out}")
    return out


def generate_all_mcr_sl_plots() -> List[Path]:
    results = _load_results()
    predictions = _load_predictions()
    generated = []

    if results:
        generated.append(plot_fusion_comparison_bar(results))
        generated.append(plot_per_seed_stability(results))
    if predictions:
        generated.append(plot_roc_pr_curves(predictions))

    return generated
