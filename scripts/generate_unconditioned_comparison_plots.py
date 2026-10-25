"""
generate_unconditioned_comparison_plots.py

Generates publication-quality comparative plots comparing:
  1. Unconditioned Default Models (Logistic Regression, XGBoost, Random Forest) trained directly on raw datasets without pipeline conditions.
  2. Our Evidence-Conditioned Pipeline Candidate Models.
  3. The Validation-Gated Multi-Model Ensemble.

Updates:
  - 17_candidate_vs_default_xgboost.png (Unconditioned Baselines vs Evidence Pipeline across cohorts)
  - 19_unconditioned_vs_evidence_pipeline_benchmark.png (Comprehensive 4-panel benchmark)
  - 06_candidate_vs_ensemble.png (Multi-cohort Candidate vs Ensemble head-to-head)

Saves all plots to evidence/final/submission/New/plots/ and synchronizes to new2/plots/.
"""

import json
import shutil
from pathlib import Path

def safe_sync(src: Path, dst: Path):
    src = src.resolve()
    dst = dst.resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(src, dst)
    except Exception:
        dst.write_bytes(src.read_bytes())
import matplotlib.pyplot as plt
import numpy as np

PALETTE = {
    "bg": "#FFFFFF",
    "card": "#F8FAFC",
    "border": "#E2E8F0",
    "text": "#0F172A",
    "subtext": "#475569",
    "grid": "#CBD5E1",
    "uncond_lr": "#64748B",       # Slate Gray
    "uncond_xgb": "#F59E0B",      # Amber
    "uncond_rf": "#06B6D4",       # Cyan
    "candidate": "#2563EB",       # Royal Blue
    "ensemble": "#10B981",        # Emerald Green
    "gain": "#8B5CF6",            # Purple
    "accent": "#F59E0B",          # Amber Accent
}

def apply_theme():
    plt.rcdefaults()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "text.color": PALETTE["text"],
        "axes.labelcolor": PALETTE["text"],
        "axes.titlecolor": PALETTE["text"],
        "xtick.color": PALETTE["text"],
        "ytick.color": PALETTE["text"],
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.labelweight": "semibold",
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "figure.facecolor": PALETTE["bg"],
        "axes.facecolor": PALETTE["card"],
        "axes.edgecolor": PALETTE["border"],
        "axes.linewidth": 1.0,
        "grid.color": PALETTE["grid"],
        "grid.linestyle": "--",
        "grid.alpha": 0.5,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

def generate_plots():
    apply_theme()
    
    comp_path = Path("evidence/final/submission/New/results/unconditioned_benchmark_comparison.json")
    with open(comp_path, encoding="utf-8") as f:
        data = json.load(f)
        
    out_dir = Path("evidence/final/submission/New/plots").resolve()
    new2_dir = Path("new2/plots").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    new2_dir.mkdir(parents=True, exist_ok=True)
    
    cohort_keys = list(data.keys())
    cohort_labels = [
        "Cohort A\n(Hancock Tabular)",
        "Cohort B\n(Lesion Tabular)",
        "Cohort C\n(Derm Image)",
        "Cohort D\n(Pathology Text)",
        "Cohort E\n(Trimodal)",
    ]
    
    lr_rocs = [data[c]["unconditioned_lr"]["roc_auc_mean"] for c in cohort_keys]
    xgb_rocs = [data[c]["unconditioned_xgb"]["roc_auc_mean"] for c in cohort_keys]
    rf_rocs = [data[c]["unconditioned_rf"]["roc_auc_mean"] for c in cohort_keys]
    cand_rocs = [data[c]["evidence_pipeline"]["roc_auc_mean"] for c in cohort_keys]
    ens_rocs = [data[c]["ensemble"]["roc_auc_mean"] for c in cohort_keys]
    
    lr_f1s = [data[c]["unconditioned_lr"]["f1_mean"] for c in cohort_keys]
    xgb_f1s = [data[c]["unconditioned_xgb"]["f1_mean"] for c in cohort_keys]
    rf_f1s = [data[c]["unconditioned_rf"]["f1_mean"] for c in cohort_keys]
    cand_f1s = [data[c]["evidence_pipeline"]["f1_mean"] for c in cohort_keys]
    ens_f1s = [data[c]["ensemble"]["f1_mean"] for c in cohort_keys]
    
    x = np.arange(len(cohort_keys))
    
    # -------------------------------------------------------------------------
    # PLOT 1: 17_candidate_vs_default_xgboost.png (Multi-Cohort Unconditioned vs Candidate)
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    w = 0.20
    
    # Panel 1: ROC-AUC
    ax1.bar(x - 1.5*w, lr_rocs, w, label="Unconditioned Logistic Regression", color=PALETTE["uncond_lr"], alpha=0.9, edgecolor="#1E293B")
    ax1.bar(x - 0.5*w, xgb_rocs, w, label="Unconditioned Default XGBoost", color=PALETTE["uncond_xgb"], alpha=0.9, edgecolor="#1E293B")
    ax1.bar(x + 0.5*w, rf_rocs, w, label="Unconditioned Default Random Forest", color=PALETTE["uncond_rf"], alpha=0.9, edgecolor="#1E293B")
    bars_cand = ax1.bar(x + 1.5*w, cand_rocs, w, label="Evidence-Conditioned Pipeline Candidate", color=PALETTE["candidate"], alpha=0.95, edgecolor="#1E293B")
    
    for i, c_roc in enumerate(cand_rocs):
        ax1.text(x[i] + 1.5*w, c_roc + 0.015, f"{c_roc:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color=PALETTE["candidate"])
        
    ax1.set_ylabel("Test ROC-AUC", fontweight="bold")
    ax1.set_ylim(0.0, 1.15)
    ax1.set_title("17. Evidence-Conditioned Pipeline Candidate vs. Unconditioned Standalone Models Across All Cohorts", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper right", framealpha=0.95, fontsize=8)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    
    # Panel 2: F1-Score
    ax2.bar(x - 1.5*w, lr_f1s, w, label="Unconditioned Logistic Regression", color=PALETTE["uncond_lr"], alpha=0.9, edgecolor="#1E293B")
    ax2.bar(x - 0.5*w, xgb_f1s, w, label="Unconditioned Default XGBoost", color=PALETTE["uncond_xgb"], alpha=0.9, edgecolor="#1E293B")
    ax2.bar(x + 0.5*w, rf_f1s, w, label="Unconditioned Default Random Forest", color=PALETTE["uncond_rf"], alpha=0.9, edgecolor="#1E293B")
    bars_cand_f1 = ax2.bar(x + 1.5*w, cand_f1s, w, label="Evidence-Conditioned Pipeline Candidate", color=PALETTE["candidate"], alpha=0.95, edgecolor="#1E293B")
    
    for i, c_f1 in enumerate(cand_f1s):
        ax2.text(x[i] + 1.5*w, c_f1 + 0.015, f"{c_f1:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color=PALETTE["candidate"])
        
    ax2.set_ylabel("F1-Score", fontweight="bold")
    ax2.set_ylim(0.0, 1.15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(cohort_labels, fontsize=8.5, fontweight="semibold")
    ax2.legend(loc="upper right", framealpha=0.95, fontsize=8)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    p17_path = out_dir / "17_candidate_vs_default_xgboost.png"
    plt.savefig(p17_path)
    plt.close()
    safe_sync(p17_path, new2_dir / p17_path.name)
    print(f"Saved and synchronized {p17_path.name}")
    
    # -------------------------------------------------------------------------
    # PLOT 2: 19_unconditioned_vs_evidence_pipeline_benchmark.png (Comprehensive 4-Panel)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    w = 0.22
    
    # Subplot A: ROC-AUC Comparison
    ax = axes[0, 0]
    ax.bar(x - 1.5*w, lr_rocs, w, label="Unconditioned LR", color=PALETTE["uncond_lr"], alpha=0.85)
    ax.bar(x - 0.5*w, xgb_rocs, w, label="Unconditioned XGB", color=PALETTE["uncond_xgb"], alpha=0.85)
    ax.bar(x + 0.5*w, cand_rocs, w, label="Evidence Pipeline", color=PALETTE["candidate"], alpha=0.95)
    ax.bar(x + 1.5*w, ens_rocs, w, label="Ensemble", color=PALETTE["ensemble"], alpha=0.9)
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.4, 1.1)
    ax.set_xticks(x)
    ax.set_xticklabels([c.split("\n")[0] for c in cohort_labels], fontsize=8)
    ax.set_title("A. Discrimination (ROC-AUC) Across Cohorts", fontsize=10, fontweight="bold")
    ax.legend(loc="lower right", fontsize=7.5)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    
    # Subplot B: F1-Score Comparison
    ax = axes[0, 1]
    ax.bar(x - 1.5*w, lr_f1s, w, label="Unconditioned LR", color=PALETTE["uncond_lr"], alpha=0.85)
    ax.bar(x - 0.5*w, xgb_f1s, w, label="Unconditioned XGB", color=PALETTE["uncond_xgb"], alpha=0.85)
    ax.bar(x + 0.5*w, cand_f1s, w, label="Evidence Pipeline", color=PALETTE["candidate"], alpha=0.95)
    ax.bar(x + 1.5*w, ens_f1s, w, label="Ensemble", color=PALETTE["ensemble"], alpha=0.9)
    ax.set_ylabel("F1-Score")
    ax.set_ylim(0.0, 1.1)
    ax.set_xticks(x)
    ax.set_xticklabels([c.split("\n")[0] for c in cohort_labels], fontsize=8)
    ax.set_title("B. Classification F1 (Evidence Imbalance Handling)", fontsize=10, fontweight="bold")
    ax.legend(loc="upper right", fontsize=7.5)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    
    # Subplot C: Brier Calibration Score (Lower is Better)
    ax = axes[1, 0]
    lr_brier = [data[c]["unconditioned_lr"]["brier_mean"] for c in cohort_keys]
    xgb_brier = [data[c]["unconditioned_xgb"]["brier_mean"] for c in cohort_keys]
    cand_brier = [data[c]["evidence_pipeline"]["brier_mean"] for c in cohort_keys]
    w_b = 0.25
    ax.bar(x - w_b, lr_brier, w_b, label="Unconditioned LR", color=PALETTE["uncond_lr"], alpha=0.85)
    ax.bar(x, xgb_brier, w_b, label="Unconditioned XGB", color=PALETTE["uncond_xgb"], alpha=0.85)
    ax.bar(x + w_b, cand_brier, w_b, label="Evidence Pipeline (Well-Calibrated)", color=PALETTE["candidate"], alpha=0.95)
    ax.set_ylabel("Brier Score Loss (Lower = Better)")
    ax.set_ylim(0.0, 0.30)
    ax.set_xticks(x)
    ax.set_xticklabels([c.split("\n")[0] for c in cohort_labels], fontsize=8)
    ax.set_title("C. Clinical Probability Calibration (Brier Score)", fontsize=10, fontweight="bold")
    ax.legend(loc="upper right", fontsize=7.5)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    
    # Subplot D: Relative F1 Gain of Evidence Pipeline over Unconditioned Baselines
    ax = axes[1, 1]
    f1_gains_lr = [(c - l) for c, l in zip(cand_f1s, lr_f1s)]
    f1_gains_xgb = [(c - xg) for c, xg in zip(cand_f1s, xgb_f1s)]
    w_g = 0.35
    ax.bar(x - w_g/2, f1_gains_lr, w_g, label="F1 Gain over Unconditioned LR", color=PALETTE["gain"], alpha=0.85)
    ax.bar(x + w_g/2, f1_gains_xgb, w_g, label="F1 Gain over Unconditioned XGB", color=PALETTE["accent"], alpha=0.85)
    ax.axhline(0, color="black", linestyle="-", linewidth=0.8)
    ax.set_ylabel("Absolute F1-Score Improvement")
    ax.set_xticks(x)
    ax.set_xticklabels([c.split("\n")[0] for c in cohort_labels], fontsize=8)
    ax.set_title("D. Evidence Conditioning Benefit (Net F1 Gain)", fontsize=10, fontweight="bold")
    ax.legend(loc="upper right", fontsize=7.5)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    
    plt.suptitle("19. Comprehensive Scientific Benchmark: Unconditioned Baselines vs. Evidence-Conditioned Pipeline vs. Ensemble", fontsize=12, fontweight="bold", y=0.995)
    plt.tight_layout()
    p19_path = out_dir / "19_unconditioned_vs_evidence_pipeline_benchmark.png"
    plt.savefig(p19_path)
    plt.close()
    safe_sync(p19_path, new2_dir / p19_path.name)
    print(f"Saved and synchronized {p19_path.name}")
    
    # -------------------------------------------------------------------------
    # PLOT 3: 06_candidate_vs_ensemble.png (Multi-Cohort Candidate vs Ensemble Head-to-Head)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    w = 0.35
    
    ax.bar(x - w/2, cand_rocs, w, label="Evidence-Conditioned Candidate Pipeline", color=PALETTE["candidate"], alpha=0.92, edgecolor="#1E293B")
    ax.bar(x + w/2, ens_rocs, w, label="Validation-Performance-Weighted Ensemble", color=PALETTE["ensemble"], alpha=0.92, edgecolor="#1E293B")
    
    for i in range(len(cohort_keys)):
        cr = cand_rocs[i]
        er = ens_rocs[i]
        ax.text(x[i] - w/2, cr + 0.012, f"{cr:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold", color=PALETTE["candidate"])
        ax.text(x[i] + w/2, er + 0.012, f"{er:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold", color=PALETTE["ensemble"])
        
    ax.set_ylabel("Test ROC-AUC", fontweight="bold")
    ax.set_ylim(0.0, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(cohort_labels, fontsize=8.5, fontweight="semibold")
    ax.set_title("06. Multi-Cohort Head-to-Head: Evidence-Conditioned Candidate vs. Ensemble Baseline\n(Computed strictly from canonical predictions across seeds [42, 100, 2026])", fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.95, fontsize=8.5)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    p06_path = out_dir / "06_candidate_vs_ensemble.png"
    plt.savefig(p06_path)
    plt.close()
    safe_sync(p06_path, new2_dir / p06_path.name)
    print(f"Saved and synchronized {p06_path.name}")

if __name__ == "__main__":
    generate_plots()
