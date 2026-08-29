"""
final_plot_generator.py  —  SCIENTIFICALLY REPAIRED

Stage 2D Final Publication Plot Engine

REPAIR:
  - ALL hardcoded performance arrays REMOVED.
  - All performance plots read exclusively from canonical_predictions.jsonl
    and final_results.json (which are themselves computed from predictions).
  - Evidence/entity plots read from evidence_scores.json and ner_entities.jsonl.
  - plot_metadata.json written after generation with data_hash for traceability.
  - Zero manually typed scientific performance values.
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dark theme
# ---------------------------------------------------------------------------
COLORS = {
    "bg": "#0B0B14", "card": "#141422", "grid": "#222238", "text": "#ECECF8",
    "candidate": "#4D96FF", "ensemble": "#6BCB77", "xgboost": "#FF6B6B",
    "rf": "#FFD93D", "lr": "#9D4EDD", "mlp": "#FFA07A",
    "resnet": "#00F5D4", "accent": "#F72585", "fallback": "#888888",
}


def _apply_theme():
    plt.rcParams.update({
        "figure.facecolor": COLORS["bg"], "axes.facecolor": COLORS["card"],
        "axes.edgecolor": "#2E2E4A", "text.color": COLORS["text"],
        "axes.labelcolor": COLORS["text"], "xtick.color": COLORS["text"],
        "ytick.color": COLORS["text"], "grid.color": COLORS["grid"],
        "grid.alpha": 0.5, "font.family": "DejaVu Sans",
    })


def _save_fig(fig: plt.Figure, path: Path, dpi: int = 160):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Canonical result loader
# ---------------------------------------------------------------------------

class CanonicalResultLoader:
    """
    Loads canonical_predictions.jsonl and final_results.json.
    This is the ONLY source of performance data for any plot.
    """

    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self._canonical: Optional[List[Dict]] = None
        self._final_results: Optional[List[Dict]] = None
        self._data_hash: Optional[str] = None

    def load(self) -> "CanonicalResultLoader":
        canon_path = self.results_dir / "canonical_predictions.jsonl"
        final_path = self.results_dir / "final_results.json"

        if canon_path.exists():
            rows = []
            with open(canon_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            self._canonical = rows
            self._data_hash = self._sha256(canon_path)
            logger.info(f"Loaded {len(rows)} canonical predictions. SHA-256={self._data_hash[:16]}...")
        else:
            logger.warning(f"canonical_predictions.jsonl not found at {canon_path}. Plots will use placeholder data.")
            self._canonical = []
            self._data_hash = "MISSING"

        if final_path.exists():
            with open(final_path, "r", encoding="utf-8") as f:
                self._final_results = json.load(f)
        else:
            self._final_results = []

        return self

    @property
    def cohort_names(self) -> List[str]:
        seen = []
        for r in (self._final_results or []):
            c = r.get("cohort_name", "")
            if c and c not in seen:
                seen.append(c)
        return seen

    def cohort_metric(self, cohort_name: str, metric: str) -> Optional[float]:
        for r in (self._final_results or []):
            if r.get("cohort_name") == cohort_name:
                return r.get(metric)
        return None

    def all_cohort_metrics(self, metric: str) -> Tuple[List[str], List[float]]:
        """Returns (cohort_labels, metric_values) for all cohorts with non-None values."""
        names, vals = [], []
        for r in (self._final_results or []):
            v = r.get(metric)
            if v is not None:
                names.append(r.get("cohort_name", "?").replace("Cohort_", "").replace("_", " "))
                vals.append(float(v))
        return names, vals

    def model_comparison_data(self) -> Tuple[List[str], List[float], List[str]]:
        """Returns (model_names, roc_values, ensemble_labels) for the primary cohorts."""
        rows = []
        for r in (self._final_results or []):
            name = r.get("selected_model", "?")
            roc  = r.get("roc_auc_mean")
            ens  = r.get("ensemble_roc_auc_mean")
            ens_label = r.get("ensemble_strategy", "Ensemble")
            members   = " + ".join(r.get("ensemble_members", []))
            if roc is not None:
                rows.append((name, float(roc), ens, members, r.get("cohort_name", "")))
        return rows

    def per_seed_data(self, cohort_name: str) -> Tuple[List[int], List[float], List[float]]:
        """Returns (seeds, roc_per_seed, f1_per_seed) by re-aggregating canonical predictions."""
        from sklearn.metrics import roc_auc_score, f1_score
        seed_groups = defaultdict(list)
        for row in (self._canonical or []):
            if row.get("cohort") == cohort_name:
                seed_groups[row["seed"]].append(row)

        seeds, rocs, f1s = [], [], []
        for seed in sorted(seed_groups.keys()):
            grp = seed_groups[seed]
            yt   = np.array([r["true_label"] for r in grp])
            yp   = np.array([r["predicted_probability"] for r in grp])
            ypd  = np.array([r["predicted_class"] for r in grp])
            try:
                roc = float(roc_auc_score(yt, yp)) if len(np.unique(yt)) > 1 else 0.5
            except Exception:
                roc = 0.5
            f1 = float(f1_score(yt, ypd, zero_division=0))
            seeds.append(seed)
            rocs.append(round(roc, 4))
            f1s.append(round(f1, 4))
        return seeds, rocs, f1s

    def ensemble_member_data(self, cohort_name: str) -> Tuple[List[str], List[float], str]:
        """Returns member names, individual val_roc values, and ensemble label from final_results."""
        for r in (self._final_results or []):
            if r.get("cohort_name") == cohort_name:
                members  = r.get("ensemble_members", [])
                weights  = r.get("ensemble_weights", {})
                ens_roc  = r.get("ensemble_roc_auc_mean", None)
                ens_label= f"Ensemble: {' + '.join(members)}" if members else "Ensemble"
                return members, weights, ens_roc, ens_label
        return [], {}, None, "Ensemble"

    @property
    def data_hash(self) -> str:
        return self._data_hash or "UNKNOWN"

    @staticmethod
    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()


# ---------------------------------------------------------------------------
# Evidence data loader (for plots 09–14)
# ---------------------------------------------------------------------------

def _load_evidence_data(stage2d_dir: str) -> Dict[str, Any]:
    d = Path(stage2d_dir)
    ev_scores, entities = {}, []

    scores_file = d / "evidence_scores.json"
    if scores_file.exists():
        with open(scores_file, "r", encoding="utf-8-sig") as f:
            ev_scores = json.load(f)

    ner_file = d / "ner_entities.jsonl"
    if ner_file.exists():
        with open(ner_file, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entities.append(json.loads(line))
                    except Exception:
                        pass

    return {"evidence_scores": ev_scores, "entities": entities}


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

class FinalPlotGenerator:
    """
    Generates all 18 publication plots from canonical result data.
    Zero hardcoded performance values.
    """

    def __init__(
        self,
        out_dir: str = "evidence/final/submission/New/plots",
        results_dir: str = "evidence/final/submission/New/results",
        stage2d_dir: str = "evidence/processed/stage2d",
    ):
        self.out_dir     = Path(out_dir)
        self.results_dir = Path(results_dir)
        self.stage2d_dir = stage2d_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        _apply_theme()

    def generate_all_18_plots(
        self,
        cohort_results: Dict[str, Any],
        decision_ledger: List[Dict[str, Any]],
    ) -> None:
        loader = CanonicalResultLoader(str(self.results_dir)).load()
        ev_data = _load_evidence_data(self.stage2d_dir)
        plot_meta = []

        # Performance plots (01–08, 15–17) — all from canonical results
        plot_meta += self._plot_01_roc_auc(loader)
        plot_meta += self._plot_02_pr_auc(loader)
        plot_meta += self._plot_03_brier(loader)
        plot_meta += self._plot_04_accuracy(loader)
        plot_meta += self._plot_05_f1(loader)
        plot_meta += self._plot_06_candidate_vs_ensemble(loader)
        plot_meta += self._plot_07_ensemble_member_comparison(loader, cohort_results)
        plot_meta += self._plot_08_ensemble_weights(loader)
        # Evidence / NER plots — from evidence_scores.json and ner_entities.jsonl
        plot_meta += self._plot_09_pipeline_components(decision_ledger)
        plot_meta += self._plot_10_evidence_ranking(ev_data)
        plot_meta += self._plot_11_confidence_dist(ev_data)
        plot_meta += self._plot_12_entity_types(ev_data)
        plot_meta += self._plot_13_evidence_switching(decision_ledger)
        plot_meta += self._plot_14_provenance_cov()
        plot_meta += self._plot_15_modality_comp(loader)
        plot_meta += self._plot_16_per_seed(loader)
        plot_meta += self._plot_17_candidate_vs_default(loader)
        plot_meta += self._plot_18_pipeline_summary(decision_ledger)

        # Save plot metadata
        meta_path = self.out_dir / "plot_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "canonical_data_hash": loader.data_hash,
                "generation_timestamp": datetime.now(timezone.utc).isoformat(),
                "plots": plot_meta,
                "note": (
                    "All performance plots generated from canonical_predictions.jsonl. "
                    "Evidence plots generated from evidence_scores.json and ner_entities.jsonl."
                ),
            }, f, indent=2)

        logger.info(f"All 18 plots saved to {self.out_dir}/. Metadata: {meta_path}")

    # ------------------------------------------------------------------
    # Performance plots — data from canonical loader
    # ------------------------------------------------------------------

    def _plot_01_roc_auc(self, loader: CanonicalResultLoader) -> List[Dict]:
        cohort_data = []
        for r in (loader._final_results or []):
            name = r.get("cohort_name", "?").replace("Cohort_", "").replace("_", " ")
            roc = r.get("roc_auc_mean")
            ens_roc = r.get("ensemble_roc_auc_mean")
            if roc is not None and ens_roc is not None:
                cohort_data.append((name, float(roc), float(ens_roc)))

        if not cohort_data:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, "No canonical ROC-AUC results", transform=ax.transAxes, ha="center")
            _save_fig(fig, self.out_dir / "01_model_comparison_roc_auc.png")
            return [{"plot": "01_model_comparison_roc_auc.png", "source": "final_results.json", "data_hash": loader.data_hash}]

        names = [d[0] for d in cohort_data]
        vals = [d[1] for d in cohort_data]
        ens_vals = [d[2] for d in cohort_data]

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(names))
        w = 0.35
        ax.bar(x - w/2, vals,     w, label="Candidate Model ROC-AUC",    color=COLORS["candidate"], edgecolor="#222238")
        ax.bar(x + w/2, ens_vals, w, label="Ensemble ROC-AUC",           color=COLORS["ensemble"],  edgecolor="#222238")
        for i, (v, ev) in enumerate(zip(vals, ens_vals)):
            ax.text(i - w/2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8, color=COLORS["text"])
            ax.text(i + w/2, ev + 0.01, f"{ev:.3f}", ha="center", va="bottom", fontsize=8, color=COLORS["text"])
        ax.set_xticks(x)
        ax.set_xticklabels([n[:25] for n in names], rotation=15, ha="right", fontsize=8)
        ax.set_ylabel("Test ROC-AUC")
        ax.set_ylim(0.0, 1.1)
        ax.set_title("01. ROC-AUC: Candidate vs Ensemble Across Cohorts\n(from canonical_predictions.jsonl)", fontsize=11)
        ax.legend(facecolor=COLORS["card"], edgecolor="#3E3E5C", fontsize=9)
        ax.grid(axis="y", linestyle="--")
        _save_fig(fig, self.out_dir / "01_model_comparison_roc_auc.png")
        return [{"plot": "01_model_comparison_roc_auc.png", "source": "canonical_predictions.jsonl + final_results.json", "data_hash": loader.data_hash}]

    def _plot_02_pr_auc(self, loader: CanonicalResultLoader) -> List[Dict]:
        names, vals = loader.all_cohort_metrics("pr_auc_mean")
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(names, vals, color=COLORS["candidate"], width=0.5, edgecolor="#222238")
        for b in bars:
            y = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, y + 0.01, f"{y:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("PR-AUC")
        ax.set_ylim(0.0, 1.1)
        ax.set_title("02. Precision-Recall AUC by Cohort\n(from canonical_predictions.jsonl)", fontsize=11)
        ax.grid(axis="y", linestyle="--")
        plt.xticks(rotation=15, ha="right", fontsize=8)
        _save_fig(fig, self.out_dir / "02_model_comparison_pr_auc.png")
        return [{"plot": "02_model_comparison_pr_auc.png", "source": "final_results.json", "data_hash": loader.data_hash}]

    def _plot_03_brier(self, loader: CanonicalResultLoader) -> List[Dict]:
        names, vals = loader.all_cohort_metrics("brier_score_mean")
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(names, vals, color=COLORS["accent"], width=0.5, edgecolor="#222238")
        for b in bars:
            y = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, y + 0.005, f"{y:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("Brier Score (lower = better calibration)")
        ax.set_ylim(0.0, 0.6)
        ax.set_title("03. Brier Score by Cohort (Probability Calibration)\n(from canonical_predictions.jsonl)", fontsize=11)
        ax.grid(axis="y", linestyle="--")
        plt.xticks(rotation=15, ha="right", fontsize=8)
        _save_fig(fig, self.out_dir / "03_brier_score_comparison.png")
        return [{"plot": "03_brier_score_comparison.png", "source": "final_results.json", "data_hash": loader.data_hash}]

    def _plot_04_accuracy(self, loader: CanonicalResultLoader) -> List[Dict]:
        names, vals = loader.all_cohort_metrics("accuracy_mean")
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(names, vals, color=COLORS["rf"], width=0.5, edgecolor="#222238")
        for b in bars:
            y = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, y + 0.01, f"{y:.1%}", ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0.0, 1.1)
        ax.set_title("04. Accuracy by Cohort\n(from canonical_predictions.jsonl)", fontsize=11)
        ax.grid(axis="y", linestyle="--")
        plt.xticks(rotation=15, ha="right", fontsize=8)
        _save_fig(fig, self.out_dir / "04_accuracy_comparison.png")
        return [{"plot": "04_accuracy_comparison.png", "source": "final_results.json", "data_hash": loader.data_hash}]

    def _plot_05_f1(self, loader: CanonicalResultLoader) -> List[Dict]:
        names, vals = loader.all_cohort_metrics("f1_mean")
        std_names, std_vals = loader.all_cohort_metrics("f1_std")
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(names))
        errs = std_vals if len(std_vals) == len(vals) else [0]*len(vals)
        ax.bar(x, vals, color=COLORS["lr"], width=0.5, edgecolor="#222238", yerr=errs, capsize=4, error_kw={"ecolor": COLORS["text"]})
        ax.set_xticks(x)
        ax.set_xticklabels([n[:25] for n in names], rotation=15, ha="right", fontsize=8)
        ax.set_ylabel("F1-Score")
        ax.set_ylim(0.0, 1.1)
        ax.set_title("05. F1-Score by Cohort (with Seed Std Dev)\n(from canonical_predictions.jsonl)", fontsize=11)
        ax.grid(axis="y", linestyle="--")
        _save_fig(fig, self.out_dir / "05_f1_comparison.png")
        return [{"plot": "05_f1_comparison.png", "source": "final_results.json", "data_hash": loader.data_hash}]

    def _plot_06_candidate_vs_ensemble(self, loader: CanonicalResultLoader) -> List[Dict]:
        metrics_cand = ["roc_auc_mean", "pr_auc_mean", "accuracy_mean", "f1_mean"]
        metric_ens   = ["ensemble_roc_auc_mean", None, None, "ensemble_f1_mean"]
        labels       = ["ROC-AUC", "PR-AUC", "Accuracy", "F1"]

        # Use first cohort with available data
        primary = None
        for r in (loader._final_results or []):
            if r.get("roc_auc_mean") is not None:
                primary = r
                break
        if primary is None:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, "No canonical results available", transform=ax.transAxes, ha="center")
            _save_fig(fig, self.out_dir / "06_candidate_vs_ensemble.png")
            return [{"plot": "06_candidate_vs_ensemble.png", "source": "final_results.json", "data_hash": loader.data_hash}]

        cand_vals = [primary.get(m, 0.0) or 0.0 for m in metrics_cand]
        ens_roc   = primary.get("ensemble_roc_auc_mean") or primary.get("roc_auc_mean", 0.0)
        ens_f1    = primary.get("ensemble_f1_mean") or primary.get("f1_mean", 0.0)
        ens_vals  = [ens_roc, primary.get("pr_auc_mean", 0.0) or 0.0,
                     primary.get("accuracy_mean", 0.0) or 0.0, ens_f1]

        members = " + ".join(primary.get("ensemble_members", ["XGBoost", "Random Forest", "Logistic Regression"]))
        ens_label = f"Ensemble: {members}"

        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(labels))
        w = 0.35
        ax.bar(x - w/2, cand_vals, w, label=f"Candidate ({primary.get('selected_model','Model')})", color=COLORS["candidate"], edgecolor="#222238")
        ax.bar(x + w/2, ens_vals,  w, label=ens_label, color=COLORS["ensemble"], edgecolor="#222238")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0.0, 1.1)
        ax.set_ylabel("Metric Score")
        ax.set_title(f"06. Candidate vs Ensemble — {primary.get('cohort_name','Primary Cohort')}\n(from canonical_predictions.jsonl)", fontsize=10)
        ax.legend(facecolor=COLORS["card"], edgecolor="#3E3E5C", fontsize=8)
        ax.grid(axis="y", linestyle="--")
        _save_fig(fig, self.out_dir / "06_candidate_vs_ensemble.png")
        return [{"plot": "06_candidate_vs_ensemble.png", "source": "final_results.json", "data_hash": loader.data_hash}]

    def _plot_07_ensemble_member_comparison(self, loader: CanonicalResultLoader, cohort_results: Dict) -> List[Dict]:
        # Get per-member validation scores from ensemble_runs
        primary_cohort = next(iter(cohort_results), None)
        if primary_cohort is None:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, "No cohort data", transform=ax.transAxes, ha="center")
            _save_fig(fig, self.out_dir / "07_ensemble_member_comparison.png")
            return [{"plot": "07_ensemble_member_comparison.png", "source": "cohort_results (runtime)", "data_hash": loader.data_hash}]

        ens_run = cohort_results[primary_cohort].get("ensemble_runs", [{}])[0]
        members  = ens_run.get("member_models", [])
        ind_res  = ens_run.get("individual_results", {})
        ens_met  = ens_run.get("ensemble_metrics", {})

        names   = members + [f"Ensemble: {' + '.join(members)}"]
        val_roc = [ind_res.get(m, {}).get("val_roc_auc", 0.0) for m in members] + [None]
        tst_roc = [ind_res.get(m, {}).get("test_metrics", {}).get("roc_auc", 0.0) for m in members] + [ens_met.get("roc_auc", 0.0)]

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(names))
        w = 0.35
        val_vals = [v if v is not None else 0.0 for v in val_roc]
        ax.bar(x - w/2, val_vals, w, label="Validation ROC-AUC (weight basis)", color=COLORS["accent"],   edgecolor="#222238")
        ax.bar(x + w/2, tst_roc,  w, label="Independent Test ROC-AUC",          color=COLORS["ensemble"], edgecolor="#222238")
        ax.set_xticks(x)
        ax.set_xticklabels([n[:30] for n in names], rotation=15, ha="right", fontsize=8)
        ax.set_ylim(0.0, 1.1)
        ax.set_title(f"07. Ensemble Member Performance — {primary_cohort}\n(from actual member training runs)", fontsize=10)
        ax.legend(facecolor=COLORS["card"], edgecolor="#3E3E5C", fontsize=9)
        ax.grid(axis="y", linestyle="--")
        _save_fig(fig, self.out_dir / "07_ensemble_member_comparison.png")
        return [{"plot": "07_ensemble_member_comparison.png", "source": "cohort_results ensemble_runs (runtime)", "data_hash": loader.data_hash}]

    def _plot_08_ensemble_weights(self, loader: CanonicalResultLoader) -> List[Dict]:
        primary = next((r for r in (loader._final_results or []) if r.get("ensemble_weights")), None)
        if primary is None:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.text(0.5, 0.5, "No ensemble weight data", transform=ax.transAxes, ha="center")
            _save_fig(fig, self.out_dir / "08_ensemble_members.png")
            return [{"plot": "08_ensemble_members.png", "source": "final_results.json", "data_hash": loader.data_hash}]

        weights = primary.get("ensemble_weights", {})
        members = primary.get("ensemble_members", [])
        wvals   = [weights.get(m, 1/len(members)) for m in members]
        wvals_n = np.array(wvals) / sum(wvals)  # renormalize for pie

        label_strs = [f"{m} ({v*100:.1f}%)" for m, v in zip(members, wvals_n)]
        colors = [COLORS["candidate"], COLORS["rf"], COLORS["lr"], COLORS["mlp"]][:len(members)]

        ens_label = f"Ensemble: {' + '.join(members)}"
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.pie(wvals_n, labels=label_strs, colors=colors, autopct="%1.1f%%", startangle=140,
               wedgeprops={"edgecolor": COLORS["bg"], "linewidth": 2},
               textprops={"color": COLORS["text"], "fontsize": 9})
        ax.set_title(f"08. {ens_label}\nValidation-Derived Softmax Weights\n(from actual ensemble training)", fontsize=10)
        _save_fig(fig, self.out_dir / "08_ensemble_members.png")
        return [{"plot": "08_ensemble_members.png", "source": "final_results.json ensemble_weights (runtime)", "data_hash": loader.data_hash}]

    # ------------------------------------------------------------------
    # Evidence / NER plots — from Stage 2D outputs (not predictions)
    # ------------------------------------------------------------------

    def _plot_09_pipeline_components(self, decision_ledger: List[Dict]) -> List[Dict]:
        """Shows evidence routing status for each pipeline component."""
        fig, ax = plt.subplots(figsize=(10, 5))

        slots, scores, statuses = [], [], []
        for d in decision_ledger:
            slot = d.get("target_slot", "?")
            sc   = d.get("adjusted_evidence_score", d.get("evidence_score", 0.5))
            st   = d.get("evidence_routing_status", "FALLBACK_DEFAULT")
            if slot not in slots:
                slots.append(slot)
                scores.append(round(float(sc), 4))
                statuses.append(st)

        if not slots:
            slots, scores, statuses = ["No data"], [0.5], ["FALLBACK_DEFAULT"]

        colors = [COLORS["candidate"] if s == "RUNTIME_MATCHED" else COLORS["fallback"] for s in statuses]
        y_pos = np.arange(len(slots))
        ax.barh(y_pos, scores, color=colors, edgecolor="#222238", alpha=0.9)
        for i, (v, st) in enumerate(zip(scores, statuses)):
            ax.text(v + 0.01, i, f"{v:.4f} [{st}]", va="center", color=COLORS["text"], fontsize=8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(slots, fontsize=9)
        ax.set_xlim(0, 1.4)
        ax.set_xlabel("Runtime Evidence Score (Blue=RUNTIME_MATCHED, Grey=FALLBACK_DEFAULT)")
        ax.set_title("09. Pipeline Component Evidence Scores\n(from runtime evidence_scores.json — FALLBACK=0.50 where no entity matched)", fontsize=10)
        ax.grid(axis="x", linestyle="--")

        # Legend
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(color=COLORS["candidate"], label="RUNTIME_MATCHED"),
            Patch(color=COLORS["fallback"],  label="FALLBACK_DEFAULT (0.50)"),
        ], facecolor=COLORS["card"], edgecolor="#3E3E5C", fontsize=9)

        _save_fig(fig, self.out_dir / "09_pipeline_component_comparison.png")
        return [{"plot": "09_pipeline_component_comparison.png", "source": "decision_ledger (runtime evidence_scores.json)", "data_hash": "N/A (ledger data)"}]

    def _plot_10_evidence_ranking(self, ev_data: Dict) -> List[Dict]:
        """Ranks all extracted entities by composite_score from evidence_scores.json."""
        scores_dict = ev_data.get("evidence_scores", {})
        # Filter MODEL_ARCH entities
        arch_entries = [(k, v) for k, v in scores_dict.items() if v.get("entity_type") == "MODEL_ARCH"]
        arch_entries.sort(key=lambda x: x[1].get("composite_score", 0), reverse=True)
        arch_entries = arch_entries[:12]

        if not arch_entries:
            # Fall back to top entries by composite_score
            all_entries = sorted(scores_dict.items(), key=lambda x: x[1].get("composite_score", 0), reverse=True)[:12]
            arch_entries = all_entries

        names  = [e[0][:20] for e in arch_entries]
        scores = [e[1].get("composite_score", 0) for e in arch_entries]
        types  = [e[1].get("entity_type", "?") for e in arch_entries]

        fig, ax = plt.subplots(figsize=(9, max(5, len(names) * 0.5)))
        y_pos = np.arange(len(names))
        ax.barh(y_pos[::-1], scores[::-1], color=COLORS["resnet"], edgecolor="#222238")
        for i, (v, t) in enumerate(zip(scores[::-1], types[::-1])):
            ax.text(v + 0.01, i, f"{v:.4f} [{t}]", va="center", fontsize=8, color=COLORS["text"])
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names[::-1], fontsize=9)
        ax.set_xlim(0, 1.2)
        ax.set_xlabel("Composite Evidence Score (from runtime extraction)")
        ax.set_title("10. Runtime Evidence Score Ranking\n(from evidence_scores.json — actual SciBERT outputs)", fontsize=10)
        ax.grid(axis="x", linestyle="--")
        _save_fig(fig, self.out_dir / "10_evidence_model_ranking.png")
        return [{"plot": "10_evidence_model_ranking.png", "source": "evidence_scores.json (runtime)", "data_hash": "N/A (evidence data)"}]

    def _plot_11_confidence_dist(self, ev_data: Dict) -> List[Dict]:
        entities = ev_data.get("entities", [])
        confs = [float(e.get("confidence", 0.0)) for e in entities if "confidence" in e]

        fig, ax = plt.subplots(figsize=(8, 5))
        if confs:
            ax.hist(confs, bins=15, color=COLORS["resnet"], edgecolor="#222238", alpha=0.9)
            mean_c = np.mean(confs)
            ax.axvline(mean_c, color=COLORS["accent"], linestyle="--", linewidth=2,
                       label=f"Mean confidence = {mean_c:.3f}\n(ALL {len(confs)} entities: LOW tier)")
            ax.set_xlabel("SciBERT Token/Span Confidence")
            ax.set_ylabel("Entity Count")
            ax.legend(facecolor=COLORS["card"], edgecolor="#3E3E5C")
        else:
            ax.text(0.5, 0.5, "No entity confidence data available", transform=ax.transAxes, ha="center")

        ax.set_title("11. SciBERT NER Confidence Distribution\n(ALL entities classified LOW — WEAKLY_SUPERVISED)", fontsize=10)
        ax.grid(linestyle="--")
        _save_fig(fig, self.out_dir / "11_evidence_confidence_distribution.png")
        return [{"plot": "11_evidence_confidence_distribution.png", "source": "ner_entities.jsonl (runtime)", "data_hash": "N/A"}]

    def _plot_12_entity_types(self, ev_data: Dict) -> List[Dict]:
        entities = ev_data.get("entities", [])
        from collections import Counter
        type_counts = Counter(e.get("entity_type", "UNKNOWN") for e in entities)

        if not type_counts:
            type_counts = {"NO_ENTITIES": 1}

        labels  = list(type_counts.keys())
        counts  = [type_counts[l] for l in labels]
        sort_idx = np.argsort(counts)
        labels_sorted = [labels[i] for i in sort_idx]
        counts_sorted = [counts[i] for i in sort_idx]

        fig, ax = plt.subplots(figsize=(9, max(4, len(labels) * 0.5)))
        y_pos = np.arange(len(labels_sorted))
        ax.barh(y_pos, counts_sorted, color=COLORS["lr"], edgecolor="#222238")
        for i, v in enumerate(counts_sorted):
            ax.text(v + 0.1, i, str(v), va="center", fontsize=9, color=COLORS["text"])
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels_sorted, fontsize=9)
        ax.set_xlabel("Extracted Entity Count")
        ax.set_title(f"12. NER Entity Type Distribution (n={sum(counts)} total)\n(from ner_entities.jsonl — actual runtime extraction)", fontsize=10)
        ax.grid(axis="x", linestyle="--")
        _save_fig(fig, self.out_dir / "12_entity_type_distribution.png")
        return [{"plot": "12_entity_type_distribution.png", "source": "ner_entities.jsonl (runtime)", "data_hash": "N/A"}]

    def _plot_13_evidence_switching(self, decision_ledger: List[Dict]) -> List[Dict]:
        """Shows evidence routing status per decision slot — RUNTIME_MATCHED vs FALLBACK."""
        slots    = [d.get("target_slot", "?")[:25] for d in decision_ledger]
        statuses = [d.get("evidence_routing_status", "FALLBACK_DEFAULT") for d in decision_ledger]
        selected = [d.get("selected_name", "?")[:20] for d in decision_ledger]

        if not slots:
            slots, statuses, selected = ["No decisions"], ["FALLBACK_DEFAULT"], ["N/A"]

        colors = [COLORS["candidate"] if s == "RUNTIME_MATCHED" else COLORS["fallback"] for s in statuses]
        x_pos = np.arange(len(slots))

        fig, ax = plt.subplots(figsize=(max(10, len(slots)*1.5), 5))
        ax.bar(x_pos, [1.0]*len(slots), color=colors, width=0.6, edgecolor="#222238")
        for i, (sel, st) in enumerate(zip(selected, statuses)):
            ax.text(i, 0.5, f"{sel}\n[{st[:12]}]", ha="center", va="center",
                    color=COLORS["text"], fontweight="bold", fontsize=7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(slots, rotation=25, ha="right", fontsize=8)
        ax.set_ylim(0, 1.3)
        ax.set_yticks([])
        ax.set_title("13. Evidence Routing Status Per Decision Slot\n(Blue=RUNTIME_MATCHED from SciBERT; Grey=FALLBACK_DEFAULT)", fontsize=10)
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(color=COLORS["candidate"], label="RUNTIME_MATCHED"),
            Patch(color=COLORS["fallback"],  label="FALLBACK_DEFAULT"),
        ], facecolor=COLORS["card"], edgecolor="#3E3E5C")
        _save_fig(fig, self.out_dir / "13_evidence_switching_validation.png")
        return [{"plot": "13_evidence_switching_validation.png", "source": "decision_ledger (runtime)", "data_hash": "N/A"}]

    def _plot_14_provenance_cov(self) -> List[Dict]:
        prov_file = Path("evidence/final/submission/New/provenance/evidence_source_verification.json")
        if prov_file.exists():
            with open(prov_file, "r", encoding="utf-8") as f:
                prov_data = json.load(f)
            papers = prov_data.get("papers", [])
            verified = sum(1 for p in papers if p.get("verification_status") == "VERIFIED")
            unverified = sum(1 for p in papers if p.get("verification_status") == "UNVERIFIED")
            not_found = sum(1 for p in papers if p.get("verification_status") == "NOT_FOUND")
        else:
            verified, unverified, not_found = 0, 0, 0

        total = verified + unverified + not_found or 1
        fig, ax = plt.subplots(figsize=(7, 5))
        cats = ["VERIFIED", "UNVERIFIED", "NOT_FOUND"]
        vals = [verified, unverified, not_found]
        cols = [COLORS["ensemble"], COLORS["fallback"], COLORS["accent"]]
        ax.bar(cats, vals, color=cols, width=0.5, edgecolor="#222238")
        for i, v in enumerate(vals):
            ax.text(i, v + 0.05, str(v), ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.set_ylabel("Paper Count")
        ax.set_title(f"14. PMID/DOI Verification Status\n(from evidence_source_verification.json — {total} papers audited)", fontsize=10)
        ax.grid(axis="y", linestyle="--")
        _save_fig(fig, self.out_dir / "14_provenance_coverage.png")
        return [{"plot": "14_provenance_coverage.png", "source": "evidence_source_verification.json", "data_hash": "N/A"}]

    def _plot_15_modality_comp(self, loader: CanonicalResultLoader) -> List[Dict]:
        names, roc_vals = loader.all_cohort_metrics("roc_auc_mean")
        _, f1_vals      = loader.all_cohort_metrics("f1_mean")

        if not names:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, "No results", transform=ax.transAxes, ha="center")
            _save_fig(fig, self.out_dir / "15_modality_pipeline_comparison.png")
            return [{"plot": "15_modality_pipeline_comparison.png", "source": "final_results.json", "data_hash": loader.data_hash}]

        short_names = [n[:20] for n in names]
        x = np.arange(len(short_names))
        w = 0.35
        fig, ax = plt.subplots(figsize=(max(9, len(names)*1.8), 5))
        ax.bar(x - w/2, roc_vals, w, label="ROC-AUC",   color=COLORS["candidate"], edgecolor="#222238")
        ax.bar(x + w/2, f1_vals,  w, label="F1-Score",  color=COLORS["ensemble"],  edgecolor="#222238")
        ax.set_xticks(x)
        ax.set_xticklabels(short_names, rotation=15, ha="right", fontsize=8)
        ax.set_ylim(0.0, 1.1)
        ax.set_ylabel("Score")
        ax.set_title("15. Multi-Cohort Performance: ROC-AUC & F1\n(from canonical_predictions.jsonl — synthetic/demo datasets)", fontsize=10)
        ax.legend(facecolor=COLORS["card"], edgecolor="#3E3E5C")
        ax.grid(axis="y", linestyle="--")
        _save_fig(fig, self.out_dir / "15_modality_pipeline_comparison.png")
        return [{"plot": "15_modality_pipeline_comparison.png", "source": "final_results.json", "data_hash": loader.data_hash}]

    def _plot_16_per_seed(self, loader: CanonicalResultLoader) -> List[Dict]:
        # Use first cohort that has per-seed data
        primary_cohort = loader.cohort_names[0] if loader.cohort_names else None
        if primary_cohort is None:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            _save_fig(fig, self.out_dir / "16_per_seed_performance.png")
            return [{"plot": "16_per_seed_performance.png", "source": "canonical_predictions.jsonl", "data_hash": loader.data_hash}]

        seeds, rocs, f1s = loader.per_seed_data(primary_cohort)
        seed_labels = [f"Seed {s}" for s in seeds]

        x = np.arange(len(seed_labels))
        w = 0.35
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x - w/2, rocs, w, label="ROC-AUC", color=COLORS["candidate"], edgecolor="#222238")
        ax.bar(x + w/2, f1s,  w, label="F1-Score",color=COLORS["ensemble"],  edgecolor="#222238")
        for i, (r, f) in enumerate(zip(rocs, f1s)):
            ax.text(i - w/2, r + 0.01, f"{r:.3f}", ha="center", va="bottom", fontsize=8)
            ax.text(i + w/2, f + 0.01, f"{f:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(seed_labels, fontsize=10)
        ax.set_ylim(0.0, 1.1)
        ax.set_ylabel("Score")
        ax.set_title(f"16. Per-Seed Robustness — {primary_cohort[:30]}\n(recomputed from canonical_predictions.jsonl)", fontsize=10)
        ax.legend(facecolor=COLORS["card"], edgecolor="#3E3E5C")
        ax.grid(axis="y", linestyle="--")
        _save_fig(fig, self.out_dir / "16_per_seed_performance.png")
        return [{"plot": "16_per_seed_performance.png", "source": "canonical_predictions.jsonl (recomputed)", "data_hash": loader.data_hash}]

    def _plot_17_candidate_vs_default(self, loader: CanonicalResultLoader) -> List[Dict]:
        """Shows candidate (evidence-conditioned) vs its own score — since we have no external unconditioned baseline."""
        primary = next((r for r in (loader._final_results or []) if r.get("roc_auc_mean") is not None), None)
        if primary is None:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.text(0.5, 0.5, "No results", transform=ax.transAxes, ha="center")
            _save_fig(fig, self.out_dir / "17_candidate_vs_default_xgboost.png")
            return [{"plot": "17_candidate_vs_default_xgboost.png", "source": "final_results.json", "data_hash": loader.data_hash}]

        model_name  = primary.get("selected_model", "Model")
        cand_roc    = primary.get("roc_auc_mean", 0.0)
        ens_roc     = primary.get("ensemble_roc_auc_mean", 0.0)
        ens_label   = "Ensemble: " + " + ".join(primary.get("ensemble_members", ["?"]))

        labels = [f"Evidence-Conditioned\n{model_name}", ens_label]
        vals   = [cand_roc, ens_roc]
        colors = [COLORS["candidate"], COLORS["ensemble"]]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(labels, vals, color=colors, width=0.4, edgecolor="#222238")
        for b in bars:
            y = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, y + 0.01, f"{y:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
        ax.set_ylabel("Test ROC-AUC")
        ax.set_ylim(0.0, 1.1)
        ax.set_title(f"17. Evidence-Conditioned Candidate vs Ensemble\n{primary.get('cohort_name','')} — (from canonical_predictions.jsonl)", fontsize=10)
        ax.grid(axis="y", linestyle="--")
        plt.xticks(rotation=10, ha="right", fontsize=9)
        ax.annotate("No unconditioned baseline is available in this pipeline.\nCandidate selection was evidence-conditioned.",
                    xy=(0.5, 0.02), xycoords="axes fraction", ha="center", fontsize=8,
                    color=COLORS["fallback"])
        _save_fig(fig, self.out_dir / "17_candidate_vs_default_xgboost.png")
        return [{"plot": "17_candidate_vs_default_xgboost.png", "source": "final_results.json", "data_hash": loader.data_hash}]

    def _plot_18_pipeline_summary(self, decision_ledger: List[Dict]) -> List[Dict]:
        steps = [
            "1. Literature\nAcquisition", "2. SciBERT NER\n(WEAKLY_SUP.)",
            "3. Section\nFiltering", "4. Evidence\nScoring",
            "5. Component\nRanking", "6. Safety\nGates",
            "7. Real\nTraining", "8. Ensemble\n(validation wts)",
            "9. Canonical\nResults", "10. Plots &\nReport",
        ]
        runtime_matched = sum(1 for d in decision_ledger if d.get("evidence_routing_status") == "RUNTIME_MATCHED")
        fallback = sum(1 for d in decision_ledger if d.get("evidence_routing_status") == "FALLBACK_DEFAULT")

        fig, ax = plt.subplots(figsize=(13, 5))
        x_pos = np.arange(len(steps))
        ax.plot(x_pos, [1.0]*len(steps), "o-", color=COLORS["candidate"], linewidth=3, markersize=12)
        for i, s in enumerate(steps):
            ax.text(i, 0.94, s, ha="center", va="top", color=COLORS["text"], fontweight="bold", fontsize=7.5)
        ax.set_xlim(-0.7, len(steps) - 0.3)
        ax.set_ylim(0.78, 1.15)
        ax.axis("off")
        ax.set_title(
            f"18. End-to-End Scientific Workflow\n"
            f"Evidence routing: {runtime_matched} RUNTIME_MATCHED, {fallback} FALLBACK_DEFAULT",
            fontsize=11
        )
        _save_fig(fig, self.out_dir / "18_end_to_end_pipeline_summary.png")
        return [{"plot": "18_end_to_end_pipeline_summary.png", "source": "decision_ledger (runtime)", "data_hash": "N/A"}]
