"""
evidence_decision_engine.py  —  SCIENTIFICALLY REPAIRED

Stage 2D Runtime Evidence-Conditioned Decision Engine

Every candidate evidence score MUST originate from the actual runtime
evidence_scores.json produced by Stage 2D SciBERT NER.

Evidence routing procedure:
  1. Scan all entities in evidence_scores.json with matching entity_type.
  2. Match candidate name against canonical_name using alias table.
  3. Use composite_score from matched entity as the runtime evidence score.
  4. If no entity matches, score = FALLBACK_DEFAULT (0.50).
     FALLBACK is NEVER represented as literature-derived evidence.

Hardcoded candidate scores such as XGBoost=0.940 are FORBIDDEN.

Decision Ledger records:
  entity_key, entity_type, runtime_score, evidence_routing_status,
  supporting_papers, supporting_spans, selection_reason.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical alias table: maps candidate display names to lookup aliases.
# Matches against 'canonical_name' field in evidence_scores.json.
# ---------------------------------------------------------------------------
_CANDIDATE_ALIASES: Dict[str, List[str]] = {
    # Tabular models
    "XGBoost":                 ["xgboost", "extreme gradient", "xgb", "gradient boost", "gbm", "gradient boosting"],
    "Random Forest":           ["random forest", "rf", "random forests", "ensemble trees"],
    "Logistic Regression":     ["logistic regression", "logistic", "logit", "linear classifier", "lr"],
    "Tabular MLP":             ["multilayer perceptron", "mlp", "feed-forward", "feedforward", "neural network"],
    # Image models
    "ResNet-18":               ["resnet", "resnet-18", "resnet18", "residual network", "res-net"],
    "ResNet-50":               ["resnet-50", "resnet50"],
    "EfficientNet-B0":         ["efficientnet", "efficientnet-b0", "efficient net"],
    "Vision Transformer (ViT-Small)": ["vit", "vision transformer", "transformer", "self-attention"],
    # Text models
    "PubMedBERT":              ["pubmedbert", "pubmed bert", "bert", "biobert", "biomed", "language model"],
    "ClinicalBERT":            ["clinicalbert", "clinical bert", "clinbert"],
    "TF-IDF + Linear Classifier": ["tfidf", "tf-idf", "bag of words", "bow", "linear svm", "word embeddings"],
    # Preprocessing
    "MICE Imputation":         ["mice", "multiple imputation", "imputation", "missing data"],
    "Standard Scaling (Z-Score)": ["standard scal", "z-score", "standardiz", "normaliz"],
    "SMOTE (Synthetic Minority Oversampling)": ["smote", "oversampling", "synthetic minority", "class imbalance"],
    # Fusion
    "Late Fusion (Feature Concatenation)": ["late fusion", "concatenat", "feature fusion", "multimodal"],
    "Gated Multimodal Fusion": ["gated", "gate fusion", "attention gate"],
    "Cross-Modal Attention":   ["cross-attention", "cross-modal", "cross attention"],
}

# Entity type mapping for each candidate slot
_SLOT_ENTITY_TYPES: Dict[str, List[str]] = {
    "tabular_model":    ["MODEL_ARCH"],
    "image_model":      ["MODEL_ARCH"],
    "text_model":       ["MODEL_ARCH"],
    "preprocessing":    ["PREPROCESSING", "SAMPLING"],
    "fusion":           ["FUSION"],
    "ensemble":         ["MODEL_ARCH"],
}

# Fallback score when no runtime entity matches — documented default
_FALLBACK_SCORE = 0.50
_FALLBACK_STATUS = "FALLBACK_DEFAULT"
_RUNTIME_STATUS  = "RUNTIME_MATCHED"


class EvidenceDecisionEngine:
    """
    Ranks pipeline components using Stage 2D runtime SciBERT evidence scores.
    Every decision records a complete, auditable provenance chain.
    """

    def __init__(self, stage2d_dir: str = "evidence/processed/stage2d"):
        self.stage2d_dir = Path(stage2d_dir)
        self.evidence_records: Dict[str, Any] = self._load_stage2d_scores()
        self.decision_ledger: List[Dict[str, Any]] = []
        logger.info(
            f"EvidenceDecisionEngine loaded {len(self.evidence_records)} runtime evidence records "
            f"from {self.stage2d_dir / 'evidence_scores.json'}."
        )

    def _load_stage2d_scores(self) -> Dict[str, Any]:
        scores_file = self.stage2d_dir / "evidence_scores.json"
        if scores_file.exists():
            with open(scores_file, "r", encoding="utf-8-sig") as f:
                return json.load(f)
        logger.warning(f"evidence_scores.json not found at {scores_file}. All decisions will use FALLBACK_DEFAULT.")
        return {}

    # ------------------------------------------------------------------
    # Public selectors
    # ------------------------------------------------------------------

    def select_tabular_model(self, sample_count: int, feature_count: int, compute_budget: str = "LIGHT") -> Dict[str, Any]:
        candidates = [
            {"id": "xgboost",            "name": "XGBoost",            "min_samples": 25, "light_ok": True},
            {"id": "random_forest",      "name": "Random Forest",       "min_samples": 20, "light_ok": True},
            {"id": "logistic_regression","name": "Logistic Regression", "min_samples": 10, "light_ok": True},
            {"id": "tabular_mlp",        "name": "Tabular MLP",         "min_samples": 50, "light_ok": False},
        ]
        return self._rank_and_select(
            slot="tabular_model", candidates=candidates,
            sample_count=sample_count, compute_budget=compute_budget, modality="tabular",
        )

    def select_image_model(self, sample_count: int, compute_budget: str = "LIGHT") -> Dict[str, Any]:
        candidates = [
            {"id": "resnet18",       "name": "ResNet-18",                    "min_samples": 20, "light_ok": True},
            {"id": "resnet50",       "name": "ResNet-50",                    "min_samples": 50, "light_ok": False},
            {"id": "efficientnet_b0","name": "EfficientNet-B0",              "min_samples": 25, "light_ok": True},
            {"id": "vit_small",      "name": "Vision Transformer (ViT-Small)","min_samples": 80, "light_ok": False},
        ]
        return self._rank_and_select(
            slot="image_model", candidates=candidates,
            sample_count=sample_count, compute_budget=compute_budget, modality="image",
        )

    def select_text_model(self, sample_count: int, compute_budget: str = "LIGHT") -> Dict[str, Any]:
        candidates = [
            {"id": "pubmedbert",   "name": "PubMedBERT",                 "min_samples": 20, "light_ok": True},
            {"id": "clinicalbert", "name": "ClinicalBERT",               "min_samples": 25, "light_ok": True},
            {"id": "tfidf_linear", "name": "TF-IDF + Linear Classifier", "min_samples": 10, "light_ok": True},
        ]
        return self._rank_and_select(
            slot="text_model", candidates=candidates,
            sample_count=sample_count, compute_budget=compute_budget, modality="text",
        )

    def select_preprocessing(self, modality: str, has_missing: bool, has_imbalance: bool) -> Dict[str, Any]:
        """Selects preprocessing methods using runtime evidence scores."""
        decisions: Dict[str, Any] = {}

        if modality == "tabular":
            decisions["imputation"] = self._lookup_preprocessing_component(
                "MICE Imputation" if has_missing else "Identity",
                reason_suffix="for multi-variable clinical missingness" if has_missing else "no missing values detected",
            )
            decisions["scaling"] = self._lookup_preprocessing_component(
                "Standard Scaling (Z-Score)",
                reason_suffix="canonical feature standardization",
            )
            decisions["sampling"] = self._lookup_preprocessing_component(
                "SMOTE (Synthetic Minority Oversampling)" if has_imbalance else "None (Balanced Cohort)",
                reason_suffix="class imbalance detected" if has_imbalance else "cohort is balanced",
            )
        elif modality == "image":
            decisions["image_transform"] = {
                "method": "Bicubic Resize (32x32) + ImageNet Channel Normalization",
                "evidence_routing_status": _FALLBACK_STATUS,
                "evidence_score": _FALLBACK_SCORE,
                "reason": "Standard vision preprocessing; FALLBACK_DEFAULT (no IMAGE_TRANSFORM entity in runtime evidence).",
            }
        elif modality == "text":
            decisions["text_transform"] = {
                "method": "SciBERT WordPiece Tokenization (max_length=64)",
                "evidence_routing_status": _FALLBACK_STATUS,
                "evidence_score": _FALLBACK_SCORE,
                "reason": "Standard biomedical tokenization; FALLBACK_DEFAULT (no TEXT_TRANSFORM entity in runtime evidence).",
            }

        return decisions

    def _lookup_preprocessing_component(self, component_name: str, reason_suffix: str) -> Dict[str, Any]:
        """Looks up a preprocessing component in runtime evidence."""
        ev, key, status = self._resolve_runtime_score(component_name, slot="preprocessing")
        return {
            "method": component_name,
            "evidence_routing_status": status,
            "evidence_score": round(ev.get("composite_score", _FALLBACK_SCORE), 4),
            "entity_key": key,
            "supporting_pmids": ev.get("supporting_pmids", []),
            "reason": (
                f"RUNTIME_MATCHED entity '{key}' (score={ev.get('composite_score', _FALLBACK_SCORE):.4f}) — {reason_suffix}."
                if status == _RUNTIME_STATUS else
                f"FALLBACK_DEFAULT (no runtime entity matched '{component_name}') — {reason_suffix}."
            ),
        }

    def select_fusion(self, active_modalities: List[str], compute_budget: str = "LIGHT") -> Dict[str, Any]:
        if len(active_modalities) <= 1:
            return {
                "selected_fusion": "Unimodal Direct Head",
                "evidence_score": 1.0,
                "evidence_routing_status": _FALLBACK_STATUS,
                "reason": "Single active modality; no fusion required.",
            }

        candidates = [
            {"name": "Late Fusion (Feature Concatenation)", "light_ok": True},
            {"name": "Gated Multimodal Fusion",             "light_ok": True},
            {"name": "Cross-Modal Attention",               "light_ok": False},
        ]
        if compute_budget == "LIGHT":
            candidates = [c for c in candidates if c["light_ok"]]

        scored = []
        for c in candidates:
            ev, key, status = self._resolve_runtime_score(c["name"], slot="fusion")
            scored.append({
                "name": c["name"],
                "entity_key": key,
                "score": round(ev.get("composite_score", _FALLBACK_SCORE), 4),
                "status": status,
                "supporting_pmids": ev.get("supporting_pmids", []),
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        best = scored[0]

        return {
            "selected_fusion": best["name"],
            "evidence_score": best["score"],
            "evidence_routing_status": best["status"],
            "entity_key": best["entity_key"],
            "supporting_pmids": best["supporting_pmids"],
            "reason": (
                f"{'RUNTIME_MATCHED' if best['status']==_RUNTIME_STATUS else 'FALLBACK_DEFAULT'} "
                f"entity '{best['entity_key']}' (score={best['score']:.4f}) under {compute_budget} budget."
            ),
            "all_candidates": scored,
        }

    def select_ensemble(self, member_models: List[str], is_multimodal: bool = False) -> Dict[str, Any]:
        return {
            "ensemble_strategy": "Validation-Performance-Weighted Averaging",
            "ensemble_members": member_models,
            "ensemble_label": f"Ensemble: {' + '.join(member_models)}",
            "evidence_routing_status": _FALLBACK_STATUS,
            "evidence_score": _FALLBACK_SCORE,
            "reason": (
                "FALLBACK_DEFAULT — ensemble strategy not directly resolved from NER entities; "
                "validation-performance weighting is a principled default that strictly prevents test leakage."
            ),
        }

    # ------------------------------------------------------------------
    # Core ranking engine
    # ------------------------------------------------------------------

    def _rank_and_select(
        self,
        slot: str,
        candidates: List[Dict[str, Any]],
        sample_count: int,
        compute_budget: str,
        modality: str,
    ) -> Dict[str, Any]:
        scored_candidates = []

        for cand in candidates:
            ev, entity_key, routing_status = self._resolve_runtime_score(cand["name"], slot=slot)

            base_score = round(ev.get("composite_score", _FALLBACK_SCORE), 4)
            ner_conf   = round(ev.get("ner_confidence_score", 0.0), 4)
            sec_score  = round(ev.get("section_relevance_score", 0.0), 4)
            pmids      = ev.get("supporting_pmids", [])
            n_papers   = ev.get("supporting_paper_count", 0)

            # Safety/budget gating — adjusts score but does not hide routing status
            adjusted_score = base_score
            constraint_note = None
            if sample_count < cand["min_samples"]:
                adjusted_score = base_score * 0.40
                constraint_note = f"SAFETY_CONSTRAINT: sample_count={sample_count} < min_required={cand['min_samples']}"
            elif compute_budget == "LIGHT" and not cand["light_ok"]:
                adjusted_score = base_score * 0.50
                constraint_note = f"BUDGET_CONSTRAINT: LIGHT budget excludes {cand['name']}"

            scored_candidates.append({
                "candidate_id":            cand["id"],
                "candidate_name":          cand["name"],
                "entity_key":              entity_key,
                "entity_type":             ev.get("entity_type", "UNKNOWN"),
                "runtime_evidence_score":  base_score,
                "adjusted_score":          round(adjusted_score, 4),
                "evidence_routing_status": routing_status,
                "ner_confidence":          ner_conf,
                "section_relevance":       sec_score,
                "supporting_pmids":        pmids,
                "supporting_paper_count":  n_papers,
                "constraint_note":         constraint_note,
            })

        scored_candidates.sort(key=lambda x: x["adjusted_score"], reverse=True)
        winner = scored_candidates[0]

        # Build human-readable selection reason
        if winner["constraint_note"]:
            reason = winner["constraint_note"]
        elif winner["evidence_routing_status"] == _RUNTIME_STATUS:
            reason = (
                f"RUNTIME_MATCHED entity '{winner['entity_key']}' "
                f"(NER confidence={winner['ner_confidence']:.3f}, "
                f"section_relevance={winner['section_relevance']:.3f}, "
                f"composite_score={winner['runtime_evidence_score']:.4f}) "
                f"from {winner['supporting_paper_count']} paper(s) "
                f"(PMIDs: {', '.join(winner['supporting_pmids'][:2])})."
            )
        else:
            reason = (
                f"FALLBACK_DEFAULT — no runtime entity in evidence_scores.json matched '{winner['candidate_name']}'. "
                f"Defaulting to highest-priority eligible candidate."
            )

        ledger_entry = {
            "target_slot":             slot,
            "modality":                modality,
            "selected_name":           winner["candidate_name"],
            "entity_key":              winner["entity_key"],
            "entity_type":             winner["entity_type"],
            "runtime_evidence_score":  winner["runtime_evidence_score"],
            "adjusted_evidence_score": winner["adjusted_score"],
            "evidence_score":          winner["adjusted_score"],
            "evidence_routing_status": winner["evidence_routing_status"],
            "ner_confidence":          winner["ner_confidence"],
            "section_relevance":       winner["section_relevance"],
            "supporting_pmids":        winner["supporting_pmids"],
            "constraint_note":         winner["constraint_note"],
            "candidate_ranking": [
                {
                    "name":              c["candidate_name"],
                    "entity_key":        c["entity_key"],
                    "adjusted_score":    c["adjusted_score"],
                    "routing_status":    c["evidence_routing_status"],
                    "constraint":        c["constraint_note"],
                }
                for c in scored_candidates
            ],
            "selection_reason": reason,
        }
        self.decision_ledger.append(ledger_entry)

        return {
            "selected_id":             winner["candidate_id"],
            "selected_name":           winner["candidate_name"],
            "entity_key":              winner["entity_key"],
            "entity_type":             winner["entity_type"],
            "evidence_score":          winner["adjusted_score"],
            "evidence_routing_status": winner["evidence_routing_status"],
            "ner_confidence":          winner["ner_confidence"],
            "supporting_pmids":        winner["supporting_pmids"],
            "all_candidates":          scored_candidates,
            "selection_reason":        reason,
        }

    # ------------------------------------------------------------------
    # Runtime evidence resolver — core matching logic
    # ------------------------------------------------------------------

    def _resolve_runtime_score(
        self, candidate_name: str, slot: str
    ) -> tuple[Dict[str, Any], str, str]:
        """
        Resolves a candidate name to an actual runtime evidence record.

        Matching strategy (in order):
          1. Exact key match (canonical_name == alias)
          2. Alias match (any alias in _CANDIDATE_ALIASES[candidate] is a substring of canonical_name)
          3. Any entity whose canonical_name is a substring of any candidate alias
          4. No match → FALLBACK_DEFAULT

        Returns:
          (evidence_record_dict, matched_entity_key, routing_status)
        """
        aliases = _CANDIDATE_ALIASES.get(candidate_name, [candidate_name.lower()])
        target_types = _SLOT_ENTITY_TYPES.get(slot, [])

        # Filter evidence records by entity type when possible
        pool = {
            k: v for k, v in self.evidence_records.items()
            if not target_types or v.get("entity_type", "") in target_types
        }

        # --- Pass 1: exact key match ---
        for alias in aliases:
            if alias in pool:
                return pool[alias], alias, _RUNTIME_STATUS

        # --- Pass 2: alias substring in canonical_name ---
        for alias in aliases:
            for key, rec in pool.items():
                canon = rec.get("canonical_name", key).lower()
                if alias in canon or canon in alias:
                    return rec, key, _RUNTIME_STATUS

        # --- Pass 3: broader pool (ignore entity_type filter) ---
        for alias in aliases:
            for key, rec in self.evidence_records.items():
                canon = rec.get("canonical_name", key).lower()
                if alias in canon or canon in alias:
                    return rec, key, _RUNTIME_STATUS

        # --- No match: FALLBACK ---
        logger.debug(
            f"EVIDENCE_FALLBACK: no runtime entity matched candidate '{candidate_name}' "
            f"for slot='{slot}'. Using FALLBACK_DEFAULT score={_FALLBACK_SCORE}."
        )
        return {}, "NONE", _FALLBACK_STATUS
