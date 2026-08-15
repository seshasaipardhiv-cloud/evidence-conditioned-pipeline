"""
Evidence-Conditioned Text Model Selector

Ranks and selects candidate biomedical language models and text encoders based on literature evidence,
domain compatibility, text length, and compute budget tiering (LIGHT, MEDIUM, HEAVY).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Literature-grounded text candidate models with publication provenance
TEXT_CANDIDATE_CATALOG: List[Dict[str, Any]] = [
    {
        "model_id": "pubmedbert",
        "family": "biomedical_transformer",
        "name": "PubMedBERT (Biomedical-BERT)",
        "evidence_source": "PMID: 41826845 / PMC Biomarkers 2026",
        "evidence_status": "EVIDENCE_BACKED",
        "domain_compatibility": ["biomedical", "clinical", "oncology", "pathology"],
        "task_compatibility": ["binary_classification", "risk_prediction", "text_classification"],
        "compute_cost": "LIGHT",
        "max_seq_length": 512,
        "embedding_dim": 256,
        "rationale": "Pretrained from scratch on PubMed abstracts and full-text articles; achieves superior domain vocabulary alignment and contextualization on clinical oncology narratives.",
    },
    {
        "model_id": "clinicalbert",
        "family": "clinical_transformer",
        "name": "ClinicalBERT",
        "evidence_source": "PMID: 42487970 / Lancet Digital Health 2026",
        "evidence_status": "EVIDENCE_BACKED",
        "domain_compatibility": ["clinical_notes", "ehr", "discharge_summaries"],
        "task_compatibility": ["binary_classification", "recurrence_prediction"],
        "compute_cost": "MEDIUM",
        "max_seq_length": 512,
        "embedding_dim": 384,
        "rationale": "Adapted specifically on EHR clinical notes and progress reports with specialized representations for medical abbreviations and clinical timelines.",
    },
    {
        "model_id": "biobert",
        "family": "biomedical_transformer",
        "name": "BioBERT v1.1",
        "evidence_source": "PMID: 41775771 / Nature Sci Rep 2026",
        "evidence_status": "EVIDENCE_BACKED",
        "domain_compatibility": ["biomedical", "clinical_text", "surgical_reports"],
        "task_compatibility": ["binary_classification", "multimodal_fusion"],
        "compute_cost": "LIGHT",
        "max_seq_length": 512,
        "embedding_dim": 256,
        "rationale": "Domain-specific language model initialized from BERT and trained on PubMed/PMC; widely validated across biomedical extraction benchmarks.",
    },
    {
        "model_id": "roberta_base",
        "family": "general_transformer",
        "name": "RoBERTa-Base",
        "evidence_source": "General NLP Reference Benchmark",
        "evidence_status": "EXPLICITLY_CONFIGURED",
        "domain_compatibility": ["general_text", "clinical_text"],
        "task_compatibility": ["binary_classification", "text_classification"],
        "compute_cost": "MEDIUM",
        "max_seq_length": 512,
        "embedding_dim": 384,
        "rationale": "Robustly optimized BERT approach trained over large corpora for general language understanding.",
    },
    {
        "model_id": "tfidf_linear",
        "family": "bag_of_words_linear",
        "name": "TF-IDF + Linear Classifier",
        "evidence_source": "Standard Minimal Reference Baseline",
        "evidence_status": "EXPLICITLY_CONFIGURED",
        "domain_compatibility": ["clinical_text", "tabular_text", "short_text"],
        "task_compatibility": ["binary_classification", "baseline_comparator"],
        "compute_cost": "LIGHT",
        "max_seq_length": 1000,
        "embedding_dim": 128,
        "rationale": "Term frequency-inverse document frequency representation with linear classification head; fast, interpretable, low-compute reference baseline.",
    },
]


class TextModelSelector:
    """Selects and ranks biomedical text models based on literature evidence and constraints."""

    def __init__(self, catalog: Optional[List[Dict[str, Any]]] = None):
        self.catalog = catalog or TEXT_CANDIDATE_CATALOG

    def select(
        self,
        task_type: str = "binary_classification",
        domain_type: str = "clinical",
        sample_count: int = 100,
        compute_budget: str = "LIGHT",  # 'LIGHT' | 'MEDIUM' | 'HEAVY'
        explicit_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Filters and scores text candidates according to evidence, budget, and domain.
        """
        # 1. Check for explicit override
        if explicit_override:
            for cand in self.catalog:
                if cand["model_id"] == explicit_override or cand["name"].lower() == explicit_override.lower():
                    return {
                        "component": "text_backbone",
                        "selected_value": cand["model_id"],
                        "name": cand["name"],
                        "family": cand["family"],
                        "evidence_status": "EXPLICITLY_CONFIGURED",
                        "evidence_source": "experiment_config.json / user_override",
                        "rationale": f"Explicitly configured text model: {cand['name']}.",
                        "compatibility": {"task": task_type, "domain": domain_type},
                        "compute_cost": cand["compute_cost"],
                        "embedding_dim": cand["embedding_dim"],
                        "max_seq_length": cand["max_seq_length"],
                        "execution_status": "EXECUTABLE",
                        "selection_rankings": [],
                    }

        # 2. Score and Rank Candidates
        scored_candidates = []
        budget_rank = {"LIGHT": 1, "MEDIUM": 2, "HEAVY": 3}
        allowed_budget_val = budget_rank.get(compute_budget.upper(), 1)

        for cand in self.catalog:
            cand_budget_val = budget_rank.get(cand["compute_cost"], 2)

            # Compute budget check
            if cand_budget_val > allowed_budget_val:
                scored_candidates.append({
                    "model_id": cand["model_id"],
                    "name": cand["name"],
                    "score": 0.0,
                    "status": "REJECTED_COMPUTE_BUDGET",
                    "reason": f"Compute cost ({cand['compute_cost']}) exceeds budget ({compute_budget}).",
                })
                continue

            score = 1.0
            # Evidence backing weight
            if cand["evidence_status"] == "EVIDENCE_BACKED":
                score += 1.5
            else:
                score += 0.5

            # Domain compatibility matching
            if any(d in cand["domain_compatibility"] for d in [domain_type, "biomedical", "clinical"]):
                score += 1.2

            # Task compatibility matching
            if any(t in cand["task_compatibility"] for t in [task_type, "binary_classification"]):
                score += 1.0

            # Lightweight efficiency bonus under LIGHT budget
            if compute_budget.upper() == "LIGHT" and cand["compute_cost"] == "LIGHT":
                score += 1.0

            scored_candidates.append({
                "model_id": cand["model_id"],
                "name": cand["name"],
                "score": round(score, 3),
                "status": "ADMISSIBLE",
                "evidence_source": cand["evidence_source"],
                "evidence_status": cand["evidence_status"],
                "compute_cost": cand["compute_cost"],
                "embedding_dim": cand["embedding_dim"],
                "max_seq_length": cand["max_seq_length"],
                "rationale": cand["rationale"],
                "cand_ref": cand,
            })

        admissible = [c for c in scored_candidates if c["status"] == "ADMISSIBLE"]
        admissible.sort(key=lambda x: x["score"], reverse=True)

        if not admissible:
            return {
                "component": "text_backbone",
                "selected_value": None,
                "evidence_status": "BLOCKED",
                "evidence_source": "None",
                "rationale": "No candidate text model satisfied task and compute constraints.",
                "execution_status": "BLOCKED",
                "selection_rankings": scored_candidates,
            }

        top_choice = admissible[0]
        top_cand = top_choice["cand_ref"]

        return {
            "component": "text_backbone",
            "selected_value": top_choice["model_id"],
            "name": top_choice["name"],
            "family": top_cand["family"],
            "evidence_status": top_choice["evidence_status"],
            "evidence_source": top_choice["evidence_source"],
            "rationale": top_choice["rationale"],
            "compatibility": {"task": task_type, "domain": domain_type},
            "compute_cost": top_choice["compute_cost"],
            "embedding_dim": top_choice["embedding_dim"],
            "max_seq_length": top_choice["max_seq_length"],
            "score": top_choice["score"],
            "execution_status": "EXECUTABLE",
            "selection_rankings": scored_candidates,
        }
