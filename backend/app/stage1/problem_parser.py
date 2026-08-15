"""
Stage 1 — Problem Parser

Deterministic NLP parser that extracts structured fields from a natural-language
problem statement without LLM calls.

Strategy:
  - Keyword/phrase matching on lowercased tokens for task type, modalities, metrics, etc.
  - All extracted fields carry explicit provenance.
  - If information is absent, the value is None with confidence=unknown.
  - Inferred fields are clearly marked confidence=inferred with evidence_text preserved.

NO hallucination. NO fabricated targets. NO invented metrics.
"""
from __future__ import annotations

import re
from typing import List, Optional, Set, Tuple

from backend.app.stage1.models import (
    ConfidenceLevel,
    ExtractedField,
    ParsedProblem,
    Provenance,
    SourceType,
    TaskType,
)


# ──────────────────────────────────────────────────────────────────────────────
# Keyword maps (order matters: more specific patterns first)
# ──────────────────────────────────────────────────────────────────────────────

TASK_PATTERNS: List[Tuple[TaskType, List[str]]] = [
    (TaskType.survival_analysis, [
        "survival analysis", "survival model", "time-to-event", "time to event",
        "kaplan", "cox", "hazard", "overall survival", "progression-free survival",
        "event-free survival",
    ]),
    (TaskType.classification, [
        "classif", "predict class", "predict label", "binary prediction",
        "multi-class", "multiclass", "predict whether", "predict if",
        "detect ", "detection", "diagnos",
    ]),
    (TaskType.regression, [
        "regress", "predict value", "predict continuous", "predict numeric",
        "predict amount", "predict score", "continuous value", "continuous predict",
        "predict a continuous", "predict the continuous",
    ]),
    (TaskType.clustering, [
        "cluster", "unsupervised group", "subgroup discover",
        "stratif", "patient stratif",
    ]),
    (TaskType.ranking, [
        "rank", "prioriti",
    ]),
    (TaskType.anomaly_detection, [
        "anomaly", "outlier", "novelty",
    ]),
    (TaskType.segmentation, [
        "segment", "delineate", "contour",
    ]),
    (TaskType.generation, [
        "generat", "synthesiz", "augment data",
    ]),
]

DOMAIN_PATTERNS: List[Tuple[str, List[str]]] = [
    ("oncology", ["cancer", "oncol", "tumor", "tumour", "carcinoma", "malignant", "neoplasm"]),
    ("radiology", ["radiol", "imaging", "mri", "ct scan", "pet scan", "x-ray"]),
    ("pathology", ["histolog", "patholog", "biopsy", "slide"]),
    ("genomics", ["genom", "genetic", "mutation", "snp", "variant"]),
    ("clinical_medicine", ["clinical", "patient", "hospital", "ehr", "electronic health"]),
]

MODALITY_PATTERNS: List[Tuple[str, List[str]]] = [
    ("clinical", ["clinical", "structured data", "demographic", "patient data"]),
    ("pathology", ["patholog", "histolog", "tissue", "biopsy"]),
    ("blood", ["blood", "lab", "laboratory", "haematolog", "hematolog", "biomarker", "analyte", "serum"]),
    ("text", ["text", "report", "narrative", "free text", "clinical note", "discharge summary", "history"]),
    ("imaging", ["image", "mri", "ct", "pet", "wsi", "whole slide", "radiology"]),
    ("genomic", ["genomic", "genome", "sequence", "rna", "dna", "snp"]),
]

METRIC_PATTERNS: List[Tuple[str, List[str]]] = [
    ("AUROC", ["auroc", "auc", "roc", "area under"]),
    ("accuracy", ["accuracy"]),
    ("f1_score", ["f1", "f-score", "f measure"]),
    ("precision", ["precision"]),
    ("recall", ["recall", "sensitivity"]),
    ("specificity", ["specificity"]),
    ("c_index", ["c-index", "concordance index", "harrell"]),
    ("rmse", ["rmse", "root mean square"]),
    ("mae", ["mae", "mean absolute error"]),
    ("r2", ["r-squared", "r2 score", "coefficient of determination"]),
    ("log_loss", ["log loss", "cross entropy"]),
]

INTERPRETABILITY_PATTERNS = ["interpret", "explainab", "shap", "lime", "transparent", "explain"]
LATENCY_PATTERNS = ["real-time", "realtime", "low latency", "fast inference", "millisecond"]
EXCLUSION_PATTERNS = [
    (r"(?:do not|don't|avoid|without|exclude|no)\s+(.{3,40}?)(?:\.|,|$)", ),
    (r"(?:without using)\s+(.{3,40}?)(?:\.|,|$)", ),
]


def _make_user_provenance(confidence: ConfidenceLevel, evidence: Optional[str] = None) -> Provenance:
    return Provenance(
        source_type=SourceType.user_input,
        source_reference="user_problem_statement",
        extraction_method="deterministic_parser",
        confidence=confidence,
        evidence_text=evidence,
    )


def _find_first_match(text_lower: str, patterns: List[str]) -> Optional[str]:
    for pattern in patterns:
        if pattern in text_lower:
            idx = text_lower.find(pattern)
            # Return a short surrounding snippet as evidence
            start = max(0, idx - 10)
            end = min(len(text_lower), idx + len(pattern) + 20)
            return text_lower[start:end].strip()
    return None


def _detect_task_type(text_lower: str) -> Tuple[TaskType, ConfidenceLevel, Optional[str]]:
    for task_type, keywords in TASK_PATTERNS:
        snippet = _find_first_match(text_lower, keywords)
        if snippet:
            return task_type, ConfidenceLevel.explicit, snippet
    return TaskType.unknown, ConfidenceLevel.unknown, None


def _detect_domain(text_lower: str) -> Tuple[Optional[str], ConfidenceLevel, Optional[str]]:
    for domain, keywords in DOMAIN_PATTERNS:
        snippet = _find_first_match(text_lower, keywords)
        if snippet:
            return domain, ConfidenceLevel.inferred, snippet
    return None, ConfidenceLevel.unknown, None


def _detect_modalities(text_lower: str) -> Tuple[List[str], ConfidenceLevel, Optional[str]]:
    found: List[str] = []
    evidence_snippets: List[str] = []
    for modality, keywords in MODALITY_PATTERNS:
        snippet = _find_first_match(text_lower, keywords)
        if snippet:
            found.append(modality)
            evidence_snippets.append(snippet)
    if found:
        return found, ConfidenceLevel.explicit, "; ".join(evidence_snippets)
    return [], ConfidenceLevel.unknown, None


def _detect_metric(text_lower: str) -> Tuple[Optional[str], ConfidenceLevel, Optional[str]]:
    for metric, keywords in METRIC_PATTERNS:
        snippet = _find_first_match(text_lower, keywords)
        if snippet:
            return metric, ConfidenceLevel.explicit, snippet
    return None, ConfidenceLevel.unknown, None


def _detect_exclusions(text: str) -> List[str]:
    exclusions: List[str] = []
    for (pattern,) in EXCLUSION_PATTERNS:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        exclusions.extend([m.strip() for m in matches if m.strip()])
    return list(dict.fromkeys(exclusions))  # deduplicate while preserving order


def _detect_interpretability(text_lower: str) -> Tuple[Optional[str], ConfidenceLevel, Optional[str]]:
    snippet = _find_first_match(text_lower, INTERPRETABILITY_PATTERNS)
    if snippet:
        return "required", ConfidenceLevel.explicit, snippet
    return None, ConfidenceLevel.unknown, None


def _detect_latency(text_lower: str) -> Tuple[Optional[str], ConfidenceLevel, Optional[str]]:
    snippet = _find_first_match(text_lower, LATENCY_PATTERNS)
    if snippet:
        return "low_latency_required", ConfidenceLevel.explicit, snippet
    return None, ConfidenceLevel.unknown, None


def parse_problem_statement(problem_statement: str) -> ParsedProblem:
    """
    Deterministically parse a natural-language problem statement into structured fields.
    All extracted values carry provenance. Missing information is represented as None,
    not fabricated.
    """
    text_lower = problem_statement.lower()

    # ── Task type ────────────────────────────────────────────────────────────
    task_type_val, task_conf, task_evidence = _detect_task_type(text_lower)
    task_field = ExtractedField(
        value=task_type_val.value,
        provenance=_make_user_provenance(task_conf, task_evidence),
    )

    # ── Domain ───────────────────────────────────────────────────────────────
    domain_val, domain_conf, domain_evidence = _detect_domain(text_lower)
    domain_field = ExtractedField(
        value=domain_val,
        provenance=_make_user_provenance(domain_conf, domain_evidence),
    )

    # ── Application area — inferred from domain + task ────────────────────
    app_area: Optional[str] = None
    if domain_val and task_type_val != TaskType.unknown:
        app_area = f"{domain_val} / {task_type_val.value}"
    app_area_field = ExtractedField(
        value=app_area,
        provenance=_make_user_provenance(
            ConfidenceLevel.inferred if app_area else ConfidenceLevel.unknown,
            None,
        ),
    )

    # ── Modalities ───────────────────────────────────────────────────────────
    mod_vals, mod_conf, mod_evidence = _detect_modalities(text_lower)
    modality_field = ExtractedField(
        value=mod_vals if mod_vals else None,
        provenance=_make_user_provenance(mod_conf, mod_evidence),
    )

    # ── Metric ───────────────────────────────────────────────────────────────
    metric_val, metric_conf, metric_evidence = _detect_metric(text_lower)
    metric_field = ExtractedField(
        value=metric_val,
        provenance=_make_user_provenance(metric_conf, metric_evidence),
    )

    # ── Interpretability ─────────────────────────────────────────────────────
    interp_val, interp_conf, interp_evidence = _detect_interpretability(text_lower)
    interp_field = ExtractedField(
        value=interp_val,
        provenance=_make_user_provenance(interp_conf, interp_evidence),
    )

    # ── Latency ──────────────────────────────────────────────────────────────
    latency_val, latency_conf, latency_evidence = _detect_latency(text_lower)
    latency_field = ExtractedField(
        value=latency_val,
        provenance=_make_user_provenance(latency_conf, latency_evidence),
    )

    # ── Exclusions ───────────────────────────────────────────────────────────
    exclusions = _detect_exclusions(problem_statement)
    exclusions_field = ExtractedField(
        value=exclusions if exclusions else None,
        provenance=_make_user_provenance(
            ConfidenceLevel.explicit if exclusions else ConfidenceLevel.unknown, None
        ),
    )

    # ── Fields never fabricated: target, objective, desired_output ────────
    no_value_field = ExtractedField(
        value=None,
        provenance=_make_user_provenance(ConfidenceLevel.unknown, None),
    )

    # Prediction objective — only if explicitly stated with "predict X"
    obj_match = re.search(r"predict\s+([\w\s]{2,40}?)(?:\.|,|\band\b|$)", problem_statement, re.IGNORECASE)
    pred_objective = obj_match.group(1).strip() if obj_match else None
    objective_field = ExtractedField(
        value=pred_objective,
        provenance=_make_user_provenance(
            ConfidenceLevel.explicit if pred_objective else ConfidenceLevel.unknown,
            obj_match.group(0).strip() if obj_match else None,
        ),
    )

    return ParsedProblem(
        domain=domain_field,
        application_area=app_area_field,
        task_type=task_field,
        prediction_objective=objective_field,
        target_variable=no_value_field,    # never inferred from text alone
        desired_output=no_value_field,     # never fabricated
        evaluation_metric=metric_field,
        computational_constraints=no_value_field,
        latency_constraints=latency_field,
        interpretability_requirements=interp_field,
        modality_requirements=modality_field,
        explicit_exclusions=exclusions_field,
        raw_problem_statement=problem_statement,
    )
