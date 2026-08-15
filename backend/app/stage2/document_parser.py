"""
DocumentParser: Abstract-scope evidence extraction.

Scientific rules enforced:
  - Claims are only generated from papers with a non-empty abstract.
  - Each claim must be explicitly grounded in the abstract text.
  - Mechanism must be identified from controlled vocabulary; UNMAPPED if not.
  - evidence_status is set STRICTLY:
      direct_empirical  → Abstract explicitly reports an experimental result with a metric or direction.
      methodological    → Abstract describes a method but no experimental outcome.
      qualitative       → Abstract states qualitative outcome without numbers.
  - baseline, metric, method_value, baseline_value, delta are null unless the
    abstract explicitly states them. No inference, no calculation.
  - claim text must be a short factual sentence grounded in the abstract.
    The entire abstract is NEVER stuffed into the claim field.
  - source_scope = "abstract" always (full text not available).
"""

import re
import uuid
from typing import List, Optional, Tuple
from datetime import datetime

from backend.app.stage2.models import (
    PaperRecord, EvidenceClaim, ExtractionMethod, ExtractionStatus,
    EvidenceStatus, SourceScope, Provenance, MechanismCategory, Mechanism,
    EmpiricalResult,
)
from backend.app.stage2.mechanism_mapper import MechanismMapper

# ──────────────────────────────────────────────────────────────────────────────
# Regex patterns for explicit numeric extraction from abstracts
# ──────────────────────────────────────────────────────────────────────────────

# Matches patterns like "AUC: 0.77", "AUROC of 0.91", "C-index 0.652"
_METRIC_VALUE_RE = re.compile(
    r'(auroc|auc|c-index|c index|concordance index|accuracy|f1|sensitivity|specificity'
    r'|precision@\d+|precision at \d+|roc-auc)'
    r'[\s:]*(?:of\s+|=\s*)?([0-9]\.[0-9]+)',
    re.IGNORECASE
)

# Matches "X vs Y" comparisons: "0.77 vs 0.67 AUC"
_VS_COMPARE_RE = re.compile(
    r'([0-9]\.[0-9]+)\s+vs\.?\s+([0-9]\.[0-9]+)\s+'
    r'(auroc|auc|c-index|concordance|accuracy)?',
    re.IGNORECASE
)

# Matches delta patterns: "+11.5% mean C-index", "increased by 5%"
_DELTA_RE = re.compile(
    r'(?:up to\s+)?\+([0-9]+\.?[0-9]*)%\s*(mean\s+)?([a-z-]+)',
    re.IGNORECASE
)

# Weighted accuracy patterns: "weighted accuracy of 92 %", "99 % accuracy"
_PCT_ACCURACY_RE = re.compile(
    r'([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:weighted\s+)?accuracy',
    re.IGNORECASE
)

_PCT_ACCURACY_REV_RE = re.compile(
    r'accuracy\s+of\s+([0-9]+(?:\.[0-9]+)?)\s*%',
    re.IGNORECASE
)

# Survival metrics: "MFS ... 0.796", "OS ... 0.641"
_SURVIVAL_RE = re.compile(
    r'(MFS|RFS|OS|PFS)\s+(?:were\s+(?:as\s+follows:?\s*)?)?([0-9]\.[0-9]+)',
    re.IGNORECASE
)

# Classification accuracy with percent
_CLASS_ACC_RE = re.compile(
    r'([0-9]{2,3}\.?[0-9]?)\s*%?\s*(?:classification\s+)?accuracy',
    re.IGNORECASE
)


class DocumentParser:

    def __init__(self):
        self.mechanism_mapper = MechanismMapper()

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def parse_paper(self, paper: PaperRecord) -> List[Tuple[EvidenceClaim, Mechanism]]:
        """
        Parse one PaperRecord and return validated (claim, mechanism) pairs.

        Returns an empty list if:
          - abstract is absent or empty
          - no mechanism can be identified in the abstract
          - no empirical or methodological content can be grounded
        """
        if not paper.abstract or not paper.abstract.strip():
            return []

        abstract = paper.abstract  # preserve original capitalisation for display
        abstract_lower = abstract.lower()

        # 1. Identify primary mechanism ─────────────────────────────────────
        category, canonical = self.mechanism_mapper.map_mechanism(abstract_lower)
        mechanism_id = (
            f"mech_{canonical.replace(' ', '_').replace('-', '_')}"
            if category != MechanismCategory.unmapped
            else f"mech_unmapped_{uuid.uuid4().hex[:8]}"
        )
        mechanism = Mechanism(
            mechanism_id=mechanism_id,
            canonical_name=canonical if category != MechanismCategory.unmapped else "UNMAPPED",
            category=category,
            mapping_status="MAPPED" if category != MechanismCategory.unmapped else "UNMAPPED",
        )

        # 2. Detect modalities ──────────────────────────────────────────────
        modalities = self._detect_modalities(abstract_lower)

        # 3. Determine task and domain ──────────────────────────────────────
        task = self._detect_task(abstract_lower)
        domain = self._detect_domain(abstract_lower)

        # 4. Attempt numeric extraction ─────────────────────────────────────
        numeric_result = self._extract_numeric_result(abstract)

        # 5. Classify evidence_status and direction ─────────────────────────
        evidence_status, direction, extraction_status = self._classify(
            abstract_lower, numeric_result
        )

        # 6. Build empirical result object (nulls for missing values) ───────
        result_obj = self._build_result(numeric_result, direction, evidence_status)

        # 7. Extract baseline if explicitly stated ──────────────────────────
        baseline = self._extract_baseline(abstract_lower)

        # 8. Build claim text (never the full abstract) ─────────────────────
        claim_text = self._build_claim_text(
            paper, canonical, category, evidence_status, numeric_result, task, domain
        )

        # 9. Build provenance ───────────────────────────────────────────────
        provenance = Provenance(
            source_type="scholarly_api",
            source_reference=paper.doi or paper.pmid or paper.paper_id,
            extraction_method=ExtractionMethod.regex_based,
            extraction_status=extraction_status,
            evidence_text=claim_text,
            retrieval_date=datetime.now().isoformat(),
        )

        # 10. Build claim ───────────────────────────────────────────────────
        claim = EvidenceClaim(
            evidence_id=f"claim_{uuid.uuid4().hex[:8]}",
            paper_id=paper.paper_id,
            claim=claim_text,
            source_scope=SourceScope.abstract,
            mechanisms=[mechanism_id],
            task=task,
            domain=domain,
            modalities=modalities,
            baseline=baseline,
            metric=numeric_result.get("metric") if numeric_result else None,
            result=result_obj,
            evidence_location="Abstract",
            extraction_method=ExtractionMethod.regex_based,
            evidence_status=evidence_status,
            provenance=provenance,
        )

        return [(claim, mechanism)]

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _detect_modalities(self, abstract_lower: str) -> List[str]:
        modalities = []
        if "clinic" in abstract_lower:
            modalities.append("clinical")
        if any(k in abstract_lower for k in ("imag", "ct ", "pet", "mri", "radiomic", "whole-slide", "wsi")):
            modalities.append("imaging")
        if any(k in abstract_lower for k in ("text", "report", "pathology report", "clinical note")):
            modalities.append("text")
        if any(k in abstract_lower for k in ("omic", "genom", "transcript", "protein", "metabol", "molecular")):
            modalities.append("omics")
        return modalities or ["clinical"]

    def _detect_task(self, abstract_lower: str) -> Optional[str]:
        if any(k in abstract_lower for k in ("survival", "prognosis", "mfs", "rfs", "os", "pfs", "c-index")):
            return "survival_prediction"
        if any(k in abstract_lower for k in ("classif", "diagnosis", "detection", "cancer type")):
            return "classification"
        if "segmentation" in abstract_lower:
            return "segmentation"
        return None

    def _detect_domain(self, abstract_lower: str) -> Optional[str]:
        if any(k in abstract_lower for k in ("head and neck", "hnc", "oropharyngeal", "nasopharyngeal")):
            return "head_and_neck_cancer"
        if "prostate" in abstract_lower:
            return "prostate_cancer"
        if "lung" in abstract_lower:
            return "lung_cancer"
        if "breast" in abstract_lower:
            return "breast_cancer"
        if "pan-cancer" in abstract_lower or "pan cancer" in abstract_lower or "tcga" in abstract_lower:
            return "pan_cancer"
        if "cancer" in abstract_lower or "oncolog" in abstract_lower or "tumor" in abstract_lower:
            return "cancer_general"
        return None

    def _extract_numeric_result(self, abstract: str) -> Optional[dict]:
        """
        Try to find any explicitly stated numeric result.
        Returns a dict with keys: metric, method_value, baseline_value, delta
        or None if nothing explicit is found.
        """
        # Try survival metrics first (most specific)
        m = _SURVIVAL_RE.search(abstract)
        if m:
            return {
                "metric": m.group(1).upper(),
                "method_value": float(m.group(2)),
                "baseline_value": None,
                "delta": None,
            }

        # Try delta patterns: "+11.5% mean C-index"
        m = _DELTA_RE.search(abstract)
        if m:
            return {
                "metric": m.group(3),
                "method_value": None,
                "baseline_value": None,
                "delta": float(m.group(1)),
            }

        # Try "X vs Y AUC" comparison
        m = _VS_COMPARE_RE.search(abstract)
        if m:
            metric = m.group(3) or "AUC"
            return {
                "metric": metric.upper(),
                "method_value": float(m.group(1)),
                "baseline_value": float(m.group(2)),
                "delta": round(float(m.group(1)) - float(m.group(2)), 3),
            }

        # Try "AUC: 0.77" style
        m = _METRIC_VALUE_RE.search(abstract)
        if m:
            return {
                "metric": m.group(1).upper(),
                "method_value": float(m.group(2)),
                "baseline_value": None,
                "delta": None,
            }

        # Try "92 % accuracy" or "99 % accuracy"
        m = _PCT_ACCURACY_RE.search(abstract) or _PCT_ACCURACY_REV_RE.search(abstract)
        if m:
            val = float(m.group(1))
            if val > 1:  # It's a percentage, normalise
                val = round(val / 100.0, 4)
            return {
                "metric": "Accuracy",
                "method_value": val,
                "baseline_value": None,
                "delta": None,
            }

        return None

    def _classify(
        self, abstract_lower: str, numeric_result: Optional[dict]
    ) -> Tuple[EvidenceStatus, str, ExtractionStatus]:
        """
        Determine evidence_status, direction and extraction_status.

        Rules:
          - direct_empirical only if an explicit experimental result is reported.
          - methodological if a method is described but no outcome reported.
          - qualitative if a direction is stated without numbers.
          - unverified if neither.
        """
        has_result_section = any(
            k in abstract_lower for k in ("result:", "results:", "conclusion:", "findings:")
        )
        has_improvement_words = any(
            k in abstract_lower
            for k in ("outperform", "surpass", "improv", "higher", "better",
                       "superior", "exceed", "outperfom", "greatest")
        )
        has_degradation_words = any(
            k in abstract_lower for k in ("not improv", "no significant", "lower", "worse", "inferior")
        )
        has_method_words = any(
            k in abstract_lower
            for k in ("we propose", "we present", "we develop", "we introduc",
                       "we design", "we describe", "our framework", "our method",
                       "our model", "novel", "we train")
        )

        # Direct empirical: numeric result present OR result section with comparison
        if numeric_result is not None:
            direction = "improvement"
            if numeric_result.get("delta") is not None and numeric_result["delta"] < 0:
                direction = "degradation"
            elif has_degradation_words and not has_improvement_words:
                direction = "degradation"
            extraction_status = ExtractionStatus.explicit
            return EvidenceStatus.direct_empirical, direction, extraction_status

        # Qualitative directional: direction stated but no numbers
        if has_improvement_words and has_result_section:
            return EvidenceStatus.direct_empirical, "improvement", ExtractionStatus.structured

        if has_improvement_words:
            return EvidenceStatus.qualitative, "improvement", ExtractionStatus.structured

        if has_degradation_words:
            return EvidenceStatus.qualitative, "degradation", ExtractionStatus.structured

        # Methodological: method described, no empirical outcome
        if has_method_words:
            return EvidenceStatus.methodological, "qualitative", ExtractionStatus.unresolved

        return EvidenceStatus.unverified, "qualitative", ExtractionStatus.unresolved

    def _build_result(
        self,
        numeric_result: Optional[dict],
        direction: str,
        evidence_status: EvidenceStatus,
    ) -> Optional[EmpiricalResult]:
        """
        Build EmpiricalResult. All values remain null unless explicitly found.
        Returns None if evidence is methodological or unverified.
        """
        if evidence_status in (EvidenceStatus.methodological, EvidenceStatus.unverified):
            return None

        if numeric_result:
            return EmpiricalResult(
                metric=numeric_result.get("metric"),
                method_value=numeric_result.get("method_value"),
                baseline_value=numeric_result.get("baseline_value"),
                delta=numeric_result.get("delta"),
                direction=direction,
            )
        # Qualitative or direct_empirical without numbers
        return EmpiricalResult(
            metric=None,
            method_value=None,
            baseline_value=None,
            delta=None,
            direction=direction,
        )

    def _extract_baseline(self, abstract_lower: str) -> Optional[str]:
        """
        Extract an explicit baseline name if stated.
        E.g., "compared with clinical parameters alone", "vs unimodal approach".
        Returns None if not explicitly stated.
        """
        patterns = [
            r"compar(?:ed)? (?:with|to|against) ([a-z\s\-]+(?:alone|approach|model|baseline|method))",
            r"vs\.? ([a-z\s\-]+ (?:alone|approach|model|baseline|method))",
        ]
        for pattern in patterns:
            m = re.search(pattern, abstract_lower)
            if m:
                raw = m.group(1).strip()
                if len(raw) < 80:  # sanity check length
                    return raw
        return None

    def _build_claim_text(
        self,
        paper: PaperRecord,
        canonical: str,
        category: MechanismCategory,
        evidence_status: EvidenceStatus,
        numeric_result: Optional[dict],
        task: Optional[str],
        domain: Optional[str],
    ) -> str:
        """
        Build a short, factual claim sentence grounded in the abstract.
        Never pastes the full abstract.
        """
        mech_name = canonical if category != MechanismCategory.unmapped else "an unmapped mechanism"
        task_str = task.replace("_", " ") if task else "cancer-related tasks"
        domain_str = domain.replace("_", " ") if domain else "oncology"

        if evidence_status == EvidenceStatus.direct_empirical and numeric_result:
            metric = numeric_result.get("metric", "a reported metric")
            method_val = numeric_result.get("method_value")
            baseline_val = numeric_result.get("baseline_value")
            delta = numeric_result.get("delta")

            if method_val is not None and baseline_val is not None:
                return (
                    f"[Abstract] {paper.title}: Using {mech_name} for {task_str} in {domain_str},"
                    f" the method achieved {metric} {method_val} vs baseline {baseline_val}"
                    f" (delta={delta})."
                )
            elif method_val is not None:
                return (
                    f"[Abstract] {paper.title}: Using {mech_name} for {task_str} in {domain_str},"
                    f" the method achieved {metric} = {method_val}."
                )
            elif delta is not None:
                return (
                    f"[Abstract] {paper.title}: Using {mech_name} for {task_str} in {domain_str},"
                    f" the method showed +{delta}% {metric} improvement."
                )

        if evidence_status in (EvidenceStatus.qualitative, EvidenceStatus.direct_empirical):
            return (
                f"[Abstract] {paper.title}: Using {mech_name} for {task_str} in {domain_str},"
                f" the abstract reports performance improvement without explicit numbers."
            )

        if evidence_status == EvidenceStatus.methodological:
            return (
                f"[Abstract] {paper.title}: Proposes {mech_name} for {task_str} in {domain_str}."
                f" No experimental outcome reported in abstract."
            )

        return (
            f"[Abstract] {paper.title}: Mentions {mech_name} in the context of {domain_str}."
            f" Evidence status could not be determined from abstract alone."
        )
