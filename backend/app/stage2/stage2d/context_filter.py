"""
context_filter.py

Stage 2D — Contextual Section Classifier & Methodological Intent Filter

Analyzes document structure and sentence context to enforce scientific validity:
  - Methodology Sections (Methods, Materials, Experimental Setup, Implementation): Weight = 1.00
  - Results & Ablations: Weight = 0.85
  - Abstract: Weight = 0.75
  - Introduction & Related Work: Weight = 0.35
  - Discussion & Limitations: Weight = 0.30

Filters out non-methodological noise (e.g. historical survey citations or hypothetical future work).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class SectionContextFilter:
    """
    Classifies section types and calculates contextual methodology relevance.
    """

    METHODOLOGY_KEYWORDS = [
        "method", "methods", "material", "materials", "experimental setup",
        "model", "architecture", "training", "implementation", "pipeline",
        "data preprocessing", "sampling strategy", "loss function", "optimization",
    ]

    RESULTS_KEYWORDS = [
        "result", "results", "ablation", "performance", "evaluation", "benchmark", "validation",
    ]

    BACKGROUND_KEYWORDS = [
        "introduction", "background", "related work", "literature review", "overview",
    ]

    DISCUSSION_KEYWORDS = [
        "discussion", "limitation", "limitations", "future work", "conclusion", "conclusions",
    ]

    def classify_section(self, section_name: Optional[str]) -> Tuple[str, float]:
        """
        Classifies a section header into a standardized category with relevance multiplier.
        Returns: (section_category, relevance_weight)
        """
        if not section_name or not section_name.strip():
            return "ABSTRACT", 0.75

        s = section_name.strip().lower()

        if any(k in s for k in self.METHODOLOGY_KEYWORDS):
            return "METHODOLOGY", 1.00
        elif any(k in s for k in self.RESULTS_KEYWORDS):
            return "RESULTS", 0.85
        elif "abstract" in s:
            return "ABSTRACT", 0.75
        elif any(k in s for k in self.BACKGROUND_KEYWORDS):
            return "BACKGROUND", 0.35
        elif any(k in s for k in self.DISCUSSION_KEYWORDS):
            return "DISCUSSION", 0.30

        return "GENERAL_BODY", 0.60

    def evaluate_sentence_relevance(self, sentence: str, section_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluates a candidate sentence for methodological active usage vs background citation.
        """
        sec_cat, sec_weight = self.classify_section(section_name)

        # Active usage pattern
        is_active = bool(re.search(
            r"\b(?:we\s+(?:use|used|apply|applied|employ|employed|implement|implemented|adopt|adopted|train|trained|evaluated)"
            r"|(?:our|the proposed)\s+(?:model|method|framework|approach|network|architecture)"
            r"|was\s+(?:trained|applied|used|built|optimized)\s+(?:with|using|on))\b",
            sentence,
            re.IGNORECASE,
        ))

        # Citation / background pattern
        is_citation = bool(re.search(
            r"\b(?:et\s+al\.|demonstrated\s+by|proposed\s+by|introduced\s+by|reported\s+in|previous\s+studies|prior\s+work)\b",
            sentence,
            re.IGNORECASE,
        ))

        # Hypothetical / future work pattern
        is_future = bool(re.search(
            r"\b(?:future\s+work|could\s+be|may\s+be\s+extended|planned\s+for|promising\s+direction)\b",
            sentence,
            re.IGNORECASE,
        ))

        if is_active and not is_citation:
            intent = "ACTIVE_METHODOLOGY"
            multiplier = 1.00
        elif is_citation:
            intent = "BACKGROUND_CITATION"
            multiplier = 0.40
        elif is_future:
            intent = "FUTURE_WORK_PROPOSAL"
            multiplier = 0.30
        else:
            intent = "NEUTRAL_STATEMENT"
            multiplier = 0.70

        composite_weight = round(sec_weight * multiplier, 4)

        return {
            "section_category": sec_cat,
            "section_weight": sec_weight,
            "intent": intent,
            "intent_multiplier": multiplier,
            "composite_context_weight": composite_weight,
            "is_actionable_evidence": composite_weight >= 0.50,
        }
