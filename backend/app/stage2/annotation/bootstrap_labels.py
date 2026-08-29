"""
bootstrap_labels.py

Stage 2C — Weak Supervision / Bootstrap Label Generator

PURPOSE:
  This module generates WEAK BIO-tagged training examples using the existing
  controlled vocabulary (the legacy MechanismMapper dictionary) as a source
  of seed annotations.

  These labels are:
    - Marked explicitly as bootstrap_weak (ExtractionMethod.bootstrap_weak)
    - Marked with confidence_status = "unresolved"
    - Marked with is_bootstrap = True
    - Given a fixed bootstrap confidence of BOOTSTRAP_CONFIDENCE (0.55)
      which is below LOW_CONFIDENCE_THRESHOLD (0.60), so they always get
      review_flag = True

  They are NEVER:
    - Presented as verified NER output
    - Relabelled as transformer_ner extraction
    - Used to inflate precision/recall numbers in comparison experiments

WHEN TO USE:
  Use bootstrap labels to:
  1. Build initial training data for fine-tuning the NER head
  2. Demonstrate the annotation schema to human annotators
  3. Produce a seed annotation file for manual correction

SCIENTIFIC TRANSPARENCY:
  Every output record explicitly documents its weak-supervision origin.
  The comparison_runner.py reports bootstrap entities separately from
  genuine Transformer NER entities.

Usage:
    generator = BootstrapLabelGenerator()
    entities = generator.generate(
        text="We used SMOTE for class balancing and trained XGBoost...",
        paper_id="paper_abc",
    )
    generator.save_to_jsonl(entities, "annotation/seed_annotations.jsonl")
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from pathlib import Path
from typing import List, Optional

from backend.app.stage2.models import ExtractionMethod, NEREntity
from backend.app.stage2.ner_entity_types import (
    CANONICAL_EXAMPLES, ENTITY_TO_MECHANISM, LOW_CONFIDENCE_THRESHOLD,
    NEREntityType,
)

logger = logging.getLogger(__name__)

# Bootstrap labels are intentionally below the review threshold
BOOTSTRAP_CONFIDENCE: float = 0.55  # < LOW_CONFIDENCE_THRESHOLD → always review_flag=True


class BootstrapLabelGenerator:
    """
    Generates weak BIO-tagged NEREntity records using vocabulary matching.

    This is the bootstrapping stage: it gives the Transformer NER head
    initial training signal before human annotations are available.

    IMPORTANT: These are not Transformer-extracted entities.
    They are vocabulary-matched entities that HELP CREATE training data.
    """

    def __init__(self):
        # Build pattern list: (compiled_regex, entity_type, canonical_text)
        self._patterns: List = []
        for entity_type, examples in CANONICAL_EXAMPLES.items():
            if entity_type == NEREntityType.O:
                continue
            for example in examples:
                # Word-boundary match, case-insensitive
                pattern = re.compile(
                    r"\b" + re.escape(example) + r"\b",
                    re.IGNORECASE,
                )
                self._patterns.append((pattern, entity_type, example))

        # Sort by length of example text (longer matches take priority)
        self._patterns.sort(key=lambda x: len(x[2]), reverse=True)

    def generate(
        self,
        text: str,
        paper_id: str,
        pmid: Optional[str] = None,
        doi: Optional[str] = None,
        section: Optional[str] = None,
    ) -> List[NEREntity]:
        """
        Generate bootstrap NER entities from text using vocabulary matching.

        All entities are explicitly marked:
          - extraction_method = bootstrap_weak
          - is_bootstrap = True
          - confidence = BOOTSTRAP_CONFIDENCE (0.55, below LOW threshold)
          - review_flag = True
          - confidence_status = "unresolved"

        These MUST NOT be presented as Transformer-extracted entities.
        """
        if not text or not text.strip():
            return []

        entities: List[NEREntity] = []
        # Track matched character ranges to avoid overlapping spans
        matched_ranges: List[tuple] = []

        for pattern, entity_type, canonical in self._patterns:
            for match in pattern.finditer(text):
                start, end = match.start(), match.end()

                # Skip if overlaps with an already-matched range
                overlaps = any(
                    not (end <= ms or start >= me)
                    for ms, me in matched_ranges
                )
                if overlaps:
                    continue

                matched_ranges.append((start, end))
                span_text = text[start:end]
                mechanism_cat = ENTITY_TO_MECHANISM.get(
                    entity_type, __import__(
                        "backend.app.stage2.models",
                        fromlist=["MechanismCategory"]
                    ).MechanismCategory.unmapped
                )

                entities.append(NEREntity(
                    entity_id=str(uuid.uuid4()),
                    text=span_text,
                    entity_type=entity_type.value,
                    mechanism_category=mechanism_cat.value,
                    start_char=start,
                    end_char=end,
                    source_text=text,
                    source_section=section,
                    source_paper_id=paper_id,
                    source_pmid=pmid,
                    source_doi=doi,
                    confidence=BOOTSTRAP_CONFIDENCE,
                    confidence_level="LOW",   # always LOW — bootstrap labels
                    review_flag=True,         # always True — requires human review
                    extraction_method=ExtractionMethod.bootstrap_weak,
                    model_version="bootstrap_vocabulary_v1.0",
                    bio_tag=f"B-{entity_type.value}",
                    confidence_status="unresolved",  # always unresolved
                    is_bootstrap=True,
                ))

        logger.info(
            f"Bootstrap generated {len(entities)} weak labels for {paper_id} "
            f"(ALL marked is_bootstrap=True, confidence={BOOTSTRAP_CONFIDENCE})"
        )
        return entities

    def generate_bio_sequence(
        self,
        text: str,
        paper_id: str,
    ) -> List[dict]:
        """
        Generate token-level BIO annotation suitable for NER training.

        Returns list of {"token": str, "label": str} dicts.
        Suitable for conversion to CoNLL or HuggingFace datasets format.
        Marks each token with is_bootstrap=True.
        """
        entities = self.generate(text, paper_id)
        # Simple whitespace tokenization for BIO sequence generation
        tokens = text.split()
        bio_sequence = []
        char_pos = 0

        for token in tokens:
            token_start = text.find(token, char_pos)
            token_end = token_start + len(token)

            # Find entity that covers this token position
            label = "O"
            for ent in entities:
                if token_start >= ent.start_char and token_end <= ent.end_char:
                    if token_start == ent.start_char:
                        label = f"B-{ent.entity_type}"
                    else:
                        label = f"I-{ent.entity_type}"
                    break

            bio_sequence.append({
                "token": token,
                "label": label,
                "is_bootstrap": True,
                "paper_id": paper_id,
            })
            char_pos = token_end

        return bio_sequence

    def save_to_jsonl(
        self,
        entities: List[NEREntity],
        output_path: str,
    ) -> None:
        """
        Save bootstrap entities to JSONL format for annotation review.
        Includes a header comment documenting bootstrap origin.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            # Write provenance header as first line
            header = {
                "_type": "bootstrap_annotation_header",
                "_warning": (
                    "ALL entities in this file are WEAK SUPERVISION bootstrap labels. "
                    "They were generated by vocabulary matching, NOT by a Transformer NER model. "
                    "Human review is REQUIRED before using these for training or reporting. "
                    "extraction_method=bootstrap_weak on all records."
                ),
                "_bootstrap_confidence": BOOTSTRAP_CONFIDENCE,
                "_review_required": True,
            }
            f.write(json.dumps(header) + "\n")

            for entity in entities:
                f.write(entity.model_dump_json() + "\n")

        logger.info(f"Saved {len(entities)} bootstrap labels to {path}")
