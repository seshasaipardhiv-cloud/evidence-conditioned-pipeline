"""
relation_extractor.py

Stage 2C — Post-NER Relation/Association Extraction

After the Transformer NER pipeline produces NEREntity records, this module
groups entities into structured RelationRecord associations.

Strategy: co-sentence proximity heuristic
  - Entities extracted from the same sentence are candidates for relations.
  - Typed relation rules are applied based on entity type pairs.
  - Confidence = min(entity_a.confidence, entity_b.confidence)
    (conservative: the weakest link bounds the association).

Relation type taxonomy:
  HAS_LOSS       : MODEL_ARCH ↔ LOSS
  HAS_OPTIMIZER  : MODEL_ARCH ↔ OPTIMIZATION
  HAS_REGULARIZATION : MODEL_ARCH ↔ REGULARIZATION
  HAS_PREPROCESSING  : DATASET ↔ PREPROCESSING
  HAS_SAMPLING   : DATASET ↔ SAMPLING
  HAS_EVALUATION : MODEL_ARCH ↔ EVALUATION
  HAS_FEATURE_REPR   : MODEL_ARCH ↔ FEATURE_REPR
  HAS_FUSION     : MODEL_ARCH ↔ FUSION
  CO_OCCURS      : any two entities in the same sentence (generic fallback)

Scientific rules:
  - Relations are only formed between entities in the same sentence.
  - Self-relations (entity_a == entity_b) are excluded.
  - Duplicate pairs (A→B and B→A) are deduplicated to one canonical direction.
  - All RelationRecord objects carry source_paper_id and source_sentence
    for full provenance.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from backend.app.stage2.models import ExtractionMethod, NEREntity, RelationRecord

# ─────────────────────────────────────────────────────────────────────────────
# Typed relation rules: (entity_type_a, entity_type_b) → relation_type
# Order of types within a pair is canonical (A comes first alphabetically).
# ─────────────────────────────────────────────────────────────────────────────

_TYPED_RELATION_RULES: Dict[Tuple[str, str], str] = {
    ("MODEL_ARCH", "LOSS"):           "HAS_LOSS",
    ("MODEL_ARCH", "OPTIMIZATION"):   "HAS_OPTIMIZER",
    ("MODEL_ARCH", "REGULARIZATION"): "HAS_REGULARIZATION",
    ("MODEL_ARCH", "EVALUATION"):     "HAS_EVALUATION",
    ("MODEL_ARCH", "FEATURE_REPR"):   "HAS_FEATURE_REPR",
    ("MODEL_ARCH", "FUSION"):         "HAS_FUSION",
    ("DATASET", "PREPROCESSING"):     "HAS_PREPROCESSING",
    ("DATASET", "SAMPLING"):          "HAS_SAMPLING",
    ("PREPROCESSING", "SAMPLING"):    "CO_PREPROCESSING_SAMPLING",
    ("FEATURE_REPR", "MODEL_ARCH"):   "HAS_FEATURE_REPR",  # reverse
}


def _canonical_pair(
    entity_a: NEREntity, entity_b: NEREntity
) -> Tuple[NEREntity, NEREntity, str]:
    """
    Return (a, b, relation_type).
    Typed rules take precedence; falls back to CO_OCCURS.
    Canonical pair order: MODEL_ARCH > DATASET > others (alphabetical tiebreak).
    """
    ta, tb = entity_a.entity_type, entity_b.entity_type

    # Try typed rule with both orders
    if (ta, tb) in _TYPED_RELATION_RULES:
        return entity_a, entity_b, _TYPED_RELATION_RULES[(ta, tb)]
    if (tb, ta) in _TYPED_RELATION_RULES:
        return entity_b, entity_a, _TYPED_RELATION_RULES[(tb, ta)]

    # Generic CO_OCCURS — alphabetical order for deduplication
    if ta <= tb:
        return entity_a, entity_b, "CO_OCCURS"
    else:
        return entity_b, entity_a, "CO_OCCURS"


class RelationExtractor:
    """
    Groups NEREntity records produced by TransformerNERPipeline into
    structured RelationRecord associations using co-sentence heuristic.
    """

    def extract(
        self,
        entities: List[NEREntity],
    ) -> List[RelationRecord]:
        """
        Given a list of NEREntity objects (potentially from multiple papers),
        produce RelationRecord objects.

        Groups entities by (paper_id, source_text) to find co-sentence pairs.
        """
        if not entities:
            return []

        # Group by (paper_id, source_sentence)
        sentence_groups: Dict[Tuple[str, str], List[NEREntity]] = defaultdict(list)
        for entity in entities:
            key = (entity.source_paper_id, entity.source_text)
            sentence_groups[key].append(entity)

        relations: List[RelationRecord] = []
        seen_pairs: Set[Tuple[str, str]] = set()  # deduplication

        for (paper_id, sentence), ents in sentence_groups.items():
            if len(ents) < 2:
                continue  # No pairs possible

            # Generate all unordered pairs
            for i in range(len(ents)):
                for j in range(i + 1, len(ents)):
                    ea, eb = ents[i], ents[j]

                    # Skip if same entity type AND same text (near-duplicate)
                    if ea.entity_type == eb.entity_type and ea.text == eb.text:
                        continue

                    a, b, rel_type = _canonical_pair(ea, eb)

                    # Deduplication key: canonical text pair in sentence
                    dedup_key = (a.entity_id, b.entity_id)
                    if dedup_key in seen_pairs:
                        continue
                    seen_pairs.add(dedup_key)

                    confidence = min(a.confidence, b.confidence)

                    relations.append(RelationRecord(
                        relation_id=str(uuid.uuid4()),
                        entity_a_id=a.entity_id,
                        entity_a_text=a.text,
                        entity_a_type=a.entity_type,
                        entity_b_id=b.entity_id,
                        entity_b_text=b.text,
                        entity_b_type=b.entity_type,
                        relation_type=rel_type,
                        confidence=confidence,
                        source_paper_id=paper_id,
                        source_sentence=sentence,
                        extraction_method=ExtractionMethod.transformer_ner,
                    ))

        return relations

    def get_typed_relations(
        self,
        relations: List[RelationRecord],
        relation_type: str,
    ) -> List[RelationRecord]:
        """Filter relations by type."""
        return [r for r in relations if r.relation_type == relation_type]

    def summary_stats(self, relations: List[RelationRecord]) -> Dict[str, int]:
        """Return count per relation type."""
        counts: Dict[str, int] = defaultdict(int)
        for r in relations:
            counts[r.relation_type] += 1
        return dict(counts)
