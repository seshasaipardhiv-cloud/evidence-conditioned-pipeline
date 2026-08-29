import re
from typing import List, Optional, Tuple
from backend.app.stage2.models import MechanismCategory


# Verbs that establish a mechanism is actively USED in the paper's own method.
# Background/citation sentences (e.g. "X et al. used PCA") are NOT sufficient.
_USAGE_VERBS = re.compile(
    r"\b(?:we\s+(?:use|used|apply|applied|employ|employed|implement|implemented|adopt|adopted|train|trained)"
    r"|(?:our|the proposed)\s+(?:model|method|framework|approach|network|architecture)"
    r"|is\s+(?:trained|applied|used|built)\s+(?:with|on|using)"
    r"|using\s+[a-z\s]{0,30}?(?=\bpca\b|\bcnn\b|\bresnet\b|\btransformer\b|\bautoencoder\b"
    r"|\bdropout\b|\blasso\b|\bsmote\b))\b",
    re.IGNORECASE,
)


class MechanismMapper:
    def __init__(self):
        # A simple keyword to category mapping for the controlled vocabulary.
        # Keys are LOWERCASE canonical phrases; matched with strict word boundaries.
        self.vocabulary = {
            "cnn": MechanismCategory.representation,
            "resnet": MechanismCategory.representation,
            "transformer": MechanismCategory.representation,
            "autoencoder": MechanismCategory.representation,
            "pca": MechanismCategory.representation,
            "principal component analysis": MechanismCategory.representation,
            "smote": MechanismCategory.sampling,
            "dropout": MechanismCategory.regularization,
            "l1 regularization": MechanismCategory.regularization,
            "l2 regularization": MechanismCategory.regularization,
            "weight decay": MechanismCategory.regularization,
            "early stopping": MechanismCategory.regularization,
            "cross entropy": MechanismCategory.loss,
            "cross-entropy": MechanismCategory.loss,
            "focal loss": MechanismCategory.loss,
            "random forest": MechanismCategory.classifier,
            "svm": MechanismCategory.classifier,
            "logistic regression": MechanismCategory.classifier,
            "xgboost": MechanismCategory.classifier,
            "late fusion": MechanismCategory.fusion,
            "early fusion": MechanismCategory.fusion,
            "cross-attention": MechanismCategory.attention,
            "cross attention": MechanismCategory.attention,
            "self-attention": MechanismCategory.attention,
            "self attention": MechanismCategory.attention,
            "bagging": MechanismCategory.ensembling,
            "boosting": MechanismCategory.ensembling,
            "imputation": MechanismCategory.preprocessing,
            "normalization": MechanismCategory.preprocessing,
            "mutual information": MechanismCategory.feature_selection,
            "lasso": MechanismCategory.feature_selection,
            "platt scaling": MechanismCategory.calibration,
            "isotonic regression": MechanismCategory.calibration,
        }

    def map_mechanism(self, raw_text: str) -> Tuple[MechanismCategory, str]:
        """
        Maps a raw textual mechanism description to a controlled Category and
        canonical name using strict word-boundary matching.

        IMPORTANT: This method does NOT require usage context — it is used
        when the calling code has already confirmed relevance.  Use
        map_mechanism_in_context() for full contextual validation.

        Returns (Category, canonical_name).
        If unmapped, returns (MechanismCategory.unmapped, raw_text).
        """
        raw_lower = raw_text.lower().strip()

        # Prefer longer matches first (avoid "l1" matching inside "l1 regularization")
        for key in sorted(self.vocabulary, key=len, reverse=True):
            category = self.vocabulary[key]
            # Strict word-boundary match: "pca" must not match inside "cspca"
            pattern = r"\b" + re.escape(key) + r"\b"
            if re.search(pattern, raw_lower):
                return category, key

        return MechanismCategory.unmapped, raw_text

    def map_mechanism_in_context(
        self, raw_text: str, context_sentence: str
    ) -> Tuple[MechanismCategory, str]:
        """
        Full contextual validation: maps a mechanism only when the
        context_sentence contains a usage verb establishing that the
        paper's OWN method employs this mechanism.

        Background/citation sentences that merely mention a technique
        are NOT sufficient.

        Returns (Category, canonical_name) or (UNMAPPED, raw_text).
        """
        # First check word-boundary match
        category, canonical = self.map_mechanism(raw_text)
        if category == MechanismCategory.unmapped:
            return category, raw_text

        # Then require usage context
        if not _USAGE_VERBS.search(context_sentence):
            return MechanismCategory.unmapped, raw_text

        return category, canonical


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2C: Transformer-Based Mechanism Mapper
# ──────────────────────────────────────────────────────────────────────────────

class TransformerMechanismMapper:
    """
    Mechanism mapper backed by the Stage 2C SciBERT NER pipeline.

    This is NOT a wrapper around the legacy MechanismMapper.  It calls
    TransformerNERPipeline directly and returns NEREntity records with:
      - extraction_method = transformer_ner
      - per-entity confidence scores
      - review_flag for uncertain extractions
      - full character-level provenance

    Scientific rules:
      - If the Transformer model is unavailable, returns empty list and logs
        a clear error. NEVER calls regex extraction and labels it as NER.
      - Entities below LOW_CONFIDENCE_THRESHOLD have review_flag = True.
      - UNMAPPED entities are returned as-is (entity_type = "O", mechanism
        category = UNMAPPED) rather than silently being upgraded.
    """

    def __init__(self):
        # Lazy import to avoid circular and to allow the rest of the system
        # to function even when torch/transformers are not installed.
        from backend.app.stage2.transformer_ner import TransformerNERPipeline
        self._pipeline = TransformerNERPipeline()
        self._legacy_mapper = MechanismMapper()  # kept for comparison only

    @property
    def model_available(self) -> bool:
        return self._pipeline.model_available

    def extract_entities(
        self,
        text: str,
        paper_id: str,
        pmid: Optional[str] = None,
        doi: Optional[str] = None,
        section: Optional[str] = None,
    ) -> "List":
        """
        Extract NEREntity records from text using SciBERT.

        Returns List[NEREntity].
        If model unavailable: returns [] with a logged error (no regex fallback).
        """
        from backend.app.stage2.models import NEREntity
        return self._pipeline.extract(
            text=text,
            paper_id=paper_id,
            pmid=pmid,
            doi=doi,
            section=section,
        )

    def extract_entities_with_relations(
        self,
        text: str,
        paper_id: str,
        pmid: Optional[str] = None,
        doi: Optional[str] = None,
        section: Optional[str] = None,
    ) -> "Tuple":
        """
        Extract NEREntity records AND RelationRecord associations.

        Returns (List[NEREntity], List[RelationRecord]).
        """
        from backend.app.stage2.relation_extractor import RelationExtractor
        entities = self.extract_entities(text, paper_id, pmid, doi, section)
        relations = RelationExtractor().extract(entities)
        return entities, relations

    def legacy_map_for_comparison(
        self, raw_text: str, context_sentence: str
    ) -> "Tuple[MechanismCategory, str]":
        """
        Run the legacy regex mapper.  For comparison experiments only.
        Result is explicitly labelled as regex_based in comparison reports.
        """
        return self._legacy_mapper.map_mechanism_in_context(raw_text, context_sentence)

