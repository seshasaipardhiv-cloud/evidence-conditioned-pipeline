"""
entity_validator.py

Entity-class validation for extracted scientific fields.

Architecture: CANDIDATE → ENTITY VALIDATION → EVIDENCE VALIDATION → ACCEPT

A candidate must pass BOTH:
  1. Context/relationship check (handled by the caller)
  2. Entity-plausibility check (handled here)

This module implements the plausibility check:
  - Is the candidate a named entity? (not a generic word / verb / article)
  - Does it have the right morphological signature for its field?
"""

import re
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Shared reject-lists
# ─────────────────────────────────────────────────────────────────────────────

# Generic English words that must NEVER be accepted as named entities.
# Includes articles, conjunctions, prepositions, common verbs/adjectives,
# and domain-generic nouns that could appear after dataset-source patterns.
_GENERIC_WORD_BLOCKLIST = frozenset({
    # Articles / conjunctions / prepositions
    "the", "a", "an", "and", "or", "but", "although", "however", "while",
    "when", "where", "which", "that", "this", "these", "those", "then",
    "since", "because", "for", "to", "of", "from", "with", "in", "on",
    "at", "by", "as", "if", "its", "our", "we", "they", "it", "is",
    # Generic domain nouns that are not named datasets/methods
    "cancer", "disease", "patients", "patient", "subjects", "samples",
    "images", "image", "data", "dataset", "databases", "cohort", "cohorts",
    "study", "studies", "work", "approach", "approaches", "model", "models",
    "method", "methods", "framework", "frameworks", "system", "systems",
    "analysis", "results", "features", "feature", "information",
    "clinical", "imaging", "omics", "genomic", "text", "raw", "standard",
    "training", "testing", "validation",
    # Common verbs / gerunds that appear after proposal verbs
    "offers", "providing", "laying", "using", "improving", "showing",
    "enabling", "combining", "integrating", "fusing", "extracting",
    "predicting", "detecting", "classifying", "performing", "achieving",
    "presenting", "introducing", "proposing", "developing", "demonstrating",
    "including", "containing", "comprising", "consisting", "representing",
    "capturing", "leveraging", "incorporating", "utilizing", "employing",
    "applying", "extending", "building", "creating", "designing",
    # Adjectives / common descriptors
    "novel", "proposed", "new", "existing", "current", "recent", "previous",
    "improved", "enhanced", "robust", "effective", "efficient", "accurate",
    "multimodal", "unimodal", "multi-modal",
    # Section headers that leak
    "supplementary", "appendix", "figure", "table", "supplemental",
    # Common words that start with uppercase mid-sentence
    "option", "foundation", "basis", "summary",
})

# Named dataset / registry / repository indicators:
# A valid dataset name must look like a proper noun or known registry pattern.
# These are structural rules; individual blocklist entries augment them.

# Pattern for a plausible named entity (dataset or method):
# - Must be 2+ characters
# - Must not be entirely lowercase common words
# - Must start with an uppercase letter OR be an all-caps acronym
# - Must not be a stopword
_PROPER_NOUN_RE = re.compile(
    r'^(?:[A-Z][A-Za-z0-9\-]{1,}|[A-Z]{2,}(?:[0-9\-][A-Za-z0-9\-]*)?)$'
)

# A known-dataset registry pattern: well-known medical/genomic databases.
# If a candidate matches this, it passes even without a PROPER_NOUN shape.
_KNOWN_DATASET_PATTERNS = re.compile(
    r'\b(?:TCGA|TCIA|GEO|NCBI|MIMIC|NLST|BraTS|LiTS|KITS|LIDC|CPTAC|SEER'
    r'|HNSC|BRCA|LUAD|LUSC|STAD|LIHC|KIRC|PAAD|OV|SARC|UCEC|UCSC|CCLE'
    r'|multicenter|multi-center|institutional|private|public)\b',
    re.IGNORECASE,
)

# Named method / framework patterns — well-known acronym-style names
_KNOWN_METHOD_PATTERNS = re.compile(
    r'\b(?:HONeYBEE|CALM|PORPOISE|MCAT|SurvPath|MotCAT|CoAttN|MMFUSION'
    r'|MMFusion|SurvPath|CLAM|TransPath|HiPT|DSMIL|TransMIL|ABMIL)\b',
    re.IGNORECASE,
)

# Words that are clearly English verbs / present participles / gerunds:
# captured by a morphological pattern — ends in -ing, -ed, -ion, -ment
# Note: valid method/dataset names like "Normalization" should not be blocked here
# so we only apply this for method validation, not dataset.
_VERB_GERUND_RE = re.compile(
    r'^(?:lay|offer|use|improv|show|enabl|combin|integrat|fus|extract|'
    r'predict|detect|classif|perform|achiev|present|introduc|propos|develop|'
    r'demonstrat|includ|contain|compris|consist|represent|captur|leverag|'
    r'incorporat|utiliz|employ|appli|extend|build|creat|design|demonstrat|'
    r'provid)(?:ing|ed|es|s)?\b',
    re.IGNORECASE,
)


def validate_dataset_candidate(candidate: str) -> bool:
    """
    Returns True if `candidate` is a plausible named dataset/registry/cohort.
    Returns False if it is a generic word, verb, article, or implausible entity.

    Validation rules:
    1. Not in the generic word blocklist
    2. Length >= 2 characters after stripping
    3. Must start with an uppercase letter (proper noun) OR be a known dataset name
    4. Not a conjunction/article (leading lowercase initial if >3 chars raises suspicion)
    5. Known datasets always pass
    """
    if not candidate or not candidate.strip():
        return False

    cand = candidate.strip()
    cand_lower = cand.lower()

    # Rule 1: blocklist
    if cand_lower in _GENERIC_WORD_BLOCKLIST:
        return False

    # Rule 2: minimum length
    if len(cand) < 2:
        return False

    # Known datasets always pass
    if _KNOWN_DATASET_PATTERNS.search(cand):
        return True

    # Rule 3: must start with uppercase (proper noun rule)
    if not cand[0].isupper():
        return False

    # Rule 4: if the candidate is a single common English word (even capitalized),
    # it likely came from a sentence-initial position — reject.
    # Specifically: single tokens that are in _GENERIC_WORD_BLOCKLIST even with
    # capital first letter are already caught by Rule 1 (lowercase comparison).

    # Rule 5: multi-word candidates — the first word must be a proper noun or acronym
    words = cand.split()
    if len(words) > 1:
        # Multi-word candidates like "Cancer Imaging Archive" are acceptable
        # as long as they don't start with a known article/conjunction
        first_word_lower = words[0].lower()
        if first_word_lower in {"the", "a", "an", "and", "or", "but", "although",
                                "however", "this", "these", "those", "although"}:
            return False
        return True

    # Single-word candidates: must look like a proper noun or acronym
    # (uppercase initial + not a blocked word — already checked above)
    # Additionally, single common English nouns that happen to be capitalized
    # (e.g. "Option", "Cancer", "Although") are caught by blocklist.
    return True


def validate_method_candidate(candidate: str) -> bool:
    """
    Returns True if `candidate` is a plausible named method/framework/model.
    Returns False if it is a common verb, gerund, article, generic descriptor,
    subordinator, or non-entity word.

    Validation rules:
    1. Not in the generic word blocklist
    2. Length >= 3 characters
    3. Not a verb / gerund / present participle
    4. Must start with uppercase OR be a known method name
    5. Not a subordinating conjunction or relative pronoun
    """
    if not candidate or not candidate.strip():
        return False

    cand = candidate.strip()
    cand_lower = cand.lower()

    # Rule 1: blocklist
    if cand_lower in _GENERIC_WORD_BLOCKLIST:
        return False

    # Rule 2: minimum length (method names are at least 3 chars)
    if len(cand) < 3:
        return False

    # Known methods always pass
    if _KNOWN_METHOD_PATTERNS.search(cand):
        return True

    # Rule 3: reject verb/gerund patterns
    if _VERB_GERUND_RE.match(cand_lower):
        return False

    # Rule 4: must start with uppercase
    if not cand[0].isupper():
        return False

    # Rule 5: subordinating conjunctions / relative pronouns
    _SUBORDINATORS = frozenset({
        "that", "which", "where", "when", "while", "although", "because",
        "since", "if", "unless", "whether", "what", "how", "who", "whose",
    })
    if cand_lower in _SUBORDINATORS:
        return False

    # Additional check: if the candidate consists of all common English words
    # (e.g. "Supplementary Information", "Our Model"), reject
    words = cand.split()
    non_proper_count = sum(
        1 for w in words if w.lower() in _GENERIC_WORD_BLOCKLIST
    )
    if non_proper_count == len(words):
        return False

    return True
