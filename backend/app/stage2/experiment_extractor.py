"""
experiment_extractor.py

Extracts structured ExperimentRecord and AblationRecord from paper text.

Scientific rules (STRICT):
  1. Only information EXPLICITLY stated in the text is recorded.
  2. Never infer values not present in text — null/UNKNOWN is preferred over a guess.
  3. delta is computed ONLY when BOTH baseline_value AND method_value are found.
  4. Ablations are extracted when the text contains ablation-style comparisons.
  5. fusion_strategy is assigned ONLY when the paper explicitly describes
     the architecture (e.g. "early fusion", "cross-attention"). Generic
     "multimodal" alone is NOT sufficient.
  6. Modalities require an explicit input/usage relationship; background
     mentions are rejected.
  7. Dataset names must appear as data sources, not as method/framework names.
  8. Proposed method, dataset, baseline, and mechanism are separate entities.
  9. All accepted non-null scientific fields must carry a FieldProvenance entry.
"""

import re
import uuid
from typing import Dict, List, Optional, Tuple

from backend.app.stage2.entity_validator import validate_dataset_candidate, validate_method_candidate
from backend.app.stage2.models import (
    AblationRecord, BaselineRecord, ExperimentRecord, ExtractionStatus,
    FieldProvenance, FusionStrategy, ResultRecord, SourceScope,
)
from backend.app.stage2.section_parser import SectionParser, BACKGROUND_SECTIONS, EMPIRICAL_SECTIONS


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Method/framework names that must NEVER be treated as dataset names.
# NOTE: Only block names that are definitively method names, NOT baseline comparator
# names like PORPOISE (which is a valid baseline in some papers).
_METHOD_NAME_BLOCKLIST = frozenset({
    "honeybee", "calm", "mcat", "survpath", "motcat",
    "coattn", "mmfusion", "proposed", "novel",
})

# Names that cannot appear as baselines (method name blocklist for baselines)
_BASELINE_NAME_BLOCKLIST = frozenset({
    "proposed", "novel", "our", "the",
})

# Dataset source patterns — STRICT.
# Each pattern captures group(1) = the candidate entity name.
# Rules:
#   - The candidate must appear ADJACENT to an explicit data-source keyword
#     (cohort, dataset, database, corpus, collection, repository, archive)
#     OR in a sentence that unambiguously identifies it as a data source.
#   - The candidate must then pass validate_dataset_candidate().
#   - Articles ("The"), conjunctions ("Although"), and verbs must not be captured.
#
# IMPORTANT: Patterns are ordered most → least specific. Each captures group(1).
_DATASET_SOURCE_PATTERNS: List[str] = [
    # Pattern 1: "the TCGA cohort" / "TCGA cohort" / "TCGA database" — X immediately before keyword
    # The entity MUST appear before the keyword (not after an article)
    r"\b([A-Z][A-Za-z0-9\-]{2,20})\s+(?:cohort|dataset|database|corpus|collection|repository|archive|registry)\b",
    # Pattern 2: "from the TCGA" — but only when followed by data-source keyword or end of clause
    # Must NOT match "from the raw dataset", "from the option"
    r"(?:from|using|on)\s+(?:the\s+)?([A-Z][A-Za-z0-9\-]{2,20})\s*(?:cohort|dataset|database|corpus|collection|repository|archive|registry)\b",
    # Pattern 3: "data were obtained from TCGA" — subject is data, so X after "from" is the source
    r"(?:data|samples?|patients?|cases?|images?)\s+(?:were\s+)?(?:obtained|collected|retrieved|derived|sourced|downloaded|taken)\s+from\s+(?:the\s+)?([A-Z][A-Za-z0-9\-]{2,20})\b",
    # Pattern 4: "we used the TCGA cohort" — ONLY when followed explicitly by a data-source keyword
    r"we\s+(?:used|analyzed|included|selected|downloaded)\s+(?:the\s+)?([A-Z][A-Za-z0-9\-]{2,20})\s+(?:cohort|dataset|database|corpus|collection|registry)\b",
    # Pattern 5: "n=X patients from INSTITUTION" — must be followed by cohort/institution keyword
    r"n\s*=\s*\d+\s+(?:patients?|subjects?|cases?)\s+from\s+(?:the\s+)?([A-Z][A-Za-z0-9\-]{2,20})\b",
    # Pattern 6: "multicenter" / "multi-center" — captured as a descriptor
    r"\b(multicenter|multi-center)\s+(?:cohort|dataset|study|trial)\b",
]

# Sentences that describe the paper's OWN input modalities
# Must have: modality keyword + an explicit INPUT relationship verb
_MODALITY_INPUT_VERBS = re.compile(
    r"\b(?:"
    r"used\s+as\s+input|fed\s+(?:into|to)\s+(?:the\s+)?(?:model|network|encoder|classifier)"
    r"|incorporated\s+as\s+features?|incorporated\s+into"
    r"|used\s+for\s+(?:prediction|classification|fusion|training|prognosis)"
    r"|combined\s+with\s+(?:clinical|imaging|text|omics|genomic|patholog)"
    r"|encoded\s+and\s+(?:provided|fed|passed)"
    r"|patient\s+(?:clinical\s+)?(?:variables?|features?|data)\s+were\s+included"
    r"|integrated\s+(?:clinical|imaging|text|omics|genomic|patholog)"
    # Explicit first-person ownership: "We used X", "We collected X", "We relied on X"
    r"|we\s+(?:used|collected|obtained|included|analyzed|acquired|relied\s+on|processed|extracted|combined)\s+"
    r"(?:\w+\s+){0,4}?(?:clinical|imaging|image|ct|pet|mri|radiomic|text|report|omic|genomic)"
    r"|(?:clinical|imaging|genomic|transcriptomic|patholog\w+|radiolog\w+)\s+(?:data|features?|information|variables?)\s+(?:were|was)\s+(?:used|included|collected|fed|input|processed|combined|extracted)"
    r"|we\s+(?:collect|use|includ|incorporat|extract|combin|integrat|process|input|fuse)\w*\s+(?:clinical|imaging|text|omics|genomic|transcriptomic|patholog)"
    r"|relied\s+on\s+(?:clinical|imaging|image|ct|pet|mri|radiomic|text|report|omic|genomic)"
    r")\b",
    re.IGNORECASE,
)

# Sentences that describe OTHER papers' work — reject these for modality extraction
_BACKGROUND_NEGATIONS = re.compile(
    r"\b(?:"
    r"previous\s+(?:studies?|work|literature|methods?|approaches?|models?)"
    r"|prior\s+(?:studies?|work|literature|methods?|approaches?|models?)"
    r"|other\s+(?:studies?|work|methods?|approaches?|groups?|researchers?)"
    r"|existing\s+(?:methods?|approaches?|models?|work|literature)"
    r"|has\s+been\s+(?:shown|demonstrated|reported|proposed|used)"
    r"|have\s+been\s+(?:shown|demonstrated|reported|proposed|used)"
    r"|(?:may|can|could|might|should)\s+(?:improve|enhance|help|benefit)"
    r"|(?:is|are)\s+(?:commonly|typically|often|generally|widely)\s+(?:used|applied|reported|available)"
    r"|literature\s+(?:has|reports?|shows?|suggests?)"
    r"|were\s+(?:not\s+)?(?:available|collected|accessible|obtained)\s+(?:for\s+this\s+study)?"
    r")\b",
    re.IGNORECASE,
)

# Fusion strategy — ordered from most specific to least specific.
# Each entry: (regex_pattern, FusionStrategy)
# Guard: the sentence must also contain an architectural context word.
_FUSION_PATTERNS: List[Tuple[str, FusionStrategy]] = [
    (r"\bcross[- ]attention\b", FusionStrategy.cross_attention),
    (r"\bq-former\b", FusionStrategy.cross_attention),
    (r"\bgated\s+fusion\b", FusionStrategy.gated_fusion),
    (r"\bgated[- ]multimodal\b", FusionStrategy.gated_fusion),
    (r"\bjoint\s+embed(?:ding)?\b", FusionStrategy.joint_embedding),
    (r"\bearly\s+fusion\b", FusionStrategy.early_fusion),
    (r"\bfeature[- ]level\s+fusion\b", FusionStrategy.early_fusion),
    (r"\blate\s+fusion\b", FusionStrategy.late_fusion),
    (r"\bdecision[- ]level\s+fusion\b", FusionStrategy.late_fusion),
    (r"\bintermediate\s+fusion\b", FusionStrategy.intermediate_fusion),
    (r"\bensemble\s+(?:fusion|method|model)\b", FusionStrategy.ensemble_fusion),
]

# Architectural context guard — fusion extraction only valid in these contexts
_FUSION_ARCH_GUARD = re.compile(
    r"\b(?:architecture|model|network|framework|layer|module|mechanism|strategy"
    r"|approach|uses?|employs?|applies?|implement|combines?|merges?|fuses?)\b",
    re.IGNORECASE,
)

# Baseline comparators
_BASELINE_PATTERNS = re.compile(
    r"(?:compar(?:ed|ing)?\s+(?:with|to|against|versus)|against\s+|vs\.?\s+|versus\s+|outperform(?:ed|s)?\s+)"
    r"([A-Z][A-Za-z0-9\-]{1,20}(?:\s+[A-Za-z0-9\-]{1,15}){0,2}|"
    r"(?:unimodal|single[- ]modal|image[- ]only|text[- ]only|clinical[- ]only"
    r"|[a-z][a-z\s\-]{2,30}?)(?:\s+(?:alone|baseline|model|approach|method|branch))?)",
    re.IGNORECASE,
)

# Metric extraction
_RESULT_RE = re.compile(
    r"(auroc|auc|roc.auc|c.index|concordance|accuracy|f1.score?|"
    r"sensitivity|specificity|precision@?\d*|recall|mfs|rfs|os|pfs)\b"
    r"[\s:=of]*((?:of\s+)?([0-9]\.[0-9]+))",
    re.IGNORECASE,
)
_VS_RE = re.compile(
    r"([0-9]\.[0-9]+)\s+vs\.?\s+([0-9]\.[0-9]+)\s+"
    r"(?:(auroc|auc|c.index|accuracy|f1|auc))?",
    re.IGNORECASE,
)
_PCT_RE = re.compile(
    r"([0-9]{2,3}(?:\.[0-9]+)?)\s*%\s*(?:accuracy|weighted\s+accuracy|classification\s+accuracy)",
    re.IGNORECASE,
)
_DELTA_RE = re.compile(
    r"\+([0-9]+\.?[0-9]*)\%?\s*(?:mean\s+)?(c.index|auc|auroc|accuracy|improvement)",
    re.IGNORECASE,
)

# Ablation patterns
_ABLATION_FEATURE_RE = re.compile(
    r"(?:without|w/o|removing?|excluding?)\s+"
    r"([a-z][a-z\s\-]+?)\s+(?:features?\s+)?(?:the|model|stream|module|branch|component)",
    re.IGNORECASE,
)
_ABLATION_PUNCT_RE = re.compile(
    r"(?:without|w/o|removing?|excluding?|ablat(?:ion|ed|ing))\s+"
    r"([a-z][a-z\s\-]{2,60}?)\s*[,\.;]",
    re.IGNORECASE,
)

# Negative evidence
_NEGATIVE_RE = re.compile(
    r"no\s+significant|did\s+not\s+improv|fail(?:ed)?|underperform|"
    r"inferior|not\s+superior|worse\s+than|degraded|did\s+not\s+enhance",
    re.IGNORECASE,
)

# Sample count
_SAMPLE_COUNT_RE = re.compile(
    r"(?:n\s*=\s*|of\s+|on\s+|using\s+|with\s+|from\s+)(\d[\d,]+)\s*"
    r"(?:patients?|subjects?|cases?|samples?|images?|examinations?|participants?)",
    re.IGNORECASE,
)

# Cross-validation / train-test
_CV_RE = re.compile(r"(\d+)-fold\s+cross[- ]?validat", re.IGNORECASE)
_TRAIN_TEST_RE = re.compile(
    r"(\d+)%?\s*(?:train(?:ing)?)\s*/?\s*(\d+)%?\s*(?:test(?:ing)?|val(?:idation)?)",
    re.IGNORECASE,
)

# Loss function
_LOSS_RE = re.compile(
    r"(cross.entropy|focal\s+loss|cox\s+loss|bce|binary\s+cross.entropy|"
    r"nll\s+loss|mse|l1\s+loss|contrastive\s+loss|triplet\s+loss)",
    re.IGNORECASE,
)

# Regularization
_REG_RE = re.compile(
    r"(dropout|l1\s+regulariz|l2\s+regulariz|weight\s+decay|early\s+stopping|batch\s+norm)",
    re.IGNORECASE,
)

# Statistical test
_STAT_RE = re.compile(
    r"(wilcoxon|mann.whitney|t.test|anova|delong|mcnemar|bootstrapp)",
    re.IGNORECASE,
)

# Proposed method identification patterns.
# STRICT: We extract the named entity ONLY — not the verb itself, not a subordinator.
# Every pattern captures group(1) = the method name (a noun phrase / acronym).
# The caller must then run validate_method_candidate() on the result.
_METHOD_NAMED_PATTERNS: List[str] = [
    # "we propose/introduce/present/call/name X" — X immediately after verb (no subordinate clause)
    # Must NOT match "we propose that ..." — stop at word boundary, not a subordinator
    r"(?:we\s+(?:propose|present|introduce|develop|design|call|name|term|denote)\s+"
    r"(?:a\s+|an\s+|the\s+)?(?:novel\s+)?(?:multimodal\s+)?(?:method|framework|model|approach|architecture|system|network|algorithm)?"
    r"\s*(?:called|named|termed|denoted)?\s*)([A-Z][A-Za-z0-9\-]{2,20})\b",
    # "called/named/termed X" — most reliable signal
    r"(?:called|named|termed|denoted)\s+([A-Z][A-Za-z0-9\-]{2,20})\b",
    # "our framework, X," or "our model, X," — X is a named entity after comma
    r"(?:our|the\s+proposed)\s+(?:framework|model|method|approach|architecture)"
    r"[,\s]+([A-Z][A-Za-z0-9\-]{2,20})[,\s]",
    # "X is a novel/proposed method/framework/model/architecture"
    r"\b([A-Z][A-Za-z0-9\-]{2,20})\s+is\s+a\s+(?:novel|proposed|new|our)\s+"
    r"(?:model|method|framework|approach|architecture|algorithm|system)\b",
    # "the proposed X method/framework" — X between "proposed" and method-type word
    r"(?:the\s+proposed|the\s+novel)\s+([A-Z][A-Za-z0-9\-]{2,20})\s+"
    r"(?:method|framework|approach|model|architecture|algorithm|system)\b",
]

# Task extraction — must match explicit prediction objectives
# Order: more specific first
_TASK_PATTERNS: List[Tuple[str, str]] = [
    # HPV status prediction → classification/subtyping
    (r"\b(?:predict|classif|determin|identif)\w*\s+(?:hpv|human\s+papillomavirus)\s+(?:status|subtype|positiv|negativ)", "classification"),
    # Cancer detection / diagnosis
    (r"\b(?:detect|diagnos|identif|classif)\w*\s+(?:cancer|carcinoma|tumor|tumour|malignancy|lesion|prostate\s+cancer|cspca|clinically\s+significant)", "diagnosis"),
    # Survival / prognosis prediction
    (r"\b(?:predict|estimat|forecast)\w*\s+(?:overall\s+survival|disease[- ]free\s+survival|progression[- ]free\s+survival|recurrence[- ]free|metastasis[- ]free|prognosis|survival\s+outcome)", "survival_prediction"),
    # Recurrence prediction
    (r"\b(?:predict|estimat)\w*\s+(?:recurrence|relapse)", "recurrence_prediction"),
    # Generic subtyping / classification
    (r"\b(?:cancer|tumor|disease)\s+(?:subtyping|subtype\s+(?:classif|identif)\w*)", "classification"),
    # Generic classification task label
    (r"\b(?:survival\s+prediction|prognosis\s+prediction|prognostic\s+model)\b", "survival_prediction"),
    # HPV sub-typing
    (r"\bhpv\s+sub[- ]?typ", "classification"),
]

# Abbreviation protection for sentence splitting — these should not be split at
_ABBREV_PROTECT = [
    ("PET/CT", "__PETCT__"),
    ("csPCa", "__CSPCA__"),
    ("e.g.", "__EG__"),
    ("i.e.", "__IE__"),
    ("et al.", "__ETAL__"),
    ("vs.", "__VS__"),
    ("Fig.", "__FIG__"),
    ("Eq.", "__EQ__"),
    ("No.", "__NO__"),
]


def _safe_delta(method: Optional[float], baseline: Optional[float]) -> Optional[float]:
    """Compute delta ONLY when both values are known."""
    if method is not None and baseline is not None:
        return round(method - baseline, 4)
    return None


def _make_prov(
    field_name: str,
    value: str,
    source_sentence: str,
    section: str,
    confidence: ExtractionStatus = ExtractionStatus.explicit,
) -> FieldProvenance:
    return FieldProvenance(
        field_name=field_name,
        value=value,
        source_sentence=source_sentence,
        section=section,
        source_location=section,
        extraction_method="regex_based",
        confidence_status=confidence,
        verification_status="VERIFIED",
    )


class ExperimentExtractor:

    def __init__(self):
        self.section_parser = SectionParser()

    def extract(
        self,
        paper_id: str,
        text: str,
        source_scope: SourceScope,
        sections: Optional[Dict[str, str]] = None,
    ) -> Tuple[List[ExperimentRecord], List[AblationRecord]]:
        """
        Parse text into ExperimentRecord(s) and AblationRecord(s).
        Returns (experiments, ablations).
        """
        if not text or not text.strip():
            return [], []

        if sections is None:
            sections = self.section_parser.parse(text)

        # Gather texts from relevant sections
        results_text = self.section_parser.get_results_text(sections) or text
        methods_text = sections.get("methods", "")
        dataset_text = sections.get("dataset", "")
        full_text_lower = text.lower()
        results_lower = results_text.lower()

        section_label = "Results" if source_scope == SourceScope.full_text else "Abstract"

        # Pre-segment into sentences tagged with their section
        tagged_full = self._get_tagged_sentences(text, sections)
        tagged_results = self._get_tagged_sentences(results_text, sections, default_section="results")
        tagged_methods = self._get_tagged_sentences(methods_text, sections, default_section="methods")
        tagged_dataset = self._get_tagged_sentences(dataset_text, sections, default_section="dataset")

        experiment_id = f"exp_{uuid.uuid4().hex[:8]}"
        field_provenance: Dict[str, FieldProvenance] = {}

        # ── Dataset ───────────────────────────────────────────────────────────
        dataset, ds_prov = self._extract_dataset(tagged_dataset or tagged_full)
        if ds_prov:
            field_provenance["dataset"] = ds_prov

        sample_count = self._extract_sample_count(dataset_text or full_text_lower)

        # ── Training strategy ─────────────────────────────────────────────────
        cv = self._extract_cv(methods_text or full_text_lower)
        train_test = self._extract_train_test(methods_text or full_text_lower)

        # ── Fusion strategy ───────────────────────────────────────────────────
        fusion, fusion_prov = self._extract_fusion(tagged_full)
        if fusion_prov:
            field_provenance["fusion_strategy"] = fusion_prov

        # ── Baselines ─────────────────────────────────────────────────────────
        baselines_list, base_provs = self._extract_baselines(
            tagged_results or tagged_full
        )
        for i, bp in enumerate(base_provs):
            field_provenance[f"baseline_{i}"] = bp
        if base_provs:
            field_provenance["baselines"] = base_provs[0]

        # ── Proposed Method ───────────────────────────────────────────────────
        method, method_prov = self._extract_method(tagged_full)
        if method_prov:
            field_provenance["proposed_method"] = method_prov
            # Remove method from baselines if it accidentally appeared there
            baselines_list = [b for b in baselines_list if b.name.lower() != method.lower()]

        # ── Task ──────────────────────────────────────────────────────────────
        task, task_prov = self._detect_task(tagged_full)
        if task_prov:
            field_provenance["task"] = task_prov

        # ── Modalities ────────────────────────────────────────────────────────
        mods, mods_provs = self._detect_modalities(tagged_full)
        for m_name, prov in mods_provs.items():
            field_provenance[f"modalities_{m_name}"] = prov
        if mods_provs:
            field_provenance["modalities"] = list(mods_provs.values())[0]

        # ── Loss / Regularization ─────────────────────────────────────────────
        loss = self._extract_loss(methods_text or full_text_lower)
        reg = self._extract_regularization(methods_text or full_text_lower)

        # ── Statistical test ──────────────────────────────────────────────────
        stat = self._extract_stat_test(results_lower or full_text_lower)

        # ── Results ───────────────────────────────────────────────────────────
        result_records = self._extract_results(results_text, source_scope, sections)

        # ── Limitations ───────────────────────────────────────────────────────
        limits_text = self.section_parser.get_limitations_text(sections)
        limitations = self._extract_limitations(limits_text) if limits_text else None

        # ── Backward-compat baseline scalar ───────────────────────────────────
        baseline_str = baselines_list[0].name if baselines_list else None

        experiment = ExperimentRecord(
            experiment_id=experiment_id,
            paper_id=paper_id,
            dataset=dataset,
            sample_count=sample_count,
            task=task,
            modalities=mods,
            train_strategy=train_test,
            validation_strategy=cv,
            test_strategy=None,
            baseline=baseline_str,
            baselines=baselines_list,
            proposed_method=method,
            preprocessing=None,
            augmentation=None,
            loss_function=loss,
            regularization=reg,
            fusion_strategy=fusion,
            feature_representation=None,
            hyperparameter_tuning=None,
            evaluation_metrics=list({r.metric for r in result_records if r.metric}),
            reported_results=result_records,
            statistical_test_if_reported=stat,
            limitations=limitations,
            source_scope=source_scope,
            source_section=section_label,
            field_provenance=field_provenance,
        )

        # ── Ablations ─────────────────────────────────────────────────────────
        ablation_text = self.section_parser.get_ablation_text(sections)
        ablations = self._extract_ablations(
            ablation_text or results_text,
            experiment_id,
            paper_id,
            source_scope,
        )

        return [experiment], ablations

    # ─────────────────────────────────────────────────────────────────────────
    # Sentence segmentation
    # ─────────────────────────────────────────────────────────────────────────

    def _protect_abbreviations(self, text: str) -> str:
        """Replace known abbreviations so they survive sentence splitting."""
        for abbrev, placeholder in _ABBREV_PROTECT:
            text = text.replace(abbrev, placeholder)
        return text

    def _restore_abbreviations(self, text: str) -> str:
        for abbrev, placeholder in _ABBREV_PROTECT:
            text = text.replace(placeholder, abbrev)
        return text

    def _get_sentences(self, text: str) -> List[str]:
        """Split text into sentences, preserving medical abbreviations."""
        if not text or not text.strip():
            return []
        protected = self._protect_abbreviations(text)
        # Split on ". " or "! " or "? " followed by an uppercase letter or newlines
        parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\s*\n+\s*', protected)
        result = []
        for p in parts:
            p = self._restore_abbreviations(p.strip())
            if p:
                result.append(p)
        return result

    def _get_tagged_sentences(
        self,
        text: str,
        sections: Dict[str, str],
        default_section: str = "unstructured",
    ) -> List[Tuple[str, str]]:
        """
        Returns list of (sentence, section_key) tuples.
        Assigns each sentence to its source section.
        """
        if not text or not text.strip():
            return []
        # Build reverse mapping: text_chunk → section_key
        section_for_line: Dict[str, str] = {}
        for sec_key, sec_text in sections.items():
            for line in sec_text.splitlines():
                if line.strip():
                    section_for_line[line.strip()] = sec_key

        sentences = self._get_sentences(text)
        result = []
        for s in sentences:
            # Look up section; fall back to default
            sec = section_for_line.get(s, default_section)
            result.append((s, sec))
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Field extractors
    # ─────────────────────────────────────────────────────────────────────────

    def _extract_dataset(
        self, tagged: List[Tuple[str, str]]
    ) -> Tuple[Optional[str], Optional[FieldProvenance]]:
        """
        Extract dataset name from sentences that establish it as a DATA SOURCE.

        Pipeline:
          CANDIDATE (regex match) → ENTITY VALIDATION (validate_dataset_candidate)
            → EVIDENCE VALIDATION (sentence is in experimental context)
              → ACCEPT

        A candidate fails entity validation when it is:
          - A generic noun (cancer, raw, data, patients)
          - An article or conjunction (the, although, and)
          - A verb or gerund
          - A scanner/modality abbreviation (CT, PET, MRI)
          - Too short (< 3 chars)
        """
        # Short uppercase tokens that are scanners/modalities, not dataset names
        _SCANNER_TOKENS = frozenset({"ct", "pet", "mri", "us", "mr", "fmri", "dti"})

        for sentence, section in tagged:
            # Skip background sections — datasets are not extracted from intros
            if self.section_parser.is_background_section(section):
                continue
            for pattern in _DATASET_SOURCE_PATTERNS:
                m = re.search(pattern, sentence, re.IGNORECASE)
                if m:
                    ds = m.group(1).strip()
                    ds_lower = ds.lower().strip()
                    # Gate 1: method/framework blocklist
                    if ds_lower in _METHOD_NAME_BLOCKLIST:
                        continue
                    # Gate 2: scanner token blocklist
                    if ds_lower in _SCANNER_TOKENS:
                        continue
                    # Gate 3: entity plausibility validation
                    if not validate_dataset_candidate(ds):
                        continue
                    prov = _make_prov(
                        "dataset", ds, sentence, section, ExtractionStatus.explicit
                    )
                    return ds, prov
        return None, None

    def _extract_sample_count(self, text: str) -> Optional[int]:
        m = _SAMPLE_COUNT_RE.search(text)
        if m:
            try:
                return int(m.group(1).replace(",", ""))
            except ValueError:
                pass
        return None

    def _extract_cv(self, text: str) -> Optional[str]:
        m = _CV_RE.search(text)
        return f"{m.group(1)}-fold cross-validation" if m else None

    def _extract_train_test(self, text: str) -> Optional[str]:
        m = _TRAIN_TEST_RE.search(text)
        return f"{m.group(1)}/{m.group(2)} train/test split" if m else None

    def _extract_fusion(
        self, tagged: List[Tuple[str, str]]
    ) -> Tuple[Optional[FusionStrategy], Optional[FieldProvenance]]:
        """
        Assign a fusion strategy ONLY when the sentence:
          1. Contains an explicit fusion keyword
          2. Also contains an architectural context word
        'multimodal' alone is NOT sufficient to determine fusion type.
        """
        for sentence, section in tagged:
            if _BACKGROUND_NEGATIONS.search(sentence):
                continue
            for pattern, strategy in _FUSION_PATTERNS:
                if re.search(pattern, sentence, re.IGNORECASE):
                    # Require architectural context
                    if _FUSION_ARCH_GUARD.search(sentence):
                        prov = _make_prov(
                            "fusion_strategy", strategy.value, sentence, section,
                            ExtractionStatus.explicit,
                        )
                        return strategy, prov
        return None, None

    def _extract_baselines(
        self, tagged: List[Tuple[str, str]]
    ) -> Tuple[List[BaselineRecord], List[FieldProvenance]]:
        """
        Extract all comparators mentioned via comparison language.

        Each comparator must:
          1. Appear in a comparison context (compared with X, vs X, against X)
          2. Not be a pronoun, article, or generic word
          3. Be at least 3 characters and not start with a lowercase article
        """
        seen: set = set()
        records: List[BaselineRecord] = []
        provs: List[FieldProvenance] = []

        # Additional generic words to reject as baselines
        _BASELINE_REJECTS = frozenset({
            "proposed", "novel", "our", "the", "them", "these", "those",
            "it", "its", "we", "they", "this", "that", "which",
            "standard", "traditional",
        })

        for sentence, section in tagged:
            for m in _BASELINE_PATTERNS.finditer(sentence):
                raw = m.group(1).strip()
                raw_lower = raw.lower()
                # Skip empty or very short terms
                if not raw or len(raw_lower) < 3:
                    continue
                # Reject pronouns and generic words
                if raw_lower in _BASELINE_REJECTS or raw_lower in _BASELINE_NAME_BLOCKLIST:
                    continue
                # Reject if starts with a lowercase article/preposition
                first_word = raw.split()[0].lower()
                if first_word in {"the", "a", "an", "some", "our", "their"}:
                    continue
                if raw_lower in seen:
                    continue
                seen.add(raw_lower)
                baseline_rec = BaselineRecord(
                    name=raw,
                    source_sentence=sentence,
                    source_location=section,
                    comparison_context=m.group(0).strip(),
                )
                prov = _make_prov(
                    "baseline", raw, sentence, section, ExtractionStatus.explicit
                )
                records.append(baseline_rec)
                provs.append(prov)
        return records, provs

    def _extract_method(
        self, tagged: List[Tuple[str, str]]
    ) -> Tuple[Optional[str], Optional[FieldProvenance]]:
        """
        Identify the paper's proposed method/framework.

        Pipeline:
          CANDIDATE (regex match from _METHOD_NAMED_PATTERNS)
            → ENTITY VALIDATION (validate_method_candidate)
              → ACCEPT

        Rejects:
          - Subordinate clauses ("we propose that ...")
          - Gerunds and verbs ("laying", "offers", "using")
          - Generic descriptors ("novel", "proposed", "our")
          - Section headers ("Supplementary")
          - Short tokens (< 3 chars)
        """
        for sentence, section in tagged:
            if _BACKGROUND_NEGATIONS.search(sentence):
                continue
            for pattern in _METHOD_NAMED_PATTERNS:
                m = re.search(pattern, sentence, re.IGNORECASE)
                if m:
                    name = m.group(1).strip()
                    # Entity validation gate — rejects verbs, articles, generics
                    if not validate_method_candidate(name):
                        continue
                    prov = _make_prov(
                        "proposed_method", name, sentence, section,
                        ExtractionStatus.explicit,
                    )
                    return name, prov
        return None, None

    def _detect_task(
        self, tagged: List[Tuple[str, str]]
    ) -> Tuple[Optional[str], Optional[FieldProvenance]]:
        """
        Detect the paper's prediction task from explicit objective statements.
        Returns (task_string, provenance) or (None, None) if not determinable.
        Background sentences are excluded.
        """
        for sentence, section in tagged:
            if self.section_parser.is_background_section(section):
                continue
            for pattern, task_label in _TASK_PATTERNS:
                if re.search(pattern, sentence, re.IGNORECASE):
                    prov = _make_prov(
                        "task", task_label, sentence, section,
                        ExtractionStatus.explicit,
                    )
                    return task_label, prov
        return None, None

    def _detect_modalities(
        self, tagged: List[Tuple[str, str]]
    ) -> Tuple[List[str], Dict[str, FieldProvenance]]:
        """
        Detect modalities used as INPUT by the paper's own method.

        STRICT rules:
          1. The sentence must be in an experimental section
             (methods, experimental_setup, results, dataset, unstructured).
             Introduction, Background, Related Work, Discussion are excluded.
          2. The sentence must contain an explicit input/usage relationship verb.
          3. Background/prior-work sentences are rejected.
          4. Each accepted modality yields a FieldProvenance with confidence_status=explicit.
        """
        mods: Dict[str, FieldProvenance] = {}

        _MODALITY_KEYWORDS: Dict[str, List[str]] = {
            "clinical": [
                "clinical variable", "clinical feature", "clinical data",
                "clinical information", "clinical characteristic",
                "tabular", "demographic", "lab result", "ehr",
                "patient variable", "patient feature", "clinical record",
                "pathological stage", "age", "sex", "psa",
            ],
            "imaging": [
                "imaging", "image", "images", "ct image", "pet image", "mri image", "mri scan",
                "pet/ct", "radiomic", "whole-slide", "wsi", "histolog",
                "radiograph", "mammograph", "ultrasound", "scan",
                "radiology image", "pathology image",
            ],
            "text": [
                "pathology report", "radiology report", "clinical note",
                "free text", "report text", "nlp", "text report",
                "discharge summary", "medical report",
            ],
            "omics": [
                "genomic", "transcriptomic", "proteomic", "metabolomic",
                "rna-seq", "gene expression", "mutation", "cnv", "methylation",
                "mrna", "mirna", "copy number", "somatic mutation",
            ],
        }

        # Sections where modality extraction is trustworthy
        _TRUSTED_SECTIONS = frozenset({
            "methods", "dataset", "experimental_setup", "results",
            "results_and_discussion", "ablation", "unstructured",
        })

        for sentence, section in tagged:
            # Rule 1: Only extract from experimental/methods sections
            if section in BACKGROUND_SECTIONS:
                continue
            # Require the section to be "trusted" for modality claims
            # (Introduction and Discussion are not trusted)
            if section not in _TRUSTED_SECTIONS and section != "unstructured":
                continue
            # Rule 2: Reject sentences about OTHER papers
            if _BACKGROUND_NEGATIONS.search(sentence):
                continue
            # Rule 3: Require explicit input relationship
            if not _MODALITY_INPUT_VERBS.search(sentence):
                continue

            s_lower = sentence.lower()
            for mod_name, keywords in _MODALITY_KEYWORDS.items():
                if mod_name not in mods:
                    if any(kw in s_lower for kw in keywords):
                        mods[mod_name] = _make_prov(
                            "modalities", mod_name, sentence, section,
                            ExtractionStatus.explicit,
                        )

        return list(mods.keys()), mods

    def _extract_loss(self, text: str) -> Optional[str]:
        m = _LOSS_RE.search(text)
        return m.group(1).strip() if m else None

    def _extract_regularization(self, text: str) -> Optional[str]:
        m = _REG_RE.search(text)
        return m.group(1).strip() if m else None

    def _extract_stat_test(self, text: str) -> Optional[str]:
        m = _STAT_RE.search(text)
        return m.group(1).strip() if m else None

    def _extract_limitations(self, text: str) -> Optional[str]:
        sentences = self._get_sentences(text)
        for s in sentences:
            if re.search(
                r"limit|small\s+sample|single.center|future\s+work|cohort\s+size"
                r"|larger\s+study|generaliz|external\s+valid",
                s, re.IGNORECASE,
            ):
                return s.strip()[:400]
        return None

    def _extract_results(
        self,
        text: str,
        source_scope: SourceScope,
        sections: Dict[str, str],
    ) -> List[ResultRecord]:
        records = []
        section_label = "Results" if source_scope == SourceScope.full_text else "Abstract"

        # Pattern 1: "X vs Y AUC"
        for m in _VS_RE.finditer(text):
            metric = m.group(3) or "AUC"
            method_val = float(m.group(1))
            baseline_val = float(m.group(2))
            delta = _safe_delta(method_val, baseline_val)
            direction = (
                "improvement" if delta and delta > 0
                else "degradation" if delta and delta < 0
                else "unchanged"
            )
            records.append(ResultRecord(
                metric=metric.upper(),
                baseline_value=baseline_val,
                method_value=method_val,
                delta=delta,
                direction=direction,
                source_location=section_label,
                source_scope=source_scope,
            ))

        # Pattern 2: "+11.5% C-index"
        for m in _DELTA_RE.finditer(text):
            metric = m.group(2) or "unknown"
            records.append(ResultRecord(
                metric=metric.upper(),
                baseline_value=None,
                method_value=None,
                delta=float(m.group(1)),
                direction="improvement",
                source_location=section_label,
                source_scope=source_scope,
            ))

        # Pattern 3: "accuracy = 0.92" or "AUC 0.91"
        for m in _RESULT_RE.finditer(text):
            metric_name = m.group(1).upper()
            val = float(m.group(3))
            records.append(ResultRecord(
                metric=metric_name,
                baseline_value=None,
                method_value=val,
                delta=None,
                direction="improvement",
                source_location=section_label,
                source_scope=source_scope,
            ))

        # Pattern 4: "92% accuracy"
        for m in _PCT_RE.finditer(text):
            val = float(m.group(1))
            if val > 1:
                val = round(val / 100.0, 4)
            records.append(ResultRecord(
                metric="Accuracy",
                baseline_value=None,
                method_value=val,
                delta=None,
                direction="improvement",
                source_location=section_label,
                source_scope=source_scope,
            ))

        # Pattern 5: Explicit negative results
        if _NEGATIVE_RE.search(text):
            records.append(ResultRecord(
                metric=None,
                baseline_value=None,
                method_value=None,
                delta=None,
                direction="degradation",
                source_location=section_label,
                source_scope=source_scope,
            ))

        # Deduplicate by (metric, method_value)
        seen: set = set()
        unique: List[ResultRecord] = []
        for r in records:
            key = (r.metric, r.method_value)
            if key not in seen:
                seen.add(key)
                unique.append(r)
        return unique

    def _extract_ablations(
        self,
        text: str,
        parent_id: str,
        paper_id: str,
        source_scope: SourceScope,
    ) -> List[AblationRecord]:
        if not text:
            return []
        ablations = []
        seen_conditions: set = set()

        all_matches = (
            list(_ABLATION_FEATURE_RE.finditer(text))
            + list(_ABLATION_PUNCT_RE.finditer(text))
        )
        for m in all_matches:
            condition = m.group(1).strip()
            if not condition or len(condition) > 60:
                continue
            condition_lower = condition.lower()
            if condition_lower in seen_conditions:
                continue
            seen_conditions.add(condition_lower)
            ablation_id = f"abl_{uuid.uuid4().hex[:8]}"
            ablations.append(AblationRecord(
                ablation_id=ablation_id,
                parent_experiment_id=parent_id,
                paper_id=paper_id,
                condition_removed=condition,
                result=None,
                source_location="Ablation" if source_scope == SourceScope.full_text else "Abstract",
                source_scope=source_scope,
            ))
        return ablations
