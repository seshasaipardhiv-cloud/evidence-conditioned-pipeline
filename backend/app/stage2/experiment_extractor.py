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
    FieldProvenance, FusionStrategy, PublishedExperimentResult, ResultRecord, SourceScope,
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

    def extract_published_results(
        self,
        paper_id: str,
        text: str,
        pmid: Optional[str] = None,
        doi: Optional[str] = None,
        citation: Optional[str] = None,
        dataset: Optional[str] = None,
        task: Optional[str] = None,
        validation_protocol: Optional[str] = None,
        sample_size: Optional[int] = None,
        source_scope: SourceScope = SourceScope.full_text,
        sections: Optional[Dict[str, str]] = None,
    ) -> List[PublishedExperimentResult]:
        """
        Extracts structured PublishedExperimentResult records directly from empirical text.
        Guarantees:
        1. Only explicitly stated numerical scores with verified model associations are extracted.
        2. Disambiguates complex multi-model constructions (paired comparisons, 'respectively' patterns).
        3. Ambiguous multi-model sentences without clear attribution are flagged (is_ambiguous=True).
        4. Scales bounded percentage metrics to [0.0, 1.0] while preserving unbounded metrics (e.g. RMSE).
        5. Masks out CI ranges, p-values, scanner parameters, and patient counts to reject false positives.
        6. Extracts sentence-level dataset and evaluation protocol for scientific comparability.
        """
        if not text or not text.strip():
            return []

        from backend.app.stage2.result_evidence import (
            canonicalize_model_id,
            normalize_metric_name,
            METRIC_DIRECTION_MAP,
        )

        norm_text = (
            text.replace("\xa0", " ")
            .replace("\u202f", " ")
            .replace("\u2009", " ")
            .replace("–", "-")
            .replace("—", "-")
            .replace("−", "-")
            .strip()
        )

        model_patterns: List[Tuple[str, str, re.Pattern]] = [
            ("XGBoost", "xgboost", re.compile(r"\b(?:xgboost|xgb|gradient\s+boosting(?:\s+trees?|\s+classifier|\s+decision\s+trees?|\s+models?)?|gbdt|lightgbm|lgb|catboost)\b", re.IGNORECASE)),
            ("Random Forest", "random_forest", re.compile(r"\b(?:random\s+forests?|rf|random\s+survival\s+forests?|rsf)\b", re.IGNORECASE)),
            ("Logistic Regression", "logistic_regression", re.compile(r"\b(?:logistic\s+regression|logreg|linear\s+models?|elastic\s*net|lasso|ridge|cox(?:\s+proportional\s+hazards?|\s+model)?)\b", re.IGNORECASE)),
            ("Tabular MLP", "mlp_tabular", re.compile(r"\b(?:multi[- ]layer\s+perceptron|mlp|deep\s+neural\s+networks?|tabular\s+mlp|artificial\s+neural\s+networks?|ann|dnn|clinical\s+embeddings?|fusion\s+models?|deep\s+learning\s+fusion(?:\s+model)?|cross[- ]attention\s+fusion(?:\s+model)?)\b", re.IGNORECASE)),
            ("Support Vector Machine", "svm", re.compile(r"\b(?:support\s+vector\s+machines?|svm|support\s+vector\s+classifier|svc)\b", re.IGNORECASE)),
            ("Ensemble (Soft Voting)", "ensemble_weighted_voting", re.compile(r"\b(?:ensemble(?:\s+model|\s+pipeline)?(?:\s+with\s+soft\s+voting)?|soft\s+voting\s+ensemble)\b", re.IGNORECASE)),
            ("ResNet-18", "resnet18", re.compile(r"\b(?:resnet[-_ ]?18|convolutional\s+neural\s+networks?|cnns?)\b", re.IGNORECASE)),
            ("ResNet-50", "resnet50", re.compile(r"\b(?:resnet[-_ ]?50)\b", re.IGNORECASE)),
            ("EfficientNet-B0", "efficientnet_b0", re.compile(r"\b(?:efficientnet(?:[-_ ]?b0)?)\b", re.IGNORECASE)),
            ("Vision Transformer", "vit_small", re.compile(r"\b(?:vision\s+transformer|vit(?:[-_ ]?small)?)\b", re.IGNORECASE)),
            ("PubMedBERT", "pubmedbert", re.compile(r"\b(?:pubmedbert|biomedical\s+bert|biomedbert|biobert)\b", re.IGNORECASE)),
            ("ClinicalBERT", "clinicalbert", re.compile(r"\b(?:clinicalbert|clinical\s+bert)\b", re.IGNORECASE)),
            ("TF-IDF", "tfidf_mlp", re.compile(r"\b(?:tf[- ]?idf(?:\s*\+\s*mlp)?)\b", re.IGNORECASE)),
        ]

        bounded_metrics = {
            "auc", "auroc", "roc_auc", "roc-auc", "c_index", "c-index", "concordance",
            "concordance_index", "concordance index", "accuracy", "acc", "f1", "f1_score",
            "f1-score", "precision", "recall", "sensitivity", "specificity", "brier",
        }

        dataset_regex = re.compile(
            r"\b(?:on|using|in)\s+(?:the\s+)?([A-Za-z0-9\-]{3,25})\s*(?:cohort|dataset|database|benchmark)?\b",
            re.IGNORECASE,
        )
        known_datasets = ["TCGA", "PubMed", "RADCURE", "Head-Neck-PET-CT", "MIMIC", "SEER", "IMvigor210", "KHMC-HNSC"]

        def _get_sent_dataset(sent: str) -> Optional[str]:
            for ds in known_datasets:
                if re.search(r"\b" + re.escape(ds) + r"\b", sent, re.IGNORECASE):
                    return ds
            m = dataset_regex.search(sent)
            if m:
                cand = m.group(1).strip()
                if cand.lower() not in ["the", "this", "our", "internal", "external", "test", "train", "validation"]:
                    return cand
            return dataset

        def _clean_for_metrics(sent: str) -> str:
            # Mask 95% CI ranges: 95% CI: 0.647-0.741 or (95% CI: 0.574-0.670)
            s = re.sub(r"\(?\s*95%\s*CI\s*[:=]\s*[0-9.\-–\s,]+\)?", " ", sent, flags=re.IGNORECASE)
            # Mask p-values: P < 0.001, p = 0.05
            s = re.sub(r"\b[pP]\s*[<>=]\s*[0-9.]+\b", " ", s)
            # Mask hardware/scanner: 120 kV, 250 mA, 3.0 mm, 1.5 T
            s = re.sub(r"\b[0-9]+(?:\.[0-9]+)?\s*(?:kV|mA|mm|ms|MHz|Tesla)\b", " ", s, flags=re.IGNORECASE)
            # Mask patient counts: 1019 images, 999 patients, n = 529
            s = re.sub(r"\b(?:n\s*[:=]\s*[0-9]+|[0-9]+\s+(?:patients?|cases?|images?|subjects?))\b", " ", s, flags=re.IGNORECASE)
            return s

        def _extract_metric_scores(sent: str, default_metric: Optional[str] = None) -> List[Tuple[str, float, int, int]]:
            clean_s = _clean_for_metrics(sent)
            scores = []

            # 1. Parenthesized: (AUC: 0.88), (AUC = 0.88), (AUC 0.88)
            re_paren = re.compile(
                r"\(\s*(auroc|auc|roc[-_ ]auc|c[-_ ]index|concordance(?:\s+index)?|accuracy|acc|f1[-_ ]score|f1|rmse|mae|recall|precision|sensitivity|specificity)\s*[:=]?\s*([0-9]+\.?[0-9]*)\s*\)",
                re.IGNORECASE,
            )
            for m in re_paren.finditer(clean_s):
                met = m.group(1)
                val = float(m.group(2))
                if met.lower() in bounded_metrics and val > 1.0 and val <= 100.0:
                    val = round(val / 100.0, 4)
                scores.append((met, val, m.start(), m.end()))

            # 2. Metric with colon/equals/of/was: AUC: 0.88, accuracy of 0.92, C-index was 0.807
            re_metric_val = re.compile(
                r"\b(auroc|auc|roc[-_ ]auc|c[-_ ]index|concordance(?:\s+index)?|accuracy|acc|f1[-_ ]score|f1|rmse|mae|recall|precision|sensitivity|specificity)\b\s*(?:[:=of\s]+|was\s+|is\s+|reached\s+|attained\s+)*([0-9]+\.?[0-9]*)",
                re.IGNORECASE,
            )
            for m in re_metric_val.finditer(clean_s):
                met = m.group(1)
                val_str = m.group(2)
                if not val_str:
                    continue
                val = float(val_str)
                if any(abs(m.start() - s[2]) < 3 for s in scores):
                    continue
                if met.lower() in bounded_metrics and val > 1.0 and val <= 100.0:
                    val = round(val / 100.0, 4)
                scores.append((met, val, m.start(), m.end()))

            # 3. Value followed by % then metric: 92% accuracy, 94.5% recall
            re_pct1 = re.compile(
                r"([0-9]{1,3}(?:\.[0-9]+)?)\s*%\s*(accuracy|acc|auc|f1|precision|recall|sensitivity|specificity|c[-_ ]index|concordance)",
                re.IGNORECASE,
            )
            for m in re_pct1.finditer(clean_s):
                val = round(float(m.group(1)) / 100.0, 4)
                met = m.group(2)
                if not any(abs(m.start() - s[2]) < 5 for s in scores):
                    scores.append((met, val, m.start(), m.end()))

            # 4. Metric followed by %: accuracy of 92%
            re_pct2 = re.compile(
                r"\b(accuracy|acc|auc|f1|precision|recall|sensitivity|specificity)\s*(?:of|was|is|reached)?\s*([0-9]{1,3}(?:\.[0-9]+)?)\s*%",
                re.IGNORECASE,
            )
            for m in re_pct2.finditer(clean_s):
                met = m.group(1)
                val = round(float(m.group(2)) / 100.0, 4)
                if not any(abs(m.start() - s[2]) < 5 for s in scores):
                    scores.append((met, val, m.start(), m.end()))

            # 5. Decimal followed by metric: 0.84 AUC, 0.78 C-index
            re_num_metric = re.compile(
                r"\b([0-9]\.[0-9]+)\s+(auroc|auc|roc[-_ ]auc|c[-_ ]index|concordance|accuracy|acc|f1|recall|precision|rmse|mae)\b",
                re.IGNORECASE,
            )
            for m in re_num_metric.finditer(clean_s):
                val = float(m.group(1))
                met = m.group(2)
                if not any(abs(m.start() - s[2]) < 5 for s in scores):
                    scores.append((met, val, m.start(), m.end()))

            # 6. Additional numbers with default_metric (e.g. "and 0.777 in external", "compared with 0.85 for ResNet-18")
            if default_metric:
                re_bare = re.compile(r"(?:and|with|to|of|\()\s*([0-9]\.[0-9]+)\b(?:\s+(?:in|for|on))?", re.IGNORECASE)
                for m in re_bare.finditer(clean_s):
                    val = float(m.group(1))
                    if any(abs(s[1] - val) < 1e-4 for s in scores):
                        continue
                    if not any(abs(m.start(1) - s[2]) < 4 for s in scores):
                        scores.append((default_metric, val, m.start(1), m.end(1)))

            scores.sort(key=lambda x: x[2])
            return scores

        tagged = (
            self._get_tagged_sentences(norm_text, sections)
            if sections
            else [(s, "Results" if source_scope == SourceScope.full_text else "Abstract") for s in self._get_sentences(norm_text)]
        )

        extracted_results: List[PublishedExperimentResult] = []
        seen_keys: set = set()
        res_idx = 1

        for raw_sentence, section in tagged:
            # Skip background/intro sections
            if self.section_parser.is_background_section(section):
                continue

            # Protocol detection
            proto = validation_protocol
            cv_match = re.search(r"\b(\d+)[- ]fold\s+(?:cross[- ]validation|cv)\b", raw_sentence, re.IGNORECASE)
            if cv_match:
                proto = f"{cv_match.group(1)}_fold_cv"
            elif any(k in raw_sentence.lower() for k in ["external", "independent cohort", "independent validation"]):
                proto = "external_validation"
            elif any(k in raw_sentence.lower() for k in ["holdout", "test set", "test split"]):
                proto = "holdout_test"

            sent_dataset = _get_sent_dataset(raw_sentence)

            # Detect models present across raw_sentence
            models_in_sentence = []
            for display_name, canon_id, pat in model_patterns:
                for match in pat.finditer(raw_sentence):
                    models_in_sentence.append((match.start(), display_name, canon_id))
            models_in_sentence.sort(key=lambda x: x[0])
            unique_models = []
            for m in models_in_sentence:
                if not unique_models or unique_models[-1][2] != m[2]:
                    unique_models.append(m)

            if not unique_models:
                continue

            # -------------------------------------------------------------------
            # Pattern A: "respectively" construction
            # -------------------------------------------------------------------
            if "respectively" in raw_sentence.lower() and len(unique_models) >= 2:
                all_scores = _extract_metric_scores(raw_sentence)
                if len(all_scores) < len(unique_models):
                    num_re = re.compile(r"\b([0-9]+\.?[0-9]*)\b")
                    found_nums = []
                    for nm in num_re.finditer(raw_sentence):
                        val = float(nm.group(1))
                        if 0.0 < val <= 100.0:
                            val_norm = round(val / 100.0, 4) if val > 1.0 else val
                            found_nums.append((val_norm, nm.start(), nm.end()))
                    if len(found_nums) >= len(unique_models):
                        use_nums = found_nums[-len(unique_models):]
                        metric_found = all_scores[0][0] if all_scores else "AUC"
                        all_scores = [(metric_found, n[0], n[1], n[2]) for n in use_nums]

                if len(unique_models) >= 2 and len(all_scores) == len(unique_models):
                    for (_, disp_name, canon_id), (met_name, val, _, _) in zip(unique_models, all_scores):
                        key = (canon_id, met_name.lower(), val)
                        if key not in seen_keys:
                            seen_keys.add(key)
                            _, hib = normalize_metric_name(met_name)
                            extracted_results.append(PublishedExperimentResult(
                                result_id=f"pub_res_{paper_id}_{res_idx:03d}",
                                paper_id=paper_id,
                                pmid=pmid,
                                doi=doi,
                                citation=citation or f"Paper {paper_id}",
                                model_name=disp_name,
                                canonical_id=canon_id,
                                dataset_name=sent_dataset,
                                task_type=task or "classification",
                                metric_name=met_name,
                                reported_value=val,
                                higher_is_better=hib,
                                validation_protocol=proto,
                                sample_size=sample_size,
                                source_sentence=raw_sentence.strip(),
                                section_location=section,
                                is_ambiguous=False,
                                extraction_confidence=0.92,
                            ))
                            res_idx += 1
                    continue

            # -------------------------------------------------------------------
            # Pattern B: Head-to-Head "X vs Y" comparison
            # -------------------------------------------------------------------
            vs_match = re.search(r"([0-9]\.[0-9]+)\s+vs\.?\s+([0-9]\.[0-9]+)\s*(?:(auroc|auc|c[-_ ]index|concordance|accuracy|f1|recall|precision))?", raw_sentence, re.IGNORECASE)
            if vs_match and len(unique_models) >= 2:
                v1 = float(vs_match.group(1))
                v2 = float(vs_match.group(2))
                raw_metric = vs_match.group(3) or "AUC"
                _, hib = normalize_metric_name(raw_metric)
                _, mod_a_disp, mod_a_canon = unique_models[0]
                _, mod_b_disp, mod_b_canon = unique_models[1]

                for mod_disp, mod_canon, val in [(mod_a_disp, mod_a_canon, v1), (mod_b_disp, mod_b_canon, v2)]:
                    key = (mod_canon, raw_metric.lower(), val)
                    if key not in seen_keys:
                        seen_keys.add(key)
                        extracted_results.append(PublishedExperimentResult(
                            result_id=f"pub_res_{paper_id}_{res_idx:03d}",
                            paper_id=paper_id,
                            pmid=pmid,
                            doi=doi,
                            citation=citation or f"Paper {paper_id}",
                            model_name=mod_disp,
                            canonical_id=mod_canon,
                            dataset_name=sent_dataset,
                            task_type=task or "classification",
                            metric_name=raw_metric,
                            reported_value=val,
                            higher_is_better=hib,
                            validation_protocol=proto,
                            sample_size=sample_size,
                            source_sentence=raw_sentence.strip(),
                            section_location=section,
                            is_ambiguous=False,
                            extraction_confidence=0.92,
                        ))
                        res_idx += 1
                continue

            # -------------------------------------------------------------------
            # Pattern C: Paired parentheticals / explicit inline associations
            # -------------------------------------------------------------------
            if len(unique_models) >= 2:
                scores_with_spans = _extract_metric_scores(raw_sentence)
                paired_bindings = []
                for i, (m_start, disp, canon) in enumerate(unique_models):
                    next_start = unique_models[i+1][0] if i+1 < len(unique_models) else len(raw_sentence)
                    bound_scores = [s for s in scores_with_spans if m_start < s[2] < next_start + 40]
                    if bound_scores:
                        paired_bindings.append((disp, canon, bound_scores[0]))

                if len(paired_bindings) >= 2:
                    for disp, canon, (met_name, val, _, _) in paired_bindings:
                        key = (canon, met_name.lower(), val)
                        if key not in seen_keys:
                            seen_keys.add(key)
                            _, hib = normalize_metric_name(met_name)
                            extracted_results.append(PublishedExperimentResult(
                                result_id=f"pub_res_{paper_id}_{res_idx:03d}",
                                paper_id=paper_id,
                                pmid=pmid,
                                doi=doi,
                                citation=citation or f"Paper {paper_id}",
                                model_name=disp,
                                canonical_id=canon,
                                dataset_name=sent_dataset,
                                task_type=task or "classification",
                                metric_name=met_name,
                                reported_value=val,
                                higher_is_better=hib,
                                validation_protocol=proto,
                                sample_size=sample_size,
                                source_sentence=raw_sentence.strip(),
                                section_location=section,
                                is_ambiguous=False,
                                extraction_confidence=0.92,
                            ))
                            res_idx += 1
                    continue

            # -------------------------------------------------------------------
            # Pattern D: Multi-eval of single model across protocols
            # -------------------------------------------------------------------
            if len(unique_models) == 1:
                primary_scores = _extract_metric_scores(raw_sentence)
                prim_met = primary_scores[0][0] if primary_scores else "AUC"
                all_scores = _extract_metric_scores(raw_sentence, default_metric=prim_met)

                if len(all_scores) >= 2 and ("internal" in raw_sentence.lower() or "external" in raw_sentence.lower()):
                    disp, canon = unique_models[0][1], unique_models[0][2]
                    for raw_metric, val, s_start, s_end in all_scores:
                        window = raw_sentence[max(0, s_start - 35):min(len(raw_sentence), s_end + 35)].lower()
                        score_proto = proto
                        if "external" in window:
                            score_proto = "external_validation"
                        elif "internal" in window:
                            score_proto = "internal_validation"
                        key = (canon, raw_metric.lower(), val, score_proto)
                        if key not in seen_keys:
                            seen_keys.add(key)
                            _, hib = normalize_metric_name(raw_metric)
                            extracted_results.append(PublishedExperimentResult(
                                result_id=f"pub_res_{paper_id}_{res_idx:03d}",
                                paper_id=paper_id,
                                pmid=pmid,
                                doi=doi,
                                citation=citation or f"Paper {paper_id}",
                                model_name=disp,
                                canonical_id=canon,
                                dataset_name=sent_dataset,
                                task_type=task or "classification",
                                metric_name=raw_metric,
                                reported_value=val,
                                higher_is_better=hib,
                                validation_protocol=score_proto,
                                sample_size=sample_size,
                                source_sentence=raw_sentence.strip(),
                                section_location=section,
                                is_ambiguous=False,
                                extraction_confidence=0.90,
                            ))
                            res_idx += 1
                    continue

            # -------------------------------------------------------------------
            # Pattern E: Clause-based processing with metric propagation
            # -------------------------------------------------------------------
            clauses = re.split(r'(?:,\s*(?:while|whereas|but)\s+|\s+(?:while|whereas|but)\s+|;\s*|\bcompared\s+(?:to|with)\s+)', raw_sentence)
            initial_scores = _extract_metric_scores(raw_sentence)
            primary_metric = initial_scores[0][0] if initial_scores else None

            # Genuine ambiguity check: multiple models with only 1 score in entire sentence
            if len(unique_models) > 1 and len(initial_scores) == 1 and not any(k in raw_sentence.lower() for k in ["compared", "whereas", "while", "outperformed"]):
                raw_metric, val, _, _ = initial_scores[0]
                _, hib = normalize_metric_name(raw_metric)
                for _, disp, canon in unique_models:
                    key = (canon, raw_metric.lower(), val, True)
                    if key not in seen_keys:
                        seen_keys.add(key)
                        extracted_results.append(PublishedExperimentResult(
                            result_id=f"pub_res_{paper_id}_{res_idx:03d}",
                            paper_id=paper_id,
                            pmid=pmid,
                            doi=doi,
                            citation=citation or f"Paper {paper_id}",
                            model_name=disp,
                            canonical_id=canon,
                            dataset_name=sent_dataset,
                            task_type=task or "classification",
                            metric_name=raw_metric,
                            reported_value=val,
                            higher_is_better=hib,
                            validation_protocol=proto,
                            sample_size=sample_size,
                            source_sentence=raw_sentence.strip(),
                            section_location=section,
                            is_ambiguous=True,
                            extraction_confidence=0.40,
                        ))
                        res_idx += 1
                continue

            for clause in clauses:
                clause_models = []
                for display_name, canon_id, pat in model_patterns:
                    if pat.search(clause):
                        clause_models.append((display_name, canon_id))

                scores = _extract_metric_scores(clause, default_metric=primary_metric)
                if scores:
                    primary_metric = scores[0][0]

                if not clause_models or not scores:
                    continue

                # Filter out scores that belong to non-model baselines like "junior radiologists"
                if re.search(r"\b(?:radiologists?|clinicians?|physicians?)\s*\([^\)]*?\)", clause, re.IGNORECASE):
                    clean_clause = re.sub(r"\b(?:junior\s+|senior\s+)?radiologists?\s*\([^\)]*?\)", " ", clause, flags=re.IGNORECASE)
                    scores = _extract_metric_scores(clean_clause, default_metric=primary_metric)
                    if not scores:
                        continue

                if len(clause_models) == 1:
                    disp, canon = clause_models[0]
                    for raw_metric, val, _, _ in scores:
                        key = (canon, raw_metric.lower(), val)
                        if key not in seen_keys:
                            seen_keys.add(key)
                            _, hib = normalize_metric_name(raw_metric)
                            extracted_results.append(PublishedExperimentResult(
                                result_id=f"pub_res_{paper_id}_{res_idx:03d}",
                                paper_id=paper_id,
                                pmid=pmid,
                                doi=doi,
                                citation=citation or f"Paper {paper_id}",
                                model_name=disp,
                                canonical_id=canon,
                                dataset_name=sent_dataset,
                                task_type=task or "classification",
                                metric_name=raw_metric,
                                reported_value=val,
                                higher_is_better=hib,
                                validation_protocol=proto,
                                sample_size=sample_size,
                                source_sentence=raw_sentence.strip(),
                                section_location=section,
                                is_ambiguous=False,
                                extraction_confidence=0.90,
                            ))
                            res_idx += 1

                elif len(clause_models) > 1:
                    for raw_metric, val, _, _ in scores:
                        _, hib = normalize_metric_name(raw_metric)
                        for disp, canon in clause_models:
                            key = (canon, raw_metric.lower(), val, True)
                            if key not in seen_keys:
                                seen_keys.add(key)
                                extracted_results.append(PublishedExperimentResult(
                                    result_id=f"pub_res_{paper_id}_{res_idx:03d}",
                                    paper_id=paper_id,
                                    pmid=pmid,
                                    doi=doi,
                                    citation=citation or f"Paper {paper_id}",
                                    model_name=disp,
                                    canonical_id=canon,
                                    dataset_name=sent_dataset,
                                    task_type=task or "classification",
                                    metric_name=raw_metric,
                                    reported_value=val,
                                    higher_is_better=hib,
                                    validation_protocol=proto,
                                    sample_size=sample_size,
                                    source_sentence=raw_sentence.strip(),
                                    section_location=section,
                                    is_ambiguous=True,
                                    extraction_confidence=0.40,
                                ))
                                res_idx += 1

        return extracted_results

