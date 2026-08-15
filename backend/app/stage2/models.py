from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

# ──────────────────────────────────────────────────────────────────────────────
# Enumerations
# ──────────────────────────────────────────────────────────────────────────────

class ExtractionMethod(str, Enum):
    manual = "manual"
    regex_based = "regex_based"
    schema_parser = "schema_parser"
    llm_assisted = "llm_assisted"

class ExtractionStatus(str, Enum):
    explicit = "explicit"      # Value directly stated in text
    structured = "structured"  # Inferable from structured section
    unresolved = "unresolved"  # Cannot be confirmed from available text

class EvidenceStatus(str, Enum):
    direct_empirical = "direct_empirical"
    secondary_empirical = "secondary_empirical"
    qualitative = "qualitative"
    methodological = "methodological"
    theoretical = "theoretical"
    unverified = "unverified"

class SourceScope(str, Enum):
    full_text = "full_text"
    abstract = "abstract"
    metadata_only = "metadata_only"
    none = "none"

class FullTextAccessStatus(str, Enum):
    accessible = "accessible"
    not_accessible = "not_accessible"
    not_found = "not_found"
    abstract_only = "abstract_only"

class FusionStrategy(str, Enum):
    early_fusion = "early_fusion"
    intermediate_fusion = "intermediate_fusion"
    late_fusion = "late_fusion"
    cross_attention = "cross_attention"
    gated_fusion = "gated_fusion"
    ensemble_fusion = "ensemble_fusion"
    joint_embedding = "joint_embedding"
    unknown = "unknown"

class MechanismCategory(str, Enum):
    representation = "Representation"
    preprocessing = "Preprocessing"
    feature_selection = "Feature Selection"
    loss = "Loss"
    sampling = "Sampling"
    regularization = "Regularization"
    fusion = "Fusion"
    attention = "Attention"
    classifier = "Classifier"
    calibration = "Calibration"
    ensembling = "Ensembling"
    unmapped = "UNMAPPED"

# ──────────────────────────────────────────────────────────────────────────────
# Provenance
# ──────────────────────────────────────────────────────────────────────────────

class Provenance(BaseModel):
    source_type: str = Field(description="E.g., scholarly_api, dataset, full_text")
    source_reference: str = Field(description="DOI, PMID, or file path")
    extraction_method: ExtractionMethod
    extraction_status: ExtractionStatus = Field(default=ExtractionStatus.explicit)
    evidence_text: Optional[str] = None
    retrieval_date: str
    section: Optional[str] = None         # e.g. "Results", "Table 2"
    page_or_table: Optional[str] = None   # e.g. "Table 3", "p.7"

class SearchMetadata(BaseModel):
    search_query: str
    source: str
    retrieval_date: str
    filters: Optional[str] = None
    year_range: Optional[str] = None
    domain: Optional[str] = None
    task: Optional[str] = None
    modality: Optional[str] = None
    candidates_returned: int = 0
    papers_selected: int = 0

# ──────────────────────────────────────────────────────────────────────────────
# Paper Representation
# ──────────────────────────────────────────────────────────────────────────────

class PaperRecord(BaseModel):
    paper_id: str
    title: str
    authors: List[str]
    publication_year: int
    journal: Optional[str] = None
    doi: Optional[str] = None
    pmid: Optional[str] = None
    pmc_id: Optional[str] = None
    openalex_id: Optional[str] = None
    source: str
    abstract: Optional[str] = None
    abstract_available: bool = False
    full_text_available: bool = False
    full_text_source: Optional[str] = None       # e.g. "PMC", "MDPI", "BioMedCentral"
    full_text_url: Optional[str] = None
    full_text_retrieved_at: Optional[str] = None
    full_text_sha256: Optional[str] = None
    full_text_license: Optional[str] = None      # e.g. "CC BY 4.0"
    full_text_access_status: FullTextAccessStatus = FullTextAccessStatus.not_found
    retrieval_date: str
    sha256: Optional[str] = None
    license_or_access_status: Optional[str] = None

# ──────────────────────────────────────────────────────────────────────────────
# Mechanisms
# ──────────────────────────────────────────────────────────────────────────────

class Mechanism(BaseModel):
    mechanism_id: str
    canonical_name: str
    category: MechanismCategory
    description: Optional[str] = None
    role: Optional[str] = None                   # actual role in the paper's architecture
    input_modality: Optional[str] = None
    output_representation: Optional[str] = None
    conditions: Optional[str] = None
    evidence_claim_ids: List[str] = Field(default_factory=list)
    transferability_notes: Optional[str] = None
    mapping_status: str = Field(default="MAPPED")

# ──────────────────────────────────────────────────────────────────────────────
# Experimental Conditions and Characteristics
# ──────────────────────────────────────────────────────────────────────────────

class DatasetCharacteristics(BaseModel):
    sample_count: Optional[int] = None
    feature_count: Optional[int] = None
    class_count: Optional[int] = None
    class_imbalance: Optional[bool] = None
    missingness: Optional[float] = None
    modality_count: Optional[int] = None
    modality_types: List[str] = Field(default_factory=list)
    text_available: Optional[bool] = None
    image_available: Optional[bool] = None
    tabular_available: Optional[bool] = None
    temporal_available: Optional[bool] = None

class ExperimentalConditions(BaseModel):
    dataset_name: Optional[str] = None
    train_test_strategy: Optional[str] = None
    cross_validation: Optional[str] = None
    random_seed_if_reported: Optional[int] = None
    baseline: Optional[str] = None
    hyperparameter_tuning: Optional[str] = None
    compute_conditions_if_reported: Optional[str] = None
    preprocessing: Optional[str] = None
    augmentation: Optional[str] = None
    evaluation_metric: Optional[str] = None

class EmpiricalResult(BaseModel):
    metric: Optional[str] = None
    baseline_value: Optional[float] = None
    method_value: Optional[float] = None
    delta: Optional[float] = None
    direction: str = Field(description="improvement, degradation, unchanged, qualitative")

# ──────────────────────────────────────────────────────────────────────────────
# Stage 2B: Structured Experiment and Ablation
# ──────────────────────────────────────────────────────────────────────────────

class FieldProvenance(BaseModel):
    field_name: str
    value: str
    source_sentence: str
    section: str
    source_location: str
    extraction_method: str = "regex_based"
    # confidence_status replaces verification_status for forward-compat;
    # maps to the ExtractionStatus enum: explicit | structured | unresolved
    confidence_status: ExtractionStatus = ExtractionStatus.explicit
    # keep verification_status as alias so old tests don't break
    verification_status: str = "VERIFIED"


class BaselineRecord(BaseModel):
    """A single comparator described in the paper."""
    name: str
    source_sentence: str
    source_location: str
    comparison_context: Optional[str] = None

class ResultRecord(BaseModel):
    """Fine-grained result, used within ExperimentRecord."""
    metric: Optional[str] = None
    baseline_value: Optional[float] = None
    method_value: Optional[float] = None
    delta: Optional[float] = None          # null unless BOTH baseline and method are known
    direction: str                         # improvement / degradation / unchanged / qualitative
    source_location: Optional[str] = None  # e.g. "Results/Table 2"
    source_scope: SourceScope = SourceScope.abstract

class ExperimentRecord(BaseModel):
    experiment_id: str
    paper_id: str
    dataset: Optional[str] = None
    sample_count: Optional[int] = None
    task: Optional[str] = None
    modalities: List[str] = Field(default_factory=list)
    train_strategy: Optional[str] = None
    validation_strategy: Optional[str] = None
    test_strategy: Optional[str] = None
    # baseline: kept as Optional[str] for backwards compatibility.
    # Populated automatically from baselines[0].name if baselines is non-empty.
    baseline: Optional[str] = None
    # baselines: structured list of all comparators identified in the paper.
    baselines: List[BaselineRecord] = Field(default_factory=list)
    proposed_method: Optional[str] = None
    preprocessing: Optional[str] = None
    augmentation: Optional[str] = None
    loss_function: Optional[str] = None
    regularization: Optional[str] = None
    fusion_strategy: Optional[FusionStrategy] = None
    feature_representation: Optional[str] = None
    hyperparameter_tuning: Optional[str] = None
    evaluation_metrics: List[str] = Field(default_factory=list)
    reported_results: List[ResultRecord] = Field(default_factory=list)
    statistical_test_if_reported: Optional[str] = None
    limitations: Optional[str] = None
    source_scope: SourceScope = SourceScope.abstract
    source_section: Optional[str] = None  # where this experiment was described
    field_provenance: Dict[str, FieldProvenance] = Field(default_factory=dict)

class AblationRecord(BaseModel):
    ablation_id: str
    parent_experiment_id: str
    paper_id: str
    condition_removed: str     # what was removed/changed in this ablation
    result: Optional[ResultRecord] = None
    source_location: Optional[str] = None
    source_scope: SourceScope = SourceScope.abstract

class ContradictionCandidate(BaseModel):
    candidate_id: str
    evidence_claim_a: str
    evidence_claim_b: str
    reason: str
    comparison_dimensions: List[str] = Field(default_factory=list)
    # Dimensions compared: task, mechanism, metric, modality, domain
    shared_task: Optional[str] = None
    shared_metric: Optional[str] = None
    shared_mechanisms: List[str] = Field(default_factory=list)
    direction_a: Optional[str] = None
    direction_b: Optional[str] = None

# ──────────────────────────────────────────────────────────────────────────────
# Evidence Claims
# ──────────────────────────────────────────────────────────────────────────────

class EvidenceClaim(BaseModel):
    evidence_id: str
    paper_id: str
    claim: str
    source_scope: SourceScope = Field(default=SourceScope.abstract)
    mechanisms: List[str] = Field(description="List of mechanism_ids")
    task: Optional[str] = None
    domain: Optional[str] = None
    modalities: List[str] = Field(default_factory=list)
    dataset_characteristics: Optional[DatasetCharacteristics] = None
    baseline: Optional[str] = None
    metric: Optional[str] = None
    result: Optional[EmpiricalResult] = None
    experimental_conditions: Optional[ExperimentalConditions] = None
    limitations: Optional[str] = None
    evidence_location: str = Field(description="E.g., Abstract, Results/Table 2")
    source_text_reference: Optional[str] = None
    extraction_method: ExtractionMethod
    evidence_status: EvidenceStatus = Field(default=EvidenceStatus.unverified)
    contradiction_candidate: bool = False
    provenance: Provenance
    # Stage 2B enrichment
    experiment_id: Optional[str] = None   # link to ExperimentRecord if extracted

# ──────────────────────────────────────────────────────────────────────────────
# Graph Representation
# ──────────────────────────────────────────────────────────────────────────────

class GraphNode(BaseModel):
    node_id: str
    node_type: str
    properties: Dict[str, Any] = Field(default_factory=dict)

class GraphRelationship(BaseModel):
    source_id: str
    target_id: str
    relationship_type: str
    properties: Dict[str, Any] = Field(default_factory=dict)

class EvidenceGraph(BaseModel):
    nodes: List[GraphNode] = Field(default_factory=list)
    relationships: List[GraphRelationship] = Field(default_factory=list)
