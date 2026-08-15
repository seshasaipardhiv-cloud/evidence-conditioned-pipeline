from enum import Enum
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field

class Component(str, Enum):
    missing_value_handling = "missing_value_handling"
    categorical_encoding = "categorical_encoding"
    feature_representation = "feature_representation"
    modality_fusion = "modality_fusion"
    base_learner = "base_learner"
    loss_function = "loss_function"
    imbalance_handling = "imbalance_handling"
    ensembling = "ensembling"

class Mechanism(str, Enum):
    # 1. missing_value_handling
    mean_imputation = "mean_imputation"
    # 2. categorical_encoding
    one_hot_encoding = "one_hot_encoding"
    # 3. feature_representation
    cnn_representation = "cnn_representation"
    transformer_representation = "transformer_representation"
    # 4. modality_fusion
    late_fusion = "late_fusion"
    early_fusion = "early_fusion"
    cross_attention = "cross_attention"
    joint_embedding = "joint_embedding"
    # 5. base_learner
    gradient_boosting = "gradient_boosting"
    # 6. loss_function
    focal_loss = "focal_loss"
    # 7. imbalance_handling
    class_weighted_sampling = "class_weighted_sampling"
    # 8. ensembling
    average_ensembling = "average_ensembling"

# Mapping from Mechanism to Component
MECHANISM_TO_COMPONENT = {
    Mechanism.mean_imputation: Component.missing_value_handling,
    Mechanism.one_hot_encoding: Component.categorical_encoding,
    Mechanism.cnn_representation: Component.feature_representation,
    Mechanism.transformer_representation: Component.feature_representation,
    Mechanism.late_fusion: Component.modality_fusion,
    Mechanism.early_fusion: Component.modality_fusion,
    Mechanism.cross_attention: Component.modality_fusion,
    Mechanism.joint_embedding: Component.modality_fusion,
    Mechanism.gradient_boosting: Component.base_learner,
    Mechanism.focal_loss: Component.loss_function,
    Mechanism.class_weighted_sampling: Component.imbalance_handling,
    Mechanism.average_ensembling: Component.ensembling,
}

class Stage3Context(BaseModel):
    task: str = "unknown"
    modalities: List[str] = Field(default_factory=list)
    sample_size: Optional[int] = None
    missingness_rate: float = 0.0
    class_imbalance: float = 0.0 # 0.0 means perfectly balanced, 1.0 means fully imbalanced (or undefined if not applicable)
    text_available: bool = False
    imaging_available: bool = False
    clinical_available: bool = False
    blood_available: bool = False
    constraints: Dict[str, Any] = Field(default_factory=dict)

class EvidenceMatch(BaseModel):
    paper_id: str
    claim_id: Optional[str] = None
    experiment_id: Optional[str] = None
    mechanism_id: Optional[str] = None
    source_location: Optional[str] = None
    source_scope: Optional[str] = None
    result: Optional[str] = None
    metric: Optional[str] = None
    baseline: Optional[str] = None
    context_similarity: float
    evidence_quality: float
    direction: str # "positive", "negative", "neutral"

class ContextualBelief(BaseModel):
    mechanism: Mechanism
    component: Component
    prior_alpha: float = 1.0
    prior_beta: float = 1.0
    alpha: float = 1.0
    beta: float = 1.0
    supporting_evidence_count: int = 0
    contradicting_evidence_count: int = 0
    neutral_evidence_count: int = 0
    supporting_matches: List[EvidenceMatch] = Field(default_factory=list)
    contradicting_matches: List[EvidenceMatch] = Field(default_factory=list)
    neutral_matches: List[EvidenceMatch] = Field(default_factory=list)
    
    @property
    def posterior_mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

class MechanismScore(BaseModel):
    mechanism: Mechanism
    component: Component
    posterior_mean: float
    evidence_count: int
    support_count: int
    contradiction_count: int
    context_similarity_sum: float
    evidence_quality_sum: float
    final_score: float

class MechanismRanking(BaseModel):
    component: Component
    winner: Optional[Mechanism] = None
    selection_status: str # "selected", "insufficient_evidence", "tie"
    alternatives: List[MechanismScore] = Field(default_factory=list)
    insufficient_evidence: List[Mechanism] = Field(default_factory=list)

class PipelineSpecification(BaseModel):
    problem_context: Stage3Context
    fixed_components: List[str]
    selected_mechanisms: Dict[str, Optional[str]]
    alternative_mechanisms: Dict[str, List[Dict[str, Any]]]
    mechanism_scores: Dict[str, Dict[str, Any]]
    contextual_beliefs: Dict[str, Dict[str, Any]]
    supporting_evidence: Dict[str, List[EvidenceMatch]]
    contradicting_evidence: Dict[str, List[EvidenceMatch]]
    uncertainty: Dict[str, float]
    selection_rationale: Dict[str, str]
    expected_baselines: List[str]
    compute_budget_placeholder: str = "TBD - Awaiting Execution Stage"
