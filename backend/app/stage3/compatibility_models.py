from enum import Enum
from typing import List, Optional, Any, Dict
from pydantic import BaseModel

class CompatibilityStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    INCOMPATIBLE = "INCOMPATIBLE"
    BLOCKED_BY_TASK = "BLOCKED_BY_TASK"
    INVALID_BASELINE_ENTITY = "INVALID_BASELINE_ENTITY"

class MechanismDecision(BaseModel):
    mechanism: str
    component: str
    decision: CompatibilityStatus
    reason: Optional[str] = None
    posterior_mean: Optional[float] = None
    evidence_count: int = 0

class BaselineDecision(BaseModel):
    baseline: str
    decision: CompatibilityStatus
    reason: Optional[str] = None

class TargetGate(BaseModel):
    task: str
    target: Optional[str] = None
    blocked: bool
    reason: Optional[str] = None

class CompatibilityAuditReport(BaseModel):
    stage: str = "3.1"
    status: str
    target_gate: TargetGate
    mechanism_decisions: List[MechanismDecision]
    baseline_decisions: List[BaselineDecision]
    incompatibilities: List[Dict[str, Any]]
    insufficient_evidence: List[Dict[str, Any]]
    invalid_baselines: List[Dict[str, Any]]
    supporting_evidence: List[Dict[str, Any]]
    provenance_checks: Dict[str, Any]
