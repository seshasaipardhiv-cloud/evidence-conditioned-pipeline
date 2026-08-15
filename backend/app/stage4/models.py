from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum

class ExecutionStatus(str, Enum):
    BLOCKED = "BLOCKED"
    CONFIGURATION_VALIDATED = "CONFIGURATION_VALIDATED"
    READY_FOR_TRAINING = "READY_FOR_TRAINING"

class ExperimentConfig(BaseModel):
    target_variable: Optional[str] = None
    task_type: Optional[str] = None
    primary_metric: Optional[str] = None
    secondary_metrics: List[str] = Field(default_factory=list)
    test_size: Optional[float] = None
    validation_size: Optional[float] = None
    random_seeds: List[int] = Field(default_factory=list)
    patient_level_split: bool = True
    stratification_policy: Optional[str] = None
    missing_target_policy: str = "exclude_from_supervised_analysis"

class ExecutionGate(BaseModel):
    execution_status: ExecutionStatus = ExecutionStatus.BLOCKED
    training_allowed: bool = False
    blocking_reasons: List[str] = Field(default_factory=list)
    validated_at: Optional[str] = None
    stage3_compatibility_valid: bool = False
    target_valid: bool = False
    task_valid: bool = False
    split_valid: bool = False
    leakage_check_passed: bool = False
    mechanism_gate_passed: bool = False
    compute_budget_valid: bool = False

class DataSplitManifest(BaseModel):
    total_patients: int = 0
    eligible_patients: int = 0
    excluded_missing_target: int = 0
    train_count: int = 0
    validation_count: int = 0
    test_count: int = 0
    train_validation_overlap: int = 0
    train_test_overlap: int = 0
    validation_test_overlap: int = 0
    split_seed: Optional[int] = None
    split_method: Optional[str] = None
    split_hash: Optional[str] = None
    task_type: Optional[str] = None
    stratification_policy: Optional[str] = None

class TargetLeakageReport(BaseModel):
    rejected_fields: List[Dict[str, str]] = Field(default_factory=list)

class MechanismGate(BaseModel):
    decisions: List[Dict[str, Any]] = Field(default_factory=list)

class ComputeBudget(BaseModel):
    max_epochs: int = 10
    max_training_time_minutes: int = 15
    max_memory_gb: int = 4
    device: str = "cpu"
    max_parallel_jobs: int = 1
