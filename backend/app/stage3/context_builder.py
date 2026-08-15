import json
from pathlib import Path
from backend.app.stage3.models import Stage3Context

class ContextBuilder:
    def __init__(self, stage1_path: str = "data/processed/hancock/stage1_problem_representation.json"):
        self.stage1_path = Path(stage1_path)
        
    def build(self) -> Stage3Context:
        if not self.stage1_path.exists():
            # If not found, return empty fallback context
            return Stage3Context()
            
        with open(self.stage1_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # Parse task
        task = data.get("task", {}).get("value", "unknown")
        if task == "unknown":
            task_type = data.get("problem", {}).get("task_type", {}).get("value", "unknown")
            if task_type != "unknown":
                task = task_type
                
        # Parse modalities
        mods_dict = data.get("modalities", {})
        modalities = [k for k, v in mods_dict.items() if v]
        
        # Parse dataset features
        features = data.get("dataset_features", {})
        sample_size = features.get("number_of_patients")
        
        # Missingness
        missingness_dict = features.get("missingness_rates", {})
        if missingness_dict:
            missingness_rate = sum(missingness_dict.values()) / len(missingness_dict)
        else:
            missingness_rate = 0.0
            
        # Class imbalance - attempt to derive if targets are known, otherwise 0.0
        # Given stage 1 target_status is AMBIGUOUS, we leave this at 0.0
        class_imbalance = 0.0
        
        # Constraints
        constraints = data.get("constraints", {})
        
        return Stage3Context(
            task=task,
            modalities=modalities,
            sample_size=sample_size,
            missingness_rate=missingness_rate,
            class_imbalance=class_imbalance,
            text_available="text" in modalities,
            imaging_available="imaging" in modalities,
            clinical_available="clinical" in modalities,
            blood_available="blood" in modalities,
            constraints=constraints
        )
