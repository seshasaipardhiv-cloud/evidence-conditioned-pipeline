import json
import hashlib
from typing import List, Dict, Any, Tuple
from pathlib import Path

import random
from collections import defaultdict

def hash_patient_ids(patient_ids: List[str]) -> str:
    sorted_ids = sorted([str(pid) for pid in patient_ids])
    h = hashlib.sha256()
    for pid in sorted_ids:
        h.update(pid.encode('utf-8'))
    return h.hexdigest()

def pure_train_test_split(patient_ids, targets, test_size, seed):
    # Group by class
    classes = defaultdict(list)
    for p_id, t in zip(patient_ids, targets):
        classes[t].append(p_id)
        
    train_ids = []
    test_ids = []
    train_y = []
    test_y = []
    
    for cls, p_list in classes.items():
        # Sort for determinism before shuffling
        p_list.sort()
        
        # Deterministic shuffle
        rng = random.Random(seed)
        rng.shuffle(p_list)
        
        n_test = int(round(len(p_list) * test_size))
        
        # Test gets the first n_test elements
        cls_test = p_list[:n_test]
        cls_train = p_list[n_test:]
        
        test_ids.extend(cls_test)
        test_y.extend([cls] * len(cls_test))
        
        train_ids.extend(cls_train)
        train_y.extend([cls] * len(cls_train))
        
    # Final shuffle to mix classes
    rng = random.Random(seed + 1)
    train_combined = list(zip(train_ids, train_y))
    rng.shuffle(train_combined)
    if train_combined:
        train_ids, train_y = zip(*train_combined)
    else:
        train_ids, train_y = [], []
        
    rng = random.Random(seed + 2)
    test_combined = list(zip(test_ids, test_y))
    rng.shuffle(test_combined)
    if test_combined:
        test_ids, test_y = zip(*test_combined)
    else:
        test_ids, test_y = [], []
        
    return list(train_ids), list(test_ids), list(train_y), list(test_y)

class PatientSplitter:
    def __init__(self, config_path: str, data_path: str):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
        with open(data_path, "r", encoding="utf-8") as f:
            self.clinical_data = json.load(f)
            
    def _extract_valid_cohort(self) -> Tuple[List[str], List[Any]]:
        target_var = self.config["target_variable"]
        patient_ids = []
        targets = []
        for row in self.clinical_data:
            val = row.get(target_var)
            if val is not None:
                patient_ids.append(row["patient_id"])
                targets.append(val)
        return patient_ids, targets

    def run_splits(self) -> Dict[str, Any]:
        patient_ids, targets = self._extract_valid_cohort()
        
        test_size = self.config["test_size"]
        val_size = self.config["validation_size"]
        
        # val_size is relative to total. Since train_test_split is sequential, 
        # first split test, then split val from the remaining.
        # test = test_size, remaining = 1 - test_size
        # val_ratio_of_remaining = val_size / (1 - test_size)
        val_ratio_of_remaining = val_size / (1.0 - test_size)
        
        manifest = {
            "target_variable": self.config["target_variable"],
            "missing_target_policy": self.config["missing_target_policy"],
            "splits": []
        }
        
        for seed in self.config["random_seeds"]:
            # First split: Test vs (Train + Val)
            train_val_ids, test_ids, train_val_y, test_y = pure_train_test_split(
                patient_ids, targets, test_size=test_size, seed=seed
            )
            
            # Second split: Val vs Train
            train_ids, val_ids, train_y, val_y = pure_train_test_split(
                train_val_ids, train_val_y, test_size=val_ratio_of_remaining, seed=seed
            )
            
            t_set, v_set, test_set = set(train_ids), set(val_ids), set(test_ids)
            overlap_1 = len(t_set.intersection(v_set))
            overlap_2 = len(t_set.intersection(test_set))
            overlap_3 = len(v_set.intersection(test_set))
            
            manifest["splits"].append({
                "seed": seed,
                "train_patient_count": len(train_ids),
                "validation_patient_count": len(val_ids),
                "test_patient_count": len(test_ids),
                "train_patient_hash": hash_patient_ids(train_ids),
                "validation_patient_hash": hash_patient_ids(val_ids),
                "test_patient_hash": hash_patient_ids(test_ids),
                "overlap_counts": {
                    "train_validation": overlap_1,
                    "train_test": overlap_2,
                    "validation_test": overlap_3
                },
                "target_distribution": {
                    "train": {
                        "yes": train_y.count("yes"),
                        "no": train_y.count("no"),
                        "missing": 0,
                        "recurrence_rate": train_y.count("yes") / len(train_y) if train_y else 0
                    },
                    "validation": {
                        "yes": val_y.count("yes"),
                        "no": val_y.count("no"),
                        "missing": 0,
                        "recurrence_rate": val_y.count("yes") / len(val_y) if val_y else 0
                    },
                    "test": {
                        "yes": test_y.count("yes"),
                        "no": test_y.count("no"),
                        "missing": 0,
                        "recurrence_rate": test_y.count("yes") / len(test_y) if test_y else 0
                    }
                }
            })
            
        out_dir = Path("data/metadata/hancock")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "data_split_manifest.json"
        
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
            
        return manifest
