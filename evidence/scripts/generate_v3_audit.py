"""
generate_v3_audit.py
Compares extracted ExperimentRecord fields against verified ground truth
and produces stage2b_claim_validation_v3.json + stage2b_claim_validation_summary_v3.json.

Run from the project root:
  python evidence/scripts/generate_v3_audit.py
"""

import json
import os
from typing import Any, Dict, List, Optional

EXPERIMENTS_PATH = "evidence/processed/experiments.jsonl"
OUT_RECORDS = "evidence/metadata/stage2b_claim_validation_v3.json"
OUT_SUMMARY = "evidence/metadata/stage2b_claim_validation_summary_v3.json"

# Ground-truth verified values per paper (from the Stage 2B clinical audit).
VERIFIED: Dict[str, Dict[str, Any]] = {
    "paper_10.1038_s42256-023-00633-5": {
        "dataset": "TCGA",
        "task": "survival_prediction",
        "modalities": ["clinical", "imaging", "omics"],
        "proposed_method": None,
        "baselines": None,        
        "metric": None,
        "result": None,
        "fusion_strategy": "early_fusion",
    },
    "paper_10.3390_bioengineering11010013": {
        "dataset": "TCIA",
        "task": "diagnosis",
        "modalities": ["imaging"],
        "proposed_method": None,
        "baselines": None,
        "metric": "AUC",
        "result": 0.76,
        "fusion_strategy": "gated_fusion",
    },
    "paper_38396486": {
        "dataset": None,          
        "task": "survival_prediction",
        "modalities": ["clinical", "imaging", "text"],
        "proposed_method": None,
        "baselines": None,
        "metric": "C-INDEX",
        "result": 0.796,
        "fusion_strategy": "cross_attention",
    },
    "paper_39074400": {
        "dataset": None,          
        "task": "diagnosis",
        "modalities": ["clinical", "imaging"],
        "proposed_method": None,
        "baselines": None,
        "metric": "AUC",
        "result": 0.77,
        "fusion_strategy": "early_fusion",
    },
    "paper_40325104": {
        "dataset": "TCGA",
        "task": "survival_prediction",
        "modalities": ["clinical", "omics"],
        "proposed_method": None,
        "baselines": None,
        "metric": None,
        "result": None,
        "fusion_strategy": "late_fusion",
    },
    "paper_40449048": {
        "dataset": None,
        "task": "diagnosis",
        "modalities": ["clinical", "imaging"],
        "proposed_method": None,
        "baselines": None,
        "metric": "Accuracy",
        "result": 0.99,
        "fusion_strategy": None,
    },
    "paper_41131352": {
        "dataset": "TCGA",
        "task": "survival_prediction",
        "modalities": ["clinical", "imaging", "omics"],
        "proposed_method": "HONeYBEE",
        "baselines": ["PORPOISE"],
        "metric": "Accuracy",
        "result": 0.9021,
        "fusion_strategy": "intermediate_fusion",
    },
    "paper_41353186": {
        "dataset": "TCGA",
        "task": "survival_prediction",
        "modalities": ["clinical", "imaging", "text"],
        "proposed_method": "CALM",
        "baselines": ["PORPOISE"],
        "metric": "C-INDEX",
        "result": 0.606,
        "fusion_strategy": "cross_attention",
    },
}

def _str_status(extracted: Any, verified: Any, fuzzy: bool = False) -> str:
    if verified is None and extracted is None:
        return "VERIFIED"
    if verified is None:
        return "INCORRECT" # If ground truth is None and we extracted something, it's a false positive (INCORRECT)
    if extracted is None:
        return "NOT_AVAILABLE"
    if fuzzy:
        e = str(extracted).lower()
        v = str(verified).lower()
        if e == v or v in e or e in v:
            return "VERIFIED"
        return "INCORRECT"
    return "VERIFIED" if extracted == verified else "INCORRECT"


def _set_status(extracted: List, verified: Optional[List]) -> str:
    if verified is None:
        if extracted: return "INCORRECT"
        return "VERIFIED"
    if not isinstance(extracted, list):
        return "INCORRECT"
    e_set = set(m.lower() for m in extracted)
    v_set = set(m.lower() for m in verified)
    if e_set == v_set:
        return "VERIFIED"
    if e_set and v_set and e_set.issubset(v_set | {"text"}):
        if v_set.issubset(e_set | {"text"}):
            return "VERIFIED"
    overlap = e_set & v_set
    if overlap and len(overlap) >= len(v_set) * 0.5:
        # If we extracted something NOT in verified, it's a false positive on that item, but might be partially verified
        # However, for false positive tracking, let's look at incorrects
        return "PARTIALLY_VERIFIED"
    if not e_set:
        return "NOT_AVAILABLE"
    return "INCORRECT"


def _baseline_status(extracted: List[Dict], verified: Optional[List]) -> str:
    if verified is None:
        if extracted: return "INCORRECT"
        return "VERIFIED"
    if not extracted:
        return "NOT_AVAILABLE"
    extracted_names = {b.get("name", "").upper() for b in extracted}
    verified_names = {v.upper() for v in verified}
    if verified_names.issubset(extracted_names):
        return "VERIFIED"
    if verified_names & extracted_names:
        return "PARTIALLY_VERIFIED"
    return "INCORRECT"


def main():
    if not os.path.exists(EXPERIMENTS_PATH):
        print(f"ERROR: {EXPERIMENTS_PATH} not found.")
        return

    with open(EXPERIMENTS_PATH) as f:
        experiments = [json.loads(line) for line in f if line.strip()]

    records = []
    field_breakdown: Dict[str, Dict[str, int]] = {}

    false_extractions = {
        "dataset": [],
        "method": [],
        "modality": [],
        "task": [],
        "mechanism": []
    }

    for exp in experiments:
        pid = exp.get("paper_id", "")
        if pid not in VERIFIED:
            continue
        vv = VERIFIED[pid]

        def add(field_name: str, extracted: Any, verified: Any, status: str):
            records.append({
                "paper_id": pid,
                "field": field_name,
                "extracted_value": extracted,
                "verified_value": verified,
                "verification_status": status,
            })
            if field_name not in field_breakdown:
                field_breakdown[field_name] = {"VERIFIED": 0, "INCORRECT": 0,
                                                "NOT_AVAILABLE": 0, "PARTIALLY_VERIFIED": 0}
            field_breakdown[field_name][status] = field_breakdown[field_name].get(status, 0) + 1
            
            # Track false positives
            if status == "INCORRECT" and extracted is not None:
                if field_name == "dataset":
                    false_extractions["dataset"].append({"pid": pid, "val": extracted})
                elif field_name == "proposed_method":
                    false_extractions["method"].append({"pid": pid, "val": extracted})
                elif field_name == "task":
                    false_extractions["task"].append({"pid": pid, "val": extracted})
            
            if field_name == "modalities" and (status == "INCORRECT" or status == "PARTIALLY_VERIFIED"):
                 if extracted:
                     v_set = set(m.lower() for m in (verified or []))
                     for m in extracted:
                         if m.lower() not in v_set:
                             false_extractions["modality"].append({"pid": pid, "val": m})
        
        # Mechanisms are part of claims, but let's check field provenance or graph directly if needed.
        # For now, let's just track the main fields.

        ext_ds = exp.get("dataset")
        add("dataset", ext_ds, vv["dataset"], _str_status(ext_ds, vv["dataset"], fuzzy=True))

        ext_task = exp.get("task")
        add("task", ext_task, vv["task"], _str_status(ext_task, vv["task"], fuzzy=(ext_task == "classification" and vv["task"] == "diagnosis")))

        ext_mods = exp.get("modalities", [])
        add("modalities", ext_mods, vv["modalities"], _set_status(ext_mods, vv["modalities"]))

        ext_meth = exp.get("proposed_method")
        add("proposed_method", ext_meth, vv["proposed_method"], _str_status(ext_meth, vv["proposed_method"], fuzzy=True))

        ext_baselines = exp.get("baselines", [])
        add("baselines", [b.get("name") for b in ext_baselines], vv["baselines"], _baseline_status(ext_baselines, vv["baselines"]))

        ext_fus = exp.get("fusion_strategy")
        add("fusion_strategy", ext_fus, vv["fusion_strategy"], _str_status(ext_fus, vv["fusion_strategy"]))

        results = exp.get("reported_results", [])
        ext_metric = results[0]["metric"] if results else None
        ver_metric = vv["metric"]
        metric_status = "VERIFIED" if (
            ext_metric and ver_metric and ext_metric.upper().replace("-", "") == ver_metric.upper().replace("-", "")
        ) else "NOT_AVAILABLE" if not ext_metric else "INCORRECT"
        if not ext_metric and not ver_metric:
            metric_status = "VERIFIED"
        add("metric", ext_metric, ver_metric, metric_status)

        ext_val = results[0]["method_value"] if results else None
        ver_val = vv["result"]
        if ext_val is None and ver_val is None:
            res_status = "VERIFIED"
        elif ext_val is None:
            res_status = "NOT_AVAILABLE"
        elif ver_val is None:
            res_status = "INCORRECT"
        elif abs(float(ext_val) - float(ver_val)) < 0.01:
            res_status = "VERIFIED"
        else:
            res_status = "INCORRECT"
        add("result", ext_val, ver_val, res_status)

    with open(OUT_RECORDS, "w") as f:
        json.dump(records, f, indent=2)

    total = len(records)
    verified_n = sum(1 for r in records if r["verification_status"] == "VERIFIED")
    incorrect_n = sum(1 for r in records if r["verification_status"] == "INCORRECT")
    partial_n = sum(1 for r in records if r["verification_status"] == "PARTIALLY_VERIFIED")
    not_avail_n = sum(1 for r in records if r["verification_status"] == "NOT_AVAILABLE")

    summary = {
        "stage": "2B_v3",
        "audit": {
            "total_fields": total,
            "verified": verified_n,
            "incorrect": incorrect_n,
            "partially_verified": partial_n,
            "not_available": not_avail_n,
        },
        "field_breakdown": field_breakdown,
        "false_extractions": false_extractions,
    }

    with open(OUT_SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)
        
    print("=== V3 AUDIT SUMMARY ===")
    print(f"VERIFIED             : {verified_n}")
    print(f"INCORRECT            : {incorrect_n}")
    print(f"PARTIALLY_VERIFIED   : {partial_n}")
    print(f"NOT_AVAILABLE        : {not_avail_n}")
    print()
    print("False Positive Extractions:")
    for k, v in false_extractions.items():
        print(f"  {k}: {len(v)} occurrences -> {v}")

if __name__ == "__main__":
    main()
