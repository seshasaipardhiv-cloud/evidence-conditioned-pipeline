import json
from pathlib import Path
from collections import Counter

def run_semantics_audit():
    clinical_path = Path("data/raw/hancock/structured/StructuredData/clinical_data.json")
    with open(clinical_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    total_records = len(data)
    
    # Analyze exactly these fields:
    candidate_fields = [
        "survival_status",
        "survival_status_with_cause",
        "recurrence",
        "days_to_recurrence",
        "days_to_last_information",
        "days_to_progress_1",
        "days_to_progress_2",
        "days_to_metastasis_1"
    ]
    
    report_items = []
    
    for field in candidate_fields:
        values = []
        missing = 0
        for row in data:
            v = row.get(field)
            if v is None:
                missing += 1
            else:
                values.append(v)
                
        types = list(set(type(v).__name__ for v in values))
        c = Counter(values)
        unique_count = len(c)
        missing_rate = missing / total_records if total_records > 0 else 0
        
        # Determine candidate role
        if field in ["survival_status", "recurrence"]:
            candidate_role = "event_indicator"
            leakage_role = "derived from outcome or is outcome"
        elif "days_to_" in field:
            if field == "days_to_recurrence":
                # Special logic found during research: it's missing for all negative recurrences
                candidate_role = "unresolved"
                leakage_role = "time of positive outcome only"
            elif field == "days_to_last_information":
                candidate_role = "time_to_event" # representing censoring time
                leakage_role = "post_outcome"
            else:
                candidate_role = "post_outcome"
                leakage_role = "post_outcome"
        elif field == "survival_status_with_cause":
            candidate_role = "potential_target"
            leakage_role = "derived from outcome"
        else:
            candidate_role = "ordinary_feature"
            leakage_role = "none"
            
        report_items.append({
            "field_name": field,
            "data_type": types[0] if types else "null",
            "unique_value_count": unique_count,
            "sample_value_types": types,
            "missing_count": missing,
            "missing_rate": missing_rate,
            "observed_value_pattern": f"Top 5 values: {[str(k) for k in list(c.keys())[:5]]}",
            "candidate_role": candidate_role,
            "possible_leakage_role": leakage_role,
            "source_schema": "clinical_data",
            "confidence_status": "HIGH" if candidate_role != "unresolved" else "LOW"
        })
        
    out_dir = Path("data/metadata/hancock")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "target_semantics_report.json"
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report_items, f, indent=2)
        
    return out_path

if __name__ == "__main__":
    run_semantics_audit()
