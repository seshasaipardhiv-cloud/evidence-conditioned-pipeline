import math
from typing import List
from backend.app.stage3.models import Stage3Context

def calculate_modality_jaccard(ctx_mods: List[str], ev_mods: List[str]) -> float:
    if not ctx_mods and not ev_mods:
        return 1.0
    set1 = set(m.lower() for m in ctx_mods)
    set2 = set(m.lower() for m in ev_mods)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    if union == 0:
        return 0.0
    return intersection / union

def calculate_sample_size_similarity(ctx_size: int, ev_size: int) -> float:
    if ctx_size is None or ev_size is None or ctx_size <= 0 or ev_size <= 0:
        return 0.5 # Neutral fallback if sample size is unknown
        
    diff = abs(math.log10(ctx_size) - math.log10(ev_size))
    # E.g. diff of 1 (10x difference) -> similarity drops
    # Let's say diff of 2 (100x difference) is 0 similarity
    sim = max(0.0, 1.0 - (diff / 2.0))
    return sim

def calculate_missingness_similarity(ctx_miss: float, ev_miss: float) -> float:
    if ctx_miss is None or ev_miss is None:
        return 0.5
    # Absolute difference [0, 1]
    diff = abs(ctx_miss - ev_miss)
    return max(0.0, 1.0 - diff)

def calculate_imbalance_similarity(ctx_imb: float, ev_imb: float) -> float:
    if ctx_imb is None or ev_imb is None:
        return 0.5
    diff = abs(ctx_imb - ev_imb)
    return max(0.0, 1.0 - diff)

def calculate_task_similarity(ctx_task: str, ev_task: str) -> float:
    ct = str(ctx_task).lower()
    et = str(ev_task).lower()
    if ct == "unknown" or et == "unknown" or et == "none":
        return 0.5 # Broad match penalty
    if ct == et:
        return 1.0
    return 0.0

def calculate_context_similarity(context: Stage3Context, experiment: dict) -> float:
    """
    Computes a deterministic similarity score [0.0, 1.0] between a Stage1 context
    and an evidence experiment/claim.
    """
    ev_mods = experiment.get("modalities") or []
    mod_sim = calculate_modality_jaccard(context.modalities, ev_mods)
    
    ev_task = experiment.get("task", "unknown")
    task_sim = calculate_task_similarity(context.task, ev_task)
    
    ev_size = experiment.get("sample_count")
    size_sim = calculate_sample_size_similarity(context.sample_size, ev_size)
    
    # Missingness and imbalance are rarely extracted directly in the evidence corpus 
    # unless part of dataset_characteristics, so we might fallback to 0.5.
    miss_sim = 0.5
    imb_sim = 0.5
    
    # Weighting scheme (Task and Modalities are most important)
    weights = [
        (task_sim, 0.4),
        (mod_sim, 0.4),
        (size_sim, 0.1),
        (miss_sim, 0.05),
        (imb_sim, 0.05)
    ]
    
    total_sim = sum(val * weight for val, weight in weights)
    return total_sim
