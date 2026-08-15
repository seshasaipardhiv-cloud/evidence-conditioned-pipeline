import json
from pathlib import Path
from typing import List, Dict, Optional
from backend.app.stage3.models import Stage3Context, Mechanism, EvidenceMatch, MECHANISM_TO_COMPONENT
from backend.app.stage3.similarity import calculate_context_similarity

class EvidenceMatcher:
    def __init__(self, processed_dir: str = "evidence/processed"):
        self.processed_dir = Path(processed_dir)
        self.experiments = []
        self.claims = []
        self.papers = {}
        self._load_data()
        
    def _load_data(self):
        exp_path = self.processed_dir / "experiments.jsonl"
        if exp_path.exists():
            with open(exp_path, "r", encoding="utf-8") as f:
                self.experiments = [json.loads(line) for line in f if line.strip()]
                
        claim_path = self.processed_dir / "evidence_claims.jsonl"
        if claim_path.exists():
            with open(claim_path, "r", encoding="utf-8") as f:
                self.claims = [json.loads(line) for line in f if line.strip()]
                
        paper_path = self.processed_dir / "papers.jsonl"
        if paper_path.exists():
            with open(paper_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        p = json.loads(line)
                        self.papers[p["paper_id"]] = p

    def match_evidence(self, context: Stage3Context) -> List[EvidenceMatch]:
        matches = []
        
        # We need a mapping from vocabulary strings to Mechanism enum
        # This translates raw text (e.g. "cross attention") into the formal vocabulary.
        mechanism_string_map = {
            "mean imputation": Mechanism.mean_imputation,
            "mean_imputation": Mechanism.mean_imputation,
            "one hot": Mechanism.one_hot_encoding,
            "one_hot": Mechanism.one_hot_encoding,
            "cnn": Mechanism.cnn_representation,
            "transformer": Mechanism.transformer_representation,
            "late fusion": Mechanism.late_fusion,
            "late_fusion": Mechanism.late_fusion,
            "early fusion": Mechanism.early_fusion,
            "early_fusion": Mechanism.early_fusion,
            "cross attention": Mechanism.cross_attention,
            "cross-attention": Mechanism.cross_attention,
            "cross_attention": Mechanism.cross_attention,
            "joint embedding": Mechanism.joint_embedding,
            "joint_embedding": Mechanism.joint_embedding,
            "gradient boosting": Mechanism.gradient_boosting,
            "xgboost": Mechanism.gradient_boosting,
            "lightgbm": Mechanism.gradient_boosting,
            "focal loss": Mechanism.focal_loss,
            "focal_loss": Mechanism.focal_loss,
            "class weight": Mechanism.class_weighted_sampling,
            "weighted sampling": Mechanism.class_weighted_sampling,
            "class_weighted_sampling": Mechanism.class_weighted_sampling,
            "ensemble": Mechanism.average_ensembling,
            "average": Mechanism.average_ensembling,
            "ensemble_fusion": Mechanism.average_ensembling,
        }
        
        def _get_evidence_quality(paper_id: str, source_scope: str) -> float:
            p = self.papers.get(paper_id, {})
            # Full-text quantitative evidence = 1.0, abstract-only = 0.6
            quality = 1.0
            if source_scope == "abstract" or not p.get("full_text_available"):
                quality = 0.6
            return quality
            
        def _map_string_to_mechanism(text: str) -> Optional[Mechanism]:
            if not text: return None
            text = text.lower()
            for key, mech in mechanism_string_map.items():
                if key in text:
                    return mech
            return None
            
        def _get_baseline_str(b_list) -> Optional[str]:
            if not b_list: return None
            b = b_list[0]
            if isinstance(b, dict):
                return str(b.get("name", b))
            return str(b)
            
        # Match from experiments (quantitative)
        for exp in self.experiments:
            paper_id = exp.get("paper_id")
            source_scope = exp.get("source_scope", "full_text")
            
            # Map mechanism from fusion_strategy
            fusion_str = exp.get("fusion_strategy")
            fusion_mech = _map_string_to_mechanism(fusion_str)
            if fusion_mech:
                sim = calculate_context_similarity(context, exp)
                qual = _get_evidence_quality(paper_id, source_scope)
                # Results in experiments imply positive evidence for the proposed method
                matches.append(EvidenceMatch(
                    paper_id=paper_id,
                    experiment_id=exp.get("experiment_id"),
                    mechanism_id=fusion_mech.value,
                    source_scope=source_scope,
                    result=str(exp.get("reported_results")),
                    metric=None,
                    baseline=_get_baseline_str(exp.get("baselines")),
                    context_similarity=sim,
                    evidence_quality=qual,
                    direction="positive" # Implied positive for reported fusion method over baseline
                ))
                
            # Map mechanism from proposed_method
            prop_str = exp.get("proposed_method")
            prop_mech = _map_string_to_mechanism(prop_str)
            if prop_mech and prop_mech != fusion_mech:
                sim = calculate_context_similarity(context, exp)
                qual = _get_evidence_quality(paper_id, source_scope)
                matches.append(EvidenceMatch(
                    paper_id=paper_id,
                    experiment_id=exp.get("experiment_id"),
                    mechanism_id=prop_mech.value,
                    source_scope=source_scope,
                    result=str(exp.get("reported_results")),
                    metric=None,
                    baseline=_get_baseline_str(exp.get("baselines")),
                    context_similarity=sim,
                    evidence_quality=qual,
                    direction="positive"
                ))

        # Match from evidence_claims (qualitative & quantitative directions)
        for claim in self.claims:
            paper_id = claim.get("paper_id")
            source_scope = claim.get("source_scope", "full_text")
            
            # Determine direction
            status = claim.get("evidence_status")
            direction = "unknown"
            if isinstance(status, dict) and "value" in status:
                direction = status["value"]
            elif hasattr(status, "value"):
                direction = status.value
            elif isinstance(status, str):
                direction = status
                
            if direction not in ["positive", "negative", "neutral"]:
                continue
                
            claim_text = claim.get("claim", "")
            matched_mech = _map_string_to_mechanism(claim_text)
            
            if matched_mech:
                # Mock an experiment dict to calculate similarity
                mock_exp = {
                    "modalities": claim.get("modalities", []),
                    "task": claim.get("task", "unknown")
                }
                sim = calculate_context_similarity(context, mock_exp)
                qual = _get_evidence_quality(paper_id, source_scope)
                
                matches.append(EvidenceMatch(
                    paper_id=paper_id,
                    claim_id=claim.get("evidence_id"),
                    mechanism_id=matched_mech.value,
                    source_scope=source_scope,
                    result=str(claim.get("result")) if claim.get("result") else None,
                    metric=claim.get("metric"),
                    baseline=str(claim.get("baseline")) if claim.get("baseline") else None,
                    context_similarity=sim,
                    evidence_quality=qual,
                    direction=direction
                ))
                
        return matches
