from typing import Dict, List, Optional
from backend.app.stage3.models import (
    Mechanism, Component, ContextualBelief, 
    MechanismScore, MechanismRanking, PipelineSpecification, Stage3Context, MECHANISM_TO_COMPONENT
)

class RankerComposer:
    def __init__(self, evidence_threshold: float = 0.1):
        # The threshold of total weighted evidence needed to confidently select a mechanism over the prior.
        self.evidence_threshold = evidence_threshold

    def rank_and_compose(
        self, 
        context: Stage3Context, 
        beliefs: Dict[Mechanism, ContextualBelief]
    ) -> PipelineSpecification:
        
        # 1. Score all mechanisms
        mechanism_scores = []
        scores_by_comp: Dict[Component, List[MechanismScore]] = {c: [] for c in Component}
        
        for mech, belief in beliefs.items():
            total_matches = belief.supporting_matches + belief.contradicting_matches + belief.neutral_matches
            
            sim_sum = sum(m.context_similarity for m in total_matches)
            qual_sum = sum(m.evidence_quality for m in total_matches)
            ev_count = len(total_matches)
            
            # Weighted support logic - score combines posterior mean and context similarity volume
            # The score must be transparent.
            score = MechanismScore(
                mechanism=mech,
                component=belief.component,
                posterior_mean=belief.posterior_mean,
                evidence_count=ev_count,
                support_count=belief.supporting_evidence_count,
                contradiction_count=belief.contradicting_evidence_count,
                context_similarity_sum=sim_sum,
                evidence_quality_sum=qual_sum,
                final_score=belief.posterior_mean * sim_sum
            )
            mechanism_scores.append(score)
            scores_by_comp[belief.component].append(score)
            
        # 2. Rank within components
        rankings: Dict[Component, MechanismRanking] = {}
        selected_mechanisms = {}
        alternative_mechanisms = {}
        selection_rationale = {}
        expected_baselines = []
        
        for comp in Component:
            comp_scores = scores_by_comp[comp]
            # Sort by final score descending, then by evidence count descending to break ties
            comp_scores.sort(key=lambda x: (x.final_score, x.evidence_count), reverse=True)
            
            winner = None
            status = "insufficient_evidence"
            alts = []
            insuff = []
            
            if comp_scores:
                best = comp_scores[0]
                # If there's literally no evidence or the final score is 0.0, we mark insufficient.
                # Since final_score = posterior_mean * sim_sum, if sim_sum == 0, score is 0.
                if best.final_score <= self.evidence_threshold:
                    status = "insufficient_evidence"
                    insuff = [s.mechanism for s in comp_scores]
                    selection_rationale[comp.value] = "No mechanism has adequate contextual support."
                else:
                    # Check for ties
                    if len(comp_scores) > 1 and abs(comp_scores[0].final_score - comp_scores[1].final_score) < 1e-5:
                        status = "tie"
                        insuff = [s.mechanism for s in comp_scores[2:]]
                        alts = [comp_scores[1]]
                        selection_rationale[comp.value] = f"Tie between {comp_scores[0].mechanism.value} and {comp_scores[1].mechanism.value}."
                    else:
                        status = "selected"
                        winner = best.mechanism
                        alts = comp_scores[1:]
                        selection_rationale[comp.value] = f"Selected {winner.value} with posterior mean {best.posterior_mean:.3f} and similarity sum {best.context_similarity_sum:.3f}."
                        
                        # Collect baselines from the winner's supporting evidence
                        for m in beliefs[winner].supporting_matches:
                            if m.baseline and m.baseline not in expected_baselines:
                                expected_baselines.append(m.baseline)
            
            rankings[comp] = MechanismRanking(
                component=comp,
                winner=winner,
                selection_status=status,
                alternatives=alts,
                insufficient_evidence=insuff
            )
            selected_mechanisms[comp.value] = winner.value if winner else None
            alternative_mechanisms[comp.value] = [a.model_dump() for a in alts]
            
        # 3. Compile specification
        mech_scores_dict = {s.mechanism.value: s.model_dump() for s in mechanism_scores}
        beliefs_dict = {k.value: v.model_dump(exclude={"supporting_matches", "contradicting_matches", "neutral_matches"}) for k, v in beliefs.items()}
        
        sup_ev = {k.value: v.supporting_matches for k, v in beliefs.items() if v.supporting_matches}
        con_ev = {k.value: v.contradicting_matches for k, v in beliefs.items() if v.contradicting_matches}
        
        uncertainties = {k.value: (v.prior_alpha + v.prior_beta) / (v.alpha + v.beta) for k, v in beliefs.items()}
        
        spec = PipelineSpecification(
            problem_context=context,
            fixed_components=[c.value for c in Component],
            selected_mechanisms=selected_mechanisms,
            alternative_mechanisms=alternative_mechanisms,
            mechanism_scores=mech_scores_dict,
            contextual_beliefs=beliefs_dict,
            supporting_evidence=sup_ev,
            contradicting_evidence=con_ev,
            uncertainty=uncertainties,
            selection_rationale=selection_rationale,
            expected_baselines=expected_baselines
        )
        
        return spec, rankings
