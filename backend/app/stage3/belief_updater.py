from typing import List, Dict
from backend.app.stage3.models import Mechanism, ContextualBelief, EvidenceMatch, MECHANISM_TO_COMPONENT

class BeliefUpdater:
    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        
    def update_beliefs(self, matches: List[EvidenceMatch]) -> Dict[Mechanism, ContextualBelief]:
        beliefs = {}
        for mech in Mechanism:
            beliefs[mech] = ContextualBelief(
                mechanism=mech,
                component=MECHANISM_TO_COMPONENT[mech],
                prior_alpha=self.prior_alpha,
                prior_beta=self.prior_beta,
                alpha=self.prior_alpha,
                beta=self.prior_beta
            )
            
        for match in matches:
            mech_id = match.mechanism_id
            if not mech_id:
                continue
                
            mech = Mechanism(mech_id)
            belief = beliefs[mech]
            
            weight = match.context_similarity * match.evidence_quality
            
            if match.direction == "positive":
                belief.alpha += weight
                belief.supporting_evidence_count += 1
                belief.supporting_matches.append(match)
            elif match.direction == "negative":
                belief.beta += weight
                belief.contradicting_evidence_count += 1
                belief.contradicting_matches.append(match)
            elif match.direction == "neutral":
                # Neutral evidence doesn't shift the mean directionally, 
                # but we could optionally increase both alpha and beta to increase certainty.
                # The requirements state: "neutral result -> no directional update"
                belief.neutral_evidence_count += 1
                belief.neutral_matches.append(match)
                
        return beliefs
