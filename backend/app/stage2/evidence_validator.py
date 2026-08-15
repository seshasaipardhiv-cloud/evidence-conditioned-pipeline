from typing import List, Dict, Any, Tuple
from backend.app.stage2.models import EvidenceClaim, EvidenceStatus

class EvidenceValidator:
    def __init__(self):
        pass

    def validate_claims(self, claims: List[EvidenceClaim]) -> List[EvidenceClaim]:
        """
        Validates all claims and identifies contradiction candidates.
        Only valid claims are returned.
        """
        valid_claims = []
        for claim in claims:
            # Basic validation: must have provenance and mechanisms
            if claim.provenance and claim.mechanisms:
                # Discard unverified numericals or hallucinated stuff by just retaining claims that meet structure
                valid_claims.append(claim)
                
        # Identify contradiction candidates
        self._flag_contradictions(valid_claims)
        
        return valid_claims

    def _flag_contradictions(self, claims: List[EvidenceClaim]) -> None:
        """
        Flags contradiction_candidate = True if two claims use similar mechanism/modality 
        but report opposing empirical directions.
        """
        for i, claim_a in enumerate(claims):
            if claim_a.evidence_status not in (EvidenceStatus.direct_empirical, EvidenceStatus.secondary_empirical):
                continue
                
            for j in range(i + 1, len(claims)):
                claim_b = claims[j]
                if claim_b.evidence_status not in (EvidenceStatus.direct_empirical, EvidenceStatus.secondary_empirical):
                    continue
                    
                # Check for overlap in mechanisms
                common_mechs = set(claim_a.mechanisms).intersection(set(claim_b.mechanisms))
                if not common_mechs:
                    continue
                    
                # Check for opposing empirical results
                if claim_a.result and claim_b.result:
                    dir_a = claim_a.result.direction
                    dir_b = claim_b.result.direction
                    
                    if (dir_a == "improvement" and dir_b == "degradation") or (dir_b == "improvement" and dir_a == "degradation"):
                        claim_a.contradiction_candidate = True
                        claim_b.contradiction_candidate = True
