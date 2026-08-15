"""
contradiction_detector.py

Generates ContradictionCandidate records by comparing EvidenceClaims
on multiple dimensions:
  - task
  - metric
  - mechanism
  - modality
  - domain

Rules:
  - Different datasets or entirely different tasks are NOT automatically contradictions.
  - Only claims with the SAME task AND metric AND at least one shared mechanism
    AND opposing directions qualify as contradiction candidates.
  - The system emits contradiction_candidate = True with a reason; it does NOT
    declare a scientific contradiction.
  - No HANCOCK data is used.
"""

import uuid
from typing import List, Optional

from backend.app.stage2.models import (
    ContradictionCandidate, EvidenceClaim, EvidenceStatus,
)


class ContradictionDetector:

    def detect(self, claims: List[EvidenceClaim]) -> List[ContradictionCandidate]:
        """
        Returns a list of ContradictionCandidate records.
        Also mutates claims in place to set contradiction_candidate = True.
        """
        candidates = []

        # Only compare direct or secondary empirical claims
        empirical = [
            c for c in claims
            if c.evidence_status in (
                EvidenceStatus.direct_empirical, EvidenceStatus.secondary_empirical
            )
            and c.result is not None
        ]

        for i, claim_a in enumerate(empirical):
            for claim_b in empirical[i + 1:]:
                candidate = self._compare(claim_a, claim_b)
                if candidate:
                    candidates.append(candidate)
                    claim_a.contradiction_candidate = True
                    claim_b.contradiction_candidate = True

        return candidates

    def _compare(
        self, a: EvidenceClaim, b: EvidenceClaim
    ) -> Optional[ContradictionCandidate]:
        """
        Returns a ContradictionCandidate only if ALL of the following hold:
          1. Both claims have an explicit result with a direction.
          2. The directions are opposing (improvement vs degradation).
          3. They share the same task (if stated in both).
          4. They share the same metric (if stated in both).
          5. They share at least one mechanism.
        """
        if not a.result or not b.result:
            return None

        dir_a = a.result.direction
        dir_b = b.result.direction

        # Directions must explicitly oppose
        if not (
            (dir_a == "improvement" and dir_b == "degradation") or
            (dir_a == "degradation" and dir_b == "improvement")
        ):
            return None

        # Tasks must match (if both specified)
        if a.task and b.task and a.task != b.task:
            return None

        # Metrics must match (if both specified)
        metric_a = a.result.metric
        metric_b = b.result.metric
        if metric_a and metric_b and metric_a.upper() != metric_b.upper():
            return None

        # Must share at least one mechanism
        shared_mechs = set(a.mechanisms) & set(b.mechanisms)
        if not shared_mechs:
            return None

        # Build comparison dimensions list
        dimensions: List[str] = ["mechanism", "direction"]
        if a.task and b.task:
            dimensions.append("task")
        if metric_a and metric_b:
            dimensions.append("metric")
        shared_mods = set(a.modalities) & set(b.modalities)
        if shared_mods:
            dimensions.append("modality")

        reason = (
            f"Claims share mechanism(s) {list(shared_mechs)} and report "
            f"opposing directions ({dir_a} vs {dir_b})"
            + (f" on metric {metric_a}" if metric_a else "")
            + (f" for task {a.task}" if a.task else "")
            + ". Requires human review to confirm or reject as contradiction."
        )

        return ContradictionCandidate(
            candidate_id=f"contra_{uuid.uuid4().hex[:8]}",
            evidence_claim_a=a.evidence_id,
            evidence_claim_b=b.evidence_id,
            reason=reason,
            comparison_dimensions=dimensions,
            shared_task=a.task if a.task == b.task else None,
            shared_metric=metric_a if metric_a and metric_b and metric_a.upper() == metric_b.upper() else None,
            shared_mechanisms=list(shared_mechs),
            direction_a=dir_a,
            direction_b=dir_b,
        )
