from backend.app.stage3.models import (
    Mechanism, Component, ContextualBelief, EvidenceMatch, Stage3Context, PipelineSpecification
)
from backend.app.stage3.ranker import RankerComposer
from backend.app.stage3.belief_updater import BeliefUpdater

def test_pipeline_composition_insufficient_evidence():
    context = Stage3Context()
    beliefs = {mech: ContextualBelief(mechanism=mech, component=Component.missing_value_handling) for mech in Mechanism}
    
    # Empty matches -> insufficient evidence
    ranker = RankerComposer(evidence_threshold=0.1)
    spec, rankings = ranker.rank_and_compose(context, beliefs)
    
    for comp in Component:
        assert rankings[comp].selection_status == "insufficient_evidence"
        assert rankings[comp].winner is None

def test_pipeline_composition_winner():
    context = Stage3Context()
    # Create mock belief that has strong support
    beliefs = {mech: ContextualBelief(mechanism=mech, component=Component.modality_fusion) for mech in Mechanism}
    beliefs[Mechanism.late_fusion].alpha = 5.0
    beliefs[Mechanism.late_fusion].supporting_evidence_count = 4
    beliefs[Mechanism.late_fusion].supporting_matches = [
        EvidenceMatch(paper_id="1", mechanism_id="late_fusion", context_similarity=0.8, evidence_quality=1.0, direction="positive")
    ]
    
    ranker = RankerComposer(evidence_threshold=0.1)
    spec, rankings = ranker.rank_and_compose(context, beliefs)
    
    assert rankings[Component.modality_fusion].selection_status == "selected"
    assert rankings[Component.modality_fusion].winner == Mechanism.late_fusion

def test_belief_update_directions():
    updater = BeliefUpdater()
    matches = [
        EvidenceMatch(paper_id="1", mechanism_id="early_fusion", context_similarity=0.5, evidence_quality=1.0, direction="positive"),
        EvidenceMatch(paper_id="2", mechanism_id="early_fusion", context_similarity=0.5, evidence_quality=1.0, direction="negative"),
        EvidenceMatch(paper_id="3", mechanism_id="early_fusion", context_similarity=0.5, evidence_quality=1.0, direction="neutral")
    ]
    
    beliefs = updater.update_beliefs(matches)
    ef = beliefs[Mechanism.early_fusion]
    
    # Prior alpha 1.0, beta 1.0
    assert ef.alpha == 1.5
    assert ef.beta == 1.5
    assert ef.supporting_evidence_count == 1
    assert ef.contradicting_evidence_count == 1
    assert ef.neutral_evidence_count == 1
    assert ef.posterior_mean == 0.5

def test_no_raw_patient_text_leakage():
    # Context should not have any raw text fields
    ctx = Stage3Context()
    assert not hasattr(ctx, "raw_text")
    assert not hasattr(ctx, "patient_data")
    assert ctx.task == "unknown"
