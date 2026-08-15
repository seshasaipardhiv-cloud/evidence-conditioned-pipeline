from backend.app.stage3.models import Component, Mechanism, MECHANISM_TO_COMPONENT

def test_fixed_component_taxonomy():
    components = list(Component)
    assert len(components) == 8
    expected = [
        "missing_value_handling", "categorical_encoding", "feature_representation",
        "modality_fusion", "base_learner", "loss_function", "imbalance_handling", "ensembling"
    ]
    for e in expected:
        assert any(c.value == e for c in components)

def test_closed_mechanism_vocabulary():
    mechanisms = list(Mechanism)
    assert len(mechanisms) == 12

def test_no_unknown_mechanisms():
    for mech in Mechanism:
        assert mech in MECHANISM_TO_COMPONENT
        assert isinstance(MECHANISM_TO_COMPONENT[mech], Component)
