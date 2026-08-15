import pytest
from backend.app.stage2.mechanism_mapper import MechanismMapper
from backend.app.stage2.normalizer import Normalizer
from backend.app.stage2.models import MechanismCategory

def test_mechanism_mapping():
    mapper = MechanismMapper()
    cat, canonical = mapper.map_mechanism("We used a CNN for feature extraction.")
    assert cat == MechanismCategory.representation
    assert canonical == "cnn"

def test_unmapped_mechanism():
    mapper = MechanismMapper()
    cat, canonical = mapper.map_mechanism("We used a magic quantum crystal.")
    assert cat == MechanismCategory.unmapped
    assert canonical == "We used a magic quantum crystal."

def test_dataset_characteristics_normalization():
    norm = Normalizer()
    raw = {
        "sample_count": "100",
        "missingness": "0.05",
        "class_imbalance": True,
        "text_available": "yes",
        "non_existent_field": 123
    }
    char = norm.normalize_dataset_characteristics(raw)
    assert char.sample_count == 100
    assert char.missingness == 0.05
    assert char.class_imbalance is True
    assert char.text_available is True
    assert char.feature_count is None
