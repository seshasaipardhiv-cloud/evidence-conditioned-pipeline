import pytest
from backend.app.stage2.graph_builder import GraphBuilder
from backend.app.stage2.models import (
    PaperRecord, EvidenceClaim, Mechanism, MechanismCategory,
    Provenance, ExtractionMethod, ExtractionStatus
)

def test_graph_node_creation():
    builder = GraphBuilder()
    
    p = PaperRecord(paper_id="1", title="Test", authors=[], publication_year=2021, source="test", retrieval_date="now")
    m = Mechanism(mechanism_id="m1", canonical_name="cnn", category=MechanismCategory.representation)
    
    prov = Provenance(source_type="test", source_reference="1", extraction_method=ExtractionMethod.manual, extraction_status=ExtractionStatus.explicit, retrieval_date="now")
    c = EvidenceClaim(
        evidence_id="c1", paper_id="1", claim="test claim", mechanisms=["m1"], modalities=["clinical"],
        evidence_location="Abstract",
        extraction_method=ExtractionMethod.manual, provenance=prov
    )
    
    graph = builder.build_graph([p], [c], [m])
    assert len(graph.nodes) == 4 # Paper, Mechanism, Claim, Modality
    
    node_types = [n.node_type for n in graph.nodes]
    assert "Paper" in node_types
    assert "Mechanism" in node_types
    assert "EvidenceClaim" in node_types
    assert "Modality" in node_types
    
def test_graph_relationship_creation():
    builder = GraphBuilder()
    
    p = PaperRecord(paper_id="1", title="Test", authors=[], publication_year=2021, source="test", retrieval_date="now")
    m = Mechanism(mechanism_id="m1", canonical_name="cnn", category=MechanismCategory.representation)
    
    prov = Provenance(source_type="test", source_reference="1", extraction_method=ExtractionMethod.manual, extraction_status=ExtractionStatus.explicit, retrieval_date="now")
    c = EvidenceClaim(
        evidence_id="c1", paper_id="1", claim="test claim", mechanisms=["m1"], modalities=["clinical"],
        evidence_location="Abstract",
        extraction_method=ExtractionMethod.manual, provenance=prov
    )
    
    graph = builder.build_graph([p], [c], [m])
    assert len(graph.relationships) == 3 # reports, uses, uses_modality
    
    rel_types = [r.relationship_type for r in graph.relationships]
    assert "reports" in rel_types
    assert "uses" in rel_types
    assert "uses_modality" in rel_types
