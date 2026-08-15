import pytest
from backend.app.stage2.acquisition import EvidenceAcquisition
from backend.app.stage2.models import PaperRecord

def test_paper_deduplication():
    acq = EvidenceAcquisition()
    # Mock records with same DOI
    r1 = PaperRecord(paper_id="1", title="A", authors=[], publication_year=2021, doi="10.1000/1", source="mock", retrieval_date="now")
    r2 = PaperRecord(paper_id="2", title="B", authors=[], publication_year=2021, doi="10.1000/1", source="mock", retrieval_date="now")
    
    unique = acq._deduplicate([r1, r2])
    assert len(unique) == 1
    assert unique[0].paper_id == "1"

def test_pmid_normalization_deduplication():
    acq = EvidenceAcquisition()
    # Mock records with same PMID but missing DOI
    r1 = PaperRecord(paper_id="1", title="A", authors=[], publication_year=2021, pmid="12345", source="mock", retrieval_date="now")
    r2 = PaperRecord(paper_id="2", title="B", authors=[], publication_year=2021, pmid="12345", source="mock", retrieval_date="now")
    
    unique = acq._deduplicate([r1, r2])
    assert len(unique) == 1
    
def test_fallback_deduplication():
    acq = EvidenceAcquisition()
    # Mock records with same title and year
    r1 = PaperRecord(paper_id="1", title="Same Title", authors=[], publication_year=2021, source="mock", retrieval_date="now")
    r2 = PaperRecord(paper_id="2", title="  same TITLE ", authors=[], publication_year=2021, source="mock", retrieval_date="now")
    
    unique = acq._deduplicate([r1, r2])
    assert len(unique) == 1

def test_missing_full_text():
    acq = EvidenceAcquisition()
    records, total, failed = acq.fetch_seed_papers()
    if records:
        assert not records[0].full_text_available
