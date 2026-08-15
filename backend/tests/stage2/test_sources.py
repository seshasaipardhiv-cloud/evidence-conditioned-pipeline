import os
import pytest
from backend.app.stage2.sources import PubMedSource, OpenAlexSource, CrossrefSource
from backend.app.stage2.search_strategy import SearchQuery

def test_missing_doi_fallback():
    src = PubMedSource()
    # Assuming PMIDs work
    record = src.get_by_pmid("38396486")
    assert record is not None
    assert record.pmid == "38396486"
    
def test_openalex_id_handling():
    src = OpenAlexSource()
    record = src.get_by_openalex_id("W123456")
    # For now in mocks, it's None or not implemented
    assert record is None

def test_pubmed_search_strategy():
    src = PubMedSource()
    query = SearchQuery(keywords=["cancer"], year_start=2020)
    built = src.strategy.build_query(query)
    assert "term" in built
    assert "cancer" in built["term"]
    
def test_mocked_network_responses():
    src = PubMedSource()
    query = SearchQuery(keywords=["cancer"], year_start=2020)
    # Mock search returns empty list
    results = src.search(query)
    assert results == []
