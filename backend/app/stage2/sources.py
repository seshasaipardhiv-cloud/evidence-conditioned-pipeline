import json
from abc import ABC, abstractmethod
from typing import List, Optional
from datetime import datetime
from pathlib import Path

from backend.app.stage2.models import PaperRecord
from backend.app.stage2.search_strategy import SearchQuery, SearchStrategy, PubMedSearchStrategy, OpenAlexSearchStrategy, CrossrefSearchStrategy

class EvidenceSource(ABC):
    def __init__(self):
        self.strategy: Optional[SearchStrategy] = None
        
    @abstractmethod
    def get_by_doi(self, doi: str) -> Optional[PaperRecord]:
        pass

    @abstractmethod
    def get_by_pmid(self, pmid: str) -> Optional[PaperRecord]:
        pass
        
    @abstractmethod
    def search(self, query: SearchQuery) -> List[PaperRecord]:
        pass
        
    def _mock_record_from_seed(self, identifier: str, key_field: str) -> Optional[PaperRecord]:
        """Helper to fetch from our local seed_papers.json for testing."""
        seed_path = Path("evidence/metadata/seed_papers.json")
        if not seed_path.exists():
            return None
            
        with open(seed_path, "r", encoding="utf-8") as f:
            seeds = json.load(f)
            
        for seed in seeds:
            if seed.get(key_field) == identifier:
                # Try to parse year, default to 2024
                year = seed.get("year", 2024)
                if isinstance(year, str) and year.isdigit():
                    year = int(year)
                    
                abstract_val = seed.get("abstract", "") or ""
                return PaperRecord(
                    paper_id=f"paper_{seed.get('pmid', seed.get('doi', '000').replace('/', '_'))}",
                    title=seed.get("title", ""),
                    authors=["Unknown"],
                    publication_year=year,
                    doi=seed.get("doi"),
                    pmid=seed.get("pmid"),
                    source=self.__class__.__name__,
                    abstract=abstract_val if abstract_val.strip() else None,
                    abstract_available=bool(abstract_val.strip()),
                    full_text_available=False,
                    retrieval_date=datetime.now().isoformat()
                )
        return None

class PubMedSource(EvidenceSource):
    def __init__(self):
        super().__init__()
        self.strategy = PubMedSearchStrategy()
        
    def get_by_doi(self, doi: str) -> Optional[PaperRecord]:
        return self._mock_record_from_seed(doi, "doi")

    def get_by_pmid(self, pmid: str) -> Optional[PaperRecord]:
        return self._mock_record_from_seed(pmid, "pmid")
        
    def search(self, query: SearchQuery) -> List[PaperRecord]:
        # Returns empty list in mock
        return []

class OpenAlexSource(EvidenceSource):
    def __init__(self):
        super().__init__()
        self.strategy = OpenAlexSearchStrategy()

    def get_by_doi(self, doi: str) -> Optional[PaperRecord]:
        return self._mock_record_from_seed(doi, "doi")

    def get_by_pmid(self, pmid: str) -> Optional[PaperRecord]:
        return self._mock_record_from_seed(pmid, "pmid")
        
    def get_by_openalex_id(self, openalex_id: str) -> Optional[PaperRecord]:
        return None
        
    def search(self, query: SearchQuery) -> List[PaperRecord]:
        return []

class CrossrefSource(EvidenceSource):
    def __init__(self):
        super().__init__()
        self.strategy = CrossrefSearchStrategy()

    def get_by_doi(self, doi: str) -> Optional[PaperRecord]:
        return self._mock_record_from_seed(doi, "doi")

    def get_by_pmid(self, pmid: str) -> Optional[PaperRecord]:
        return None
        
    def search(self, query: SearchQuery) -> List[PaperRecord]:
        return []
