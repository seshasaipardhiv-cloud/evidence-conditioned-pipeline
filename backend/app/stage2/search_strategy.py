from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

class SearchQuery(BaseModel):
    keywords: List[str]
    year_start: Optional[int] = None
    year_end: Optional[int] = None
    must_include_modalities: List[str] = []
    task_filter: Optional[str] = None
    limit: int = 10

class SearchStrategy(ABC):
    @abstractmethod
    def build_query(self, query: SearchQuery) -> Dict[str, Any]:
        """Convert a standard SearchQuery into a source-specific query payload."""
        pass

class PubMedSearchStrategy(SearchStrategy):
    def build_query(self, query: SearchQuery) -> Dict[str, Any]:
        # E.g. (head and neck cancer) AND (multimodal) AND ("2020"[Date - Publication] : "3000"[Date - Publication])
        terms = ["(" + " OR ".join(query.keywords) + ")"]
        if query.must_include_modalities:
            mod_term = " AND ".join(query.must_include_modalities)
            terms.append(f"({mod_term})")
            
        term_str = " AND ".join(terms)
        
        if query.year_start:
            end_yr = query.year_end or 3000
            term_str += f' AND ("{query.year_start}"[Date - Publication] : "{end_yr}"[Date - Publication])'
            
        return {
            "term": term_str,
            "retmax": query.limit,
            "retmode": "json"
        }

class OpenAlexSearchStrategy(SearchStrategy):
    def build_query(self, query: SearchQuery) -> Dict[str, Any]:
        filters = []
        if query.year_start:
            filters.append(f"from_publication_date:{query.year_start}-01-01")
        if query.year_end:
            filters.append(f"to_publication_date:{query.year_end}-12-31")
            
        search_terms = " ".join(query.keywords)
        if query.must_include_modalities:
            search_terms += " " + " ".join(query.must_include_modalities)
            
        return {
            "search": search_terms,
            "filter": ",".join(filters) if filters else None,
            "per-page": query.limit
        }

class CrossrefSearchStrategy(SearchStrategy):
    def build_query(self, query: SearchQuery) -> Dict[str, Any]:
        search_terms = " ".join(query.keywords)
        filters = []
        if query.year_start:
            filters.append(f"from-pub-date:{query.year_start}-01-01")
            
        payload = {
            "query": search_terms,
            "rows": query.limit
        }
        if filters:
            payload["filter"] = ",".join(filters)
            
        return payload
