import logging
import json
from typing import List, Dict, Set, Tuple
from pathlib import Path

from backend.app.stage2.models import PaperRecord
from backend.app.stage2.sources import PubMedSource, OpenAlexSource, CrossrefSource

logger = logging.getLogger(__name__)


class EvidenceAcquisition:
    def __init__(self):
        self.sources = [
            PubMedSource(),
            OpenAlexSource(),
            CrossrefSource()
        ]

    def fetch_seed_papers(self) -> Tuple[List[PaperRecord], int, int]:
        """
        Fetches and deduplicates seed papers.

        Returns:
            (unique_papers, raw_candidate_count, failed_retrieval_count)
        """
        seed_path = Path("evidence/metadata/seed_papers.json")
        if not seed_path.exists():
            logger.warning("No seed_papers.json found.")
            return [], 0, 0

        with open(seed_path, "r", encoding="utf-8") as f:
            seeds = json.load(f)

        raw_records: List[PaperRecord] = []
        failed = 0

        for seed in seeds:
            doi = seed.get("doi")
            pmid = seed.get("pmid")

            record = None
            for source in self.sources:
                if doi:
                    record = source.get_by_doi(doi)
                if not record and pmid:
                    record = source.get_by_pmid(pmid)
                if record:
                    raw_records.append(record)
                    break

            if record is None:
                failed += 1
                logger.warning(f"Failed to retrieve paper: doi={doi} pmid={pmid}")

        unique = self._deduplicate(raw_records)
        return unique, len(seeds), failed

    def _deduplicate(self, records: List[PaperRecord]) -> List[PaperRecord]:
        """
        Deduplicate using DOI → PMID → title+year priority.
        """
        seen_dois: Set[str] = set()
        seen_pmids: Set[str] = set()
        seen_title_years: Set[str] = set()

        unique_records = []

        for record in records:
            # DOI-level dedup
            if record.doi:
                doi_key = record.doi.lower().strip()
                if doi_key in seen_dois:
                    continue
                seen_dois.add(doi_key)

            # PMID-level dedup
            if record.pmid:
                if record.pmid in seen_pmids:
                    continue
                seen_pmids.add(record.pmid)

            # Title+year fallback dedup
            title_year = f"{record.title.lower().strip()}_{record.publication_year}"
            if title_year in seen_title_years:
                continue
            seen_title_years.add(title_year)

            unique_records.append(record)

        return unique_records
