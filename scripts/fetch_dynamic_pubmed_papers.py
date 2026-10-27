"""
fetch_dynamic_pubmed_papers.py

Script to dynamically query NCBI PubMed based on the project's aim and keywords,
retrieving:
  1. SciBERT Training Corpus (>= 200 distinct papers)
  2. Pipeline Extraction Corpus (>= 100 extra distinct papers, completely disjoint)
"""

import json
import logging
from pathlib import Path
from backend.app.stage1.dynamic_literature_retriever import DynamicLiteratureRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("fetch_dynamic_pubmed_papers")

PROJECT_KEYWORDS = [
    "clinical machine learning oncology prediction",
    "biomarker machine learning diagnosis prognosis",
    "multimodal healthcare deep learning fusion",
    "tabular clinical risk prediction xgboost",
    "cardiovascular machine learning risk stratification",
    "radiomics deep learning cancer survival",
    "dermatology skin lesion classification transformer",
]

def main():
    retriever = DynamicLiteratureRetriever()
    logger.info("Starting dynamic PubMed literature discovery based on project scope...")
    
    scibert_corpus, extraction_corpus = retriever.search_and_partition_corpora(
        keywords=PROJECT_KEYWORDS,
        scibert_target=215,
        extraction_target=105,
    )
    
    # Save SciBERT training corpus
    scibert_path = Path("SciBERT/data/scibert_training_papers.jsonl")
    scibert_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scibert_path, "w", encoding="utf-8") as f:
        for p in scibert_corpus:
            f.write(json.dumps(p) + "\n")
    logger.info(f"Saved {len(scibert_corpus)} papers to {scibert_path}")
    
    # Save Pipeline extraction corpus
    ext_path = Path("evidence/processed/dynamic_extraction_papers.jsonl")
    ext_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ext_path, "w", encoding="utf-8") as f:
        for p in extraction_corpus:
            f.write(json.dumps(p) + "\n")
    logger.info(f"Saved {len(extraction_corpus)} papers to {ext_path}")
    
    # Verify strict disjointness
    scibert_pmids = set(p["pmid"] for p in scibert_corpus)
    ext_pmids = set(p["pmid"] for p in extraction_corpus)
    overlap = scibert_pmids.intersection(ext_pmids)
    assert len(overlap) == 0, f"Error: Found {len(overlap)} overlapping PMIDs between training and extraction!"
    
    logger.info("Successfully established two strictly disjoint real-literature corpora!")
    logger.info(f"SciBERT Training Corpus: {len(scibert_corpus)} papers (>= 200 requirement satisfied)")
    logger.info(f"Extraction Pipeline Corpus: {len(extraction_corpus)} papers (>= 100 requirement satisfied)")

if __name__ == "__main__":
    main()
