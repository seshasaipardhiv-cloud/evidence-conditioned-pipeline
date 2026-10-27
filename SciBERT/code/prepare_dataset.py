"""
prepare_dataset.py

Constructs the complete SciBERT NER training, validation, and test datasets
derived 100% from the 30 published research papers in the evidence corpus.
Zero synthetic sentences are used.
"""

from __future__ import annotations

import json
import logging
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from backend.app.stage2.stage2d.advanced_weak_labeler import AdvancedWeakLabeler
from backend.app.stage2.ner_entity_types import LABEL2ID, NUM_LABELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("prepare_scibert_dataset")


def segment_sentences(text: str) -> List[str]:
    """Cleanly split abstract text into individual sentences."""
    # Handle common scientific abbreviations (e.g., vs., et al., e.g., i.e., Fig., no.)
    clean_text = text.strip()
    clean_text = re.sub(r'\bet\s+al\.', 'et_al', clean_text)
    clean_text = re.sub(r'\be\.g\.,?', 'for_example', clean_text)
    clean_text = re.sub(r'\bi\.e\.,?', 'that_is', clean_text)
    clean_text = re.sub(r'\bvs\.', 'versus', clean_text)
    clean_text = re.sub(r'\bFig\.', 'Figure', clean_text)
    clean_text = re.sub(r'\bNo\.', 'Number', clean_text)

    raw_sents = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', clean_text)
    sents = []
    for s in raw_sents:
        # Revert placeholders
        s = s.replace('et_al', 'et al.').replace('for_example', 'e.g.,').replace('that_is', 'i.e.,').replace('versus', 'vs.').replace('Figure', 'Fig.').replace('Number', 'No.')
        s = s.strip()
        if len(s) >= 15:
            sents.append(s)
    return sents


def build_real_scibert_dataset(
    papers_path: Optional[str] = None,
    out_dir: str = "SciBERT/data",
    seed: int = 42,
) -> Dict[str, Any]:
    if papers_path is None:
        if Path("SciBERT/data/scibert_training_papers.jsonl").exists():
            papers_path = "SciBERT/data/scibert_training_papers.jsonl"
        else:
            papers_path = "evidence/processed/papers.jsonl"

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with open(papers_path, "r", encoding="utf-8") as f:
        papers = [json.loads(line) for line in f if line.strip()]

    logger.info(f"Loaded {len(papers)} papers from {papers_path}")

    labeler = AdvancedWeakLabeler()
    all_examples: List[Dict[str, Any]] = []
    entity_counts: Dict[str, int] = {}

    total_abstract_chars = 0
    paper_contributions: Dict[str, int] = {}

    for paper in papers:
        pid = paper["paper_id"]
        pmid = paper.get("pmid")
        doi = paper.get("doi")
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")

        if not abstract or not abstract.strip():
            logger.warning(f"Skipping paper without abstract: {pid}")
            continue

        total_abstract_chars += len(abstract)
        sentences = segment_sentences(abstract)
        paper_contributions[pid] = len(sentences)

        for sent_idx, sent_text in enumerate(sentences):
            # Extract grounded methodology spans
            weak_entities = labeler.extract_weak_labels(
                text=sent_text,
                paper_id=pid,
                section_name="abstract",
                pmid=pmid,
                doi=doi,
            )

            spans = []
            for ent in weak_entities:
                etype = ent.entity_type.value if hasattr(ent.entity_type, "value") else str(ent.entity_type)
                spans.append({
                    "start_char": ent.start_char,
                    "end_char": ent.end_char,
                    "text": ent.text,
                    "entity_type": etype,
                    "bio_tag": f"B-{etype}",
                    "confidence": ent.confidence,
                })
                entity_counts[etype] = entity_counts.get(etype, 0) + 1

            constraints = labeler.extract_experimental_constraints(sent_text)

            example = {
                "example_id": f"{pid}_s{sent_idx:02d}",
                "source_paper_id": pid,
                "source_pmid": pmid,
                "source_doi": doi,
                "source_title": title,
                "sentence_text": sent_text,
                "char_length": len(sent_text),
                "word_count": len(sent_text.split()),
                "entities_count": len(spans),
                "spans": spans,
                "constraints": constraints,
                "has_methodology_mention": len(spans) > 0,
            }
            all_examples.append(example)

    # Deterministic train/val/test split (70% / 15% / 15%)
    random.seed(seed)
    # Group by paper to prevent data leakage across splits
    papers_with_examples = list(paper_contributions.keys())
    random.shuffle(papers_with_examples)

    n_papers = len(papers_with_examples)
    n_train = int(n_papers * 0.70)
    n_val = int(n_papers * 0.15)

    train_papers = set(papers_with_examples[:n_train])
    val_papers = set(papers_with_examples[n_train:n_train + n_val])
    test_papers = set(papers_with_examples[n_train + n_val:])

    train_examples = [ex for ex in all_examples if ex["source_paper_id"] in train_papers]
    val_examples = [ex for ex in all_examples if ex["source_paper_id"] in val_papers]
    test_examples = [ex for ex in all_examples if ex["source_paper_id"] in test_papers]

    # Save splits
    with open(out_path / "full_ner_dataset.json", "w", encoding="utf-8") as f:
        json.dump(all_examples, f, indent=2)

    with open(out_path / "train_ner_dataset.json", "w", encoding="utf-8") as f:
        json.dump(train_examples, f, indent=2)

    with open(out_path / "val_ner_dataset.json", "w", encoding="utf-8") as f:
        json.dump(val_examples, f, indent=2)

    with open(out_path / "test_ner_dataset.json", "w", encoding="utf-8") as f:
        json.dump(test_examples, f, indent=2)

    stats = {
        "dataset_name": "SciBERT Real Scientific Methodology NER Benchmark",
        "provenance": "100% genuine published scientific literature from PubMed/CrossRef (zero synthetic data)",
        "total_source_papers": len(papers),
        "papers_with_abstracts": len(paper_contributions),
        "total_sentences": len(all_examples),
        "sentences_with_entities": sum(1 for ex in all_examples if ex["has_methodology_mention"]),
        "total_entity_spans": sum(len(ex["spans"]) for ex in all_examples),
        "entity_distribution_across_classes": entity_counts,
        "split_summary": {
            "train": {
                "paper_count": len(train_papers),
                "sentence_count": len(train_examples),
                "entity_count": sum(len(ex["spans"]) for ex in train_examples),
            },
            "validation": {
                "paper_count": len(val_papers),
                "sentence_count": len(val_examples),
                "entity_count": sum(len(ex["spans"]) for ex in val_examples),
            },
            "test": {
                "paper_count": len(test_papers),
                "sentence_count": len(test_examples),
                "entity_count": sum(len(ex["spans"]) for ex in test_examples),
            },
        },
        "leakage_isolation": "Paper-level disjoint split (no sentences from test papers exist in train or val)",
    }

    with open(out_path / "dataset_statistics.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    logger.info("================================================================================")
    logger.info("SCIBERT REAL LITERATURE DATASET GENERATED SUCCESSFULLY")
    logger.info(f"Total Sentences: {len(all_examples)} across {len(paper_contributions)} real papers")
    logger.info(f"Total Methodology Spans: {stats['total_entity_spans']}")
    logger.info(f"Train: {len(train_examples)} | Val: {len(val_examples)} | Test: {len(test_examples)}")
    logger.info(f"Saved to: {out_path}")
    logger.info("================================================================================")

    return stats


if __name__ == "__main__":
    build_real_scibert_dataset()
