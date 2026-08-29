# Annotation Infrastructure — Stage 2C

## Purpose

This directory contains the annotation scaffolding for building supervised
training data for the Stage 2C Transformer NER head.

## Current State (Honest)

The SciBERT classification head **requires supervised training data** to
produce meaningful NER predictions. Without fine-tuning:

- The encoder produces real, meaningful contextual embeddings (SciBERT is
  trained on 1.14M scientific papers)
- The classification head is **randomly initialized** and produces random
  projections of those embeddings
- Real-world NER performance from the untrained head is near-random

This is explicitly documented throughout the codebase and in all output files.

## Bootstrapping Stage

`bootstrap_labels.py` uses the legacy vocabulary (from `mechanism_mapper.py`)
to generate weak BIO-tagged training candidates. These are:

- Clearly marked `extraction_method = bootstrap_weak`
- Marked `is_bootstrap = True`
- Given `confidence = 0.55` (below the 0.60 LOW_CONFIDENCE_THRESHOLD)
- All flagged `review_flag = True`

**They are not NER model output. They are vocabulary-matched seed labels.**

## Annotation Workflow

1. Run `BootstrapLabelGenerator.save_to_jsonl()` to produce
   `seed_annotations.jsonl`
2. Human annotator reviews each entity, corrects labels, removes noise
3. Verified annotations saved to `verified_annotations.jsonl`
4. Fine-tune the NER head:
   ```bash
   python -m backend.app.stage2.annotation.fine_tune \
       --train verified_annotations.jsonl \
       --output models/ner_head_v1.pt
   ```
5. Re-run Stage 2C with fine-tuned head

## Entity Schema

```json
{
  "entity_id": "uuid4",
  "text": "SMOTE",
  "entity_type": "SAMPLING",
  "mechanism_category": "Sampling",
  "start_char": 14,
  "end_char": 19,
  "source_text": "We applied SMOTE for class balancing...",
  "source_paper_id": "paper_abc",
  "confidence": 0.55,
  "confidence_level": "LOW",
  "review_flag": true,
  "extraction_method": "bootstrap_weak",
  "is_bootstrap": true
}
```

## Files

| File | Description |
|---|---|
| `bootstrap_labels.py` | Generates weak BIO labels from vocabulary |
| `seed_annotations.jsonl` | Bootstrap candidates for human review |
| `verified_annotations.jsonl` | (created after human review) |
| `README.md` | This file |
