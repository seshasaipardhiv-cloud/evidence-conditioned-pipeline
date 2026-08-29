# Evidence-Conditioned Compositional Pipeline Synthesis for Multimodal Cancer Problems

**Only the backend foundation is implemented. Scientific stages have not yet been implemented.**

## Purpose
This project is the scientific framework for the "Evidence-Conditioned Compositional Pipeline Synthesis for Multimodal Cancer Problems" research project. It integrates evidence ingestion, Transformer-based scientific NLP entity extraction, evidence graph synthesis, multimodal candidate generation, and rigorous forensic benchmarking.

## Current Implementation Status
- **Stage 1**: Foundation & Data Schema
- **Stage 2A/2B**: Evidence Acquisition (PubMed/PMC/Unpaywall) & Controlled Graph Ingestion
- **Stage 2C**: Deep-Learning NLP Mechanism Extraction (SciBERT-based Transformer NER, Relation Extraction, Provenance Tracking)
- **Stage 5B/6A**: Primary Clinical Tabular Experiment & Automated Pipelines
- **Stage 6A–6I**: Manuscript, Audit, & Provenance Infrastructure
- **Stage 10/10.5**: Multimodal Evidence-Conditioned Automation & Forensic Validation
- **Stage 11/11.x**: Model Alternative & Transparent Ensemble Benchmarking

## Stage 2C: Deep-Learning NLP Extraction Architecture
- **Model**: `allenai/scibert_scivocab_uncased` (SciBERT 12-layer Transformer)
- **Token Classification**: 23 BIO labels mapping to 11 methodology entity classes (`MODEL_ARCH`, `SAMPLING`, `PREPROCESSING`, `LOSS`, `REGULARIZATION`, etc.)
- **Association**: Co-sentence proximity and typed relation extraction (`HAS_LOSS`, `HAS_OPTIMIZER`, etc.)
- **Safety & Provenance**: Strict confidence thresholding, human-review flags (`review_flag=True`), and immutable paper/DOI/PMID/span offsets. Zero silent regex fallback.

## Setup Instructions

### 1. Create Python Environment
```powershell
cd evidence-conditioned-pipeline
python -m venv venv
.\venv\Scripts\activate
```

### 2. Install Dependencies
```powershell
pip install -r backend/requirements.txt
```

### 3. Configure Environment
Copy the example configuration to `.env`:
```powershell
cp .env.example .env
```

### 4. Start PostgreSQL
Run the database using Docker Compose:
```powershell
docker compose up -d postgres
```

### 5. Start FastAPI
Navigate to `backend` and run:
```powershell
cd backend
uvicorn app.main:app --reload
```

### 6. Run Tests
From the `backend` directory, run:
```powershell
pytest
```
