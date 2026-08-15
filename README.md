# Evidence-Conditioned Compositional Pipeline Synthesis for Multimodal Cancer Problems

**Only the backend foundation is implemented. Scientific stages have not yet been implemented.**

## Purpose
This project is the foundational backend infrastructure for the "Evidence-Conditioned Compositional Pipeline Synthesis for Multimodal Cancer Problems" research project. It sets up the basic directories, PostgreSQL database via Docker, FastAPI app, configuration and testing foundation.

## Current Implementation Status
Currently, only STEP 1 (Project and Backend Foundation) is complete. The system is scaffolded without any ML components, RAG, vector search, or external APIs.

## Architecture
- FastAPI application (Python 3.11+)
- PostgreSQL (running in Docker)
- SQLAlchemy ORM & Alembic for database operations
- Pytest for testing

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
