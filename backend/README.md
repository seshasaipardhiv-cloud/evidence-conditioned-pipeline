# Evidence-Conditioned Compositional Pipeline Synthesis for Multimodal Cancer Problems

Current phase: STEP 1 — Backend Foundation

**Note:** Scientific stages, ML logic, LLM/RAG, and frontend are NOT implemented yet.

## Setup Instructions

### 1. Python version
Ensure you have Python 3.11+ installed.

### 2. Creating virtual environment
```powershell
python -m venv .venv
```

### 3. Activating virtual environment on Windows PowerShell
```powershell
.\.venv\Scripts\activate
```

### 4. Installing requirements
```powershell
pip install -r backend/requirements.txt
```

### 5. Starting PostgreSQL
Start the PostgreSQL container:
```powershell
docker compose up -d postgres
```

### 6. Checking PostgreSQL
Verify that the container is running and healthy:
```powershell
docker compose ps
```

### 7. Starting FastAPI
Run the FastAPI application locally:
```powershell
uvicorn backend.app.main:app --reload
```
*(Make sure you run this from the project root where `backend` is a subdirectory)*

### 8. Opening Swagger
Navigate to the Swagger UI:
[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 9. Running tests
Run the test suite using pytest:
```powershell
pytest backend/tests -v
```
