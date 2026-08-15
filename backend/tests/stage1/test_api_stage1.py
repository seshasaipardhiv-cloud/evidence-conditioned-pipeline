"""Stage 1 API endpoint tests using TestClient."""
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

VALID_STMT = (
    "We want to develop a multimodal machine learning system for cancer research "
    "using clinical, pathology, blood and text data."
)


def test_stage1_analyze_returns_200():
    response = client.post("/api/v1/stage1/analyze", json={"problem_statement": VALID_STMT})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "representation" in data


def test_stage1_response_has_key_fields():
    response = client.post("/api/v1/stage1/analyze", json={"problem_statement": VALID_STMT})
    rep = response.json()["representation"]
    assert "problem" in rep
    assert "dataset" in rep
    assert "task" in rep
    assert "modalities" in rep
    assert "target_information" in rep
    assert "compatibility" in rep
    assert "warnings" in rep
    assert "provenance" in rep


def test_stage1_provenance_in_response():
    response = client.post("/api/v1/stage1/analyze", json={"problem_statement": VALID_STMT})
    rep = response.json()["representation"]
    assert rep["provenance"]["source_type"] is not None
    assert rep["problem"]["task_type"]["provenance"] is not None


def test_stage1_empty_statement_rejected():
    response = client.post("/api/v1/stage1/analyze", json={"problem_statement": ""})
    assert response.status_code == 422


def test_stage1_too_short_statement_rejected():
    response = client.post("/api/v1/stage1/analyze", json={"problem_statement": "Hi"})
    assert response.status_code == 422


def test_stage1_classification_task_detected():
    stmt = "Classify cancer patients for recurrence using clinical data."
    response = client.post("/api/v1/stage1/analyze", json={"problem_statement": stmt})
    assert response.status_code == 200
    rep = response.json()["representation"]
    assert rep["task"]["value"] == "classification"


def test_stage1_no_raw_text_in_response():
    response = client.post("/api/v1/stage1/analyze", json={"problem_statement": VALID_STMT})
    body = str(response.json())
    assert "patient history:" not in body.lower()
    assert "surgical report:" not in body.lower()
