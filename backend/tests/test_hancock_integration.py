import pytest
from backend.app.data.hancock.loader import ingest_hancock_dataset
from backend.app.data.hancock.config import STRUCTURED_ZIP, TEXT_ZIP

def test_integration_ingest_hancock():
    if not STRUCTURED_ZIP.exists() and not TEXT_ZIP.exists():
        pytest.skip("Real HANCOCK integration test skipped because dataset files are not present.")
    
    success = ingest_hancock_dataset()
    assert success
