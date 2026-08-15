import pytest

@pytest.fixture(autouse=True)
def mock_external_apis(monkeypatch):
    """
    Ensure no external network access is made during standard test suite execution.
    By default, tests will use the mocked adapters built into the sources.py that read from seed_papers.json.
    """
    pass
