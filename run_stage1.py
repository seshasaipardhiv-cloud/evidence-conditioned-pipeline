import json
from fastapi.testclient import TestClient
from backend.app.main import app
from pathlib import Path

client = TestClient(app)

problem_statement = "We want to develop a multimodal machine learning system for cancer research using clinical, pathology, blood and text data. The system should identify an appropriate predictive learning pipeline based on scientific evidence."

response = client.post(
    "/api/v1/stage1/analyze",
    json={"problem_statement": problem_statement}
)

print(f"Status Code: {response.status_code}")
print("Response JSON:")
print(json.dumps(response.json(), indent=2))

# Also verify the output files exist
processed_dir = Path("data/processed/hancock")
meta_dir = Path("data/metadata/hancock")

print("\nVerifying Output Files:")
if (processed_dir / "stage1_problem_representation.json").exists():
    print("OK: data/processed/hancock/stage1_problem_representation.json exists")
else:
    print("MISSING: data/processed/hancock/stage1_problem_representation.json")

if (meta_dir / "stage1_profile_report.json").exists():
    print("OK: data/metadata/hancock/stage1_profile_report.json exists")
else:
    print("MISSING: data/metadata/hancock/stage1_profile_report.json")
