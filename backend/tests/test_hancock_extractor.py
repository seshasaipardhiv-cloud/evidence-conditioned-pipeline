import pytest
import zipfile
from pathlib import Path
from backend.app.data.hancock.extractor import safe_extract_zip, calculate_sha256

def test_calculate_sha256(tmp_path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("hello world")
    hash_val = calculate_sha256(test_file)
    assert hash_val == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"

def test_safe_extract_zip(tmp_path):
    zip_path = tmp_path / "test.zip"
    extract_dir = tmp_path / "extract"
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("test1.json", '{"patient_id": "123"}')
        zf.writestr("subdir/test2.json", '{"patient_id": "456"}')
        
    success, msg = safe_extract_zip(zip_path, extract_dir)
    assert success
    assert (extract_dir / "test1.json").exists()
    assert (extract_dir / "subdir" / "test2.json").exists()

def test_path_traversal_zip(tmp_path):
    zip_path = tmp_path / "evil.zip"
    extract_dir = tmp_path / "extract"
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("../evil.txt", "evil")
        
    success, msg = safe_extract_zip(zip_path, extract_dir)
    assert not success
    assert "Path traversal attempt detected" in msg
