import pytest
from backend.app.data.hancock.text_extractor import HancockTextExtractor

def test_extract_id():
    extractor = HancockTextExtractor()
    assert extractor._extract_id("SurgeryReport_065.txt") == "065"
    assert extractor._extract_id("SurgeryReport_History_001.txt") == "001"
    assert extractor._extract_id("SurgeryDescriptionEnglish_123.txt") == "123"
    assert extractor._extract_id("InvalidName.txt") == ""

def test_determine_type():
    extractor = HancockTextExtractor()
    assert extractor._determine_type("SurgeryReport_History_001.txt") == "history"
    assert extractor._determine_type("SurgeryReport_001.txt") == "report"
    assert extractor._determine_type("SurgeryDescription_001.txt") == "description"
    assert extractor._determine_type("SurgeryDescriptionEnglish_001.txt") == "description"
    assert extractor._determine_type("SurgeryReport_ICD_Codes_001.txt") == "unmapped"

def test_process_directory_synthetic(tmp_path):
    text_dir = tmp_path / "text"
    text_dir.mkdir()
    
    # Valid files
    (text_dir / "SurgeryReport_001.txt").write_text("Report content", encoding="utf-8")
    (text_dir / "SurgeryReport_History_001.txt").write_text("History content", encoding="utf-8")
    (text_dir / "SurgeryDescription_001.txt").write_text("Description content", encoding="utf-8")
    (text_dir / "SurgeryDescriptionEnglish_001.txt").write_text("English desc", encoding="utf-8")
    
    # Leading zero preservation
    (text_dir / "SurgeryReport_002.txt").write_text("Report 2", encoding="utf-8")
    
    # Empty file
    (text_dir / "SurgeryReport_003.txt").write_text("", encoding="utf-8")
    
    # Unmapped file
    (text_dir / "SurgeryReport_ICD_Codes_004.txt").write_text("ICD", encoding="utf-8")
    
    extractor = HancockTextExtractor()
    records = extractor.process_directory(text_dir)
    
    # Convert list of dicts to a dict mapped by patient_id for easy assertions
    recs_by_id = {r["patient_id"]: r for r in records}
    
    assert "001" in recs_by_id
    rec_1 = recs_by_id["001"]
    assert len(rec_1["report"]) == 1
    assert rec_1["report"][0]["text"] == "Report content"
    assert rec_1["report"][0]["source_file"] == "SurgeryReport_001.txt"
    
    assert len(rec_1["history"]) == 1
    assert rec_1["history"][0]["text"] == "History content"
    
    assert len(rec_1["description"]) == 2
    desc_texts = [d["text"] for d in rec_1["description"]]
    assert "Description content" in desc_texts
    assert "English desc" in desc_texts
    
    assert "002" in recs_by_id
    assert recs_by_id["002"]["report"][0]["text"] == "Report 2"
    
    assert "003" in recs_by_id
    assert not recs_by_id["003"]["report"][0]["text"]
    
    # 004 only had unmapped text, shouldn't create a record
    assert "004" not in recs_by_id
    assert "SurgeryReport_ICD_Codes_004.txt" in extractor.unmapped_files
    
    assert extractor.stats["text_file_count"] == 7
    assert extractor.stats["empty_text_count"] == 1
    assert extractor.stats["description_count"] == 2
