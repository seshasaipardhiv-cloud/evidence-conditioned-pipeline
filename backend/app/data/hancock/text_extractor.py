import logging
import re
from pathlib import Path
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class TextRecord:
    def __init__(self, patient_id: str):
        self.patient_id = patient_id
        self.history: List[Dict[str, str]] = []
        self.report: List[Dict[str, str]] = []
        self.description: List[Dict[str, str]] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "history": self.history,
            "report": self.report,
            "description": self.description,
            "metadata": {
                "history_count": len(self.history),
                "report_count": len(self.report),
                "description_count": len(self.description)
            }
        }


class HancockTextExtractor:
    def __init__(self):
        self.records: Dict[str, TextRecord] = {}
        self.unmapped_files: List[str] = []
        self.errors: List[str] = []
        
        self.stats = {
            "text_file_count": 0,
            "history_count": 0,
            "report_count": 0,
            "description_count": 0,
            "empty_text_count": 0
        }

    def _extract_id(self, filename: str) -> str:
        """Extracts patient ID assuming it is the last numeric part before the extension."""
        match = re.search(r'_([^_\.]+)\.txt$', filename)
        if match:
            return match.group(1).strip()
        return ""

    def _determine_type(self, filename: str) -> str:
        """Maps file name to text type deterministically based on Hancock structure."""
        lower_name = filename.lower()
        if "surgeryreport_history" in lower_name:
            return "history"
        elif "surgeryreport_" in lower_name and "history" not in lower_name and "icd" not in lower_name and "ops" not in lower_name:
            return "report"
        elif "surgerydescription_" in lower_name or "surgerydescriptionenglish_" in lower_name:
            return "description"
        else:
            return "unmapped"

    def process_directory(self, text_dir: Path) -> List[Dict[str, Any]]:
        if not text_dir.exists():
            return []
            
        for txt_file in text_dir.rglob("*.txt"):
            self.stats["text_file_count"] += 1
            filename = txt_file.name
            
            raw_patient_id = self._extract_id(filename)
            if not raw_patient_id:
                self.errors.append(f"Cannot extract patient_id from filename: {filename}")
                self.unmapped_files.append(filename)
                continue
                
            canonical_patient_id = raw_patient_id.strip()
            
            text_type = self._determine_type(filename)
            if text_type == "unmapped":
                self.unmapped_files.append(filename)
                continue
                
            try:
                with open(txt_file, "r", encoding="utf-8") as f:
                    content = f.read()
            except UnicodeDecodeError:
                self.errors.append(f"Encoding error (not UTF-8): {filename}")
                continue
            except Exception as e:
                self.errors.append(f"Failed to read file: {filename} - {str(e)}")
                continue
                
            if not content.strip():
                self.stats["empty_text_count"] += 1
                
            if canonical_patient_id not in self.records:
                self.records[canonical_patient_id] = TextRecord(raw_patient_id)
                
            record = self.records[canonical_patient_id]
            
            if text_type == "history":
                record.history.append({"source_file": filename, "text": content})
                self.stats["history_count"] += 1
            elif text_type == "report":
                record.report.append({"source_file": filename, "text": content})
                self.stats["report_count"] += 1
            elif text_type == "description":
                record.description.append({"source_file": filename, "text": content})
                self.stats["description_count"] += 1

        return [rec.to_dict() for rec in self.records.values()]
