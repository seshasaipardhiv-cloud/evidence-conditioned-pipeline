from pathlib import Path

# Paths relative to the project root
ROOT_DIR = Path(".")
DATA_DIR = ROOT_DIR / "data"

RAW_DIR = DATA_DIR / "raw" / "hancock"
INTERIM_DIR = DATA_DIR / "interim" / "hancock"
PROCESSED_DIR = DATA_DIR / "processed" / "hancock"
METADATA_DIR = DATA_DIR / "metadata" / "hancock"

STRUCTURED_ZIP = RAW_DIR / "StructuredData.zip"
TEXT_ZIP = RAW_DIR / "TextData.zip"

STRUCTURED_EXTRACT_DIR = RAW_DIR / "structured"
TEXT_EXTRACT_DIR = RAW_DIR / "text"

INGESTION_REPORT_PATH = METADATA_DIR / "ingestion_report.json"
FILE_MANIFEST_PATH = METADATA_DIR / "file_manifest.json"
