import sys
import logging
from backend.app.data.hancock.loader import ingest_hancock_dataset
from backend.app.logging_config import setup_logging

def main():
    setup_logging("INFO")
    logger = logging.getLogger("ingest_hancock")
    
    try:
        success = ingest_hancock_dataset()
        if not success:
            logger.info("Ingestion skipped due to missing files.")
            sys.exit(0)
    except Exception as e:
        logger.error(f"Critical error during ingestion: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
