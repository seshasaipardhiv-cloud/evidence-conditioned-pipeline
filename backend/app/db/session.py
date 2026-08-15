import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import OperationalError
from backend.app.config import settings

logger = logging.getLogger(__name__)

engine = create_engine(settings.database_url, echo=(settings.environment == "development"))
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def check_db_connection() -> bool:
    try:
        with engine.connect() as connection:
            return True
    except OperationalError:
        logger.error("Database connection failed due to OperationalError")
        return False
    except Exception:
        logger.error("Database connection failed due to unexpected error")
        return False

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
