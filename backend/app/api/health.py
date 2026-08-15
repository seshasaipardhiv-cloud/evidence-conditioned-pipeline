from fastapi import APIRouter, HTTPException
from backend.app.db.session import check_db_connection
from backend.app.config import settings

router = APIRouter()

@router.get("/health")
def health_check():
    return {"status": "ok"}

@router.get("/health/db")
def health_check_db():
    is_connected = check_db_connection()
    if is_connected:
        return {"status": "ok", "database": "connected"}
    else:
        raise HTTPException(status_code=503, detail="Database connection failed")

@router.get("/version")
def version():
    return {
        "project": settings.app_name,
        "version": settings.app_version,
        "stage": "foundation"
    }
