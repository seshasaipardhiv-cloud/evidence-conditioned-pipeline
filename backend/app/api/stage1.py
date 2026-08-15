"""
Stage 1 API — POST /api/v1/stage1/analyze
"""
from __future__ import annotations

import logging
from fastapi import APIRouter, HTTPException

from backend.app.stage1.models import Stage1Request, Stage1Response
from backend.app.stage1.orchestrator import run_stage1

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/stage1", tags=["Stage 1 — Problem Understanding"])


@router.post("/analyze", response_model=Stage1Response)
async def analyze_problem(request: Stage1Request) -> Stage1Response:
    """
    Submit a natural-language problem statement and receive a deterministic
    Structured Problem Representation grounded in the HANCOCK dataset.

    Does not call an LLM. Does not train models. Does not generate pipelines.
    """
    try:
        representation = run_stage1(
            problem_statement=request.problem_statement,
            write_outputs=True,
        )
        return Stage1Response(status="ok", representation=representation)
    except ValueError as e:
        logger.warning(f"Stage 1 invalid input: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        logger.error(f"Stage 1 safety violation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Stage 1 internal safety check failed.")
    except Exception as e:
        logger.error(f"Stage 1 unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")
