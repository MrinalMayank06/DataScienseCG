from fastapi import APIRouter, Query

from app.schemas.probability import ProbabilityResponse
from app.services.probability_service import ProbabilityService

router = APIRouter(prefix="/probability")


@router.get("/pass", response_model=ProbabilityResponse)
async def pass_probability(
    passed_count: int = Query(..., ge=0),
    total_count: int = Query(..., gt=0),
):
    return ProbabilityService().calculate_pass_probability(
        passed_count=passed_count,
        total_count=total_count,
    )
