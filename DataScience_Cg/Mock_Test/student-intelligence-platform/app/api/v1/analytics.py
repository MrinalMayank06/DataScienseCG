from fastapi import APIRouter

from app.schemas.analytics import AnalyticsSummaryResponse, ChartPathsResponse
from app.services.analytics_service import AnalyticsService

router = APIRouter(prefix="/analytics")


@router.get("/summary", response_model=AnalyticsSummaryResponse)
async def analytics_summary():
    return AnalyticsService().generate_summary()


@router.get("/charts", response_model=ChartPathsResponse)
async def analytics_charts():
    return AnalyticsService().generate_charts()
