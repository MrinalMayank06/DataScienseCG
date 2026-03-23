from fastapi import APIRouter

from app.api.v1.analytics import router as analytics_router
from app.api.v1.ml import router as ml_router
from app.api.v1.probability import router as probability_router
from app.api.v1.students import router as students_router

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(students_router, tags=["students"])
api_router.include_router(analytics_router, tags=["analytics"])
api_router.include_router(probability_router, tags=["probability"])
api_router.include_router(ml_router, tags=["ml"])
