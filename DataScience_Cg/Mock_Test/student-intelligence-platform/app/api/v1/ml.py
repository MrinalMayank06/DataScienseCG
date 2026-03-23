from fastapi import APIRouter

from app.schemas.ml import MLTrainResponse, PredictionRequest, PredictionResponse
from app.services.ml_service import MLService

router = APIRouter(prefix="/ml")


@router.post("/train", response_model=MLTrainResponse)
async def train_model():
    return MLService().train()


@router.post("/predict", response_model=PredictionResponse)
async def predict(payload: PredictionRequest):
    return MLService().predict(payload)
