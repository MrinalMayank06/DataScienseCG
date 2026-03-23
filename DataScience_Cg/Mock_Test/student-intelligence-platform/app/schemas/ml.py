from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    hours_studied: float = Field(..., ge=0, le=24)
    attendance: float = Field(..., ge=0, le=100)
    assignments_completed: int = Field(..., ge=0, le=20)
    age: int = Field(..., ge=16, le=100)


class MLTrainResponse(BaseModel):
    message: str
    model_path: str
    accuracy: float
    classification_report: dict


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    confidence_note: str
