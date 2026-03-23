from app.core.config import get_settings
from app.schemas.ml import PredictionRequest
from ml.predict import load_model_and_predict
from ml.train import train_pipeline


class MLService:
    def __init__(self):
        self.settings = get_settings()

    def train(self) -> dict:
        accuracy, report = train_pipeline(model_path=self.settings.model_path)
        return {
            "message": "Model trained successfully",
            "model_path": self.settings.model_path,
            "accuracy": accuracy,
            "classification_report": report,
        }

    def predict(self, payload: PredictionRequest) -> dict:
        prediction = load_model_and_predict(self.settings.model_path, payload.model_dump())
        return {
            "prediction": int(prediction),
            "label": "pass" if int(prediction) == 1 else "fail",
            "confidence_note": "This is a model-based prediction using the persisted classification pipeline.",
        }
