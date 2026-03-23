import joblib
import pandas as pd


def load_model_and_predict(model_path: str, features: dict) -> int:
    model = joblib.load(model_path)
    df = pd.DataFrame([features])
    prediction = model.predict(df)[0]
    return int(prediction)
