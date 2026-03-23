from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.features import FEATURE_COLUMNS, TARGET_COLUMN

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "student_performance.csv"


def build_pipeline() -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, FEATURE_COLUMNS),
        ]
    )

    model = LogisticRegression(max_iter=500, random_state=42)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    return pipeline


def train_pipeline(model_path: str = "artifacts/student_model.joblib"):
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    accuracy = round(float(accuracy_score(y_test, predictions)), 4)
    report = classification_report(y_test, predictions, output_dict=True)

    model_file = BASE_DIR / model_path
    model_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_file)

    return accuracy, report


if __name__ == "__main__":
    accuracy, _ = train_pipeline()
    print(f"Model trained successfully. Accuracy: {accuracy}")
