import json
import joblib
import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any


class PredictRequest(BaseModel):
    features: List[float]  


def create_app(model_path: str = "models/model.pkl", metadata_path: str = "models/metadata.json"):
    if not Path(model_path).exists():
        raise RuntimeError(
            f"Model file not found at '{model_path}'. "
            "Run the Airflow DAG ml_training_pipeline_v2 first."
        )

    model = joblib.load(model_path)

    metadata: Dict[str, Any] = {}
    if Path(metadata_path).exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {"warning": f"metadata not found at {metadata_path}"}

    app = FastAPI(title="Breast Cancer Model API")

    target_names = {0: "malignant", 1: "benign"}

    @app.get("/")
    def root():
        return {"message": "Model is ready for inference!", "classes": target_names}

    @app.get("/model/info")
    def model_info():
        return metadata

    @app.post("/predict")
    def predict(request: PredictRequest):
        if len(request.features) != 30:
            raise HTTPException(status_code=400, detail="features must have length 30")

        X = np.array([request.features], dtype=float)
        try:
            idx = int(model.predict(X)[0])
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

        return {"prediction": target_names.get(idx, str(idx)), "class_index": idx}

    return app