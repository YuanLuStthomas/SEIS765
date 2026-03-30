import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import boto3



def train_model(df: pd.DataFrame, model_path: str = "models/iris_model.pkl") -> float:
    """Train a logistic regression classifier and save it (Iris)."""
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"[ml_pipeline.model] Model accuracy: {acc:.4f}")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(clf, model_path)
    print(f"[ml_pipeline.model] Saved model to {model_path}")

    return acc


# -------------------------
#pipeline functions
# -------------------------

def _paths(model_dir: str):
    os.makedirs(model_dir, exist_ok=True)
    return {
        "model": os.path.join(model_dir, "model.pkl"),
        "metrics": os.path.join(model_dir, "metrics.json"),
        "metadata": os.path.join(model_dir, "metadata.json"),
    }


def train_model_v2(model_dir: str = "models", model_version: str = "dev") -> str:
    """
    Train Logistic Regression on breast cancer dataset and save model artifact.
    Returns model path.
    """
    paths = _paths(model_dir)

    data = load_breast_cancer()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, y_train)

    joblib.dump(clf, paths["model"])
    print(f"[hw3] Saved model to {paths['model']} (version={model_version})")

    
    return paths["model"]


def evaluate_model_v2(model_dir: str = "models", model_version: str = "dev") -> float:
    """
    Evaluate model on test set and write:
      models/metrics.json
      models/metadata.json
    """
    paths = _paths(model_dir)

    if not os.path.exists(paths["model"]):
        raise FileNotFoundError(f"Model not found at {paths['model']}")

    clf = joblib.load(paths["model"])

    data = load_breast_cancer()
    X = data.data
    y = data.target

 
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preds = clf.predict(X_test)
    acc = float(accuracy_score(y_test, preds))

  
    with open(paths["metrics"], "w") as f:
        json.dump({"accuracy": acc}, f, indent=2)
    print(f"[hw3] Wrote metrics to {paths['metrics']}")

   
    metadata = {
        "model_version": model_version,
        "dataset": "breast_cancer",
        "model_type": "logistic_regression",
        "accuracy": acc,
    }
    with open(paths["metadata"], "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[hw3] Wrote metadata to {paths['metadata']}")

    return acc


def promote_model_v2(
    model_dir: str = "models",
    model_version: str = "dev",
    threshold: float = 0.94,
) -> str:
    """
    Promote model if it meets threshold; upload artifacts to S3:
      s3://<bucket>/models/<model_version>/{model.pkl, metrics.json, metadata.json}
    If fails threshold, task should fail.
    """
    paths = _paths(model_dir)

    if not os.path.exists(paths["metrics"]):
        raise FileNotFoundError(f"metrics.json not found at {paths['metrics']}")
    if not os.path.exists(paths["metadata"]):
        raise FileNotFoundError(f"metadata.json not found at {paths['metadata']}")
    if not os.path.exists(paths["model"]):
        raise FileNotFoundError(f"model.pkl not found at {paths['model']}")

    with open(paths["metrics"], "r") as f:
        metrics = json.load(f)

    acc = float(metrics.get("accuracy", 0.0))
    print(f"[hw3] Promote check: accuracy={acc:.4f}, threshold={threshold:.2f}")

   
    if acc < threshold:
        raise ValueError(f"Model failed quality gate: accuracy {acc:.4f} < {threshold:.2f}")

    bucket = os.environ.get("S3_BUCKET")
    if not bucket:
        raise EnvironmentError("S3_BUCKET environment variable is not set.")

    prefix = f"models/{model_version}/"

    s3 = boto3.client("s3")
    for local_path, name in [
        (paths["model"], "model.pkl"),
        (paths["metrics"], "metrics.json"),
        (paths["metadata"], "metadata.json"),
    ]:
        key = prefix + name
        s3.upload_file(local_path, bucket, key)
        print(f"[hw3] Uploaded s3://{bucket}/{key}")

    return f"s3://{bucket}/{prefix}"